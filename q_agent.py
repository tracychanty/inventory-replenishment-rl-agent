"""
State space is discretized into bins so the Q-table remains small enough
for tabular learning while still capturing the key state dimensions.

The Q-state is intentionally simpler than the full observation:
    (inventory_bin, is_weekend, pending_bin, lead_time_bin, days_remaining_bin)
"""

import numpy as np
import pickle
from pathlib import Path
from env import (
    InventoryEnv, ORDER_OPTIONS, EPISODE_DAYS,
    MAX_INVENTORY, MAX_PENDING, LEAD_TIME_DAYS
)
from baseline import (
    evaluate_policy,
    print_results,
    ReorderPointPolicy,
    RandomPolicy,
    tune_reorder_policy,
)


# ---------------------------------------------------------------------------
# State discretization
# ---------------------------------------------------------------------------
# Bin each continuous/large dimension into a small number of buckets.
# Fewer bins = faster learning but less precision.
# The default bins are intentionally conservative so the user can sweep them.
DEFAULT_INVENTORY_BINS = [0, 30, 60, 90, 130, MAX_INVENTORY]
DEFAULT_PENDING_BINS = [0, 1, 40, MAX_PENDING]
DEFAULT_LEAD_TIME_BINS = [0, 1, LEAD_TIME_DAYS + 1]
DEFAULT_DAYS_REMAINING_BINS = [0, 2, 7, 15, EPISODE_DAYS + 1]

DEFAULT_BINS = {
    "inventory": DEFAULT_INVENTORY_BINS,
    "pending": DEFAULT_PENDING_BINS,
    "lead_time": DEFAULT_LEAD_TIME_BINS,
    "days_remaining": DEFAULT_DAYS_REMAINING_BINS,
}

N_DAY_TYPE = 2
N_ACTIONS = len(ORDER_OPTIONS)

Q_SHAPE = (
    len(DEFAULT_INVENTORY_BINS) - 1,
    N_DAY_TYPE,
    len(DEFAULT_PENDING_BINS) - 1,
    len(DEFAULT_LEAD_TIME_BINS) - 1,
    len(DEFAULT_DAYS_REMAINING_BINS) - 1,
    N_ACTIONS,
)


def build_q_shape(bins: dict[str, list[int]]) -> tuple[int, ...]:
    return (
        len(bins["inventory"]) - 1,       # on-hand inventory bins
        N_DAY_TYPE,                       # weekday/weekend flag
        len(bins["pending"]) - 1,         # in-transit quantity bins
        len(bins["lead_time"]) - 1,       # lead-time bins
        len(bins["days_remaining"]) - 1,  # episode-timing bins
        N_ACTIONS,                        # available ordering actions
    )


def discretize(obs: np.ndarray, bins: dict[str, list[int]] | None = None) -> tuple:
    """Map the raw environment observation into a smaller discrete state index."""
    bins = bins or DEFAULT_BINS

    inventory_bin = np.digitize(obs[0], bins["inventory"][1:])
    is_weekend = int(obs[1] in (5, 6))
    pending_bin = np.digitize(obs[3], bins["pending"][1:])
    lead_bin = np.digitize(obs[4], bins["lead_time"][1:])
    days_remaining_bin = np.digitize(obs[6], bins["days_remaining"][1:])

    inventory_bin = min(inventory_bin, len(bins["inventory"]) - 2)
    pending_bin = min(pending_bin, len(bins["pending"]) - 2)
    lead_bin = min(lead_bin, len(bins["lead_time"]) - 2)
    days_remaining_bin = min(days_remaining_bin, len(bins["days_remaining"]) - 2)

    return (inventory_bin, is_weekend, pending_bin, lead_bin, days_remaining_bin)


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning with epsilon-greedy exploration.

    Hyperparameters (chosen conservatively for a 30-day horizon):
        alpha   : learning rate      — how fast Q-values update
        gamma   : discount factor    — how much future rewards matter
        epsilon : exploration rate   — starts high, decays over training
    """

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        bins: dict[str, list[int]] | None = None,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.bins = bins or DEFAULT_BINS
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros(build_q_shape(self.bins))

    def _available_actions(self, obs: np.ndarray) -> list[int]:
        """
        Restrict clearly dominated actions:
        - if an order is already in transit, extra order actions have no effect
        """
        pending_order_qty = int(obs[3])

        if pending_order_qty > 0:
            return [0]
        return list(range(N_ACTIONS))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Use epsilon-greedy logic: explore randomly during training, exploit best action during evaluation."""
        available_actions = self._available_actions(obs)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))
        state = discretize(obs, self.bins)
        q_values = self.Q[state]
        best_local_idx = int(np.argmax(q_values[available_actions]))
        return int(available_actions[best_local_idx])

    # ------------------------------------------------------------------
    # Q-update (single step)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        s = discretize(obs, self.bins)
        s_ = discretize(next_obs, self.bins)

        current_q = self.Q[s][action]
        next_actions = self._available_actions(next_obs)
        target_q = reward + (
            0.0 if done else self.gamma * np.max(self.Q[s_][next_actions])
        )
        self.Q[s][action] += self.alpha * (target_q - current_q)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    # Gradually reduce exploration so the agent relies more on learned behavior over time
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_episodes: int = 3000,
    agent_kwargs: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[QLearningAgent, list[float]]:
    """
    Train a Q-learning agent and return (agent, episode_reward_history).
    """
    env = InventoryEnv(seed=seed)
    agent = QLearningAgent(seed=seed, **(agent_kwargs or {}))

    reward_history = []
    log_every = max(1, n_episodes // 10)  

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for _ in range(EPISODE_DAYS):
            # Select an action, apply it to the environment, and observe the next state and reward
            action = agent.select_action(obs)
            next_obs, reward, done, trunc, _  = env.step(action)
            agent.update(obs, action, reward, next_obs, done or trunc)

            # Move to the next state and accumulate episode reward
            obs = next_obs
            total_reward += reward

            if done or trunc:
                break

        agent.decay_epsilon()
        reward_history.append(total_reward)

        if verbose and (ep + 1) % log_every == 0:
            recent_mean = np.mean(reward_history[-log_every:])
            print(f"  Episode {ep+1:>5d}/{n_episodes} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"mean reward (last {log_every}): ${recent_mean:.2f}")

    return agent, reward_history


def tune_q_learning(
    candidate_configs: list[dict] | None = None,
    train_episodes: int = 1500,
    validation_episodes: int = 100,
    train_seed: int = 42,
    validation_seed: int = 100_000,
) -> tuple[dict, dict]:
    """
    Sweep a small set of agent configurations on a held-out validation split.
    """
    if candidate_configs is None:
        candidate_configs = [
            {
                "name": "default",
                "alpha": 0.2,
                "gamma": 0.95,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.995,
            },
            {
                "name": "lower_lr",
                "alpha": 0.1,
                "gamma": 0.95,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.995,
            },
            {
                "name": "higher_gamma",
                "alpha": 0.1,
                "gamma": 0.98,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.997,
            },
            {
                "name": "coarser_days",
                "alpha": 0.2,
                "gamma": 0.95,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.995,
                "bins": {
                    "inventory": DEFAULT_INVENTORY_BINS,
                    "pending": DEFAULT_PENDING_BINS,
                    "lead_time": DEFAULT_LEAD_TIME_BINS,
                    "days_remaining": [0, 3, 10, EPISODE_DAYS + 1],
                },
            },
            {
                "name": "finer_state",
                "alpha": 0.1,
                "gamma": 0.95,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.995,
                "bins": {
                    "inventory": [0, 20, 40, 60, 80, 110, 150, MAX_INVENTORY],
                    "pending": [0, 1, 20, 60, MAX_PENDING],
                    "lead_time": DEFAULT_LEAD_TIME_BINS,
                    "days_remaining": [0, 2, 5, 10, 20, EPISODE_DAYS + 1],
                },
            },
        ]

    best_config = None
    best_results = None
    best_reward = -np.inf

    print("\n" + "=" * 50)
    print("  Validation Sweep")
    print("=" * 50)

    for i, config in enumerate(candidate_configs, start=1):
        config_name = config.get("name", f"config_{i}")
        agent_kwargs = {k: v for k, v in config.items() if k != "name"}
        agent, _ = train(
            n_episodes=train_episodes,
            agent_kwargs=agent_kwargs,
            seed=train_seed,
            verbose=False,
        )
        results = evaluate_policy(
            GreedyAgentPolicy(agent),
            n_episodes=validation_episodes,
            seed=validation_seed,
        )
        print(
            f"  {config_name:<12} | "
            f"mean reward ${results['mean_reward']:.2f} | "
            f"stockouts {results['mean_stockouts']:.1f} | "
            f"holding ${results['mean_holding_cost']:.2f}"
        )

        if results["mean_reward"] > best_reward:
            best_reward = results["mean_reward"]
            best_config = config
            best_results = results

    return best_config, best_results


# ---------------------------------------------------------------------------
# Greedy eval wrapper (matches evaluate_policy interface in baseline.py)
# ---------------------------------------------------------------------------

class GreedyAgentPolicy:
    """Wrap the trained agent so it can be evaluated like the baseline policies."""
    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def select_action(self, obs: np.ndarray) -> int:
        return self.agent.select_action(obs, greedy=True)   # Always choose the best learned action during evaluation


# ---------------------------------------------------------------------------
# Main: train + compare against baselines
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    TRAIN_EPISODES = 3000
    EVAL_EPISODES = 200
    TRAIN_SEED = 42
    TEST_SEED = 200_000

    print("=" * 50)
    print("  Training Q-learning agent ...")
    print(f"  Episodes: {TRAIN_EPISODES}  |  Eval episodes: {EVAL_EPISODES}")
    print("=" * 50)

    best_config, validation_results = tune_q_learning(
        train_episodes=1500,
        validation_episodes=100,
        train_seed=TRAIN_SEED,
        validation_seed=100_000,
    )
    print(
        f"\nBest validation config: {best_config.get('name', 'unnamed')} | "
        f"mean reward ${validation_results['mean_reward']:.2f}"
    )

    agent_kwargs = {k: v for k, v in best_config.items() if k != "name"}
    agent, reward_history = train(
        n_episodes=TRAIN_EPISODES,
        agent_kwargs=agent_kwargs,
        seed=TRAIN_SEED,
    )

    # Save the learned Q-table and training rewards for later reuse
    save_path = Path("q_table.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"Q": agent.Q, "reward_history": reward_history}, f)
    print(f"\nQ-table saved to {save_path}")

    # --- Plot reward curve with 100-episode moving average (show whether learning improved over time) ---
    window = 100
    moving_avg = np.convolve(reward_history, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(range(window - 1, len(reward_history)),
             moving_avg, color="steelblue", linewidth=2, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward ($)")
    plt.title("Q-Learning Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150)
    plt.close()
    print("Reward curve saved to reward_curve.png")

    # --- Evaluate all three policies on the same seeds ---
    env = InventoryEnv(seed=TEST_SEED)

    random_results = evaluate_policy(RandomPolicy(env),
                                     n_episodes=EVAL_EPISODES, seed=TEST_SEED)
    reorder_results = evaluate_policy(ReorderPointPolicy(reorder_point=60, order_up_to=120),
                                      n_episodes=EVAL_EPISODES, seed=TEST_SEED)
    tuned_reorder_config, tuned_reorder_results = tune_reorder_policy(
        n_episodes=EVAL_EPISODES,
        seed=TEST_SEED,
    )
    agent_results = evaluate_policy(GreedyAgentPolicy(agent),
                                    n_episodes=EVAL_EPISODES, seed=TEST_SEED)

    print_results("Random Policy (lower bound)", random_results)
    print_results("Reorder-Point s=60 S=120 (rule-based)", reorder_results)
    print_results(
        f"Best tuned reorder s={tuned_reorder_config['reorder_point']} "
        f"S={tuned_reorder_config['order_up_to']}",
        tuned_reorder_results,
    )
    print_results("Q-Learning Agent (trained)", agent_results)

    print("\n" + "=" * 50)
    print("  Δ vs Random : "
          f"${agent_results['mean_reward'] - random_results['mean_reward']:+.2f}\n")
    print("  Δ vs Reorder-Point : "
          f"${agent_results['mean_reward'] - reorder_results['mean_reward']:+.2f}\n")
    print("  Δ vs Best Tuned Reorder : "
          f"${agent_results['mean_reward'] - tuned_reorder_results['mean_reward']:+.2f}\n")
    print("  The RL agent balanced stockout penalties against holding costs:")
    print(f"    Ordering cost — Agent: ${agent_results['mean_ordering_cost']:.2f}"
          f"  vs Reorder: ${reorder_results['mean_ordering_cost']:.2f}")
    print(f"    Holding cost — Agent: ${agent_results['mean_holding_cost']:.2f}"
          f"  vs Reorder: ${reorder_results['mean_holding_cost']:.2f}")
    print(f"    Stockouts/ep — Agent: {agent_results['mean_stockouts']:.1f}"
          f"  vs Reorder: {reorder_results['mean_stockouts']:.1f}")
    print("=" * 50)
