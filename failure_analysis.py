"""
Four failure modes:
  1. Reward hacking    — agent exploits reward structure in unintended ways
  2. Unsafe behavior   — agent causes stockouts by under-ordering
  3. Instability       — Q-values diverge or oscillate under bad hyperparameters
  4. Overfitting       — agent trained on one demand pattern fails on another
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from env import (
    InventoryEnv, ORDER_OPTIONS, EPISODE_DAYS,
    HOLDING_COST, STOCKOUT_PENALTY, ORDERING_COST, DEMAND_LAMBDA
)
from baseline import evaluate_policy, ReorderPointPolicy
from q_agent import QLearningAgent, GreedyAgentPolicy, train

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

SEED = 42


# ---------------------------------------------------------------------------
# Load the trained Q-learning agent and its reward history from disk
# ---------------------------------------------------------------------------

def load_agent() -> tuple[QLearningAgent, list[float]]:
    path = Path("q_table.pkl")
    if not path.exists():
        raise FileNotFoundError("q_table.pkl not found. Run 'python q_agent.py' first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent(seed=SEED)
    agent.Q = data["Q"]
    agent.epsilon = 0.0
    return agent, data["reward_history"]


# ---------------------------------------------------------------------------
# Failure 1: Reward Hacking
# ---------------------------------------------------------------------------
# The reward is: revenue - holding - stockout - ordering.
# 
# If the stockout penalty is too small, the agent may learn to keep
# inventory levels very low to avoid holding costs, accepting stockouts
# as a cheaper alternative. This is a form of reward hacking because the
# agent exploits the reward design rather than achieving the true business goal.

def analyse_reward_hacking(agent: QLearningAgent):
    print("\n--- Failure 1: Reward Hacking ---")

    # Demonstrate the under-stock exploit: reduce stockout penalty to $1 and retrain a fresh agent
    class HackedEnv(InventoryEnv):
        """Same env but with artificially low stockout penalty."""
        def _simulate_day(self, units_to_order):
            profit, info = super()._simulate_day(units_to_order)
            # Replace the original stockout penalty with a much smaller penalty
            original_penalty = info["units_lost"] * STOCKOUT_PENALTY
            reduced_penalty = info["units_lost"] * 1.0          # $1 instead of $5
            adjustment = original_penalty - reduced_penalty
            profit += adjustment
            info["stockout_penalty"] = reduced_penalty
            return profit, info

    # Train a fresh Q-learning agent under the weakened reward setting
    hacked_env = HackedEnv(seed=SEED)
    hacked_agent = QLearningAgent(seed=SEED, epsilon_start=1.0)

    for ep in range(500):
        obs, _ = hacked_env.reset(seed=SEED + ep)
        for _ in range(EPISODE_DAYS):
            action = hacked_agent.select_action(obs)
            next_obs, reward, done, _, _ = hacked_env.step(action)
            hacked_agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break
        hacked_agent.decay_epsilon()

    # Measure order-0 rate GREEDILY after training (not during — that would
    # include random exploration and misrepresent learned behaviour)
    zero_order_steps = 0
    total_steps = 0
    for ep in range(100):
        obs, _ = hacked_env.reset(seed=SEED + 1000 + ep)
        for _ in range(EPISODE_DAYS):
            action = hacked_agent.select_action(obs, greedy=True)

            # Count how often the learned policy chooses to order nothing
            if ORDER_OPTIONS[action] == 0:
                zero_order_steps += 1

            total_steps += 1
            obs, _, done, _, _ = hacked_env.step(action)
            if done:
                break

    zero_order_rate = zero_order_steps / total_steps

    print(f"  With reduced stockout penalty ($1): agent chose 'order 0' "
          f"{zero_order_rate:.1%} of steps (exploitation of weak penalty).")
    print("  Mitigation: calibrate STOCKOUT_PENALTY to reflect true lost-sale cost.")
    print("  In production: monitor inventory levels — sustained near-zero stock")
    print("  is a red flag that the agent is gaming the reward.")

    return zero_order_rate


# ---------------------------------------------------------------------------
# Failure 2: Unsafe Behaviour — chronic stockouts
# ---------------------------------------------------------------------------
# The agent is trained on Poisson(λ=20) weekday / Poisson(λ=25) weekend demand.
# If real demand is higher (e.g. a sales promotion, holiday season), the agent
# will chronically under-order and cause stockouts.

def analyse_unsafe_behaviour(agent: QLearningAgent):
    print("\n--- Failure 2: Unsafe Behaviour (demand surge) ---")

    class SurgeEnv(InventoryEnv):
        """Demand is 50% higher than training distribution."""
        def _simulate_day(self, units_to_order):
            # Temporarily boost lambda
            original_lam = 25 if self.day_of_week in [5, 6] else 20
            # Increase the demand rate to simulate a promotion or holiday demand surge
            surge_lam = int(original_lam * 1.5)
            # Generate customer demand from the higher-demand distribution
            demand = int(self.np_rng.poisson(lam=surge_lam))

            # Manually process order arrival, new order placement, demand fulfillment, and reward calculation
            ordering_cost = 0.0
            if self.lead_time_left > 0:
                self.lead_time_left -= 1
                if self.lead_time_left == 0:
                    from env import MAX_INVENTORY
                    self.inventory = min(self.inventory + self.pending_order, MAX_INVENTORY)
                    self.pending_order = 0
            if units_to_order > 0 and self.pending_order == 0:
                self.pending_order = units_to_order
                self.lead_time_left = 2
                ordering_cost = ORDERING_COST

            self.demand_history.append(demand)
            if len(self.demand_history) > 7:
                self.demand_history.pop(0)

            units_sold = min(demand, self.inventory)
            units_lost = demand - units_sold
            self.inventory -= units_sold
            self.inventory = max(self.inventory, 0)

            revenue = units_sold  * 10.0
            holding_cost = self.inventory * HOLDING_COST
            stockout_penalty = units_lost * STOCKOUT_PENALTY
            profit = revenue - holding_cost - stockout_penalty - ordering_cost

            info = {
                "demand": demand, "units_sold": units_sold, "units_lost": units_lost,
                "revenue": revenue, "holding_cost": holding_cost,
                "stockout_penalty": stockout_penalty, "ordering_cost": ordering_cost,
                "profit": profit,
            }
            return profit, info

    normal_env = InventoryEnv(seed=SEED)
    surge_env  = SurgeEnv(seed=SEED)

    # Evaluate normal demand performance using the original environment
    normal_results = evaluate_policy(GreedyAgentPolicy(agent), n_episodes=100,
                                     seed=SEED)
    
    # Evaluate the trained agent manually under surge-demand conditions
    surge_stockouts = []
    for ep in range(100):
        obs, _ = surge_env.reset(seed=SEED + ep)
        ep_stockouts = 0
        for _ in range(EPISODE_DAYS):
            action = agent.select_action(obs, greedy=True)
            obs, _, done, _, info = surge_env.step(action)
            ep_stockouts += info["units_lost"]
            if done:
                break
        surge_stockouts.append(ep_stockouts)

    print(f"  Normal demand  — mean stockouts/ep: {normal_results['mean_stockouts']:.1f}")
    print(f"  Surge  demand  — mean stockouts/ep: {np.mean(surge_stockouts):.1f}")
    print("  Mitigation: add a demand-surge safety buffer (e.g. reorder point")
    print("  scales with rolling_avg_demand), or retrain on a wider demand distribution.")

    return normal_results["mean_stockouts"], float(np.mean(surge_stockouts))


# ---------------------------------------------------------------------------
# Failure 3: Training Instability
# ---------------------------------------------------------------------------
# A learning rate that is too high causes Q-values to overshoot and oscillate.
# Training with alpha=0.9 is demonstrated and plotted the reward curve
# against the well-tuned alpha=0.1 agent.

def analyse_instability():
    print("\n--- Failure 3: Training Instability (high learning rate) ---")

    N = 800 

    # Train the baseline stable agent using the default learning rate
    _, stable_history = train(n_episodes=N, seed=SEED, verbose=False)

    # Train a second agent with a very high learning rate to demonstrate instability
    from env import InventoryEnv as _Env
    from q_agent import QLearningAgent as _Agent
    env = _Env(seed=SEED)
    unstable_agent = _Agent(seed=SEED, alpha=0.9)
    unstable_history = []
    for ep in range(N):
        obs, _ = env.reset(seed=SEED + ep)
        total = 0.0
        for _ in range(EPISODE_DAYS):
            action = unstable_agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            unstable_agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total += reward
            if done:
                break
        unstable_agent.decay_epsilon()
        unstable_history.append(total)

    window = 50

    # Smooth noisy reward histories using a moving average
    def smooth(h):
        return np.convolve(h, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smooth(stable_history), color="#f72585", linewidth=2, label="α=0.1 (stable)")
    ax.plot(smooth(unstable_history), color="#4361ee", linewidth=2,
            linestyle="--", label="α=0.9 (unstable)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Training Instability: Effect of Learning Rate", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig("instability.png")
    plt.close(fig)

    stable_std = float(np.std(stable_history[-200:]))
    unstable_std = float(np.std(unstable_history[-200:]))
    print(f"  α=0.1 — reward std (last 200 eps): {stable_std:.1f}")
    print(f"  α=0.9 — reward std (last 200 eps): {unstable_std:.1f}")
    print("  Mitigation: use a small, fixed learning rate or a learning rate schedule.")

    return stable_std, unstable_std


# ---------------------------------------------------------------------------
# Failure 4: Overfitting to Training Demand Distribution
# ---------------------------------------------------------------------------
# The agent is trained on the original demand distribution but tested on
# a different seasonal demand pattern. Performance degradation indicates
# that the learned policy may not generalize well to unseen conditions.

# Test whether the agent generalizes to a different demand distribution
def analyse_overfitting(agent: QLearningAgent):
    print("\n--- Failure 4: Overfitting to Training Distribution ---")

    class SeasonalEnv(InventoryEnv):
        """Demand follows a sinusoidal weekly pattern instead of Poisson."""
        def _simulate_day(self, units_to_order):
            # Generate demand using a weekly sinusoidal pattern instead of the original pattern
            seasonal_lam = int(20 + 10 * np.sin(2 * np.pi * self.day_of_week / 7))
            seasonal_lam = max(seasonal_lam, 5)   # floor at 5 units/day
            demand = int(self.np_rng.poisson(lam=seasonal_lam))

            ordering_cost = 0.0
            if self.lead_time_left > 0:
                self.lead_time_left -= 1
                if self.lead_time_left == 0:
                    from env import MAX_INVENTORY
                    self.inventory = min(self.inventory + self.pending_order, MAX_INVENTORY)
                    self.pending_order = 0
            if units_to_order > 0 and self.pending_order == 0:
                self.pending_order = units_to_order
                self.lead_time_left = 2
                ordering_cost = ORDERING_COST

            self.demand_history.append(demand)
            if len(self.demand_history) > 7:
                self.demand_history.pop(0)

            units_sold = min(demand, self.inventory)
            units_lost = demand - units_sold
            self.inventory -= units_sold
            self.inventory = max(self.inventory, 0)

            revenue = units_sold  * 10.0
            holding_cost = self.inventory * HOLDING_COST
            stockout_penalty = units_lost  * STOCKOUT_PENALTY
            profit = revenue - holding_cost - stockout_penalty - ordering_cost

            info = {
                "demand": demand, "units_sold": units_sold, "units_lost": units_lost,
                "revenue": revenue, "holding_cost": holding_cost,
                "stockout_penalty": stockout_penalty, "ordering_cost": ordering_cost,
                "profit": profit,
            }
            return profit, info

    # Compare agent vs reorder-point on seasonal env
    seasonal_env = SeasonalEnv(seed=SEED)
    agent_seasonal = []
    reorder_seasonal = []
    reorder_policy = ReorderPointPolicy(reorder_point=60, order_up_to=120)

    for ep in range(100):
        # Agent
        obs, _ = seasonal_env.reset(seed=SEED + ep)
        total = 0.0
        for _ in range(EPISODE_DAYS):
            action = agent.select_action(obs, greedy=True)
            obs, reward, done, _, _ = seasonal_env.step(action)
            total += reward
            if done:
                break
        agent_seasonal.append(total)

        # Reorder-point
        obs, _ = seasonal_env.reset(seed=SEED + ep)
        total  = 0.0
        for _ in range(EPISODE_DAYS):
            action = reorder_policy.select_action(obs)
            obs, reward, done, _, _ = seasonal_env.step(action)
            total += reward
            if done:
                break
        reorder_seasonal.append(total)

    agent_mean = float(np.mean(agent_seasonal))
    reorder_mean = float(np.mean(reorder_seasonal))

    print(f"  Agent mean reward on seasonal demand   : ${agent_mean:.2f}")
    print(f"  Reorder-point mean reward on seasonal  : ${reorder_mean:.2f}")

    # Compare average rewards to check whether the learned agent overfits the original simulator
    delta = agent_mean - reorder_mean
    if delta < 0:
        print(f"  Agent underperforms rule-based by ${abs(delta):.2f} — overfitting confirmed.")
    else:
        print(f"  Agent still outperforms by ${delta:.2f} — robust to this distribution shift.")
    print("  Mitigation: train on a mixture of demand distributions (domain randomisation),")
    print("  or add a safety override that triggers the reorder-point policy when")
    print("  rolling_avg_demand deviates significantly from training distribution.")

    return agent_mean, reorder_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Failure Analysis — Inventory Replenishment Agent")
    print("=" * 55)

    agent, reward_history = load_agent()

    zero_order_rate = analyse_reward_hacking(agent)
    normal_stockouts, surge_stockouts = analyse_unsafe_behaviour(agent)
    stable_std, unstable_std = analyse_instability()
    agent_seasonal, reorder_seasonal = analyse_overfitting(agent)

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  1. Reward hacking   — 'order 0' rate under weak penalty: {zero_order_rate:.1%}")
    print(f"  2. Unsafe behaviour — stockouts surge vs normal: "
          f"{surge_stockouts:.1f} vs {normal_stockouts:.1f} units/ep")
    print(f"  3. Instability      — reward std α=0.9 vs α=0.1: "
          f"{unstable_std:.1f} vs {stable_std:.1f}")
    print(f"  4. Overfitting      — agent vs reorder on seasonal demand: "
          f"${agent_seasonal:.0f} vs ${reorder_seasonal:.0f}")