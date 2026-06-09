"""
Two baselines:
  1. RandomPolicy       — orders a random quantity each day (lower bound)
  2. ReorderPointPolicy — classic (s, S) rule: if inventory <= s, order up to S
"""

import numpy as np
from env import InventoryEnv, ORDER_OPTIONS, EPISODE_DAYS


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

class RandomPolicy:
    """
    Random policy baseline used as a lower-performance benchmark
    """
    def __init__(self, env: InventoryEnv):
        self.env = env

    def select_action(self, obs: np.ndarray) -> int:
        return self.env.action_space.sample()


class ReorderPointPolicy:
    """
    Rule-based inventory policy using a classic (s, S) replenishment strategy

    Rule:
      - If inventory_level <= reorder_point (s) AND no order is pending:
            order enough to bring expected stock up to target level (S), rounded to the nearest available ORDER_OPTION.
      - Otherwise: do nothing (action = 0).

    Obs indices:
        0 = inventory_level
        1 = day_of_week
        2 = rolling_avg_demand * 10
        3 = pending_order_qty
        4 = lead_time_left
        5 = inventory_position
        6 = days_remaining
    """

    def __init__(self, reorder_point: int = 40, order_up_to: int = 80):
        """
        Args:
            reorder_point: trigger an order when inventory falls to or below this level.
                           Default 40 ≈ 2 days of mean weekday demand (20 units/day).
            order_up_to:   target inventory level to order toward.
                           Default 80 ≈ 4 days of mean demand, giving a buffer.
        """
        self.reorder_point = reorder_point
        self.order_up_to   = order_up_to

    def select_action(self, obs: np.ndarray) -> int:
        # Extract current inventory information from the environment state
        inventory_level = obs[0]
        pending_order_qty = obs[3]
        inventory_position = obs[5]

        # Avoid placing a new order when inventory is already in transit
        if pending_order_qty > 0:
            return 0    # action 0 = order nothing

        # Only replenish inventory when stock falls below the reorder threshold
        if inventory_level > self.reorder_point:
            return 0

        # Calculate the quantity needed to reach the target inventory level
        desired_order = self.order_up_to - inventory_position
        desired_order = max(desired_order, 0)

        # Pick the closest available order quantity (without exceeding desired)
        best_action = 0
        for i, qty in enumerate(ORDER_OPTIONS):
            if qty <= desired_order:
                best_action = i   # take the largest feasible option
        return best_action


# Search for the best-performing (s, S) inventory policy by evaluating multiple reorder-point and order-up-to combinations
def tune_reorder_policy(
    reorder_points: list[int] | None = None,
    order_up_to_levels: list[int] | None = None,
    n_episodes: int = 200,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Grid-search a simple (s, S) baseline so the RL agent is compared against
    a strong heuristic rather than an arbitrary one-off setting.
    """
    reorder_points = reorder_points or list(range(20, 91, 10))
    order_up_to_levels = order_up_to_levels or list(range(60, 151, 10))

    best_config = None
    best_results = None
    best_reward = -np.inf

    for reorder_point in reorder_points:
        for order_up_to in order_up_to_levels:
            if order_up_to <= reorder_point:
                continue

            results = evaluate_policy(
                ReorderPointPolicy(reorder_point=reorder_point, order_up_to=order_up_to),
                n_episodes=n_episodes,
                seed=seed,
            )

            # Keep the configuration that achieves the highest average episode reward
            if results["mean_reward"] > best_reward:
                best_reward = results["mean_reward"]
                best_config = {
                    "reorder_point": reorder_point,
                    "order_up_to": order_up_to,
                }
                best_results = results

    return best_config, best_results


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_policy(policy, n_episodes: int = 100, seed: int = 0) -> dict:
    """
    Run `policy` for `n_episodes` and return summary statistics.

    Returns a dict with keys:
        mean_reward, std_reward, mean_stockouts, mean_units_sold, episode_rewards
    """
    env = InventoryEnv(seed=seed)
    episode_rewards   = []
    episode_stockouts = []
    episode_sold      = []
    episode_ordering  = []
    episode_holding   = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        total_stockout = 0
        total_sold = 0
        total_ordering = 0.0
        total_holding = 0.0

        for _ in range(EPISODE_DAYS):
            # Select an action using the policy and apply it to the environment
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_stockout += info["units_lost"]
            total_sold += info["units_sold"]
            total_ordering += info["ordering_cost"]
            total_holding += info["holding_cost"]
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_stockouts.append(total_stockout)
        episode_sold.append(total_sold)
        episode_ordering.append(total_ordering)
        episode_holding.append(total_holding)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_stockouts": float(np.mean(episode_stockouts)),
        "mean_units_sold": float(np.mean(episode_sold)),
        "mean_ordering_cost": float(np.mean(episode_ordering)),
        "mean_holding_cost": float(np.mean(episode_holding)),
        "episode_rewards": episode_rewards,
    }


def print_results(name: str, results: dict) -> None:
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Mean episode reward : ${results['mean_reward']:>8.2f}")
    print(f"  Std  episode reward : ${results['std_reward']:>8.2f}")
    print(f"  Mean units sold/ep  :  {results['mean_units_sold']:>7.1f}")
    print(f"  Mean stockouts/ep   :  {results['mean_stockouts']:>7.1f}")
    print(f"  Mean ordering cost  : ${results['mean_ordering_cost']:>8.2f}")
    print(f"  Mean holding cost   : ${results['mean_holding_cost']:>8.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_EPISODES = 200
    TEST_SEED = 200_000

    env = InventoryEnv(seed=TEST_SEED)

    # 1. Random policy
    random_policy = RandomPolicy(env)
    random_results = evaluate_policy(random_policy, n_episodes=N_EPISODES, seed=TEST_SEED)
    print_results("Random Policy (lower bound)", random_results)

    # 2. Rule-based (s, S) policy
    reorder_policy = ReorderPointPolicy(reorder_point=60, order_up_to=120)
    reorder_results = evaluate_policy(reorder_policy, n_episodes=N_EPISODES, seed=TEST_SEED)
    print_results("Reorder-Point Policy  s=60, S=120  (rule-based)", reorder_results)

    best_config, best_results = tune_reorder_policy(
        n_episodes=N_EPISODES,
        seed=TEST_SEED,
    )
    print_results(
        f"Best tuned reorder policy  s={best_config['reorder_point']}, S={best_config['order_up_to']}",
        best_results,
    )

    # Summary comparison
    print(f"\n{'='*45}")
    print("  Improvement: reorder-point vs random")
    delta = reorder_results["mean_reward"] - random_results["mean_reward"]
    print(f"  Δ mean reward = ${delta:+.2f} per episode")
    print(f"{'='*45}\n")
