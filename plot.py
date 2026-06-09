"""
Plots produced:
  1. reward_curve.png       — Q-learning training reward + 100-ep moving average
  2. policy_comparison.png  — Mean reward ± std for all three policies
  3. cost_breakdown.png     — Stacked cost components per policy
  4. behavior_episode.png   — One greedy episode: inventory, demand, orders over 30 days
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
 
from env import InventoryEnv, ORDER_OPTIONS, EPISODE_DAYS, STOCKOUT_PENALTY
from baseline import evaluate_policy, ReorderPointPolicy, RandomPolicy
from q_agent import QLearningAgent, GreedyAgentPolicy, Q_SHAPE

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})
 
COLORS = {
    "random": "#adb5bd", 
    "reorder": "#4895ef", 
    "agent": "#f72585", 
}
 
SEED = 42
EVAL_EPISODES = 200


# ---------------------------------------------------------------------------
# Load trained Q-learning agent
# ---------------------------------------------------------------------------

def load_agent() -> tuple[QLearningAgent, list[float]]:
    path = Path("q_table.pkl")
    if not path.exists():
        raise FileNotFoundError(
            "q_table.pkl not found. Run 'python q_agent.py' first to train the agent."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent(seed=SEED)
    agent.Q = data["Q"]
    agent.epsilon = 0.0   # fully greedy at eval time
    return agent, data["reward_history"]


# ---------------------------------------------------------------------------
# Plot 1: Training reward curve
# ---------------------------------------------------------------------------

# Visualize learning progress throughout training
def plot_reward_curve(reward_history: list[float], save_path: str = "reward_curve.png"):
    window = 100
 
    # Smooth noisy episode rewards using a moving average
    moving_avg = np.convolve(reward_history, np.ones(window) / window, mode="valid")
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_history, alpha=0.2, color=COLORS["agent"], label="Episode reward")
    ax.plot(
        range(window - 1, len(reward_history)),
        moving_avg,
        color=COLORS["agent"], linewidth=2,
        label=f"{window}-episode moving average",
    )
 
    ax.axhline(moving_avg[-1], linestyle="--", color="black", linewidth=0.8,
               label=f"Final avg: ${moving_avg[-1]:.0f}")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Episode Reward ($)")
    ax.set_title("Q-Learning Training Reward Curve", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Policy comparison — mean reward ± 1 std
# ---------------------------------------------------------------------------

# Compare average episode rewards across all policies
def plot_policy_comparison(results: dict, save_path: str = "policy_comparison.png"):
    labels = ["Random\n(lower bound)", "Reorder-Point\n(rule-based)", "Q-Learning\n(trained)"]
    means = [results[k]["mean_reward"] for k in ["random", "reorder", "agent"]]
    stds = [results[k]["std_reward"]  for k in ["random", "reorder", "agent"]]
    colors = [COLORS["random"], COLORS["reorder"], COLORS["agent"]]
 
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=6,
                  error_kw={"linewidth": 1.5}, width=0.5)
 
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 10,
            f"${mean:.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
 
    ax.set_ylabel("Mean Episode Reward ($)")
    ax.set_title("Policy Comparison — Mean Reward ± 1 Std\n"
                 f"({EVAL_EPISODES} evaluation episodes)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Cost breakdown (stacked bar) per policy
# ---------------------------------------------------------------------------

# Compare the major cost components incurred by each policy
def plot_cost_breakdown(results: dict, save_path: str = "cost_breakdown.png"):
    policies = [
        "Random\n(lower bound)",
        "Reorder-Point\ns=60 S=120",
        "Reorder-Point\ns=70 S=130\n(best tuned)",
        "Q-Learning\n(trained)",
    ]
    keys = ["random", "reorder", "tuned", "agent"]
 
    ordering = [results[k]["mean_ordering_cost"] for k in keys]
    holding  = [results[k]["mean_holding_cost"]  for k in keys]
    stockouts = [results[k]["mean_stockouts"] * STOCKOUT_PENALTY for k in keys]
    reward   = [results[k]["mean_reward"]        for k in keys]
 
    x = np.arange(len(policies))
    w = 0.45
 
    fig, ax = plt.subplots(figsize=(11, 6))
 
    ax.bar(x, [-o for o in ordering], width=w, label="Ordering cost", color="#4361ee")
    ax.bar(x, [-h for h in holding], width=w, bottom=[-o for o in ordering],
           label="Holding cost", color="#7209b7")
    ax.bar(x, [-s for s in stockouts], width=w,
           bottom=[-o - h for o, h in zip(ordering, holding)],
           label="Stockout penalty", color="#e63946")
 
    ax.plot(x, reward, "o--", color="black", linewidth=1.5,
            zorder=5, label="Net reward")
 
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=9)
    ax.set_ylabel("$ per episode")
    ax.set_title("Cost Breakdown by Policy", fontweight="bold")
 
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=True)
 
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4: One greedy episode — inventory, demand, and orders over 30 days
# ---------------------------------------------------------------------------

# Visualize the decisions made by the trained agent during one episode
def plot_behavior_episode(agent: QLearningAgent, save_path: str = "behavior_episode.png"):
    """
    Runs one greedy episode and plots:
      - Inventory level over time
      - Daily demand
      - Order quantities placed
    """
    env = InventoryEnv(seed=SEED + 999)   # fresh seed, not seen during training
    obs, _ = env.reset(seed=SEED + 999)
 
    days = []
    inventories = []
    demands = []
    orders = []
 
    # Run one episode using the learned policy
    for day in range(EPISODE_DAYS):
        action = agent.select_action(obs, greedy=True)
        units_ordered = ORDER_OPTIONS[action]
        obs, _, done, _, info = env.step(action)
 
        days.append(day + 1)
        inventories.append(env.inventory)
        demands.append(info["demand"])
        orders.append(units_ordered)
 
        if done:
            break
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
 
    # Top panel: inventory vs demand
    ax1.plot(days, inventories, color=COLORS["agent"], linewidth=2, label="Inventory level")
    ax1.fill_between(days, inventories, alpha=0.15, color=COLORS["agent"])
    ax1.plot(days, demands, color=COLORS["reorder"], linewidth=1.5,
             linestyle="--", label="Daily demand")
    ax1.axhline(40, color="grey", linestyle=":", linewidth=1, label="Reorder point (s=40)")
    ax1.set_ylabel("Units")
    ax1.set_title("Greedy Q-Agent Behavior — Single Episode", fontweight="bold")
    ax1.legend(loc="upper right")
 
    # Bottom panel: orders placed
    ax2.bar(days, orders, color=COLORS["agent"], alpha=0.8, label="Units ordered")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Units Ordered")
    ax2.set_title("Orders Placed by Agent", fontweight="bold")
    ax2.legend()
 
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading trained agent ...")
    agent, reward_history = load_agent()
 
    print("Evaluating policies ...")
    env = InventoryEnv(seed=SEED)
    results = {
        "random": evaluate_policy(RandomPolicy(env),
                                  n_episodes=EVAL_EPISODES, seed=SEED),
        "reorder": evaluate_policy(ReorderPointPolicy(reorder_point=60, order_up_to=120),
                                   n_episodes=EVAL_EPISODES, seed=SEED),
        "tuned": evaluate_policy(ReorderPointPolicy(reorder_point=70, order_up_to=130),
                                 n_episodes=EVAL_EPISODES, seed=SEED),
        "agent": evaluate_policy(GreedyAgentPolicy(agent),
                                 n_episodes=EVAL_EPISODES, seed=SEED),
    }
 
    print("\nGenerating plots ...")
    plot_reward_curve(reward_history)
    plot_policy_comparison(results)
    plot_cost_breakdown(results)
    plot_behavior_episode(agent)
 
    print("\nFour plots saved:")
    print("  reward_curve.png      — training convergence")
    print("  policy_comparison.png — mean reward ± std across policies")
    print("  cost_breakdown.png    — ordering / holding / stockout costs per policy")
    print("  behavior_episode.png  — what the agent actually does in one episode")