# Tabular Q-Learning Agent for Retail Inventory Replenishment

A reinforcement learning agent trained to optimise daily inventory replenishment decisions for a single retail SKU over a 30-day simulated episode. The objective is to determine daily inventory ordering decisions that maximize long-term profit while balancing sales revenue, inventory holding costs, stockout penalties, and ordering costs. under stochastic demand and a 2-day supplier lead time.

The agent was evaluated against two baselines:
1. Random Policy (lower bound)
2. Reorder-Point Policy (traditional inventory heuristic)

---

## Problem Framing

### State

Each state contains operational information available to the inventory manager:

| Variable | Description |
|---|---|
| **Inventory Level** | Current inventory on hand |
| **Day of Week** | Monday–Sunday indicator |
| **Rolling Average Demand** | 7-day moving average demand |
| **Pending Order Quantity** | Inventory currently in transit |
| **Lead Time Remaining** | Days until order arrival |
| **Inventory Position** | On-hand + in-transit inventory |
| **Days Remaining** | Remaining days in episode |

### Action

The agent chooses one inventory order quantity per day:

Order quantity âˆˆ {0, 20, 40, 50, 60, 70, 80} units

### Reward

The reward function represents daily business profit:

Reward = Sales Revenue âˆ’ Holding Cost âˆ’ Stockout Penalty âˆ’ Ordering Cost

Where:
- Revenue = units sold × selling price
- Holding Cost = inventory remaining at end of day
- Stockout Penalty = unmet customer demand
- Ordering Cost = fixed cost per replenishment order

### Transition

The environment includes:
- Poisson customer demand (Î»=20 weekdays, Î»=25 weekends)
- Two-day supplier lead time

### Horizon

30 simulated days per episode

### Cost parameters

| Parameter | Value |
|---|---|
| Sale price | $10 / unit |
| Holding cost | $0.50 / unit / day |
| Stockout penalty | $5 / unit of unmet demand |
| Ordering cost | $20 flat fee per order |

---

## Project Structure

```
.
â”œâ”€â”€ env.py                  # Custom Gymnasium inventory environment
â”œâ”€â”€ baseline.py             # Random policy and (s, S) reorder-point policy
â”œâ”€â”€ q_agent.py              # Tabular Q-learning agent implementation and training
â”œâ”€â”€ plot.py                 # Visualization generation
â”œâ”€â”€ failure_analysis.py     # Safety and robustness experiments (reward hacking, unsafe behaviour, instability, overfitting)
â”œâ”€â”€ q_table.pkl             # Pre-trained Q-table (skip retraining)
â”œâ”€â”€
â”œâ”€â”€ plots                   # behavior_episode.png, cost_breakdown.png, policy_comparison.png, reward_curve.png, instability.png
â”œâ”€â”€ business_memo.docx      # Deployment recommendation memo
â””â”€â”€ README.md
```

---

## Agent Design

**State discretization:** The full 7-dimensional observation is projected into a 5-dimensional discrete Q-state:

| Q-state dimension | Source | Bins |
|---|---|---|
| Inventory level | `obs[0]` | [0, 30, 60, 90, 130, 200] |
| Day type | `obs[1]` | weekday / weekend (2 buckets) |
| Pending order qty | `obs[3]` | [0, 1, 40, 100] |
| Lead time left | `obs[4]` | [0, 1, 3] |
| Days remaining | `obs[6]` | [0, 2, 7, 15, 31] |

Rolling average demand (`obs[2]`) and inventory position (`obs[5]`) are present in the environment observation but not used in the Q-state.

Q-table size: **5 Ã— 2 Ã— 3 Ã— 2 Ã— 4 Ã— 7 = 1,680 entries**

**Hyperparameter tuning:** Before full training, `q_agent.py` runs a validation sweep across five candidate configurations (varying Î±, Î³, Îµ-decay, and bin granularity) on a held-out seed to select the best configuration automatically.

**Best configuration (selected by sweep):**

| Parameter | Value |
|---|---|
| Learning rate Î± | 0.1 |
| Discount factor Î³ | 0.98 |
| Îµ start / end | 1.0 â†’ 0.05 |
| Îµ decay | 0.997 per episode |
| Training episodes | 3,000 |

**Action masking:** When a pending order is already in transit, the agent is restricted to action 0 (order nothing), preventing illegal double-ordering and reducing wasted exploration.

---

## Setup

```bash
pip install gymnasium numpy matplotlib
```

Python 3.10+ required.

---

## How to Run

**1. Sanity-check the environment**
```bash
python env.py
```
Runs one 30-day episode with a random policy and prints daily state. Confirms the environment is working correctly.

**2. Evaluate baseline policies**
```bash
python baseline.py
```
Evaluates the random policy (lower bound) and the (s=60, S=120) reorder-point policy over 200 episodes. Also runs a grid search to find the best-tuned reorder-point thresholds.

**3. Train the Q-learning agent**
```bash
python q_agent.py
```
Trains the agent for 3,000 episodes (~2â€“3 minutes on a standard laptop), saves the Q-table to `q_table.pkl`, plots the training reward curve, and prints a three-policy comparison.

**4. Generate all plots**
```bash
python plot.py
```
Loads `q_table.pkl` and generates four plots (see Plots section below).

**5. Run failure analysis**
```bash
python failure_analysis.py
```
Loads `q_table.pkl` and runs four failure mode tests: reward hacking, demand surge, training instability, and distributional overfitting.

---

## Key Results

Evaluated over 200 episodes with seed=42:

| Policy | Mean Reward | Std | Stockouts / ep |
|---|---|---|---|
| Random (lower bound) | $4,675 | $499 | 18.5 |
| Reorder-Point s=60 S=120 | $5,512 | $222 | 15.5 |
| Reorder-Point s=70 S=130 (best tuned) | $5,573 | $262 | 3.1 |
| **Q-Learning Agent** | **$5,521** | **$257** | **9.4** |

The Q-learning agent matches the standard reorder-point policy on profit and reduces stockouts by 39% relative to that baseline. The grid-searched reorder-point policy (s=70, S=130) edges ahead on raw profit but carries higher holding costs ($574 vs $521). The Q-learning agent targets leaner inventory profile that accepts more stockout risk in exchange for lower carrying costs. Whether this trade-off is better off will depend on how much weight the business places on storage costs versus service level consistency.

---

## Plots

| File | Description |
|---|---|
| `reward_curve.png` | Training reward per episode + 100-episode moving average. Shows convergence to ~$5,000 by episode 500. |
| `policy_comparison.png` | Mean reward Â± 1 std for all policies across 200 evaluation episodes. |
| `cost_breakdown.png` | Stacked cost components (ordering, holding, stockout) per policy with net reward overlay. |
| `behavior_episode.png` | Single greedy episode: inventory level, daily demand, and orders placed over 30 days. |
| `instability.png` | Training reward comparison: Î±=0.1 (stable) vs Î±=0.9 (unstable). |

---

## Failure Analysis Summary

| Failure Mode | Finding | Mitigation |
|---|---|---|
| Reward hacking | Under a $1 stockout penalty, agent chose "order 0" (order nothing) in 66.7% of steps | Validate stockout penalty against true lost-sale cost; alert on sustained near-zero stock |
| Demand surge | Stockouts rose from 10.7 to 287.7/ep under 1.5Ã— demand | Fallback to reorder-point policy when rolling avg demand > 1.3Ã— training mean; quarterly retraining on recent demand data |
| Training instability | Î±=0.9 produced reward std of 1,002 vs 283 for Î±=0.1 | Lock hyperparameters; require hold-out validation before retraining |
| Distributional shift | Agent earned $4,979 vs reorder-point $4,886 on seasonal demand | Robust â€” agent still outperforms. Domain randomisation recommended for future work |

---

## Deployment Recommendation

**Shadow deployment recommended.** The agent demonstrates strong performance and consistently outperforms the random baseline while matching the performance of a traditional reorder-point policy. However, failure analysis revealed several vulnerabilities. Therefore, the agent is not ready for autonomous operation but is ready for a 30-day shadow period where its recommendations are logged and compared against current operations without being executed. See `business_memo.docx` for the full deployment argument and governance recommendations.


