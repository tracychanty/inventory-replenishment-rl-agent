"""
State:  (
    inventory_level,
    day_of_week,
    rolling_avg_demand,
    pending_order_quantity,
    lead_time_left,
    inventory_position,
    days_remaining,
)
Action: units to order, chosen from ORDER_OPTIONS
Reward: Sales Revenue - Holding Cost - Stockout Penalty - Ordering Cost
Horizon: 30 simulated days per episode
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------
# Environment constants
# ---------------------------------------------------------
MAX_INVENTORY = 200                           # warehouse capacity (units)
MAX_PENDING = 100                             # max units that can be in a pending order
LEAD_TIME_DAYS = 2                            # days until an order arrives
ORDER_OPTIONS = [0, 20, 40, 50, 60, 70, 80]   # discrete action space

DEMAND_LAMBDA = 20                    # Poisson mean daily demand (units)
SALE_PRICE = 10.0                     # revenue per unit sold ($)
HOLDING_COST = 0.50                   # cost per unit held overnight ($)
STOCKOUT_PENALTY = 5.0                # penalty per unit of unmet demand ($)
ORDERING_COST = 20.0                  # flat cost per order placed ($)

EPISODE_DAYS = 30                     # horizon H
ROLLING_WINDOW = 7                    # days used for rolling average demand


class InventoryEnv(gym.Env):
    """
    A discrete-time retail inventory MDP.

    Observation (all integers, clipped to their ranges):
        inventory_level     : units currently on hand             [0, MAX_INVENTORY]
        day_of_week         : 0 = Monday … 6 = Sunday             [0, 6]
        rolling_avg_demand  : 7-day rolling mean demand * 10      [0, MAX_INVENTORY * 10]
        pending_order_qty   : units currently in transit          [0, MAX_PENDING]
        lead_time_left      : days until pending order arrives    [0, LEAD_TIME_DAYS]
        inventory_position  : on-hand + in-transit units          [0, MAX_INVENTORY + MAX_PENDING]
        days_remaining      : decision steps left in episode      [0, EPISODE_DAYS]

    Action:
        Integer index into ORDER_OPTIONS, i.e. 0→order 0, 1→order 20, …

    Reward:
        profit = revenue - holding_cost - stockout_penalty - ordering_cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = None):
        super().__init__()

        # --- action space ---
        self.action_space = spaces.Discrete(len(ORDER_OPTIONS))

        # --- observation space ---
        # rolling_avg_demand stored as int (actual_mean * 10) to avoid floats
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            high = np.array(
                [
                    MAX_INVENTORY,
                    6,
                    MAX_INVENTORY * 10,
                    MAX_PENDING,
                    LEAD_TIME_DAYS,
                    MAX_INVENTORY + MAX_PENDING,
                    EPISODE_DAYS,
                ],
                dtype=np.int32,
            ),
            dtype = np.int32,
        )

        self.np_rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

        # Initialize the starting business conditions for the episode
        self.inventory = 50                                      # start with 50 units on hand
        self.day = 0                                             # episode step counter
        self.day_of_week = 0                                     # Monday
        self.pending_order = 0                                   # no order in transit
        self.lead_time_left = 0                                  # days until pending order lands
        self.demand_history = [DEMAND_LAMBDA] * ROLLING_WINDOW   # warm-start history

        info = {}
        return self._get_obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Execute one decision step: convert the selected action into an order quantity
        units_to_order = ORDER_OPTIONS[action]

        # Simulate one business day and calculate the reward from that decision
        reward, info = self._simulate_day(units_to_order)

        self.day += 1
        self.day_of_week = (self.day_of_week + 1) % 7

        terminated = (self.day >= EPISODE_DAYS)
        truncated = False

        # Terminal: penalise leftover inventory (holding cost for unsold stock)
        if terminated:
            terminal_penalty = self.inventory * HOLDING_COST
            reward -= terminal_penalty
            info["terminal_holding_penalty"] = terminal_penalty

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        print(
            f"Day {self.day:02d} | "
            f"Inventory: {self.inventory:3d} | "
            f"Day-of-week: {self.day_of_week} | "
            f"Pending: {self.pending_order} (arrives in {self.lead_time_left}d) | "
            f"Inv position: {self.inventory + self.pending_order:3d} | "
            f"Days left: {EPISODE_DAYS - self.day:2d} | "
            f"Rolling avg demand: {self._rolling_avg():.1f}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_day(self, units_to_order: int) -> tuple[float, dict]:
        """Run one day of the simulation and return (reward, info dict)."""
        ordering_cost = 0.0

        # 1. Receive pending order if lead time elapsed (before placing new order)
        if self.lead_time_left > 0:
            self.lead_time_left -= 1
            if self.lead_time_left == 0:
                # Add the arriving inventory to stock without exceeding warehouse capacity
                self.inventory = min(self.inventory + self.pending_order,
                                     MAX_INVENTORY)
                self.pending_order = 0

        # 2. Allow a new order only when there is no outstanding order in transit
        if units_to_order > 0 and self.pending_order == 0:
            self.pending_order = units_to_order
            self.lead_time_left = LEAD_TIME_DAYS
            ordering_cost = ORDERING_COST

        # 3. Random demand (Poisson) — higher demand on weekends
        lam = 25 if self.day_of_week in [5, 6] else 20  # Sat/Sun = 25, weekdays = 20
        demand = int(self.np_rng.poisson(lam=lam))
        self.demand_history.append(demand)
        if len(self.demand_history) > ROLLING_WINDOW:
            self.demand_history.pop(0)

        # 4. Sell as many units as available and record any unmet demand
        units_sold = min(demand, self.inventory)
        units_lost = demand - units_sold           # unmet demand
        self.inventory -= units_sold               # update inventory after fulfilling customer demand
        self.inventory = max(self.inventory, 0)    # safety clip

        # 5. Compute reward components
        revenue = units_sold * SALE_PRICE
        holding_cost = self.inventory * HOLDING_COST
        stockout_penalty = units_lost * STOCKOUT_PENALTY
        profit = revenue - holding_cost - stockout_penalty - ordering_cost

        info = {
            "demand": demand,
            "units_sold": units_sold,
            "units_lost": units_lost,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "ordering_cost": ordering_cost,
            "profit": profit,
        }
        return profit, info

    def _rolling_avg(self) -> float:
        return float(np.mean(self.demand_history))

    def _get_obs(self) -> np.ndarray:
        inventory_position = self.inventory + self.pending_order
        days_remaining = EPISODE_DAYS - self.day
        return np.array([
            int(np.clip(self.inventory, 0, MAX_INVENTORY)),
            int(self.day_of_week),
            int(np.clip(round(self._rolling_avg() * 10), 0, MAX_INVENTORY * 10)),
            int(np.clip(self.pending_order, 0, MAX_PENDING)),
            int(np.clip(self.lead_time_left, 0, LEAD_TIME_DAYS)),
            int(np.clip(inventory_position, 0, MAX_INVENTORY + MAX_PENDING)),
            int(np.clip(days_remaining, 0, EPISODE_DAYS)),
        ], dtype=np.int32)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

# Run a simple simulation using a random policy to verify the environment works correctly
if __name__ == "__main__":
    env = InventoryEnv(seed=42)
    obs, _ = env.reset()
    print("Starting obs:", obs)

    total_reward = 0.0
    for step in range(EPISODE_DAYS):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        if terminated:
            break

    print(f"\nEpisode total reward (random policy): ${total_reward:.2f}")
