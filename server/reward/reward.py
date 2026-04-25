from __future__ import annotations

from models import StratArenaReward


def compute_reward(
    *,
    task_key: str,
    won: bool,
    my_bid: float,
    resource_value: float,
    market_pressure: float,
    exploit_opportunity: bool,
    exploit_signal: float,
    strong_opponent_pressure: bool,
    budget_ratio: float,
) -> StratArenaReward:
    reward = StratArenaReward()

    if won:
        reward.value_component = 0.12 * resource_value
        edge = resource_value - my_bid
        reward.efficiency_component = max(-3.0, min(6.0, edge * 0.08))
        if exploit_opportunity:
            reward.strategy_component += 4.5 + (2.5 * exploit_signal)
    else:
        if my_bid <= 0.05 and strong_opponent_pressure:
            reward.strategy_component += 1.5 + (1.5 * exploit_signal)
        elif my_bid <= 0.05 and resource_value < 35.0:
            reward.efficiency_component += 0.8

    overbid = max(0.0, my_bid - (resource_value * 1.10))
    reward.penalty_component -= overbid * 0.12

    if budget_ratio < 0.10:
        reward.penalty_component -= 2.5
    elif budget_ratio < 0.20:
        reward.penalty_component -= 1.0

    if task_key == "easy":
        reward.efficiency_component *= 1.20
    elif task_key == "medium":
        reward.strategy_component *= 1.25
    else:
        reward.strategy_component *= 1.10
        reward.penalty_component -= max(0.0, market_pressure - 1.45) * 0.8 if my_bid > 0.0 and not won else 0.0

    reward.reward = (
        reward.value_component
        + reward.efficiency_component
        + reward.strategy_component
        + reward.penalty_component
    )
    reward.reward = round(reward.reward, 4)
    reward.value_component = round(reward.value_component, 4)
    reward.efficiency_component = round(reward.efficiency_component, 4)
    reward.strategy_component = round(reward.strategy_component, 4)
    reward.penalty_component = round(reward.penalty_component, 4)
    return reward
