TASK_CONFIG = {
    "id": "basic_competition",
    "name": "balanced_strategy_learning",
    "description": (
        "Learn to trade off value capture and budget discipline in a stable multi-agent arena. "
        "The agent should win strong resources without burning its budget too early."
    ),
    "difficulty": "easy",
    "max_steps": 50,
    "initial_budget": 500.0,
    "target_value": 850.0,
}


def grade(env_state: dict) -> float:
    value_score = min(float(env_state.get("total_value_won", 0.0)) / TASK_CONFIG["target_value"], 1.0)
    efficiency_score = min(float(env_state.get("efficiency_ratio", 0.0)) / 1.20, 1.0)
    budget_ratio = float(env_state.get("budget_remaining_ratio", 0.0))
    if 0.08 <= budget_ratio <= 0.45:
        budget_score = 1.0
    elif budget_ratio < 0.08:
        budget_score = max(0.0, budget_ratio / 0.08)
    else:
        budget_score = max(0.0, 1.0 - min((budget_ratio - 0.45) / 0.55, 1.0))

    score = (0.45 * value_score) + (0.35 * efficiency_score) + (0.20 * budget_score)
    return round(min(max(score, 0.0), 1.0), 4)
