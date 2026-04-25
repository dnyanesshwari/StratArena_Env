TASK_CONFIG = {
    "id": "adaptive_strategy",
    "name": "dynamic_opponent_adaptation",
    "description": (
        "Opponent behavior shifts mid-episode. The agent must update its theory of mind, "
        "stay efficient under higher pressure, and recover after the strategic regime change."
    ),
    "difficulty": "hard",
    "max_steps": 50,
    "initial_budget": 900.0,
    "regime_shift_step": 25,
}


def grade(env_state: dict) -> float:
    adaptation_score = float(env_state.get("adaptation_score", 0.0))
    belief_score = float(env_state.get("belief_alignment", 0.0))
    post_shift_efficiency = min(float(env_state.get("post_shift_efficiency", 0.0)) / 1.05, 1.0)
    exploit_score = float(env_state.get("exploit_success_rate", 0.0))

    score = (
        (0.40 * adaptation_score)
        + (0.25 * belief_score)
        + (0.20 * post_shift_efficiency)
        + (0.15 * exploit_score)
    )
    return round(min(max(score, 0.0), 1.0), 4)
