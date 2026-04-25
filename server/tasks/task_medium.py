TASK_CONFIG = {
    "id": "opponent_exploitation",
    "name": "exploit_weak_opponents",
    "description": (
        "Identify when opponents are likely depleted or predictable, then concentrate spending "
        "into those windows while avoiding high-pressure waste."
    ),
    "difficulty": "medium",
    "max_steps": 50,
    "initial_budget": 500.0,
}


def grade(env_state: dict) -> float:
    exploit_score = float(env_state.get("exploit_success_rate", 0.0))
    smart_pass_score = float(env_state.get("smart_pass_rate", 0.0))
    belief_score = float(env_state.get("belief_alignment", 0.0))
    efficiency_score = min(float(env_state.get("efficiency_ratio", 0.0)) / 1.10, 1.0)

    score = (
        (0.40 * exploit_score)
        + (0.20 * smart_pass_score)
        + (0.25 * belief_score)
        + (0.15 * efficiency_score)
    )
    return round(min(max(score, 0.0), 1.0), 4)
