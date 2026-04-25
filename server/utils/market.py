from __future__ import annotations

import math
import random


def sample_resource_round(
    rng: random.Random,
    step_idx: int,
    max_steps: int,
    task_key: str,
) -> dict[str, float]:
    phase = step_idx / max(max_steps - 1, 1)
    seasonal = 0.5 + (0.5 * math.sin((2.0 * math.pi * phase) + 0.35))

    base_value = {"easy": 42.0, "medium": 50.0, "hard": 56.0}[task_key]
    volatility = {"easy": 10.0, "medium": 16.0, "hard": 22.0}[task_key]

    scarcity = 0.18 + (0.48 * seasonal) + rng.uniform(-0.10, 0.10)
    scarcity = max(0.05, min(1.0, scarcity))

    market_pressure = 0.75 + (0.55 * scarcity) + rng.uniform(-0.12, 0.14)
    if task_key == "hard" and step_idx >= max_steps // 2:
        market_pressure += 0.20 + rng.uniform(0.0, 0.10)
        scarcity = max(0.10, min(1.0, scarcity + rng.uniform(0.05, 0.18)))

    peak_bonus = 1.0 + (0.25 if phase > 0.65 else 0.0)
    raw_value = rng.gauss(base_value * (0.82 + (0.55 * scarcity)) * peak_bonus, volatility)
    resource_value = max(8.0, raw_value)

    return {
        "resource_value": round(resource_value, 4),
        "resource_scarcity": round(scarcity, 4),
        "market_pressure": round(max(0.4, market_pressure), 4),
    }
