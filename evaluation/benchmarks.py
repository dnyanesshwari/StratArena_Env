from __future__ import annotations

from typing import Callable

from server.evaluate_tasks import run_task
from models import StratArenaObservation
from server.tasks import TASKS


PolicyFn = Callable[[StratArenaObservation], tuple[float, str]]


def benchmark_all_tasks(policy: PolicyFn | None = None) -> list[dict[str, float | str | int]]:
    rows = []
    for task in TASKS:
        result = run_task(task, policy=policy)
        rows.append(
            {
                "task": result.task,
                "score": result.score,
                "value": result.total_value,
                "spend": result.spend,
                "wins": result.wins,
                "exploit": result.exploit,
                "belief": result.belief,
                "adaptation": result.adaptation,
            }
        )
    return rows
