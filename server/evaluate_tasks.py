from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from inference import heuristic_allocation
from models import StratArenaAction, StratArenaObservation
from server.stratarena_environment import StratArenaEnvironment
from server.tasks import TASKS


PolicyFn = Callable[[StratArenaObservation], tuple[float, str]]


@dataclass(frozen=True)
class PolicyResult:
    task: str
    score: float
    total_value: float
    spend: float
    wins: int
    exploit: float
    belief: float
    adaptation: float


def run_task(task: str, policy: PolicyFn | None = None, seed: int | None = None) -> PolicyResult:
    env = StratArenaEnvironment()
    obs = env.reset(task=task, seed=seed)
    policy_fn = policy or heuristic_allocation
    while not obs.done:
        allocation, _ = policy_fn(obs)
        obs = env.step(StratArenaAction(allocation=allocation))

    summary = env.summary_metrics()
    return PolicyResult(
        task=task,
        score=env.grade(),
        total_value=obs.total_value_won,
        spend=obs.spend_so_far,
        wins=obs.wins,
        exploit=summary["exploit_success_rate"],
        belief=summary["belief_alignment"],
        adaptation=summary["adaptation_score"],
    )


def main() -> None:
    print("StratArena heuristic evaluation")
    print("-" * 92)
    print(
        f"{'task':>6} | {'score':>6} | {'value':>8} | {'spend':>8} | "
        f"{'wins':>4} | {'exploit':>7} | {'belief':>6} | {'adapt':>5}"
    )
    print("-" * 92)

    total = 0.0
    for task_key in TASKS:
        result = run_task(task_key)
        total += result.score
        print(
            f"{result.task:>6} | {result.score:>6.4f} | {result.total_value:>8.2f} | "
            f"{result.spend:>8.2f} | {result.wins:>4} | {result.exploit:>7.3f} | "
            f"{result.belief:>6.3f} | {result.adaptation:>5.3f}"
        )

    print("-" * 92)
    print(f"{'average':>6} | {total / max(len(TASKS), 1):>6.4f}")


if __name__ == "__main__":
    main()
