from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import task_easy, task_hard, task_medium


GradeFn = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class TaskDefinition:
    key: str
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    initial_budget: float
    grader: GradeFn
    regime_shift_step: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "id": self.task_id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "max_steps": self.max_steps,
            "initial_budget": self.initial_budget,
            "regime_shift_step": self.regime_shift_step,
        }


def _build_definition(module: Any) -> TaskDefinition:
    config = module.TASK_CONFIG
    return TaskDefinition(
        key=config["difficulty"],
        task_id=config["id"],
        name=config["name"],
        difficulty=config["difficulty"],
        description=config["description"],
        max_steps=int(config["max_steps"]),
        initial_budget=float(config["initial_budget"]),
        grader=module.grade,
        regime_shift_step=config.get("regime_shift_step"),
    )


TASKS: dict[str, TaskDefinition] = {
    definition.key: definition
    for definition in (
        _build_definition(task_easy),
        _build_definition(task_medium),
        _build_definition(task_hard),
    )
}

TASK_ALIASES: dict[str, str] = {}
for task in TASKS.values():
    TASK_ALIASES[task.key] = task.key
    TASK_ALIASES[task.task_id] = task.key
    TASK_ALIASES[task.name] = task.key


def get_task_definition(task: str | None) -> TaskDefinition:
    normalized = TASK_ALIASES.get((task or "medium").strip().lower(), "medium")
    if normalized not in TASKS:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {supported}")
    return TASKS[normalized]


def list_task_metadata() -> list[dict[str, Any]]:
    return [task.to_metadata() for task in TASKS.values()]
