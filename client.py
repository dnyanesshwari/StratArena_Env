from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import StratArenaAction, StratArenaObservation, StratArenaState


class StratArenaEnv(EnvClient[StratArenaAction, StratArenaObservation, StratArenaState]):
    def _step_payload(self, action: StratArenaAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[StratArenaObservation]:
        observation = StratArenaObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> StratArenaState:
        return StratArenaState(**payload)
