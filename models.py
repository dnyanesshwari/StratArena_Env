from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State


class StratArenaReward(BaseModel):
    reward: float = 0.0
    value_component: float = 0.0
    efficiency_component: float = 0.0
    strategy_component: float = 0.0
    penalty_component: float = 0.0


class StratArenaStepInfo(BaseModel):
    action_taken: float = Field(default=0.0, ge=0.0, le=2.0)
    winner: str = "none"
    resource_value: float = Field(default=0.0, ge=0.0)
    scarcity: float = Field(default=0.0, ge=0.0, le=1.0)
    market_pressure: float = Field(default=0.0, ge=0.0)
    my_bid: float = Field(default=0.0, ge=0.0)
    aggressive_bid: float = Field(default=0.0, ge=0.0)
    conservative_bid: float = Field(default=0.0, ge=0.0)
    my_budget: float = Field(default=0.0, ge=0.0)
    exploit_opportunity: bool = False
    reward_breakdown: StratArenaReward = Field(default_factory=StratArenaReward)


class StratArenaObservation(Observation):
    task: str = "medium"
    task_id: str = "opponent_exploitation"
    task_description: str = ""
    step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=50, ge=1)
    my_budget: float = Field(default=500.0, ge=0.0)
    my_budget_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    resource_value: float = Field(default=50.0, ge=0.0)
    resource_scarcity: float = Field(default=0.0, ge=0.0, le=1.0)
    market_pressure: float = Field(default=1.0, ge=0.0)
    opponent_signals: dict[str, dict[str, float]] = Field(default_factory=dict)
    tom_features: list[float] = Field(default_factory=list)
    exploit_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_signal: float = Field(default=1.0, ge=0.0, le=1.0)
    total_value_won: float = Field(default=0.0, ge=0.0)
    spend_so_far: float = Field(default=0.0, ge=0.0)
    wins: int = Field(default=0, ge=0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_winner: str = "none"
    reward_breakdown: StratArenaReward = Field(default_factory=StratArenaReward)


class StratArenaAction(Action):
    allocation: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Allocation intensity. 0=skip, 1=match estimated value, 2=aggressive.",
    )


class StratArenaState(State):
    task: str = "medium"
    task_id: str = "opponent_exploitation"
    max_steps: int = Field(default=50, ge=1)
    my_budget: float = Field(default=500.0, ge=0.0)
    opponent_budgets: dict[str, float] = Field(default_factory=dict)
    opponent_modes: dict[str, str] = Field(default_factory=dict)
    current_round: dict[str, float] = Field(default_factory=dict)
    total_value_won: float = Field(default=0.0, ge=0.0)
    total_resource_value_seen: float = Field(default=0.0, ge=0.0)
    spend_so_far: float = Field(default=0.0, ge=0.0)
    wins: int = Field(default=0, ge=0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary_metrics: dict[str, float] = Field(default_factory=dict)
    belief_summary: dict[str, Any] = Field(default_factory=dict)
    last_info: StratArenaStepInfo | None = None
