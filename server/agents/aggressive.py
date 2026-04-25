from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggressiveAgent:
    initial_budget: float = 500.0
    budget: float = 500.0
    wins: int = 0
    recent_wins: list[int] = field(default_factory=list)
    bid_history: list[float] = field(default_factory=list)
    last_bid_ratio: float = 0.0

    def reset(self, budget: float | None = None) -> None:
        self.initial_budget = float(budget or self.initial_budget)
        self.budget = self.initial_budget
        self.wins = 0
        self.recent_wins = []
        self.bid_history = []
        self.last_bid_ratio = 0.0

    def strategy_name(self, step_idx: int, max_steps: int, task_key: str) -> str:
        if task_key == "hard" and step_idx >= max_steps // 2:
            return "selective"
        return "aggressive"

    def strategy_score(self, step_idx: int, max_steps: int, task_key: str) -> float:
        return 0.35 if self.strategy_name(step_idx, max_steps, task_key) == "selective" else 1.0

    def act(self, round_state: dict[str, float], step_idx: int, max_steps: int, task_key: str) -> float:
        if self.budget <= 0.0:
            return 0.0

        resource_value = round_state["resource_value"]
        scarcity = round_state["resource_scarcity"]
        pressure = round_state["market_pressure"]
        spend_anchor = resource_value * (0.55 + (0.35 * pressure))
        phase = 1.0 - (step_idx / max(max_steps - 1, 1))
        budget_ratio = self.budget / max(self.initial_budget, 1.0)

        if task_key == "hard" and step_idx >= max_steps // 2:
            intensity = 0.78 + (0.40 * scarcity) + (0.22 * (1.0 - budget_ratio))
        else:
            intensity = 1.05 + (0.48 * phase) + (0.24 * scarcity) + (0.12 * pressure)

        raw_bid = spend_anchor * intensity
        return max(0.0, min(raw_bid, self.budget))

    def update(self, bid: float, won: bool, round_state: dict[str, float]) -> None:
        if won:
            self.budget = max(0.0, self.budget - bid)
            self.wins += 1
        base = max(round_state["resource_value"] * (0.55 + (0.35 * round_state["market_pressure"])), 1.0)
        self.last_bid_ratio = max(0.0, min(2.5, bid / base))
        self.bid_history.append(self.last_bid_ratio)
        self.recent_wins.append(int(won))
        if len(self.bid_history) > 8:
            self.bid_history.pop(0)
        if len(self.recent_wins) > 6:
            self.recent_wins.pop(0)

    def signal(self) -> dict[str, float]:
        avg_bid_ratio = sum(self.bid_history) / max(len(self.bid_history), 1)
        recent_wins = sum(self.recent_wins)
        return {
            "recent_wins": float(recent_wins),
            "last_bid_ratio": round(self.last_bid_ratio, 4),
            "avg_bid_ratio": round(avg_bid_ratio, 4),
            "win_rate": round(recent_wins / max(len(self.recent_wins), 1), 4),
        }
