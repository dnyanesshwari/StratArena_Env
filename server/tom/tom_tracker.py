from __future__ import annotations

from dataclasses import dataclass, field


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class OpponentBelief:
    label: str
    budget_belief: float = 1.0
    aggression_belief: float = 0.5
    confidence: float = 0.0
    volatility_belief: float = 0.5
    bid_ratio_history: list[float] = field(default_factory=list)
    win_history: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        bid_ratio: float,
        won: bool,
        resource_value: float,
        market_pressure: float,
        step_ratio: float,
        regime_shift: bool,
    ) -> None:
        self.bid_ratio_history.append(bid_ratio)
        self.win_history.append(1.0 if won else 0.0)
        if len(self.bid_ratio_history) > 10:
            self.bid_ratio_history.pop(0)
        if len(self.win_history) > 10:
            self.win_history.pop(0)

        avg_bid = sum(self.bid_ratio_history) / len(self.bid_ratio_history)
        recent_win_rate = sum(self.win_history) / len(self.win_history)
        early_slice = self.bid_ratio_history[: max(1, len(self.bid_ratio_history) // 2)]
        late_slice = self.bid_ratio_history[max(1, len(self.bid_ratio_history) // 2) :]
        trend = (sum(late_slice) / max(len(late_slice), 1)) - (sum(early_slice) / max(len(early_slice), 1))

        aggression_signal = (0.60 * _clip(avg_bid / 1.5)) + (0.25 * recent_win_rate) + (0.15 * (1.0 - step_ratio))
        self.aggression_belief = _clip((0.7 * self.aggression_belief) + (0.3 * aggression_signal))

        spend_signal = (0.55 * _clip(avg_bid / 1.4)) + (0.25 * recent_win_rate) + (0.20 * _clip(market_pressure / 1.5))
        if regime_shift:
            self.volatility_belief = _clip((0.6 * self.volatility_belief) + 0.4)
            self.confidence *= 0.75
        else:
            self.volatility_belief = _clip((0.75 * self.volatility_belief) + (0.25 * abs(trend)))

        self.budget_belief = _clip(
            self.budget_belief - (0.05 * spend_signal) - (0.05 if won else 0.0) + (0.02 if bid_ratio < 0.5 else 0.0)
        )
        self.confidence = _clip((len(self.bid_ratio_history) / 10.0) * (1.0 - (0.35 * self.volatility_belief)))

    def to_features(self) -> list[float]:
        return [
            round(self.budget_belief, 4),
            round(self.aggression_belief, 4),
            round(self.confidence, 4),
            round(self.volatility_belief, 4),
        ]


class ToMTracker:
    def __init__(self, labels: tuple[str, ...] = ("aggressive", "conservative")):
        self.labels = labels
        self.beliefs: dict[str, OpponentBelief] = {}
        self.reset()

    def reset(self) -> None:
        self.beliefs = {label: OpponentBelief(label=label) for label in self.labels}

    def update(
        self,
        *,
        label: str,
        bid_ratio: float,
        won: bool,
        resource_value: float,
        market_pressure: float,
        step_ratio: float,
        regime_shift: bool = False,
    ) -> None:
        self.beliefs[label].update(
            bid_ratio=bid_ratio,
            won=won,
            resource_value=resource_value,
            market_pressure=market_pressure,
            step_ratio=step_ratio,
            regime_shift=regime_shift,
        )

    def get_features(self) -> list[float]:
        features: list[float] = []
        for label in self.labels:
            features.extend(self.beliefs[label].to_features())
        features.append(round(self.get_exploit_signal(), 4))
        features.append(round(self.get_uncertainty_signal(), 4))
        return features

    def get_exploit_signal(self) -> float:
        signals = []
        for belief in self.beliefs.values():
            depletion = 1.0 - belief.budget_belief
            predictability = belief.confidence * (1.0 - (0.5 * belief.volatility_belief))
            signals.append(depletion * predictability)
        return _clip(sum(signals) / max(len(signals), 1))

    def get_uncertainty_signal(self) -> float:
        if not self.beliefs:
            return 1.0
        confidence = sum(b.confidence for b in self.beliefs.values()) / len(self.beliefs)
        volatility = sum(b.volatility_belief for b in self.beliefs.values()) / len(self.beliefs)
        return _clip((1.0 - confidence) * 0.7 + (volatility * 0.3))

    def summary(self) -> dict[str, dict[str, float | str]]:
        result: dict[str, dict[str, float | str]] = {}
        for label, belief in self.beliefs.items():
            result[label] = {
                "budget_belief": round(belief.budget_belief, 4),
                "aggression_belief": round(belief.aggression_belief, 4),
                "confidence": round(belief.confidence, 4),
                "volatility": round(belief.volatility_belief, 4),
                "inferred_style": "aggressive" if belief.aggression_belief >= 0.5 else "conservative",
            }
        return result
