from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import StratArenaAction, StratArenaReward, StratArenaStepInfo
from server.agents import AggressiveAgent, ConservativeAgent
from server.observation.builder import build_observation, build_state
from server.reward import compute_reward
from server.tasks import TaskDefinition, get_task_definition, list_task_metadata
from server.tom import ToMTracker
from server.utils.market import sample_resource_round


class StratArenaEnvironment(Environment[StratArenaAction, object, object]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self.aggressive = AggressiveAgent()
        self.conservative = ConservativeAgent()
        self.tom_tracker = ToMTracker()
        self.task: TaskDefinition = get_task_definition("medium")
        self.max_steps = self.task.max_steps
        self.initial_budget = self.task.initial_budget
        self.episode_id = ""
        self.step_idx = 0
        self.my_budget = self.initial_budget
        self.total_value_won = 0.0
        self.total_resource_value_seen = 0.0
        self.spend_so_far = 0.0
        self.my_wins = 0
        self.last_winner = "none"
        self.last_reward = StratArenaReward()
        self.last_info: StratArenaStepInfo | None = None
        self.current_round = {"resource_value": 50.0, "resource_scarcity": 0.5, "market_pressure": 1.0}
        self.exploit_opportunities = 0
        self.exploit_wins = 0
        self.smart_pass_opportunities = 0
        self.smart_passes = 0
        self.regime_adapted_actions = 0
        self.regime_rounds = 0
        self.post_shift_value = 0.0
        self.post_shift_spend = 0.0
        self.belief_alignment_total = 0.0
        self.belief_alignment_count = 0
        self.reset()

    def reset(self, seed: int | None = None, episode_id: str | None = None, task: str | None = None, **kwargs):
        self.task = get_task_definition(task)
        self.max_steps = self.task.max_steps
        self.initial_budget = self.task.initial_budget
        if seed is None:
            seed = kwargs.get("task_seed")
        self._rng = random.Random(seed if seed is not None else 17)

        self.episode_id = episode_id or str(uuid4())
        self.step_idx = 0
        self.my_budget = self.initial_budget
        self.total_value_won = 0.0
        self.total_resource_value_seen = 0.0
        self.spend_so_far = 0.0
        self.my_wins = 0
        self.last_winner = "none"
        self.last_reward = StratArenaReward()
        self.last_info = None
        self.exploit_opportunities = 0
        self.exploit_wins = 0
        self.smart_pass_opportunities = 0
        self.smart_passes = 0
        self.regime_adapted_actions = 0
        self.regime_rounds = 0
        self.post_shift_value = 0.0
        self.post_shift_spend = 0.0
        self.belief_alignment_total = 0.0
        self.belief_alignment_count = 0

        self.aggressive.reset(self.initial_budget)
        self.conservative.reset(self.initial_budget)
        self.tom_tracker.reset()
        self.current_round = sample_resource_round(self._rng, self.step_idx, self.max_steps, self.task.key)
        return build_observation(self, done=False)

    def step(self, action: StratArenaAction, timeout_s: float | None = None, **kwargs):
        if self.step_idx >= self.max_steps or self.my_budget <= 0.0:
            return build_observation(self, done=True)

        resource_value = self.current_round["resource_value"]
        scarcity = self.current_round["resource_scarcity"]
        pressure = self.current_round["market_pressure"]
        self.total_resource_value_seen += resource_value

        market_anchor = max(resource_value * (0.55 + (0.35 * pressure)), 1.0)
        my_bid = min(float(action.allocation) * market_anchor, self.my_budget)
        aggressive_bid = self.aggressive.act(self.current_round, self.step_idx, self.max_steps, self.task.key)
        conservative_bid = self.conservative.act(self.current_round, self.step_idx, self.max_steps, self.task.key)

        bids = {
            "me": round(my_bid, 4),
            "aggressive": round(aggressive_bid, 4),
            "conservative": round(conservative_bid, 4),
        }
        highest_bid = max(bids.values())
        winners = [label for label, bid in bids.items() if abs(bid - highest_bid) < 1e-9]
        winner = self._rng.choice(winners) if len(winners) > 1 else winners[0]
        self.last_winner = winner

        exploit_opportunity = self._is_exploit_opportunity(resource_value)
        strong_opponent_pressure = max(aggressive_bid, conservative_bid) > (resource_value * pressure * 1.15)

        if exploit_opportunity:
            self.exploit_opportunities += 1
        if my_bid <= 0.05 and strong_opponent_pressure:
            self.smart_pass_opportunities += 1

        if winner == "me":
            self.my_budget = max(0.0, self.my_budget - my_bid)
            self.spend_so_far += my_bid
            self.total_value_won += resource_value
            self.my_wins += 1
            if exploit_opportunity:
                self.exploit_wins += 1
        elif my_bid <= 0.05 and strong_opponent_pressure:
            self.smart_passes += 1

        self.aggressive.update(aggressive_bid, winner == "aggressive", self.current_round)
        self.conservative.update(conservative_bid, winner == "conservative", self.current_round)

        step_ratio = (self.step_idx + 1) / max(self.max_steps, 1)
        regime_shift = self.task.regime_shift_step is not None and self.step_idx >= self.task.regime_shift_step
        self.tom_tracker.update(
            label="aggressive",
            bid_ratio=aggressive_bid / market_anchor,
            won=winner == "aggressive",
            resource_value=resource_value,
            market_pressure=pressure,
            step_ratio=step_ratio,
            regime_shift=regime_shift,
        )
        self.tom_tracker.update(
            label="conservative",
            bid_ratio=conservative_bid / market_anchor,
            won=winner == "conservative",
            resource_value=resource_value,
            market_pressure=pressure,
            step_ratio=step_ratio,
            regime_shift=regime_shift,
        )
        self._update_belief_alignment()

        if regime_shift:
            self.regime_rounds += 1
            if self._is_adaptive_action(my_bid, resource_value, pressure):
                self.regime_adapted_actions += 1
            if winner == "me":
                self.post_shift_value += resource_value
                self.post_shift_spend += my_bid

        self.last_reward = compute_reward(
            task_key=self.task.key,
            won=winner == "me",
            my_bid=my_bid,
            resource_value=resource_value,
            market_pressure=pressure,
            exploit_opportunity=exploit_opportunity,
            exploit_signal=self.tom_tracker.get_exploit_signal(),
            strong_opponent_pressure=strong_opponent_pressure,
            budget_ratio=self.my_budget / max(self.initial_budget, 1.0),
        )
        self.last_info = StratArenaStepInfo(
            action_taken=round(float(action.allocation), 4),
            winner=winner,
            resource_value=round(resource_value, 4),
            scarcity=round(scarcity, 4),
            market_pressure=round(pressure, 4),
            my_bid=round(my_bid, 4),
            aggressive_bid=round(aggressive_bid, 4),
            conservative_bid=round(conservative_bid, 4),
            my_budget=round(self.my_budget, 4),
            exploit_opportunity=exploit_opportunity,
            reward_breakdown=self.last_reward,
        )

        self.step_idx += 1
        done = self.step_idx >= self.max_steps or self.my_budget <= 0.0
        if not done:
            self.current_round = sample_resource_round(self._rng, self.step_idx, self.max_steps, self.task.key)
        return build_observation(self, done=done)

    @property
    def state(self):
        return build_state(self)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="StratArena",
            description="Multi-agent OpenEnv benchmark for theory-of-mind strategic allocation under uncertainty.",
            version="1.0.0",
            author="stratarena-team",
            readme_content=None,
        )

    def grade(self) -> float:
        return self.task.grader(self.summary_metrics())

    def summary_metrics(self) -> dict[str, float]:
        efficiency_ratio = self.total_value_won / max(self.spend_so_far, 1.0)
        exploit_success_rate = self.exploit_wins / max(self.exploit_opportunities, 1) if self.exploit_opportunities else 0.0
        smart_pass_rate = self.smart_passes / max(self.smart_pass_opportunities, 1) if self.smart_pass_opportunities else 0.0
        belief_alignment = self.belief_alignment_total / max(self.belief_alignment_count, 1) if self.belief_alignment_count else 0.0
        adaptation_score = self.regime_adapted_actions / max(self.regime_rounds, 1) if self.regime_rounds else 0.0
        post_shift_efficiency = self.post_shift_value / max(self.post_shift_spend, 1.0) if self.post_shift_spend > 0.0 else 0.0
        return {
            "total_value_won": round(self.total_value_won, 4),
            "total_resource_value_seen": round(self.total_resource_value_seen, 4),
            "spend_so_far": round(self.spend_so_far, 4),
            "budget_remaining_ratio": round(self.my_budget / max(self.initial_budget, 1.0), 4),
            "efficiency_ratio": round(efficiency_ratio, 4),
            "exploit_success_rate": round(exploit_success_rate, 4),
            "smart_pass_rate": round(smart_pass_rate, 4),
            "belief_alignment": round(belief_alignment, 4),
            "adaptation_score": round(adaptation_score, 4),
            "post_shift_efficiency": round(post_shift_efficiency, 4),
        }

    def get_opponent_signals(self) -> dict[str, dict[str, float]]:
        return {
            "aggressive": self.aggressive.signal(),
            "conservative": self.conservative.signal(),
        }

    def task_catalog(self) -> list[dict[str, object]]:
        return list_task_metadata()

    def _is_exploit_opportunity(self, resource_value: float) -> bool:
        aggressive_weak = self.aggressive.budget / max(self.initial_budget, 1.0) < 0.40
        moderate_value = resource_value >= 28.0
        conservative_not_spiking = self.conservative.last_bid_ratio < 0.85
        return aggressive_weak and moderate_value and conservative_not_spiking

    def _is_adaptive_action(self, my_bid: float, resource_value: float, pressure: float) -> bool:
        if resource_value >= 70.0 and my_bid > 0.0:
            return True
        if pressure > 1.45 and my_bid <= 0.05:
            return True
        return False

    def _update_belief_alignment(self) -> None:
        aggressive_belief = self.tom_tracker.beliefs["aggressive"]
        conservative_belief = self.tom_tracker.beliefs["conservative"]

        actual_pairs = [
            (
                aggressive_belief,
                self.aggressive.budget / max(self.initial_budget, 1.0),
                self.aggressive.strategy_score(self.step_idx, self.max_steps, self.task.key),
            ),
            (
                conservative_belief,
                self.conservative.budget / max(self.initial_budget, 1.0),
                self.conservative.strategy_score(self.step_idx, self.max_steps, self.task.key),
            ),
        ]
        for belief, actual_budget, actual_style in actual_pairs:
            budget_alignment = 1.0 - abs(belief.budget_belief - actual_budget)
            style_alignment = 1.0 - abs(belief.aggression_belief - actual_style)
            self.belief_alignment_total += max(0.0, min((budget_alignment + style_alignment) / 2.0, 1.0))
            self.belief_alignment_count += 1

__all__ = ["StratArenaEnvironment"]