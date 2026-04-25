from __future__ import annotations

from models import StratArenaObservation, StratArenaState


def build_observation(env, done: bool) -> StratArenaObservation:
    task = env.task
    round_state = env.current_round
    return StratArenaObservation(
        done=done,
        reward=env.last_reward.reward,
        task=task.key,
        task_id=task.task_id,
        task_description=task.description,
        step=env.step_idx,
        max_steps=env.max_steps,
        my_budget=round(env.my_budget, 4),
        my_budget_ratio=round(env.my_budget / max(env.initial_budget, 1.0), 4),
        resource_value=round(round_state["resource_value"], 4),
        resource_scarcity=round(round_state["resource_scarcity"], 4),
        market_pressure=round(round_state["market_pressure"], 4),
        opponent_signals=env.get_opponent_signals(),
        tom_features=env.tom_tracker.get_features(),
        exploit_signal=round(env.tom_tracker.get_exploit_signal(), 4),
        uncertainty_signal=round(env.tom_tracker.get_uncertainty_signal(), 4),
        total_value_won=round(env.total_value_won, 4),
        spend_so_far=round(env.spend_so_far, 4),
        wins=env.my_wins,
        task_score=env.grade(),
        last_winner=env.last_winner,
        reward_breakdown=env.last_reward,
        metadata={
            "regime_shift_step": task.regime_shift_step,
            "belief_summary": env.tom_tracker.summary(),
        },
    )


def build_state(env) -> StratArenaState:
    return StratArenaState(
        episode_id=env.episode_id,
        step_count=env.step_idx,
        task=env.task.key,
        task_id=env.task.task_id,
        max_steps=env.max_steps,
        my_budget=round(env.my_budget, 4),
        opponent_budgets={
            "aggressive": round(env.aggressive.budget, 4),
            "conservative": round(env.conservative.budget, 4),
        },
        opponent_modes={
            "aggressive": env.aggressive.strategy_name(env.step_idx, env.max_steps, env.task.key),
            "conservative": env.conservative.strategy_name(env.step_idx, env.max_steps, env.task.key),
        },
        current_round={k: round(v, 4) for k, v in env.current_round.items()},
        total_value_won=round(env.total_value_won, 4),
        total_resource_value_seen=round(env.total_resource_value_seen, 4),
        spend_so_far=round(env.spend_so_far, 4),
        wins=env.my_wins,
        task_score=env.grade(),
        summary_metrics=env.summary_metrics(),
        belief_summary=env.tom_tracker.summary(),
        last_info=env.last_info,
    )
