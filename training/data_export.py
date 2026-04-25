from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import build_prompt, heuristic_allocation
from models import StratArenaAction, StratArenaObservation
from server.stratarena_environment import StratArenaEnvironment


PolicyFn = Callable[[StratArenaObservation], tuple[float, str]]


def random_policy_factory(seed: int) -> PolicyFn:
    rng = random.Random(seed)

    def _policy(obs: StratArenaObservation) -> tuple[float, str]:
        return rng.uniform(0.0, 2.0), "random"

    return _policy


def mixed_policy_factory(seed: int) -> PolicyFn:
    rng = random.Random(seed)
    random_policy = random_policy_factory(seed + 1000)

    def _policy(obs: StratArenaObservation) -> tuple[float, str]:
        if rng.random() < 0.7:
            return heuristic_allocation(obs)
        return random_policy(obs)

    return _policy


def select_policy(name: str, seed: int) -> PolicyFn:
    if name == "heuristic":
        return heuristic_allocation
    if name == "random":
        return random_policy_factory(seed)
    if name == "mixed":
        return mixed_policy_factory(seed)
    raise ValueError(f"Unsupported policy '{name}'")


def export_dataset(
    output_path: Path,
    episodes_per_task: int,
    policy_name: str,
    base_seed: int,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    sft_rows_written = 0

    with output_path.open("w", encoding="utf-8") as fp:
        for task_idx, task in enumerate(("easy", "medium", "hard")):
            for episode_idx in range(episodes_per_task):
                episode_seed = base_seed + (task_idx * 10_000) + episode_idx
                policy = select_policy(policy_name, episode_seed)
                env = StratArenaEnvironment()
                obs = env.reset(task=task, seed=episode_seed)
                trace = []

                while not obs.done:
                    prompt = build_prompt(task, obs, trace)
                    allocation, note = policy(obs)
                    next_obs = env.step(StratArenaAction(allocation=allocation))
                    completion = json.dumps(
                        {
                            "allocation": round(allocation, 4),
                            "reason": note,
                        },
                        separators=(",", ":"),
                    )
                    row = {
                        "task": task,
                        "episode_seed": episode_seed,
                        "step": obs.step,
                        "prompt": prompt,
                        "completion": completion,
                        "reward": float(next_obs.reward or 0.0),
                        "done": bool(next_obs.done),
                        "task_score": float(next_obs.task_score),
                        "winner": next_obs.last_winner,
                        "my_budget_ratio": float(next_obs.my_budget_ratio),
                        "exploit_signal": float(next_obs.exploit_signal),
                        "uncertainty_signal": float(next_obs.uncertainty_signal),
                        "metadata": {
                            "policy": policy_name,
                            "resource_value": float(obs.resource_value),
                            "market_pressure": float(obs.market_pressure),
                            "tom_features": list(obs.tom_features),
                        },
                    }
                    fp.write(json.dumps(row) + "\n")
                    rows_written += 1
                    if policy_name != "random":
                        sft_rows_written += 1

                    trace.append(
                        {
                            "step": next_obs.step,
                            "allocation": allocation,
                            "reward": float(next_obs.reward or 0.0),
                            "exploit_signal": float(next_obs.exploit_signal),
                            "task_score": float(next_obs.task_score),
                            "winner": next_obs.last_winner,
                        }
                    )
                    obs = next_obs

    return {
        "rows_written": rows_written,
        "sft_candidate_rows": sft_rows_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StratArena trajectories for SFT / GRPO")
    parser.add_argument(
        "--output",
        default="outputs/stratarena_rollouts.jsonl",
        help="JSONL file to write prompts, actions, and rewards into.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=40,
        help="Number of rollouts to export for each task.",
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "mixed", "random"],
        default="mixed",
        help="Policy used to generate actions for the dataset.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    args = parser.parse_args()

    summary = export_dataset(
        output_path=Path(args.output),
        episodes_per_task=args.episodes_per_task,
        policy_name=args.policy,
        base_seed=args.seed,
    )
    print(
        "Export complete: "
        f"rows={summary['rows_written']} "
        f"sft_candidates={summary['sft_candidate_rows']} "
        f"path={args.output}"
    )


if __name__ == "__main__":
    main()
