from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import heuristic_allocation
from models import StratArenaAction
from server.stratarena_environment import StratArenaEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one StratArena episode step by step.")
    parser.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    env = StratArenaEnvironment()
    obs = env.reset(task=args.task, seed=args.seed)

    print(f"task={args.task} seed={args.seed}")
    print("=" * 80)
    for _ in range(args.steps):
        if obs.done:
            break
        allocation, note = heuristic_allocation(obs)
        obs = env.step(StratArenaAction(allocation=allocation))
        state = env.state

        payload = {
            "step": obs.step,
            "allocation": round(allocation, 4),
            "policy_note": note,
            "winner": obs.last_winner,
            "reward": round(float(obs.reward or 0.0), 4),
            "my_budget": round(state.my_budget, 4),
            "opponent_budgets": state.opponent_budgets,
            "round": state.current_round,
            "last_info": state.last_info.model_dump() if state.last_info else None,
            "belief_summary": state.belief_summary,
            "task_score": round(obs.task_score, 4),
        }
        print(json.dumps(payload, indent=2))
        print("-" * 80)


if __name__ == "__main__":
    main()
