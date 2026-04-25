# StratArena

StratArena is an OpenEnv-compatible multi-agent reinforcement learning environment for theory-of-mind training under uncertainty. A learner competes against aggressive and conservative opponents for limited resources, sees only behavioral signals, and must infer hidden opponent state
such as budget depletion and strategic posture.

## Core idea

- multi-agent strategic allocation rather than a single-agent toy benchmark
- partial observability through opponent signals instead of direct hidden-state access
- theory-of-mind features generated from an online belief tracker
- task-specific grading for balanced play, opponent exploitation, and dynamic adaptation

## Tasks

- `easy` / `basic_competition`: balance value capture with budget discipline
- `medium` / `opponent_exploitation`: exploit depleted or predictable opponents
- `hard` / `adaptive_strategy`: adapt after the opponent regime shifts mid-episode

## OpenEnv contract

- `reset(task=..., seed=...) -> StratArenaObservation`
- `step(StratArenaAction) -> StratArenaObservation`
- `state -> StratArenaState`

Validate the environment:

```bash
uv run openenv validate .
```

## Run it

Install dependencies:

```bash
uv sync
```

Run the local server:

```bash
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run the heuristic benchmark:

```bash
uv run python inference.py --policy heuristic --task all
uv run python evaluate_tasks.py
```

Run a trained RL model inside the live environment through the same inference entrypoint:

```bash
uv run python inference.py --policy trained --model-path outputs/stratarena_grpo --task all
```

Inspect one episode and verify all 3 agents are active:

```bash
uv run python training/inspect_episode.py --task hard --steps 10
```

Export rollouts for training:

```bash
uv run python training/data_export.py --policy mixed --episodes-per-task 80 --output outputs/stratarena_rollouts.jsonl
```

In Google Colab with Unsloth, use the exported JSONL and run:

```bash
python training/unsloth_colab_train.py --dataset outputs/stratarena_rollouts.jsonl --mode sft
python training/unsloth_colab_train.py --dataset outputs/stratarena_rollouts.jsonl --mode grpo
```

Or run the full local pipeline with the same notebook logic:

```bash
uv run python training/unsloth_colab_train.py --dataset outputs/stratarena_rollouts.jsonl --mode all
uv run python training/unsloth_colab_train.py --dataset outputs/stratarena_rollouts.jsonl --mode eval
```

Training notes:

- `training/unsloth_colab_train.py` ports the notebook into a repo script.
- `SFT` uses rollout prompts plus expert completions from the exported dataset.
- `GRPO` uses two rewards: JSON-format compliance and an observation-based environment-shaped reward computed from the prompt payload.
- `inference.py --policy trained` loads the saved RL/SFT model folder and runs it back through the actual StratArena environment.

## Observation summary

Each step exposes:

- current step and remaining budget
- current resource value, scarcity, and market pressure
- opponent behavior signals such as recent wins and bid intensity
- ToM feature vector: inferred budget, aggression, confidence, volatility, exploit signal, uncertainty
- cumulative value won, spend so far, reward breakdown, and live task score

## Why it matters

This environment is designed for training and benchmarking agents that must reason about other agents instead of merely reacting to visible state. It is suitable for strategic allocation research across auctions, negotiation, resource scheduling, and other competitive partially observable settings.
