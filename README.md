# StratArena 🏟️

**A multi-agent reinforcement learning environment for teaching LLMs strategic reasoning, opponent modeling, and Theory-of-Mind under uncertainty.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-llama/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What is StratArena?

StratArena is a fully self-contained RL training and evaluation environment built on the [OpenEnv](https://github.com/meta-llama/openenv) standard. It puts a learning agent into competitive multi-round auctions against two scripted opponents — one **aggressive**, one **conservative** — and forces the agent to infer hidden opponent state from observable behavior alone.

The agent never sees its opponents' real budgets or strategies directly. It must build beliefs, spot weakness, exploit opportunity at the right moment, and adapt when the opponents change behavior mid-episode. That is what makes StratArena interesting: it is not a single-agent optimization task — it is a social reasoning task wrapped in a reinforcement learning shell.

> *"We were not only building a system. We were trying to teach a machine one small part of strategic human intelligence: the ability to form a belief about another agent and act under uncertainty."*

---

## Key Innovations

### 1. Theory-of-Mind in the Reward Loop

Most RL environments give the agent clean state. StratArena deliberately withholds it. Instead, a **`ToMTracker`** component runs every step and maintains evolving belief estimates for each opponent:

| Belief signal | What it captures |
|---|---|
| `budget_belief` | Estimated fraction of opponent budget remaining |
| `aggression_belief` | Inferred bid intensity level |
| `confidence` | How certain the agent's beliefs are (grows with observation count) |
| `volatility_belief` | How unpredictably the opponent's behavior is changing |
| `exploit_signal` | Combined measure: depletion × predictability — when to strike |
| `uncertainty_signal` | When to hold back |

These 10 features are fed directly into every observation, making Theory-of-Mind a first-class training signal rather than a post-hoc analysis tool.

### 2. Structured, Three-Tier Task Progression

The environment ships three tasks that build on each other:

| Task | ID | Core challenge | Grading weights |
|---|---|---|---|
| **Easy** | `basic_competition` | Capture value without burning budget | 45% value + 35% efficiency + 20% budget |
| **Medium** | `opponent_exploitation` | Exploit depleted opponents at the right moment | 40% exploit rate + 25% belief alignment + 20% smart passes + 15% efficiency |
| **Hard** | `adaptive_strategy` | Adapt when opponent strategy shifts mid-episode (step 25) | 40% adaptation + 25% belief alignment + 20% post-shift efficiency + 15% exploit |

This mirrors how real strategic intelligence develops: first discipline, then exploitation, then adaptation.

### 3. Shaped, Decomposed Reward Function

A shallow win/loss reward is not enough for teaching strategic behavior. StratArena's reward function has four explicit components, each returned in every observation:

```
reward = value_component       # 0.12 × resource_value on a win
       + efficiency_component  # edge between value and bid cost
       + strategy_component    # bonus for exploiting weak opponents (4.5 + 2.5 × exploit_signal)
       + penalty_component     # overbid and budget exhaustion penalties
```

Task-specific scaling then amplifies the signal most relevant to each challenge level.

### 4. Adaptive Strategy Controller + Online Bandit Learning

The built-in inference engine goes beyond a static heuristic. It pairs a **state machine** (PROBE → EXPLOIT → DEFEND → RECOVER) with a lightweight **epsilon-greedy BanditAdapter** that updates Q-values from live reward signal every step — no GPU, no model, just within-episode learning.

```
PROBE   → gather opponent intel, bid lightly (0.3–0.6)
EXPLOIT → opponent depleted, press hard (0.8–1.4)
DEFEND  → market crowded or uncertain, near-zero bid (0.0–0.3)
RECOVER → loss streak, rebuild cautiously (0.4–0.7)
```

Strategy transitions are logged, returned to the UI, and fed into the LLM prompt context — so the language model knows *when* to pivot, not just *how much* to bid.

### 5. Three Policy Tiers — from Zero-Cost Heuristic to Fine-Tuned LLM

StratArena supports three interchangeable inference backends:

| Tier | What runs | Use case |
|---|---|---|
| **Heuristic** | `heuristic_allocation()` | Fast baseline, zero API cost |
| **OpenAI LLM** | `OpenAIPolicy` (GPT-4o-mini default) | Cloud-based reasoning, JSON output |
| **Local fine-tuned** | `LocalTrainedPolicy` (LoRA / Unsloth) | Trained checkpoints in `outputs/` |

All three policies share the same prompt format and observation schema, making it trivial to swap backends and compare behavior.

### 6. End-to-End RL Training Pipeline

The project ships a complete two-stage training pipeline:

```
1. Collect rollouts   →  outputs/stratarena_rollouts.jsonl   (heuristic expert trajectories)
2. SFT               →  outputs/stratarena_sft/              (supervised warm-start)
3. GRPO              →  outputs/stratarena_grpo/             (reward-driven optimization)
```

The GRPO stage uses the same `compute_reward()` function that runs live inside the environment, closing the training-inference loop. The base model is `meta-llama/Llama-3.2-1B-Instruct`, trained via [Unsloth](https://github.com/unslothai/unsloth) for efficiency.

### 7. Live Inspection Dashboard

A full-stack arena dashboard lets you observe every mechanic in real time:

- Animated per-round bids for all three agents
- ToM belief table (budget belief, aggression, confidence, predicted style)
- Exploit signal and uncertainty signal bars
- Decomposed reward components (value / efficiency / strategy / penalty)
- Strategy transition log with bandit Q-values
- Cumulative reward and bid comparison charts (Chart.js)
- Configurable scenario framing: **Auction**, **Negotiation**, or **Resource Allocation**

---

## Architecture at a Glance

```
┌──────────────────────────────────┐
│  Arena UI  (browser SPA)         │  ← Vanilla JS + Chart.js
│  arena_ui/index.html             │
└──────────────┬───────────────────┘
               │ HTTP REST (JSON)
               ▼
┌──────────────────────────────────┐
│  FastAPI Server  (server/app.py) │  ← OpenEnv routes + Dashboard API
│  /reset  /step  /state  /health  │
│  /api/episode/*                  │
└──────────────┬───────────────────┘
               │
  ┌────────────┴──────────────────────────┐
  │                                       │
  ▼                                       ▼
EpisodeSession                  StratArenaEnvironment
(dashboard_api.py)              (stratarena_environment.py)
  AdaptiveStrategyController      AggressiveAgent
  BanditAdapter                   ConservativeAgent
  StepTrace list                  ToMTracker
  heuristic_allocation()          compute_reward()
                                  sample_resource_round()
```

---

## Project Structure

```
StratArena_Env/
├── models.py                    # Pydantic schemas (Observation, Action, State, Reward)
├── inference.py                 # Heuristic, OpenAI, LocalModel policies + Bandit + Controller
├── client.py                    # OpenEnv client wrapper (for external agent evaluation)
├── evaluate_tasks.py            # Top-level eval entry point
├── app.py                       # Dashboard-only launcher (port 8001)
├── openenv.yaml                 # OpenEnv manifest
│
├── server/
│   ├── app.py                   # Main FastAPI app (OpenEnv + Dashboard + static UI)
│   ├── dashboard_api.py         # EpisodeSession orchestrator + REST endpoints
│   ├── stratarena_environment.py # Core Environment class
│   ├── agents/                  # AggressiveAgent, ConservativeAgent
│   ├── observation/builder.py   # env state → Observation / State conversion
│   ├── reward/reward.py         # 4-component reward function
│   ├── tasks/                   # TaskDefinition registry (easy / medium / hard)
│   ├── tom/tom_tracker.py       # Theory-of-Mind belief tracker
│   ├── utils/market.py          # Stochastic market generator
│   └── evaluate_tasks.py        # run_task() heuristic benchmark runner
│
├── evaluation/
│   ├── benchmarks.py            # benchmark_all_tasks()
│   ├── evaluate.py              # Standalone eval main
│   └── plots.py                 # ASCII bar charts and metric tables
│
├── arena_ui/                    # Frontend (HTML + CSS + JS)
├── training/                    # Unsloth SFT + GRPO notebook
└── outputs/                     # Saved LoRA adapters (SFT + GRPO)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) CUDA GPU for local model inference

### Installation

```bash
git clone https://github.com/dnyanesshwari/StratArena_Env.git
cd StratArena_Env
pip install -r requirements.txt
```

Or use `uv` (recommended):

```bash
uv sync
```

### Run the full server (OpenEnv + Dashboard + Arena UI)

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Run the dashboard-only launcher

```bash
python app.py          # port 8001 by default
# or
PORT=8001 python app.py
```

### Docker

```bash
# Full server (OpenEnv-compatible, port 7860)
docker build -f server/Dockerfile -t stratarena .
docker run -p 7860:7860 stratarena

# Dashboard-only
docker build -t stratarena-ui .
docker run -p 8001:8001 stratarena-ui
```

---

## Running Evaluation

### Heuristic benchmark across all tasks

```bash
python server/evaluate_tasks.py
```

Output example:

```
  task |  score |    value |    spend | wins | exploit | belief | adapt
----------------------------------------------------------------------
  easy | 0.9020 |  1021.34 |   382.10 |   28 |   0.833 |  0.741 | 0.000
medium | 0.6640 |   891.22 |   441.67 |   23 |   0.611 |  0.688 | 0.000
  hard | 0.5550 |   934.45 |   512.33 |   21 |   0.500 |  0.622 | 0.320
```

### Benchmark with a custom policy

```python
from evaluation.benchmarks import benchmark_all_tasks
from evaluation.plots import print_full_report

rows = benchmark_all_tasks(policy=my_policy_fn)  # policy: (obs) -> (float, str)
print_full_report(rows)
```

---

## Environment Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server listen port |
| `DASHBOARD_PORT` | `8001` | Dashboard-only launcher port |
| `OPENAI_API_KEY` | — | API key for `OpenAIPolicy` |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model name |
| `API_BASE_URL` | `https://api.openai.com/v1` | Compatible with any OpenAI-format endpoint |

---

## OpenEnv API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit one action, receive observation |
| `GET` | `/state` | Full internal environment state |
| `GET` | `/health` | Health check |
| `POST` | `/api/episode/start` | Start a dashboard episode |
| `POST` | `/api/episode/{id}/step` | Advance one round |
| `GET` | `/api/episode/{id}/summary` | Final score and metrics |

---

## Training Your Own Model

Open `training/StratArena_Training_unsloth.ipynb` and follow the two stages:

1. **SFT** — trains on `outputs/stratarena_rollouts.jsonl` using cross-entropy on expert heuristic rollouts.
2. **GRPO** — continues from the SFT checkpoint using Group Relative Policy Optimization, with StratArena's `compute_reward()` as the live reward signal.

Pre-trained LoRA adapters are included in `outputs/stratarena_sft/` and `outputs/stratarena_grpo/` and can be loaded immediately:

```python
from inference import LocalTrainedPolicy

policy = LocalTrainedPolicy("outputs/stratarena_grpo")
allocation, reason = policy.act(task="medium", obs=obs, trace=trace)
```

---

## Why This Matters

Most LLM benchmarks test knowledge retrieval or single-agent reasoning. StratArena targets a different, harder capability: **understanding another agent's hidden state and adapting your behavior in response**.

This matters because many real-world decisions happen in interactive settings — negotiation, resource allocation, competitive bidding, coalition building. In those settings, logic alone is not enough. What matters is:

- Reading weakness before it becomes obvious
- Knowing when to press and when to conserve
- Updating beliefs when the world changes
- Not being fooled by apparent patterns that are about to break

StratArena is designed to stress-test and train exactly those capabilities.

---

## Citation / Acknowledgements

Built by **TechnoAIGirls** during a hackathon on multi-agent RL for LLMs.
Base model: `meta-llama/Llama-3.2-1B-Instruct`.
Training framework: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl).
Environment standard: [OpenEnv](https://github.com/meta-llama/openenv).

---

## License

MIT
