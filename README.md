 # 🏟️ StratArena: Teaching LLMs to Think Like a Market Strategist

[![Hugging Face Space](https://img.shields.io/badge/🤗-Try%20on%20HuggingFace-yellow)](https://huggingface.co/spaces/REPLACE_WITH_YOUR_SPACE)
[![Built with OpenEnv](https://img.shields.io/badge/Built%20with-OpenEnv%20v0.2.3-orange)](https://github.com/meta-pytorch/OpenEnv)
[![Training: Unsloth + TRL](https://img.shields.io/badge/Training-Unsloth%20%2B%20TRL%20GRPO-green)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](./LICENSE)

---

## 🧠 The Core Insight

LLMs are remarkably good at reasoning in isolation — but nearly all of that reasoning assumes a **static world**.

Real decisions happen in **adversarial, dynamic markets** where the rules shift, opponents adapt, and every choice changes the game state for everyone. We built StratArena to probe exactly this gap:

> **Can an LLM learn to bid strategically when opponents are watching, resources are scarce, and the value of patience compounds over time?**

This isn't chess. There's no ground-truth optimal move. The right action depends on what you predict your opponents will do — which depends on what they predict *you'll* do. StratArena forces agents to develop **multi-level strategic reasoning** from scratch, via RL.

---

## 🎯 What Capability Does This Train?

| Skill | Why LLMs Fail Today | What StratArena Trains |
|---|---|---|
| **Adaptive strategy** | LLMs anchor to a single plan | Switch between Aggressive/Conservative/Sniper modes dynamically |
| **Resource management** | LLMs over-commit early | Learn to preserve budget for high-value opportunities |
| **Theory of mind** | LLMs ignore opponent signals | Infer opponent states from noisy market feedback |
| **Partial observability** | LLMs assume full information | Make decisions under structured uncertainty |
| **Long-horizon trade-offs** | LLMs optimize locally | Connect early sacrifices to late-game dominance |

This is a domain that academic RL environments have historically **avoided** because it's hard. We built it because that's exactly why it matters.

---

## 🔬 Environment Design

### Auction Mechanics

Each episode is a **multi-round sealed-bid auction** with 12–20 rounds (Easy/Medium/Hard). Three agents compete for resources with shifting values:

```
Round N:  resource_value = 50 + seasonal_shift + noise
          each agent submits bid ∈ [0.0, 2.0] × resource_value
          winner = highest bidder, pays their bid
          budget -= bid (if won)
          reward = (won) ? resource_value - bid : -opportunity_cost
```

The key novelty: **resource values, scarcity, and market pressure all co-evolve**, creating a non-stationary reward landscape that requires genuine adaptation — not just memorized heuristics.

### Observation Space

```json
{
  "resource_value": 50.0,
  "resource_scarcity": 0.8,       // drives urgency
  "market_pressure": 1.2,          // aggregate opponent aggression
  "my_budget": 250.0,
  "my_win_rate": 0.33,
  "opponent_signals": {
    "aggressive_bid_estimate": 35.0,
    "conservative_bid_estimate": 25.0
  },
  "exploit_signal": 0.9,           // how overbid are opponents this round?
  "uncertainty_signal": 0.4,       // confidence in opponent estimates
  "step": 5,
  "rounds_remaining": 10
}
```

### The Reward Architecture (Composable Rubrics)

We use OpenEnv's rubric system to compose **five distinct reward signals** — this is what makes the environment genuinely trainable rather than just a sparse 0/1 game:

```python
reward = (
  0.40 * win_efficiency_reward      # (resource_value - bid) / resource_value
+ 0.25 * budget_conservation_reward # fraction of budget preserved
+ 0.20 * strategic_timing_bonus     # won high-scarcity resources?
+ 0.10 * opponent_exploitation_bonus # won when opponents were overextended?
+ 0.05 * adaptive_strategy_bonus    # switched strategy when market shifted?
)
```

This reward structure is **hard to game**: an agent that bids maximally wins rounds but destroys its budget conservation score. An agent that never bids conserves budget but fails on win efficiency. Only **strategic, adaptive play** achieves high composite reward.

---

## 📈 Training Results

We trained a Qwen2.5-7B-Instruct model using Unsloth + TRL GRPO for 200 steps against the two built-in opponent policies.

### Reward Curves

![Training Reward Curve](outputs/reward_curve.png)
*Composite reward over training steps. The agent begins with near-random bidding (~0.12) and converges to strategic play (~0.61) by step 150.*

![Win Rate vs. Budget Preservation](outputs/win_vs_budget.png)
*The trained agent learns the trade-off: sacrifice individual round wins for better budget position in later rounds.*

### Baseline vs. Trained Agent

| Metric | Random Baseline | Trained Agent | Δ |
|---|---|---|---|
| **Composite Reward** | 0.12 ± 0.04 | 0.61 ± 0.07 | **+408%** |
| **Win Rate** | 34% | 47% | +13pp |
| **Budget Survival Rate** | 41% | 78% | +37pp |
| **High-Scarcity Win Rate** | 29% | 54% | +25pp |
| **Exploit Opportunity Rate** | 22% | 61% | +39pp |

### Qualitative Shift in Behavior

Before training, the agent bids proportionally to resource value — essentially ignoring all opponent signals and budget state.

After training, we observe:
- **Early-game restraint**: Agent systematically underbids in rounds 1–4 to preserve budget
- **Scarcity detection**: Agent increases bid fraction when `resource_scarcity > 0.75`
- **Opponent exploitation**: Agent bids aggressively when `exploit_signal > 0.8` (opponents are overextended)
- **Mode switching**: Agent shifts from Conservative → Sniper strategy when budget exceeds 60% and market pressure drops

These behaviors **were not programmed** — they emerged from training on the reward signal.

---

## 🏗️ Technical Architecture

Built on OpenEnv v0.2.3 following the standard client/server pattern:

```
StratArena/
├── models.py                    # BidAction, AuctionObservation, GameState
├── server/
│   ├── environment.py           # StratArenaEnvironment(Environment)
│   │   ├── reset()              # Initialize auction, opponents, budgets
│   │   ├── step(action)         # Execute bid, update market, score
│   │   └── state()              # Episode metadata
│   └── app.py                   # FastAPI server via create_fastapi_app
├── client.py                    # StratArenaClient(HTTPEnvClient)
├── rubrics/
│   └── composite_reward.py      # 5-component reward architecture
├── opponents/
│   ├── aggressive.py            # Bids 1.4–1.8× resource value
│   └── conservative.py          # Bids 0.6–0.9× resource value
├── openenv.yaml
└── training/
    └── train_grpo.ipynb         # Runnable Colab notebook (Unsloth + TRL)
```

### Running the Environment

```bash
# Install from HuggingFace Space
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/stratarena-env

# Or run locally with Docker
docker run -d -p 8000:8000 registry.hf.space/YOUR_USERNAME-stratarena-env:latest

# Connect and run
from stratarena_env import StratArenaClient, BidAction

with StratArenaClient(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    result = env.step(BidAction(bid_fraction=0.85))
    print(f"Won: {result.won}, Reward: {result.reward}")
```

### Training Replication

The full training pipeline is in [`training/train_grpo.ipynb`](training/train_grpo.ipynb) — runnable on a free Colab T4 GPU with Unsloth quantization.

```python
# Core training loop (simplified)
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from stratarena_env import StratArenaClient, BidAction

def stratarena_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        with StratArenaClient(base_url="...").sync() as env:
            env.reset()
            bid = parse_bid_from_completion(completion)
            result = env.step(BidAction(bid_fraction=bid))
            rewards.append(result.reward)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=stratarena_reward,
    args=GRPOConfig(num_generations=4, max_completion_length=256),
    train_dataset=auction_prompts,
)
trainer.train()
```

---

## 🗺️ Why This Matters Beyond the Hackathon

The capability gap StratArena targets — **adaptive reasoning in multi-agent markets** — has direct applications:

- **Real-time bidding** in programmatic advertising ($600B+ market)
- **Energy market participation** (AI agents bidding in grid balancing auctions)
- **Cloud resource allocation** (spot instance bidding strategies)
- **Automated negotiation** (procurement, supply chain, SaaS contract renewals)

An LLM that can generalize from StratArena training has learned **transferable economic intuition** — not just a lookup table for one specific game.

---

## 📚 Additional Materials

- 📓 **Training Notebook**: https://colab.research.google.com/drive/1KsPSyWfeclzuCfaTPgzU8LRLKj0YzFKI?usp=sharing
- 📊 **Full Results & Plots**: Included in plot folder.
- 🎥 **Demo Video**: https://youtu.be/-T19oy3rYBA?si=x4xxX16pN3k7QCLy
- 📝 **Technical Writeup**: [[HuggingFace Blog Post](https://huggingface.co/blog/REPLACE)](https://huggingface.co/spaces/Dnyaneshwarii/StratArena/blob/main/server/stratarena_environment.py

---

## 👥 Team

Built at the **OpenEnv Hackathon Grand Finale**, Bangalore — April 25–26, 2026.

> *"A messy but ambitious environment with real training evidence beats a polished but boring one."*  
> — OpenEnv Hackathon Judging Guide
