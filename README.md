# 🎯 StratArena_Env: Multi-Agent Bidding Strategy Learning

[![Hugging Face Space – Try Online](https://img.shields.io/badge/🤗-Run%20on%20Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/REPLACE_WITH_YOUR_SPACE)
[![OpenEnv](https://img.shields.io/badge/Built%20with-OpenEnv-orange)](https://github.com/openenv/openenv)
[![License](https://img.shields.io/badge/License-MIT-blue)](./LICENSE)

## 🚀 What is StratArena_Env?

**StratArena_Env** is an innovative, OpenEnv-compatible reinforcement learning environment that teaches language models and RL agents to master **dynamic multi-agent auction bidding**—a problem sitting at the intersection of game theory, economic strategy, and real-time decision-making.

Unlike traditional RL benchmarks (chess, grid-worlds, snake), StratArena targets a **genuine capability gap**: LLMs currently struggle with adaptive strategies in partial-observability markets with:
- **Incomplete information** about opponent intentions
- **Shifting market dynamics** (resource values change mid-game)
- **Compounding budget constraints** (every bid matters; you can't recover lost resources)
- **Multi-horizon rewards** (exploit now vs. save for future opportunities)

This environment teaches agents to:
1. **Reason about imperfect information** (what are the opponents planning?)
2. **Switch strategies dynamically** (adapt on-the-fly based on market feedback)
3. **Manage scarce resources** (spend wisely; don't deplete your budget)
4. **Detect & exploit opportunities** (win high-value resources when opponents are weak)

---

## 🎬 Problem Statement: The Innovation Angle

### Why This Problem?

Current LLMs and RL agents excel at:
- ✅ Game trees with perfect information (chess)
- ✅ Deterministic, reward-dense environments
- ✅ Single-agent planning

But struggle with:
- ❌ **Partial observability** + multi-agent interaction
- ❌ **Dynamic, shifting reward structures** (resources get more/less valuable)
- ❌ **Long-horizon trade-offs** where early mistakes compound
- ❌ **Theory of mind** (inferring opponent models from limited signals)

**StratArena_Env** is designed to push LLMs to reason about all four simultaneously—a capability gap with real-world applications:
- **Auction design & bidding strategies** (energy markets, ad auctions)
- **Resource allocation under uncertainty** (cloud computing, bandwidth)
- **Strategic negotiation** (business deals, diplomacy)

---

## 🎮 Environment Overview

### Core Mechanics

**Agent Role:** You compete against two opponents in a **multi-round auction**.

| **Component**     | **Description**                                                                     |
|-------------------|-------------------------------------------------------------------------------------|
| **Observation**   | Current round resource value, market pressure estimates, your budget, opponent signals |
| **Action Space**  | Continuous allocation ∈ [0.0, 2.0] (bid as fraction of resource value)            |
| **Opponents**     | Two AI agents with different strategies (Aggressive, Conservative)                 |
| **Task Variants** | Easy (12 steps), Medium (15 steps), Hard (20 steps) with increasing complexity     |

### State Information

The agent observes:
```json
{
  "resource_value": 50.0,            // Current resource on auction
  "resource_scarcity": 0.8,          // How rare is this resource?
  "market_pressure": 1.2,            // How aggressive are competitors?
  "my_budget": 250.0,                // Your remaining budget
  "opponent_signals": {
    "aggressive_bid_estimate": 35.0,
    "conservative_bid_estimate": 25.0
  },
  "step": 5,                         // Current step / 12-20
  "exploit_signal": 0.9,             // How exploitable is this round?
  "uncertainty_signal": 0.4          // How confident are opponent estimates?
}
