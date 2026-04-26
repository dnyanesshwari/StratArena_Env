# How We Tried to Teach an LLM to Understand an Opponent

When I first learned reinforcement learning, I understood it in the simplest way: an agent learns through rewards and penalties. That is the definition most of us start with. It is correct, but while building this project, I realized that reinforcement learning becomes much more meaningful when it is used not just to optimize actions, but to shape behavior in uncertain, human-like situations.

That realization did not come from theory alone. It came from this hackathon.

My team and I were not trying to build just another benchmark or another model demo. We were trying to explore a real weakness we kept noticing in modern LLMs. Today, LLMs can write code, summarize documents, answer questions, and reason surprisingly well. In many cases, they can imitate parts of human thinking. But there is still a very important gap.

LLMs are still weak at understanding an opponent.

That sounds simple, but it matters a lot.

In real life, many important decisions are not made in isolation. They happen in negotiation, auctions, resource allocation, strategic planning, and any setting where another agent is also trying to win. In those situations, success does not come only from solving a visible problem. It comes from reading the other side.

Humans do this naturally. During a negotiation, a person can often sense when the other side is under pressure, hiding weakness, becoming aggressive, or changing strategy. We may not know their exact internal state, but we infer it from behavior. We observe timing, confidence, hesitation, pressure, and patterns. Then we adjust.

That is a very powerful kind of intelligence.

And we felt that most LLM systems still do not do this well.

That is where our project began.

## The question that pushed us

We asked ourselves a simple but deep question:

**Can we create an environment where an LLM learns not just to act, but to understand the behavior of other agents under uncertainty?**

That question led us to build **StratArena**.

StratArena is our multi-agent reinforcement learning environment designed for strategic settings where opponent modeling matters. Instead of giving the learner a clean, fully visible world, we designed a setting where important information is hidden. The model cannot directly see the true budget or full strategy of its opponents. It only sees signals from their actions.

That design choice was intentional.

We wanted the model to face the same kind of uncertainty humans face in real strategic interaction. You do not get perfect information. You only get behavior. From that behavior, you try to build belief.

## From idea to environment

Once we became clear on the problem, the next challenge was to translate it into a trainable environment.

We built StratArena around competitive decision-making. The learner competes against two different kinds of opponents: one aggressive and one conservative. Each of them behaves differently, spends differently, and shifts pressure differently across an episode. The learner has to decide how much to allocate in each round while the environment keeps changing.

To make the environment meaningful, we made it partially observable. The learner does not get direct access to hidden internal state. Instead, it observes practical signals such as recent wins, bid intensity, market pressure, resource scarcity, and behavioral trends over time.

This was important for us because we did not want to build a toy task where the right answer is obvious from the state itself. We wanted to create a setting where the model has to infer what is happening behind the scenes.

That is where our most important innovation came in.

## The heart of the project: Theory of Mind

The novelty of our work is the use of a **Theory of Mind-inspired mechanism** inside the environment.

In simple terms, Theory of Mind is the ability to form an internal estimate of what another agent may be thinking, intending, or hiding. Humans use this constantly. In strategic interaction, it is often the difference between average decision-making and strong decision-making.

We wanted to bring that idea into LLM training.

So in our environment, we built a belief-tracking component that continuously estimates opponent state from observable behavior. Instead of revealing truth directly, the environment produces evolving belief signals such as estimated budget, aggression level, confidence, volatility, exploitability, and uncertainty.

This changed the whole nature of the problem.

Now the learner was not only deciding, “Should I bid high or low?”  
It was learning to ask, in effect, “What do I think my opponent’s condition is right now? How certain am I? Is this the right moment to exploit, defend, or wait?”

That made the project feel much closer to real strategic intelligence.

## Why we chose these kinds of tasks

We did not want the environment to be narrow or useful for only one artificial setting. We wanted it to generalize to classes of real-world interactions where reading the other side matters.

That is why we shaped the environment around tasks like:

- negotiation-like strategic interaction
- resource allocation under competition
- auction-style decision-making

These are all scenarios where single-agent logic is not enough. A model can be intelligent in isolation and still fail badly if it cannot model the behavior of others.

That was the core problem we were trying to solve.

## The three stages of challenge

To make the learning process structured, we designed three tasks.

The first task focuses on **balanced strategy learning**. Here the learner must capture value without destroying its own budget too early. This is the foundation: discipline before aggression.

The second task focuses on **opponent exploitation**. In this setting, the learner must notice when an opponent is weak, depleted, or predictable, and then act more aggressively in those specific moments.

The third task focuses on **dynamic adaptation**. This is the hardest setting. In the middle of the episode, the opponent behavior changes. That means the learner cannot survive with one fixed pattern. It has to update its beliefs and adapt after the regime shift.

This progression mattered to us. It mirrors how strategic understanding develops: first control yourself, then read weakness, then adapt when the world changes.

## Reward was not enough by itself, so we shaped it carefully

One thing we learned quickly is that if you want the agent to learn rich strategic behavior, you cannot rely on a shallow reward alone.

So we designed reward functions that captured more than just winning.

Our reward considered things like:

- value captured
- spending efficiency
- whether the action exploited a real opportunity
- whether the agent avoided waste under pressure
- penalties for overbidding or poor budget management

This was one of the most educational parts of the project for me. It showed that building an RL environment is not only about coding transitions and actions. It is also about deciding what kind of intelligence you want the system to value.

In our case, we wanted the environment to reward strategic judgment, not blind aggression.

## One of the most exciting parts: no fixed dataset

Another thing that made this project feel genuinely innovative was that we were not training from a traditional fixed dataset alone.

The environment itself generates rollouts in real time. Those interactions become the training material.

That means the data is not just static examples collected once and frozen. It is created from live strategic behavior inside the environment. The model learns from situations that emerge through interaction.

For me, this was one of the most beautiful parts of the whole system. It felt like we were not only feeding data to the model. We were building a world in which the model could learn.

## How we trained the LLM

To make this practical, we used a two-stage pipeline.

First, we used **SFT** to help the model learn the structure of the task and action behavior from rollout data. This gave the model a supervised starting point.

After that, we used **GRPO** so the model could improve through reward-driven optimization. This stage mattered because imitation alone is not enough for strategic reasoning. A model may learn the format of good behavior from examples, but it becomes more interesting when it starts receiving feedback from consequences.

We used **`meta-llama/Llama-3.2-1B-Instruct`** in this training flow. That gave us a practical model size for experimentation while still letting us test the full idea.

## What we built beyond the core environment

As the project grew, it became more than just an environment file or a notebook experiment.

We built the surrounding system too:

- an OpenEnv-compatible environment
- training scripts for SFT and GRPO
- rollout export pipeline
- evaluation across tasks
- inference flow for trained models
- a dashboard and API to inspect episodes, beliefs, bids, rewards, and adaptation behavior

That part mattered a lot to us because we wanted this project to be inspectable. We wanted to see the model not just as a final score, but as a learner moving through uncertainty.

## What this project taught me

This project changed how I think about reinforcement learning for LLMs.

Before this, RL felt to me like a framework mostly described through definitions: reward, penalty, policy, optimization. After building StratArena, it started to feel much more alive. It became a way to teach a model something subtle: how to behave when the answer is not fully visible and when another mind is part of the problem.

That is what made this hackathon special for me.

We were not only building a system. We were trying to teach a machine one small part of strategic human intelligence: the ability to form a belief about another agent and act under uncertainty.

That is still a hard problem. We are not claiming it is solved. But I believe this project is a genuine step in that direction.

## Closing reflection

Our honest goal in this hackathon was to build an RL environment that helps an LLM learn opponent behavior in settings where one-way reasoning is not enough.

In tasks like negotiation, auctions, and resource allocation, success depends on more than logic. It depends on timing, adaptation, uncertainty handling, and mental modeling of the other side.

That is what we tried to capture with StratArena.

As AI keeps advancing, I believe reinforcement learning can help LLMs grow beyond static response generation and become more adaptive, more strategic, and more useful in interactive real-world settings.

This project was our attempt to explore that future, not just by talking about it, but by building toward it. +
 
