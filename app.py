"""StratArena Gradio UI launcher.

This app is optimized for Hugging Face Spaces or local demos.
It runs a single episode, visualizes strategy adaptation, and
shows a compact RL dashboard for the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt

from inference import (
    AdaptiveStrategyController,
    LocalTrainedPolicy,
    OpenAIPolicy,
    clamp_allocation,
    heuristic_allocation,
    _per_round_spend_cap,
)
from models import StratArenaAction
from server.stratarena_environment import StratArenaEnvironment

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUCCESS_SCORE_THRESHOLD = 0.55

STRATEGIES = {
    "PROBE": 0,
    "EXPLOIT": 1,
    "DEFEND": 2,
    "RECOVER": 3,
}


def plot_line(steps: list[int], values: list[float], title: str, ylabel: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    ax.plot(steps, values, marker="o", color=color, linewidth=2, alpha=0.92)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_strategy_timeline(steps: list[int], strategies: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    values = [STRATEGIES.get(mode, 0) for mode in strategies]
    ax.step(steps, values, where="post", color="#2a9d8f", linewidth=2)
    ax.scatter(steps, values, color="#264653", s=40)
    ax.set_yticks(list(STRATEGIES.values()))
    ax.set_yticklabels(list(STRATEGIES.keys()), fontsize=9)
    ax.set_title("Strategy timeline", fontsize=12, pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Strategy")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def run_episode_ui(
    task: str,
    policy_name: str,
    model_path: str,
    seed: int,
) -> tuple[str, plt.Figure, plt.Figure, plt.Figure, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episode_seed = int(seed) if seed is not None else None
    env = StratArenaEnvironment()
    obs = env.reset(task=task, seed=episode_seed)

    policy = None
    policy_note = "Adaptive heuristic"
    if policy_name == "openai":
        if OPENAI_KEY:
            policy = OpenAIPolicy(model=MODEL_NAME, api_key=OPENAI_KEY, base_url=API_BASE_URL)
            policy_note = f"OpenAI {MODEL_NAME}"
        else:
            policy_note = "OPENAI_API_KEY missing — fallback to heuristic"
    elif policy_name == "trained":
        model_path = model_path.strip() if model_path else ""
        if model_path:
            policy = LocalTrainedPolicy(model_path=model_path)
            policy_note = f"Trained model {Path(model_path).name}"
        else:
            policy_note = "Trained model path not set — fallback to heuristic"

    controller = AdaptiveStrategyController(task=task)
    steps_taken = 0
    last_reward = 0.0

    allocations: list[float] = []
    rewards: list[float] = []
    strategies: list[str] = []
    step_history: list[dict[str, Any]] = []

    while not obs.done:
        current_strategy = controller.update(obs, step_history, steps_taken, last_reward)

        if policy is None:
            allocation, _ = heuristic_allocation(obs, step_history, current_strategy)
        else:
            try:
                allocation, _ = policy.act(task, obs, step_history, controller)
                h_alloc, _ = heuristic_allocation(obs, step_history, current_strategy)
                allocation = clamp_allocation(0.70 * allocation + 0.30 * h_alloc)
                allocation = min(allocation, _per_round_spend_cap(obs))
            except Exception as exc:
                allocation, _ = heuristic_allocation(obs, step_history, current_strategy)
                policy_note = f"{policy_note} (failed, using heuristic)"

        obs = env.step(StratArenaAction(allocation=allocation))
        steps_taken += 1

        base_reward = float(getattr(obs, "reward", 0.0))
        task_score = float(getattr(obs, "task_score", 0.0))
        exploit_signal = float(obs.exploit_signal)
        uncertainty_signal = float(obs.uncertainty_signal)
        remaining_budget = float(getattr(obs, "my_budget", 0.0))

        budget_bonus = 0.5 if remaining_budget > 200 else -0.5
        missed_opportunity = exploit_signal > 0.6 and allocation < 0.5
        strategy_bonus = 0.0
        if current_strategy == "EXPLOIT" and allocation >= 0.7:
            strategy_bonus = 0.3
        elif current_strategy == "DEFEND" and allocation <= 0.2:
            strategy_bonus = 0.2

        reward = (
            base_reward
            + 0.5 * task_score
            + 0.3 * exploit_signal
            - 0.2 * uncertainty_signal
            + budget_bonus
            + strategy_bonus
        )
        if missed_opportunity:
            reward -= 1.5

        last_reward = reward
        allocations.append(allocation)
        rewards.append(reward)
        strategies.append(current_strategy)

        step_history.append(
            {
                "step": steps_taken,
                "allocation": round(allocation, 4),
                "reward": round(reward, 4),
                "strategy": current_strategy,
                "exploit_signal": round(exploit_signal, 4),
                "uncertainty_signal": round(uncertainty_signal, 4),
                "task_score": round(task_score, 4),
                "budget_ratio": round(obs.my_budget_ratio, 4),
            }
        )

    score = env.grade()
    success = score >= SUCCESS_SCORE_THRESHOLD
    transitions = [
        {
            "step": t.step,
            "from": t.from_mode,
            "to": t.to_mode,
            "trigger": t.trigger,
        }
        for t in controller.transitions
    ]

    summary = f"""
### StratArena episode complete
- **Task:** {task}
- **Policy:** {policy_name}
- **Effective mode:** {policy_note}
- **Seed:** {episode_seed}
- **Final score:** {score:.4f}
- **Success:** {'Yes' if success else 'No'}
- **Steps:** {steps_taken}
- **Adaptation transitions:** {len(transitions)}
- **Final bandit values:** {controller.bandit.values()}
"""

    allocation_figure = plot_line(list(range(1, steps_taken + 1)), allocations, "Allocation per step", "Allocation", "#1d3557")
    reward_figure = plot_line(list(range(1, steps_taken + 1)), rewards, "Shaped reward per step", "Reward", "#e63946")
    strategy_figure = plot_strategy_timeline(list(range(1, steps_taken + 1)), strategies)

    return (
        summary,
        allocation_figure,
        reward_figure,
        strategy_figure,
        step_history,
        transitions,
        controller.bandit.values(),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="StratArena RL Explorer") as demo:
        gr.Markdown(
            """
# StratArena RL Explorer

A compact Gradio dashboard for the StratArena environment.
Run a single episode and see how strategy, allocation, and reward evolve.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    label="Task",
                    choices=["easy", "medium", "hard"],
                    value="medium",
                )
                policy_dropdown = gr.Radio(
                    label="Policy",
                    choices=["heuristic", "trained", "openai"],
                    value="heuristic",
                )
                model_path = gr.Textbox(
                    label="Trained model folder",
                    value="outputs/stratarena_sft",
                    placeholder="Leave blank if not using trained policy",
                )
                seed_input = gr.Number(
                    label="Episode seed",
                    value=42,
                    precision=0,
                )
                run_button = gr.Button("Run episode", variant="primary")
                info_text = """
- `heuristic` is the default adaptive baseline.
- `trained` loads a local checkpoint.
- `openai` uses the OpenAI API if `OPENAI_API_KEY` is set.
"""
                gr.Markdown(info_text)

            with gr.Column(scale=2):
                summary = gr.Markdown("### Ready to run an episode")
                allocation_plot = gr.Plot(label="Allocation chart")
                reward_plot = gr.Plot(label="Reward chart")
                strategy_plot = gr.Plot(label="Strategy timeline")

        with gr.Tabs():
            with gr.Tab("Episode trace"):
                trace_table = gr.Dataframe(
                    headers=[
                        "step",
                        "allocation",
                        "reward",
                        "strategy",
                        "exploit_signal",
                        "uncertainty_signal",
                        "task_score",
                        "budget_ratio",
                    ],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Strategy history"):
                transition_table = gr.Dataframe(
                    headers=["step", "from", "to", "trigger"],
                    interactive=False,
                    wrap=True,
                )
                bandit_json = gr.JSON(label="Bandit Q-values")

        run_button.click(
            fn=run_episode_ui,
            inputs=[task_dropdown, policy_dropdown, model_path, seed_input],
            outputs=[
                summary,
                allocation_plot,
                reward_plot,
                strategy_plot,
                trace_table,
                transition_table,
                bandit_json,
            ],
        )

    return demo


gradio_app = build_ui()

if __name__ == "__main__":
    gradio_app.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=False,
        share=False,
    )
