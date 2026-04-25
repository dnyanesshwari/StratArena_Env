from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference import clamp_allocation


SYSTEM_PROMPT = """You are the learner policy for StratArena.
You receive a partial-observability game state.
Return only JSON with this schema:
{"allocation": <float 0-2>, "reason": "<short phrase>"}
Pick stronger allocations only when value is strong or an opponent looks weak.
Reduce allocation when pressure or uncertainty is high."""

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 1024


@dataclass(frozen=True)
class PredictionRecord:
    model: str
    task: str
    step: int
    predicted_alloc: float | None
    reference_alloc: float | None
    reference_reward: float
    simulated_reward: float
    format_ok: bool
    alloc_error: float | None
    raw_output: str


def load_rollout_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def build_sft_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            "completion": row["completion"],
            "task": row["task"],
            "reward": row["reward"],
            "task_score": row["task_score"],
        }
        for row in rows
    ]


def build_grpo_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            "task": row["task"],
            "reference_reward": row["reward"],
            "reference_score": row["task_score"],
            "metadata": row.get("metadata", {}),
        }
        for row in rows
    ]


def extract_allocation(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r'"allocation"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not match:
        return None
    try:
        return clamp_allocation(float(match.group(1)))
    except ValueError:
        return None


def parse_completion_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        parts: list[str] = []
        for chunk in item:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text", chunk.get("content", ""))))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    if isinstance(item, dict):
        return str(item.get("content", item))
    return str(item)


def format_reward_func(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        rewards.append(0.2 if extract_allocation(parse_completion_text(completion)) is not None else -1.0)
    return rewards


def simulated_env_reward(allocation: float, obs: dict[str, Any]) -> float:
    resource_value = float(obs.get("resource_value", 50.0))
    market_pressure = float(obs.get("market_pressure", 1.0))
    uncertainty = float(obs.get("uncertainty_signal", 0.5))
    exploit = float(obs.get("exploit_signal", 0.0))

    value_norm = resource_value / 100.0
    ideal_bid = value_norm * (1.0 - 0.3 * uncertainty) * min(1.5, market_pressure)
    ideal_bid = clamp_allocation(ideal_bid)

    dist = abs(allocation - ideal_bid) / 2.0
    reward = max(-1.0, 1.0 - 2.0 * dist) + 0.3 * exploit * float(allocation > 0.8)
    return float(reward)


def grpo_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return format_reward_func(completions, **kwargs)


def grpo_env_reward(
    prompts: list[list[dict[str, str]]],
    completions: list[Any],
    task: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for prompt_messages, completion, _task_name in zip(prompts, completions, task):
        user_content = prompt_messages[-1]["content"]
        allocation = extract_allocation(parse_completion_text(completion))
        if allocation is None:
            rewards.append(-2.0)
            continue
        try:
            obs = json.loads(user_content)
        except json.JSONDecodeError:
            rewards.append(-1.5)
            continue
        rewards.append(simulated_env_reward(allocation, obs))
    return rewards


def sample_rollout_rows(rows: list[dict[str, Any]], max_samples: int = 60) -> list[dict[str, Any]]:
    if not rows:
        return []
    step = max(1, len(rows) // max_samples)
    return rows[::step][:max_samples]


def _get_value(item: PredictionRecord | dict[str, Any], key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key)


def summarize_predictions(results: list[PredictionRecord | dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "format_rate": 0.0,
            "avg_reward": 0.0,
            "avg_alloc_error": 0.0,
        }
    format_rate = sum(1.0 for item in results if bool(_get_value(item, "format_ok"))) / len(results)
    avg_reward = sum(float(_get_value(item, "simulated_reward")) for item in results) / len(results)
    alloc_errors = [
        float(_get_value(item, "alloc_error"))
        for item in results
        if _get_value(item, "alloc_error") is not None
    ]
    avg_alloc_error = sum(alloc_errors) / len(alloc_errors) if alloc_errors else 0.0
    return {
        "format_rate": format_rate,
        "avg_reward": avg_reward,
        "avg_alloc_error": avg_alloc_error,
    }


def build_chat_text(tokenizer: Any, prompt_messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_lines = [f"{message['role'].upper()}: {message['content']}" for message in prompt_messages]
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)
