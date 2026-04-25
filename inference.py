from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import StratArenaAction, StratArenaObservation
from server.stratarena_environment import StratArenaEnvironment

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str | None = os.getenv("OPENAI_API_KEY")

BENCHMARK = "stratarena"
SUCCESS_SCORE_THRESHOLD = 0.55

SYSTEM_PROMPT = """You are a strategic allocation policy for StratArena.
Return JSON only: {"allocation": <float 0-2>, "reason": "<short>"}
Use theory-of-mind signals:
- low opponent budget belief -> exploit with higher allocation on good value rounds
- high uncertainty or high market pressure -> reduce allocation unless value is strong
- hard task after the regime shift requires rapid adaptation, not constant aggression
Never output anything except JSON."""


@dataclass
class StepTrace:
    step: int
    allocation: float
    reward: float
    exploit_signal: float
    task_score: float
    winner: str


def _tom_dict(obs: StratArenaObservation) -> dict[str, float]:
    features = obs.tom_features + ([0.0] * max(0, 10 - len(obs.tom_features)))
    return {
        "agg_budget": features[0],
        "agg_aggr": features[1],
        "agg_conf": features[2],
        "agg_vol": features[3],
        "con_budget": features[4],
        "con_aggr": features[5],
        "con_conf": features[6],
        "con_vol": features[7],
        "exploit": features[8],
        "uncertainty": features[9],
    }


def heuristic_allocation(obs: StratArenaObservation) -> tuple[float, str]:
    tom = _tom_dict(obs)
    value_score = obs.resource_value * obs.market_pressure
    exploit_window = tom["exploit"] > 0.32 or tom["agg_budget"] < 0.45
    strong_pressure = obs.market_pressure > 1.35
    uncertain = obs.uncertainty_signal > 0.58 or tom["uncertainty"] > 0.58

    if obs.task == "easy":
        if value_score < 28.0:
            return 0.0, "skip weak"
        if value_score > 82.0 and obs.my_budget_ratio > 0.20:
            return 1.10, "take value"
        if exploit_window and value_score > 48.0:
            return 1.20, "press edge"
        return (0.65, "balanced probe") if value_score > 42.0 else (0.0, "wait")

    if obs.task == "medium":
        if exploit_window and value_score > 46.0:
            return 1.35, "exploit weak"
        if strong_pressure and not exploit_window:
            return 0.0, "avoid crowd"
        if uncertain and value_score < 60.0:
            return 0.0, "uncertain skip"
        if value_score > 72.0:
            return 1.00, "good value"
        return (0.55, "light entry") if value_score > 44.0 else (0.0, "pass")

    late_game = obs.step >= (obs.max_steps // 2)
    if late_game and exploit_window and value_score > 50.0:
        return 1.25, "adapt strike"
    if strong_pressure and value_score < 62.0:
        return 0.0, "protect budget"
    if uncertain and value_score < 68.0:
        return 0.0, "reassess"
    if value_score > 78.0:
        return 1.10, "high value"
    return (0.70, "selective test") if value_score > 54.0 else (0.0, "hold")


def build_prompt(task: str, obs: StratArenaObservation, trace: list[StepTrace]) -> str:
    recent_trace: list[dict[str, Any]] = []
    for trace_item in trace[-4:]:
        if hasattr(trace_item, "__dict__"):
            recent_trace.append(trace_item.__dict__)
        elif isinstance(trace_item, dict):
            recent_trace.append(trace_item)
        else:
            recent_trace.append({"value": str(trace_item)})

    payload: dict[str, Any] = {
        "task": task,
        "step": obs.step,
        "budget_ratio": round(obs.my_budget_ratio, 3),
        "resource_value": round(obs.resource_value, 2),
        "resource_scarcity": round(obs.resource_scarcity, 3),
        "market_pressure": round(obs.market_pressure, 3),
        "exploit_signal": round(obs.exploit_signal, 3),
        "uncertainty_signal": round(obs.uncertainty_signal, 3),
        "opponent_signals": obs.opponent_signals,
        "tom_features": obs.tom_features,
        "recent": recent_trace,
    }
    return json.dumps(payload)


def parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def clamp_allocation(value: Any) -> float:
    try:
        return max(0.0, min(2.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class OpenAIPolicy:
    def __init__(self, model: str, api_key: str | None, base_url: str = API_BASE_URL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI policy requires the `openai` package. Install project dependencies first."
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def act(
        self,
        task: str,
        obs: StratArenaObservation,
        trace: list[StepTrace],
    ) -> tuple[float, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(task, obs, trace)},
            ],
        )
        data = parse_json(response.choices[0].message.content or "{}")
        return clamp_allocation(data.get("allocation")), str(data.get("reason", "model"))


def build_chat_text(prompt_messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_lines = [f"{message['role'].upper()}: {message['content']}" for message in prompt_messages]
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)


class LocalTrainedPolicy:
    def __init__(self, model_path: str, max_seq_length: int = 1024) -> None:
        self.max_seq_length = max_seq_length
        self.model_path = self._normalize_model_path(model_path)
        self.model, self.tokenizer = self._load_model(self.model_path, max_seq_length)

    def _normalize_model_path(self, model_path: str) -> str:
        path = Path(model_path)
        if path.is_file():
            return str(path.parent)
        return str(path)

    def _load_model(self, model_path: str, max_seq_length: int) -> tuple[Any, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            FastLanguageModel = None

        if FastLanguageModel is not None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local trained policy requires unsloth, or transformers+torch."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            try:
                from peft import AutoPeftModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "This trained model looks like a PEFT adapter. Install `peft` to load it."
                ) from exc
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        model.eval()
        return model, tokenizer

    def act(
        self,
        task: str,
        obs: StratArenaObservation,
        trace: list[StepTrace],
    ) -> tuple[float, str]:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task, obs, trace)},
        ]
        text = build_chat_text(messages, tokenizer=self.tokenizer)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            data = parse_json(generated or "{}")
        except Exception:
            data = {"allocation": 0.0, "reason": "invalid-json"}
        return clamp_allocation(data.get("allocation")), str(data.get("reason", "trained-model"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_episode(
    task: str,
    *,
    model: str = MODEL_NAME,
    seed: int | None = None,
    policy_name: str = "heuristic",
    base_url: str = API_BASE_URL,
    api_key: str | None = None,
    model_path: str | None = None,
    max_seq_length: int = 1024,
) -> float:
    env = StratArenaEnvironment()
    obs = env.reset(task=task, seed=seed)
    policy: Any | None = None
    policy_label = "heuristic"
    if policy_name == "openai":
        if api_key or API_KEY:
            policy = OpenAIPolicy(model=model, api_key=api_key or API_KEY, base_url=base_url)
            policy_label = model
    elif policy_name == "trained":
        if not model_path:
            raise ValueError("`--model-path` is required when `--policy trained` is used.")
        policy = LocalTrainedPolicy(model_path=model_path, max_seq_length=max_seq_length)
        policy_label = Path(policy.model_path).name or policy.model_path

    rewards: list[float] = []
    trace: list[StepTrace] = []
    steps_taken = 0

    log_start(task, BENCHMARK, policy_label)
    while not obs.done:
        error: str | None = None
        if policy is None:
            allocation, note = heuristic_allocation(obs)
        else:
            try:
                allocation, note = policy.act(task, obs, trace)
            except Exception as exc:
                allocation, note = heuristic_allocation(obs)
                error = str(exc)

        obs = env.step(StratArenaAction(allocation=allocation))
        reward = float(obs.reward or 0.0)
        rewards.append(reward)
        steps_taken += 1
        log_step(steps_taken, f"allocation={allocation:.3f}", reward, obs.done, error)
        trace.append(
            StepTrace(
                step=obs.step,
                allocation=allocation,
                reward=reward,
                exploit_signal=obs.exploit_signal,
                task_score=obs.task_score,
                winner=obs.last_winner,
            )
        )

    score = env.grade()
    log_end(score >= SUCCESS_SCORE_THRESHOLD, steps_taken, score, rewards)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="StratArena inference runner")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "openai", "trained"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores: dict[str, float] = {}
    for task in tasks:
        scores[task] = run_episode(
            task,
            model=args.model,
            seed=args.seed,
            policy_name=args.policy,
            base_url=args.base_url,
            api_key=args.api_key,
            model_path=args.model_path,
            max_seq_length=args.max_seq_length,
        )

    if len(scores) > 1:
        average = sum(scores.values()) / len(scores)
        print(
            "[DEBUG] SUMMARY "
            + " | ".join(f"{task}={score:.3f}" for task, score in scores.items())
            + f" | avg={average:.3f}"
        )


__all__ = [
    "API_BASE_URL",
    "API_KEY",
    "BENCHMARK",
    "MODEL_NAME",
    "LocalTrainedPolicy",
    "OpenAIPolicy",
    "SUCCESS_SCORE_THRESHOLD",
    "SYSTEM_PROMPT",
    "build_chat_text",
    "StepTrace",
    "build_prompt",
    "clamp_allocation",
    "heuristic_allocation",
    "log_end",
    "log_start",
    "log_step",
    "main",
    "parse_json",
    "run_episode",
]


if __name__ == "__main__":
    main()
