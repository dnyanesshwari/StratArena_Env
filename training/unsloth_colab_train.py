from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.llm_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_SEQ_LENGTH,
    build_chat_text,
    build_grpo_records,
    build_sft_records,
    extract_allocation,
    grpo_env_reward,
    grpo_format_reward,
    load_rollout_rows,
    sample_rollout_rows,
    simulated_env_reward,
    summarize_predictions,
    SYSTEM_PROMPT,
)


def _require_unsloth_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Training requires unsloth, transformers, datasets, trl, and torch. "
            "Use the original Colab stack or install the same packages locally."
        ) from exc
    return torch, Dataset, TrainerCallback, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer, FastLanguageModel


def _generate_predictions(
    model: Any,
    tokenizer: Any,
    sample_rows: list[dict[str, Any]],
    max_seq_length: int,
    label: str,
    max_new_tokens: int = 80,
) -> list[dict[str, Any]]:
    import json

    torch, *_rest = _require_unsloth_stack()

    results: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in sample_rows:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ]
            text = build_chat_text(tokenizer, prompt_messages)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            predicted_alloc = extract_allocation(generated)
            reference_alloc = extract_allocation(row["completion"])
            obs = json.loads(row["prompt"])
            reward = simulated_env_reward(predicted_alloc or 0.0, obs)
            results.append(
                {
                    "model": label,
                    "task": row["task"],
                    "step": row["step"],
                    "predicted_alloc": predicted_alloc,
                    "reference_alloc": reference_alloc,
                    "reference_reward": row["reward"],
                    "simulated_reward": reward,
                    "format_ok": predicted_alloc is not None,
                    "alloc_error": (
                        abs(predicted_alloc - reference_alloc)
                        if predicted_alloc is not None and reference_alloc is not None
                        else None
                    ),
                    "raw_output": generated,
                }
            )
    return results


def _load_model_for_inference(model_name: str, max_seq_length: int) -> tuple[Any, Any]:
    *_prefix, FastLanguageModel = _require_unsloth_stack()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def train_sft(
    dataset_path: str,
    base_model_name: str,
    output_dir: str,
    max_seq_length: int,
    epochs: float,
    learning_rate: float,
) -> str:
    (
        _torch,
        Dataset,
        TrainerCallback,
        _GRPOConfig,
        _GRPOTrainer,
        SFTConfig,
        SFTTrainer,
        FastLanguageModel,
    ) = _require_unsloth_stack()

    rows = load_rollout_rows(dataset_path)
    sft_records = build_sft_records(rows)
    sft_dataset = Dataset.from_list(sft_records)

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    sft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    class SftLogCapture(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[dict[str, float]] = []

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if logs and "loss" in logs:
                self.history.append({"step": float(state.global_step), "loss": float(logs["loss"])})

    callback = SftLogCapture()

    def formatting_func(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        output_texts: list[str] = []
        for prompt_messages, completion_text in zip(examples["prompt"], examples["completion"]):
            chat_prompt = build_chat_text(base_tokenizer, prompt_messages)
            output_texts.append(chat_prompt + completion_text)
        return {"text": output_texts}

    sft_dataset = sft_dataset.map(
        formatting_func,
        batched=True,
        remove_columns=sft_dataset.column_names,
    )

    trainer = SFTTrainer(
        model=sft_model,
        tokenizer=base_tokenizer,
        train_dataset=sft_dataset,
        dataset_text_field="text",
        callbacks=[callback],
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            logging_steps=5,
            save_steps=100,
            fp16=True,
            bf16=False,
            optim="adamw_8bit",
            report_to="none",
            completion_only_loss=False,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    base_tokenizer.save_pretrained(output_dir)
    return output_dir


def train_grpo(
    dataset_path: str,
    init_model_name: str,
    output_dir: str,
    max_seq_length: int,
    max_steps: int,
    learning_rate: float,
) -> str:
    (
        _torch,
        Dataset,
        TrainerCallback,
        GRPOConfig,
        GRPOTrainer,
        _SFTConfig,
        _SFTTrainer,
        FastLanguageModel,
    ) = _require_unsloth_stack()

    rows = load_rollout_rows(dataset_path)
    grpo_records = build_grpo_records(rows)
    grpo_dataset = Dataset.from_list(grpo_records)

    grpo_model, grpo_tokenizer = FastLanguageModel.from_pretrained(
        model_name=init_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    class GrpoLogCapture(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[dict[str, float]] = []

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            entry: dict[str, float] = {"step": float(state.global_step)}
            for key in ("reward", "reward/grpo_format_reward", "reward/grpo_env_reward", "loss", "kl"):
                if key in logs:
                    entry[key] = float(logs[key])
            if len(entry) > 1:
                self.history.append(entry)

    callback = GrpoLogCapture()

    trainer = GRPOTrainer(
        model=grpo_model,
        tokenizer=grpo_tokenizer,
        reward_funcs=[grpo_format_reward, grpo_env_reward],
        train_dataset=grpo_dataset,
        callbacks=[callback],
        args=GRPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=max_seq_length,
            max_completion_length=96,
            max_steps=max_steps,
            logging_steps=5,
            save_steps=50,
            report_to="none",
            bf16=False,
            fp16=True,
            use_vllm=False,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    grpo_tokenizer.save_pretrained(output_dir)
    return output_dir


def evaluate_models(
    dataset_path: str,
    base_model_name: str,
    sft_model_name: str | None,
    grpo_model_name: str | None,
    max_seq_length: int,
    max_samples: int,
) -> None:
    rows = load_rollout_rows(dataset_path)
    sample_rows = sample_rollout_rows(rows, max_samples=max_samples)

    model_specs = [("Untrained", base_model_name)]
    if sft_model_name:
        model_specs.append(("SFT", sft_model_name))
    if grpo_model_name:
        model_specs.append(("GRPO", grpo_model_name))

    for label, model_name in model_specs:
        model, tokenizer = _load_model_for_inference(model_name, max_seq_length)
        predictions = _generate_predictions(model, tokenizer, sample_rows, max_seq_length, label)
        summary = summarize_predictions(predictions)
        print(
            f"{label:10s} | format={summary['format_rate']:.1%} | "
            f"avg_reward={summary['avg_reward']:.4f} | "
            f"avg_alloc_error={summary['avg_alloc_error']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo-native SFT/GRPO trainer for StratArena rollouts")
    parser.add_argument("--dataset", default="outputs/stratarena_rollouts.jsonl")
    parser.add_argument("--mode", choices=["sft", "grpo", "all", "eval"], default="all")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-output", default="outputs/stratarena_sft")
    parser.add_argument("--grpo-output", default="outputs/stratarena_grpo")
    parser.add_argument("--grpo-init-model", default=None)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--eval-samples", type=int, default=60)
    parser.add_argument("--sft-epochs", type=float, default=1.0)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-4)
    parser.add_argument("--grpo-max-steps", type=int, default=300)
    parser.add_argument("--grpo-learning-rate", type=float, default=5e-6)
    args = parser.parse_args()

    sft_model_name = args.sft_output if Path(args.sft_output).exists() else None
    grpo_model_name = args.grpo_output if Path(args.grpo_output).exists() else None

    if args.mode in {"sft", "all"}:
        sft_model_name = train_sft(
            dataset_path=args.dataset,
            base_model_name=args.base_model,
            output_dir=args.sft_output,
            max_seq_length=args.max_seq_length,
            epochs=args.sft_epochs,
            learning_rate=args.sft_learning_rate,
        )
        print(f"SFT complete: {sft_model_name}")

    if args.mode in {"grpo", "all"}:
        init_model_name = args.grpo_init_model or sft_model_name or args.sft_output
        grpo_model_name = train_grpo(
            dataset_path=args.dataset,
            init_model_name=init_model_name,
            output_dir=args.grpo_output,
            max_seq_length=args.max_seq_length,
            max_steps=args.grpo_max_steps,
            learning_rate=args.grpo_learning_rate,
        )
        print(f"GRPO complete: {grpo_model_name}")

    if args.mode in {"eval", "all"}:
        evaluate_models(
            dataset_path=args.dataset,
            base_model_name=args.base_model,
            sft_model_name=sft_model_name,
            grpo_model_name=grpo_model_name,
            max_seq_length=args.max_seq_length,
            max_samples=args.eval_samples,
        )


if __name__ == "__main__":
    main()
