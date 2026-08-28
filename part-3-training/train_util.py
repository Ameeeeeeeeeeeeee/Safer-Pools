from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import (
    CHECKPOINT_KEEP,
    GLOBAL_BATCH,
    LORA_COMMON,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_PATHS,
    RUN_DIR,
    TRAIN_METRIC_DIR,
    TRAIN_PER_DEVICE_BATCH,
    TRAIN_SUMMARY_DIR,
    resolve_model_path,
)
from common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from common.prompts import as_user, judge_prompt


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def model_key(alias_or_path: str) -> str:
    if alias_or_path in MODEL_PATHS:
        return alias_or_path
    resolved = Path(alias_or_path).resolve()
    for key, path in MODEL_PATHS.items():
        if Path(path).resolve() == resolved:
            return key
    raise KeyError(alias_or_path)


def load_tokenizer(alias_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(alias_or_path), trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def ids_from(encoded) -> list[int]:
    ids = getattr(encoded, "input_ids", encoded)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def chat_len(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> int:
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    return len(ids_from(encoded))


def render_messages(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return "\n".join(f"{row['role']}: {row['content']}" for row in messages) + "\nassistant:"


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": str(row["role"]), "content": str(row["content"])} for row in messages]


def make_lora(alias_or_path: str, task_type: TaskType = TaskType.CAUSAL_LM) -> LoraConfig:
    key = model_key(alias_or_path)
    return LoraConfig(
        task_type=task_type,
        r=LORA_COMMON["r"],
        lora_alpha=LORA_COMMON["lora_alpha"],
        lora_dropout=LORA_COMMON["lora_dropout"],
        bias="none",
        target_modules=LORA_TARGET_MODULES[key],
    )


def model_kwargs() -> dict[str, Any]:
    return {"torch_dtype": torch.bfloat16, "trust_remote_code": True}


def train_defaults() -> dict[str, Any]:
    return {
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "logging_first_step": True,
        "report_to": "none",
        "save_only_model": False,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "dataset_num_proc": 1,
    }


def per_device(algorithm: str) -> int:
    return TRAIN_PER_DEVICE_BATCH[algorithm]


def grad_accum(algorithm: str, per_device_batch: int | None = None) -> int:
    batch = per_device_batch or per_device(algorithm)
    denom = max(1, world_size()) * batch
    global_batch = GLOBAL_BATCH[algorithm]
    if global_batch % denom != 0:
        raise RuntimeError(f"global batch {global_batch} not divisible by {denom}")
    return global_batch // denom


def total_steps(train_rows: int, algorithm: str, per_device_batch: int, epochs: float, max_steps: int) -> int:
    if max_steps and max_steps > 0:
        return max_steps
    effective = max(1, world_size()) * per_device_batch * grad_accum(algorithm, per_device_batch)
    return max(1, math.ceil(math.ceil(train_rows / effective) * epochs))


def save_interval(train_rows: int, algorithm: str, per_device_batch: int, epochs: float, max_steps: int) -> int:
    return max(1, math.ceil(total_steps(train_rows, algorithm, per_device_batch, epochs, max_steps) / CHECKPOINT_KEEP))


def checkpoint_step(path: Path | str) -> int:
    match = re.search(r"checkpoint-(\d+)$", str(path))
    return int(match.group(1)) if match else 0


def latest_checkpoint(output_dir: Path) -> str | None:
    return get_last_checkpoint(str(output_dir))


def resume_checkpoint(output_dir: Path) -> str | None:
    return latest_checkpoint(output_dir)


def keep_steps(steps: list[int], max_points: int = CHECKPOINT_KEEP) -> list[int]:
    ordered = sorted(set(step for step in steps if step > 0))
    if len(ordered) <= max_points:
        return ordered
    chosen = []
    for idx in range(max_points):
        pos = round(idx * (len(ordered) - 1) / (max_points - 1))
        value = ordered[pos]
        if not chosen or chosen[-1] != value:
            chosen.append(value)
    if chosen[-1] != ordered[-1]:
        chosen[-1] = ordered[-1]
    return chosen


def retain_checkpoints(output_dir: Path) -> list[int]:
    if not is_main():
        return []
    dirs = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step)
    keep = set(keep_steps([checkpoint_step(path) for path in dirs]))
    for path in dirs:
        if checkpoint_step(path) not in keep:
            shutil.rmtree(path, ignore_errors=True)
    return sorted(keep)


def build_sft(path: Path) -> Dataset:
    rows = []
    for row in read_jsonl(path):
        rows.append({
            "id": row["id"], "prompt": clean_messages(row["prompt_messages"]),
            "completion": [{"role": "assistant", "content": row["completion"]}],
            "metadata": row.get("metadata", {}), "chat_template_kwargs": {"enable_thinking": False},
        })
    return Dataset.from_list(rows)


def build_dpo(path: Path) -> Dataset:
    rows = []
    for row in read_jsonl(path):
        rows.append({
            "id": row["id"], "prompt": clean_messages(row["prompt_messages"]),
            "chosen": clean_messages(row["chosen_messages"]), "rejected": clean_messages(row["rejected_messages"]),
            "metadata": row.get("metadata", {}), "chat_template_kwargs": {"enable_thinking": False},
        })
    return Dataset.from_list(rows)


def build_kto(path: Path) -> Dataset:
    rows = []
    for row in read_jsonl(path):
        rows.append({
            "id": row["id"], "prompt": clean_messages(row["prompt_messages"]),
            "completion": clean_messages(row["completion_messages"]), "label": bool(row["label"]),
            "metadata": row.get("metadata", {}), "chat_template_kwargs": {"enable_thinking": False},
        })
    return Dataset.from_list(rows)


def build_rm(path: Path, tokenizer, max_length: int = MAX_SEQ_LENGTH) -> Dataset:
    rows = []
    for row in read_jsonl(path):
        text = render_messages(tokenizer, clean_messages(row["prompt_messages"]) + [{"role": "assistant", "content": row["answer_text"]}], add_generation_prompt=False)
        rows.append({"text": text, "labels": [float(row["supportiveness_score"]) / 7.0, float(row["social_risk_score"]) / 7.0]})
    return Dataset.from_list(rows)


def score_to_unit(score: float) -> float:
    return (score - 1.0) / 6.0


def build_judge_rm(path: Path, tokenizer, limit: int = 0) -> Dataset:
    rows = []
    source = read_jsonl(path)
    if limit > 0:
        source = source[:limit]
    for row in source:
        question = row.get("question_text") or row["prompt_messages"][-1]["content"]
        text = render_messages(tokenizer, as_user(judge_prompt(question, row["answer_text"])), add_generation_prompt=False)
        rows.append({"text": text, "labels": [score_to_unit(float(row["supportiveness_score"])), score_to_unit(float(row["social_risk_score"]))]})
    return Dataset.from_list(rows)


def persist_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    if not is_main():
        return
    ensure_dir(output_dir)
    write_json(output_dir / "run_summary.json", summary)
    ensure_dir(TRAIN_SUMMARY_DIR)
    write_json(TRAIN_SUMMARY_DIR / f"{summary['run_id']}.json", summary)


def persist_history(output_dir: Path, run_id: str, log_history: list[dict[str, Any]]) -> None:
    if not is_main():
        return
    write_jsonl(output_dir / "log_history.jsonl", log_history)
    ensure_dir(TRAIN_METRIC_DIR)
    with (TRAIN_METRIC_DIR / "loss.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "step", "epoch", "loss", "eval_loss", "learning_rate"])
        if handle.tell() == 0:
            writer.writeheader()
        for row in log_history:
            if "loss" in row or "eval_loss" in row:
                writer.writerow({
                    "run_id": run_id,
                    "step": row.get("step"),
                    "epoch": row.get("epoch"),
                    "loss": row.get("loss"),
                    "eval_loss": row.get("eval_loss"),
                    "learning_rate": row.get("learning_rate"),
                })


def adapter_needs_arch(checkpoint: str | Path | None) -> bool:
    if checkpoint is None:
        return False
    path = Path(checkpoint) / "adapter_model.safetensors"
    if not path.exists():
        return False
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(".language_model." in key for key in handle.keys())


def load_generation_model(model: str, checkpoint: str | None = None):
    model_path = resolve_model_path(model)
    loader = AutoModelForCausalLM
    if adapter_needs_arch(checkpoint):
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch_name = (getattr(cfg, "architectures", None) or [None])[0]
        loader = getattr(transformers, arch_name, AutoModelForCausalLM)
    loaded = loader.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    if checkpoint:
        loaded = PeftModel.from_pretrained(loaded, checkpoint)
    loaded.eval()
    return loaded
