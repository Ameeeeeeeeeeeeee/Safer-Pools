#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import EVAL_CHUNK_SIZE, EVAL_POINTS, PRIMARY_MODEL, STEP_MAX_NEW_TOKENS, dataset_path, resolve_model_path
from common.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text
from paths import QUEUE_DIR, ensure_dirs
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--base", action="store_true")
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--step-limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=STEP_MAX_NEW_TOKENS)
    p.add_argument("--chunk-size", type=int, default=EVAL_CHUNK_SIZE)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def checkpoint_step(path: Path | str) -> int:
    match = re.search(r"checkpoint-(\d+)$", str(path))
    return int(match.group(1)) if match else 0


def render_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{row['role']}: {row['content']}" for row in messages) + "\nassistant:"


def prompt_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_prompts(limit: int) -> list[dict]:
    path = dataset_path("sft", "risk_bounded_support", "test")
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"no test prompts at {path}")
    return rows if limit <= 0 else rows[:limit]


def select_steps(run_dir: Path, step_limit: int) -> list[tuple[int, Path]]:
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    steps = [checkpoint_step(path) for path in ckpts]
    if len(steps) <= EVAL_POINTS:
        chosen = steps
    else:
        first = steps[:20]
        rest = steps[20:]
        chosen_rest = []
        for idx in range(15):
            pos = round(idx * (len(rest) - 1) / 14)
            chosen_rest.append(rest[pos])
        chosen = sorted(set(first + chosen_rest))
    if step_limit > 0:
        chosen = chosen[:step_limit]
    return [(step, run_dir / f"checkpoint-{step}") for step in chosen]


def run_id_from(args: argparse.Namespace) -> tuple[str, str]:
    if args.base:
        return f"base__{args.model}", args.model
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --base")
    summary = read_json(args.run_dir / "run_summary.json")
    return summary["run_id"], summary.get("model", args.model)


def write_chunks(run_id: str, step: int, rows: list[dict], chunk_size: int) -> None:
    base = QUEUE_DIR / "responses" / run_id / f"step_{step:06d}"
    ensure_dir(base)
    for chunk_idx, start in enumerate(range(0, len(rows), chunk_size)):
        chunk = rows[start:start + chunk_size]
        path = base / f"chunk_{chunk_idx:05d}.jsonl"
        write_jsonl(path, chunk)
        write_text(path.with_suffix(".ready"), "ready\n")


def done_path(run_id: str, step: int) -> Path:
    return QUEUE_DIR / "responses" / run_id / f"step_{step:06d}" / "step.done"


def build_llm(model_path: str, use_lora: bool, args: argparse.Namespace) -> LLM:
    return LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_lora=use_lora,
        max_loras=1,
        max_lora_rank=args.max_lora_rank,
        disable_log_stats=True,
    )


def generate_step(llm: LLM, tokenizer, prompts: list[dict], sampling: SamplingParams, run_id: str, model_name: str, step: int, checkpoint: Path | None, lora_idx: int, args: argparse.Namespace) -> None:
    done = done_path(run_id, step)
    if done.exists() and not args.force:
        print(f"skip generated {run_id} step {step}", flush=True)
        return
    texts = [render_messages(tokenizer, row["prompt_messages"]) for row in prompts]
    prompt_tokens = [prompt_len(tokenizer, text) for text in texts]
    lora_request = None if checkpoint is None else LoRARequest(f"{run_id}_{step}", lora_idx, str(checkpoint))
    start = time.time()
    outputs = llm.generate(texts, sampling, lora_request=lora_request, use_tqdm=False)
    gen_seconds = time.time() - start
    rows = []
    for source, output, tokens in zip(prompts, outputs, prompt_tokens):
        completion = output.outputs[0]
        rows.append({
            "run_id": run_id,
            "step": step,
            "checkpoint_path": None if checkpoint is None else str(checkpoint),
            "generation_model": model_name,
            "profile_id": source["profile_id"],
            "question_id": source["question_id"],
            "target": source.get("target"),
            "question_text": source["prompt_text"],
            "prompt_messages": source["prompt_messages"],
            "answer_text": completion.text.strip(),
            "generation_prompt_tokens": int(tokens),
            "generation_completion_tokens": len(completion.token_ids),
        })
    write_chunks(run_id, step, rows, args.chunk_size)
    write_text(done, "done\n")
    write_json(done.with_suffix(".summary.json"), {
        "run_id": run_id,
        "step": step,
        "rows": len(rows),
        "gen_seconds": round(gen_seconds, 4),
        "rows_per_second": round(len(rows) / gen_seconds, 4) if gen_seconds else None,
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
    })
    print(f"generated {run_id} step {step} rows={len(rows)} seconds={gen_seconds:.2f}", flush=True)


def main() -> None:
    ensure_dirs()
    args = parse_args()
    run_id, model_name = run_id_from(args)
    if args.base:
        steps: list[tuple[int, Path | None]] = [(0, None)]
    else:
        steps = select_steps(args.run_dir, args.step_limit)
    if steps and not args.force and all(done_path(run_id, step).exists() for step, _ in steps):
        print(f"skip run generated {run_id} steps={len(steps)}", flush=True)
        return
    model_path = resolve_model_path(model_name)
    prompts = test_prompts(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    load_start = time.time()
    llm = build_llm(model_path, not args.base, args)
    print(f"loaded {run_id} seconds={time.time() - load_start:.2f} steps={len(steps)} prompts={len(prompts)}", flush=True)
    for idx, (step, checkpoint) in enumerate(steps, start=1):
        generate_step(llm, tokenizer, prompts, sampling, run_id, model_name, step, checkpoint, idx, args)


if __name__ == "__main__":
    main()
