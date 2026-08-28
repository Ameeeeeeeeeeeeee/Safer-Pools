#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART3 = ROOT / "part-3-training"
if str(PART3) not in sys.path:
    sys.path.insert(0, str(PART3))

from common.config import MAX_SEQ_LENGTH, dataset_path, resolve_model_path
from common.io import ensure_dir, read_jsonl
from train_util import build_sft, grad_accum, load_tokenizer, make_lora, model_kwargs, per_device, persist_history, persist_summary, resume_checkpoint, retain_checkpoints, save_interval, train_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    train = build_sft(dataset_path("sft", args.target, "train"))
    eval_ds = build_sft(dataset_path("sft", args.target, "val")) if read_jsonl(dataset_path("sft", args.target, "val")) else None
    batch = per_device("sft")
    save_steps = save_interval(len(train), "sft", batch, args.epochs, args.max_steps)
    cfg = SFTConfig(
        output_dir=str(args.output_dir), learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum("sft", batch), eval_strategy="epoch" if eval_ds else "no",
        save_strategy="steps", save_steps=save_steps, max_length=args.max_length,
        model_init_kwargs=model_kwargs(), **train_defaults()
    )
    cfg.completion_only_loss = True
    trainer = SFTTrainer(model=resolve_model_path(args.model), args=cfg, train_dataset=train, eval_dataset=eval_ds, processing_class=load_tokenizer(args.model), peft_config=make_lora(args.model))
    result = trainer.train(resume_from_checkpoint=resume_checkpoint(args.output_dir))
    history = [dict(row) for row in trainer.state.log_history]
    retained = retain_checkpoints(args.output_dir)
    summary = {"run_id": args.output_dir.name, "algorithm": "sft", "target": args.target, "model": args.model, "output_dir": str(args.output_dir), "train_rows": len(train), "eval_rows": 0 if eval_ds is None else len(eval_ds), "train_loss": result.metrics.get("train_loss"), "global_step": trainer.state.global_step, "epochs": args.epochs, "save_steps": save_steps, "retained_checkpoint_steps": retained}
    persist_history(args.output_dir, args.output_dir.name, history)
    persist_summary(args.output_dir, summary)


if __name__ == "__main__":
    main()
