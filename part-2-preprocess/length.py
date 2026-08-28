#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART3 = ROOT / "part-3-training"
if str(PART3) not in sys.path:
    sys.path.insert(0, str(PART3))

from common.config import DATASET_DIR, MAX_SEQ_LENGTH, PRIMARY_MODEL
from common.io import read_jsonl, write_json
from train_util import chat_len, load_tokenizer


def summarize(vals: list[int]) -> dict:
    vals = sorted(vals)
    def q(p: int) -> int | None:
        if not vals:
            return None
        return vals[round((len(vals) - 1) * p / 100)]
    return {
        "n": len(vals), "min": vals[0] if vals else None, "p50": q(50), "p75": q(75), "p90": q(90),
        "p95": q(95), "p99": q(99), "max": vals[-1] if vals else None,
        "mean": round(statistics.fmean(vals), 2) if vals else None,
        f">{MAX_SEQ_LENGTH}": sum(v > MAX_SEQ_LENGTH for v in vals),
    }


def main() -> None:
    tok = load_tokenizer(PRIMARY_MODEL)
    rows = []
    for path in sorted(DATASET_DIR.glob("*/*/*.jsonl")):
        kind = path.parts[-3]
        vals = []
        for row in read_jsonl(path):
            prompt = row.get("prompt_messages") or []
            if kind == "sft":
                vals.append(chat_len(tok, prompt + [{"role": "assistant", "content": row["completion"]}]))
            elif kind == "dpo":
                vals.append(max(chat_len(tok, prompt + row["chosen_messages"]), chat_len(tok, prompt + row["rejected_messages"])))
            elif kind == "kto":
                vals.append(chat_len(tok, prompt + row["completion_messages"]))
            elif kind == "rm":
                vals.append(chat_len(tok, prompt))
        info = {"path": str(path), **summarize(vals)}
        if info[f">{MAX_SEQ_LENGTH}"]:
            raise RuntimeError(f"length overflow: {info}")
        rows.append(info)
    write_json(DATASET_DIR / "length_stats.json", {"rows": rows, "max_seq_length": MAX_SEQ_LENGTH})
    print(json.dumps({"rows": rows[:3], "count": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
