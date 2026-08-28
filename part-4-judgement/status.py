#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import EVAL_POINTS, RUN_DIR
from paths import QUEUE_DIR


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def selected_steps(run_dir: Path) -> int:
    steps = sorted(int(p.name.split("-")[-1]) for p in run_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit())
    return min(len(steps), EVAL_POINTS)


def expected(limit: int = 1000) -> dict:
    runs = [p for p in RUN_DIR.glob("qwen3.5-9b__*") if (p / "run_summary.json").exists()]
    steps = sum(selected_steps(p) for p in runs)
    return {"runs": len(runs), "steps": steps + 1, "responses": (steps + 1) * limit, "judge_calls": (steps + 1) * limit * 2}


def main() -> None:
    response_files = [p for p in (QUEUE_DIR / "responses").glob("*/step_*/chunk_*.jsonl")]
    judgment_files = [p for p in (QUEUE_DIR / "judgments").glob("*/step_*/chunk_*.jsonl") if not p.name.endswith("__raw.jsonl") and not p.name.endswith("__failed.jsonl")]
    raw_files = [p for p in (QUEUE_DIR / "judgments").glob("*/step_*/chunk_*__raw.jsonl")]
    failed_files = [p for p in (QUEUE_DIR / "judgments").glob("*/step_*/chunk_*__failed.jsonl")]
    data = {
        "expected": expected(),
        "response_files": len(response_files),
        "response_rows": sum(line_count(p) for p in response_files),
        "judgment_files": len(judgment_files),
        "judgment_rows": sum(line_count(p) for p in judgment_files),
        "raw_files": len(raw_files),
        "raw_rows": sum(line_count(p) for p in raw_files),
        "failed_files": len(failed_files),
        "failed_calls": sum(line_count(p) for p in failed_files),
        "gen_done": (QUEUE_DIR / "gen.done").exists(),
    }
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
