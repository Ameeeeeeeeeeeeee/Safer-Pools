#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.api import Client
from common.config import GLOBAL_API_WORKERS, JUDGE_ALIASES, resolve_api_model
from common.io import append_jsonl, ensure_dir, read_jsonl, write_jsonl, write_text
from prompt import judge_messages
from paths import QUEUE_DIR, ensure_dirs

JUDGE_ATTEMPTS = 5
JUDGE_MAX_TOKENS = 1024
FAIL_RATE_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--idle-stop", type=int, default=120)
    return p.parse_args()


def pending_markers() -> list[Path]:
    response_root = QUEUE_DIR / "responses"
    if not response_root.exists():
        return []
    rows = []
    for marker in response_root.glob("*/step_*/chunk_*.ready"):
        rel = marker.relative_to(response_root)
        out_marker = QUEUE_DIR / "judgments" / rel
        if not out_marker.exists():
            rows.append(marker)
    return sorted(rows)


def score_value(payload: dict, key: str) -> int:
    value = payload[key]
    if isinstance(value, int) and not isinstance(value, bool):
        score = value
    else:
        match = re.search(r"([1-7])\s*分?", str(value))
        if not match:
            raise ValueError(f"{key} is not a 1-7 score: {value!r}")
        score = int(match.group(1))
    if score < 1 or score > 7:
        raise ValueError(f"{key} out of range: {value}")
    return score


def text_value(payload: dict, *keys: str) -> str:
    for key in keys:
        if key in payload:
            return str(payload[key])
    raise KeyError(keys[0])


def score_alias(row: dict, alias: str, client: VolcClient) -> dict:
    tag = f"vllm_step:{row['run_id']}:{row['step']}:{row['question_id']}:{alias}"
    last_error: Exception | None = None
    last_model = resolve_api_model(alias)
    for attempt in range(JUDGE_ATTEMPTS):
        payload: dict | None = None
        model_name = resolve_api_model(alias)
        try:
            payload, result = client.chat_json(judge_messages(row), model=alias, max_tokens=JUDGE_MAX_TOKENS, tag=tag)
            model_name = result.model
            supportiveness_score = score_value(payload, "supportiveness_score")
            social_risk_score = score_value(payload, "social_risk_score")
            return {
                **row,
                "judge_alias": alias,
                "judge_model": model_name,
                "supportiveness_score": supportiveness_score,
                "social_risk_score": social_risk_score,
                "supportiveness_rationale": text_value(payload, "supportiveness_rationale"),
                "social_risk_rationale": text_value(payload, "social_risk_rationale"),
                "judge_prompt_tokens": result.prompt_tokens,
                "judge_completion_tokens": result.completion_tokens,
            }
        except Exception as exc:
            last_error = exc
            last_model = model_name
            fail = {"ts": time.time(), "phase": client.phase, "tag": tag, "model": model_name, "attempt": attempt + 1, "error": repr(exc)}
            if payload is not None:
                fail["payload"] = payload
            append_jsonl(client.fail_path, fail)
            if attempt < JUDGE_ATTEMPTS - 1:
                time.sleep(1.0)
    return {
        **row,
        "judge_alias": alias,
        "judge_model": last_model,
        "judge_failed": True,
        "judge_error": repr(last_error),
    }


def average(raw: list[dict]) -> list[dict]:
    by_key: dict[str, list[dict]] = {}
    for row in raw:
        if row.get("judge_failed"):
            continue
        by_key.setdefault(row["question_id"], []).append(row)
    rows = []
    for qid, group in sorted(by_key.items()):
        if {row["judge_alias"] for row in group} != set(JUDGE_ALIASES):
            continue
        base_keys = ["run_id", "step", "checkpoint_path", "profile_id", "question_id", "question_text", "prompt_messages", "answer_text", "generation_model", "generation_prompt_tokens", "generation_completion_tokens"]
        base = {key: group[0][key] for key in base_keys}
        row = {**base, "judge_alias": "avg", "supportiveness_score": round(statistics.fmean(item["supportiveness_score"] for item in group), 4), "social_risk_score": round(statistics.fmean(item["social_risk_score"] for item in group), 4)}
        for item in group:
            alias = item["judge_alias"]
            row[f"{alias}_supportiveness_score"] = item["supportiveness_score"]
            row[f"{alias}_social_risk_score"] = item["social_risk_score"]
        rows.append(row)
    return rows


def process(marker: Path, client: VolcClient) -> None:
    chunk_path = marker.with_suffix(".jsonl")
    rows = read_jsonl(chunk_path)
    raw: list[dict] = []
    jobs = [(row, alias) for row in rows for alias in JUDGE_ALIASES]
    workers = min(GLOBAL_API_WORKERS, len(jobs)) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_alias, row, alias, client) for row, alias in jobs]
        for future in concurrent.futures.as_completed(futures):
            raw.append(future.result())
    fails = [row for row in raw if row.get("judge_failed")]
    fail_limit = max(8, int(len(jobs) * FAIL_RATE_LIMIT))
    if len(fails) > fail_limit:
        raise RuntimeError(f"too many judge failures: {len(fails)}/{len(jobs)} in {chunk_path}")
    rel = marker.relative_to(QUEUE_DIR / "responses")
    out_dir = QUEUE_DIR / "judgments" / rel.parent
    ensure_dir(out_dir)
    out_path = out_dir / chunk_path.name
    write_jsonl(out_path.with_name(out_path.stem + "__raw.jsonl"), sorted(raw, key=lambda row: (row["question_id"], row["judge_alias"])))
    if fails:
        write_jsonl(out_path.with_name(out_path.stem + "__failed.jsonl"), sorted(fails, key=lambda row: (row["question_id"], row["judge_alias"])))
    avg = average(raw)
    write_jsonl(out_path, avg)
    write_text(out_path.with_suffix(".ready"), "ready\n")
    print(f"judged {chunk_path} rows={len(avg)} failed_calls={len(fails)}", flush=True)


def generation_done() -> bool:
    return (QUEUE_DIR / "gen.done").exists()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    client = Client("part4_vllm_judge")
    idle = 0
    while True:
        markers = pending_markers()
        if markers:
            idle = 0
            for marker in markers:
                process(marker, client)
        elif args.once or (generation_done() and idle >= args.idle_stop):
            break
        else:
            time.sleep(2)
            idle += 2


if __name__ == "__main__":
    main()
