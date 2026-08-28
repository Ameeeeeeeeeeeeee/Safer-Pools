#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api import VolcClient
from common.config import GLOBAL_API_WORKERS, JUDGE_ALIASES, JUDGE_DIR, QA_DIR, ensure_dirs, resolve_api_model
from common.io import read_jsonl, write_jsonl
from common.prompts import as_user, judge_prompt

JUDGE_PART_DIR = JUDGE_DIR / "raw_parts"
REQUIRED_FIELDS = ["supportiveness_score", "social_risk_score", "supportiveness_rationale", "social_risk_rationale"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def safe_name(answer_id: str, judge_alias: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{answer_id}__{judge_alias}")
    return f"{name}.jsonl"


def judge_path(row: dict, judge_alias: str) -> Path:
    return JUDGE_PART_DIR / safe_name(row["answer_id"], judge_alias)


def score_value(value) -> int:
    if isinstance(value, int):
        score = value
    else:
        match = re.search(r"([1-7])\s*分?", str(value))
        if not match:
            raise RuntimeError(f"bad judge score: {value}")
        score = int(match.group(1))
    if not 1 <= score <= 7:
        raise RuntimeError(f"score out of range: {value}")
    return score


def check_payload(payload: dict) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise RuntimeError("judge payload is not object")
    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    if missing:
        raise RuntimeError(f"judge payload missing fields: {missing}")
    return score_value(payload["supportiveness_score"]), score_value(payload["social_risk_score"])


def build_score(row: dict, judge_alias: str, payload: dict, result) -> dict:
    supportiveness_score, social_risk_score = check_payload(payload)
    return {
        **row,
        "judge_alias": judge_alias,
        "judge_model": resolve_api_model(judge_alias),
        "supportiveness_score": supportiveness_score,
        "social_risk_score": social_risk_score,
        "supportiveness_rationale": str(payload["supportiveness_rationale"]),
        "social_risk_rationale": str(payload["social_risk_rationale"]),
        "judge_prompt_tokens": result.prompt_tokens,
        "judge_completion_tokens": result.completion_tokens,
    }


def score_one(row: dict, judge_alias: str, client: VolcClient, force: bool = False) -> dict:
    path = judge_path(row, judge_alias)
    if path.exists() and not force:
        cached = read_jsonl(path)
        if len(cached) == 1:
            return cached[0]
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            payload, result = client.chat_json(as_user(judge_prompt(row["question_text"], row["answer_text"])), model=judge_alias, max_tokens=1024, tag=f"judge:{row['answer_id']}:{judge_alias}:try{attempt}")
            item = build_score(row, judge_alias, payload, result)
        except Exception as exc:
            last_error = exc
            continue
        write_jsonl(path, [item])
        return item
    raise RuntimeError(f"{row['answer_id']}/{judge_alias} judge validation failed after retries: {last_error}")


def average_rows(raw: list[dict]) -> list[dict]:
    by_answer: dict[str, list[dict]] = {}
    for row in raw:
        by_answer.setdefault(row["answer_id"], []).append(row)
    rows = []
    for answer_id, group in sorted(by_answer.items()):
        if {row["judge_alias"] for row in group} != set(JUDGE_ALIASES):
            raise RuntimeError(f"missing judges for {answer_id}")
        base_keys = [
            "profile_id", "question_id", "question_index", "question_model", "question_model_name",
            "question_variant", "question_variant_title", "topic_id", "topic_title", "question_text",
            "answer_id", "answer_alias", "answer_model", "answer_variant", "answer_variant_title", "answer_group", "answer_text",
        ]
        base = {key: group[0].get(key) for key in base_keys}
        row = {
            **base,
            "judge_alias": "avg",
            "supportiveness_score": round(statistics.fmean(item["supportiveness_score"] for item in group), 4),
            "social_risk_score": round(statistics.fmean(item["social_risk_score"] for item in group), 4),
        }
        for item in group:
            alias = item["judge_alias"]
            row[f"{alias}_supportiveness_score"] = item["supportiveness_score"]
            row[f"{alias}_social_risk_score"] = item["social_risk_score"]
            row[f"{alias}_supportiveness_rationale"] = item["supportiveness_rationale"]
            row[f"{alias}_social_risk_rationale"] = item["social_risk_rationale"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    ensure_dirs()
    qa_rows = [row for row in read_jsonl(QA_DIR / "answers.jsonl") if int(row["profile_id"]) <= args.profile_count]
    jobs = [(row, alias) for row in qa_rows for alias in JUDGE_ALIASES]
    client = VolcClient("part2_judge")
    raw: list[dict] = []
    workers = min(args.workers, GLOBAL_API_WORKERS, len(jobs)) or 1
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(score_one, row, alias, client, args.force): (row, alias) for row, alias in jobs}
        for future in concurrent.futures.as_completed(future_map):
            raw.append(future.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"judge {done}/{len(jobs)}", flush=True)
    raw.sort(key=lambda row: (row["answer_id"], row["judge_alias"]))
    avg = average_rows(raw)
    write_jsonl(JUDGE_DIR / "raw.jsonl", raw)
    write_jsonl(JUDGE_DIR / "avg.jsonl", avg)
    write_jsonl(JUDGE_DIR / "supportiveness.jsonl", [{**row, "score": row["supportiveness_score"]} for row in avg])
    write_jsonl(JUDGE_DIR / "social_risk.jsonl", [{**row, "score": row["social_risk_score"]} for row in avg])
    print(f"judge avg rows {len(avg)}", flush=True)


if __name__ == "__main__":
    main()
