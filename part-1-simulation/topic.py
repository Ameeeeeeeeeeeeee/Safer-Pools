#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api import VolcClient
from common.config import GLOBAL_API_WORKERS, MINIBIO_DIR, TOPIC_DIR, TOPICS, ensure_dirs
from common.io import read_jsonl, read_text, write_jsonl
from common.prompts import as_user, topic_prompt

TOPIC_BY_ID = {tid: {"topic_id": tid, "topic_title": title, "topic_desc": desc} for tid, title, desc in TOPICS}
REQUIRED_FIELDS = ["topic_id", "fit_score", "askability_score", "controversy_score", "why_fit"]
SCORE_PART_DIR = TOPIC_DIR / "score_parts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1000)
    parser.add_argument("--questions-per-model", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_scores(profile_id: str, payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{profile_id} topic payload is not object")
    rows = payload.get("topics")
    if not isinstance(rows, list):
        raise RuntimeError(f"{profile_id} topics is not list")
    if len(rows) != len(TOPICS):
        raise RuntimeError(f"{profile_id} topic count {len(rows)} != {len(TOPICS)}")
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"{profile_id} topic row {idx} is not object")
        missing = [key for key in REQUIRED_FIELDS if key not in row]
        if missing:
            raise RuntimeError(f"{profile_id} topic row {idx} missing fields: {missing}")
    seen = {row["topic_id"] for row in rows}
    expected = {tid for tid, _, _ in TOPICS}
    if seen != expected:
        raise RuntimeError(f"{profile_id} topic ids mismatch")
    out = []
    for row in rows:
        tid = row["topic_id"]
        out.append({
            "profile_id": profile_id,
            **TOPIC_BY_ID[tid],
            "fit_score": int(row["fit_score"]),
            "askability_score": int(row["askability_score"]),
            "controversy_score": int(row["controversy_score"]),
            "why_fit": str(row["why_fit"]),
        })
    return out


def score_path(pid: str) -> Path:
    return SCORE_PART_DIR / f"{pid}.jsonl"


def score_one(pid: str, client: VolcClient, force: bool) -> list[dict]:
    path = score_path(pid)
    if path.exists() and not force:
        rows = read_jsonl(path)
        if len(rows) == len(TOPICS):
            return rows
    summary = read_text(MINIBIO_DIR / f"{pid}.txt")
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            payload, _ = client.chat_json(as_user(topic_prompt(summary)), model="deepseek", max_tokens=3072, tag=f"topic:{pid}:try{attempt}")
            rows = check_scores(pid, payload)
        except Exception as exc:
            last_error = exc
            continue
        write_jsonl(path, rows)
        return rows
    raise RuntimeError(f"{pid} topic validation failed after retries: {last_error}")


def sample_slots(rows: list[dict], count: int) -> list[dict]:
    ranked = sorted(rows, key=lambda r: (-(r["fit_score"] * 2 + r["askability_score"] + r["controversy_score"]), r["topic_id"]))
    pool: list[dict] = []
    for row in ranked:
        combined = row["fit_score"] * 2 + row["askability_score"] + row["controversy_score"]
        pool.extend([row] * max(1, round(combined / 7)))
    if len(pool) < count * 2:
        pool = ranked * (count * 2)
    return pool


def build_samples(score_rows: list[dict], count: int) -> list[dict]:
    by_profile: dict[str, list[dict]] = {}
    for row in score_rows:
        by_profile.setdefault(row["profile_id"], []).append(row)
    samples = []
    for pid, rows in sorted(by_profile.items()):
        pool = sample_slots(rows, count)
        for alias, offset in [("doubao", 0), ("deepseek", count)]:
            for idx in range(count):
                topic = pool[(offset + idx) % len(pool)]
                samples.append({"profile_id": pid, "question_model": alias, "slot_index": idx + 1, **{k: topic[k] for k in ["topic_id", "topic_title", "topic_desc", "fit_score", "askability_score", "controversy_score"]}})
    return samples


def main() -> None:
    args = parse_args()
    ensure_dirs()
    score_path = TOPIC_DIR / "score.jsonl"
    sample_path = TOPIC_DIR / "sample.jsonl"
    client = VolcClient("part1_topic", default_model="deepseek")
    pids = [f"{idx:04d}" for idx in range(1, args.profile_count + 1)]
    all_rows: list[dict] = []
    workers = min(GLOBAL_API_WORKERS, len(pids)) or 1
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(score_one, pid, client, args.force): pid for pid in pids}
        for future in concurrent.futures.as_completed(future_map):
            all_rows.extend(future.result())
            done += 1
            if done % 50 == 0 or done == len(pids):
                print(f"topic scores {done}/{len(pids)}", flush=True)
    all_rows.sort(key=lambda row: (row["profile_id"], row["topic_id"]))
    write_jsonl(score_path, all_rows)
    write_jsonl(sample_path, build_samples(all_rows, args.questions_per_model))
    print(f"topic rows {len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
