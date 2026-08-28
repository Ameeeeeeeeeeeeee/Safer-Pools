#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api import VolcClient
from common.config import GLOBAL_API_WORKERS, MINIBIO_DIR, QUESTION_ALIASES, QUESTION_DIR, QUESTIONS_PER_PROFILE, ensure_dirs, resolve_api_model
from common.io import read_jsonl, read_text, write_jsonl
from common.prompts import FORMAL_QUESTION_VARIANTS, as_user, formal_question_prompt

HAN = re.compile(r"[\u4e00-\u9fff]")
BAD_QUESTION = re.compile(r'```|//|JSON|输出格式|实际输出|本注释|注释：|基于人物摘要|严格遵循|仅包含|^\s*[{}]|[{}]\s*$')
QUESTION_PART_DIR = QUESTION_DIR / "parts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1500)
    parser.add_argument("--questions-per-profile", type=int, default=QUESTIONS_PER_PROFILE)
    parser.add_argument("--questions-per-model", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def q_model(index: int) -> str:
    return QUESTION_ALIASES[(index - 1) % len(QUESTION_ALIASES)]


def question_path(pid: str) -> Path:
    return QUESTION_PART_DIR / f"{pid}.jsonl"


def check_question(text: str) -> None:
    if not text or not HAN.search(text) or BAD_QUESTION.search(text):
        raise RuntimeError(f"bad question: {text[:160]}")


def normalize_items(payload: Any, variants: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise RuntimeError("question payload is not object")
    items = payload.get("questions")
    if not isinstance(items, list):
        raise RuntimeError("questions is not list")
    if len(items) != len(variants):
        raise RuntimeError(f"question count {len(items)} != {len(variants)}")
    rows = []
    for item, variant in zip(items, variants):
        if isinstance(item, str):
            vid = variant["id"]
            text = item.strip()
        elif isinstance(item, dict):
            vid = str(item.get("variant_id", "")).strip()
            text = str(item.get("question", "")).strip()
        else:
            raise RuntimeError(f"bad question item: {item}")
        if vid and vid != variant["id"]:
            vid = variant["id"]
        check_question(text)
        rows.append({"variant_id": variant["id"], "question": text})
    if len({row["question"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate questions")
    return rows


def build_rows(pid: str, alias: str, variants: list[dict[str, Any]], items: list[dict[str, str]], result) -> list[dict[str, Any]]:
    rows = []
    for idx, (variant, item) in enumerate(zip(variants, items), start=1):
        rows.append({
            "profile_id": pid,
            "question_id": f"{pid}-q{idx}",
            "question_index": idx,
            "question_model": alias,
            "question_model_name": resolve_api_model(alias),
            "question_variant": variant["id"],
            "question_variant_title": variant["title"],
            "topic_id": variant["id"],
            "topic_title": variant["title"],
            "question_text": item["question"],
            "question_prompt_tokens": result.prompt_tokens,
            "question_completion_tokens": result.completion_tokens,
        })
    return rows


def gen_one(index: int, variants: list[dict[str, Any]], client: VolcClient, force: bool) -> list[dict[str, Any]]:
    pid = f"{index:04d}"
    path = question_path(pid)
    if path.exists() and not force:
        rows = read_jsonl(path)
        if len(rows) == len(variants):
            for row in rows:
                check_question(row["question_text"])
            return rows
    alias = q_model(index)
    summary = read_text(MINIBIO_DIR / f"{pid}.txt")
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            payload, result = client.chat_json(as_user(formal_question_prompt(summary, variants, pid)), model=alias, max_tokens=3072, tag=f"question:{pid}:{alias}:try{attempt}")
            items = normalize_items(payload, variants)
            rows = build_rows(pid, alias, variants, items, result)
        except Exception as exc:
            last_error = exc
            continue
        write_jsonl(path, rows)
        return rows
    raise RuntimeError(f"{pid}/{alias} question validation failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    variants = FORMAL_QUESTION_VARIANTS[: args.questions_per_profile]
    if len(variants) != args.questions_per_profile:
        raise RuntimeError(f"questions_per_profile {args.questions_per_profile} > available variants {len(FORMAL_QUESTION_VARIANTS)}")
    client = VolcClient("part1_question")
    rows: list[dict[str, Any]] = []
    jobs = list(range(1, args.profile_count + 1))
    workers = min(GLOBAL_API_WORKERS, len(jobs)) or 1
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(gen_one, idx, variants, client, args.force) for idx in jobs]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"questions {done}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (row["profile_id"], row["question_index"]))
    write_jsonl(QUESTION_DIR / "questions.jsonl", rows)
    print(f"question rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
