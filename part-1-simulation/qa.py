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
from common.config import ASSISTANT_ALIASES, GLOBAL_API_WORKERS, QA_DIR, QUESTION_DIR, ensure_dirs, resolve_api_model
from common.io import read_jsonl, write_jsonl
from common.prompts import FORMAL_ANSWER_VARIANTS, answer_messages

HAN = re.compile(r"[\u4e00-\u9fff]")
BAD_ANSWER = re.compile(r'```|"content"\s*:|\n\s*[{}]\s*\n')
ANSWER_PART_DIR = QA_DIR / "parts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def answer_path(row: dict[str, Any], variant: dict[str, Any], alias: str) -> Path:
    return ANSWER_PART_DIR / f"{row['question_id']}__{variant['id']}__{alias}.jsonl"


def check_answer(row: dict[str, Any], variant: dict[str, Any], alias: str, text: str) -> None:
    if not text:
        raise RuntimeError(f"empty answer {row['question_id']} {variant['id']} {alias}")
    if not HAN.search(text) or BAD_ANSWER.search(text):
        raise RuntimeError(f"malformed answer {row['question_id']} {variant['id']} {alias}: {text[:120]}")


def build_answer(row: dict[str, Any], variant: dict[str, Any], alias: str, result) -> dict[str, Any]:
    answer = result.text.strip()
    check_answer(row, variant, alias, answer)
    return {
        **row,
        "answer_id": f"{row['question_id']}::{variant['id']}::{alias}",
        "answer_alias": alias,
        "answer_model": resolve_api_model(alias),
        "answer_variant": variant["id"],
        "answer_variant_title": variant["title"],
        "answer_group": variant.get("group", variant["id"]),
        "answer_text": answer,
        "answer_prompt_tokens": result.prompt_tokens,
        "answer_completion_tokens": result.completion_tokens,
    }


def answer_one(row: dict[str, Any], variant: dict[str, Any], alias: str, client: VolcClient, force: bool = False) -> dict[str, Any]:
    path = answer_path(row, variant, alias)
    if path.exists() and not force:
        cached = read_jsonl(path)
        if len(cached) == 1:
            check_answer(row, variant, alias, cached[0]["answer_text"])
            return cached[0]
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            result = client.chat_text(answer_messages(row["question_text"], variant["id"]), model=alias, max_tokens=896, tag=f"qa:{row['question_id']}:{variant['id']}:{alias}:try{attempt}")
            answer = build_answer(row, variant, alias, result)
        except Exception as exc:
            last_error = exc
            continue
        write_jsonl(path, [answer])
        return answer
    raise RuntimeError(f"{row['question_id']}/{variant['id']}/{alias} answer validation failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    questions = [row for row in read_jsonl(QUESTION_DIR / "questions.jsonl") if int(row["profile_id"]) <= args.profile_count]
    jobs = [(row, variant, alias) for row in questions for variant in FORMAL_ANSWER_VARIANTS for alias in ASSISTANT_ALIASES]
    client = VolcClient("part1_qa")
    rows: list[dict[str, Any]] = []
    workers = min(GLOBAL_API_WORKERS, len(jobs)) or 1
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(answer_one, row, variant, alias, client, args.force): (row, variant, alias) for row, variant, alias in jobs}
        for future in concurrent.futures.as_completed(future_map):
            rows.append(future.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"qa {done}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (row["question_id"], row["answer_variant"], row["answer_alias"]))
    write_jsonl(QA_DIR / "answers.jsonl", rows)
    print(f"qa rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
