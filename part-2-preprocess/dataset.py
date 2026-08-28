#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import (
    DATASET_DIR,
    DATASET_SCENARIO,
    DATASET_VARIANTS,
    JUDGE_DIR,
    SPLIT_RATIOS,
    TARGETS,
    dataset_path,
    ensure_dirs,
    rm_path,
)
from common.io import read_jsonl, write_json, write_jsonl
from common.prompts import answer_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1500)
    parser.add_argument("--scenario", default=DATASET_SCENARIO)
    parser.add_argument("--variants", nargs="+", default=DATASET_VARIANTS)
    return parser.parse_args()


def split_profiles(profile_ids: list[str]) -> dict[str, str]:
    ordered = sorted(profile_ids, key=lambda value: int(value))
    n = len(ordered)
    train_n = int(n * SPLIT_RATIOS["train"])
    val_n = int(n * SPLIT_RATIOS["val"])
    lookup = {}
    for idx, pid in enumerate(ordered):
        if idx < train_n:
            split = "train"
        elif idx < train_n + val_n:
            split = "val"
        else:
            split = "test"
        lookup[pid] = split
    return lookup


def by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["answer_variant"], row["answer_alias"]))


def rank_pair(target: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    ordered = by_variant(rows)
    if target == "support_maximizing":
        ranked = sorted(ordered, key=lambda row: (-row["supportiveness_score"], row["social_risk_score"], row["answer_variant"], row["answer_alias"]))
        return ranked[0], ranked[-1], "support_maximizing"
    if target == "risk_minimizing":
        ranked = sorted(ordered, key=lambda row: (row["social_risk_score"], -row["supportiveness_score"], row["answer_variant"], row["answer_alias"]))
        return ranked[0], ranked[-1], "risk_minimizing"
    if target == "risk_bounded_support":
        safe = [row for row in ordered if row["social_risk_score"] <= 3]
        if safe:
            winner = sorted(safe, key=lambda row: (-row["supportiveness_score"], row["social_risk_score"], row["answer_variant"], row["answer_alias"]))[0]
            rest = [row for row in ordered if row["answer_id"] != winner["answer_id"]]
            loser = sorted(rest, key=lambda row: (-row["social_risk_score"], row["supportiveness_score"], row["answer_variant"], row["answer_alias"]))[0]
            return winner, loser, "risk_bounded_support_safe"
        ranked = sorted(ordered, key=lambda row: (row["social_risk_score"], -row["supportiveness_score"], row["answer_variant"], row["answer_alias"]))
        return ranked[0], ranked[-1], "risk_bounded_support_fallback"
    raise KeyError(target)


def meta(scenario: str, variants: list[str], target: str, rule: str, winner: dict[str, Any], loser: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "candidate_variants": variants,
        "target": target,
        "winner_rule": rule,
        "winner_id": winner["answer_id"],
        "winner_variant": winner["answer_variant"],
        "winner_group": winner.get("answer_group"),
        "winner_alias": winner["answer_alias"],
        "winner_supportiveness": winner["supportiveness_score"],
        "winner_social_risk": winner["social_risk_score"],
        "loser_id": loser["answer_id"],
        "loser_variant": loser["answer_variant"],
        "loser_group": loser.get("answer_group"),
        "loser_alias": loser["answer_alias"],
        "loser_supportiveness": loser["supportiveness_score"],
        "loser_social_risk": loser["social_risk_score"],
        "supportiveness_gap": winner["supportiveness_score"] - loser["supportiveness_score"],
        "social_risk_gap": winner["social_risk_score"] - loser["social_risk_score"],
    }


def rm_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["answer_id"],
        "profile_id": row["profile_id"],
        "question_id": row["question_id"],
        "question_variant": row.get("question_variant"),
        "answer_variant": row.get("answer_variant"),
        "answer_group": row.get("answer_group"),
        "answer_alias": row["answer_alias"],
        "question_text": row["question_text"],
        "prompt_messages": answer_messages(row["question_text"]),
        "answer_text": row["answer_text"],
        "supportiveness_score": row["supportiveness_score"],
        "social_risk_score": row["social_risk_score"],
        "deepseek_supportiveness_score": row["deepseek_supportiveness_score"],
        "deepseek_social_risk_score": row["deepseek_social_risk_score"],
        "doubao_supportiveness_score": row["doubao_supportiveness_score"],
        "doubao_social_risk_score": row["doubao_social_risk_score"],
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.fmean(float(row[key]) for row in rows), 4) if rows else 0.0


def target_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "winner_supportiveness": mean(rows, "winner_supportiveness"),
        "winner_social_risk": mean(rows, "winner_social_risk"),
        "loser_supportiveness": mean(rows, "loser_supportiveness"),
        "loser_social_risk": mean(rows, "loser_social_risk"),
        "winner_variant": dict(Counter(row["winner_variant"] for row in rows)),
        "loser_variant": dict(Counter(row["loser_variant"] for row in rows)),
    }


def write_report(path: Path, stats: dict[str, Any]) -> None:
    lines = ["# Dataset Selection", ""]
    lines.append(f"scenario：`{stats['scenario']}`。")
    lines.append(f"candidate variants：{json.dumps(stats['variants'], ensure_ascii=False)}。")
    lines.append(f"profiles：{stats['profiles']}；questions：{stats['questions']}；candidate answers：{stats['answers']}。")
    lines.append("")
    lines.append("| target | train | val | test | winner supportiveness | winner social risk | loser supportiveness | loser social risk | winner variant | loser variant |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for target in TARGETS:
        row = stats["targets"][target]
        split = row["split_rows"]
        desc = row["selection"]
        lines.append(
            f"| `{target}` | {split['train']} | {split['val']} | {split['test']} | "
            f"{desc['winner_supportiveness']} | {desc['winner_social_risk']} | {desc['loser_supportiveness']} | {desc['loser_social_risk']} | "
            f"{json.dumps(desc['winner_variant'], ensure_ascii=False)} | {json.dumps(desc['loser_variant'], ensure_ascii=False)} |"
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rows = [
        row for row in read_jsonl(JUDGE_DIR / "avg.jsonl")
        if int(row["profile_id"]) <= args.profile_count and row.get("answer_variant") in args.variants
    ]
    profiles = sorted({row["profile_id"] for row in rows}, key=lambda value: int(value))
    split_lookup = split_profiles(profiles)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    expected = len(args.variants) * 2
    for qid, group in by_question.items():
        if len(group) != expected:
            raise RuntimeError(f"{args.scenario}/{qid} has {len(group)} candidates, expected {expected}")

    rm_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        rm_rows[split_lookup[row["profile_id"]]].append(rm_item(row))

    stats: dict[str, Any] = {
        "scenario": args.scenario,
        "variants": args.variants,
        "profiles": len(profiles),
        "questions": len(by_question),
        "answers": len(rows),
        "profile_split_counts": {split: list(split_lookup.values()).count(split) for split in ["train", "val", "test"]},
        "targets": {},
    }

    for target in TARGETS:
        sft = {"train": [], "val": [], "test": []}
        dpo = {"train": [], "val": [], "test": []}
        kto = {"train": [], "val": [], "test": []}
        selections = []
        for qid, group in sorted(by_question.items()):
            winner, loser, rule = rank_pair(target, group)
            split = split_lookup[winner["profile_id"]]
            prompt = answer_messages(winner["question_text"])
            info = meta(args.scenario, args.variants, target, rule, winner, loser)
            selections.append({"question_id": qid, "profile_id": winner["profile_id"], "split": split, **info})
            base = {
                "profile_id": winner["profile_id"],
                "question_id": qid,
                "question_variant": winner.get("question_variant"),
                "target": target,
                "prompt_messages": prompt,
                "prompt_text": winner["question_text"],
                "metadata": info,
            }
            sft[split].append({
                **base,
                "id": f"{qid}::{target}::sft",
                "completion": winner["answer_text"],
                "answer_alias": winner["answer_alias"],
                "answer_variant": winner["answer_variant"],
            })
            dpo[split].append({
                **base,
                "id": f"{qid}::{target}::dpo",
                "chosen": winner["answer_text"],
                "rejected": loser["answer_text"],
                "chosen_messages": [{"role": "assistant", "content": winner["answer_text"]}],
                "rejected_messages": [{"role": "assistant", "content": loser["answer_text"]}],
            })
            kto[split].append({
                **base,
                "id": f"{qid}::{target}::kto::pos",
                "completion": winner["answer_text"],
                "completion_messages": [{"role": "assistant", "content": winner["answer_text"]}],
                "label": True,
            })
            kto[split].append({
                **base,
                "id": f"{qid}::{target}::kto::neg",
                "completion": loser["answer_text"],
                "completion_messages": [{"role": "assistant", "content": loser["answer_text"]}],
                "label": False,
            })
        for kind, bucket in [("sft", sft), ("dpo", dpo), ("kto", kto)]:
            for split, items in bucket.items():
                write_jsonl(dataset_path(kind, target, split), items)
        write_jsonl(DATASET_DIR / "selection" / args.scenario / target / "rows.jsonl", selections)
        stats["targets"][target] = {
            "split_rows": {split: len(sft[split]) for split in ["train", "val", "test"]},
            "kto_split_rows": {split: len(kto[split]) for split in ["train", "val", "test"]},
            "selection": target_stats(selections),
        }

    for split, items in rm_rows.items():
        write_jsonl(rm_path(split), items)
    write_json(DATASET_DIR / "stats.json", stats)
    write_report(DATASET_DIR / "selection" / args.scenario / "summary.md", stats)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
