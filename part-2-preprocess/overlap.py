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

from common.config import JUDGE_DIR, STATS_DIR, TARGETS, ensure_dirs
from common.io import read_jsonl, write_json, write_jsonl

SCENARIOS = {
    "engaged_guarded": ["engaged", "guarded"],
    "engaged_neutral": ["engaged", "neutral"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1500)
    return parser.parse_args()


def choose(target: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    ordered = sorted(rows, key=lambda row: (row["answer_variant"], row["answer_alias"]))
    if target == "support_maximizing":
        return sorted(ordered, key=lambda row: (-row["supportiveness_score"], row["social_risk_score"], row["answer_variant"], row["answer_alias"]))[0], "support_maximizing"
    if target == "risk_minimizing":
        return sorted(ordered, key=lambda row: (row["social_risk_score"], -row["supportiveness_score"], row["answer_variant"], row["answer_alias"]))[0], "risk_minimizing"
    if target == "risk_bounded_support":
        safe = [row for row in ordered if row["social_risk_score"] <= 3]
        if safe:
            return sorted(safe, key=lambda row: (-row["supportiveness_score"], row["social_risk_score"], row["answer_variant"], row["answer_alias"]))[0], "risk_bounded_support_safe"
        return sorted(ordered, key=lambda row: (row["social_risk_score"], -row["supportiveness_score"], row["answer_variant"], row["answer_alias"]))[0], "risk_bounded_support_fallback"
    raise KeyError(target)


def desc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supportiveness_vals = [float(row["supportiveness_score"]) for row in rows]
    social_risk_vals = [float(row["social_risk_score"]) for row in rows]
    return {
        "n": len(rows),
        "supportiveness_mean": round(statistics.fmean(supportiveness_vals), 4) if rows else 0,
        "social_risk_mean": round(statistics.fmean(social_risk_vals), 4) if rows else 0,
        "variant": dict(Counter(row["answer_variant"] for row in rows)),
        "alias": dict(Counter(row["answer_alias"] for row in rows)),
    }


def overlap_rate(a: list[dict[str, Any]], b: list[dict[str, Any]], key: str) -> float:
    lookup = {row["question_id"]: row for row in b}
    same = sum(1 for row in a if lookup[row["question_id"]][key] == row[key])
    return round(same / len(a), 6) if a else 0.0


def scenario_rows(rows: list[dict[str, Any]], name: str, variants: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["answer_variant"] in variants:
            by_question[row["question_id"]].append(row)
    expected = len(variants) * 2
    for qid, group in by_question.items():
        if len(group) != expected:
            raise RuntimeError(f"{name}/{qid} has {len(group)} candidates, expected {expected}")
    chosen_rows = []
    chosen_by_target: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for qid, group in sorted(by_question.items()):
        for target in TARGETS:
            winner, rule = choose(target, group)
            item = {
                "scenario": name,
                "target": target,
                "rule": rule,
                "profile_id": winner["profile_id"],
                "question_id": qid,
                "question_variant": winner.get("question_variant"),
                "answer_id": winner["answer_id"],
                "answer_variant": winner["answer_variant"],
                "answer_alias": winner["answer_alias"],
                "supportiveness_score": winner["supportiveness_score"],
                "social_risk_score": winner["social_risk_score"],
            }
            chosen_rows.append(item)
            chosen_by_target[target].append(item)
    pairwise = {}
    for idx, left in enumerate(TARGETS):
        for right in TARGETS[idx + 1:]:
            pairwise[f"{left}__{right}"] = {
                "same_answer": overlap_rate(chosen_by_target[left], chosen_by_target[right], "answer_id"),
                "same_variant": overlap_rate(chosen_by_target[left], chosen_by_target[right], "answer_variant"),
            }
    all_same_answer = 0
    all_same_variant = 0
    target_lookup = {target: {row["question_id"]: row for row in rows} for target, rows in chosen_by_target.items()}
    for qid in by_question:
        picks = {target: target_lookup[target][qid] for target in TARGETS}
        if len({row["answer_id"] for row in picks.values()}) == 1:
            all_same_answer += 1
        if len({row["answer_variant"] for row in picks.values()}) == 1:
            all_same_variant += 1
    summary = {
        "scenario": name,
        "variants": variants,
        "questions": len(by_question),
        "targets": {target: desc(items) for target, items in chosen_by_target.items()},
        "pairwise_overlap": pairwise,
        "all_three_same_answer": round(all_same_answer / len(by_question), 6),
        "all_three_same_variant": round(all_same_variant / len(by_question), 6),
    }
    return chosen_rows, summary


def write_report(summary: dict[str, Any]) -> None:
    lines = ["# Formal Selection Overlap", ""]
    lines.append("只分析 Part2 judge 后的选择重叠，不生成 SFT/DPO/KTO 数据集。")
    for name, item in summary["scenarios"].items():
        lines.append("")
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"候选 answer_variant：{', '.join(item['variants'])}。question 数：{item['questions']}。")
        lines.append("")
        lines.append("### target 分布")
        lines.append("")
        lines.append("| target | n | supportiveness | social risk | variant | alias |")
        lines.append("|---|---:|---:|---:|---|---|")
        for target in TARGETS:
            row = item["targets"][target]
            lines.append(f"| `{target}` | {row['n']} | {row['supportiveness_mean']} | {row['social_risk_mean']} | {json.dumps(row['variant'], ensure_ascii=False)} | {json.dumps(row['alias'], ensure_ascii=False)} |")
        lines.append("")
        lines.append("### overlap")
        lines.append("")
        lines.append("| pair | same answer | same variant |")
        lines.append("|---|---:|---:|")
        for pair, values in item["pairwise_overlap"].items():
            lines.append(f"| `{pair}` | {values['same_answer']} | {values['same_variant']} |")
        lines.append(f"\nall three same answer：{item['all_three_same_answer']}；all three same variant：{item['all_three_same_variant']}。")
    (STATS_DIR / "selection_overlap.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rows = [row for row in read_jsonl(JUDGE_DIR / "avg.jsonl") if int(row["profile_id"]) <= args.profile_count]
    chosen = []
    scenarios = {}
    for name, variants in SCENARIOS.items():
        items, summary = scenario_rows(rows, name, variants)
        chosen.extend(items)
        scenarios[name] = summary
    write_jsonl(STATS_DIR / "selection_overlap_rows.jsonl", chosen)
    summary = {"profiles": args.profile_count, "answer_rows": len(rows), "scenarios": scenarios}
    write_json(STATS_DIR / "selection_overlap.json", summary)
    write_report(summary)
    print(json.dumps({"profiles": args.profile_count, "answer_rows": len(rows), "scenarios": scenarios}, ensure_ascii=False)[:4000], flush=True)


if __name__ == "__main__":
    main()
