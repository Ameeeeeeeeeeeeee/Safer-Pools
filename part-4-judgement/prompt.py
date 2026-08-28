from __future__ import annotations

from common.prompts import as_user, judge_prompt


def judge_messages(row: dict) -> list[dict[str, str]]:
    return as_user(judge_prompt(row["question_text"], row["answer_text"]))
