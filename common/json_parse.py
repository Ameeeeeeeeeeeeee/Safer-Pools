from __future__ import annotations

import json
import re
from typing import Any

STRUCT = {
    "\uff1a": ":",
    "\uff0c": ",",
    "\uff5b": "{",
    "\uff5d": "}",
    "\u3010": "[",
    "\u3011": "]",
}
DQUOTE = {"\u201c", "\u201d", "\u201e", "\u300c", "\u300d"}
TAIL_FIELD = re.compile(r'}\s*,\s*("[A-Za-z0-9_]+"\s*:)', re.S)


def strip_fence(text: str) -> str:
    value = text.strip().lstrip("\ufeff")
    value = re.sub(r"^```(?:\s*[A-Za-z0-9_-]+)?\s*", "", value, count=1)
    if value.endswith("```"):
        value = value[:-3].strip()
    return value


def strip_controls(text: str) -> str:
    return "".join(ch for ch in text if ch in "\n\r\t" or ord(ch) >= 32)


def next_real(text: str, pos: int) -> str:
    for idx in range(pos + 1, len(text)):
        if not text[idx].isspace():
            return text[idx]
    return ""


def normalize(text: str) -> str:
    text = strip_controls(text)
    chars: list[str] = []
    in_string = False
    escaped = False
    for pos, ch in enumerate(text):
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            chars.append(ch)
            escaped = True
            continue
        if ch == '"' or ch == "\uff02":
            chars.append('"')
            in_string = not in_string
            continue
        if ch in DQUOTE:
            if not in_string:
                chars.append('"')
                in_string = True
                continue
            if next_real(text, pos) in {":", ",", "]", "}"}:
                chars.append('"')
                in_string = False
                continue
            chars.append(ch)
            continue
        if not in_string:
            chars.append(STRUCT.get(ch, ch))
        else:
            chars.append(ch)
    return "".join(chars)


def fix_tail_field(text: str) -> str:
    return TAIL_FIELD.sub(r",\1", text)


def first_json(text: str) -> str:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        raise ValueError("no JSON object or list found")
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ValueError("unclosed JSON object or list")


def parse_json(text: str) -> Any:
    raw = strip_controls(strip_fence(text))
    repaired = fix_tail_field(raw)
    candidates = [raw]
    if repaired != raw:
        candidates.append(repaired)
    for item in list(candidates):
        norm = normalize(item)
        if norm != item:
            candidates.append(norm)
    for item in list(candidates):
        try:
            picked = first_json(item)
        except ValueError:
            continue
        if picked not in candidates:
            candidates.append(picked)
    errors = []
    for item in candidates:
        try:
            return json.loads(item)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("JSON parse failed: " + "; ".join(errors[:3]))
