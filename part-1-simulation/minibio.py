#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import MINIBIO_DIR, PROFILE_DIR, ensure_dirs
from common.io import read_json, write_text
from common.prompts import minibio_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    for index in range(1, args.profile_count + 1):
        pid = f"{index:04d}"
        profile = read_json(PROFILE_DIR / f"{pid}.json")
        text = minibio_text(profile)
        write_text(MINIBIO_DIR / f"{pid}.txt", text)
    print(f"minibio {args.profile_count}", flush=True)


if __name__ == "__main__":
    main()
