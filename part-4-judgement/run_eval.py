#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import PRIMARY_MODEL, RUN_DIR
from common.io import write_text
from paths import LOG_DIR, QUEUE_DIR, STAT_DIR, ensure_dirs

PART = Path(__file__).resolve().parent
VLLM_PY = "/root/miniconda3/envs/trl-vllm/bin/python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=["support_maximizing", "risk_minimizing", "risk_bounded_support"])
    p.add_argument("--algorithms", nargs="+", default=["sft", "dpo_b005", "dpo_b03", "kto_b03"])
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--step-limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-base", action="store_true")
    return p.parse_args()


def run_dirs(args: argparse.Namespace) -> list[Path]:
    rows = []
    for path in sorted(RUN_DIR.glob("*")):
        if not path.is_dir() or not (path / "run_summary.json").exists():
            continue
        if not any(f"__{algo}__" in path.name for algo in args.algorithms):
            continue
        if not any(path.name.endswith(f"__{target}") for target in args.targets):
            continue
        rows.append(path)
    return rows


def gpu_list(args: argparse.Namespace) -> list[str]:
    values = [item.strip() for item in args.gpus.split(",") if item.strip()]
    return values or ["0"]


def gen_cmd(args: argparse.Namespace, run_dir: Path | None) -> tuple[str, list[str]]:
    cmd = [VLLM_PY, str(PART / "gen_server.py"), "--model", args.model, "--limit", str(args.limit), "--max-new-tokens", str(args.max_new_tokens), "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", str(args.max_num_batched_tokens), "--max-model-len", str(args.max_model_len)]
    if args.step_limit:
        cmd.extend(["--step-limit", str(args.step_limit)])
    if args.force:
        cmd.append("--force")
    if run_dir is None:
        cmd.append("--base")
        return f"base__{args.model}", cmd
    cmd.extend(["--run-dir", str(run_dir)])
    return run_dir.name, cmd


def open_log(name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return (LOG_DIR / f"{name}.log").open("ab")


def start_gen(name: str, cmd: list[str], gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    env.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    env["PYTHONUNBUFFERED"] = "1"
    handle = open_log("gen_" + name.replace("/", "_"))
    print(f"start gen gpu={gpu} name={name}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    return proc, handle


def start_judge():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    handle = open_log("judge")
    cmd = [sys.executable, str(PART / "judge_server.py"), "--idle-stop", "120"]
    print(f"start judge", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    return proc, handle


def run_generation(args: argparse.Namespace, judge_ref: list) -> None:
    gpus = gpu_list(args)
    jobs = [] if args.no_base else [gen_cmd(args, None)]
    jobs.extend(gen_cmd(args, run_dir) for run_dir in run_dirs(args))
    available = list(gpus)
    active = []
    next_job = 0
    judge_errors = 0
    print(f"gen workers={len(gpus)} jobs={len(jobs)}", flush=True)
    while next_job < len(jobs) or active:
        judge_code = judge_ref[0].poll()
        if judge_code is not None:
            judge_ref[1].close()
            if judge_code != 0:
                judge_errors += 1
            if judge_errors > 3:
                for proc, _, _, handle in active:
                    proc.terminate()
                    handle.close()
                raise RuntimeError(f"judge_server exited too often, last code={judge_code}")
            print(f"restart judge after exit code={judge_code}", flush=True)
            judge_ref[:] = list(start_judge())
        while available and next_job < len(jobs):
            gpu = available.pop(0)
            name, cmd = jobs[next_job]
            next_job += 1
            proc, handle = start_gen(name, cmd, gpu)
            active.append((proc, gpu, cmd, handle))
        time.sleep(5)
        keep = []
        for proc, gpu, cmd, handle in active:
            code = proc.poll()
            if code is None:
                keep.append((proc, gpu, cmd, handle))
                continue
            handle.close()
            available.append(gpu)
            if code != 0:
                for other, _, _, other_handle in keep:
                    other.terminate()
                    other_handle.close()
                raise subprocess.CalledProcessError(code, cmd)
        active = keep


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.force:
        for path in [QUEUE_DIR, STAT_DIR]:
            if path.exists():
                shutil.rmtree(path)
        ensure_dirs()
    gen_done = QUEUE_DIR / "gen.done"
    if gen_done.exists():
        gen_done.unlink()
    judge_ref = list(start_judge())
    try:
        run_generation(args, judge_ref)
        write_text(gen_done, "done\n")
        judge_code = judge_ref[0].wait()
        if judge_code != 0:
            raise subprocess.CalledProcessError(judge_code, [sys.executable, str(PART / "judge_server.py")])
    finally:
        if judge_ref[0].poll() is None:
            judge_ref[0].terminate()
        judge_ref[1].close()


if __name__ == "__main__":
    main()
