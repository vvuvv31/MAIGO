#!/usr/bin/env python3
"""Run reproducible alternating transport A/B pairs without attaching a profiler."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--pairs", type=int, default=5)
    return p.parse_args()


def extract(text, pattern):
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"missing output field: {pattern}")
    return float(match.group(1))


def main():
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    binaries = {"control": args.control.resolve(), "candidate": args.candidate.resolve()}
    env = dict(os.environ)
    env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    env["LD_LIBRARY_PATH"] = "/home/wuwei/sycl_workspace/llvm/build/install/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["CARBON_RUNTIME_BREAKDOWN"] = "1"
    schedule = []
    for phase, count in (("warm", args.warmups), ("run", args.pairs)):
        for pair in range(count):
            variants = ("control", "candidate") if pair % 2 == 0 else ("candidate", "control")
            schedule.extend((phase, pair, variant) for variant in variants)
    rows = []
    for phase, pair, variant in schedule:
        run = args.output / f"{phase}_{pair:02d}_{variant}"
        run.mkdir()
        (run / "data").symlink_to(args.data.resolve(), target_is_directory=True)
        before = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits"], text=True).strip()
        started = time.perf_counter()
        with (run / "run.log").open("w") as log:
            subprocess.run([str(binaries[variant]), "--config", str(args.config.resolve()),
                            "--device", "cuda"], cwd=run, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        wall = time.perf_counter() - started
        after = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits"], text=True).strip()
        output = (run / "run.log").read_text()
        row = {
            "phase": phase, "pair": pair, "variant": variant, "wall_s": wall,
            "elapsed_s": extract(output, r"Elapsed: ([0-9.]+) s"),
            "primary_s": extract(output, r"Kernel time: primary=([0-9.]+) s"),
            "secondary_s": extract(output, r" secondary=([0-9.]+) s"),
            "temperature_clock_before": before, "temperature_clock_after": after,
        }
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
