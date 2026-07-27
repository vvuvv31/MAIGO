#!/usr/bin/env python3
"""Run reproducible GPU scorerLET on/off timing measurements."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


ELAPSED_RE = re.compile(r"^Elapsed:\s+([0-9.eE+-]+)\s+s$", re.MULTILINE)
THROUGHPUT_RE = re.compile(
    r"^Throughput:\s+([0-9.eE+-]+)\s+histories/s$", re.MULTILINE
)


def run_once(
    executable: Path,
    config: Path,
    histories: int,
    enabled: bool,
    output_dir: Path,
    index: int,
) -> dict[str, object]:
    state = "on" if enabled else "off"
    command = [
        str(executable),
        "--config",
        str(config),
        "--device",
        "cuda",
        "--histories",
        str(histories),
        "--scorer-let" if enabled else "--no-scorer-let",
    ]
    if enabled:
        command.extend(
            ["--let-output", str(output_dir / f"gpu_letd_{state}_{index}.csv")]
        )
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    wall_seconds = time.perf_counter() - start
    log_path = output_dir / f"gpu_{state}_{index}.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    elapsed = ELAPSED_RE.search(completed.stdout)
    throughput = THROUGHPUT_RE.search(completed.stdout)
    if elapsed is None or throughput is None:
        raise RuntimeError(f"Could not parse GPU timing from {log_path}")
    return {
        "scorerLET": enabled,
        "transport_elapsed_seconds": float(elapsed.group(1)),
        "wall_seconds": wall_seconds,
        "histories_per_second": float(throughput.group(1)),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "on", "ab"), default="ab")
    parser.add_argument("--histories", type=int, default=100_000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-histories", type=int, default=10_000)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/oneapi-release/carbon_mc")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/beam_200MeVu_letd.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/letd_water/benchmark")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states = [False, True] if args.mode == "ab" else [args.mode == "on"]
    if args.warmup_histories > 0:
        for enabled in states:
            run_once(
                args.executable,
                args.config,
                args.warmup_histories,
                enabled,
                args.output_dir,
                -1,
            )

    runs: list[dict[str, object]] = []
    for index in range(args.repetitions):
        for enabled in states:
            runs.append(
                run_once(
                    args.executable,
                    args.config,
                    args.histories,
                    enabled,
                    args.output_dir,
                    index,
                )
            )

    summary: dict[str, object] = {
        "histories": args.histories,
        "repetitions": args.repetitions,
        "runs": runs,
    }
    if args.mode == "ab":
        off = [
            float(run["histories_per_second"])
            for run in runs
            if not bool(run["scorerLET"])
        ]
        on = [
            float(run["histories_per_second"])
            for run in runs
            if bool(run["scorerLET"])
        ]
        off_median = statistics.median(off)
        on_median = statistics.median(on)
        summary["median_off_histories_per_second"] = off_median
        summary["median_on_histories_per_second"] = on_median
        summary["throughput_change_percent"] = 100.0 * (on_median / off_median - 1.0)
        summary["runtime_increase_percent"] = 100.0 * (off_median / on_median - 1.0)

    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
