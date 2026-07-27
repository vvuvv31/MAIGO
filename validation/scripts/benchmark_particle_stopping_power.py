#!/usr/bin/env python3
"""Interleaved timing of C-12 scaling and isotope-specific stopping tables."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path


ELAPSED = re.compile(r"(?m)^Elapsed:\s+([0-9.eE+-]+)\s+s$")
HISTORIES = re.compile(r"(?m)^Histories:\s+([0-9]+)$")
THROUGHPUT = re.compile(
    r"(?m)^Throughput:\s+([0-9.eE+-]+)\s+histories/s$"
)


def set_mode(text: str, enabled: bool) -> str:
    pattern = re.compile(
        r"(?m)^use_particle_specific_stopping_power:\s*(?:true|false)\s*$"
    )
    result, count = pattern.subn(
        f"use_particle_specific_stopping_power: {str(enabled).lower()}", text
    )
    if count != 1:
        raise RuntimeError(
            "Base config needs exactly one use_particle_specific_stopping_power key"
        )
    return result


def replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}:\s*.*$")
    result, count = pattern.subn(f"{key}: {value}", text)
    if count != 1:
        raise RuntimeError(f"Base config needs exactly one {key} key")
    return result


def run(binary: Path, config: Path, log: Path) -> dict[str, float | int | str]:
    completed = subprocess.run(
        [str(binary), "--config", str(config), "--device", "cuda"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(completed.stdout)
    elapsed = ELAPSED.search(completed.stdout)
    histories = HISTORIES.search(completed.stdout)
    throughput = THROUGHPUT.search(completed.stdout)
    if elapsed is None or histories is None or throughput is None:
        raise RuntimeError(f"Could not parse timing from {log}")
    return {
        "elapsed_seconds": float(elapsed.group(1)),
        "histories": int(histories.group(1)),
        "histories_per_second": float(throughput.group(1)),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", type=Path, default=Path("build/oneapi-release/carbon_mc")
    )
    parser.add_argument(
        "--base-config", type=Path, default=Path("config/beam_sobp_water_letd.yaml")
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cascade-package", type=Path, default=None)
    parser.add_argument("--condition-cascade-depth", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/letd_sobp/particle_table_benchmark/interleaved"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_config.read_text()
    if args.cascade_package is not None:
        base = replace_scalar(
            base, "cascade_package_file", str(args.cascade_package)
        )
    base = replace_scalar(
        base,
        "cascade_condition_on_reference_depth",
        str(args.condition_cascade_depth).lower(),
    )
    configs = {}
    for name, enabled in (("approx", False), ("exact", True)):
        path = args.output_dir / f"{name}.yaml"
        path.write_text(set_mode(base, enabled))
        configs[name] = path

    runs: dict[str, list[dict[str, float | int | str]]] = {
        "approx": [],
        "exact": [],
    }
    # Alternate AB/BA to suppress temperature and background-load ordering bias.
    for repetition in range(args.repetitions):
        order = ("approx", "exact") if repetition % 2 == 0 else ("exact", "approx")
        for mode in order:
            result = run(
                args.binary,
                configs[mode],
                args.output_dir / f"{mode}_{repetition}.log",
            )
            runs[mode].append(result)
            print(mode, repetition, result)

    medians = {
        mode: statistics.median(
            float(result["histories_per_second"]) for result in results
        )
        for mode, results in runs.items()
    }
    report = {
        "base_config": str(args.base_config),
        "repetitions": args.repetitions,
        "runs": runs,
        "median_histories_per_second": medians,
        "exact_throughput_change_percent": 100.0
        * (medians["exact"] / medians["approx"] - 1.0),
        "exact_runtime_change_percent": 100.0
        * (medians["approx"] / medians["exact"] - 1.0),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
