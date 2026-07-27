#!/usr/bin/env python3
"""Run the same-version GPU LET_d model at 100, 200, 300 and 400 MeV/u."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ENERGIES = (100, 200, 300, 400)


def replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}:\s*.*$")
    replacement = f"{key}: {value}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {key!r} entry, found {count}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", type=Path, default=Path("config/beam_200MeVu_letd.yaml")
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("build/oneapi-release/carbon_mc")
    )
    parser.add_argument("--histories", type=int, default=100_000)
    parser.add_argument(
        "--particle-specific",
        action="store_true",
        help="Use isotope-specific G4 stopping-power tables for fragments.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/letd_energy_sweep/gpu")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_config.read_text()
    summaries = {}
    for energy in ENERGIES:
        run_dir = args.output_dir / f"e{energy}"
        run_dir.mkdir(parents=True, exist_ok=True)
        text = replace_scalar(base, "number_of_histories", str(args.histories))
        text = replace_scalar(text, "initial_energy_MeVu", f"{energy}.0")
        text = replace_scalar(text, "scorerLET", "true")
        text = replace_scalar(
            text,
            "use_particle_specific_stopping_power",
            str(args.particle_specific).lower(),
        )
        text = replace_scalar(
            text, "output_file", str(run_dir / "gpu_idd.csv")
        )
        text = replace_scalar(
            text, "let_output_file", str(run_dir / "gpu_letd.csv")
        )
        # Avoid CUDA auto secondary-queue caps truncating cascade products.
        queue_cap = max(2_500_000, int(args.histories) * 25)
        text = replace_scalar(
            text, "secondary_queue_capacity", str(queue_cap)
        )
        config_path = run_dir / "config.yaml"
        config_path.write_text(text)
        completed = subprocess.run(
            [str(args.binary), "--config", str(config_path), "--device", "cuda"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (run_dir / "run.log").write_text(completed.stdout)
        elapsed = re.search(r"(?m)^Elapsed:\s+([0-9.eE+-]+)\s+s$", completed.stdout)
        throughput = re.search(
            r"(?m)^Throughput:\s+([0-9.eE+-]+)\s+histories/s$",
            completed.stdout,
        )
        summaries[str(energy)] = {
            "histories": args.histories,
            "elapsed_seconds": float(elapsed.group(1)) if elapsed else None,
            "histories_per_second": (
                float(throughput.group(1)) if throughput else None
            ),
            "config": str(config_path),
            "let_csv": str(run_dir / "gpu_letd.csv"),
            "particle_specific_stopping_power": args.particle_specific,
        }
        print(f"{energy} MeV/u: {summaries[str(energy)]}")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
