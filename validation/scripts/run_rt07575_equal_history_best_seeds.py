#!/usr/bin/env python3
"""Run two exact-history RT07575 best-profile GPU seeds with dose and LET."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def replace_one(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return updated


def render_config(
    template: Path,
    output: Path,
    run_dir: Path,
    spots: Path,
    histories: int,
    seed: int,
) -> None:
    text = template.read_text(encoding="utf-8")
    for key, value in (
        ("number_of_histories", str(histories)),
        ("random_seed", str(seed)),
        ("topas_spots_file", str(spots)),
        ("voxel_dose_mhd_output_file", str(run_dir / "dose.mhd")),
        ("let_voxel_mhd_output_file", str(run_dir / "letd")),
    ):
        text = replace_one(text, key, value)
    output.write_text(text, encoding="utf-8")


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def number(pattern: str) -> float:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            raise ValueError(f"{path}: missing {pattern}")
        return float(match.group(1))

    def integer(pattern: str) -> int:
        return int(round(number(pattern)))

    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    return {
        "histories": integer(r"^Histories:\s*(\d+)$"),
        "elapsed_seconds": number(r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": number(
            r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"
        ),
        "energy_balance_error": number(
            r"^Energy balance error:\s*([0-9.eE+-]+)$"
        ),
        "secondary_queue_overflow": integer(
            r"^Secondary queue overflow:\s*(\d+)$"
        ),
        "cascade_queue_overflow": integer(
            r"^Cascade queue overflow:\s*(\d+)$"
        ),
        "backend": backend.group(1) if backend else "unknown",
    }


def complete(run_dir: Path, histories: int) -> dict[str, object] | None:
    required = (
        run_dir / "dose.mhd",
        run_dir / "dose.raw",
        run_dir / "letd_primary_c12.mhd",
        run_dir / "letd_primary_c12.raw",
        run_dir / "letd_all_hadron.mhd",
        run_dir / "letd_all_hadron.raw",
        run_dir / "run.log",
    )
    if not all(path.exists() for path in required):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != histories:
        return None
    if stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]:
        return None
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc")
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"),
    )
    parser.add_argument(
        "--spots",
        type=Path,
        default=Path("ct/fullplan_result/RT07575/spots_full_plan.txt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/ct/RT07575/equal_history_best_seeds"),
    )
    parser.add_argument("--histories", type=int, default=12_963_817)
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260801, 20260802))
    args = parser.parse_args()

    for name in ("binary", "template", "spots", "output_root"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.repo_root / value)

    report: dict[str, object] = {
        "profile": "best",
        "template": str(args.template),
        "spots": str(args.spots),
        "histories_per_seed": args.histories,
        "runs": {},
    }
    runs = report["runs"]
    assert isinstance(runs, dict)
    for seed in args.seeds:
        run_dir = args.output_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stats = complete(run_dir, args.histories)
        if stats is None:
            config = run_dir / "config.yaml"
            render_config(
                args.template,
                config,
                run_dir,
                args.spots,
                args.histories,
                seed,
            )
            print(f"RUN seed {seed}", flush=True)
            with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    [str(args.binary), "--config", str(config), "--device", "cuda"],
                    cwd=args.repo_root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"seed {seed} failed with {completed.returncode}; "
                    f"see {run_dir / 'run.log'}"
                )
            stats = complete(run_dir, args.histories)
            if stats is None:
                raise RuntimeError(f"seed {seed} output is incomplete or overflowed")
        else:
            print(f"SKIP complete seed {seed}", flush=True)
        print(
            f"DONE seed {seed}: {stats['elapsed_seconds']:.2f} s, "
            f"{stats['throughput_histories_per_second']:.0f} histories/s",
            flush=True,
        )
        runs[str(seed)] = stats

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
