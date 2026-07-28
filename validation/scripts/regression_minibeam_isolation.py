#!/usr/bin/env python3
"""Three-gate minibeam isolation regression.

Gate 1 — CARBON_ENABLE_MINIBEAM=OFF vs master (or reference binary):
  non-minibeam dose as close as FP32 atomics allow; history/step/reaction
  counts identical; kernel time not significantly slower.

Gate 2 — CARBON_ENABLE_MINIBEAM=ON + minibeam:false vs OFF binary:
  dose consistent; performance close.

Gate 3 — CARBON_ENABLE_MINIBEAM=ON + minibeam:true:
  optional smoke against a frozen dose CSV, or just that the case runs and
  prints backend/minibeam diagnostics.

Usage examples:

  # Compare two already-produced depth-dose CSVs
  python3 validation/scripts/regression_minibeam_isolation.py compare \\
      --ref out/reg/master_water.csv --test out/reg/off_water.csv

  # Full harness (expects binaries and a GPU). Builds are external.
  python3 validation/scripts/regression_minibeam_isolation.py run \\
      --master-bin build/master-nvidia/carbon_mc \\
      --off-bin build/oneapi-nvidia-release/carbon_mc \\
      --on-bin build/oneapi-nvidia-minibeam/carbon_mc \\
      --outdir out/reg_isolation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_depth_dose_csv(path: Path) -> Tuple[List[float], List[float]]:
    depths: List[float] = []
    doses: List[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(
            (row for row in handle if row.strip() and not row.lstrip().startswith("#"))
        )
        # Accept several common column names used in this repo.
        fieldnames = reader.fieldnames or []
        depth_key = next(
            (
                k
                for k in fieldnames
                if k.lower()
                in {
                    "depth_mm",
                    "z_mm",
                    "depth",
                    "z",
                }
            ),
            None,
        )
        dose_key = next(
            (
                k
                for k in fieldnames
                if k.lower()
                in {
                    "energy_deposition_mev",
                    "dose_gy",
                    "dose",
                    "deposited_mev",
                    "energy_mev",
                    "mev",
                }
            ),
            None,
        )
        if depth_key is None or dose_key is None:
            # Fallback: first two numeric columns
            handle.seek(0)
            raw = csv.reader(
                (
                    row
                    for row in handle
                    if row.strip() and not row.lstrip().startswith("#")
                )
            )
            header = next(raw, None)
            for row in raw:
                if len(row) < 2:
                    continue
                try:
                    depths.append(float(row[0]))
                    doses.append(float(row[1]))
                except ValueError:
                    continue
            if not depths:
                raise ValueError(f"Could not parse depth dose CSV: {path}")
            return depths, doses
        for row in reader:
            try:
                depths.append(float(row[depth_key]))
                doses.append(float(row[dose_key]))
            except (KeyError, ValueError):
                continue
    if not depths:
        raise ValueError(f"Empty depth dose CSV: {path}")
    return depths, doses


def compare_series(
    ref: List[float],
    test: List[float],
    *,
    label: str,
) -> Dict[str, float]:
    if len(ref) != len(test):
        raise ValueError(f"{label}: length mismatch {len(ref)} vs {len(test)}")
    ref_sum = sum(ref)
    test_sum = sum(test)
    abs_diff = [abs(a - b) for a, b in zip(ref, test)]
    peak = max(abs(x) for x in ref) if ref else 0.0
    l1 = sum(abs_diff)
    max_abs = max(abs_diff) if abs_diff else 0.0
    integral_diff_pct = (
        100.0 * (test_sum - ref_sum) / ref_sum if ref_sum != 0.0 else float("nan")
    )
    l1_pct = 100.0 * l1 / sum(abs(x) for x in ref) if ref_sum != 0.0 else float("nan")
    max_bin_frac = max_abs / peak if peak > 0.0 else float("nan")
    bitwise_equal = all(a == b for a, b in zip(ref, test))
    return {
        "integral_diff_pct": integral_diff_pct,
        "normalized_l1_pct": l1_pct,
        "max_bin_over_peak": max_bin_frac,
        "bitwise_equal": 1.0 if bitwise_equal else 0.0,
        "ref_integral": ref_sum,
        "test_integral": test_sum,
    }


def parse_run_log(text: str) -> Dict[str, float]:
    """Extract a few summary counters / timings from carbon_mc stdout."""
    patterns = {
        "nuclear_interactions": r"nuclear[_ ]interactions[=: ]+(\d+)",
        "total_steps": r"total[_ ]steps[=: ]+(\d+)",
        "transport_seconds": r"(?:transport|elapsed)[^=\d]*([0-9]*\.?[0-9]+)\s*s",
        "primary_kernel_seconds": r"primary[_ ]kernel[^=\d]*([0-9]*\.?[0-9]+)\s*s",
        "secondary_kernel_seconds": r"secondary[_ ]kernel[^=\d]*([0-9]*\.?[0-9]+)\s*s",
    }
    out: Dict[str, float] = {}
    for key, pat in patterns.items():
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            out[key] = float(match.group(1))
    return out


def run_case(
    binary: Path,
    config: Path,
    out_csv: Path,
    *,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, float]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Keep a local YAML copy so concurrent gates do not race shared configs.
    tmp_cfg = out_csv.with_suffix(".yaml")
    shutil.copyfile(config, tmp_cfg)
    cmd = [
        str(binary),
        "--config",
        str(tmp_cfg),
        "--device",
        "cuda",
    ]
    # Some minibeam configs score only voxel MHD and require empty depth-dose
    # paths when fragment species scoring is off. Only force --output when the
    # caller wants a depth CSV and the config allows it.
    text = tmp_cfg.read_text()
    allow_depth = True
    # Do not let \s match newlines or "output_file:" will swallow the next key.
    has_empty_output = bool(re.search(r"(?m)^output_file:[ \t]*$", text))
    has_nonempty_output = bool(re.search(r"(?m)^output_file:[ \t]*\S+", text))
    has_mhd = bool(re.search(r"(?m)^voxel_dose_mhd_output_file:[ \t]*\S+", text))
    species_off = bool(
        re.search(r"(?m)^enable_fragment_species_scoring:[ \t]*false[ \t]*$", text)
    )
    if species_off and has_mhd and has_empty_output and not has_nonempty_output:
        allow_depth = False
    if allow_depth:
        cmd.extend(["--output", str(out_csv)])
    if extra_args:
        cmd.extend(extra_args)
    run_env = dict(**{k: v for k, v in __import__("os").environ.items()})
    run_env.setdefault("ONEAPI_DEVICE_SELECTOR", "cuda:gpu")
    if env:
        run_env.update(env)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = out_csv.with_suffix(".log")
    log_path.write_text(log)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} failed ({proc.returncode}) for {config}:\n{log[-4000:]}"
        )
    if allow_depth and not out_csv.is_file():
        raise RuntimeError(f"Expected output CSV missing: {out_csv}\n{log[-2000:]}")
    if not allow_depth:
        # Smoke success: write a tiny marker CSV so downstream compare is optional.
        out_csv.write_text(
            "depth_mm,energy_deposition_MeV\n0.0,0.0\n"
        )
    return log, elapsed


def gate_compare(
    name: str,
    ref_csv: Path,
    test_csv: Path,
    *,
    max_integral_diff_pct: float,
    max_l1_pct: float,
    require_bitwise: bool,
    ref_log: str = "",
    test_log: str = "",
    max_time_ratio: Optional[float] = None,
    ref_elapsed: Optional[float] = None,
    test_elapsed: Optional[float] = None,
) -> Dict[str, object]:
    _, ref_dose = load_depth_dose_csv(ref_csv)
    _, test_dose = load_depth_dose_csv(test_csv)
    metrics = compare_series(ref_dose, test_dose, label=name)
    ref_stats = parse_run_log(ref_log) if ref_log else {}
    test_stats = parse_run_log(test_log) if test_log else {}
    count_ok = True
    count_notes = []
    for key in ("nuclear_interactions", "total_steps"):
        if key in ref_stats and key in test_stats:
            if ref_stats[key] != test_stats[key]:
                count_ok = False
                count_notes.append(
                    f"{key} ref={ref_stats[key]:.0f} test={test_stats[key]:.0f}"
                )
    time_ok = True
    time_ratio = float("nan")
    if (
        max_time_ratio is not None
        and ref_elapsed is not None
        and test_elapsed is not None
        and ref_elapsed > 0
    ):
        time_ratio = test_elapsed / ref_elapsed
        time_ok = time_ratio <= max_time_ratio
    dose_ok = (
        abs(metrics["integral_diff_pct"]) <= max_integral_diff_pct
        and metrics["normalized_l1_pct"] <= max_l1_pct
    )
    if require_bitwise:
        dose_ok = dose_ok and bool(metrics["bitwise_equal"])
    passed = dose_ok and count_ok and time_ok
    return {
        "gate": name,
        "passed": passed,
        "dose_ok": dose_ok,
        "count_ok": count_ok,
        "time_ok": time_ok,
        "time_ratio": time_ratio,
        "count_notes": count_notes,
        **metrics,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    result = gate_compare(
        args.name,
        Path(args.ref),
        Path(args.test),
        max_integral_diff_pct=args.max_integral_diff_pct,
        max_l1_pct=args.max_l1_pct,
        require_bitwise=args.require_bitwise,
        max_time_ratio=args.max_time_ratio,
        ref_elapsed=args.ref_elapsed,
        test_elapsed=args.test_elapsed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def make_non_minibeam_config(src: Path, dst: Path) -> None:
    text = src.read_text()
    # Force legacy contract flags.
    replacements = {
        r"(?m)^minibeam:.*$": "minibeam: false",
        r"(?m)^secondary_persistent_workers:.*$": "secondary_persistent_workers: 0",
        r"(?m)^secondary_condensed_step_mm:.*$": "secondary_condensed_step_mm: 0",
        r"(?m)^secondary_fp32_energy_residual:.*$": "secondary_fp32_energy_residual: false",
        r"(?m)^robust_boundary_nudge:.*$": "robust_boundary_nudge: false",
        r"(?m)^electronic_buildup_lateral_sigma_mm:.*$": "electronic_buildup_lateral_sigma_mm: 0",
        r"(?m)^dose_output_scale:.*$": "dose_output_scale: 1.0",
    }
    for pat, rep in replacements.items():
        if re.search(pat, text):
            text = re.sub(pat, rep, text, count=1)
        else:
            text += f"\n{rep}\n"
    # Strip copper-only required paths when minibeam is forced false — leave them;
    # they are ignored when minibeam is false.
    dst.write_text(text)


def cmd_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []

    water_cfg = Path(args.water_config)
    legacy_cfg = outdir / "legacy_water.yaml"
    make_non_minibeam_config(water_cfg, legacy_cfg)

    # Gate 1: master vs OFF (if master binary provided)
    off_bin = Path(args.off_bin)
    if not off_bin.is_file():
        raise SystemExit(f"OFF binary missing: {off_bin}")

    off_csv = outdir / "gate1_off_water.csv"
    off_log, off_elapsed = run_case(off_bin, legacy_cfg, off_csv)

    if args.master_bin:
        master_bin = Path(args.master_bin)
        master_csv = outdir / "gate1_master_water.csv"
        master_log, master_elapsed = run_case(master_bin, legacy_cfg, master_csv)
        g1 = gate_compare(
            "gate1_off_vs_master",
            master_csv,
            off_csv,
            max_integral_diff_pct=args.g1_integral_pct,
            max_l1_pct=args.g1_l1_pct,
            require_bitwise=args.g1_bitwise,
            ref_log=master_log,
            test_log=off_log,
            max_time_ratio=args.g1_max_time_ratio,
            ref_elapsed=master_elapsed,
            test_elapsed=off_elapsed,
        )
    else:
        # Self-check only: OFF ran successfully; mark incomplete.
        g1 = {
            "gate": "gate1_off_vs_master",
            "passed": False,
            "note": "master binary not provided; OFF run stored for manual compare",
            "off_elapsed": off_elapsed,
        }
    results.append(g1)

    # Gate 2: ON + minibeam false vs OFF
    on_bin = Path(args.on_bin)
    if not on_bin.is_file():
        raise SystemExit(f"ON binary missing: {on_bin}")
    on_false_csv = outdir / "gate2_on_false_water.csv"
    on_false_log, on_false_elapsed = run_case(on_bin, legacy_cfg, on_false_csv)
    g2 = gate_compare(
        "gate2_on_false_vs_off",
        off_csv,
        on_false_csv,
        max_integral_diff_pct=args.g2_integral_pct,
        max_l1_pct=args.g2_l1_pct,
        require_bitwise=False,
        ref_log=off_log,
        test_log=on_false_log,
        max_time_ratio=args.g2_max_time_ratio,
        ref_elapsed=off_elapsed,
        test_elapsed=on_false_elapsed,
    )
    results.append(g2)

    # Gate 3: ON + minibeam true smoke (or CSV compare if ref given)
    mini_cfg = Path(args.minibeam_config)
    mini_csv = outdir / "gate3_minibeam.csv"
    mini_log, mini_elapsed = run_case(on_bin, mini_cfg, mini_csv)
    g3: Dict[str, object] = {
        "gate": "gate3_minibeam_true",
        "passed": True,
        "elapsed": mini_elapsed,
        "log_excerpt": mini_log[-1500:],
    }
    if args.minibeam_ref_csv:
        g3 = gate_compare(
            "gate3_minibeam_true",
            Path(args.minibeam_ref_csv),
            mini_csv,
            max_integral_diff_pct=args.g3_integral_pct,
            max_l1_pct=args.g3_l1_pct,
            require_bitwise=False,
            ref_log="",
            test_log=mini_log,
        )
        g3["elapsed"] = mini_elapsed
    results.append(g3)

    summary = {
        "results": results,
        "all_passed": all(bool(r.get("passed")) for r in results),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Compare two depth-dose CSVs")
    compare.add_argument("--ref", required=True)
    compare.add_argument("--test", required=True)
    compare.add_argument("--name", default="compare")
    compare.add_argument("--max-integral-diff-pct", type=float, default=1e-3)
    compare.add_argument("--max-l1-pct", type=float, default=1e-2)
    compare.add_argument("--require-bitwise", action="store_true")
    compare.add_argument("--max-time-ratio", type=float, default=None)
    compare.add_argument("--ref-elapsed", type=float, default=None)
    compare.add_argument("--test-elapsed", type=float, default=None)
    compare.set_defaults(func=cmd_compare)

    run = sub.add_parser("run", help="Run three-gate isolation harness")
    run.add_argument("--master-bin", default="")
    run.add_argument("--off-bin", required=True)
    run.add_argument("--on-bin", required=True)
    run.add_argument("--outdir", default="out/reg_isolation")
    run.add_argument(
        "--water-config",
        default="config/beam_200MeVu_fragment_cascade_smoke.yaml",
        help="Non-minibeam water/cascade case used for gates 1–2",
    )
    run.add_argument(
        "--minibeam-config",
        default="config/beam_minibeam_center_copper_em_10k.yaml",
        help="Minibeam true smoke case for gate 3",
    )
    run.add_argument("--minibeam-ref-csv", default="")
    run.add_argument("--g1-integral-pct", type=float, default=1e-4)
    run.add_argument("--g1-l1-pct", type=float, default=1e-3)
    run.add_argument("--g1-bitwise", action="store_true")
    run.add_argument("--g1-max-time-ratio", type=float, default=1.05)
    run.add_argument("--g2-integral-pct", type=float, default=1e-3)
    run.add_argument("--g2-l1-pct", type=float, default=1e-2)
    run.add_argument("--g2-max-time-ratio", type=float, default=1.15)
    run.add_argument("--g3-integral-pct", type=float, default=1.0)
    run.add_argument("--g3-l1-pct", type=float, default=2.5)
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
