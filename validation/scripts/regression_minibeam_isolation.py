#!/usr/bin/env python3
"""Three-gate minibeam isolation regression.

Gate 1 — CARBON_ENABLE_MINIBEAM=OFF vs master (or reference binary):
  non-minibeam dose bitwise/near-bitwise; Steps and nuclear counts match;
  transport/kernel times not significantly slower (median of repeats).

Gate 2 — CARBON_ENABLE_MINIBEAM=ON + minibeam:false vs OFF binary:
  dose consistent; uses legacy kernel (no +minibeam in Backend); kernel
  time close to OFF (not process wall time).

Gate 3 — CARBON_ENABLE_MINIBEAM=ON + minibeam:true:
  Backend contains minibeam; diagnostics non-zero when enabled; MHD/raw
  output present with non-zero dose when configured; optional depth CSV
  vs frozen reference.

Usage:
  python3 validation/scripts/regression_minibeam_isolation.py run \\
      --off-bin build/oneapi-nvidia-release/carbon_mc \\
      --on-bin build/oneapi-nvidia-minibeam/carbon_mc \\
      --outdir out/reg_isolation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_depth_dose_csv(path: Path) -> Tuple[List[float], List[float]]:
    depths: List[float] = []
    doses: List[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(
            (row for row in handle if row.strip() and not row.lstrip().startswith("#"))
        )
        fieldnames = reader.fieldnames or []
        depth_key = next(
            (
                k
                for k in fieldnames
                if k.lower() in {"depth_mm", "z_mm", "depth", "z"}
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
            handle.seek(0)
            raw = csv.reader(
                (
                    row
                    for row in handle
                    if row.strip() and not row.lstrip().startswith("#")
                )
            )
            next(raw, None)
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
    l1_pct = (
        100.0 * l1 / sum(abs(x) for x in ref) if ref_sum != 0.0 else float("nan")
    )
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


def parse_run_log(text: str) -> Dict[str, object]:
    """Parse carbon_mc summary lines (Steps:, Kernel time:, Backend:, ...)."""
    out: Dict[str, object] = {}
    m = re.search(r"(?m)^Steps:\s*(\d+)\s*$", text)
    if m:
        out["total_steps"] = float(m.group(1))
    m = re.search(r"(?m)^Nuclear interactions:\s*(\d+)\s*$", text)
    if m:
        out["nuclear_interactions"] = float(m.group(1))
    m = re.search(r"(?m)^Elapsed:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*s\s*$", text)
    if m:
        out["elapsed_seconds"] = float(m.group(1))
    m = re.search(
        r"(?m)^Kernel time:\s*primary=([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*s\s+"
        r"secondary=([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*s\s+"
        r"neutral=([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*s\s+"
        r"charged-after-neutral=([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*s\s*$",
        text,
    )
    if m:
        out["primary_kernel_seconds"] = float(m.group(1))
        out["secondary_kernel_seconds"] = float(m.group(2))
        out["neutral_kernel_seconds"] = float(m.group(3))
        out["charged_after_neutral_kernel_seconds"] = float(m.group(4))
        out["transport_kernel_seconds"] = (
            float(m.group(1))
            + float(m.group(2))
            + float(m.group(3))
            + float(m.group(4))
        )
    m = re.search(r"(?m)^Backend:\s*(.+)\s*$", text)
    if m:
        out["backend"] = m.group(1).strip()
    m = re.search(r"(?m)^Histories:\s*(\d+)\s*$", text)
    if m:
        out["histories"] = float(m.group(1))
    m = re.search(
        r"Minibeam water entrance primary C-12 by slit:\s*([0-9/]+)", text
    )
    if m:
        slits = [int(x) for x in m.group(1).split("/") if x]
        out["minibeam_water_entrance_c12"] = float(sum(slits))
    m = re.search(r"Minibeam incident histories[^:]*:\s*(\d+)", text, re.I)
    if m:
        out["minibeam_incident"] = float(m.group(1))
    return out


def require_log_fields(stats: Dict, fields: Sequence[str], label: str) -> None:
    missing = [f for f in fields if f not in stats]
    if missing:
        raise RuntimeError(
            f"{label}: missing required log fields {missing}. "
            f"Present keys: {sorted(stats)}"
        )


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def inspect_mhd_dose(mhd_path: Path) -> Dict[str, object]:
    """Validate MHD exists, has a raw sibling, and contains non-zero dose."""
    if not mhd_path.is_file():
        raise RuntimeError(f"MHD missing: {mhd_path}")
    text = mhd_path.read_text(errors="replace")
    dims_m = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
    elem_m = re.search(r"ElementType\s*=\s*(\S+)", text)
    data_m = re.search(r"ElementDataFile\s*=\s*(\S+)", text)
    if not dims_m or not data_m:
        raise RuntimeError(f"MHD header incomplete: {mhd_path}")
    nx, ny, nz = (int(dims_m.group(i)) for i in range(1, 4))
    if min(nx, ny, nz) <= 0:
        raise RuntimeError(f"Invalid DimSize in {mhd_path}")
    raw_name = data_m.group(1)
    raw_path = (mhd_path.parent / raw_name).resolve()
    if not raw_path.is_file():
        raise RuntimeError(f"MHD raw missing: {raw_path}")
    raw = raw_path.read_bytes()
    elem = (elem_m.group(1) if elem_m else "MET_FLOAT").upper()
    if "DOUBLE" in elem:
        width, fmt = 8, "<d"
    else:
        width, fmt = 4, "<f"
    expected = nx * ny * nz * width
    if len(raw) < expected:
        raise RuntimeError(
            f"Raw size {len(raw)} < expected {expected} for {raw_path}"
        )
    n_vals = expected // width
    # Sample up to 256k values for speed; require at least one finite non-zero.
    step = max(1, n_vals // 262144)
    nonzero = 0
    total = 0.0
    peak = 0.0
    for i in range(0, n_vals, step):
        (val,) = struct.unpack_from(fmt, raw, i * width)
        if not math.isfinite(val):
            continue
        total += abs(val)
        peak = max(peak, abs(val))
        if val != 0.0:
            nonzero += 1
    if nonzero == 0 or peak <= 0.0:
        raise RuntimeError(f"MHD dose is all-zero or non-finite: {mhd_path}")
    return {
        "mhd": str(mhd_path),
        "raw": str(raw_path),
        "dim": [nx, ny, nz],
        "element": elem,
        "raw_bytes": len(raw),
        "nonzero_samples": nonzero,
        "peak_abs": peak,
        "sum_abs_samples": total,
    }


def config_mhd_path(config_text: str, config_path: Path) -> Optional[Path]:
    m = re.search(r"(?m)^voxel_dose_mhd_output_file:[ \t]*(\S+)\s*$", config_text)
    if not m:
        return None
    p = Path(m.group(1))
    if not p.is_absolute():
        # Paths in YAML are repo-relative for this project.
        p = Path.cwd() / p
    return p


def run_case(
    binary: Path,
    config: Path,
    out_csv: Path,
    *,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    force_output: Optional[bool] = None,
) -> Tuple[str, Dict[str, float], float]:
    """Run once. Returns (log, parsed_stats, process_wall_s)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_csv.with_suffix(".yaml")
    shutil.copyfile(config, tmp_cfg)
    text = tmp_cfg.read_text()
    has_empty_output = bool(re.search(r"(?m)^output_file:[ \t]*$", text))
    has_nonempty_output = bool(re.search(r"(?m)^output_file:[ \t]*\S+", text))
    has_mhd = bool(re.search(r"(?m)^voxel_dose_mhd_output_file:[ \t]*\S+", text))
    species_off = bool(
        re.search(r"(?m)^enable_fragment_species_scoring:[ \t]*false[ \t]*$", text)
    )
    allow_depth = True
    if force_output is not None:
        allow_depth = force_output
    elif species_off and has_mhd and has_empty_output and not has_nonempty_output:
        allow_depth = False

    cmd = [str(binary), "--config", str(tmp_cfg), "--device", "cuda"]
    if allow_depth:
        cmd.extend(["--output", str(out_csv)])
    if extra_args:
        cmd.extend(extra_args)
    run_env = dict(os.environ)
    run_env.setdefault("ONEAPI_DEVICE_SELECTOR", "cuda:gpu")
    if env:
        run_env.update(env)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env, check=False)
    wall = time.perf_counter() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_csv.with_suffix(".log").write_text(log)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} failed ({proc.returncode}) for {config}:\n{log[-4000:]}"
        )
    stats = parse_run_log(log)
    if allow_depth:
        if not out_csv.is_file():
            raise RuntimeError(f"Expected output CSV missing: {out_csv}\n{log[-2000:]}")
    return log, stats, wall  # type: ignore[return-value]


def run_case_median(
    binary: Path,
    config: Path,
    out_stem: Path,
    *,
    repeats: int,
    warmup: int,
    time_field: str = "transport_kernel_seconds",
    require_fields: Sequence[str] = (
        "total_steps",
        "nuclear_interactions",
        "elapsed_seconds",
        "primary_kernel_seconds",
        "secondary_kernel_seconds",
        "transport_kernel_seconds",
    ),
    force_output: Optional[bool] = None,
) -> Tuple[str, Dict[str, float], Path]:
    """Warmup then repeat; return last log, median stats, last CSV path."""
    times: List[float] = []
    primary_times: List[float] = []
    secondary_times: List[float] = []
    elapsed_times: List[float] = []
    last_log = ""
    last_stats: Dict[str, float] = {}
    last_csv = out_stem
    total_runs = warmup + repeats
    for i in range(total_runs):
        csv_path = out_stem if i == total_runs - 1 else out_stem.with_name(
            out_stem.stem + f"_r{i}" + out_stem.suffix
        )
        log, stats, _wall = run_case(
            binary, config, csv_path, force_output=force_output
        )
        require_log_fields(stats, require_fields, f"{binary.name} run {i}")
        if i >= warmup:
            times.append(float(stats[time_field]))
            primary_times.append(float(stats["primary_kernel_seconds"]))
            secondary_times.append(float(stats["secondary_kernel_seconds"]))
            elapsed_times.append(float(stats["elapsed_seconds"]))
        last_log, last_stats, last_csv = log, stats, csv_path
    last_stats = dict(last_stats)
    last_stats["median_transport_kernel_seconds"] = median(times)
    last_stats["median_primary_kernel_seconds"] = median(primary_times)
    last_stats["median_secondary_kernel_seconds"] = median(secondary_times)
    last_stats["median_elapsed_seconds"] = median(elapsed_times)
    last_stats["timing_repeats"] = float(repeats)
    last_stats["timing_warmup"] = float(warmup)
    return last_log, last_stats, last_csv


def gate_compare_dose(
    name: str,
    ref_csv: Path,
    test_csv: Path,
    *,
    max_integral_diff_pct: float,
    max_l1_pct: float,
    require_bitwise: bool,
    ref_stats: Dict[str, float],
    test_stats: Dict[str, float],
    max_time_ratio: Optional[float],
    time_key: str = "median_transport_kernel_seconds",
    require_count_match: bool = True,
) -> Dict[str, object]:
    _, ref_dose = load_depth_dose_csv(ref_csv)
    _, test_dose = load_depth_dose_csv(test_csv)
    metrics = compare_series(ref_dose, test_dose, label=name)
    count_ok = True
    count_notes: List[str] = []
    if require_count_match:
        for key in ("nuclear_interactions", "total_steps"):
            if key not in ref_stats or key not in test_stats:
                count_ok = False
                count_notes.append(f"{key} missing from logs")
            elif ref_stats[key] != test_stats[key]:
                count_ok = False
                count_notes.append(
                    f"{key} ref={ref_stats[key]:.0f} test={test_stats[key]:.0f}"
                )
    time_ok = True
    time_ratio = float("nan")
    if max_time_ratio is not None:
        if time_key not in ref_stats or time_key not in test_stats:
            time_ok = False
            count_notes.append(f"timing field {time_key} missing")
        else:
            ref_t = float(ref_stats[time_key])
            test_t = float(test_stats[time_key])
            if ref_t <= 0.0:
                time_ok = False
                count_notes.append(f"ref {time_key} <= 0")
            else:
                time_ratio = test_t / ref_t
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
        "time_key": time_key,
        "ref_time": ref_stats.get(time_key),
        "test_time": test_stats.get(time_key),
        "count_notes": count_notes,
        "ref_backend": ref_stats.get("backend"),
        "test_backend": test_stats.get("backend"),
        **metrics,
    }


def make_non_minibeam_config(src: Path, dst: Path) -> None:
    text = src.read_text()
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
    dst.write_text(text)


def gate3_minibeam(
    binary: Path,
    config: Path,
    outdir: Path,
    *,
    ref_csv: str,
    max_integral_diff_pct: float,
    max_l1_pct: float,
) -> Dict[str, object]:
    out_csv = outdir / "gate3_minibeam.csv"
    # Unique MHD path under outdir so we can verify fresh output.
    tmp_cfg = outdir / "gate3_minibeam_cfg.yaml"
    text = config.read_text()
    mhd_out = outdir / "gate3_minibeam_dose.mhd"
    if re.search(r"(?m)^voxel_dose_mhd_output_file:", text):
        text = re.sub(
            r"(?m)^voxel_dose_mhd_output_file:.*$",
            f"voxel_dose_mhd_output_file: {mhd_out.as_posix()}",
            text,
            count=1,
        )
    else:
        text += f"\nvoxel_dose_mhd_output_file: {mhd_out.as_posix()}\n"
    # Prefer enabling diagnostics for validation when the case is minibeam.
    if re.search(r"(?m)^minibeam_diagnostics:", text):
        text = re.sub(
            r"(?m)^minibeam_diagnostics:.*$",
            "minibeam_diagnostics: true",
            text,
            count=1,
        )
    else:
        text += "\nminibeam_diagnostics: true\n"
    tmp_cfg.write_text(text)
    # Remove stale MHD so we can prove the run created a new one.
    for stale in (mhd_out, mhd_out.with_suffix(".raw"), outdir / "gate3_minibeam_dose.raw"):
        if stale.exists():
            stale.unlink()

    log, stats, _wall = run_case(binary, tmp_cfg, out_csv)
    notes: List[str] = []
    passed = True

    backend = str(stats.get("backend", ""))
    if "minibeam" not in backend.lower():
        passed = False
        notes.append(f"Backend lacks minibeam marker: {backend!r}")

    # Diagnostics: require some evidence of beamline activity.
    diag_ok = False
    if "minibeam_water_entrance_c12" in stats and stats["minibeam_water_entrance_c12"] > 0:
        diag_ok = True
    if re.search(r"Minibeam", log):
        # Any minibeam diagnostic block is required.
        if re.search(r"Minibeam water entrance|Minibeam Copper|beamline", log, re.I):
            diag_ok = True
    if not diag_ok:
        passed = False
        notes.append("minibeam diagnostics missing or zero")

    mhd_info: Dict[str, object] = {}
    try:
        mhd_info = inspect_mhd_dose(mhd_out)
    except Exception as exc:  # noqa: BLE001 - surface as gate failure
        # Some smoke configs may not write MHD if voxel scoring off.
        cfg_has_voxel = bool(
            re.search(r"(?m)^enable_voxel_scoring:[ \t]*true\s*$", text)
        )
        if cfg_has_voxel:
            passed = False
            notes.append(f"MHD validation failed: {exc}")
        else:
            notes.append(f"MHD skipped (voxel scoring off): {exc}")

    dose_metrics: Dict[str, object] = {}
    if ref_csv:
        if not out_csv.is_file() or out_csv.stat().st_size < 32:
            passed = False
            notes.append("depth CSV missing for reference compare")
        else:
            try:
                _, ref_dose = load_depth_dose_csv(Path(ref_csv))
                _, test_dose = load_depth_dose_csv(out_csv)
                dose_metrics = compare_series(
                    ref_dose, test_dose, label="gate3_ref"
                )
                if (
                    abs(dose_metrics["integral_diff_pct"]) > max_integral_diff_pct
                    or dose_metrics["normalized_l1_pct"] > max_l1_pct
                ):
                    passed = False
                    notes.append("depth dose vs reference out of tolerance")
            except Exception as exc:  # noqa: BLE001
                passed = False
                notes.append(f"reference compare failed: {exc}")
    else:
        notes.append(
            "no --minibeam-ref-csv: depth reference not checked "
            "(backend/diagnostics/MHD still required)"
        )

    return {
        "gate": "gate3_minibeam_true",
        "passed": passed,
        "backend": backend,
        "notes": notes,
        "stats": {k: v for k, v in stats.items() if not isinstance(v, str)},
        "backend_str": backend,
        "mhd": mhd_info,
        "dose_metrics": dose_metrics,
        "log_excerpt": log[-2000:],
    }


def cmd_compare(args: argparse.Namespace) -> int:
    ref_stats = parse_run_log(Path(args.ref_log).read_text()) if args.ref_log else {}
    test_stats = parse_run_log(Path(args.test_log).read_text()) if args.test_log else {}
    if args.ref_log:
        require_log_fields(
            ref_stats,
            ["total_steps", "primary_kernel_seconds", "transport_kernel_seconds"],
            "ref",
        )
    if args.test_log:
        require_log_fields(
            test_stats,
            ["total_steps", "primary_kernel_seconds", "transport_kernel_seconds"],
            "test",
        )
    # Allow injecting median from single-run elapsed if medians absent.
    for s in (ref_stats, test_stats):
        if "median_transport_kernel_seconds" not in s and "transport_kernel_seconds" in s:
            s["median_transport_kernel_seconds"] = s["transport_kernel_seconds"]
    result = gate_compare_dose(
        args.name,
        Path(args.ref),
        Path(args.test),
        max_integral_diff_pct=args.max_integral_diff_pct,
        max_l1_pct=args.max_l1_pct,
        require_bitwise=args.require_bitwise,
        ref_stats=ref_stats,
        test_stats=test_stats,
        max_time_ratio=args.max_time_ratio,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []

    water_cfg = Path(args.water_config)
    legacy_cfg = outdir / "legacy_water.yaml"
    make_non_minibeam_config(water_cfg, legacy_cfg)

    off_bin = Path(args.off_bin)
    if not off_bin.is_file():
        raise SystemExit(f"OFF binary missing: {off_bin}")
    on_bin = Path(args.on_bin)
    if not on_bin.is_file():
        raise SystemExit(f"ON binary missing: {on_bin}")

    repeats = max(1, args.repeats)
    warmup = max(0, args.warmup)

    # --- Gate 1 ---
    off_log, off_stats, off_csv = run_case_median(
        off_bin,
        legacy_cfg,
        outdir / "gate1_off_water.csv",
        repeats=repeats,
        warmup=warmup,
    )
    # OFF backend must not claim minibeam.
    if "minibeam" in str(off_stats.get("backend", "")).lower():
        raise RuntimeError(f"OFF backend unexpectedly minibeam: {off_stats.get('backend')}")

    if args.master_bin:
        master_bin = Path(args.master_bin)
        master_log, master_stats, master_csv = run_case_median(
            master_bin,
            legacy_cfg,
            outdir / "gate1_master_water.csv",
            repeats=repeats,
            warmup=warmup,
        )
        g1 = gate_compare_dose(
            "gate1_off_vs_master",
            master_csv,
            off_csv,
            max_integral_diff_pct=args.g1_integral_pct,
            max_l1_pct=args.g1_l1_pct,
            require_bitwise=args.g1_bitwise,
            ref_stats=master_stats,
            test_stats=off_stats,
            max_time_ratio=args.g1_max_time_ratio,
        )
    else:
        g1 = {
            "gate": "gate1_off_vs_master",
            "passed": False,
            "note": "master binary not provided; OFF median timing stored",
            "off_stats": {
                k: off_stats[k]
                for k in off_stats
                if k.startswith("median_") or k in ("total_steps", "backend")
            },
        }
    results.append(g1)

    # --- Gate 2: ON + minibeam false must use legacy kernel ---
    on_false_log, on_false_stats, on_false_csv = run_case_median(
        on_bin,
        legacy_cfg,
        outdir / "gate2_on_false_water.csv",
        repeats=repeats,
        warmup=warmup,
    )
    on_backend = str(on_false_stats.get("backend", ""))
    if "minibeam" in on_backend.lower():
        results.append(
            {
                "gate": "gate2_on_false_vs_off",
                "passed": False,
                "note": f"ON+minibeam:false still reports minibeam backend: {on_backend}",
            }
        )
    else:
        g2 = gate_compare_dose(
            "gate2_on_false_vs_off",
            off_csv,
            on_false_csv,
            max_integral_diff_pct=args.g2_integral_pct,
            max_l1_pct=args.g2_l1_pct,
            require_bitwise=args.g2_bitwise,
            ref_stats=off_stats,
            test_stats=on_false_stats,
            max_time_ratio=args.g2_max_time_ratio,
        )
        results.append(g2)

    # --- Gate 3 ---
    g3 = gate3_minibeam(
        on_bin,
        Path(args.minibeam_config),
        outdir,
        ref_csv=args.minibeam_ref_csv,
        max_integral_diff_pct=args.g3_integral_pct,
        max_l1_pct=args.g3_l1_pct,
    )
    results.append(g3)

    summary = {
        "results": results,
        "all_passed": all(bool(r.get("passed")) for r in results),
        "timing": {
            "warmup": warmup,
            "repeats": repeats,
            "time_metric": "median_transport_kernel_seconds",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_passed"] else 1


def cmd_selftest(_args: argparse.Namespace) -> int:
    sample = """
Backend: sycl-cuda+straggling+fp64-dose+attenuation
Histories: 1000
Steps: 4680768
Elapsed: 0.0495809 s
Kernel time: primary=0.007434849 s secondary=0.009331966 s neutral=0 s charged-after-neutral=0 s
Nuclear interactions: 341
Minibeam water entrance primary C-12 by slit: 0/0/0/0/6/49/244/487/283/38/5/0/0/0/0
"""
    stats = parse_run_log(sample)
    require_log_fields(
        stats,
        [
            "total_steps",
            "nuclear_interactions",
            "elapsed_seconds",
            "primary_kernel_seconds",
            "secondary_kernel_seconds",
            "transport_kernel_seconds",
            "backend",
        ],
        "selftest",
    )
    assert stats["total_steps"] == 4680768.0
    assert abs(float(stats["transport_kernel_seconds"]) - (0.007434849 + 0.009331966)) < 1e-12
    assert str(stats["backend"]).startswith("sycl-cuda")
    assert float(stats["minibeam_water_entrance_c12"]) == 1112.0
    # Missing fields must fail.
    try:
        require_log_fields({}, ["total_steps"], "empty")
    except RuntimeError:
        pass
    else:
        raise AssertionError("require_log_fields should fail on empty")
    print("selftest ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    selftest = sub.add_parser("selftest", help="Parse-log unit checks (no GPU)")
    selftest.set_defaults(func=cmd_selftest)

    compare = sub.add_parser("compare", help="Compare two depth-dose CSVs + optional logs")
    compare.add_argument("--ref", required=True)
    compare.add_argument("--test", required=True)
    compare.add_argument("--ref-log", default="")
    compare.add_argument("--test-log", default="")
    compare.add_argument("--name", default="compare")
    compare.add_argument("--max-integral-diff-pct", type=float, default=1e-3)
    compare.add_argument("--max-l1-pct", type=float, default=1e-2)
    compare.add_argument("--require-bitwise", action="store_true")
    compare.add_argument("--max-time-ratio", type=float, default=None)
    compare.set_defaults(func=cmd_compare)

    run = sub.add_parser("run", help="Run three-gate isolation harness")
    run.add_argument("--master-bin", default="")
    run.add_argument("--off-bin", required=True)
    run.add_argument("--on-bin", required=True)
    run.add_argument("--outdir", default="out/reg_isolation")
    run.add_argument(
        "--water-config",
        default="config/beam_200MeVu_fragment_cascade_smoke.yaml",
    )
    run.add_argument(
        "--minibeam-config",
        default="config/beam_minibeam_center_copper_em_10k.yaml",
    )
    run.add_argument("--minibeam-ref-csv", default="")
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--g1-integral-pct", type=float, default=1e-4)
    run.add_argument("--g1-l1-pct", type=float, default=1e-3)
    run.add_argument("--g1-bitwise", action="store_true")
    run.add_argument("--g1-max-time-ratio", type=float, default=1.10)
    run.add_argument("--g2-integral-pct", type=float, default=1e-3)
    run.add_argument("--g2-l1-pct", type=float, default=1e-2)
    run.add_argument("--g2-bitwise", action="store_true")
    run.add_argument("--g2-max-time-ratio", type=float, default=1.10)
    run.add_argument("--g3-integral-pct", type=float, default=1.0)
    run.add_argument("--g3-l1-pct", type=float, default=2.5)
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
