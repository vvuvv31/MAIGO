#!/usr/bin/env python3
"""RT07575 diagnostic: physically propagate each source energy through World air.

This intentionally changes no production code.  It reproduces the TPS-90
source-to-corrected-CT-entry geometry in ``apply_spot_to_config`` and reduces
only TOPAS spot channel L2 (total C-12 kinetic energy) by an RK4 integration
of the G4_AIR stopping-power table.  It omits air straggling and MCS.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_rt07575_equal_history_groups import load_gpu_mapped, load_topas  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402
from match_gpu_to_physical_dose import dose_only_pass_rate, gamma_3d  # noqa: E402


HISTORIES = 12_963_817
SEED = 20_260_801
MASS_NUMBER = 12.0
SAD_MM = 450.0
PATIENT_TRANSLATION_MM = (-42.8515, -12.7636, 1.3617)
PATIENT_ROT_Z_DEG = 90.0
CT_AXIS_MIN_MM = -104.25
QUANTITIES = {"dose": "dose.mhd", "primary_c12_letd": "letd_primary_c12.mhd",
              "all_hadron_letd": "letd_all_hadron.mhd"}
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))
LAYER_RE = re.compile(r"^\s*[diu]v:Tf/Scatterer1/L(\d+)/(Times|Values)\s*=\s*(.*?)\s*(?:ms|MeV|mm|deg)?\s*$")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def replace_one(text: str, key: str, value: str) -> str:
    result, count = re.subn(rf"^{re.escape(key)}:.*$", f"{key}: {value}", text,
                            flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return result


def parse_values(body: str) -> list[float]:
    numbers = [float(token) for token in body.split()]
    if len(numbers) < 2:
        raise ValueError("TOPAS channel needs count and at least one value")
    declared = int(round(numbers[0]))
    if declared != len(numbers) - 1:
        raise ValueError(f"TOPAS channel count {declared} != {len(numbers) - 1}")
    return numbers[1:]


def spot_channels(path: Path) -> tuple[list[str], dict[int, list[float]], int]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    channels: dict[int, list[float]] = {}
    l2_line = -1
    for index, line in enumerate(lines):
        matched = LAYER_RE.match(line.rstrip("\n"))
        if not matched or matched.group(2) != "Values":
            continue
        layer = int(matched.group(1))
        channels[layer] = parse_values(matched.group(3))
        if layer == 2:
            l2_line = index
    if set(range(15)) - set(channels):
        raise ValueError("spots file does not contain every L0..L14 Values channel")
    count = len(channels[0])
    if any(len(value) != count for value in channels.values()):
        raise ValueError("TOPAS channel lengths differ")
    if l2_line < 0:
        raise ValueError("spots file lacks L2 Values")
    return lines, channels, l2_line


def rotate_rx_ry(rx_deg: float, ry_deg: float, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    rx, ry = np.deg2rad(rx_deg), np.deg2rad(ry_deg)
    y, z = np.cos(rx) * y - np.sin(rx) * z, np.sin(rx) * y + np.cos(rx) * z
    x, z = np.cos(ry) * x + np.sin(ry) * z, -np.sin(ry) * x + np.cos(ry) * z
    return float(x), float(y), float(z)


def source_to_ct_entrance_mm(trans_x: float, trans_z: float, rot_x: float, rot_y: float) -> float:
    """Literal Python counterpart of TPS-90 pose conversion plus distance_to_entrance."""
    # tps_zero_beam_pose_for_spot: translation is in world coordinates, and the
    # local source axes use inverse component Rx/Ry rotations.
    wx, wy, wz = trans_x, -SAD_MM, trans_z
    dx, dy, dz = rotate_rx_ry(-rot_x, -rot_y, (0.0, 0.0, 1.0))
    px, py, pz = PATIENT_TRANSLATION_MM
    angle = np.deg2rad(PATIENT_ROT_Z_DEG)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    # world -> patient point/vector, exactly as transform_tps_90_pose_to_ct.
    ox, oy = c * (wx - px) - s * (wy - py), s * (wx - px) + c * (wy - py)
    uxx, _uxy = c * dx - s * dy, s * dx + c * dy
    _oz = wz - pz
    _uzx, _uzy, _uzz = c * dx - s * dy, s * dx + c * dy, dz
    if _uzx >= 0.0:
        origin_z, direction_z = ox - CT_AXIS_MIN_MM, _uzx
    else:
        origin_z, direction_z = -CT_AXIS_MIN_MM - ox, -_uzx
    if direction_z <= 1.0e-6:
        raise ValueError("TPS spot does not point into reoriented CT")
    distance = -origin_z / direction_z
    if distance < 0.0:
        raise ValueError("TPS source is downstream of CT entrance")
    # Keep unused geometry variables visibly evaluated: this makes the source
    # conversion auditable against the C++ transform's full world/patient pose.
    _ = (oy, pz, _oz, uxx, _uzy, _uzz)
    return distance


def air_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # The archived table has two provenance comments followed by its CSV header.
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=3)
    energy, stopping = data[:, 0], data[:, 1]
    if not (np.all(np.diff(energy) > 0) and np.all(stopping > 0)):
        raise ValueError("invalid stopping-power table")
    return energy, stopping


def air_loss_rk4(total_energy_mev: float, distance_mm: float,
                 energy_mevu: np.ndarray, stopping_mev_per_mm: np.ndarray,
                 step_mm: float) -> float:
    """Integrate dE_total/dx=-S_air(E_total/A) with bounded RK4 substeps."""
    energy = total_energy_mev
    remaining = distance_mm
    while remaining > 1.0e-12:
        h = min(step_mm, remaining)
        def derivative(value: float) -> float:
            per_u = value / MASS_NUMBER
            if per_u < energy_mevu[0] or per_u > energy_mevu[-1]:
                raise ValueError(f"air-loss energy {per_u} MeV/u outside table")
            return -float(np.interp(per_u, energy_mevu, stopping_mev_per_mm))
        k1 = derivative(energy)
        k2 = derivative(energy + 0.5 * h * k1)
        k3 = derivative(energy + 0.5 * h * k2)
        k4 = derivative(energy + h * k3)
        energy += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        remaining -= h
    if energy <= 0.0:
        raise ValueError("air loss exhausted spot energy")
    return total_energy_mev - energy


def air_losses_rk4(total_energy_mev: np.ndarray, distance_mm: np.ndarray,
                   energy_mevu: np.ndarray, stopping_mev_per_mm: np.ndarray,
                   step_mm: float) -> np.ndarray:
    """Vectorized equivalent of ``air_loss_rk4`` for all plan spots."""
    energy = total_energy_mev.astype(np.float64, copy=True)
    remaining = distance_mm.astype(np.float64, copy=True)
    while float(np.max(remaining)) > 1.0e-12:
        h = np.minimum(step_mm, remaining)
        def derivative(value: np.ndarray) -> np.ndarray:
            per_u = value / MASS_NUMBER
            if float(np.min(per_u)) < float(energy_mevu[0]) or float(np.max(per_u)) > float(energy_mevu[-1]):
                raise ValueError("air-loss energy outside table")
            return -np.interp(per_u, energy_mevu, stopping_mev_per_mm)
        k1 = derivative(energy)
        k2 = derivative(energy + 0.5 * h * k1)
        k3 = derivative(energy + 0.5 * h * k2)
        k4 = derivative(energy + h * k3)
        energy += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        remaining -= h
    return total_energy_mev - energy


def distribution(values: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(values)), "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.median(values)), "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)), "mean": float(np.mean(values))}


def generate_spots(args: argparse.Namespace) -> dict[str, Any]:
    lines, channels, l2_line = spot_channels(args.spots_source)
    n = len(channels[0])
    energies = np.asarray(channels[2], dtype=np.float64)
    paths = np.asarray([source_to_ct_entrance_mm(channels[5][i], channels[6][i],
                                                   channels[7][i], channels[8][i])
                        for i in range(n)], dtype=np.float64)
    table_energy, table_stopping = air_table(args.air_stopping_table)
    losses = air_losses_rk4(energies, paths, table_energy, table_stopping, 0.10)
    # Independent finer calculation demonstrates quadrature convergence without
    # adding a fitted correction or a physics approximation.
    losses_fine = air_losses_rk4(energies, paths, table_energy, table_stopping, 0.05)
    if float(np.max(np.abs(losses - losses_fine))) > 1.0e-8:
        raise RuntimeError("air-loss quadrature failed 0.10/0.05 mm convergence")
    corrected = energies - losses
    active = np.asarray(channels[4], dtype=np.int64) > 0
    replacement = "dv:Tf/Scatterer1/L2/Values = " + str(n) + " " + " ".join(
        f"{value:.12g}" for value in corrected) + " MeV\n"
    output_lines = lines.copy()
    output_lines[l2_line] = replacement
    args.generated_spots.parent.mkdir(parents=True, exist_ok=True)
    args.generated_spots.write_text("".join(output_lines), encoding="utf-8")
    source_text = args.spots_source.read_text(encoding="utf-8")
    generated_text = args.generated_spots.read_text(encoding="utf-8")
    expected_text = "".join(lines[:l2_line] + [replacement] + lines[l2_line + 1:])
    if generated_text != expected_text:
        raise RuntimeError("generated spots changed text outside L2")
    _, generated_channels, _ = spot_channels(args.generated_spots)
    for layer in range(15):
        if layer != 2 and generated_channels[layer] != channels[layer]:
            raise RuntimeError(f"generated spots changed L{layer}")
    if not np.allclose(np.asarray(generated_channels[2]), corrected, rtol=0.0, atol=5e-9):
        raise RuntimeError("generated L2 is not the derived corrected energy")
    audit = {
        "method": "production TPS-90 source-to-entry geometry; RK4 G4_AIR electronic stopping, 0.10 mm step",
        "scope": "L2 total kinetic energy only; no upstream air straggling or MCS",
        "spots_total": n, "spots_active": int(np.count_nonzero(active)),
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "changed_text_line": l2_line + 1, "only_l2_changed": True,
        "source_to_corrected_ct_entrance_mm": {"all": distribution(paths), "active": distribution(paths[active])},
        "energy_loss_total_MeV": {"all": distribution(losses), "active": distribution(losses[active])},
        "energy_loss_MeVu": {"all": distribution(losses / MASS_NUMBER), "active": distribution(losses[active] / MASS_NUMBER)},
        "quadrature_0p10_minus_0p05_mm_max_abs_MeV": float(np.max(np.abs(losses - losses_fine))),
        "energy_total_MeV": {"source": distribution(energies), "at_ct_entrance": distribution(corrected)},
    }
    write_json(args.output_root / "spots_audit.json", audit)
    return audit


def number(text: str, pattern: str, default: float | None = None) -> float:
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(found.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    return {"histories": int(number(text, r"^Histories:\s*(\d+)$")),
            "elapsed_seconds": number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
            "throughput_histories_per_second": number(text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"),
            "energy_balance_error": number(text, r"^Energy balance error:\s*([0-9.eE+-]+)$"),
            "secondary_queue_overflow": int(number(text, r"^Secondary queue overflow:\s*(\d+)$")),
            "cascade_queue_overflow": int(number(text, r"^Cascade queue overflow:\s*(\d+)$")),
            "neutral_queue_overflow": int(number(text, r"^Neutral queue overflow:\s*(\d+)$", 0)),
            "backend": backend.group(1) if backend else "unknown"}


def complete(run_dir: Path) -> dict[str, Any] | None:
    required = [run_dir / value for value in QUANTITIES.values()] + [run_dir / "dose.raw",
                run_dir / "letd_primary_c12.raw", run_dir / "letd_all_hadron.raw", run_dir / "run.log",
                run_dir / "config.yaml"]
    if not all(path.exists() for path in required):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES or any(stats[key] for key in
            ("secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow")):
        return None
    return stats


def render_config(args: argparse.Namespace, path: Path, run_dir: Path) -> list[str]:
    text = args.template.read_text(encoding="utf-8")
    overrides = {"number_of_histories": str(HISTORIES), "random_seed": str(SEED),
                 "ct_grid_file": "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin",
                 "spots_ct_axis_min_mm": str(CT_AXIS_MIN_MM), "topas_spots_file": str(args.generated_spots),
                 "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
                 "let_voxel_mhd_output_file": str(run_dir / "letd")}
    for key, value in overrides.items():
        text = replace_one(text, key, value)
    path.write_text(text, encoding="utf-8")
    return list(overrides)


def run_air_arm(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    run_dir = args.output_root / "air_loss"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = complete(run_dir)
    if saved is not None:
        print("SKIP complete air_loss", flush=True)
        return saved, ["resumed complete output"]
    changed = render_config(args, run_dir / "config.yaml", run_dir)
    print("RUN air_loss", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(args.binary), "--config", str(run_dir / "config.yaml"), "--device", "cuda"],
                                cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT,
                                text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"air_loss failed ({result.returncode}); see {run_dir / 'run.log'}")
    saved = complete(run_dir)
    if saved is None:
        raise RuntimeError("air_loss is incomplete, wrong-history, or overflowed")
    return saved, changed


def quantity_metrics(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, body: np.ndarray,
                     shape: tuple[int, int, int], spacing: tuple[float, float, float], cache: Path) -> dict[str, Any]:
    item: dict[str, Any] = json.loads(cache.read_text()) if cache.exists() else {}
    ref_body, eval_body = np.where(body, reference, 0.0), np.where(body, evaluation, 0.0)
    ref_selected, eval_selected = reference[selection], evaluation[selection]
    if not item:
        delta = eval_selected - ref_selected
        item = {"selected_voxels": int(np.count_nonzero(selection)),
                "evaluation_over_reference_body_integral": float(np.sum(eval_body, dtype=np.float64) / max(np.sum(ref_body, dtype=np.float64), 1e-30)),
                "selected_mean_reference": float(np.mean(ref_selected)), "selected_mean_evaluation": float(np.mean(eval_selected)),
                "selected_nrmse_over_reference_max_percent": float(100 * np.sqrt(np.mean(delta * delta)) / max(float(np.max(ref_selected)), 1e-30)),
                "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]), "gamma": {}}
        write_json(cache, item)
    ref_flat, eval_flat = array.array("f", ref_body.astype(np.float32, copy=False)), eval_body.reshape(-1).tolist()
    for percent, distance, label in CRITERIA:
        criterion = item["gamma"].setdefault(label, {})
        for local, mode in ((False, "global"), (True, "local")):
            if mode not in criterion:
                print(f"Computing {cache.stem}: {label} {mode}", flush=True)
                criterion[mode] = gamma_3d(ref_flat, eval_flat, shape, spacing, percent, distance, 10.0, 50_000, 0,
                                           local_dose=local, interpolation_step_mm=0.5, selection_mask=selection.reshape(-1))
                write_json(cache, item)
    if "30" not in item["gamma"]:
        item["gamma"]["30"] = {"global": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0, selection_mask=selection.reshape(-1)),
                                 "local": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0, local_dose=True, selection_mask=selection.reshape(-1))}
        write_json(cache, item)
    return item


def comparison(args: argparse.Namespace) -> dict[str, Any]:
    topas, gpu = load_topas(args.topas_dir), load_gpu_mapped(args.output_root / "air_loss")
    metadata, values = load(args.body_mask)
    shape = tuple(int(v) for v in metadata["DimSize"].split())
    spacing = tuple(float(v) for v in metadata["ElementSpacing"].split())
    if shape != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape}")
    body = values > 0.5
    selection = body & (topas["dose"] >= 0.10 * float(np.max(topas["dose"][body])))
    return {"reference": str(args.topas_dir), "evaluation": str(args.output_root / "air_loss"),
            "normalization": "absolute equal-history scale; no fitted normalization",
            "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
            "let_selection": "same TOPAS seed 1 dose mask; no LET threshold", "selected_voxels": int(np.count_nonzero(selection)),
            "gamma_sampling": {"3d_points": 50_000, "available_points": int(np.count_nonzero(selection)),
                               "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
            "quantities": {name: quantity_metrics(topas[name], gpu[name], selection, body, shape, spacing,
                                                    args.output_root / "gamma_checkpoints" / "air_loss" / f"{name}.json")
                           for name in QUANTITIES}}


def gamma_deltas(baseline: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    return {name: {label: {mode: trial["quantities"][name]["gamma"][label][mode]["pass_percent"] - baseline["quantities"][name]["gamma"][label][mode]["pass_percent"]
                           for mode in ("global", "local")} for _, _, label in CRITERIA + ((3.0, 0.0, "30"),)} for name in QUANTITIES}


def scalar_deltas(baseline: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    return {name: {"body_integral_ratio": trial["quantities"][name]["evaluation_over_reference_body_integral"] - baseline["quantities"][name]["evaluation_over_reference_body_integral"],
                   "selected_nrmse_over_reference_max_percent": trial["quantities"][name]["selected_nrmse_over_reference_max_percent"] - baseline["quantities"][name]["selected_nrmse_over_reference_max_percent"]}
            for name in QUANTITIES}


def markdown(report: dict[str, Any]) -> str:
    air = report["air_path_and_loss"]
    lines = ["# RT07575 upstream-air energy-loss diagnostic", "",
             f"Both GPU arms use {HISTORIES:,} histories, seed {SEED}, and no fitted normalization or alignment.",
             "The diagnostic starts from the corrected-origin geometry and changes only L2 total C-12 energy after RK4 G4_AIR stopping-power propagation to CT entry. Air straggling/MCS are intentionally omitted.", "",
             "## Air-path and energy audit", "",
             f"Source→corrected CT entry path (active spots), mm: {air['source_to_corrected_ct_entrance_mm']['active']}.",
             f"Total C-12 energy loss (active spots), MeV: {air['energy_loss_total_MeV']['active']}.",
             f"Energy loss per nucleon (active spots), MeV/u: {air['energy_loss_MeVu']['active']}.",
             f"Source/generated spots SHA-256: `{air['source_sha256']}` / `{air['generated_sha256']}`; only L2 changed: {air['only_l2_changed']}.", "",
             "## Runtime and safety", "", "| Arm | Elapsed (s) | Throughput (hist/s) | Energy balance error | Secondary / cascade / neutral overflow |", "|---|---:|---:|---:|---:|"]
    for arm, stats in report["runs"].items():
        lines.append(f"| {arm} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | {stats['energy_balance_error']:.8g} | {stats['secondary_queue_overflow']} / {stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |")
    for arm, result in report["comparisons"].items():
        lines += ["", f"## {arm} vs TOPAS seed 1", "", "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, metric in result["quantities"].items():
            g = metric["gamma"]
            cells = [f"{g[label]['global']['pass_percent']:.3f} / {g[label]['local']['pass_percent']:.3f}" for label in ("33", "22", "11", "30")]
            lines.append(f"| {name} | {metric['evaluation_over_reference_body_integral']:.6f} | {metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | {metric['selected_nrmse_over_reference_max_percent']:.3f}% | " + " | ".join(cells) + " |")
    lines += ["", "## Air-loss minus corrected-origin baseline", "", "| Quantity | Δ integral E/R | Δ NRMSE/Dmax (pp) | Δ 3%/3mm G/L (pp) | Δ 2%/2mm G/L (pp) | Δ 1%/1mm G/L (pp) | Δ 3%/0mm G/L (pp) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in QUANTITIES:
        scalar, gamma = report["strict_deltas_vs_corrected_origin"][name], report["gamma_deltas_vs_corrected_origin_pp"][name]
        cells = [f"{gamma[label]['global']:+.3f} / {gamma[label]['local']:+.3f}" for label in ("33", "22", "11", "30")]
        lines.append(f"| {name} | {scalar['body_integral_ratio']:+.6f} | {scalar['selected_nrmse_over_reference_max_percent']:+.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--spots-source", type=Path, default=Path("ct/fullplan_result/RT07575/spots_full_plan.txt"))
    parser.add_argument("--air-stopping-table", type=Path, default=Path("data/stopping_power_air_geant4_11_3_2.csv"))
    parser.add_argument("--corrected-baseline-dir", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation/edge_corrected"))
    parser.add_argument("--corrected-baseline-summary", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation/summary.json"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/upstream_air_loss_ablation"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for field, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    args.generated_spots = args.output_root / "spots_full_plan_air_loss.txt"
    for path in (args.binary, args.template, args.spots_source, args.air_stopping_table, args.corrected_baseline_dir,
                 args.corrected_baseline_summary, args.topas_dir, args.body_mask):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = generate_spots(args)
    if args.prepare_only:
        print(f"Wrote {args.generated_spots}", flush=True)
        return 0
    baseline_stats = complete(args.corrected_baseline_dir)
    if baseline_stats is None:
        raise RuntimeError("corrected-origin baseline is not exact-history and zero-overflow")
    baseline = json.loads(args.corrected_baseline_summary.read_text(encoding="utf-8"))["comparisons"]["edge_corrected"]
    for name in QUANTITIES:
        for _, _, label in CRITERIA + ((3.0, 0.0, "30"),):
            if not {"global", "local"}.issubset(baseline["quantities"][name]["gamma"].get(label, {})):
                raise ValueError(f"incomplete corrected-origin gamma: {name}/{label}")
    air_stats, changed = run_air_arm(args)
    trial = comparison(args)
    report = {"case": "RT07575", "experiment": "upstream World-air energy loss", "histories": HISTORIES, "seed": SEED,
              "template": str(args.template), "generated_spots": str(args.generated_spots), "air_path_and_loss": audit,
              "rendered_config_changed_keys": changed,
              "only_mechanism_change": "per-spot L2 total kinetic energy loss through G4_AIR to corrected CT entrance; no straggling/MCS",
              "corrected_origin_gpu_seed1": str(args.corrected_baseline_dir), "topas_seed1": str(args.topas_dir),
              "runs": {"corrected_origin_baseline": baseline_stats, "upstream_air_loss": air_stats},
              "comparisons": {"corrected_origin_baseline": baseline, "upstream_air_loss": trial},
              "gamma_deltas_vs_corrected_origin_pp": gamma_deltas(baseline, trial),
              "strict_deltas_vs_corrected_origin": scalar_deltas(baseline, trial)}
    write_json(args.output_root / "summary.json", report)
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
