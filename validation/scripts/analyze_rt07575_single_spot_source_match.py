#!/usr/bin/env python3
"""Compare the archived RT07575 TOPAS 100k single spot with current GPU best.

This is a reproducible, no-fit source-geometry diagnostic.  It converts the
existing TOPAS binary scorer to the native patient grid, renders the current
best GPU template with only run-path/history/seed/spot substitutions, executes
the one 100,000-history spot on CUDA, maps the GPU dose to patient axes, and
reports absolute-dose moments and fixed-coordinate profiles.  It never runs
TOPAS and never registers, rescales, or broadens either dose map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PATIENT_SHAPE = (417, 505, 35)  # x, y, z
EXPECTED_GPU_SHAPE = (505, 35, 417)  # patient-y, patient-z, depth=-patient-x
ALLOWED_CONFIG_KEYS = {
    "number_of_histories",
    "random_seed",
    "topas_spots_file",
    "voxel_dose_mhd_output_file",
    "let_voxel_mhd_output_file",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mhd(path: Path) -> tuple[dict[str, str], np.ndarray]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    shape = tuple(int(v) for v in meta["DimSize"].split())
    raw = path.parent / meta["ElementDataFile"]
    values = np.fromfile(raw, dtype="<f4")
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{raw}: expected {np.prod(shape)} float32 values, got {values.size}")
    return meta, values.reshape(shape[2], shape[1], shape[0]).astype(np.float64)


def write_mhd(path: Path, values: np.ndarray, spacing: tuple[float, float, float], offset: tuple[float, float, float], units: str = "Gy") -> None:
    if values.ndim != 3:
        raise ValueError("MHD output must be z,y,x")
    nz, ny, nx = values.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw")
    values.astype("<f4", copy=False).tofile(raw)
    path.write_text("\n".join((
        "ObjectType = Image", "NDims = 3", "BinaryData = True",
        "BinaryDataByteOrderMSB = False", "CompressedData = False",
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        f"Offset = {' '.join(f'{v:.12g}' for v in offset)}",
        "CenterOfRotation = 0 0 0",
        f"ElementSpacing = {' '.join(f'{v:.12g}' for v in spacing)}",
        f"DimSize = {nx} {ny} {nz}", "ElementType = MET_FLOAT",
        f"DoseUnits = {units}", f"ElementDataFile = {raw.name}", "",
    )), encoding="ascii")


def topas_geometry(header: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    axes: dict[str, tuple[int, float]] = {}
    pattern = re.compile(r"^#\s*([XYZ])\s+in\s+(\d+)\s+bins?\s+of\s+([0-9.eE+-]+)\s+(mm|cm)\s*$")
    for line in header.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            axis, count, width, unit = match.groups()
            axes[axis] = (int(count), float(width) * (10.0 if unit == "cm" else 1.0))
    if set(axes) != {"X", "Y", "Z"}:
        raise ValueError(f"{header}: incomplete TOPAS xyz geometry")
    return ((axes["X"][0], axes["Y"][0], axes["Z"][0]),
            (axes["X"][1], axes["Y"][1], axes["Z"][1]))


def convert_topas(source: Path, header: Path, ct_metadata: Path, output: Path) -> dict[str, Any]:
    shape, spacing = topas_geometry(header)
    if shape != EXPECTED_PATIENT_SHAPE:
        raise ValueError(f"{source}: expected patient shape {EXPECTED_PATIENT_SHAPE}, got {shape}")
    count = int(np.prod(shape))
    if source.stat().st_size == count * 8:
        dtype = np.dtype("<f8")
    elif source.stat().st_size == count * 4:
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"{source}: unsupported binary byte count")
    values = np.memmap(source, dtype=dtype, mode="r", shape=(shape[2], shape[1], shape[0]))
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite dose")
    metadata = json.loads(ct_metadata.read_text(encoding="utf-8"))
    edge = tuple(float(v) for v in metadata["origin_xyz_mm"])
    ct_spacing = tuple(float(v) for v in metadata["spacing_xyz_mm"])
    if spacing != ct_spacing:
        raise ValueError(f"TOPAS spacing {spacing} != CT patient spacing {ct_spacing}")
    offset = tuple(edge[i] + 0.5 * spacing[i] for i in range(3))
    write_mhd(output, np.asarray(values), spacing, offset)
    return {
        "source": str(source), "source_sha256": sha256(source), "header": str(header),
        "shape_xyz": list(shape), "spacing_xyz_mm": list(spacing),
        "offset_first_center_xyz_mm": list(offset), "dtype": dtype.name,
        "voxel_count": count, "sum": float(np.sum(values, dtype=np.float64)),
        "maximum": float(np.max(values)), "output_mhd": str(output),
    }


def replace_one(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    result, count = pattern.subn(f"{key}: {value}", text)
    if count != 1:
        raise ValueError(f"expected exactly one {key} in template, found {count}")
    return result


def render_config(template: Path, output: Path, run_dir: Path, spots: Path, histories: int, seed: int) -> dict[str, Any]:
    original = template.read_text(encoding="utf-8")
    rendered = original
    for key, value in (
        ("number_of_histories", str(histories)), ("random_seed", str(seed)),
        ("topas_spots_file", str(spots)),
        ("voxel_dose_mhd_output_file", str(run_dir / "dose.mhd")),
        ("let_voxel_mhd_output_file", str(run_dir / "letd")),
    ):
        rendered = replace_one(rendered, key, value)
    output.write_text(rendered, encoding="utf-8")
    changes: list[str] = []
    before = dict(re.findall(r"^([A-Za-z0-9_]+):\s*(.*)$", original, re.MULTILINE))
    after = dict(re.findall(r"^([A-Za-z0-9_]+):\s*(.*)$", rendered, re.MULTILINE))
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append(key)
    unexpected = sorted(set(changes) - ALLOWED_CONFIG_KEYS)
    if unexpected:
        raise ValueError(f"rendered config changed non-run keys: {unexpected}")
    return {"changed_keys": changes, "unchanged_physics_config": not unexpected}


def spot_histories(path: Path) -> list[int]:
    match = re.search(r"^iv:Tf/Scatterer1/L4/Values\s*=\s*(\d+)\s+(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"{path}: missing L4 history channel")
    expected, values = int(match.group(1)), [int(v) for v in match.group(2).split()]
    if expected != len(values):
        raise ValueError(f"{path}: L4 declared {expected} values, found {len(values)}")
    return values


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    def find(pattern: str, cast=float) -> Any:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            raise ValueError(f"{path}: missing {pattern}")
        return cast(match.group(1))
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    return {
        "histories": find(r"^Histories:\s*(\d+)$", int),
        "elapsed_seconds": find(r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": find(r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"),
        "energy_balance_error": find(r"^Energy balance error:\s*([0-9.eE+-]+)$"),
        "secondary_queue_overflow": find(r"^Secondary queue overflow:\s*(\d+)$", int),
        "cascade_queue_overflow": find(r"^Cascade queue overflow:\s*(\d+)$", int),
        "backend": backend.group(1) if backend else "unknown",
    }


def moment_report(dose: np.ndarray, spacing: tuple[float, float, float], offset: tuple[float, float, float], threshold_fraction: float) -> dict[str, Any]:
    """Dose-weighted moments on z,y,x data without materializing 3-D coordinates."""
    peak = float(np.max(dose))
    weights = np.where(dose >= threshold_fraction * peak, dose, 0.0)
    total = float(weights.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("empty moment support")
    nz, ny, nx = dose.shape
    x = offset[0] + spacing[0] * np.arange(nx)
    y = offset[1] + spacing[1] * np.arange(ny)
    z = offset[2] + spacing[2] * np.arange(nz)
    wx, wy, wz = weights.sum(axis=(0, 1)), weights.sum(axis=(0, 2)), weights.sum(axis=(1, 2))
    com = np.array([np.dot(wx, x), np.dot(wy, y), np.dot(wz, z)], dtype=np.float64) / total
    xx = float(np.dot(wx, (x - com[0]) ** 2) / total)
    yy = float(np.dot(wy, (y - com[1]) ** 2) / total)
    zz = float(np.dot(wz, (z - com[2]) ** 2) / total)
    xy = float(np.sum(weights.sum(axis=0) * (y[:, None] - com[1]) * (x[None, :] - com[0])) / total)
    xz = float(np.sum(weights.sum(axis=1) * (z[:, None] - com[2]) * (x[None, :] - com[0])) / total)
    yz = float(np.sum(weights.sum(axis=2) * (z[:, None] - com[2]) * (y[None, :] - com[1])) / total)
    covariance = np.array(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {
        "threshold_fraction_of_own_dmax": threshold_fraction,
        "support_voxels": int(np.count_nonzero(weights)), "support_dose_sum": total,
        "com_patient_xyz_mm": com.tolist(), "covariance_patient_xyz_mm2": covariance.tolist(),
        "axis_rms_width_xyz_mm": np.sqrt(np.diag(covariance)).tolist(),
        "principal_rms_widths_mm": np.sqrt(np.maximum(values, 0.0)).tolist(),
        "principal_axes_columns_patient_xyz": vectors.tolist(),
    }


def depth_metrics(dose: np.ndarray, spacing: tuple[float, float, float], offset: tuple[float, float, float]) -> dict[str, Any]:
    """Integrated-depth dose with depth increasing from +patient-X entrance to -X."""
    idd = dose.sum(axis=(0, 1))
    x = offset[0] + spacing[0] * np.arange(dose.shape[2])
    depth = x[-1] - x  # beam is +patient-X -> -patient-X
    peak_index = int(np.argmax(idd))
    peak = float(idd[peak_index])
    # In depth order, indices decrease.  Scan from peak towards increasing depth.
    ordered = np.arange(peak_index, -1, -1)
    below = ordered[idd[ordered] <= 0.8 * peak]
    r80 = None
    if below.size:
        high_index = int(below[0] + 1)
        low_index = int(below[0])
        if high_index < idd.size and idd[high_index] != idd[low_index]:
            fraction = (0.8 * peak - idd[low_index]) / (idd[high_index] - idd[low_index])
            r80 = float(depth[low_index] + fraction * (depth[high_index] - depth[low_index]))
        else:
            r80 = float(depth[low_index])
    return {"beam_direction_patient": "+X to -X", "depth_coordinate_mm": "x_max - patient_x",
            "idd_peak_depth_mm": float(depth[peak_index]), "idd_peak_patient_x_mm": float(x[peak_index]),
            "idd_peak_sum": peak, "r80_distal_depth_mm": r80,
            "depth_mm": depth.tolist(), "idd_sum": idd.tolist()}


def profile_rows(topas: np.ndarray, gpu: np.ndarray, topas_moments: dict[str, Any], topas_depth: dict[str, Any], spacing: tuple[float, float, float], offset: tuple[float, float, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Unfitted lateral profile rows at fixed TOPAS COM Y/Z and three TOPAS depths."""
    com = topas_moments["com_patient_xyz_mm"]
    y_index = int(np.clip(round((com[1] - offset[1]) / spacing[1]), 0, topas.shape[1] - 1))
    z_index = int(np.clip(round((com[2] - offset[2]) / spacing[2]), 0, topas.shape[0] - 1))
    idd = np.asarray(topas_depth["idd_sum"])
    peak = int(np.argmax(idd))
    # Beam depth increases as patient-X index decreases.  Entrance is 20% of
    # TOPAS IDD before peak; middle is halfway from that entrance to peak.
    pre_peak = np.arange(idd.size - 1, peak - 1, -1)
    crossed = pre_peak[idd[pre_peak] >= 0.2 * idd[peak]]
    entrance = int(crossed[0]) if crossed.size else int(pre_peak[-1])
    mid = int(round((entrance + peak) / 2))
    xcoord = offset[0] + spacing[0] * np.arange(topas.shape[2])
    ycoord = offset[1] + spacing[1] * np.arange(topas.shape[1])
    zcoord = offset[2] + spacing[2] * np.arange(topas.shape[0])
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for label, xi in (("entrance", entrance), ("mid", mid), ("peak", peak)):
        for axis, coordinate, ref, ev in (
            ("patient_y", ycoord, topas[z_index, :, xi], gpu[z_index, :, xi]),
            ("patient_z", zcoord, topas[:, y_index, xi], gpu[:, y_index, xi]),
        ):
            support = ref >= 0.10 * max(float(ref.max()), 1.0e-30)
            if np.count_nonzero(support) < 2:
                support = ref > 0.0
            corr = float(np.corrcoef(ref[support], ev[support])[0, 1]) if np.count_nonzero(support) > 1 and np.std(ref[support]) > 0 and np.std(ev[support]) > 0 else float("nan")
            summary.append({"depth_label": label, "patient_x_mm": float(xcoord[xi]), "axis": axis,
                            "fixed_patient_y_mm": float(ycoord[y_index]), "fixed_patient_z_mm": float(zcoord[z_index]),
                            "support_points": int(np.count_nonzero(support)), "pearson_r": corr,
                            "rmse_over_topas_plane_peak_percent": float(100.0 * np.sqrt(np.mean((ev[support] - ref[support]) ** 2)) / max(float(ref.max()), 1.0e-30))})
            for c, reference, evaluation in zip(coordinate, ref, ev, strict=True):
                rows.append({"depth_label": label, "patient_x_mm": float(xcoord[xi]), "axis": axis,
                             "coordinate_mm": float(c), "topas_dose_Gy": float(reference), "gpu_dose_Gy": float(evaluation)})
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--spots", type=Path, default=Path("ct/fullplan_result/RT07575/spots_single_spot_100k_local40.generated.txt"))
    parser.add_argument("--topas-binary", type=Path, default=Path("ct/fullplan_result/RT07575/OSMK_Dtotal_single_spot_100k_local40.bin"))
    parser.add_argument("--topas-header", type=Path, default=Path("ct/fullplan_result/RT07575/OSMK_Dtotal_single_spot_100k_local40.binheader"))
    parser.add_argument("--ct-metadata", type=Path, default=Path("ct/grid/patient_ct.metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/ct/RT07575/single_spot_source_diagnostic"))
    parser.add_argument("--histories", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    for key in ("binary", "template", "spots", "topas_binary", "topas_header", "ct_metadata", "output_dir"):
        value = getattr(args, key)
        if not value.is_absolute():
            setattr(args, key, args.repo_root / value)
    if args.histories != 100_000:
        raise ValueError("this fixed diagnostic requires exactly 100,000 histories")
    spot_counts = spot_histories(args.spots)
    if spot_counts != [args.histories]:
        raise ValueError(f"single spot must contain exactly [{args.histories}] L4 histories, got {spot_counts}")
    run_dir = args.output_dir / "gpu_seed"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conversion = convert_topas(args.topas_binary, args.topas_header, args.ct_metadata, args.output_dir / "topas_patient.mhd")
    config = run_dir / "config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_diff = render_config(args.template, config, run_dir, args.spots, args.histories, args.seed)
    required = (run_dir / "dose.mhd", run_dir / "dose.raw", run_dir / "run.log")
    stats: dict[str, Any] | None = None
    if all(path.exists() for path in required):
        try:
            candidate = parse_log(run_dir / "run.log")
            if candidate["histories"] == args.histories and not candidate["secondary_queue_overflow"] and not candidate["cascade_queue_overflow"]:
                stats = candidate
                print("SKIP completed GPU single spot", flush=True)
        except ValueError:
            pass
    if stats is None:
        print("RUN GPU single spot on CUDA", flush=True)
        started = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
            finished = subprocess.run([str(args.binary), "--config", str(config), "--device", "cuda"], cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT, check=False, text=True)
        if finished.returncode != 0:
            raise RuntimeError(f"GPU run failed ({finished.returncode}); see {run_dir / 'run.log'}")
        stats = parse_log(run_dir / "run.log")
        stats["wall_seconds_runner"] = time.monotonic() - started
    if stats["histories"] != args.histories or stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]:
        raise RuntimeError(f"invalid GPU run stats: {stats}")
    topas_meta, topas = read_mhd(args.output_dir / "topas_patient.mhd")
    gpu_meta, gpu_native = read_mhd(run_dir / "dose.mhd")
    if tuple(int(v) for v in topas_meta["DimSize"].split()) != EXPECTED_PATIENT_SHAPE:
        raise ValueError("converted TOPAS MHD shape mismatch")
    if tuple(int(v) for v in gpu_meta["DimSize"].split()) != EXPECTED_GPU_SHAPE:
        raise ValueError("GPU MHD shape mismatch")
    # GPU [depth=patient-X, patient-Z, patient-Y] -> patient [Z,Y,X].
    gpu = np.transpose(gpu_native, (1, 2, 0))[:, :, ::-1]
    spacing = tuple(float(v) for v in topas_meta["ElementSpacing"].split())
    offset = tuple(float(v) for v in topas_meta["Offset"].split())
    write_mhd(args.output_dir / "gpu_patient.mhd", gpu, spacing, offset)
    diff = gpu - topas
    topas_sum = float(topas.sum(dtype=np.float64))
    gpu_sum = float(gpu.sum(dtype=np.float64))
    selection = (topas >= 0.10 * topas.max()) | (gpu >= 0.10 * gpu.max())
    corr = float(np.corrcoef(topas[selection], gpu[selection])[0, 1])
    topas_moments = {f"threshold_{pct}pct": moment_report(topas, spacing, offset, pct / 100.0) for pct in (1, 10)}
    gpu_moments = {f"threshold_{pct}pct": moment_report(gpu, spacing, offset, pct / 100.0) for pct in (1, 10)}
    moment_delta: dict[str, Any] = {}
    for key in topas_moments:
        ref, ev = topas_moments[key], gpu_moments[key]
        moment_delta[key] = {
            "gpu_minus_topas_com_patient_xyz_mm": (np.asarray(ev["com_patient_xyz_mm"]) - np.asarray(ref["com_patient_xyz_mm"])).tolist(),
            "axis_rms_width_ratio_gpu_over_topas_xyz": (np.asarray(ev["axis_rms_width_xyz_mm"]) / np.asarray(ref["axis_rms_width_xyz_mm"])).tolist(),
            "principal_rms_width_ratio_gpu_over_topas": (np.asarray(ev["principal_rms_widths_mm"]) / np.asarray(ref["principal_rms_widths_mm"])).tolist(),
        }
    topas_depth, gpu_depth = depth_metrics(topas, spacing, offset), depth_metrics(gpu, spacing, offset)
    depth_path = args.output_dir / "integrated_depth_dose.csv"
    with depth_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patient_x_mm", "beam_depth_mm", "topas_idd_sum", "gpu_idd_sum"))
        writer.writeheader()
        for x_index, (beam_depth, topas_idd, gpu_idd) in enumerate(zip(topas_depth["depth_mm"], topas_depth["idd_sum"], gpu_depth["idd_sum"], strict=True)):
            writer.writerow({"patient_x_mm": float(offset[0] + spacing[0] * x_index), "beam_depth_mm": beam_depth, "topas_idd_sum": topas_idd, "gpu_idd_sum": gpu_idd})
    profiles, profile_metrics = profile_rows(topas, gpu, topas_moments["threshold_10pct"], topas_depth, spacing, offset)
    profile_path = args.output_dir / "lateral_profiles.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profiles[0]))
        writer.writeheader(); writer.writerows(profiles)
    fullplan_y = -0.20389
    single_y = float(moment_delta["threshold_10pct"]["gpu_minus_topas_com_patient_xyz_mm"][1])
    widths = moment_delta["threshold_10pct"]["axis_rms_width_ratio_gpu_over_topas_xyz"]
    delta = moment_delta["threshold_10pct"]["gpu_minus_topas_com_patient_xyz_mm"]
    source_correction_present = abs(single_y - fullplan_y) <= 0.10
    depth_summary = lambda values: {key: value for key, value in values.items() if key not in {"depth_mm", "idd_sum"}}
    source_conclusion = (
        "The single spot reproduces the full-plan patient-Y displacement and has "
        "materially narrower GPU high-dose lateral RMS widths. Source pose/emittance "
        "is therefore a main contributor to the rigid/high-gradient residual, although "
        "this one 100k realization cannot apportion every remaining transport residual."
        if source_correction_present and min(widths) < 0.95 else
        "The single spot does not reproduce the full-plan source-pose/width signature; "
        "source geometry/emittance is not supported as the main residual cause."
    )
    report: dict[str, Any] = {
        "case": "RT07575", "purpose": "single-spot source-pose/width diagnostic against archived TOPAS; no fit",
        "normalization": "absolute scored dose; no fitted scale, translation, or width", "histories": args.histories, "seed": args.seed,
        "inputs": {"template": str(args.template), "template_sha256": sha256(args.template), "spots": str(args.spots), "spots_sha256": sha256(args.spots), "topas_binary": str(args.topas_binary), "topas_header": str(args.topas_header)},
        "topas_conversion": conversion, "config_diff_check": config_diff, "gpu_run": stats,
        "mapping": "GPU MHD [depth=patient-X, patient-Z, patient-Y] -> patient [Z,Y,X] via transpose(1,2,0), then reverse patient X",
        "mapping_validation": {"patient_shape_xyz": list(EXPECTED_PATIENT_SHAPE), "mapped_gpu_sum_minus_native_sum": float(gpu.sum(dtype=np.float64) - gpu_native.sum(dtype=np.float64)), "gpu_patient_mhd": str(args.output_dir / "gpu_patient.mhd")},
        "absolute_dose": {"topas_integral_sum": topas_sum, "gpu_integral_sum": gpu_sum, "gpu_over_topas_integral_ratio": gpu_sum / topas_sum, "union_10pct_dmax_voxels": int(np.count_nonzero(selection)), "union_10pct_dmax_pearson_r": corr, "union_10pct_dmax_nrmse_over_topas_dmax_percent": float(100.0 * np.sqrt(np.mean(diff[selection] ** 2)) / topas.max())},
        "moments": {"topas": topas_moments, "gpu": gpu_moments, "gpu_minus_topas": moment_delta},
        "depth_metrics": {"topas": depth_summary(topas_depth), "gpu": depth_summary(gpu_depth), "gpu_minus_topas": {"idd_peak_depth_mm": gpu_depth["idd_peak_depth_mm"] - topas_depth["idd_peak_depth_mm"], "r80_distal_depth_mm": None if topas_depth["r80_distal_depth_mm"] is None or gpu_depth["r80_distal_depth_mm"] is None else gpu_depth["r80_distal_depth_mm"] - topas_depth["r80_distal_depth_mm"]}, "integrated_depth_csv": str(depth_path)},
        "lateral_profiles": {"csv": str(profile_path), "definition": "TOPAS fixed dose-COM Y/Z, TOPAS entrance/mid/peak depths; no alignment or scale", "metrics": profile_metrics},
        "fullplan_rigid_correction_reference": {"gpu_mean_minus_topas_mean_patient_y_mm": fullplan_y, "single_spot_gpu_minus_topas_patient_y_mm_10pct": single_y, "already_present_within_0p10mm": source_correction_present},
        "conclusion": source_conclusion,
        "limitations": "Each engine has one 100k realization. Moment/profile evidence is primary; no gamma is reported because low statistics make it secondary.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# RT07575 single-spot source diagnostic", "", "Absolute 100,000-history TOPAS/GPU comparison. No fitted dose scale, translation, or width.", "", f"GPU runtime: {stats['elapsed_seconds']:.3f} s ({stats['throughput_histories_per_second']:.0f} histories/s); overflows: secondary={stats['secondary_queue_overflow']}, cascade={stats['cascade_queue_overflow']}.", f"Integral GPU/TOPAS: {gpu_sum / topas_sum:.6f}; union-10%-Dmax correlation: {corr:.6f}; NRMSE/Dmax: {report['absolute_dose']['union_10pct_dmax_nrmse_over_topas_dmax_percent']:.3f}%.", "", "| Moment support | COM GPU − TOPAS (x, y, z) mm | GPU/TOPAS RMS width (x, y, z) |", "|---|---:|---:|", f"| own Dmax ≥1% | {[round(v, 4) for v in moment_delta['threshold_1pct']['gpu_minus_topas_com_patient_xyz_mm']]} | {[round(v, 4) for v in moment_delta['threshold_1pct']['axis_rms_width_ratio_gpu_over_topas_xyz']]} |", f"| own Dmax ≥10% | {[round(v, 4) for v in delta]} | {[round(v, 4) for v in widths]} |", "", f"Depth: IDD peak GPU − TOPAS = {report['depth_metrics']['gpu_minus_topas']['idd_peak_depth_mm']:.3f} mm; R80 GPU − TOPAS = {report['depth_metrics']['gpu_minus_topas']['r80_distal_depth_mm']!s} mm.", f"Full-plan patient-Y correction is already present (±0.10 mm test): {source_correction_present}; full-plan ΔY={fullplan_y:.4f} mm, single-spot ΔY={single_y:.4f} mm.", "", "## Conclusion", "", source_conclusion, "", f"Detailed fixed-coordinate lateral profiles: `{profile_path}`. Integrated depth dose: `{depth_path}`."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gpu_run": stats, "com_delta_10pct_mm": delta, "width_ratios_10pct": widths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
