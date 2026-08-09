#!/usr/bin/env python3
"""Reproduce the three-case equal-history GPU--TOPAS dose comparison.

The case table below is intentionally explicit: dose paths, raw TOPAS format,
GPU-to-patient mapping, grid shape, and history count are all declared rather
than inferred from filenames.  No scale, translation, rotation, or other fit is
performed by this script.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    name: str
    histories: int
    setup: str
    gpu_mhd: str
    gpu_log: str
    gpu_config: str
    gpu_mapping: str
    topas_kind: str
    topas_raw: str
    topas_header: str | None
    topas_mhd_check: str
    topas_log: str
    body_mhd: str
    patient_shape_zyx: tuple[int, int, int]
    patient_spacing_xyz_mm: tuple[float, float, float]


CASES = (
    CaseSpec(
        name="RT07575",
        histories=12_963_817,
        setup="skew-0.065 + rot+3.5 + heat 0.5/0.9",
        gpu_mhd=(
            "out/ct/RT07575/cascade_secondary_ablation/"
            "fullplan_yz_rot_p3p5/dose.mhd"
        ),
        gpu_log=(
            "out/ct/RT07575/cascade_secondary_ablation/"
            "fullplan_yz_rot_p3p5/run.log"
        ),
        gpu_config=(
            "out/ct/RT07575/cascade_secondary_ablation/"
            "fullplan_yz_rot_p3p5/config.yaml"
        ),
        gpu_mapping="tps_90_patient_xneg",
        topas_kind="binary_header",
        topas_raw="ct/fullplan_result/RT07575/OSMK_Dtotal_full_plan.bin",
        topas_header=(
            "ct/fullplan_result/RT07575/OSMK_Dtotal_full_plan.binheader"
        ),
        topas_mhd_check="out/fullplan_result/RT07575/topas/dose.mhd",
        topas_log="ct/fullplan_result/RT07575/slurm_rt07575_full.out",
        body_mhd="out/fullplan_result/RT07575/body_mask.mhd",
        patient_shape_zyx=(35, 505, 417),
        patient_spacing_xyz_mm=(0.5, 0.5, 2.0),
    ),
    CaseSpec(
        name="RT06423",
        histories=15_108_664,
        setup="heat 0.5/0.9 + air, skew=0 rot=0",
        gpu_mhd="out/ct/cross_case_equal_history/RT06423/dose.mhd",
        gpu_log="out/ct/cross_case_equal_history/RT06423/run.log",
        gpu_config="out/ct/cross_case_equal_history/RT06423/config.yaml",
        gpu_mapping="tps_90_patient_xneg",
        topas_kind="binary_header",
        topas_raw="ct/fullplan_result/RT06423/OSMK_Dtotal_full_plan.bin",
        topas_header=(
            "ct/fullplan_result/RT06423/OSMK_Dtotal_full_plan.binheader"
        ),
        topas_mhd_check="out/fullplan_result/RT06423/topas/dose.mhd",
        topas_log="ct/fullplan_result/RT06423/slurm_rt06423_full.out",
        body_mhd="out/fullplan_result/RT06423/body_mask.mhd",
        patient_shape_zyx=(34, 440, 440),
        patient_spacing_xyz_mm=(0.5, 0.5, 2.0),
    ),
    CaseSpec(
        name="20022516",
        histories=17_717_177,
        setup="heat 0.5/0.9 + air, skew=0 rot=0",
        gpu_mhd="out/ct/cross_case_equal_history/20022516/dose.mhd",
        gpu_log="out/ct/cross_case_equal_history/20022516/run.log",
        gpu_config="out/ct/cross_case_equal_history/20022516/config.yaml",
        gpu_mapping="tps_gantry_y_patient_ypos",
        topas_kind="rtdose_dicom",
        topas_raw="ct/fullplan_result/20022516/OSMK_Dtotal_full_plan.dcm",
        topas_header=None,
        topas_mhd_check="out/fullplan_profiles/20022516/topas/dose.mhd",
        topas_log="ct/fullplan_result/20022516/slurm_lung20022516_full.out",
        body_mhd="out/fullplan_profiles/20022516/body_mask.mhd",
        patient_shape_zyx=(42, 607, 960),
        patient_spacing_xyz_mm=(0.5, 0.5, 2.0),
    ),
)


def fail(message: str) -> None:
    raise ValueError(message)


def parse_mhd(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    dims_xyz = tuple(int(value) for value in metadata["DimSize"].split())
    types = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8", "MET_UCHAR": "u1"}
    element_type = metadata.get("ElementType")
    if element_type not in types:
        fail(f"{path}: unsupported ElementType {element_type!r}")
    raw_path = path.parent / metadata["ElementDataFile"]
    values = np.fromfile(raw_path, dtype=types[element_type])
    expected = int(np.prod(dims_xyz))
    if values.size != expected:
        fail(f"{raw_path}: found {values.size} values, expected {expected}")
    return metadata, values.reshape(tuple(reversed(dims_xyz)))


def parse_binary_header(path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    text = path.read_text(encoding="ascii")
    dims: list[int] = []
    spacing: list[float] = []
    for axis in "XYZ":
        match = re.search(
            rf"^# {axis} in (\d+) bins of ([0-9.eE+-]+) (mm|cm)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            fail(f"{path}: missing {axis} bin declaration")
        dims.append(int(match.group(1)))
        width = float(match.group(2))
        spacing.append(width * (10.0 if match.group(3) == "cm" else 1.0))
    return (dims[2], dims[1], dims[0]), tuple(spacing)


def load_topas_raw(root: Path, spec: CaseSpec) -> tuple[np.ndarray, tuple[float, float, float]]:
    path = root / spec.topas_raw
    if spec.topas_kind == "binary_header":
        assert spec.topas_header is not None
        shape, spacing = parse_binary_header(root / spec.topas_header)
        count = int(np.prod(shape))
        byte_count = path.stat().st_size
        if byte_count == count * 8:
            dtype = "<f8"
        elif byte_count == count * 4:
            dtype = "<f4"
        else:
            fail(f"{path}: {byte_count} bytes is not {count} float32/float64 values")
        # GPU and converted reference artifacts are float32.  Normalize the raw
        # scorer to that precision so the historical formal arithmetic is exact.
        dose = np.fromfile(path, dtype=dtype).reshape(shape).astype(np.float32)
        return dose, spacing
    if spec.topas_kind == "rtdose_dicom":
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError("pydicom is required for the RTDOSE case") from exc
        dataset = pydicom.dcmread(str(path))
        if str(dataset.Modality) != "RTDOSE" or str(dataset.DoseUnits).upper() != "GY":
            fail(f"{path}: expected an RTDOSE scorer in Gy")
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        if not np.allclose(orientation, [1, 0, 0, 0, 1, 0], atol=1e-8):
            fail(f"{path}: unsupported ImageOrientationPatient {orientation.tolist()}")
        frame_offsets = np.asarray(dataset.GridFrameOffsetVector, dtype=np.float64)
        dz = float(np.median(np.diff(frame_offsets)))
        if not np.allclose(np.diff(frame_offsets), dz, rtol=0.0, atol=1e-6):
            fail(f"{path}: nonuniform frame spacing is unsupported")
        row_mm, column_mm = (float(value) for value in dataset.PixelSpacing)
        dose = (
            dataset.pixel_array.astype(np.float32)
            * np.float32(dataset.DoseGridScaling)
        )
        return dose, (column_mm, row_mm, dz)
    fail(f"{spec.name}: unsupported TOPAS kind {spec.topas_kind!r}")


def map_gpu_to_patient(native_zyx: np.ndarray, mapping: str) -> np.ndarray:
    if mapping == "tps_90_patient_xneg":
        # Native axes are (patient X depth, patient Z, patient Y).  X depth is
        # reversed because the beam enters from patient +X and travels toward -X.
        return np.transpose(native_zyx, (1, 2, 0))[:, :, ::-1]
    if mapping == "tps_gantry_y_patient_ypos":
        # Native axes are (patient Y depth, patient Z, patient X), with +Y depth.
        return np.transpose(native_zyx, (1, 0, 2))
    fail(f"unsupported GPU mapping {mapping!r}")


def flat_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip("'\"")
    return result


def single_positive_history(path: Path, pattern: str) -> int:
    values = {int(value) for value in re.findall(pattern, path.read_text(errors="replace"))}
    values.discard(0)
    if len(values) != 1:
        fail(f"{path}: expected one positive history count, found {sorted(values)}")
    return values.pop()


def assert_close_sequence(label: str, actual: Any, expected: Any, atol: float = 1e-7) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        fail(f"{label}: got {actual}, expected {expected}")


def evaluate(
    root: Path,
    spec: CaseSpec,
    gpu_mhd_path: Path,
    gpu_config_path: Path,
    gpu_log_path: Path,
    gpu_bundle_overridden: bool,
) -> dict[str, Any]:
    topas, raw_spacing = load_topas_raw(root, spec)
    topas_check_meta, topas_check = parse_mhd(root / spec.topas_mhd_check)
    gpu_meta, gpu_native = parse_mhd(gpu_mhd_path)
    body_meta, body = parse_mhd(root / spec.body_mhd)
    gpu = map_gpu_to_patient(gpu_native, spec.gpu_mapping)

    expected_shape = spec.patient_shape_zyx
    for label, values in (("TOPAS raw", topas), ("TOPAS MHD check", topas_check),
                          ("mapped GPU", gpu), ("BODY", body)):
        if values.shape != expected_shape:
            fail(f"{spec.name} {label}: shape {values.shape}, expected {expected_shape}")
    assert_close_sequence(f"{spec.name} raw spacing", raw_spacing,
                          spec.patient_spacing_xyz_mm)
    for label, metadata in (("TOPAS MHD", topas_check_meta), ("BODY", body_meta)):
        spacing = tuple(float(v) for v in metadata["ElementSpacing"].split())
        assert_close_sequence(f"{spec.name} {label} spacing", spacing,
                              spec.patient_spacing_xyz_mm)
    raw_mhd_max_abs_delta = float(
        np.max(np.abs(topas.astype(np.float64) - topas_check.astype(np.float64)))
    )
    if raw_mhd_max_abs_delta > 1e-7:
        fail(f"{spec.name}: raw TOPAS and MHD check differ by {raw_mhd_max_abs_delta} Gy")

    config = flat_yaml(gpu_config_path)
    gpu_histories = single_positive_history(
        gpu_log_path, r"Histories:\s*(\d+)"
    )
    topas_histories = single_positive_history(
        root / spec.topas_log, r"Total number of histories:\s*(\d+)"
    )
    config_histories = int(config["number_of_histories"])
    if {gpu_histories, topas_histories, config_histories} != {spec.histories}:
        fail(
            f"{spec.name}: unequal histories: manifest={spec.histories}, "
            f"config={config_histories}, GPU={gpu_histories}, TOPAS={topas_histories}"
        )
    expected_geometry_mode = (
        "tps_90" if spec.gpu_mapping == "tps_90_patient_xneg" else "tps_gantry_y"
    )
    if config.get("spots_geometry_mode") != expected_geometry_mode:
        fail(
            f"{spec.name}: mapping requires spots_geometry_mode={expected_geometry_mode}, "
            f"got {config.get('spots_geometry_mode')!r}"
        )

    body_selection = body > 0.5
    if not np.any(body_selection):
        fail(f"{spec.name}: BODY mask is empty")
    dmax = float(np.max(topas[body_selection]))
    selected = body_selection & (topas >= 0.10 * dmax)
    reference = topas[selected]
    evaluated = gpu[selected]
    delta = evaluated - reference
    global_tolerance = 0.03 * dmax
    local_tolerance = 0.03 * np.abs(reference)

    return {
        "case": spec.name,
        "histories": spec.histories,
        "setup": (
            "explicit GPU iteration override; see config"
            if gpu_bundle_overridden
            else spec.setup
        ),
        "gpu_mapping": spec.gpu_mapping,
        "selected_voxels": int(np.count_nonzero(selected)),
        "body_voxels": int(np.count_nonzero(body_selection)),
        "topas_dmax_Gy": dmax,
        "nrmse_percent": float(100.0 * np.sqrt(np.mean(delta * delta)) / dmax),
        "global_3pct_0mm_pass_percent": float(
            100.0 * np.mean(np.abs(delta) <= global_tolerance)
        ),
        "local_3pct_0mm_pass_percent": float(
            100.0 * np.mean(np.abs(delta) <= local_tolerance)
        ),
        "selected_integral_gpu_over_topas": float(
            np.sum(evaluated) / np.sum(reference)
        ),
        "topas_raw_vs_mhd_max_abs_delta_Gy": raw_mhd_max_abs_delta,
        "inputs": {
            "gpu_mhd": str(gpu_mhd_path),
            "gpu_mhd_manifest_default": spec.gpu_mhd,
            "gpu_config": str(gpu_config_path),
            "gpu_config_manifest_default": spec.gpu_config,
            "gpu_log": str(gpu_log_path),
            "gpu_log_manifest_default": spec.gpu_log,
            "gpu_bundle_overridden": gpu_bundle_overridden,
            "topas_raw": spec.topas_raw,
            "topas_header": spec.topas_header,
            "topas_log": spec.topas_log,
            "topas_mhd_integrity_check": spec.topas_mhd_check,
            "body_mhd": spec.body_mhd,
        },
        "config": {
            key: config.get(key)
            for key in (
                "physics_profile",
                "dose_output_scale",
                "nuclear_residual_heat_mfp_mm",
                "nuclear_residual_heat_scale",
                "spots_enable_upstream_air_energy_loss",
                "spots_geometry_mode",
                "spots_patient_rot_z_deg",
                "spots_lateral_yz_skew",
                "spots_lateral_yz_rotation_deg",
                "spots_lateral_yz_skew_auto_pivot",
                "spots_lateral_yz_skew_pivot_mm",
            )
        },
        "history_validation": {
            "manifest": spec.histories,
            "gpu_config": config_histories,
            "gpu_log": gpu_histories,
            "topas_log": topas_histories,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-case equal-history GPU–TOPAS dose comparison",
        "",
        "Protocol: BODY ∩ raw TOPAS ≥10% of BODY TOPAS Dmax; identical-voxel "
        "local/global 3%/0 mm; no fitted scale or spatial transform.",
        "",
        "| Case | Histories | Voxels | Local 3%/0mm | Global 3%/0mm | NRMSE | E/R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["cases"]:
        lines.append(
            f"| {item['case']} | {item['histories']:,} | "
            f"{item['selected_voxels']:,} | "
            f"{item['local_3pct_0mm_pass_percent']:.4f}% | "
            f"{item['global_3pct_0mm_pass_percent']:.4f}% | "
            f"{item['nrmse_percent']:.6f}% | "
            f"{item['selected_integral_gpu_over_topas']:.6f} |"
        )
    lines.extend(("", "TOPAS dose values are read from each raw scorer; the MHD copy is "
                  "checked only for conversion integrity.", ""))
    return "\n".join(lines)


def parse_case_path_overrides(
    option: str,
    values: list[str] | None,
    root: Path,
    suffixes: tuple[str, ...],
) -> dict[str, Path]:
    known_cases = {spec.name for spec in CASES}
    overrides: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            fail(f"{option} must be CASE=PATH, got {value!r}")
        case_name, raw_path = value.split("=", 1)
        case_name = case_name.strip()
        raw_path = raw_path.strip()
        if case_name not in known_cases:
            fail(
                f"{option} has unknown case {case_name!r}; "
                f"expected one of {sorted(known_cases)}"
            )
        if case_name in overrides:
            fail(f"{option} was specified more than once for {case_name}")
        if not raw_path:
            fail(f"{option} path is empty for {case_name}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.suffix.lower() not in suffixes:
            fail(
                f"{option} for {case_name} must have suffix "
                f"{' or '.join(suffixes)}: {path}"
            )
        if not path.is_file():
            fail(f"{option} for {case_name} does not exist: {path}")
        overrides[case_name] = path
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", choices=[spec.name for spec in CASES],
        help="evaluate only this case (repeatable); default: all formal cases",
    )
    parser.add_argument(
        "--gpu-dose", action="append", metavar="CASE=PATH",
        help=(
            "GPU dose MHD override for CASE (repeatable); requires matching "
            "--gpu-config and --gpu-log from the same directory"
        ),
    )
    parser.add_argument(
        "--gpu-config", action="append", metavar="CASE=PATH",
        help="GPU YAML config paired with --gpu-dose (repeatable)",
    )
    parser.add_argument(
        "--gpu-log", action="append", metavar="CASE=PATH",
        help="GPU run log paired with --gpu-dose (repeatable)",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    requested = set(args.case or [spec.name for spec in CASES])
    selected_specs = [spec for spec in CASES if spec.name in requested]
    gpu_dose_overrides = parse_case_path_overrides(
        "--gpu-dose", args.gpu_dose, root, (".mhd",)
    )
    gpu_config_overrides = parse_case_path_overrides(
        "--gpu-config", args.gpu_config, root, (".yaml", ".yml")
    )
    gpu_log_overrides = parse_case_path_overrides(
        "--gpu-log", args.gpu_log, root, (".log",)
    )
    override_cases = (
        set(gpu_dose_overrides) | set(gpu_config_overrides) | set(gpu_log_overrides)
    )
    for case_name in sorted(override_cases):
        present = {
            "--gpu-dose": case_name in gpu_dose_overrides,
            "--gpu-config": case_name in gpu_config_overrides,
            "--gpu-log": case_name in gpu_log_overrides,
        }
        if not all(present.values()):
            missing = [option for option, exists in present.items() if not exists]
            fail(f"incomplete GPU override for {case_name}; missing {', '.join(missing)}")
        parents = {
            gpu_dose_overrides[case_name].parent,
            gpu_config_overrides[case_name].parent,
            gpu_log_overrides[case_name].parent,
        }
        if len(parents) != 1:
            fail(
                f"GPU dose/config/log override for {case_name} must come from "
                f"one directory, got {sorted(str(path) for path in parents)}"
            )
    unused_overrides = override_cases - requested
    if unused_overrides:
        fail(
            "GPU override supplied for case not selected by --case: "
            + ", ".join(sorted(unused_overrides))
        )
    report = {
        "protocol": {
            "selection": "BODY > 0.5 and raw TOPAS >= 10% of TOPAS Dmax in BODY",
            "comparison": "identical voxel; no scale or spatial fit",
            "local_pass": "abs(GPU - TOPAS) <= 3% * abs(TOPAS)",
            "global_pass": "abs(GPU - TOPAS) <= 3% * TOPAS Dmax in BODY",
        },
        "case_manifest": [asdict(spec) for spec in selected_specs],
        "cases": [
            evaluate(
                root,
                spec,
                gpu_dose_overrides.get(spec.name, (root / spec.gpu_mhd).resolve()),
                gpu_config_overrides.get(
                    spec.name, (root / spec.gpu_config).resolve()
                ),
                gpu_log_overrides.get(spec.name, (root / spec.gpu_log).resolve()),
                spec.name in override_cases,
            )
            for spec in selected_specs
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
