#!/usr/bin/env python3
"""Audit a completed fluctuation campaign and compile its runtime package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


CAMPAIGN_SCHEMA = "maigo-energy-loss-fluctuation-campaign-v1"
RUN_SCHEMA = "maigo-energy-loss-fluctuation-run-v1"
RESULT_SCHEMA = "maigo-energy-loss-fluctuation-run-result-v1"
POINT_SCHEMA = "maigo-energy-loss-fluctuation-point-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return value


def resolve_inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes campaign root: {value}")
    return path


def require_digest(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def audit_campaign(
    campaign_root: Path,
    expected_point_count: int | None,
    expected_probability_count: int,
    maximum_inverse_cdf_compression_mean_error: float,
    verify_raw_source_hashes: bool,
) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = campaign_root / "campaign_manifest.json"
    campaign = load_json(manifest_path, "campaign manifest")
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"unexpected campaign schema: {manifest_path}")
    points = campaign.get("points")
    grid = campaign.get("grid")
    if not isinstance(points, list) or not isinstance(grid, dict):
        raise ValueError("campaign points/grid are missing")
    if len(points) != grid.get("point_count"):
        raise ValueError("campaign point count differs from grid point_count")
    if expected_point_count is not None and len(points) != expected_point_count:
        raise ValueError(
            f"campaign contains {len(points)} points; expected {expected_point_count}"
        )

    campaign_identity = campaign.get("projectile")
    campaign_material = campaign.get("material")
    histories_per_point = campaign.get("histories_per_point")
    if not isinstance(campaign_identity, dict):
        raise ValueError("campaign projectile identity is missing")
    if not isinstance(campaign_material, str) or not campaign_material:
        raise ValueError("campaign material is missing")
    if not isinstance(histories_per_point, int) or histories_per_point <= 0:
        raise ValueError("campaign histories_per_point is invalid")
    probability_grid: list[float] | None = None
    point_paths: list[Path] = []
    coordinates: set[tuple[float, float]] = set()
    maximum_compression_error = 0.0
    source_bytes = 0

    for ordinal, reference in enumerate(points):
        if not isinstance(reference, dict):
            raise ValueError(f"campaign point {ordinal} is not an object")
        label = reference.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"campaign point {ordinal} has no label")
        manifest_reference = reference.get("run_manifest")
        if not isinstance(manifest_reference, dict):
            raise ValueError(f"{label}: run manifest reference is missing")
        run_manifest_path = resolve_inside(
            campaign_root, manifest_reference.get("path"), f"{label} run manifest"
        )
        require_digest(
            run_manifest_path, manifest_reference.get("sha256"), f"{label} run manifest"
        )
        run = load_json(run_manifest_path, f"{label} run manifest")
        if run.get("schema") != RUN_SCHEMA or run.get("label") != label:
            raise ValueError(f"{label}: invalid run manifest identity")
        if (
            run.get("projectile") != campaign_identity
            or run.get("material") != campaign_material
            or run.get("histories") != histories_per_point
        ):
            raise ValueError(f"{label}: run identity or histories differ from campaign")
        coordinate = (run.get("energy_MeV_per_u"), run.get("areal_density_g_per_cm2"))
        if coordinate != (
            reference.get("energy_MeV_per_u"),
            reference.get("areal_density_g_per_cm2"),
        ):
            raise ValueError(f"{label}: run coordinates differ from campaign point")
        if coordinate in coordinates:
            raise ValueError(f"{label}: duplicate campaign coordinate {coordinate}")
        coordinates.add(coordinate)

        files = run.get("files")
        if not isinstance(files, dict) or not isinstance(files.get("config"), dict):
            raise ValueError(f"{label}: run file references are missing")
        config_path = resolve_inside(
            campaign_root, files["config"].get("path"), f"{label} config"
        )
        require_digest(config_path, files["config"].get("sha256"), f"{label} config")
        run_dir = config_path.parent
        result_path = run_dir / "run_result.json"
        point_path = run_dir / "fluctuation_point.json"
        result = load_json(result_path, f"{label} run result")
        point = load_json(point_path, f"{label} fluctuation point")
        if result.get("schema") != RESULT_SCHEMA or result.get("label") != label:
            raise ValueError(f"{label}: invalid run result identity")
        point_reference = result.get("prepared_point")
        if not isinstance(point_reference, dict):
            raise ValueError(f"{label}: prepared-point reference is missing")
        require_digest(point_path, point_reference.get("sha256"), f"{label} point")
        if point.get("schema") != POINT_SCHEMA:
            raise ValueError(f"{label}: invalid point schema")
        if (
            point.get("projectile")
            != {"Z": campaign_identity.get("Z"), "A": campaign_identity.get("A")}
            or point.get("material") != campaign_material
            or point.get("histories") != histories_per_point
            or (point.get("energy_MeV_per_u"), point.get("areal_density_g_per_cm2"))
            != coordinate
        ):
            raise ValueError(f"{label}: prepared point differs from campaign/run identity")

        inverse_cdf = point.get("inverse_cdf")
        if not isinstance(inverse_cdf, dict):
            raise ValueError(f"{label}: inverse CDF is missing")
        probabilities = inverse_cdf.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != expected_probability_count:
            raise ValueError(
                f"{label}: expected {expected_probability_count} probabilities"
            )
        if probability_grid is None:
            probability_grid = probabilities
        elif probabilities != probability_grid:
            raise ValueError(f"{label}: probability grid differs from other points")
        integral_before = inverse_cdf.get("piecewise_linear_mean_before_normalization")
        if not isinstance(integral_before, (int, float)) or not math.isfinite(integral_before):
            raise ValueError(f"{label}: invalid pre-normalization inverse-CDF mean")
        empirical_type7_mean = inverse_cdf.get("empirical_type7_mean_exact")
        compression_error = inverse_cdf.get(
            "piecewise_linear_compression_mean_relative_error"
        )
        normalization_factor = inverse_cdf.get("unit_mean_normalization_factor")
        if (
            not isinstance(empirical_type7_mean, (int, float))
            or not math.isfinite(empirical_type7_mean)
            or empirical_type7_mean <= 0.0
            or not isinstance(compression_error, (int, float))
            or not math.isfinite(compression_error)
            or not isinstance(normalization_factor, (int, float))
            or not math.isfinite(normalization_factor)
        ):
            raise ValueError(f"{label}: inverse-CDF compression diagnostics are invalid")
        reconstructed_error = float(integral_before) / float(empirical_type7_mean) - 1.0
        if abs(float(compression_error) - reconstructed_error) > 1.0e-12:
            raise ValueError(f"{label}: inverse-CDF compression error is inconsistent")
        if abs(float(normalization_factor) * float(integral_before) - 1.0) > 1.0e-12:
            raise ValueError(f"{label}: inverse-CDF normalization factor is inconsistent")
        error = abs(reconstructed_error)
        if error >= maximum_inverse_cdf_compression_mean_error:
            raise ValueError(
                f"{label}: inverse-CDF compression mean error {error:.9g} is not below "
                f"{maximum_inverse_cdf_compression_mean_error:.9g}"
            )
        maximum_compression_error = max(maximum_compression_error, error)

        sources = point.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{label}: raw source references are missing")
        for kind in ("header", "phsp"):
            source = sources.get(kind)
            if not isinstance(source, dict):
                raise ValueError(f"{label}: {kind} source reference is missing")
            source_path = Path(source.get("path", "")).resolve()
            if not source_path.is_relative_to(run_dir.resolve()):
                raise ValueError(f"{label}: {kind} source is outside its run directory")
            if verify_raw_source_hashes:
                require_digest(source_path, source.get("sha256"), f"{label} {kind}")
                source_bytes += source_path.stat().st_size
        point_paths.append(point_path)

    audit = {
        "schema": "maigo-energy-loss-fluctuation-finalization-audit-v1",
        "campaign": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "projectile": campaign_identity,
        "material": campaign_material,
        "points": len(point_paths),
        "histories_per_point": histories_per_point,
        "total_histories": histories_per_point * len(point_paths),
        "probability_count": len(probability_grid or []),
        "maximum_inverse_cdf_compression_mean_relative_error": (
            maximum_compression_error
        ),
        "maximum_allowed_inverse_cdf_compression_mean_relative_error": (
            maximum_inverse_cdf_compression_mean_error
        ),
        "raw_source_hashes_verified": verify_raw_source_hashes,
        "raw_source_bytes_verified": source_bytes,
    }
    return point_paths, audit


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument(
        "--compiler",
        type=Path,
        default=here.parent / "package_tools/compile_energy_loss_fluctuation_grid.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-point-count", type=int)
    parser.add_argument("--expected-probability-count", type=int, default=335)
    parser.add_argument(
        "--maximum-inverse-cdf-compression-mean-error", type=float, default=0.001
    )
    parser.add_argument(
        "--verify-raw-source-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        campaign_root = args.campaign_dir.resolve()
        compiler = args.compiler.resolve()
        outputs = [args.output.resolve(), args.output_metadata.resolve(), args.audit_output.resolve()]
        if any(path.exists() for path in outputs):
            raise ValueError(f"refusing to overwrite finalization output: {outputs}")
        if args.expected_point_count is not None and args.expected_point_count <= 0:
            raise ValueError("expected point count must be positive")
        if args.expected_probability_count < 3:
            raise ValueError("expected probability count must be at least three")
        if not (0.0 < args.maximum_inverse_cdf_compression_mean_error < 1.0):
            raise ValueError(
                "maximum inverse-CDF compression mean error must be in (0,1)"
            )
        if not compiler.is_file():
            raise FileNotFoundError(f"compiler is missing: {compiler}")
        point_paths, audit = audit_campaign(
            campaign_root,
            args.expected_point_count,
            args.expected_probability_count,
            args.maximum_inverse_cdf_compression_mean_error,
            args.verify_raw_source_hashes,
        )
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(compiler), "--points",
                *(str(path) for path in point_paths),
                "--output", str(args.output.resolve()),
                "--output-metadata", str(args.output_metadata.resolve()),
            ],
            check=True,
        )
        metadata = load_json(args.output_metadata.resolve(), "compiled metadata")
        if metadata.get("grid", {}).get("point_count") != len(point_paths):
            raise ValueError("compiled metadata point count differs from campaign")
        audit["runtime_package"] = {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output.resolve()),
        }
        audit["runtime_metadata"] = {
            "path": str(args.output_metadata.resolve()),
            "sha256": sha256(args.output_metadata.resolve()),
        }
        args.audit_output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.resolve().write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"finalized fluctuation campaign: {len(point_paths)} points, "
            "max inverse-CDF compression mean error="
            f"{audit['maximum_inverse_cdf_compression_mean_relative_error']:.9g}"
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"runtime package compiler failed: {error.returncode}") from error


if __name__ == "__main__":
    main()
