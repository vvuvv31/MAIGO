#!/usr/bin/env python3
"""Run or collect a generated thin-slab energy-loss fluctuation campaign."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


CAMPAIGN_SCHEMA = "maigo-energy-loss-fluctuation-campaign-v1"
RUN_SCHEMA = "maigo-energy-loss-fluctuation-run-v1"
RESULT_SCHEMA = "maigo-energy-loss-fluctuation-run-result-v1"
TOPAS_PATTERN = re.compile(r"Welcome to TOPAS.*\(Version\s+([^\)]+)\)")
GEANT4_PATTERN = re.compile(
    r"Geant4 version Name:\s*geant4-(\d+)-(\d+)(?:-patch-(\d+))?"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def resolve_inside(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes campaign directory: {relative}")
    return path


def canonical_geant4_version(match: re.Match[str]) -> str:
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return f"{major}.{minor}.{patch}"


def parse_versions(log_text: str, label: str) -> dict[str, str]:
    topas = TOPAS_PATTERN.search(log_text)
    geant4 = GEANT4_PATTERN.search(log_text)
    if topas is None or geant4 is None:
        raise ValueError(f"{label}: runtime log does not identify TOPAS and Geant4")
    return {
        "TOPAS": topas.group(1).strip(),
        "Geant4": canonical_geant4_version(geant4),
    }


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def validate_digest(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def selected_points(
    campaign: dict[str, Any], requested_labels: list[str] | None
) -> list[dict[str, Any]]:
    points = campaign.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("campaign contains no points")
    if any(not isinstance(point, dict) for point in points):
        raise ValueError("campaign point entries must be objects")
    labels = [point.get("label") for point in points]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("campaign point label is missing")
    if len(set(labels)) != len(labels):
        raise ValueError("campaign point labels are not unique")
    if requested_labels is None:
        return points
    unknown = sorted(set(requested_labels) - set(labels))
    if unknown:
        raise ValueError(f"unknown campaign point labels: {unknown}")
    requested = set(requested_labels)
    return [point for point in points if point["label"] in requested]


def validate_run_manifest(
    campaign_root: Path,
    campaign: dict[str, Any],
    point: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path]:
    manifest_ref = require_mapping(
        point.get("run_manifest"), f"point {point.get('label')}: run_manifest"
    )
    manifest_path = resolve_inside(
        campaign_root,
        manifest_ref.get("path"),
        f"point {point.get('label')} run manifest",
    )
    validate_digest(
        manifest_path,
        manifest_ref.get("sha256"),
        f"point {point.get('label')} run manifest",
    )
    run = load_json(manifest_path, "run manifest")
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"unexpected run manifest schema: {manifest_path}")
    for field in ("projectile", "material", "physics"):
        if run.get(field) != campaign.get(field):
            raise ValueError(f"{manifest_path}: {field} differs from campaign manifest")
    if (
        run.get("label") != point.get("label")
        or run.get("energy_MeV_per_u") != point.get("energy_MeV_per_u")
        or run.get("areal_density_g_per_cm2")
        != point.get("areal_density_g_per_cm2")
        or run.get("seed") != point.get("seed")
    ):
        raise ValueError(f"{manifest_path}: point coordinates or seed differ from campaign")
    files = require_mapping(run.get("files"), f"{manifest_path}: files")
    config_ref = require_mapping(files.get("config"), f"{manifest_path}: config")
    config_path = resolve_inside(
        campaign_root, config_ref.get("path"), f"{manifest_path}: config"
    )
    validate_digest(config_path, config_ref.get("sha256"), "TOPAS config")
    return manifest_path, run, config_path


def collect_point(
    campaign_root: Path,
    run: dict[str, Any],
    config_path: Path,
    topas_binary: Path,
    tools_dir: Path,
) -> Path:
    run_dir = config_path.parent
    files = require_mapping(run["files"], f"{run['label']}: files")
    log_path = run_dir / files["runtime_log"]
    if not log_path.is_file():
        raise FileNotFoundError(f"runtime log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "TOPAS run sequence complete." not in log_text:
        raise ValueError(f"{run['label']}: TOPAS completion marker is missing")
    if not re.search(r"Fluctuations of dE/dx are enabled\s+1", log_text):
        raise ValueError(f"{run['label']}: Geant4 dE/dx fluctuations were not enabled")
    actual_versions = parse_versions(log_text, run["label"])
    target_versions = require_mapping(
        require_mapping(run["physics"], f"{run['label']}: physics").get(
            "target_versions"
        ),
        f"{run['label']}: target versions",
    )
    if actual_versions != target_versions:
        raise ValueError(
            f"{run['label']}: runtime versions {actual_versions} do not match "
            f"targets {target_versions}"
        )

    output_stem = run["output_stem"]
    if not isinstance(output_stem, str) or not output_stem:
        raise ValueError(f"{run['label']}: output stem is missing")
    header_path = run_dir / files["expected_header"]
    phsp_path = run_dir / files["expected_phsp"]
    expected_header = run_dir / f"{output_stem}.header"
    expected_phsp = run_dir / f"{output_stem}.phsp"
    if header_path != expected_header or phsp_path != expected_phsp:
        raise ValueError(f"{run['label']}: expected scorer names do not match output stem")
    point_output = run_dir / "fluctuation_point.json"
    if point_output.exists():
        raise ValueError(f"refusing to overwrite prepared point: {point_output}")
    prepare = [
        sys.executable,
        str(tools_dir / "prepare_topas_fluctuation.py"),
        "--input", str(run_dir / output_stem),
        "--output", str(point_output),
        "--projectile-z", str(run["projectile"]["Z"]),
        "--projectile-a", str(run["projectile"]["A"]),
        "--material", str(run["material"]),
        "--energy-mev-per-u", str(run["energy_MeV_per_u"]),
        "--areal-density-g-per-cm2", str(run["areal_density_g_per_cm2"]),
        "--expected-histories", str(run["histories"]),
    ]
    subprocess.run(prepare, cwd=run_dir, check=True)
    result = {
        "schema": RESULT_SCHEMA,
        "label": run["label"],
        "actual_versions": actual_versions,
        "target_versions": target_versions,
        "topas_binary": {
            "path": str(topas_binary),
            "sha256": sha256(topas_binary),
        },
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "runtime_log": {"path": str(log_path), "sha256": sha256(log_path)},
        "prepared_point": {
            "path": str(point_output),
            "sha256": sha256(point_output),
        },
    }
    result_path = run_dir / "run_result.json"
    if result_path.exists():
        point_output.unlink()
        raise ValueError(f"refusing to overwrite run result: {result_path}")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return point_output


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run TOPAS then collect")
    mode.add_argument(
        "--collect-only", action="store_true", help="validate existing outputs and collect"
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--topas-bin", type=Path, required=True)
    parser.add_argument("--g4-data-dir", type=Path)
    parser.add_argument("--tools-dir", type=Path, default=here.parent / "package_tools")
    parser.add_argument("--point", action="append", dest="points")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        campaign_root = args.campaign_dir.resolve()
        campaign_path = campaign_root / "campaign_manifest.json"
        campaign = load_json(campaign_path, "campaign manifest")
        if campaign.get("schema") != CAMPAIGN_SCHEMA:
            raise ValueError(f"unexpected campaign schema: {campaign_path}")
        topas_binary = args.topas_bin.resolve()
        if not topas_binary.is_file() or not os.access(topas_binary, os.X_OK):
            raise ValueError(f"TOPAS binary is not executable: {topas_binary}")
        tools_dir = args.tools_dir.resolve()
        if not (tools_dir / "prepare_topas_fluctuation.py").is_file():
            raise ValueError(f"fluctuation preparation tool is missing: {tools_dir}")
        if args.run:
            if args.g4_data_dir is None or not args.g4_data_dir.resolve().is_dir():
                raise ValueError("--run requires an existing --g4-data-dir")
            g4_data_dir = args.g4_data_dir.resolve()
        else:
            g4_data_dir = None

        points = selected_points(campaign, args.points)
        prepared: list[Path] = []
        lock_dir = campaign_root / ".fluctuation_campaign_locks"
        lock_dir.mkdir(exist_ok=True)
        with ExitStack() as locks:
            for point in points:
                label = point["label"]
                lock_name = hashlib.sha256(label.encode("utf-8")).hexdigest() + ".lock"
                lock_path = lock_dir / lock_name
                lock = locks.enter_context(lock_path.open("w", encoding="utf-8"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(
                        f"another process owns point lock for {label}: {lock_path}"
                    ) from error
            # Validate every selected manifest and config before starting any
            # Monte Carlo. A late hash failure must not leave a partially run
            # campaign merely because an earlier point happened to be valid.
            validated_runs = [
                validate_run_manifest(
                    campaign_root, campaign, point
                )
                for point in points
            ]
            for _, run, config_path in validated_runs:
                run_dir = config_path.parent
                log_path = run_dir / run["files"]["runtime_log"]
                if args.run:
                    forbidden = [
                        log_path,
                        run_dir / run["files"]["expected_header"],
                        run_dir / run["files"]["expected_phsp"],
                        run_dir / "fluctuation_point.json",
                        run_dir / "run_result.json",
                    ]
                    existing = [path for path in forbidden if path.exists()]
                    if existing:
                        raise ValueError(
                            f"refusing to overwrite existing outputs for {run['label']}: "
                            f"{existing}"
                        )
                    environment = os.environ.copy()
                    environment["TOPAS_G4_DATA_DIR"] = str(g4_data_dir)
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            [str(topas_binary), config_path.name],
                            cwd=run_dir,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    if completed.returncode != 0:
                        raise ValueError(
                            f"TOPAS failed for {run['label']} with exit code "
                            f"{completed.returncode}; see {log_path}"
                        )
                prepared.append(
                    collect_point(
                        campaign_root, run, config_path, topas_binary, tools_dir
                    )
                )
                print(f"collected {run['label']}: {prepared[-1]}")
        print(f"campaign points prepared: {len(prepared)}")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"point preparation failed with exit code {error.returncode}"
        ) from error


if __name__ == "__main__":
    main()
