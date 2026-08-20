#!/usr/bin/env python3
"""Strictly merge prepared elastic runs and optionally compile one ELPKG."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"Missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_xs(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["energy_MeV_per_u", "water_macroscopic_cross_section_per_mm"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def source_path(metadata_path: Path, entry: object, label: str) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise SystemExit(f"{metadata_path}: missing files.{label}.path")
    recorded = Path(entry["path"])
    candidates = (recorded, metadata_path.parent / recorded.name)
    for candidate in candidates:
        if candidate.is_file():
            expected = entry.get("sha256")
            if isinstance(expected, str) and sha256(candidate) != expected:
                raise SystemExit(f"{metadata_path}: SHA-256 mismatch for {label}")
            return candidate
    raise SystemExit(f"{metadata_path}: cannot locate {label} CSV ({recorded})")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-metadata", type=Path, action="append", required=True,
                        help="prepared metadata JSON; repeat for each independent run")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--energy-min-mevu", type=float, default=0.0)
    parser.add_argument("--energy-max-mevu", type=float, default=250.0,
                        help="lower edge of the final included 1 MeV/u bin")
    parser.add_argument("--min-events-per-bin", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="compile only if coverage passes")
    parser.add_argument("--compiler", type=Path, default=here.parent / "package_tools" / "compile_elastic_package.py")
    args = parser.parse_args()
    if not math.isfinite(args.energy_min_mevu) or not math.isfinite(args.energy_max_mevu):
        parser.error("energy bounds must be finite")
    span = args.energy_max_mevu - args.energy_min_mevu
    if args.energy_min_mevu < 0.0 or span < 0.0 or abs(span - round(span)) > 1.0e-9:
        parser.error("energy min/max must define integer-aligned 1 MeV/u bins")
    if args.min_events_per_bin <= 0:
        parser.error("min-events-per-bin must be positive")
    return args


def main() -> None:
    args = parse_args()
    bin_count = int(round(args.energy_max_mevu - args.energy_min_mevu)) + 1
    upper_exclusive = args.energy_max_mevu + 1.0
    output_prefix = args.output_prefix
    interactions_output = Path(f"{output_prefix}.interactions.csv.gz")
    products_output = Path(f"{output_prefix}.products.csv.gz")
    metadata_output = Path(f"{output_prefix}.metadata.json")
    report_output = Path(f"{output_prefix}.coverage.json")
    package_output = Path(f"{output_prefix}.bin")
    sidecar_output = Path(f"{output_prefix}.compiled.json")
    xs_output = Path(f"{output_prefix}.elastic_xs.csv")
    xs_sidecar_output = Path(f"{output_prefix}.elastic_xs.metadata.json")

    identity: tuple[int, int, str, str] | None = None
    interaction_fields: list[str] | None = None
    product_fields: list[str] | None = None
    merged_interactions: list[dict[str, str]] = []
    merged_products: list[dict[str, str]] = []
    sources: list[dict[str, object]] = []
    extraction_modes: set[str] = set()
    excluded_below_cutoff = 0

    for metadata_path in args.input_metadata:
        metadata_path = metadata_path.resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        projectile = metadata.get("projectile")
        if not isinstance(projectile, dict):
            raise SystemExit(f"{metadata_path}: missing projectile identity")
        current = (projectile.get("Z"), projectile.get("A"), metadata.get("material"), metadata.get("physics_model"))
        if not isinstance(current[0], int) or not isinstance(current[1], int) or not all(isinstance(value, str) and value for value in current[2:]):
            raise SystemExit(f"{metadata_path}: invalid projectile/material/physics identity")
        if identity is None:
            identity = current  # type: ignore[assignment]
        elif current != identity:
            raise SystemExit(f"{metadata_path}: campaign identity differs from the first input")
        provenance = metadata.get("provenance")
        processes = provenance.get("processes") if isinstance(provenance, dict) else None
        if processes != ["hadElastic"]:
            raise SystemExit(f"{metadata_path}: expected only hadElastic, found {processes!r}")
        extraction_mode = provenance.get("extraction_mode") if isinstance(provenance, dict) else None
        continuous_em_loss = provenance.get("continuous_em_loss_enabled") if isinstance(provenance, dict) else None
        sampling_purpose = provenance.get("sampling_purpose") if isinstance(provenance, dict) else None
        provenance_source = "prepared_metadata"
        if not isinstance(extraction_mode, str) or not isinstance(continuous_em_loss, bool):
            manifest_path = metadata_path.parent / "campaign_run.json"
            if not manifest_path.is_file():
                raise SystemExit(f"{metadata_path}: missing explicit extraction provenance and campaign manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            modules = manifest.get("physics_modules")
            if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
                raise SystemExit(f"{manifest_path}: invalid physics_modules")
            continuous_em_loss = any(item.startswith("g4em-") for item in modules)
            extraction_mode = "transport-degrading" if continuous_em_loss else "fixed-energy-package-sampling"
            sampling_purpose = (
                "elastic-package-transport-degrading-extraction"
                if continuous_em_loss else "elastic-package-fixed-energy-sampling-not-dose-reference"
            )
            provenance_source = "campaign_manifest_inference"
        if not isinstance(sampling_purpose, str) or not sampling_purpose:
            raise SystemExit(f"{metadata_path}: missing sampling_purpose")
        extraction_modes.add(extraction_mode)

        files = metadata.get("files")
        if not isinstance(files, dict):
            raise SystemExit(f"{metadata_path}: missing files metadata")
        source_interactions = source_path(metadata_path, files.get("interactions"), "interactions")
        source_products = source_path(metadata_path, files.get("products"), "products")
        i_fields, i_rows = read_csv(source_interactions)
        p_fields, p_rows = read_csv(source_products)
        if interaction_fields is None:
            interaction_fields, product_fields = i_fields, p_fields
        elif i_fields != interaction_fields or p_fields != product_fields:
            raise SystemExit(f"{metadata_path}: prepared CSV schemas differ")

        seen_ids: set[int] = set()
        expected_offset = 0
        for row in i_rows:
            old_id = int(row["interaction_id"])
            if old_id in seen_ids:
                raise SystemExit(f"{metadata_path}: duplicate interaction ID {old_id}")
            seen_ids.add(old_id)
            if row.get("process_name") != "hadElastic":
                raise SystemExit(f"{metadata_path}: interaction {old_id} is not hadElastic")
            offset = int(row["product_offset_zero_based"])
            count = int(row["product_count"])
            if offset != expected_offset or count < 0 or offset + count > len(p_rows):
                raise SystemExit(f"{metadata_path}: invalid product range for interaction {old_id}")
            members = p_rows[offset:offset + count]
            if any(int(product["interaction_id"]) != old_id for product in members):
                raise SystemExit(f"{metadata_path}: product range ownership mismatch for interaction {old_id}")
            for expected_index, product in enumerate(members, 1):
                if int(product["product_index"]) != expected_index:
                    raise SystemExit(f"{metadata_path}: non-contiguous product index for interaction {old_id}")
            expected_offset += len(members)
            energy = float(row["incident_energy_MeV_per_u"])
            if not math.isfinite(energy):
                raise SystemExit(f"{metadata_path}: non-finite interaction energy")
            if energy < args.energy_min_mevu:
                excluded_below_cutoff += 1
                continue
            if energy >= upper_exclusive:
                raise SystemExit(f"{metadata_path}: interaction energy {energy:g} lies above formal range")
            try:
                elastic_xs = float(row["macroscopic_elastic_per_mm"])
            except (KeyError, ValueError) as error:
                raise SystemExit(
                    f"{metadata_path}: interaction {old_id} has invalid macroscopic_elastic_per_mm"
                ) from error
            if not math.isfinite(elastic_xs) or elastic_xs < 0.0:
                raise SystemExit(
                    f"{metadata_path}: interaction {old_id} has non-finite or negative elastic XS"
                )
            new_id = len(merged_interactions) + 1
            new_row = dict(row)
            new_row["interaction_id"] = str(new_id)
            new_row["product_offset_zero_based"] = str(len(merged_products))
            merged_interactions.append(new_row)
            for product in members:
                new_product = dict(product)
                new_product["interaction_id"] = str(new_id)
                merged_products.append(new_product)
        if expected_offset != len(p_rows):
            raise SystemExit(f"{metadata_path}: product offsets do not cover the table")
        sources.append({"metadata": str(metadata_path), "sha256": sha256(metadata_path),
                        "interactions": len(i_rows), "products": len(p_rows),
                        "extraction_mode": extraction_mode,
                        "continuous_em_loss_enabled": continuous_em_loss,
                        "sampling_purpose": sampling_purpose,
                        "extraction_provenance_source": provenance_source})

    if identity is None or interaction_fields is None or product_fields is None:
        raise SystemExit("No campaign inputs")
    counts = [0] * bin_count
    xs_values: list[list[float]] = [[] for _ in range(bin_count)]
    for row in merged_interactions:
        index = math.floor(float(row["incident_energy_MeV_per_u"]) - args.energy_min_mevu)
        counts[index] += 1
        xs_values[index].append(float(row["macroscopic_elastic_per_mm"]))
    bins = []
    runtime_xs_rows: list[dict[str, object]] = []
    for index, count in enumerate(counts):
        values = xs_values[index]
        entry: dict[str, object] = {
            "index": index,
            "minimum_MeV_per_u": args.energy_min_mevu + index,
            "maximum_MeV_per_u": args.energy_min_mevu + index + 1.0,
            "event_count": count,
            "elastic_xs_sample_count": len(values),
        }
        if values:
            minimum, maximum = min(values), max(values)
            median = statistics.median(values)
            entry["elastic_xs_per_mm"] = {
                "median": median, "minimum": minimum, "maximum": maximum,
                "absolute_spread": maximum - minimum,
                "relative_spread_to_median": (
                    (maximum - minimum) / median if median > 0.0
                    else (0.0 if maximum == 0.0 else None)
                ),
            }
            runtime_xs_rows.append({
                "energy_MeV_per_u": f"{args.energy_min_mevu + index + 0.5:.9g}",
                "water_macroscopic_cross_section_per_mm": f"{median:.17g}",
            })
        bins.append(entry)
    empty_bins = [entry["index"] for entry in bins if entry["event_count"] == 0]
    low_bins = [entry["index"] for entry in bins if entry["event_count"] < args.min_events_per_bin]
    count_summary = {"minimum": min(counts), "p05_nearest_rank": percentile(counts, 0.05),
                     "median_nearest_rank": percentile(counts, 0.5)}
    report = {
        "schema_version": 1,
        "energy_grid": {"minimum_MeV_per_u": args.energy_min_mevu,
                        "maximum_exclusive_MeV_per_u": upper_exclusive,
                        "requested_maximum_MeV_per_u": args.energy_max_mevu,
                        "width_MeV_per_u": 1.0, "interval": "[minimum, maximum)"},
        "events": len(merged_interactions), "excluded_below_cutoff": excluded_below_cutoff,
        "minimum_events_per_bin": args.min_events_per_bin,
        "event_count_summary": count_summary, "empty_bins": empty_bins,
        "bins_below_minimum": low_bins,
        "xs_empty_bins": [index for index, values in enumerate(xs_values) if not values],
        "xs_aggregation": "median of finite non-negative event samples in each bin",
        "xs_nearest_fill": False,
        "bins": bins, "eligible_for_compile": not low_bins,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    write_csv(interactions_output, interaction_fields, merged_interactions)
    write_csv(products_output, product_fields, merged_products)
    write_runtime_xs(xs_output, runtime_xs_rows)
    xs_sidecar = {
        "schema_version": 1,
        "kind": "runtime_elastic_cross_section_table",
        "energy_unit": "MeV/u", "cross_section_unit": "1/mm",
        "bin_width_MeV_per_u": 1.0, "representative_energy": "bin center",
        "aggregation": "median", "nearest_fill": False,
        "source_interactions": {
            "path": str(interactions_output), "sha256": sha256(interactions_output),
        },
        "source_metadata": sources,
        "output": {"path": str(xs_output), "sha256": sha256(xs_output)},
        "covered_bins": len(runtime_xs_rows), "required_bins": bin_count,
        "complete": len(runtime_xs_rows) == bin_count,
    }
    xs_sidecar_output.write_text(json.dumps(xs_sidecar, indent=2) + "\n", encoding="utf-8")
    z, a, material, physics_model = identity
    aggregate_metadata = {
        "schema_version": 1, "kind": "elastic_campaign_aggregate",
        "projectile": {"Z": z, "A": a}, "material": material,
        "physics_model": physics_model,
        "provenance": {"scorer": "CarbonElasticNtuple", "processes": ["hadElastic"],
                       "aggregation": "strict concatenation with ID/offset reindexing",
                       "nearest_fill": False,
                       "extraction_modes": sorted(extraction_modes),
                       "mixed_extraction_modes": len(extraction_modes) > 1,
                       "contains_non_dose_reference_sampling": any(
                           not bool(source["continuous_em_loss_enabled"]) for source in sources
                       ),
                       "sources": sources},
        "interactions": len(merged_interactions), "products": len(merged_products),
        "files": {"interactions": {"path": str(interactions_output), "sha256": sha256(interactions_output)},
                  "products": {"path": str(products_output), "sha256": sha256(products_output)},
                  "coverage": {"path": str(report_output), "sha256": sha256(report_output)},
                  "elastic_cross_section": {"path": str(xs_output), "sha256": sha256(xs_output)},
                  "elastic_cross_section_metadata": {"path": str(xs_sidecar_output), "sha256": sha256(xs_sidecar_output)}},
    }
    metadata_output.write_text(json.dumps(aggregate_metadata, indent=2) + "\n", encoding="utf-8")

    if low_bins:
        detail = f"{len(empty_bins)} empty bins" if empty_bins else "no empty bins"
        raise SystemExit(f"Coverage rejected: {detail}; {len(low_bins)} bins below {args.min_events_per_bin} events (report: {report_output})")
    print(f"coverage accepted: min={count_summary['minimum']} p05={count_summary['p05_nearest_rank']} median={count_summary['median_nearest_rank']}")
    if args.compile:
        command = [
            sys.executable, str(args.compiler), "--metadata", str(metadata_output),
            "--interactions", str(interactions_output), "--products", str(products_output),
            "--output", str(package_output), "--output-metadata", str(sidecar_output),
            "--energy-bin-min-mevu", str(args.energy_min_mevu), "--energy-bin-width-mevu", "1",
            "--energy-bin-count", str(bin_count), "--fill-empty", "none",
        ]
        subprocess.run(command, check=True)
        print(f"compiled {package_output}")


if __name__ == "__main__":
    main()
