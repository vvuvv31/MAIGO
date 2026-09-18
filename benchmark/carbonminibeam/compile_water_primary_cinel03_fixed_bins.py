#!/usr/bin/env python3
"""Build a diagnostic primary CINEL03 package with full-stat H/O bins.

Only the hydrogen and oxygen channels are replaced.  Their accepted raw
production events are grouped into fixed-width energy bins; all other target
channels are copied byte-for-byte from the released v2.1 package.  The exact
released channel endpoints are retained because the Schneider bundle requires
the package and rate domains to agree exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REFERENCE_REPO = Path("/mnt/sdb/wuwei/MAIGO")
RAW_ROOT = Path("/mnt/sda/wuwei/cinel03-campaigns/production")
DEFAULT_PACKAGE = REFERENCE_REPO / "data/schneider/cinel03_c12_targets_v2_1.bin"
DEFAULT_CHANNELS = REFERENCE_REPO / "data/schneider/cinel03_c12_targets_v2_1.channels.json"
DEFAULT_BUNDLE = REFERENCE_REPO / "data/schneider/schneider_physics_bundle_v2_1.json"
TARGET_DIRS = {1: "H", 8: "O"}
CAMPAIGN_UUID = "00000000-0000-4000-8000-000000000026"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--input-channels", type=Path, default=DEFAULT_CHANNELS)
    parser.add_argument("--input-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("--bin-width must be finite and positive")

    sys.path.insert(0, str(REFERENCE_REPO / "startup" / "package_tools"))
    import cinel02  # pylint: disable=import-error,import-outside-toplevel
    import cinel03  # pylint: disable=import-error,import-outside-toplevel

    released = cinel03.Cinel03Package.read_binary(args.input_package)
    released_channels = json.loads(args.input_channels.read_text())["channels"]
    domains = {
        int(channel["target_element_z"]): (
            f32(channel["energy_min_MeV_per_u"]),
            f32(channel["energy_max_MeV_per_u"]),
        )
        for channel in released_channels
    }
    if set(TARGET_DIRS) - domains.keys():
        raise RuntimeError("released sidecar does not contain both H and O")

    # Recover correlated event/product records from the released package.
    released_events = []
    product_cursor = 0
    for fixed in released.interactions:
        record = cinel02._unpack_fixed(fixed)  # package tool's canonical codec
        count = int(record["direct_product_count"])
        product_bytes = released.products[product_cursor:product_cursor + count]
        if len(product_bytes) != count:
            raise RuntimeError("released product table does not close")
        products = [cinel02._unpack_product(item, 0) for item in product_bytes]
        released_events.append((record, products))
        product_cursor += count
    if product_cursor != len(released.products):
        raise RuntimeError("released package has trailing products")

    output_events = [
        (record, products, f32(record["collision_energy_MeV_per_u"]))
        for record, products in released_events
        if int(record["target_z"]) not in TARGET_DIRS
    ]
    target_stats = {}
    for target_z, directory in TARGET_DIRS.items():
        lower, upper = domains[target_z]
        accepted = []
        raw_files = sorted((args.raw_root / directory).rglob("worker_*.cinel02"))
        if not raw_files:
            raise RuntimeError(f"no raw files for target Z={target_z}")
        rejected_closure = 0
        rejected_domain = 0
        for raw_file in raw_files:
            for record, products in cinel02.read_raw(raw_file):
                if (int(record["projectile_z"]), int(record["projectile_a"]),
                        int(record["target_z"])) != (6, 12, target_z):
                    raise RuntimeError(f"unexpected channel in {raw_file}")
                collision = float(record["collision_energy_MeV"])
                accounted = (
                    float(record["parent_energy_MeV"])
                    + float(record["process_local_deposit_MeV"])
                    + float(record["unsupported_product_energy_MeV"])
                    + sum(float(product["kinetic_energy_MeV"]) for product in products)
                )
                if accounted > collision + max(200.0, 0.20 * collision):
                    rejected_closure += 1
                    continue
                energy = f32(record["collision_energy_MeV_per_u"])
                if energy < lower or energy > upper:
                    rejected_domain += 1
                    continue
                accepted.append((record, products, energy))

        # Ensure exact released endpoints remain present even if stride-20 had
        # selected a boundary event not encountered after current filtering.
        boundary_events = []
        for boundary in (lower, upper):
            candidates = [
                (record, products, f32(record["collision_energy_MeV_per_u"]))
                for record, products in released_events
                if int(record["target_z"]) == target_z
                and f32(record["collision_energy_MeV_per_u"]) == boundary
            ]
            if not candidates:
                raise RuntimeError(f"released boundary event missing for Z={target_z}, E={boundary}")
            boundary_events.append(candidates[0])
        accepted.extend(boundary_events)

        counts = defaultdict(int)
        quantized = []
        for record, products, energy in accepted:
            if energy == lower or energy == upper:
                node = energy
            else:
                bin_index = math.floor((energy - 0.5) / args.bin_width)
                center = 0.5 + (bin_index + 0.5) * args.bin_width
                node = f32(min(upper, max(lower, center)))
            updated = dict(record)
            updated["collision_energy_MeV_per_u"] = node
            quantized.append((updated, products, node))
            counts[node] += 1
        output_events.extend(quantized)
        per_bin = list(counts.values())
        target_stats[str(target_z)] = {
            "raw_files": len(raw_files),
            "accepted_events_including_two_boundary_anchors": len(quantized),
            "rejected_closure": rejected_closure,
            "rejected_outside_released_domain": rejected_domain,
            "energy_nodes": len(per_bin),
            "events_per_node": {
                "minimum": min(per_bin),
                "median": percentile(per_bin, 0.5),
                "p95": percentile(per_bin, 0.95),
                "maximum": max(per_bin),
            },
        }

    output_events.sort(key=lambda item: (
        int(item[0]["projectile_z"]), int(item[0]["projectile_a"]),
        int(item[0]["target_z"]), item[2]
    ))

    package = cinel03.Cinel03Package()
    package.minimum_energy_MeV_per_u = released.minimum_energy_MeV_per_u
    package.energy_bin_width_MeV_per_u = args.bin_width
    package.minimum_events_per_bin = 1
    package.campaign_uuid = CAMPAIGN_UUID
    for record, products, _ in output_events:
        package.interactions.append(cinel02._pack_fixed(record))
        package.products.extend(cinel02._pack_product(product) for product in products)

    # Cell ranges are retained for the package contract, although production
    # lookup uses the global energy-node index below.
    cursor = 0
    while cursor < len(output_events):
        first = output_events[cursor]
        energy = float(first[2])
        energy_bin = max(0, int(math.floor((energy - 0.5) / args.bin_width)))
        key = (int(first[0]["projectile_z"]), int(first[0]["projectile_a"]),
               int(first[0]["target_z"]), energy_bin)
        start = cursor
        cursor += 1
        while cursor < len(output_events):
            current = output_events[cursor]
            current_bin = max(0, int(math.floor((float(current[2]) - 0.5) / args.bin_width)))
            current_key = (int(current[0]["projectile_z"]), int(current[0]["projectile_a"]),
                           int(current[0]["target_z"]), current_bin)
            if current_key != key:
                break
            cursor += 1
        package.cells.append({
            "projectile_z": key[0], "projectile_a": key[1],
            "target_element_z": key[2], "energy_bin": key[3],
            "interaction_offset": start, "interaction_count": cursor - start,
            "energy_lower_MeV_per_u": 0.5 + key[3] * args.bin_width,
            "energy_upper_MeV_per_u": 0.5 + (key[3] + 1) * args.bin_width,
        })

    cursor = 0
    while cursor < len(output_events):
        first = output_events[cursor]
        key = (int(first[0]["projectile_z"]), int(first[0]["projectile_a"]),
               int(first[0]["target_z"]), f32(first[2]))
        start = cursor
        cursor += 1
        while cursor < len(output_events):
            current = output_events[cursor]
            current_key = (int(current[0]["projectile_z"]), int(current[0]["projectile_a"]),
                           int(current[0]["target_z"]), f32(current[2]))
            if current_key != key:
                break
            cursor += 1
        package.energy_nodes.append(key)
        package.event_offsets.append(start)
    package.event_offsets.append(len(output_events))
    package.event_indices = list(range(len(output_events)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "cinel03_c12_targets_v2_1_water_fixed_1mev"
    output_package = args.output_dir / f"{stem}.bin"
    temporary_dir = Path(tempfile.mkdtemp(prefix="cinel03_fixed_", dir=args.output_dir))
    temporary_package = temporary_dir / output_package.name
    package.write_binary(temporary_package)
    reread = cinel03.Cinel03Package.read_binary(temporary_package)
    if len(reread.interactions) != len(output_events):
        raise RuntimeError("package re-read interaction mismatch")
    temporary_package.replace(output_package)
    shutil.rmtree(temporary_dir)

    channels = []
    grouped_nodes = defaultdict(list)
    for pz, pa, tz, energy in package.energy_nodes:
        grouped_nodes[(pz, pa, tz)].append(energy)
    for (pz, pa, tz), energies in sorted(grouped_nodes.items()):
        gaps = [energies[index + 1] - energies[index]
                for index in range(len(energies) - 1)]
        channels.append({
            "projectile_z": pz, "projectile_a": pa, "target_element_z": tz,
            "energy_min_MeV_per_u": energies[0],
            "energy_max_MeV_per_u": energies[-1],
            "energy_nodes": len(energies),
            "maximum_observed_node_gap_MeV_per_u": max(gaps) if gaps else 0.0,
            "maximum_allowed_node_gap_MeV_per_u": 5.0,
            "interpolation_policy": "stochastic_bracketing",
            "target_alias_allowed": False,
        })
    channels_path = args.output_dir / f"{stem}.channels.json"
    channels_path.write_text(json.dumps({"channels": channels}, indent=2) + "\n")
    metadata = {
        "schema_version": 1,
        "format": "CINPKG04",
        "purpose": "diagnostic full-stat fixed-energy-bin H/O primary final states",
        "data_filename": output_package.name,
        "data_sha256": sha256_file(output_package),
        "channels_filename": channels_path.name,
        "channels_sha256": sha256_file(channels_path),
        "file_size_bytes": output_package.stat().st_size,
        "input_package": str(args.input_package),
        "input_package_sha256": sha256_file(args.input_package),
        "raw_root": str(args.raw_root),
        "bin_width_MeV_per_u": args.bin_width,
        "total_interactions": len(output_events),
        "total_products": len(package.products),
        "replaced_targets": target_stats,
        "campaign_uuid": CAMPAIGN_UUID,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = args.output_dir / f"{stem}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    bundle = json.loads(args.input_bundle.read_text())
    bundle["bundle_name"] = "schneider_physics_bundle_v2_1_water_fixed_1mev_diagnostic"
    bundle["primary_package"] = {
        "file": str(output_package),
        "sha256": metadata["data_sha256"],
        "channels_file": str(channels_path),
        "channels_sha256": metadata["channels_sha256"],
    }
    bundle["diagnostic_note"] = (
        "Full-stat production H/O primary events in fixed 1 MeV/u nodes; "
        "all other target channels copied from v2.1"
    )
    bundle_path = args.output_dir / "schneider_physics_bundle_water_fixed_1mev.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    metadata["bundle_filename"] = bundle_path.name
    metadata["bundle_sha256"] = sha256_file(bundle_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
