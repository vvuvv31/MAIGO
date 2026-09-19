#!/usr/bin/env python3
"""Compile process-faithful charged-fragment + Cu CINPKG04 events.

Each input is an independently captured TOPAS primary-only campaign.  The
compiler rejects a campaign when its projectile, target, or selected Geant4
model differs from the declared reference physics list.  Proton campaigns use
Binary Cascade; light ions and GenericIon campaigns use INCL++ in the energy
range relevant to the current minibeam reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


TOOLS = Path("/mnt/sdb/wuwei/MAIGO/startup/package_tools")
sys.path.insert(0, str(TOOLS))
import cinel02  # type: ignore
import cinel03  # type: ignore

from compile_copper_physics import f32, runtime_supported, sha256


def parse_input(value: str) -> tuple[int, int, str, str, Path]:
    pieces = value.split(",", 4)
    if len(pieces) != 5:
        raise argparse.ArgumentTypeError(
            "input must be Z,A,EXPECTED_MODEL,CAMPAIGN_UUID,RUN_ROOT"
        )
    return int(pieces[0]), int(pieces[1]), pieces[2], pieces[3], Path(pieces[4])


def parse_preserved_package(value: str) -> tuple[int, int, Path]:
    pieces = value.split(",", 2)
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(
            "preserve-projectile-package must be Z,A,PACKAGE"
        )
    return int(pieces[0]), int(pieces[1]), Path(pieces[2])


def projectile_payload_signature(
    package: cinel03.Cinel03Package, z: int, a: int
) -> tuple[list[tuple[bytes, tuple[bytes, ...]]], list[tuple[int, int, int, float]]]:
    """Return byte-exact event/product payload and nodes for one projectile."""
    events: list[tuple[bytes, tuple[bytes, ...]]] = []
    product_offset = 0
    for interaction in package.interactions:
        record = cinel02._unpack_fixed(interaction)
        product_count = int(record["direct_product_count"])
        products = tuple(
            package.products[product_offset:product_offset + product_count]
        )
        product_offset += product_count
        if (int(record["projectile_z"]), int(record["projectile_a"])) == (z, a):
            events.append((interaction, products))
    if product_offset != len(package.products):
        raise ValueError(
            f"package product ledger mismatch: {product_offset} != "
            f"{len(package.products)}"
        )
    nodes = [node for node in package.energy_nodes if node[:2] == (z, a)]
    return events, nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, default=[])
    parser.add_argument(
        "--reuse-source-metadata",
        action="append",
        type=Path,
        default=[],
        help=("Reuse every raw source declared by an earlier compiler metadata "
              "file; intended for adding coverage without replacing its C12 "
              "campaign"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--preserve-projectile-package",
        action="append",
        type=parse_preserved_package,
        default=[],
        help=("Require the compiled package to preserve byte-exact interaction, "
              "product and energy-node payloads for Z,A from PACKAGE"),
    )
    parser.add_argument(
        "--package-campaign", default="20260918-0000-4000-8000-000000000040"
    )
    args = parser.parse_args()

    inputs = list(args.input)
    for metadata_path in args.reuse_source_metadata:
        metadata = json.loads(metadata_path.read_text())
        for source in metadata.get("sources", []):
            projectile = source["projectile"]
            inputs.append((
                int(projectile["z"]),
                int(projectile["a"]),
                str(source["expected_model"]),
                str(source["campaign_uuid"]),
                Path(source["run_root"]),
            ))
    inputs = list(dict.fromkeys(inputs))
    if not inputs:
        parser.error("at least one --input or --reuse-source-metadata is required")

    events: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    sources: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    declared: dict[tuple[int, int], str] = {}
    for z, a, expected_model, campaign, root in inputs:
        if z <= 0 or a < z:
            raise ValueError(f"invalid projectile Z={z} A={a}")
        previous_model = declared.get((z, a))
        if previous_model is not None and previous_model != expected_model:
            raise ValueError(
                f"inconsistent models for Z={z} A={a}: "
                f"{previous_model!r} versus {expected_model!r}"
            )
        declared[(z, a)] = expected_model
        files = sorted((root / "raw" / campaign).glob("worker_*.cinel02"))
        if not files:
            raise ValueError(f"no CINEL02 files for {campaign} below {root}")
        accepted = 0
        models: Counter[str] = Counter()
        for path in files:
            for record, products in cinel02.read_raw(path):
                key = (int(record["projectile_z"]), int(record["projectile_a"]))
                if key != (z, a):
                    rejected["wrong_projectile"] += 1
                    continue
                if int(record["target_z"]) != 29:
                    rejected["not_copper"] += 1
                    continue
                model = str(record["model_name"])
                models[model] += 1
                if model != expected_model:
                    raise ValueError(
                        f"{root}: expected model {expected_model!r}, observed {model!r}"
                    )
                energy = float(record["collision_energy_MeV_per_u"])
                if not math.isfinite(energy) or energy <= 0.0:
                    rejected["bad_energy"] += 1
                    continue
                events.append((record, products))
                accepted += 1
        if accepted == 0:
            raise ValueError(f"no accepted Z={z} A={a} Copper events below {root}")
        sources.append({
            "projectile": {"z": z, "a": a},
            "expected_model": expected_model,
            "observed_models": dict(models),
            "campaign_uuid": campaign,
            "run_root": str(root),
            "events": accepted,
            "raw_files": {str(path): sha256(path) for path in files},
        })

    events.sort(key=lambda item: (
        int(item[0]["projectile_z"]), int(item[0]["projectile_a"]),
        f32(float(item[0]["collision_energy_MeV_per_u"])),
    ))
    package = cinel03.Cinel03Package()
    package.campaign_uuid = args.package_campaign
    package.minimum_energy_MeV_per_u = min(
        f32(float(record["collision_energy_MeV_per_u"]))
        for record, _ in events
    )
    package.energy_bin_width_MeV_per_u = 1.0
    package.minimum_events_per_bin = 1

    for record, products in events:
        package.interactions.append(cinel02._pack_fixed(record))
        for product in products:
            product = dict(product)
            product["role"] = 0 if runtime_supported(product) else 2
            package.products.append(cinel02._pack_product(product))

    cursor = 0
    while cursor < len(events):
        record = events[cursor][0]
        z = int(record["projectile_z"])
        a = int(record["projectile_a"])
        energy = f32(float(record["collision_energy_MeV_per_u"]))
        energy_bin = max(0, int(math.floor(energy)))
        start = cursor
        while cursor < len(events):
            candidate = events[cursor][0]
            candidate_key = (
                int(candidate["projectile_z"]), int(candidate["projectile_a"]),
                max(0, int(math.floor(f32(float(
                    candidate["collision_energy_MeV_per_u"]))))),
            )
            if candidate_key != (z, a, energy_bin):
                break
            cursor += 1
        package.cells.append({
            "projectile_z": z, "projectile_a": a,
            "target_element_z": 29, "energy_bin": energy_bin,
            "interaction_offset": start, "interaction_count": cursor - start,
            "energy_lower_MeV_per_u": float(energy_bin),
            "energy_upper_MeV_per_u": float(energy_bin + 1),
        })

    offsets = [0]
    cursor = 0
    while cursor < len(events):
        record = events[cursor][0]
        z = int(record["projectile_z"])
        a = int(record["projectile_a"])
        energy = f32(float(record["collision_energy_MeV_per_u"]))
        package.energy_nodes.append((z, a, 29, energy))
        while cursor < len(events):
            candidate = events[cursor][0]
            if (int(candidate["projectile_z"]),
                int(candidate["projectile_a"]),
                f32(float(candidate["collision_energy_MeV_per_u"]))) != (z, a, energy):
                break
            package.event_indices.append(cursor)
            cursor += 1
        offsets.append(cursor)
    package.event_offsets = offsets

    # Read and fingerprint every preservation reference before touching the
    # requested output. This also makes output==reference safe: validation is
    # against the original package, never against the newly written file.
    preserved_references: list[dict[str, object]] = []
    for z, a, reference_path in args.preserve_projectile_package:
        reference = cinel03.Cinel03Package.read_binary(reference_path)
        preserved_references.append({
            "z": z,
            "a": a,
            "path": reference_path,
            "sha256": sha256(reference_path),
            "signature": projectile_payload_signature(reference, z, a),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".tmp",
        dir=args.output.parent, delete=False)
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        package.write_binary(temporary_path)
        reread = cinel03.Cinel03Package.read_binary(temporary_path)
        preservation_checks: list[dict[str, object]] = []
        for preserved in preserved_references:
            z = int(preserved["z"])
            a = int(preserved["a"])
            reference_path = Path(preserved["path"])
            actual_signature = projectile_payload_signature(reread, z, a)
            if actual_signature != preserved["signature"]:
                raise ValueError(
                    f"compiled package changed preserved Z={z} A={a} payload "
                    f"from {reference_path}"
                )
            preservation_checks.append({
                "projectile": {"z": z, "a": a},
                "reference_package": str(reference_path),
                "reference_sha256": str(preserved["sha256"]),
                "events": len(actual_signature[0]),
                "energy_nodes": len(actual_signature[1]),
                "status": "byte_exact",
            })
        output_sha256 = sha256(temporary_path)
        os.replace(temporary_path, args.output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    metadata = {
        "schema": "MAIGO_COPPER_FRAGMENT_CASCADE_V1",
        "reference_physics": {
            "proton_0_to_6_GeV": "Binary Cascade",
            "d_t_He_GenericIon_0_to_3_GeV_per_u": "INCL++ undefined",
            "source_log": "/mnt/sda/wuwei/minibeam_phase_scan_6255_1.log",
        },
        "target": {"material": "G4_Cu", "element_z": 29},
        "package": str(args.output),
        "sha256": output_sha256,
        "projectiles": [{"z": z, "a": a, "model": declared[(z, a)]}
                        for z, a in sorted(declared)],
        "interactions": len(reread.interactions),
        "products": len(reread.products),
        "energy_nodes": len(reread.energy_nodes),
        "sources": sources,
        "reused_source_metadata": [str(path) for path in args.reuse_source_metadata],
        "preservation_checks": preservation_checks,
        "rejected": dict(rejected),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
