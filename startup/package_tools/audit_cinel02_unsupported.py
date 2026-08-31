#!/usr/bin/env python3
"""Stream a CINEL02 campaign and classify unsupported product energy."""

from __future__ import annotations

import argparse
import binascii
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import cinel02


def iter_records(path: Path) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Decode one worker without retaining the worker or campaign in memory."""
    with path.open("rb") as stream:
        header_bytes = stream.read(cinel02.RAW_HEADER_SIZE)
        if len(header_bytes) != cinel02.RAW_HEADER_SIZE:
            raise ValueError(f"CINEL02 raw file is shorter than its header: {path}")
        header = cinel02.RAW_HEADER.unpack(header_bytes)
        magic, version, header_size, endian, _, expected_records, expected_products, bytes_written, _, _ = header
        if magic != cinel02.RAW_MAGIC or version != cinel02.RAW_VERSION or header_size != cinel02.RAW_HEADER_SIZE:
            raise ValueError(f"unsupported CINEL02 raw header: {path}")
        if endian != 0x01020304:
            raise ValueError(f"CINEL02 raw file is not little-endian: {path}")

        record_count = 0
        product_count = 0
        while True:
            prefix = stream.read(cinel02.RECORD_PREFIX.size)
            if not prefix:
                break
            if len(prefix) != cinel02.RECORD_PREFIX.size:
                raise ValueError(f"truncated CINEL02 record prefix: {path}")
            magic, version, _, record_length, payload_length = cinel02.RECORD_PREFIX.unpack(prefix)
            if magic != cinel02.RECORD_MAGIC or version != cinel02.RECORD_VERSION:
                raise ValueError(f"invalid CINEL02 record prefix: {path}")
            if record_length != cinel02.RECORD_PREFIX.size + payload_length + cinel02.CRC_FORMAT.size:
                raise ValueError(f"invalid CINEL02 record length: {path}")
            payload = stream.read(payload_length)
            crc_bytes = stream.read(cinel02.CRC_FORMAT.size)
            if len(payload) != payload_length or len(crc_bytes) != cinel02.CRC_FORMAT.size:
                raise ValueError(f"truncated CINEL02 record: {path}")
            expected_crc = cinel02.CRC_FORMAT.unpack(crc_bytes)[0]
            if (binascii.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                raise ValueError(f"CINEL02 CRC mismatch: {path}")

            record = cinel02._unpack_fixed(payload)
            count = int(record["direct_product_count"])
            product_bytes = payload[cinel02.RAW_FIXED_FORMAT.size :]
            if len(product_bytes) != count * cinel02.PRODUCT_FORMAT.size:
                raise ValueError(f"CINEL02 product count/length mismatch: {path}")
            products = [
                cinel02._unpack_product(product_bytes, index * cinel02.PRODUCT_FORMAT.size)
                for index in range(count)
            ]
            record_count += 1
            product_count += count
            yield record, products

        actual_size = stream.tell()
        if bytes_written not in (0, actual_size):
            raise ValueError(f"CINEL02 file-size field does not match: {path}")
        if expected_records not in (0, record_count):
            raise ValueError(f"CINEL02 interaction-count mismatch: {path}")
        if expected_products not in (0, product_count):
            raise ValueError(f"CINEL02 product-count mismatch: {path}")


def audit(paths: list[Path]) -> dict[str, Any]:
    by_identity: dict[tuple[Any, ...], dict[str, float | int | str]] = defaultdict(
        lambda: {"count": 0, "kinetic_energy_MeV": 0.0}
    )
    interactions = products_seen = unsupported_count = isomer_count = 0
    captured_energy_terms: list[float] = []
    unsupported_energy_terms: list[float] = []
    isomer_energy_terms: list[float] = []

    for path in paths:
        for record, products in iter_records(path):
            interactions += 1
            captured_energy_terms.append(float(record["collision_energy_MeV"]))
            for product in products:
                products_seen += 1
                normalised = cinel02._normalise_product(product)
                isomer = cinel02._ground_state_isomer_product(normalised)
                if isomer is not None:
                    isomer_count += 1
                    isomer_energy_terms.append(float(normalised["kinetic_energy_MeV"]))
                    normalised = isomer
                if cinel02._supported_product_identity(normalised):
                    continue
                kind = cinel02._classify_inelastic_identity(
                    normalised["pdg"], normalised["z"], normalised["a"],
                    normalised["charge"], normalised["rest_mass"], normalised["excitation"],
                )
                key = (
                    int(normalised["pdg"]), int(normalised["z"]), int(normalised["a"]),
                    float(normalised["charge"]), float(normalised["excitation"]),
                    int(normalised["role"]), kind or "invalid_or_unknown",
                )
                entry = by_identity[key]
                kinetic = float(normalised["kinetic_energy_MeV"])
                entry["count"] = int(entry["count"]) + 1
                entry["kinetic_energy_MeV"] = float(entry["kinetic_energy_MeV"]) + kinetic
                unsupported_count += 1
                unsupported_energy_terms.append(kinetic)

    captured_energy = math.fsum(captured_energy_terms)
    unsupported_energy = math.fsum(unsupported_energy_terms)
    rows = []
    for key, values in by_identity.items():
        pdg, z, a, charge, excitation, role, kind = key
        rows.append({
            "pdg": pdg, "z": z, "a": a, "charge": charge,
            "excitation_MeV": excitation, "role": role, "classification": kind,
            **values,
        })
    rows.sort(key=lambda row: (-float(row["kinetic_energy_MeV"]), int(row["pdg"])))
    return {
        "format": "CINEL02_UNSUPPORTED_AUDIT_V1",
        "input_files": [str(path) for path in paths],
        "interaction_count": interactions,
        "product_count": products_seen,
        "captured_incident_energy_MeV": captured_energy,
        "isomer_normalisation": {
            "count": isomer_count,
            "kinetic_energy_MeV": math.fsum(isomer_energy_terms),
        },
        "unsupported": {
            "count": unsupported_count,
            "kinetic_energy_MeV": unsupported_energy,
            "energy_fraction_of_captured_incident": unsupported_energy / captured_energy,
            "identities": rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "interaction_count": result["interaction_count"],
        "product_count": result["product_count"],
        "unsupported": result["unsupported"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
