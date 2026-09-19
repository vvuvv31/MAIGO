#!/usr/bin/env python3
"""Compile one Copper extraction run into the optional minibeam inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "package_tools"
sys.path.insert(0, str(TOOLS))
import cinel02  # type: ignore  # authoritative TOPAS raw schema
import cinel03  # type: ignore  # shared runtime CINPKG04 schema


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def runtime_supported(product: dict[str, object]) -> bool:
    pdg = int(product["pdg"])
    z = int(product["z"])
    a = int(product["a"])
    charge = float(product["charge"])
    mass = float(product["rest_mass"])
    excitation = float(product["excitation"])
    if not (0.0 <= excitation <= 1.0e-4):
        return False
    if pdg == 22:
        return z == 0 and a == 0 and abs(charge) <= 1.0e-3 and abs(mass) <= 1.0e-3
    if pdg == 2112:
        return z == 0 and a == 1 and abs(charge) <= 1.0e-3 and abs(mass - 939.56542052) <= 0.94
    if pdg == 2212:
        return z == 1 and a == 1 and abs(charge - 1.0) <= 1.0e-3 and abs(mass - 938.27208816) <= 0.94
    if z <= 0 or a < z or pdg != 1_000_000_000 + z * 10_000 + a * 10:
        return False
    expected = a * 931.49410242
    return abs(charge - z) <= 1.0e-3 and abs(mass - expected) <= max(25.0, 0.02 * expected)


def read_ascii_ntuple(stem: Path) -> list[dict[str, str]]:
    data = stem.with_suffix(".phsp")
    header = stem.with_suffix(".header")
    columns: list[tuple[int, str]] = []
    pattern = re.compile(r"^\s*(\d+):\s*(.*?)\s*$")
    for line in header.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            columns.append((int(match.group(1)), match.group(2)))
    columns.sort()
    names = [name for _, name in columns]
    rows = []
    for line in data.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if values:
            if len(values) != len(names):
                raise ValueError(f"{data}: expected {len(names)} columns, got {len(values)}")
            rows.append(dict(zip(names, values)))
    if not rows:
        raise ValueError(f"empty TOPAS n-tuple: {data}")
    return rows


def compile_tables(run_root: Path, output: Path) -> dict[str, object]:
    stopping_rows = read_ascii_ntuple(run_root / "c12_copper_stopping")
    stopping = output / "c12_copper_stopping_geant4_11_3_2.csv"
    stopping_by_energy: dict[float, float] = {}
    for row in stopping_rows:
        energy = float(row["Energy (MeV/u)"])
        value = float(row["Electronic dE/dx (MeV/mm)"])
        previous = stopping_by_energy.setdefault(energy, value)
        if not math.isclose(previous, value, rel_tol=1e-6):
            raise ValueError(f"thread-dependent Copper stopping at {energy} MeV/u")
    with stopping.open("w", encoding="utf-8") as stream:
        stream.write("energy_MeVu,stopping_power_MeV_per_mm\n")
        for energy, value in sorted(stopping_by_energy.items()):
            stream.write(f"{energy},{value}\n")

    rate_rows = read_ascii_ntuple(run_root / "c12_copper_inelastic_rate")
    rate = output / "c12_copper_inelastic_rate_geant4_11_3_2.csv"
    rate_by_energy: dict[float, float] = {}
    for row in rate_rows:
        energy = float(row["Energy (MeV/u)"])
        value = float(row["Water Macroscopic Cross Section (1/mm)"])
        previous = rate_by_energy.setdefault(energy, value)
        if not math.isclose(previous, value, rel_tol=1e-6):
            raise ValueError(f"thread-dependent Copper inelastic rate at {energy} MeV/u")
    with rate.open("w", encoding="utf-8") as stream:
        stream.write("energy_MeV_per_u,c12_copper_macroscopic_inelastic_cross_section_per_mm\n")
        for energy, value in sorted(rate_by_energy.items()):
            stream.write(f"{energy},{value}\n")
    elastic_rows = read_ascii_ntuple(run_root / "c12_copper_elastic")
    elastic = output / "c12_copper_general_ion_elastic.csv"
    carbon_mass = 11174.86323534
    with elastic.open("w", encoding="utf-8") as stream:
        stream.write(
            "energy_MeV_per_u,macroscopic_cross_section_per_mm,"
            "transfer_fraction,target_a,target_mass_MeV\n"
        )
        for row in elastic_rows:
            energy_u = float(row["Incident Energy (MeV/u)"])
            outgoing_u = float(row["Outgoing Projectile Energy (MeV/u)"])
            target_a = int(row["Target Mass Number A"])
            target_mass = target_a * 931.49410242
            kinetic = 12.0 * energy_u
            maximum_transfer = (
                2.0 * target_mass * kinetic * (kinetic + 2.0 * carbon_mass)
                / ((carbon_mass + target_mass) ** 2 + 2.0 * target_mass * kinetic)
            )
            transfer = max(0.0, 12.0 * (energy_u - outgoing_u))
            fraction = min(1.0, transfer / maximum_transfer) if maximum_transfer > 0 else 0.0
            stream.write(
                f'{energy_u},{row["Macroscopic Elastic Cross Section (1/mm)"]},'
                f"{fraction},{target_a},{target_mass}\n"
            )

    return {
        "stopping_file": stopping.name,
        "stopping_sha256": sha256(stopping),
        "inelastic_rate_file": rate.name,
        "inelastic_rate_sha256": sha256(rate),
        "elastic_file": elastic.name,
        "elastic_sha256": sha256(elastic),
        "elastic_events": len(elastic_rows),
    }


def compile_inclxx(run_root: Path, output: Path, campaign: str) -> dict[str, object]:
    raw_files = sorted((run_root / "raw" / campaign).glob("worker_*.cinel02"))
    if not raw_files:
        raise ValueError(f"no CINEL02 worker files below {run_root}")
    events = []
    rejected = defaultdict(int)
    for raw in raw_files:
        for record, products in cinel02.read_raw(raw):
            if (record["projectile_z"], record["projectile_a"]) != (6, 12):
                rejected["not_c12"] += 1
                continue
            if record["target_z"] != 29:
                rejected["not_copper"] += 1
                continue
            energy = float(record["collision_energy_MeV_per_u"])
            if not math.isfinite(energy) or energy <= 0.0:
                rejected["bad_energy"] += 1
                continue
            events.append((record, products))
    if not events:
        raise ValueError("no valid C-12 + Cu INCLXX events")

    events.sort(key=lambda item: f32(float(item[0]["collision_energy_MeV_per_u"])))
    package = cinel03.Cinel03Package()
    package.campaign_uuid = campaign
    package.minimum_energy_MeV_per_u = min(
        f32(float(record["collision_energy_MeV_per_u"])) for record, _ in events
    )
    package.energy_bin_width_MeV_per_u = 1.0
    package.minimum_events_per_bin = 1

    for record, products in events:
        package.interactions.append(cinel02._pack_fixed(record))
        for product in products:
            product = dict(product)
            product["role"] = 0 if runtime_supported(product) else 2
            package.products.append(cinel02._pack_product(product))

    # Organizational 1-MeV/u cells; device lookup uses the exact global nodes.
    cursor = 0
    while cursor < len(events):
        energy = f32(float(events[cursor][0]["collision_energy_MeV_per_u"]))
        energy_bin = max(0, int(math.floor(energy)))
        start = cursor
        while cursor < len(events):
            candidate = f32(float(events[cursor][0]["collision_energy_MeV_per_u"]))
            if max(0, int(math.floor(candidate))) != energy_bin:
                break
            cursor += 1
        package.cells.append({
            "projectile_z": 6,
            "projectile_a": 12,
            "target_element_z": 29,
            "energy_bin": energy_bin,
            "interaction_offset": start,
            "interaction_count": cursor - start,
            "energy_lower_MeV_per_u": float(energy_bin),
            "energy_upper_MeV_per_u": float(energy_bin + 1),
        })

    offsets = [0]
    index = 0
    while index < len(events):
        energy = f32(float(events[index][0]["collision_energy_MeV_per_u"]))
        package.energy_nodes.append((6, 12, 29, energy))
        while index < len(events) and f32(
            float(events[index][0]["collision_energy_MeV_per_u"])
        ) == energy:
            package.event_indices.append(index)
            index += 1
        offsets.append(index)
    package.event_offsets = offsets

    path = output / "c12_copper_inclxx.cinpkg"
    package.write_binary(path)
    reread = cinel03.Cinel03Package.read_binary(path)
    energies = [node[3] for node in reread.energy_nodes]
    maximum_gap = max((b - a for a, b in zip(energies, energies[1:])), default=0.0)
    return {
        "inclxx_file": path.name,
        "inclxx_sha256": sha256(path),
        "interactions": len(package.interactions),
        "products": len(package.products),
        "energy_nodes": len(package.energy_nodes),
        "energy_min_MeV_per_u": min(energies),
        "energy_max_MeV_per_u": max(energies),
        "maximum_node_gap_MeV_per_u": maximum_gap,
        "rejected": dict(rejected),
        "raw_files": {str(path): sha256(path) for path in raw_files},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--campaign", default="20260917-0000-4000-8000-000000000030"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "MAIGO_COPPER_MINIBEAM_PHYSICS_V1",
        "target": {"material": "G4_Cu", "element_z": 29},
        "projectile": {"z": 6, "a": 12},
        "physics": {
            "inelastic": "G4INCLXXInterface via g4ion-inclxx",
            "elastic": "G4IonElasticPhysics/NNDiffuseElastic",
            "em": "g4em-standard_opt4",
        },
    }
    metadata.update(compile_tables(args.run_root, args.output))
    metadata.update(compile_inclxx(args.run_root, args.output, args.campaign))
    metadata_path = args.output / "copper_minibeam_physics.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
