#!/usr/bin/env python3
"""Generate and validate TOPAS CT/Schneider CINEL02 rate extraction runs.

The generated TOPAS jobs use ``Cinel02MaterialRateNtuple``.  That scorer
queries Geant4's authoritative per-atom inelastic cross-section store for the
same H-1/O-16 target pair consumed by ``cinel02_select_water_target_device``.
Each Schneider section is represented by its tabulated elemental mass
fractions at a reference density (one g/cm3 by default); the GPU can later
scale the resulting macroscopic rate by the voxel density.

This tool deliberately does not modify the GPU runtime or compile a rate
table.  ``--generate`` is the safe default.  ``--run`` executes locally for a
small smoke; ``--submit`` writes and submits a local Slurm array, as required
for the full TOPAS campaign.  ``--compile`` validates the resulting ntuples
and writes a section-aware CSV plus provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHNEIDER = REPO_ROOT / "data" / "HUtoMaterialSchneider.txt"
DEFAULT_WORK_ROOT = REPO_ROOT / "startup" / "work" / "ct_cinel02_schneider"
DEFAULT_TOPAS = Path("/home/wuwei/topas/topas-build/topas")

# The GPU charged-isotope table has 18 slots, but 6Be is intentionally not a
# transported ion under the TOPAS compatibility policy.  Do not create a
# misleading rate group for it.  C-12 is retained because it is the primary
# CINEL02 projectile and can also be a secondary continuation.
TRANSPORTABLE_PROJECTILES = (
    (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6),
    (3, 6), (3, 7), (4, 7), (4, 9), (4, 10),
    (5, 8), (5, 10), (5, 11), (6, 10), (6, 11), (6, 12),
)
TARGETS = ((1, 1), (8, 16))
EXPECTED_COLUMNS = (
    "material_section",
    "material_name",
    "material_density_g_per_cm3",
    "projectile_z",
    "projectile_a",
    "target_z",
    "target_a",
    "target_mass_fraction",
    "target_number_density_per_mm3",
    "energy_MeV_per_u",
    "microscopic_cross_section_barn",
    "macroscopic_cross_section_per_mm",
)


@dataclass(frozen=True)
class SchneiderTable:
    elements: tuple[str, ...]
    boundaries: tuple[int, ...]
    fractions: tuple[tuple[float, ...], ...]

    @property
    def section_count(self) -> int:
        return len(self.fractions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_counted_tokens(line: str, *, label: str) -> list[str]:
    if "=" not in line:
        raise ValueError(f"{label}: missing '='")
    tokens = _strip_comment(line.split("=", 1)[1]).split()
    if not tokens:
        raise ValueError(f"{label}: empty value")
    try:
        count = int(tokens[0])
    except ValueError as error:
        raise ValueError(f"{label}: missing value count") from error
    values = tokens[1:]
    if len(values) != count:
        raise ValueError(f"{label}: declared {count} values, found {len(values)}")
    return values


def parse_schneider(path: Path) -> SchneiderTable:
    elements: tuple[str, ...] | None = None
    boundaries: tuple[int, ...] | None = None
    rows: dict[int, tuple[float, ...]] = {}
    element_pattern = re.compile(r"^sv:Ge/Patient/SchneiderElements\s*=")
    boundary_pattern = re.compile(r"^iv:Ge/Patient/SchneiderHUToMaterialSections\s*=")
    row_pattern = re.compile(r"^uv:Ge/Patient/SchneiderMaterialsWeight(\d+)\s*=")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        if element_pattern.match(line):
            values = _parse_counted_tokens(line, label=f"{path}:{line_number}")
            elements = tuple(value.strip('"') for value in values)
        elif boundary_pattern.match(line):
            values = _parse_counted_tokens(line, label=f"{path}:{line_number}")
            try:
                boundaries = tuple(int(value) for value in values)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: non-integer HU boundary") from error
        else:
            match = row_pattern.match(line)
            if match:
                section = int(match.group(1)) - 1
                values = _parse_counted_tokens(line, label=f"{path}:{line_number}")
                try:
                    parsed = tuple(float(value) for value in values)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: non-numeric fraction") from error
                if any(not math.isfinite(value) or value < 0.0 for value in parsed):
                    raise ValueError(f"{path}:{line_number}: invalid material fraction")
                rows[section] = parsed
    if elements is None or boundaries is None:
        raise ValueError(f"{path}: Schneider elements or section boundaries are missing")
    section_count = len(boundaries) - 1
    if section_count <= 0 or len(rows) != section_count:
        raise ValueError(
            f"{path}: expected {section_count} material rows, found {len(rows)}"
        )
    fractions: list[tuple[float, ...]] = []
    for section in range(section_count):
        if section not in rows:
            raise ValueError(f"{path}: missing Schneider material row {section + 1}")
        row = rows[section]
        if len(row) != len(elements):
            raise ValueError(f"{path}: section {section} fraction width mismatch")
        total = sum(row)
        if not math.isfinite(total) or abs(total - 1.0) > 2.0e-3:
            raise ValueError(f"{path}: section {section} fractions sum to {total}")
        fractions.append(row)
    return SchneiderTable(elements, boundaries, tuple(fractions))


def parse_sections(value: str, section_count: int) -> list[int]:
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid descending section range: {token}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(token))
    if not selected:
        raise ValueError("at least one Schneider section is required")
    invalid = sorted(section for section in selected if section < 0 or section >= section_count)
    if invalid:
        raise ValueError(f"section(s) outside [0,{section_count - 1}]: {invalid}")
    return sorted(selected)


def _format_fraction(value: float) -> str:
    return f"{value:.12g}"


def _material_component(name: str) -> str:
    if name == "Hydrogen":
        return "CTHydrogenH1"
    if name == "Oxygen":
        return "CTOxygenO16"
    return name


def render_config(
    table: SchneiderTable,
    section: int,
    destination: Path,
    output_stem: str,
    *,
    histories: int,
    threads: int,
    seed: int,
    energy_min: float,
    energy_width: float,
    energy_count: int,
    reference_density: float,
) -> None:
    material = f"CTSchneiderSection{section:02d}"
    components = " ".join(f'"{_material_component(name)}"' for name in table.elements)
    fractions = " ".join(_format_fraction(value) for value in table.fractions[section])
    content = f'''# Generated by extract_ct_cinel02_rates.py; section {section}
i:Ts/Seed = {seed}
i:Ts/NumberOfThreads = {threads}
i:Ts/ShowHistoryCountAtInterval = {max(histories, 1)}
i:Ts/ParameterizationErrorMaxReports = 10
i:Ts/UnscoredHitMaxReports = 10
i:Ts/IndexErrorMaxReports = 10
b:Gr/Enable = "False"

# Pure H-1/O-16 elements make target identities exactly match the GPU selector.
i:Is/CTH1/Z = 1
i:Is/CTH1/N = 1
d:Is/CTH1/A = 1.00782503223 g/mole
s:El/CTHydrogenH1/Symbol = "H"
sv:El/CTHydrogenH1/IsotopeNames = 1 "CTH1"
uv:El/CTHydrogenH1/IsotopeAbundances = 1 1.0
i:Is/CTO16/Z = 8
i:Is/CTO16/N = 16
d:Is/CTO16/A = 15.99491461957 g/mole
s:El/CTOxygenO16/Symbol = "O"
sv:El/CTOxygenO16/IsotopeNames = 1 "CTO16"
uv:El/CTOxygenO16/IsotopeAbundances = 1 1.0

s:Ma/{material}/State = "Solid"
d:Ma/{material}/Density = {reference_density:.12g} g/cm3
sv:Ma/{material}/Components = {len(table.elements)} {components}
uv:Ma/{material}/Fractions = {len(table.elements)} {fractions}

s:Ge/World/Material = "G4_Galactic"
d:Ge/World/HLX = 2.0 m
d:Ge/World/HLY = 2.0 m
d:Ge/World/HLZ = 2.0 m
b:Ge/World/Invisible = "TRUE"
s:Ge/BeamPosition/Parent = "World"
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransX = 0.0 mm
d:Ge/BeamPosition/TransY = 0.0 mm
d:Ge/BeamPosition/TransZ = Ge/World/HLZ m
d:Ge/BeamPosition/RotX = 180.0 deg
d:Ge/BeamPosition/RotY = 0.0 deg
d:Ge/BeamPosition/RotZ = 0.0 deg
s:Ge/Phantom/Parent = "World"
s:Ge/Phantom/Type = "TsBox"
s:Ge/Phantom/Material = "{material}"
d:Ge/Phantom/HLX = 50.0 mm
d:Ge/Phantom/HLY = 50.0 mm
d:Ge/Phantom/HLZ = 10.0 mm
d:Ge/Phantom/MaxStepSize = 1.0 mm

s:So/PrimaryBeam/Type = "Beam"
s:So/PrimaryBeam/Component = "BeamPosition"
s:So/PrimaryBeam/BeamParticle = "GenericIon(6,12)"
d:So/PrimaryBeam/BeamEnergy = 2400.0 MeV
u:So/PrimaryBeam/BeamEnergySpread = 0.0
s:So/PrimaryBeam/BeamPositionDistribution = "None"
s:So/PrimaryBeam/BeamAngularDistribution = "None"
i:So/PrimaryBeam/NumberOfHistoriesInRun = {histories}

s:Sc/Cinel02MaterialRate/Quantity = "Cinel02MaterialRateNtuple"
s:Sc/Cinel02MaterialRate/Component = "Phantom"
s:Sc/Cinel02MaterialRate/OutputType = "ASCII"
s:Sc/Cinel02MaterialRate/OutputFile = "{output_stem}"
s:Sc/Cinel02MaterialRate/IfOutputFileAlreadyExists = "Overwrite"
i:Sc/Cinel02MaterialRate/MaterialSection = {section}
d:Sc/Cinel02MaterialRate/EnergyMin = {energy_min:.12g} MeV
d:Sc/Cinel02MaterialRate/EnergyWidth = {energy_width:.12g} MeV
i:Sc/Cinel02MaterialRate/EnergyCount = {energy_count}

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4decay"
d:Ph/Default/CutForAllParticles = 0.05 mm
'''
    destination.write_text(content, encoding="utf-8")


def _read_header(path: Path) -> tuple[int | None, tuple[str, ...]]:
    count: int | None = None
    columns: list[tuple[int, str]] = []
    count_pattern = re.compile(r"^\s*Number of Scored Entries:\s*(\d+)\s*$")
    column_pattern = re.compile(r"^\s*(\d+):\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = count_pattern.match(line)
        if match:
            count = int(match.group(1))
            continue
        match = column_pattern.match(line)
        if match:
            columns.append((int(match.group(1)), match.group(2).strip()))
    indices = [index for index, _ in columns]
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"{path}: non-contiguous column indices {indices}")
    return count, tuple(name for _, name in columns)


def read_section_rows(run_dir: Path) -> list[dict[str, str]]:
    phsp_files = sorted(run_dir.glob("*.phsp"))
    if len(phsp_files) != 1:
        raise ValueError(f"{run_dir}: expected one .phsp, found {len(phsp_files)}")
    phsp = phsp_files[0]
    header = phsp.with_suffix(".header")
    count, columns = _read_header(header)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"{header}: expected {EXPECTED_COLUMNS}, got {columns}")
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(phsp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != len(columns):
            raise ValueError(f"{phsp}:{line_number}: expected {len(columns)} values")
        rows.append(dict(zip(columns, values)))
    if count is not None and count != len(rows):
        raise ValueError(f"{phsp}: header count {count}, parsed {len(rows)} rows")
    return rows


def _number(row: dict[str, str], key: str, *, nonnegative: bool = True) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid {key}={row.get(key)!r}") from error
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError(f"invalid finite/non-negative {key}={value}")
    return value


def compile_outputs(
    work_root: Path,
    sections: Iterable[int],
    output_csv: Path,
    metadata_path: Path,
    *,
    source_path: Path,
    energy_min: float,
    energy_width: float,
    energy_count: int,
    reference_density: float,
) -> dict[str, object]:
    all_rows: list[dict[str, str]] = []
    seen: set[tuple[int, int, int, int, int, float]] = set()
    section_list = list(sections)
    for section in section_list:
        rows = read_section_rows(work_root / f"section_{section:02d}")
        if not rows:
            raise ValueError(f"section {section}: empty extraction")
        energies_by_identity: dict[tuple[int, int, int, int], list[float]] = {}
        for row in rows:
            section_value = int(row["material_section"])
            if section_value != section:
                raise ValueError(f"section {section}: scorer emitted section {section_value}")
            projectile = (int(row["projectile_z"]), int(row["projectile_a"]))
            target = (int(row["target_z"]), int(row["target_a"]))
            if projectile not in TRANSPORTABLE_PROJECTILES:
                raise ValueError(f"unsupported/excluded projectile in output: {projectile}")
            if target not in TARGETS:
                raise ValueError(f"unexpected target in output: {target}")
            density = _number(row, "material_density_g_per_cm3")
            if abs(density - reference_density) > max(1.0e-8, reference_density * 1.0e-6):
                raise ValueError(f"section {section}: reference density mismatch {density}")
            energy = _number(row, "energy_MeV_per_u")
            microscopic = _number(row, "microscopic_cross_section_barn")
            macroscopic = _number(row, "macroscopic_cross_section_per_mm")
            number_density = _number(row, "target_number_density_per_mm3")
            if number_density == 0.0 and (microscopic != 0.0 or macroscopic != 0.0):
                raise ValueError(f"section {section}: non-zero rate with zero target density")
            key = (section, *projectile, *target, energy)
            if key in seen:
                raise ValueError(f"duplicate rate row {key}")
            seen.add(key)
            identity = (*projectile, *target)
            energies_by_identity.setdefault(identity, []).append(energy)
            all_rows.append(row)
        expected_energies = [energy_min + i * energy_width for i in range(energy_count)]
        for identity, energies in energies_by_identity.items():
            if len(energies) != energy_count or any(
                abs(lhs - rhs) > max(1.0e-7, abs(rhs) * 1.0e-7)
                for lhs, rhs in zip(energies, expected_energies)
            ):
                raise ValueError(f"section {section}: non-rectangular grid for {identity}")
    all_rows.sort(
        key=lambda row: (
            int(row["material_section"]), int(row["projectile_z"]),
            int(row["projectile_a"]), int(row["target_z"]),
            int(row["target_a"]), float(row["energy_MeV_per_u"]),
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    metadata: dict[str, object] = {
        "schema": "CINEL02_CT_MATERIAL_RATE_V1",
        "source_schneider_file": str(source_path),
        "source_schneider_sha256": _sha256(source_path),
        "sections": section_list,
        "section_count": len(section_list),
        "target_model": "pure H-1/O-16 custom elements; other Schneider elements retained",
        "transportable_projectiles": [list(item) for item in TRANSPORTABLE_PROJECTILES],
        "excluded_prompt_unstable_projectiles": [[4, 6]],
        "energy_grid": {
            "min_MeV_per_u": energy_min,
            "width_MeV_per_u": energy_width,
            "count": energy_count,
        },
        "reference_density_g_per_cm3": reference_density,
        "normalization": "Geant4 GetInelasticCrossSectionPerAtom times target number density",
        "rate_units": "macroscopic_cross_section_per_mm",
        "row_count": len(all_rows),
        "output_sha256": _sha256(output_csv),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _write_slurm(
    work_root: Path, sections: list[int], topas_bin: Path, threads: int,
) -> Path:
    script = work_root / "run_array.slurm"
    section_list = " ".join(str(section) for section in sections)
    content = f'''#!/usr/bin/env bash
# Generated locally; no remote host or cluster is used.
#SBATCH --job-name=ct-cinel02-rate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem=4G
#SBATCH --array=0-{len(sections)-1}%8
#SBATCH --output=/mnt/sda/%u/ct_cinel02_rate_%A_%a.log
#SBATCH --error=/mnt/sda/%u/ct_cinel02_rate_%A_%a.err
set -euo pipefail
sections=({section_list})
section="${{sections[$SLURM_ARRAY_TASK_ID]}}"
run_dir="{work_root}/section_$(printf '%02d' "$section")"
cd "$run_dir"
exec {shlex.quote(str(topas_bin))} "config.txt"
'''
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schneider-file", type=Path, default=DEFAULT_SCHNEIDER)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--topas-bin", type=Path, default=DEFAULT_TOPAS)
    parser.add_argument("--sections", default="0-24")
    parser.add_argument("--histories", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026090201)
    parser.add_argument("--energy-min", type=float, default=0.0)
    parser.add_argument("--energy-width", type=float, default=1.0)
    parser.add_argument("--energy-count", type=int, default=401)
    parser.add_argument("--reference-density", type=float, default=1.0)
    parser.add_argument("--generate", action="store_true", help="generate configs (default action)")
    parser.add_argument("--run", action="store_true", help="run generated configs directly")
    parser.add_argument("--submit", action="store_true", help="submit a local Slurm array")
    parser.add_argument("--compile", action="store_true", help="compile/validate output ntuples")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    if not args.generate and not args.run and not args.submit and not args.compile:
        args.generate = True
    if args.histories <= 0 or args.threads <= 0 or args.energy_count <= 0:
        parser.error("histories, threads and energy-count must be positive")
    if args.energy_min < 0.0 or args.energy_width <= 0.0 or args.reference_density <= 0.0:
        parser.error("energy-min/reference-density must be non-negative/positive and width positive")
    table = parse_schneider(args.schneider_file)
    sections = parse_sections(args.sections, table.section_count)
    args.work_root.mkdir(parents=True, exist_ok=True)
    manifest_runs: list[dict[str, object]] = []
    for offset, section in enumerate(sections):
        run_dir = args.work_root / f"section_{section:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = run_dir / "config.txt"
        output_stem = str(run_dir / "ct_cinel02_rate")
        render_config(
            table, section, config, output_stem, histories=args.histories,
            threads=args.threads, seed=args.seed + offset,
            energy_min=args.energy_min, energy_width=args.energy_width,
            energy_count=args.energy_count,
            reference_density=args.reference_density,
        )
        manifest_runs.append({"section": section, "config": str(config), "sha256": _sha256(config)})
    manifest = {
        "schema": "CINEL02_CT_MATERIAL_RATE_CAMPAIGN_V1",
        "schneider_file": str(args.schneider_file),
        "schneider_sha256": _sha256(args.schneider_file),
        "sections": sections,
        "histories": args.histories,
        "threads": args.threads,
        "energy_min_MeV_per_u": args.energy_min,
        "energy_width_MeV_per_u": args.energy_width,
        "energy_count": args.energy_count,
        "reference_density_g_per_cm3": args.reference_density,
        "runs": manifest_runs,
    }
    (args.work_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.run:
        if not args.topas_bin.is_file() or not os.access(args.topas_bin, os.X_OK):
            parser.error(f"TOPAS binary is not executable: {args.topas_bin}")
        for section in sections:
            run_dir = args.work_root / f"section_{section:02d}"
            with (run_dir / "topas.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [str(args.topas_bin), "config.txt"], cwd=run_dir,
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
    if args.submit:
        if not args.topas_bin.is_file() or not os.access(args.topas_bin, os.X_OK):
            parser.error(f"TOPAS binary is not executable: {args.topas_bin}")
        script = _write_slurm(args.work_root, sections, args.topas_bin, args.threads)
        result = subprocess.run(["sbatch", str(script)], check=True, text=True, capture_output=True)
        print(result.stdout.strip())
    if args.compile:
        output_csv = args.output_csv or args.work_root / "ct_cinel02_rates.csv"
        metadata = args.metadata or output_csv.with_suffix(".metadata.json")
        result = compile_outputs(
            args.work_root, sections, output_csv, metadata,
            source_path=args.schneider_file, energy_min=args.energy_min,
            energy_width=args.energy_width, energy_count=args.energy_count,
            reference_density=args.reference_density,
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
