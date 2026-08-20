#!/usr/bin/env python3
"""Convert one TOPAS proton-water reference run to strict depth CSVs.

The input is the output stem produced by ``proton_water_cascade.txt.in``.
TOPAS' XYZ CSV format is parsed from its grid metadata, not by assuming a
fixed number of rows.  The output files contain only the headers consumed by
``scripts/compare_water_ion_validation.py`` (plus the primary dose/energy
profiles needed for the reference audit).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


AXIS_RE = re.compile(
    r"^#\s*(?P<axis>[XYZ])\s+in\s+(?P<count>\d+)\s+bins?\s+of\s+"
    r"(?P<width>[0-9.eE+-]+)\s*(?P<unit>[A-Za-z]+)\s*$",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"^#\s*[^#(]+\((?P<unit>[^)]+)\)", re.IGNORECASE)
SCORER_RE = re.compile(r"^#\s*Results for scorer:\s*(?P<scorer>[^\s]+)\s*$", re.IGNORECASE)
HISTORY_RE = re.compile(r"Number of Original Histories:\s*(\d+)")


@dataclass(frozen=True)
class Grid:
    depth_mm: tuple[float, ...]
    values: tuple[float, ...]
    unit: str
    area_mm2: float
    scorer: str
    quantity: str


def _finite_nonnegative(value: str, path: Path, line_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: non-numeric value {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{path}:{line_number}: value must be finite and non-negative")
    return parsed


def _positive_index(value: str, path: Path, line_number: int, axis: str) -> int:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid {axis} index") from error
    if not math.isfinite(parsed) or parsed != math.floor(parsed) or parsed < 0.0:
        raise ValueError(f"{path}:{line_number}: invalid {axis} index")
    return int(parsed)


def parse_topas_grid(path: Path) -> Grid:
    """Read a one-voxel-wide TOPAS XYZ scorer and validate its depth grid."""
    if not path.is_file():
        raise FileNotFoundError(f"required TOPAS scorer is missing: {path}")
    axis: dict[str, tuple[int, float]] = {}
    unit = ""
    scorer = ""
    quantity = ""
    rows: dict[tuple[int, int, int], float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        match = AXIS_RE.match(line.strip())
        if match:
            name = match.group("axis").upper()
            width = float(match.group("width"))
            if not math.isfinite(width) or width <= 0.0:
                raise ValueError(f"{path}:{line_number}: invalid {name} bin width")
            unit_name = match.group("unit").lower()
            width_mm = width * (10.0 if unit_name == "cm" else 1.0 if unit_name == "mm" else 0.0)
            if width_mm <= 0.0:
                raise ValueError(f"{path}:{line_number}: unsupported {name} grid unit {unit_name!r}")
            if name in axis:
                raise ValueError(f"{path}: duplicate {name} grid declaration")
            axis[name] = (int(match.group("count")), width_mm)
            continue
        if line.lstrip().startswith("#"):
            scorer_match = SCORER_RE.match(line.strip())
            if scorer_match:
                scorer = scorer_match.group("scorer")
            comment = line.strip()[1:].strip()
            if comment.lower().endswith(": sum") and not comment.lower().startswith("results for scorer"):
                quantity = comment.split(" (", 1)[0].split(":", 1)[0].strip()
            if not unit:
                unit_match = UNIT_RE.match(line.strip())
                if unit_match:
                    unit = unit_match.group("unit").strip()
            continue
        if not line.strip():
            continue
        fields = next(csv.reader([line]))
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: TOPAS row requires x,y,z,value")
        x = _positive_index(fields[0].strip(), path, line_number, "X")
        y = _positive_index(fields[1].strip(), path, line_number, "Y")
        z = _positive_index(fields[2].strip(), path, line_number, "Z")
        if (x, y, z) in rows:
            raise ValueError(f"{path}:{line_number}: duplicate voxel index {(x, y, z)}")
        rows[(x, y, z)] = _finite_nonnegative(fields[3].strip(), path, line_number)

    if set(axis) != {"X", "Y", "Z"}:
        raise ValueError(f"{path}: TOPAS X/Y/Z grid declarations are required")
    x_count, x_width_mm = axis["X"]
    y_count, y_width_mm = axis["Y"]
    z_count, z_width_mm = axis["Z"]
    if x_count != 1 or y_count != 1:
        raise ValueError(f"{path}: reference scorer must have XBins=YBins=1")
    expected = x_count * y_count * z_count
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} voxels, got {len(rows)}")
    expected_keys = {(0, 0, z) for z in range(z_count)}
    if set(rows) != expected_keys:
        raise ValueError(f"{path}: voxel indices do not cover Z=0..{z_count - 1}")
    # The reference beam enters at +Z and travels toward -Z.  TOPAS writes
    # voxel rows in world-Z order, so reverse the rows into depth-from-entrance
    # order while keeping the public depth grid increasing from 0.
    depth = tuple((z + 0.5) * z_width_mm for z in range(z_count))
    if any(not math.isfinite(value) or value <= 0.0 for value in depth):
        raise ValueError(f"{path}: invalid depth grid")
    values = tuple(rows[(0, 0, z_count - 1 - z)] for z in range(z_count))
    return Grid(depth, values, unit, x_width_mm * y_width_mm, scorer, quantity)


def ensure_same_grid(reference: Grid, candidate: Grid, label: str) -> None:
    if len(reference.depth_mm) != len(candidate.depth_mm):
        raise ValueError(f"{label}: depth grid length mismatch")
    for index, (left, right) in enumerate(zip(reference.depth_mm, candidate.depth_mm)):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"{label}: depth grid mismatch at index {index}")


def write_profile(
    path: Path,
    depth: tuple[float, ...],
    values: tuple[float, ...],
    column: str,
    metadata: tuple[str, ...],
) -> None:
    if len(depth) != len(values) or not depth:
        raise ValueError(f"{path}: cannot write an empty or mismatched profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        for line in metadata:
            stream.write(f"# {line}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("depth_mm", column))
        for depth_mm, value in zip(depth, values):
            if not math.isfinite(depth_mm) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{path}: output profile contains an invalid value")
            writer.writerow((f"{depth_mm:.17g}", f"{value:.17g}"))


def parse_history_count(path: Path, expected: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required TOPAS cascade header is missing: {path}")
    match = HISTORY_RE.search(path.read_text(encoding="utf-8", errors="strict"))
    if match is None:
        raise ValueError(f"{path}: Number of Original Histories is required")
    actual = int(match.group(1))
    if actual != expected:
        raise ValueError(f"{path}: history count {actual} does not match {expected}")


def parse_primary_reactions(
    phsp: Path,
    header: Path,
    histories: int,
    projectile_z: int,
    projectile_a: int,
    depth: tuple[float, ...],
    depth_width_mm: float,
    phantom_half_length_mm: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    parse_history_count(header, histories)
    if not phsp.is_file():
        raise FileNotFoundError(f"required TOPAS cascade phase space is missing: {phsp}")
    first_by_event: dict[tuple[int, int], float] = {}
    rate = [0.0] * len(depth)
    maximum_depth = depth[-1] + depth_width_mm * 0.5
    for line_number, line in enumerate(phsp.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) not in (29, 30):
            raise ValueError(f"{phsp}:{line_number}: expected 29 or 30 columns, got {len(fields)}")
        kind = fields[0].lower()
        if kind not in {"interaction", "reaction"}:
            if kind not in {"product", "secondary"}:
                raise ValueError(f"{phsp}:{line_number}: unknown record kind {fields[0]!r}")
            continue
        try:
            run_id = int(fields[1])
            event_id = int(fields[3])
            parent_id = int(fields[7])
            record_z = int(fields[10])
            record_a = int(fields[11])
            offset = 1 if len(fields) == 30 else 0
            record_projectile_z = int(fields[16 + offset])
            record_projectile_a = int(fields[17 + offset])
            world_z_mm = float(fields[20 + offset])
        except (IndexError, ValueError) as error:
            raise ValueError(f"{phsp}:{line_number}: malformed interaction record") from error
        numeric_values = (world_z_mm, float(fields[13]), float(fields[14]))
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError(f"{phsp}:{line_number}: non-finite interaction value")
        if parent_id != 0 or (record_projectile_z, record_projectile_a) != (projectile_z, projectile_a):
            continue
        if (record_z, record_a) != (projectile_z, projectile_a):
            continue
        # The TOPAS proton source is placed at +phantom_half_length_mm and
        # travels toward -Z, hence physical depth is measured from +Z.
        depth_mm = phantom_half_length_mm - world_z_mm
        if depth_mm < -1.0e-9 or depth_mm > maximum_depth + 1.0e-9:
            raise ValueError(f"{phsp}:{line_number}: interaction depth {depth_mm:g} mm is outside scorer grid")
        bin_index = min(len(depth) - 1, max(0, int(math.floor(depth_mm / depth_width_mm))))
        rate[bin_index] += 1.0
        event_key = (run_id, event_id)
        previous = first_by_event.get(event_key)
        if previous is None or depth_mm < previous:
            first_by_event[event_key] = depth_mm

    first = [0.0] * len(depth)
    for depth_mm in first_by_event.values():
        bin_index = min(len(depth) - 1, max(0, int(math.floor(depth_mm / depth_width_mm))))
        first[bin_index] += 1.0
    return tuple(first), tuple(rate)


def survival_from_fluence(fluence: Grid) -> tuple[float, ...]:
    normalized = fluence.unit.replace(" ", "").lower()
    if normalized in {"/mm2", "1/mm2", "mm-2", "mm^-2"}:
        area = fluence.area_mm2
    elif normalized in {"/cm2", "1/cm2", "cm-2", "cm^-2"}:
        area = fluence.area_mm2 / 100.0
    else:
        raise ValueError(f"primary fluence: unsupported TOPAS unit {fluence.unit!r}")
    return tuple(value * area for value in fluence.values)


def validate_primary_crossing_count(crossing: Grid, histories: int) -> tuple[float, ...]:
    """Validate and normalize the per-history primary crossing scorer.

    TOPAS writes binned scorer values as floating-point text even when the
    custom scorer accumulates integer counts.  Accept only a small formatting
    tolerance, then expose exact integer-valued counts to downstream tools.
    """
    tolerance = max(1.0e-6, histories * 1.0e-12)
    counts: list[float] = []
    for index, value in enumerate(crossing.values):
        nearest = float(round(value))
        if abs(value - nearest) > tolerance:
            raise ValueError(
                f"primary crossing count: bin {index} value {value:g} is not an integer"
            )
        if nearest < 0.0 or nearest > float(histories):
            raise ValueError(
                f"primary crossing count: bin {index} value {nearest:g} is outside [0, {histories}]"
            )
        counts.append(nearest)

    # In a one-way uniform-water primary beam, a primary can disappear but
    # cannot be created at a deeper bin.  Keep a small tolerance for scorer
    # formatting, while rejecting a real depth-direction or identity error.
    for index, (previous, current) in enumerate(zip(counts, counts[1:]), 1):
        if current > previous + tolerance:
            raise ValueError(
                "primary crossing count: survival must be non-increasing with depth "
                f"(bin {index - 1}={previous:g}, bin {index}={current:g})"
            )
    return tuple(counts)


def ensure_quantity(grid: Grid, expected: str, label: str) -> None:
    if grid.quantity.lower() != expected.lower():
        raise ValueError(f"{label}: expected TOPAS quantity {expected!r}, got {grid.quantity or '<missing>'!r}")


def let_numerator(ratio: Grid, denominator: Grid, label: str) -> tuple[float, ...]:
    ensure_same_grid(ratio, denominator, f"{label}: ratio/denominator")
    values = tuple(value * weight for value, weight in zip(ratio.values, denominator.values))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{label}: reconstructed LET numerator is invalid")
    return values


def convert_em_only_reference(
    input_dir: Path,
    output_dir: Path,
    stem: str,
    histories: int,
    projectile_z: int,
    projectile_a: int,
) -> None:
    """Convert the three scorer outputs from an EM-only reference run.

    EM-only runs deliberately do not produce reaction, LET, or total-dose
    scorers.  Keep this path separate from the full nuclear reference
    conversion so missing nuclear files cannot be silently treated as zero.
    """
    base = input_dir / stem

    def source(suffix: str) -> Path:
        return base.with_name(base.name + suffix + ".csv")

    primary_energy = parse_topas_grid(source("_primary_energy_deposit"))
    primary_dose = parse_topas_grid(source("_primary_dose"))
    primary_crossing = parse_topas_grid(source("_primary_crossing_count"))
    ensure_quantity(primary_energy, "EnergyDeposit", "EM-only primary energy")
    ensure_quantity(primary_dose, "DoseToMedium", "EM-only primary dose")
    ensure_quantity(primary_crossing, "PrimaryCrossingCount", "EM-only primary crossing count")
    ensure_same_grid(primary_energy, primary_dose, "EM-only primary dose")
    ensure_same_grid(primary_energy, primary_crossing, "EM-only primary crossing count")
    survival = validate_primary_crossing_count(primary_crossing, histories)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = f"primary projectile (Z={projectile_z},A={projectile_a},generation=Primary)"
    write_profile(
        output_dir / "primary_energy_deposit.csv",
        primary_energy.depth_mm,
        primary_energy.values,
        "energy_deposition_MeV",
        ("scorer: EnergyDeposit", f"coverage: {coverage}"),
    )
    write_profile(
        output_dir / "primary_dose.csv",
        primary_dose.depth_mm,
        primary_dose.values,
        "dose_Gy",
        ("scorer: DoseToMedium", f"coverage: {coverage}"),
    )
    write_profile(
        output_dir / "primary_survival.csv",
        primary_crossing.depth_mm,
        survival,
        "count",
        (
            "scorer: PrimaryCrossingCount",
            "semantics: per-history primary crossing count; each history contributes at most once per depth bin",
            f"coverage: {coverage}",
        ),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "em_only",
                "histories": histories,
                "projectile": {"Z": projectile_z, "A": projectile_a},
                "depth_bins": len(primary_energy.depth_mm),
                "depth_bin_width_mm": primary_energy.depth_mm[1] - primary_energy.depth_mm[0],
                "scorers": {
                    "primary_energy_deposit": "primary_energy_deposit.csv",
                    "primary_dose": "primary_dose.csv",
                    "primary_survival": "primary_survival.csv",
                },
                "nuclear_scorers": "not produced by EM-only physics list",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True, help="TOPAS output stem without scorer suffix")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--phantom-half-length-mm", type=float, required=True)
    parser.add_argument("--projectile-z", type=int, default=1)
    parser.add_argument("--projectile-a", type=int, default=1)
    parser.add_argument(
        "--em-only",
        action="store_true",
        help="convert only primary energy, dose, and crossing-count scorers",
    )
    args = parser.parse_args()
    if args.histories <= 0 or args.phantom_half_length_mm <= 0.0:
        raise SystemExit("histories and phantom half length must be positive")
    if args.projectile_z <= 0 or args.projectile_a <= 0:
        raise SystemExit("projectile Z/A must be positive")

    if args.em_only:
        convert_em_only_reference(
            args.input_dir,
            args.output_dir,
            args.stem,
            args.histories,
            args.projectile_z,
            args.projectile_a,
        )
        return

    base = args.input_dir / args.stem

    def source(suffix: str) -> Path:
        return base.with_name(base.name + suffix + ".csv")

    total_energy = parse_topas_grid(source("_total_energy_deposit"))
    primary_energy = parse_topas_grid(source("_primary_proton_energy_deposit"))
    total_dose = parse_topas_grid(source("_total_dose"))
    primary_dose = parse_topas_grid(source("_primary_proton_dose"))
    primary_let = parse_topas_grid(source("_primary_let"))
    primary_let_denominator = parse_topas_grid(source("_primary_let_denominator"))
    all_hadron_let = parse_topas_grid(source("_all_hadron_let"))
    all_hadron_let_denominator = parse_topas_grid(source("_all_hadron_let_denominator"))
    primary_crossing = parse_topas_grid(source("_primary_crossing_count"))
    primary_fluence = parse_topas_grid(source("_primary_fluence"))
    ensure_quantity(primary_let, "myHadronLET", "primary LET")
    ensure_quantity(primary_let_denominator, "myHadronLET_Denominator", "primary LET denominator")
    ensure_quantity(all_hadron_let, "myHadronLET", "all charged-ion LET")
    ensure_quantity(all_hadron_let_denominator, "myHadronLET_Denominator", "all charged-ion LET denominator")
    ensure_quantity(primary_crossing, "PrimaryCrossingCount", "primary crossing count")
    for label, candidate in (
        ("primary energy", primary_energy),
        ("total dose", total_dose),
        ("primary dose", primary_dose),
        ("primary LET", primary_let),
        ("primary LET denominator", primary_let_denominator),
        ("all-hadron LET", all_hadron_let),
        ("all-hadron LET denominator", all_hadron_let_denominator),
        ("primary crossing count", primary_crossing),
        ("primary fluence", primary_fluence),
    ):
        ensure_same_grid(total_energy, candidate, label)

    if len(total_energy.depth_mm) < 2:
        raise ValueError("TOPAS reference needs at least two depth bins")
    depth_width_mm = total_energy.depth_mm[1] - total_energy.depth_mm[0]
    if not math.isfinite(depth_width_mm) or depth_width_mm <= 0.0:
        raise ValueError("TOPAS reference depth grid must be strictly increasing")
    cascade = base.with_name(base.name + "_cascade")
    first_reactions, reaction_rate = parse_primary_reactions(
        cascade.with_suffix(".phsp"),
        cascade.with_suffix(".header"),
        args.histories,
        args.projectile_z,
        args.projectile_a,
        total_energy.depth_mm,
        depth_width_mm,
        args.phantom_half_length_mm,
    )
    survival = validate_primary_crossing_count(primary_crossing, args.histories)
    fluence_survival_diagnostic = survival_from_fluence(primary_fluence)
    primary_let_numerator = let_numerator(primary_let, primary_let_denominator, "primary LET")
    all_hadron_let_numerator = let_numerator(
        all_hadron_let, all_hadron_let_denominator, "all charged-ion LET"
    )
    output = args.output_dir
    write_profile(output / "total_energy_deposit.csv", total_energy.depth_mm, total_energy.values, "energy_deposition_MeV", ("scorer: EnergyDeposit", "coverage: all particles in G4_WATER"))
    write_profile(output / "primary_proton_energy_deposit.csv", primary_energy.depth_mm, primary_energy.values, "energy_deposition_MeV", ("scorer: EnergyDeposit", "coverage: primary proton (Z=1,A=1,generation=Primary)"))
    write_profile(output / "total_dose.csv", total_dose.depth_mm, total_dose.values, "dose_Gy", ("scorer: DoseToMedium", "coverage: all particles in G4_WATER"))
    write_profile(output / "primary_proton_dose.csv", primary_dose.depth_mm, primary_dose.values, "dose_Gy", ("scorer: DoseToMedium", "coverage: primary proton (Z=1,A=1,generation=Primary)"))
    write_profile(output / "primary_let.csv", primary_let.depth_mm, primary_let_numerator, "numerator_MeV", (f"scorer: {primary_let.scorer}", f"quantity: {primary_let.quantity}", "weighting: dose", "moment: numerator reconstructed from ratio x denominator", "coverage: primary proton (Z=1,A=1,generation=Primary)", "denominator: primary_let_denominator.csv"))
    write_profile(output / "primary_let_denominator.csv", primary_let_denominator.depth_mm, primary_let_denominator.values, "denominator_MeV", (f"scorer: {primary_let_denominator.scorer}", f"quantity: {primary_let_denominator.quantity}", "weighting: dose", "coverage: primary proton (Z=1,A=1,generation=Primary)"))
    write_profile(output / "all_hadron_let.csv", all_hadron_let.depth_mm, all_hadron_let_numerator, "numerator_MeV", (f"scorer: {all_hadron_let.scorer}", f"quantity: {all_hadron_let.quantity}", "weighting: dose", "moment: numerator reconstructed from ratio x denominator", "coverage: charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded", "denominator: all_hadron_let_denominator.csv"))
    write_profile(output / "all_hadron_let_denominator.csv", all_hadron_let_denominator.depth_mm, all_hadron_let_denominator.values, "denominator_MeV", (f"scorer: {all_hadron_let_denominator.scorer}", f"quantity: {all_hadron_let_denominator.quantity}", "weighting: dose", "coverage: charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded"))
    write_profile(output / "primary_survival.csv", primary_crossing.depth_mm, survival, "count", ("scorer: PrimaryCrossingCount", "semantics: per-history primary crossing count; each history contributes at most once per depth bin", "coverage: primary projectile (Z=1,A=1,generation=Primary)"))
    write_profile(output / "primary_fluence_diagnostic.csv", primary_fluence.depth_mm, fluence_survival_diagnostic, "count", ("scorer: Fluence", "semantics: diagnostic track-length fluence converted to an estimated count; not formal survival", "coverage: primary proton (Z=1,A=1,generation=Primary)"))
    write_profile(output / "first_reactions.csv", total_energy.depth_mm, first_reactions, "count", ("scorer: CarbonCascadeNtuple", "coverage: primary proton inelastic interactions (Z=1,A=1)"))
    write_profile(output / "reaction_rate.csv", total_energy.depth_mm, reaction_rate, "count", ("scorer: CarbonCascadeNtuple", "coverage: primary proton inelastic interaction records (Z=1,A=1)"))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "histories": args.histories,
                "projectile": {"Z": args.projectile_z, "A": args.projectile_a},
                "depth_bins": len(total_energy.depth_mm),
                "depth_bin_width_mm": depth_width_mm,
                "let_definition": {
                    "weighting": "dose",
                    "scorer": "myHadronLET",
                    "denominator_scorer": "myHadronLET_Denominator",
                    "all_hadron_coverage": "charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded",
                },
                "scorers": {
                    "primary_survival": {
                        "file": "primary_survival.csv",
                        "scorer": primary_crossing.scorer,
                        "quantity": primary_crossing.quantity,
                        "semantics": "per-history primary crossing count; each history contributes at most once per depth bin",
                        "coverage": "primary projectile (Z=1,A=1,generation=Primary)",
                        "monotonic_depth": "non-increasing for one-way uniform-water primary transport",
                    },
                    "primary_fluence_diagnostic": {
                        "file": "primary_fluence_diagnostic.csv",
                        "scorer": primary_fluence.scorer,
                        "quantity": primary_fluence.quantity,
                        "semantics": "diagnostic track-length fluence converted to an estimated count; not formal survival",
                        "coverage": "primary proton (Z=1,A=1,generation=Primary)",
                    },
                    "primary_let": {
                        "file": "primary_let.csv",
                        "scorer": primary_let.scorer,
                        "quantity": primary_let.quantity,
                        "coverage": "primary proton (Z=1,A=1,generation=Primary)",
                        "denominator": "primary_let_denominator.csv",
                    },
                    "primary_let_denominator": {
                        "file": "primary_let_denominator.csv",
                        "scorer": primary_let_denominator.scorer,
                        "quantity": primary_let_denominator.quantity,
                        "coverage": "primary proton (Z=1,A=1,generation=Primary)",
                    },
                    "all_hadron_let": {
                        "file": "all_hadron_let.csv",
                        "scorer": all_hadron_let.scorer,
                        "quantity": all_hadron_let.quantity,
                        "coverage": "charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded",
                        "denominator": "all_hadron_let_denominator.csv",
                    },
                    "all_hadron_let_denominator": {
                        "file": "all_hadron_let_denominator.csv",
                        "scorer": all_hadron_let_denominator.scorer,
                        "quantity": all_hadron_let_denominator.quantity,
                        "coverage": "charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded",
                    },
                },
                "outputs": {
                    "dose": "total_energy_deposit.csv",
                    "primary_dose": "primary_proton_energy_deposit.csv",
                    "primary_let": "primary_let.csv",
                    "all_hadron_let": "all_hadron_let.csv",
                    "primary_let_denominator": "primary_let_denominator.csv",
                    "all_hadron_let_denominator": "all_hadron_let_denominator.csv",
                    "primary_survival": "primary_survival.csv",
                    "primary_fluence_diagnostic": "primary_fluence_diagnostic.csv",
                    "first_reactions": "first_reactions.csv",
                    "reaction_rate": "reaction_rate.csv",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"converted TOPAS proton reference: {output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
