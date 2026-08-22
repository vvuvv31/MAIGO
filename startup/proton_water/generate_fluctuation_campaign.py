#!/usr/bin/env python3
"""Render a reproducible EM-only thin-water-slab fluctuation campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "maigo-energy-loss-fluctuation-campaign-v1"
POINT_SCHEMA = "maigo-energy-loss-fluctuation-run-v1"
PHYSICS_MODULES = ["g4em-standard_opt4"]
MATERIAL = "G4_WATER"
WATER_DENSITY_G_PER_CM3 = 1.0
STEP_CONTROL_FORMULA = (
    "max_step_mm = min(maximum_step_mm, "
    "maximum_relative_energy_loss * (A * energy_MeV_per_u) / "
    "stopping_power_MeV_per_mm); "
    "areal_density_g_per_cm2 = water_density_g_per_cm3 * "
    "max_step_mm / 10 * density_fraction"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_grid(value: str) -> list[float]:
    try:
        result = [finite_positive(field.strip()) for field in value.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            "grid must contain comma-separated finite positive numbers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("grid must not be empty")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return sorted(result)


def parse_density_fractions(value: str) -> list[float]:
    result = parse_grid(value)
    if len(result) < 2:
        raise argparse.ArgumentTypeError(
            "density fractions must contain at least two values"
        )
    if result[-1] > 1.0:
        raise argparse.ArgumentTypeError(
            "density fractions must be no greater than one"
        )
    return result


def read_stopping_power_table(path: Path) -> list[tuple[float, float]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"stopping-power table is missing: {resolved}")
    samples: list[tuple[float, float]] = []
    with resolved.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["energy_MeVu", "stopping_power_MeV_per_mm"]
        if reader.fieldnames != expected:
            raise ValueError(
                f"stopping-power table must have columns {expected}: {resolved}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                energy = float(row[expected[0]])
                stopping_power = float(row[expected[1]])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid stopping-power row at line {line_number}: {resolved}"
                ) from error
            if not all(map(math.isfinite, (energy, stopping_power))):
                raise ValueError(
                    f"nonfinite stopping-power row at line {line_number}: {resolved}"
                )
            if energy <= 0.0 or stopping_power <= 0.0:
                raise ValueError(
                    f"stopping-power values must be positive at line "
                    f"{line_number}: {resolved}"
                )
            if samples and energy <= samples[-1][0]:
                relation = "duplicate" if energy == samples[-1][0] else "unsorted"
                raise ValueError(
                    f"{relation} stopping-power energy at line {line_number}: "
                    f"{resolved}"
                )
            samples.append((energy, stopping_power))
    if len(samples) < 2:
        raise ValueError(
            f"stopping-power table must contain at least two samples: {resolved}"
        )
    return samples


def interpolate_stopping_power(
    samples: list[tuple[float, float]], energy: float
) -> float:
    if energy < samples[0][0] or energy > samples[-1][0]:
        raise ValueError(
            f"campaign energy {energy:g} MeV/u is outside stopping-power table "
            f"range [{samples[0][0]:g}, {samples[-1][0]:g}] MeV/u"
        )
    for index in range(1, len(samples)):
        upper_energy, upper_value = samples[index]
        if energy <= upper_energy:
            lower_energy, lower_value = samples[index - 1]
            fraction = (energy - lower_energy) / (upper_energy - lower_energy)
            return lower_value + fraction * (upper_value - lower_value)
    return samples[-1][1]


def projectile_expression(name: str | None, z: int, a: int) -> str:
    expected = "proton" if (z, a) == (1, 1) else f"GenericIon({z},{a})"
    if name is None:
        return expected
    if name != expected:
        raise ValueError(
            f"projectile name {name!r} does not match requested Z{z}A{a}; "
            f"expected {expected!r}"
        )
    return name


def numeric_tag(value: float) -> str:
    return (
        format(value, ".12g")
        .replace(".", "p")
        .replace("+", "")
        .replace("-", "m")
    )


def render(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved template tokens: {unresolved}")
    return rendered


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=here / "energy_loss_fluctuation_thin_slab.txt.in",
    )
    parser.add_argument("--projectile-z", type=positive_integer, required=True)
    parser.add_argument("--projectile-a", type=positive_integer, required=True)
    parser.add_argument("--projectile-name")
    parser.add_argument("--energies-mevu", type=parse_grid, required=True)
    density_mode = parser.add_mutually_exclusive_group(required=True)
    density_mode.add_argument("--areal-densities-g-per-cm2", type=parse_grid)
    density_mode.add_argument("--stopping-power-file", type=Path)
    parser.add_argument("--maximum-step-mm", type=finite_positive, default=0.025)
    parser.add_argument(
        "--maximum-relative-energy-loss", type=finite_positive, default=0.005
    )
    parser.add_argument(
        "--density-fractions",
        type=parse_density_fractions,
        default=parse_density_fractions("0.125,0.25,0.5,1"),
    )
    parser.add_argument("--histories", type=positive_integer, required=True)
    parser.add_argument("--threads", type=positive_integer, default=56)
    parser.add_argument("--seed-base", type=positive_integer, required=True)
    parser.add_argument(
        "--production-cut-mm", type=finite_positive, default=0.05
    )
    parser.add_argument("--topas-version", required=True)
    parser.add_argument("--geant4-version", required=True)
    args = parser.parse_args()
    if args.projectile_a < args.projectile_z:
        parser.error("projectile A must be at least Z")
    if args.threads > 56:
        parser.error("threads must be in 1..56")
    if args.maximum_relative_energy_loss > 1.0:
        parser.error("maximum relative energy loss must be no greater than one")
    return args


def build_density_grid(
    args: argparse.Namespace,
) -> tuple[dict[float, list[float]], dict[str, Any]]:
    if args.stopping_power_file is None:
        densities = args.areal_densities_g_per_cm2
        return (
            {energy: densities for energy in args.energies_mevu},
            {
                "density_mode": "explicit_common",
                "areal_densities_g_per_cm2": densities,
            },
        )

    table_path = args.stopping_power_file.resolve()
    samples = read_stopping_power_table(table_path)
    densities_by_energy: dict[float, list[float]] = {}
    for energy in args.energies_mevu:
        stopping_power = interpolate_stopping_power(samples, energy)
        total_energy = args.projectile_a * energy
        maximum_density = (
            WATER_DENSITY_G_PER_CM3
            * min(
                args.maximum_step_mm,
                args.maximum_relative_energy_loss
                * total_energy
                / stopping_power,
            )
            / 10.0
        )
        densities_by_energy[energy] = [
            maximum_density * fraction for fraction in args.density_fractions
        ]

    grid = {
        "density_mode": "step_control",
        "areal_densities_by_energy_g_per_cm2": [
            {"energy_MeV_per_u": energy, "values": densities_by_energy[energy]}
            for energy in args.energies_mevu
        ],
        "step_control": {
            "maximum_step_mm": args.maximum_step_mm,
            "maximum_relative_energy_loss": args.maximum_relative_energy_loss,
            "density_fractions": args.density_fractions,
            "stopping_power_file": {
                "path": str(table_path),
                "sha256": sha256(table_path),
            },
            "formula": STEP_CONTROL_FORMULA,
        },
    }
    return densities_by_energy, grid


def build_campaign(args: argparse.Namespace) -> dict[str, Any]:
    template_path = args.template.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"template is missing: {template_path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    projectile = projectile_expression(
        args.projectile_name, args.projectile_z, args.projectile_a
    )
    densities_by_energy, density_grid_manifest = build_density_grid(args)
    point_count = sum(map(len, densities_by_energy.values()))
    if args.seed_base + point_count - 1 > 2_147_483_647:
        raise ValueError("campaign seeds exceed signed 32-bit range")
    template = template_path.read_text(encoding="utf-8")
    plans: list[dict[str, Any]] = []
    labels: set[str] = set()
    ordinal = 0
    for energy in args.energies_mevu:
        for areal_density in densities_by_energy[energy]:
            full_thickness_mm = (
                10.0 * areal_density / WATER_DENSITY_G_PER_CM3
            )
            half_thickness_mm = 0.5 * full_thickness_mm
            source_z_mm = half_thickness_mm + 1.0
            if source_z_mm >= 100.0:
                raise ValueError(
                    f"areal density {areal_density:g} g/cm2 places the source "
                    "outside the 100 mm world half-length"
                )
            label = (
                f"e{numeric_tag(energy)}mevu_"
                f"ad{numeric_tag(areal_density)}gcm2"
            )
            if label in labels:
                raise ValueError(f"numeric tag collision for campaign point {label}")
            labels.add(label)
            seed = args.seed_base + ordinal
            ordinal += 1
            output_stem = "energy_loss_fluctuation"
            rendered = render(
                template,
                {
                    "SEED": str(seed),
                    "THREADS": str(args.threads),
                    "HISTORIES": str(args.histories),
                    "PROJECTILE_NAME": projectile,
                    "PROJECTILE_Z": str(args.projectile_z),
                    "PROJECTILE_A": str(args.projectile_a),
                    "TOTAL_ENERGY_MEV": format(
                        energy * args.projectile_a, ".17g"
                    ),
                    "SLAB_HALF_THICKNESS_MM": format(
                        half_thickness_mm, ".17g"
                    ),
                    "SOURCE_Z_MM": format(source_z_mm, ".17g"),
                    "PRODUCTION_CUT_MM": format(
                        args.production_cut_mm, ".17g"
                    ),
                    "OUTPUT_STEM": output_stem,
                },
            )
            plans.append(
                {
                    "label": label,
                    "energy_MeV_per_u": energy,
                    "total_energy_MeV": energy * args.projectile_a,
                    "areal_density_g_per_cm2": areal_density,
                    "slab_full_thickness_mm": full_thickness_mm,
                    "slab_half_thickness_mm": half_thickness_mm,
                    "source_z_mm": source_z_mm,
                    "seed": seed,
                    "output_stem": output_stem,
                    "rendered": rendered,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_points: list[dict[str, Any]] = []
    for plan in plans:
        run_dir = output_dir / "runs" / plan["label"]
        run_dir.mkdir(parents=True)
        config_path = run_dir / "run.txt"
        config_path.write_text(plan.pop("rendered"), encoding="utf-8")
        run_manifest = {
            "schema": POINT_SCHEMA,
            "projectile": {
                "name": projectile,
                "Z": args.projectile_z,
                "A": args.projectile_a,
            },
            "material": MATERIAL,
            "physics": {
                "modules": PHYSICS_MODULES,
                "nuclear_processes_enabled": False,
                "production_cut_mm": args.production_cut_mm,
                "target_versions": {
                    "TOPAS": args.topas_version,
                    "Geant4": args.geant4_version,
                },
            },
            "histories": args.histories,
            "threads": args.threads,
            **plan,
            "files": {
                "config": {
                    "path": str(config_path.relative_to(output_dir)),
                    "sha256": sha256(config_path),
                },
                "expected_header": f"{plan['output_stem']}.header",
                "expected_phsp": f"{plan['output_stem']}.phsp",
                "runtime_log": "topas.log",
            },
        }
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_points.append(
            {
                "label": plan["label"],
                "energy_MeV_per_u": plan["energy_MeV_per_u"],
                "areal_density_g_per_cm2": plan["areal_density_g_per_cm2"],
                "seed": plan["seed"],
                "run_manifest": {
                    "path": str(run_manifest_path.relative_to(output_dir)),
                    "sha256": sha256(run_manifest_path),
                },
            }
        )

    campaign = {
        "schema": SCHEMA,
        "projectile": {
            "name": projectile,
            "Z": args.projectile_z,
            "A": args.projectile_a,
        },
        "material": MATERIAL,
        "material_density_g_per_cm3": WATER_DENSITY_G_PER_CM3,
        "physics": {
            "modules": PHYSICS_MODULES,
            "nuclear_processes_enabled": False,
            "production_cut_mm": args.production_cut_mm,
            "target_versions": {
                "TOPAS": args.topas_version,
                "Geant4": args.geant4_version,
            },
        },
        "grid": {
            "energies_MeV_per_u": args.energies_mevu,
            **density_grid_manifest,
            "point_count": len(plans),
        },
        "histories_per_point": args.histories,
        "threads": args.threads,
        "seed_base": args.seed_base,
        "template": {"path": str(template_path), "sha256": sha256(template_path)},
        "points": manifest_points,
    }
    campaign_path = output_dir / "campaign_manifest.json"
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return campaign


def main() -> None:
    args = parse_args()
    try:
        campaign = build_campaign(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"generated fluctuation campaign: {args.output_dir.resolve()} "
        f"({campaign['grid']['point_count']} points)"
    )


if __name__ == "__main__":
    main()
