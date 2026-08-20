#!/usr/bin/env python3
"""Render or run isolated multi-energy TOPAS hadronic-elastic extractions."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ENERGIES = "70,100,150,200,250"
DEFAULT_MODULES = (
    "g4em-standard_opt4",
    "g4h-phy_QGSP_BIC_HP",
    "g4decay",
    "g4h-elastic_HP",
    "g4stopping",
)
FIXED_ENERGY_MODULES = (
    "g4h-phy_QGSP_BIC_HP",
    "g4h-elastic_HP",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_energies(value: str) -> list[float]:
    try:
        energies = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("energies must be comma-separated numbers") from error
    if not energies or any(energy <= 0.0 for energy in energies):
        raise argparse.ArgumentTypeError("energies must be positive")
    if len(set(energies)) != len(energies):
        raise argparse.ArgumentTypeError("energies must be unique")
    return energies


def slug(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
        raise SystemExit(f"{label} contains unsafe characters: {value!r}")
    return value


def projectile_expression(value: str | None, z: int, a: int) -> str:
    if value is None:
        return "proton" if (z, a) == (1, 1) else f"GenericIon({z},{a})"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", value):
        return value
    expected = f"GenericIon({z},{a})"
    if value == expected:
        return value
    raise SystemExit(f"Unsafe or identity-mismatched TOPAS projectile expression: {value!r}")


def energy_tag(energy: float) -> str:
    return f"{energy:g}".replace(".", "p")


def render(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", rendered)))
    if unresolved:
        raise SystemExit(f"Unresolved template tokens: {unresolved}")
    return rendered


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan (default)")
    mode.add_argument("--run", action="store_true", help="run TOPAS and prepare each energy")
    parser.add_argument("--projectile-name", default=None,
                        help="TOPAS name; defaults to proton or GenericIon(Z,A)")
    parser.add_argument("--projectile-z", type=positive_int, default=1)
    parser.add_argument("--projectile-a", type=positive_int, default=1)
    parser.add_argument("--material", default="G4_WATER")
    parser.add_argument("--energies-mevu", type=parse_energies, default=parse_energies(DEFAULT_ENERGIES))
    parser.add_argument("--histories", type=positive_int, default=1_000_000)
    parser.add_argument("--threads", type=positive_int, default=56)
    parser.add_argument("--remote-root", type=Path, default=Path("/home/v/maigo_elastic_campaign"))
    parser.add_argument("--topas-bin", type=Path)
    parser.add_argument("--template", type=Path, default=here / "elastic_only.txt.in")
    parser.add_argument("--tools-dir", type=Path, default=here.parent / "package_tools")
    parser.add_argument("--physics-model", default="G4HadronElasticPhysicsHP")
    parser.add_argument("--physics-modules", nargs="+", default=list(DEFAULT_MODULES))
    parser.add_argument(
        "--extraction-mode", choices=("transport-degrading", "fixed-energy-package-sampling"),
        default="transport-degrading",
        help="fixed-energy package sampling disables continuous EM loss/MSC and is not a dose reference",
    )
    parser.add_argument("--seed-base", type=int, default=2026082000)
    args = parser.parse_args()
    if args.projectile_a < args.projectile_z:
        parser.error("projectile A must be >= Z")
    if args.extraction_mode == "fixed-energy-package-sampling":
        if args.physics_modules != list(DEFAULT_MODULES):
            parser.error("fixed-energy-package-sampling does not accept custom physics modules")
        args.physics_modules = list(FIXED_ENERGY_MODULES)
    return args


def main() -> None:
    args = parse_args()
    projectile = projectile_expression(args.projectile_name, args.projectile_z, args.projectile_a)
    projectile_tag = "proton" if (args.projectile_z, args.projectile_a) == (1, 1) else f"ion_z{args.projectile_z}a{args.projectile_a}"
    material_tag = slug(args.material, "material").lower()
    root = args.remote_root.resolve()
    topas = (args.topas_bin or root / "topas-install/bin/topas").resolve()
    template = args.template.read_text(encoding="utf-8")
    modules = " ".join(json.dumps(module) for module in args.physics_modules)
    continuous_em_loss_enabled = args.extraction_mode == "transport-degrading"
    sampling_purpose = (
        "elastic-package-fixed-energy-sampling-not-dose-reference"
        if not continuous_em_loss_enabled else "elastic-package-transport-degrading-extraction"
    )
    if not args.run:
        print(f"[dry-run] projectile={projectile} Z={args.projectile_z} A={args.projectile_a}")
        print(f"[dry-run] extraction_mode={args.extraction_mode} physics_modules={args.physics_modules}")

    for ordinal, energy in enumerate(args.energies_mevu):
        seed = args.seed_base + ordinal
        label = f"{projectile_tag}_{energy_tag(energy)}mevu_{material_tag}"
        run_dir = root / "runs" / label
        config = run_dir / "run.txt"
        log = run_dir / "topas.log"
        output_stem = label
        prepared_interactions = run_dir / f"{label}_interactions.csv.gz"
        prepared_products = run_dir / f"{label}_products.csv.gz"
        prepared_metadata = run_dir / f"{label}.metadata.json"
        rendered = render(template, {
            "SEED": str(seed), "THREADS": str(args.threads),
            "HISTORIES": str(args.histories), "PROJECTILE_NAME": projectile,
            "PROJECTILE_Z": str(args.projectile_z), "PROJECTILE_A": str(args.projectile_a),
            "TOTAL_ENERGY_MEV": f"{energy * args.projectile_a:g}",
            "MATERIAL": args.material, "OUTPUT_STEM": output_stem,
            "PHYSICS_MODULE_COUNT": str(len(args.physics_modules)), "PHYSICS_MODULES": modules,
            "EXTRACTION_MODE": args.extraction_mode,
            "SAMPLING_PURPOSE": sampling_purpose,
            "CONTINUOUS_EM_LOSS_ENABLED": str(continuous_em_loss_enabled).lower(),
        })
        prepare = [
            sys.executable, str(args.tools_dir / "prepare_topas_elastic.py"),
            "--case", label, "--histories", str(args.histories),
            "--input-dir", str(run_dir), "--stem", f"{label}_elastic",
            "--projectile-z", str(args.projectile_z), "--projectile-a", str(args.projectile_a),
            "--material", args.material, "--physics-model", args.physics_model,
            "--interactions-output", str(prepared_interactions),
            "--products-output", str(prepared_products), "--metadata", str(prepared_metadata),
            "--phantom-half-length-mm", "350", "--runtime-log", str(log),
            "--extraction-mode", args.extraction_mode,
            "--sampling-purpose", sampling_purpose,
            "--continuous-em-loss-enabled", str(continuous_em_loss_enabled).lower(),
        ]
        if not args.run:
            print(f"[dry-run] energy={energy:g} MeV/u total={energy * args.projectile_a:g} MeV")
            print(f"[dry-run] config {config}")
            print(f"[dry-run] TOPAS {shlex.join([str(topas), str(config)])}")
            print(f"[dry-run] prepare {shlex.join(prepare)}")
            continue

        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
        if not topas.is_file():
            raise SystemExit(f"TOPAS executable not found: {topas}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config.write_text(rendered, encoding="utf-8")
        lock_path = root / ".elastic_campaign.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(f"Another elastic campaign owns {lock_path}") from error
            with log.open("w", encoding="utf-8") as output:
                subprocess.run([str(topas), str(config)], cwd=run_dir, stdout=output,
                               stderr=subprocess.STDOUT, check=True)
        subprocess.run(prepare, check=True)
        manifest = {
            "schema_version": 1, "incident_energy_MeV_per_u": energy,
            "histories": args.histories, "seed": seed,
            "projectile": {"name": projectile, "Z": args.projectile_z, "A": args.projectile_a},
            "material": args.material, "physics_model": args.physics_model,
            "physics_modules": args.physics_modules, "prepared_metadata": str(prepared_metadata),
            "extraction_mode": args.extraction_mode,
            "sampling_purpose": sampling_purpose,
            "continuous_em_loss_enabled": continuous_em_loss_enabled,
            "not_for_dose_reference": not continuous_em_loss_enabled,
        }
        (run_dir / "campaign_run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"prepared {prepared_metadata}")


if __name__ == "__main__":
    main()
