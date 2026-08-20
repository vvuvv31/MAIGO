#!/usr/bin/env python3
"""Render proton water resolved MAIGO configs and record independent seeds.

The script deliberately writes only under out/.  It neither generates nor
copies packages: package generation and TOPAS reference jobs run remotely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ENERGIES_MEV = (70, 100, 150, 200, 250)
PROTON_Z = 1
PROTON_A = 1
PROTON_REST_MASS_MEV = 938.27208816


def artifact_record(path: Path) -> dict[str, object]:
    """Describe an expected campaign artifact without requiring it to exist."""
    result: dict[str, object] = {"path": str(path), "sha256": None, "bytes": None}
    if not path.is_file():
        return result
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result["sha256"] = digest.hexdigest()
    result["bytes"] = path.stat().st_size
    return result


def compiled_sidecar_path(package: Path) -> Path:
    return package.with_name(package.stem + ".compiled.json")


def render(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)
    unresolved = [line for line in template.splitlines() if "@" in line]
    if unresolved:
        raise ValueError(f"unresolved template marker: {unresolved[0]}")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path, default=Path("config/proton_water_qgsp_bic_hp.yaml.template")
    )
    parser.add_argument("--output-root", type=Path, default=Path("out/proton_water_qgsp_bic_hp"))
    parser.add_argument("--histories", type=int, default=1_000_000)
    parser.add_argument("--gpu-seed-base", type=int, default=2026084900)
    parser.add_argument("--topas-package-seed-base", type=int, default=2026082900)
    parser.add_argument("--topas-reference-seed-base", type=int, default=2026083900)
    parser.add_argument("--enable-xs-correction", action="store_true")
    args = parser.parse_args()
    if args.histories <= 0:
        raise SystemExit("--histories must be positive")

    template = args.template.read_text(encoding="utf-8")
    root = args.output_root
    configs_dir = root / "resolved_configs"
    packages_dir = root / "packages"
    tables_dir = root / "tables"
    results_dir = root / "gpu"
    configs_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    for energy in ENERGIES_MEV:
        case_id = f"proton_water_{energy}MeV"
        package_prefix = packages_dir / case_id
        correction = tables_dir / f"{case_id}_primary_inelastic_xs_correction.csv"
        values = {
            "HISTORIES": str(args.histories),
            "ENERGY_MEV": str(energy),
            "GPU_SEED": str(args.gpu_seed_base + energy),
            "STOPPING_POWER": str(tables_dir / f"{case_id}_stopping_power.csv"),
            "INELASTIC_XS": str(tables_dir / f"{case_id}_inelastic_xs.csv"),
            "PRIMARY_PACKAGE": str(package_prefix) + "_primary_3d.bin",
            "CASCADE_PACKAGE": str(package_prefix) + "_cascade_3d.bin",
            "ENABLE_XS_CORRECTION": "true" if args.enable_xs_correction else "false",
            "XS_CORRECTION": str(correction) if args.enable_xs_correction else "",
            "OUTPUT_DIR": str(results_dir / f"{energy}MeV"),
        }
        config_path = configs_dir / f"{case_id}.yaml"
        rendered = render(template, values)
        config_path.write_text(rendered, encoding="utf-8")
        cases.append(
            {
                "case": case_id,
                "energy_MeV": energy,
                "projectile": {"Z": PROTON_Z, "A": PROTON_A, "rest_mass_MeV": PROTON_REST_MASS_MEV},
                "histories": args.histories,
                "gpu_seed": args.gpu_seed_base + energy,
                "topas_package_seed": args.topas_package_seed_base + energy,
                "topas_reference_seed": args.topas_reference_seed_base + energy,
                "resolved_config": artifact_record(config_path),
                "package_paths": {
                    "primary": values["PRIMARY_PACKAGE"],
                    "cascade": values["CASCADE_PACKAGE"],
                },
                "xs_correction": {
                    "enabled": args.enable_xs_correction,
                    "path": values["XS_CORRECTION"] or None,
                },
                "artifacts": {
                    "stopping_power": artifact_record(Path(values["STOPPING_POWER"])),
                    "inelastic_cross_section": artifact_record(Path(values["INELASTIC_XS"])),
                    "primary_package": artifact_record(Path(values["PRIMARY_PACKAGE"])),
                    "primary_sidecar": artifact_record(compiled_sidecar_path(Path(values["PRIMARY_PACKAGE"]))),
                    "cascade_package": artifact_record(Path(values["CASCADE_PACKAGE"])),
                    "cascade_sidecar": artifact_record(compiled_sidecar_path(Path(values["CASCADE_PACKAGE"]))),
                    "xs_correction": artifact_record(correction) if args.enable_xs_correction else None,
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "campaign": "proton_water_qgsp_bic_hp",
        "material": "G4_WATER",
        "physics_model": "QGSP_BIC_HP/BinaryCascade",
        "scope": {"water_only": True, "neutral_transport": False, "ct": False, "minibeam": False},
        "cases": cases,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
