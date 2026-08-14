#!/usr/bin/env python3
"""Prepare fixed-seed CT material-model A/B configs without touching results."""

import argparse
from pathlib import Path

import yaml


CASES = ("20022516", "RT06423", "RT07575")
SECTION07_PRIMARY_PACKAGE = (
    "data/packages/"
    "topas_400MeVu_schneider_section07_inclxx_100k_primary_3d.bin"
)
SECTION07_CASCADE_PACKAGE = (
    "data/packages/"
    "topas_400MeVu_schneider_section07_inclxx_100k_cascade_3d.bin"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--profile", default="balanced")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for package in (SECTION07_PRIMARY_PACKAGE, SECTION07_CASCADE_PACKAGE):
        if not (repo / package).is_file():
            raise SystemExit(f"missing section-7 runtime package: {package}")
    root = repo / "benchmark/ct/result/diagnostics/hu_lut_ab"
    for case in CASES:
        source = repo / "benchmark/ct/result" / case / f"{args.profile}.yaml"
        config = yaml.safe_load(source.read_text(encoding="ascii"))
        config["random_seed"] = args.seed
        # Origin maps make primary/secondary attribution possible. They are
        # intentionally local diagnostic outputs, not production scorers.
        config["enable_charged_origin_voxel_scoring"] = True
        config["charged_origin_voxel_output_file"] = ""
        config["charged_origin_voxel_dose_Gy_output_file"] = ""
        for model in ("density_spr", "topas_hu_lut",
                      "topas_hu_lut_section07_primary_100k",
                      "topas_hu_lut_section07_100k"):
            candidate = dict(config)
            if model == "density_spr":
                candidate.pop("ct_hu_stopping_power_lut_file", None)
                candidate["ct_use_density_mass_spr"] = True
            else:
                candidate["ct_hu_stopping_power_lut_file"] = (
                    "data/hu_stopping_power_lut_geant4_11_3_2.csv")
                # This is redundant under explicit-LUT precedence but records
                # the intended A/B physics unambiguously in the artifact.
                candidate["ct_use_density_mass_spr"] = False
            if model in ("topas_hu_lut_section07_primary_100k",
                         "topas_hu_lut_section07_100k"):
                candidate["reaction_package_file"] = SECTION07_PRIMARY_PACKAGE
            if model == "topas_hu_lut_section07_100k":
                candidate["cascade_package_file"] = SECTION07_CASCADE_PACKAGE
            destination = root / case / model / f"{args.profile}.yaml"
            output_dir = destination.parent / args.profile
            candidate["charged_origin_voxel_mhd_output_prefix"] = str(
                output_dir.relative_to(repo) / "dose_origin")
            destination.parent.mkdir(parents=True, exist_ok=True)
            serialized = yaml.safe_dump(candidate, sort_keys=False)
            # carbon_mc intentionally supports a constrained YAML subset and
            # treats an empty value as disabled; it does not unquote "''".
            serialized = serialized.replace(": ''\n", ":\n")
            destination.write_text(serialized, encoding="ascii")
            print(destination.relative_to(repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
