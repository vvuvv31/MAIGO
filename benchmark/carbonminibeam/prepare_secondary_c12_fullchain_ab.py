#!/usr/bin/env python3
"""Generate staged secondary-C12 full-chain A/B diagnostic configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


BASES = {
    150: Path("config/beam_minibeam_field3cm_copper_e150_256k.yaml"),
    250: Path("config/beam_minibeam_field3cm_copper_e250_256k.yaml"),
    300: Path("config/beam_minibeam_field3cm_copper_e300_256k.yaml"),
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def write(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "# Diagnostic only: formal minibeam configurations are unchanged.\n"
        )
        yaml.safe_dump(config, stream, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", type=int, choices=BASES, default=300)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=256_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--component-scoring", action="store_true")
    args = parser.parse_args()

    base = load(BASES[args.energy])
    base_histories = int(base["number_of_histories"])
    if "tps_histories_scale" in base:
        base["tps_histories_scale"] = (
            float(base["tps_histories_scale"]) * args.histories /
            base_histories
        )
    base["number_of_histories"] = args.histories
    if args.seed is not None:
        base["random_seed"] = args.seed
    # This experiment must never route every supported secondary through
    # Unified EM. The new C12-only selector is the sole EM-path difference.
    base["enable_secondary_unified_em"] = False
    if args.component_scoring:
        base["enable_charged_origin_voxel_scoring"] = True

    cases = {
        "A_formal_em_scale1_highland": {
            "minibeam_water_secondary_c12_enable_unified_em": False,
            "enable_secondary_energy_straggling": False,
            "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
            "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
        },
        "B_unified_scale1_highland": {
            "minibeam_water_secondary_c12_enable_unified_em": True,
            "enable_secondary_energy_straggling": True,
            "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
            "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
        },
        "C_unified_scale09958_highland": {
            "minibeam_water_secondary_c12_enable_unified_em": True,
            "enable_secondary_energy_straggling": True,
            "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
            "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
        },
        "D_unified_scale09958_fe": {
            "minibeam_water_secondary_c12_enable_unified_em": True,
            "enable_secondary_energy_straggling": True,
            "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
            "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
            "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.10,
        },
    }
    for name, overrides in cases.items():
        config = dict(base)
        config.update(overrides)
        if args.component_scoring:
            config["charged_origin_voxel_mhd_output_prefix"] = str(
                args.root / name / "component"
            )
        write(args.root / name / "config.yaml", config)


if __name__ == "__main__":
    main()
