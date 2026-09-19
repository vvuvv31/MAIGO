#!/usr/bin/env python3
"""Prepare controlled primary/secondary C12 energy-loss diagnostic configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PRIMARY_BASE = Path(
    "/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/"
    "gpu_fe_candidate/config.yaml"
)
SECONDARY_BASE = Path(
    "config/beam_minibeam_water_secondary_c12_replay_e250_1622795_legacy.yaml"
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def write_case(root: Path, name: str, base: dict, overrides: dict) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    config = dict(base)
    config.update(overrides)
    disable_planes = bool(config.pop("_disable_plane_scoring", False))
    if disable_planes:
        config.pop("minibeam_water_primary_plane_output_file", None)
        config.pop("minibeam_water_primary_plane_depths_mm", None)
    else:
        config["minibeam_water_primary_plane_output_file"] = str(
            case_dir / "primary_planes.csv"
        )
    with (case_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        stream.write(
            "# Generated diagnostic only; formal minibeam configs are unchanged.\n"
        )
        yaml.safe_dump(config, stream, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/sda/wuwei/minibeam_secondary_c12_energy_path_e250_20260919"
        ),
    )
    args = parser.parse_args()

    primary = load_config(PRIMARY_BASE)
    secondary = load_config(SECONDARY_BASE)
    # The legacy replay config deliberately exercised the historical secondary
    # stopping path.  This closure experiment instead routes secondary C12
    # through the same unified-EM package as the primary; otherwise the
    # straggling switch is a no-op and the comparison mixes two EM models.
    secondary["enable_secondary_unified_em"] = False
    secondary["minibeam_water_secondary_c12_enable_unified_em"] = True
    secondary["minibeam_water_entry_secondary_replay_file"] = str(
        args.output_root / "input_aligned" /
        "water_entrance_primary-c12_identity.csv"
    )
    common = {
        "enable_inelastic": False,
        "enable_multiple_scattering": False,
    }

    cases = (
        (
            "primary_nostrag_scale1",
            primary,
            {
                **common,
                "enable_energy_straggling": False,
                "minibeam_water_primary_stopping_power_scale": 1.0,
            },
        ),
        (
            "secondary_nostrag_scale1",
            secondary,
            {
                **common,
                "enable_energy_straggling": False,
                "enable_secondary_energy_straggling": False,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
            },
        ),
        (
            "primary_strag_scale1",
            primary,
            {
                **common,
                "enable_energy_straggling": True,
                "minibeam_water_primary_stopping_power_scale": 1.0,
            },
        ),
        (
            "primary_strag_scale09958",
            primary,
            {
                **common,
                "enable_energy_straggling": True,
                "minibeam_water_primary_stopping_power_scale": 0.9958,
            },
        ),
        (
            "secondary_strag_scale1",
            secondary,
            {
                **common,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
            },
        ),
        (
            "secondary_strag_scale09958",
            secondary,
            {
                **common,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
            },
        ),
        (
            "secondary_legacy_scale09958",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
            },
        ),
        (
            "secondary_fe010_scale09958",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
                "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.10,
            },
        ),
        (
            "secondary_fe005_scale09958",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
                "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.05,
            },
        ),
        (
            "secondary_fe010_scale09958_noplanes",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
                "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.10,
                "_disable_plane_scoring": True,
            },
        ),
        (
            "secondary_legacy_scale09958_noplanes",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
                "_disable_plane_scoring": True,
            },
        ),
        (
            "secondary_fe005_scale09958_noplanes",
            secondary,
            {
                "enable_inelastic": False,
                "enable_multiple_scattering": True,
                "enable_energy_straggling": True,
                "enable_secondary_energy_straggling": True,
                "minibeam_water_secondary_c12_post_sample_loss_scale": 0.9958,
                "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
                "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.05,
                "_disable_plane_scoring": True,
            },
        ),
    )
    for name, base, overrides in cases:
        write_case(args.output_root, name, base, overrides)


if __name__ == "__main__":
    main()
