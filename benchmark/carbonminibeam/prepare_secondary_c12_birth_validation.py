#!/usr/bin/env python3
"""Prepare diagnostic export/replay configs for true water-born C12 states."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Diagnostic only; formal configurations are unchanged.\n")
        yaml.safe_dump(config, stream, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    export = load(Path("config/beam_minibeam_field3cm_copper_e250_256k.yaml"))
    export["fragment_birth_spectrum_output_file"] = str(
        args.root / "export" / "birth"
    )
    # Keep Copper-born C12 at its transported water-entrance state separate
    # from true water-born C12, which must replay from its actual birth point.
    export["minibeam_fragment_phase_space_output_file"] = str(
        args.root / "export" / "copper_exit_charged.csv"
    )
    write(args.root / "export" / "config.yaml", export)

    replay = load(Path(
        "config/beam_minibeam_water_secondary_c12_replay_e250_1622795_legacy.yaml"
    ))
    replay.update({
        "number_of_histories": int(export["number_of_histories"]),
        "minibeam_water_entry_secondary_replay_file": str(
            args.root / "replay" / "water_c12_identity.csv"
        ),
        "minibeam_water_entry_secondary_replay_allow_primary_c12": False,
        "minibeam_water_entry_secondary_replay_allow_internal_births": True,
        "minibeam_water_secondary_c12_mcs_model": "fermi_eyges_tail",
        "minibeam_water_secondary_c12_mcs_max_segment_mm": 0.10,
        "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
        "enable_secondary_unified_em": False,
        "minibeam_water_secondary_c12_enable_unified_em": True,
        "enable_inelastic": False,
        "enable_multiple_scattering": True,
        "enable_energy_straggling": True,
        "enable_secondary_energy_straggling": True,
    })
    replay.pop("minibeam_water_entry_secondary_replay_allow_primary_c12", None)
    replay.pop("minibeam_water_primary_plane_output_file", None)
    replay.pop("minibeam_water_primary_plane_depths_mm", None)
    replay["secondary_queue_capacity"] = max(
        int(replay.get("secondary_queue_capacity", 0)),
        int(export["number_of_histories"]),
    )
    write(args.root / "replay" / "config.yaml", replay)


if __name__ == "__main__":
    main()
