#!/usr/bin/env python3
"""Document TOPAS/GPU C12 water-replay provenance and comparable dose files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def phsp_stats(path: Path) -> dict:
    data = np.loadtxt(path)
    return {
        "path": str(path),
        "n_particles": int(data.shape[0]),
        "weight_unique": [float(x) for x in np.unique(data[:, 6])],
        "weight_sum": float(data[:, 6].sum()),
        "first_scored_sum": float(data[:, 9].sum()),
        "run_id_unique": [int(x) for x in np.unique(data[:, 10])],
        "event_id_min": int(data[:, 11].min()),
        "event_id_max": int(data[:, 11].max()),
        "pdg_unique": [int(x) for x in np.unique(data[:, 7])],
        "parent_id_unique": [int(x) for x in np.unique(data[:, 13])],
    }


def main() -> None:
    out = Path("/mnt/sda/wuwei/minibeam_homologous_c12_20260919")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Homologous C12 water-transport normalization",
        "entrance_replacement_sensitivity": {
            "note": (
                "The GPU-entry vs TOPAS-entry pair that drives the same GPU "
                "water kernel is an entrance-replacement sensitivity, not a "
                "cross-engine homologous comparison."
            ),
            "directory": "/mnt/sda/wuwei/minibeam_homologous_entry_20260919",
        },
        "do_not": [
            "Treat TOPAS MT EventID as a source-file row number",
            "Mechanically scale dose by survivor/incident unless that is the "
            "documented original-history convention of that exact run",
            "Merge empty-history and no-empty TOPAS dose files",
        ],
        "e250": {
            "source_phsp": phsp_stats(Path(
                "/mnt/sda/wuwei/minibeam_water_replay_e250_256k/input/"
                "water_entrance_primary_c12.phsp")),
            "header_original_histories": 256000,
            "header_reached_phase_space": 30878,
            "phase_space_multiple_use": 1,
            "topas_6363": {
                "include_empty_histories": True,
                "log": "Will append 225122 empty histories",
                "reported_histories": 256000,
                "dose_file": "/mnt/sda/wuwei/minibeam_water_replay_e250_256k/topas_6363/dose.bin",
                "comparable_to_30878_gpu": False,
                "note": (
                    "DoseToMedium Sum is 8.28x the 30878-history run, equal "
                    "to original/survivor. Do not use for homologous dose."
                ),
            },
            "topas_6364": {
                "include_empty_histories": False,
                "reported_histories": 30878,
                "dose_file": "/mnt/sda/wuwei/minibeam_water_replay_e250_256k/topas_6364/dose.bin",
                "physics": "full list including INCL++",
                "comparable_to_30878_gpu": True,
                "note": "One emission per PHSP particle, weight=1, MultipleUse=1.",
            },
            "gpu_same_topas_entrance": {
                "directory": "/mnt/sda/wuwei/minibeam_homologous_entry_20260919/e250/topas_entry",
                "histories_emitted": 30878,
                "nuclear_interactions": 12891,
                "dose_sum_over_topas_6364": 0.9962748336211018,
            },
        },
        "e300": {
            "source_phsp": phsp_stats(Path(
                "/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/input/"
                "water_entrance_primary_c12.phsp")),
            "header_original_histories": 256000,
            "header_reached_phase_space": 34557,
            "topas": {
                "include_empty_histories": False,
                "physics": "EM-only g4em-standard_opt4 + g4decay",
                "dose_file": "/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/topas/dose.bin",
                "planes": "/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/topas/output",
            },
            "gpu_em_only": {
                "directory": "/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/gpu",
                "histories_emitted": 34557,
                "nuclear_interactions": 0,
                "dose_sum_over_topas": 1.0000241647398513,
                "phase_comparison": "/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/phase_comparison/water_primary_phase_space_metrics.json",
            },
            "gpu_full_physics_same_entrance": {
                "directory": "/mnt/sda/wuwei/minibeam_homologous_entry_20260919/e300/topas_entry",
                "nuclear_interactions": 16718,
                "comparable_to_em_only_topas": False,
            },
        },
        "identity": (
            "Keep (RunID,EventID,TrackID) from the original source run. "
            "Do not join TOPAS MT replay EventID to entrance row numbers."
        ),
    }
    path = out / "normalization_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
