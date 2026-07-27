#!/usr/bin/env python3
"""Run multi-energy GPU birth-spectrum suite and compare to TOPAS packages.

Energies: 100, 200, 300, 400 MeV/u (override with --energies).
Uses existing reaction/cascade packages (G4 11.3.2 400 MeV cascade where available).

Example:
  ONEAPI_DEVICE_SELECTOR=cuda:gpu \\
  python3 validation/scripts/run_birth_spectrum_energy_suite.py \\
    --carbon-mc build/oneapi-release/carbon_mc \\
    --device cuda --histories 10000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PACKAGE_BY_ENERGY = {
    # Prefer G4 11.3.2 400-MeV cascade/primary packages for high-E LET work.
    100: {
        "reaction": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d.bin",
        "cascade": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_3d.bin",
        "primary_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_secondaries.csv.gz",
        "primary_reactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_reactions.csv.gz",
        "cascade_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz",
        "cascade_interactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz",
        "topas_histories": 100000,
    },
    200: {
        "reaction": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d.bin",
        "cascade": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_3d.bin",
        "primary_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_secondaries.csv.gz",
        "primary_reactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_reactions.csv.gz",
        "cascade_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz",
        "cascade_interactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz",
        "topas_histories": 100000,
    },
    300: {
        "reaction": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d.bin",
        "cascade": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_3d.bin",
        "primary_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_secondaries.csv.gz",
        "primary_reactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_reactions.csv.gz",
        "cascade_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz",
        "cascade_interactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz",
        "topas_histories": 100000,
    },
    400: {
        "reaction": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d.bin",
        "cascade": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_3d.bin",
        "primary_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_secondaries.csv.gz",
        "primary_reactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_aligned_primary_3d_reactions.csv.gz",
        "cascade_products": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_products.csv.gz",
        "cascade_interactions": "validation/results/topas_400MeVu_cascade_g4_11_3_2_100k_interactions.csv.gz",
        "topas_histories": 100000,
    },
}


def write_config(path: Path, energy: int, histories: int, out_prefix: Path, pkgs: dict) -> None:
    phantom = 120.0 if energy <= 150 else 400.0
    text = f"""number_of_histories: {histories}
initial_energy_MeVu: {float(energy)}
mass_number: 12
phantom_length_mm: {phantom}
depth_bin_width_mm: 1.0
maximum_step_mm: 0.5
maximum_relative_energy_loss: 0.005
energy_cutoff_MeV: 0.1
water_density_g_per_cm3: 1.0
scorer_area_mm2: 90000.0
enable_voxel_scoring: false
enable_energy_straggling: true
straggling_scale: 1.2
enable_multiple_scattering: true
enable_primary_attenuation: true
enable_secondary_generation: true
enable_secondary_transport: true
enable_fragment_cascade: true
enable_neutral_transport: false
maximum_cascade_generations: 2
secondary_queue_capacity: 500000
max_device_memory_fraction: 0.50
random_seed: 20260727
stopping_power_file: data/stopping_power_water_geant4_11_3_2.csv
use_particle_specific_stopping_power: true
particle_stopping_power_file: data/ion_stopping_power_water_geant4_11_3_2.csv
nuclear_cross_section_file: data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv
reaction_package_file: {pkgs['reaction']}
cascade_package_file: {pkgs['cascade']}
output_file: {out_prefix}_idd.csv
fragment_species_output_file:
dose_output_file:
fragment_species_dose_output_file:
scorerLET: false
fragment_birth_spectrum_output_file: {out_prefix}
device: cuda
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon-mc", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--histories", type=int, default=10000)
    parser.add_argument("--energies", type=int, nargs="+", default=[100, 200, 300, 400])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "validation/results/birth_spectrum_energy_suite",
    )
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-topas-prep", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare = ROOT / "validation/scripts/prepare_topas_birth_spectrum.py"
    compare = ROOT / "validation/scripts/compare_fragment_birth_spectra.py"

    suite = {"histories": args.histories, "energies": {}, "device": args.device}

    # TOPAS package histograms (same packages for all energies here).
    if not args.skip_topas_prep:
        pkgs0 = PACKAGE_BY_ENERGY[400]
        topas_primary = out_dir / "topas_primary_package_g4_11_3_2"
        topas_cascade = out_dir / "topas_cascade_package_g4_11_3_2"
        subprocess.check_call(
            [
                sys.executable,
                str(prepare),
                "--products",
                str(ROOT / pkgs0["primary_products"]),
                "--reactions",
                str(ROOT / pkgs0["primary_reactions"]),
                "--source",
                "primary",
                "--histories",
                str(pkgs0["topas_histories"]),
                "--output-prefix",
                str(topas_primary),
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                str(prepare),
                "--products",
                str(ROOT / pkgs0["cascade_products"]),
                "--interactions",
                str(ROOT / pkgs0["cascade_interactions"]),
                "--source",
                "cascade",
                "--histories",
                str(pkgs0["topas_histories"]),
                "--output-prefix",
                str(topas_cascade),
            ]
        )
    else:
        topas_primary = out_dir / "topas_primary_package_g4_11_3_2"
        topas_cascade = out_dir / "topas_cascade_package_g4_11_3_2"

    for energy in args.energies:
        pkgs = PACKAGE_BY_ENERGY[energy]
        gpu_prefix = out_dir / f"gpu_e{energy}"
        if not args.skip_gpu:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tf:
                cfg_path = Path(tf.name)
                write_config(cfg_path, energy, args.histories, gpu_prefix, pkgs)
            try:
                env = os.environ.copy()
                if args.device in ("cuda", "nvidia"):
                    env.setdefault("ONEAPI_DEVICE_SELECTOR", "cuda:gpu")
                elif args.device in ("level_zero", "intel", "arc"):
                    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
                    env.setdefault(
                        "UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS", "1"
                    )
                cmd = [
                    str(args.carbon_mc),
                    "--config",
                    str(cfg_path),
                    "--device",
                    args.device,
                ]
                print("Running:", " ".join(cmd), flush=True)
                subprocess.check_call(cmd, cwd=str(ROOT), env=env)
            finally:
                cfg_path.unlink(missing_ok=True)

        gen0_json = out_dir / f"compare_e{energy}_gen0_vs_primary_package.json"
        gen1_json = out_dir / f"compare_e{energy}_gen1_vs_cascade_package.json"
        subprocess.check_call(
            [
                sys.executable,
                str(compare),
                "--gpu-prefix",
                str(gpu_prefix),
                "--topas-prefix",
                str(topas_primary),
                "--generation",
                "0",
                "--output-json",
                str(gen0_json),
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                str(compare),
                "--gpu-prefix",
                str(gpu_prefix),
                "--topas-prefix",
                str(topas_cascade),
                "--generation",
                "1",
                "--output-json",
                str(gen1_json),
            ]
        )
        with open(gen0_json, encoding="utf-8") as f:
            gen0 = json.load(f)
        with open(gen1_json, encoding="utf-8") as f:
            gen1 = json.load(f)
        suite["energies"][str(energy)] = {
            "gpu_prefix": str(gpu_prefix),
            "gen0": gen0["by_species"],
            "gen1": gen1["by_species"],
        }

    summary_path = out_dir / "suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    print(f"Wrote {summary_path}")

    # Compact table
    print("\nenergy  species  gen0_yield_ratio  gen0_meanKE_ratio  gen1_yield_ratio  gen1_meanKE_ratio")
    for energy in args.energies:
        e = suite["energies"][str(energy)]
        for sp in ("proton", "deuteron", "triton", "he3", "he4"):
            g0 = e["gen0"][sp]
            g1 = e["gen1"][sp]
            print(
                f"{energy:3d}  {sp:8s}  "
                f"{g0.get('yield_ratio_gpu_over_topas'):8.3f}  "
                f"{g0.get('mean_ke_ratio_gpu_over_topas'):8.3f}  "
                f"{g1.get('yield_ratio_gpu_over_topas'):8.3f}  "
                f"{g1.get('mean_ke_ratio_gpu_over_topas'):8.3f}"
            )


if __name__ == "__main__":
    main()
