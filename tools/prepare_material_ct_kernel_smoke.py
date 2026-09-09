"""Generate a tiny synthetic CT run for the real carbon_mc entry point.

No patient data, TOPAS campaign, physics fitting or package modification.
Uses the pinned bank's measured density nodes and current nuclear bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-mode", choices=["device", "host_mapped"], default="device")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    index = root / "benchmark/benchmark20260908/material_response_states_450_candidate/material_response_runtime_index.json"
    pin = "79bb2a62c8080ee9850845dd5b104936cc06db007f9c19c3fb10b20b42fd271e"
    if hashlib.sha256(index.read_bytes()).hexdigest() != pin:
        raise ValueError("Runtime index pin mismatch")
    tables = {t["tag"]: t["material_identity"] for t in json.loads(index.read_text())["tables"]}
    rho = np.empty((80, 16, 16), dtype="<f4")
    section = np.empty(rho.shape, dtype="u1")
    for low, high, tag in [(0, 20, "hu99"), (20, 40, "hu-950"), (40, 80, "hu99")]:
        m = tables[tag]
        # Match CCTG and the runtime loader: float32 of the HU formula,
        # not TOPAS's rounded material-report decimal.
        rho[low:high] = m["schneider_identity"]["density_g_cm3"]
        section[low:high] = m["schneider_identity"]["material_section"]
    # CCTG v2. Legacy mass-SP factors are unused: this run requires SCHNSTOP.
    with (dest / "synthetic_ct.bin").open("wb") as f:
        f.write(struct.pack("<5I6f", 0x47544343, 2, 16, 16, 80, -16, -16, 0, 2, 2, 2))
        f.write(rho.tobytes()); f.write(section.tobytes())
        f.write(struct.pack("<I25f", 25, *([1.] * 25)))
    config = yaml.safe_load((root / "config/unified_water_production.yaml").read_text())
    for key in ("unified_water_material_file", "unified_water_material_sha256"):
        config.pop(key, None)
    for key, value in list(config.items()):
        if isinstance(value, str) and value.startswith("data/"):
            config[key] = str(root / value)
    config.update(run_mode="research", scorer_mode="production", number_of_histories=16,
                  history_chunk_size=8, enable_ct_grid=True, ct_grid_file=str(dest / "synthetic_ct.bin"),
                  ct_schneider_file=str(root / "data/HUtoMaterialSchneider.txt"),
                  initial_energy_MeVu=200, beam_energy_spread=0.01,
                  voxel_bins_x=16, voxel_bins_y=16, voxel_bins_z=80,
                  voxel_size_x_mm=2., voxel_size_y_mm=2., voxel_size_z_mm=2.,
                  depth_bin_width_mm=2., phantom_length_mm=160., maximum_step_mm=0.5,
                  scorer_area_mm2=1024., enable_ct_material_mcs=True,
                  material_electron_response_index_file=str(index),
                  material_electron_response_index_sha256=pin,
                  material_electron_response_device_budget_MiB=8192,
                  material_electron_response_memory_mode=args.memory_mode,
                  material_electron_response_host_budget_MiB=98304,
                  enable_charged_origin_voxel_scoring=True,
                  validation_output_directory=str(dest), dose_to_medium_name="dose")
    # Disable optional sparse/1D exports; only dense 3D dose is requested.
    for key in ["output_file", "voxel_dose_output_file",
                "charged_origin_voxel_output_file", "charged_origin_voxel_dose_Gy_output_file",
                "neutral_origin_voxel_output_file"]:
        config[key] = ""
    # Repository's flat YAML reader uses bare empty values, not quoted ''.
    name="material_ct_kernel_smoke"+("_mapped" if args.memory_mode=="host_mapped" else "")+".yaml"
    (dest / name).write_text(
        yaml.safe_dump(config, sort_keys=False).replace(": ''\n", ":\n"))
    print(dest / name)


if __name__ == "__main__":
    main()
