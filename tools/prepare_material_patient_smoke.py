"""Bounded real-patient candidate run; no CT/spot edits, no acceptance claim."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((root / "config/20022516_electron_segment_transport.yaml").read_text())
    index = root / "benchmark/benchmark20260908/material_response_states_450_candidate/material_response_runtime_index.json"
    pin = "79bb2a62c8080ee9850845dd5b104936cc06db007f9c19c3fb10b20b42fd271e"
    if hashlib.sha256(index.read_bytes()).hexdigest() != pin:
        raise ValueError("Index SHA mismatch")
    with Path(cfg["ct_grid_file"]).open("rb") as f:
        magic, version, nx, ny, nz, ox, oy, oz, sx, sy, sz = struct.unpack("<5I6f", f.read(44))
        if magic != 0x47544343 or version not in (2, 3) or oz != 0:
            raise ValueError("Unsupported CT layout")
        rho = np.fromfile(f, dtype="<f4", count=nx*ny*nz)
        section = np.fromfile(f, dtype="u1", count=rho.size)
    tables = json.loads(index.read_text())["tables"]
    selected = set()
    demand = []
    for sec in np.unique(section):
        densities = np.unique(rho[section == sec])
        nodes = sorted((float(np.float32(t["material_identity"]["schneider_identity"]["density_g_cm3"])), i)
                       for i, t in enumerate(tables)
                       if t["material_identity"].get("schneider_identity", {}).get("material_section") == int(sec))
        if not nodes or not np.isfinite(densities).all() or densities[0] < nodes[0][0] or densities[-1] > nodes[-1][0]:
            raise ValueError(f"Uncovered section/density: {sec}, {densities[0]}..{densities[-1]}")
        values = np.array([r for r, _ in nodes])
        for density in densities:
            hi = int(np.searchsorted(values, density))
            selected.add(nodes[hi][1])
            if values[hi] != density:
                selected.add(nodes[hi-1][1])
        demand.append(dict(section=int(sec), min=float(densities[0]), max=float(densities[-1]),
                           unique=int(len(densities))))
    raw_bytes = sum(s["rows"]*312 for i in selected for s in tables[i]["transport_states"]["sources"])
    # Raw lower bound only; C++ enforces the actual resident allocation budget.
    if raw_bytes > 64*1024**3:
        raise ValueError("Raw bank exceeds preflight allowance")
    for key in list(cfg):
        if key.startswith("ct_electron_") or key.startswith("ct_schneider_delta_"):
            del cfg[key]
    cfg.update(run_mode="research", number_of_histories=64, history_chunk_size=16,
               tps_spots_file=str(root / "benchmark/topas10x/20022516/spots.csv"),
               # Same weights normalized to a fixed 64-history budget. Avoid
               # histories mode's minimum one history for every positive spot.
               tps_spot_weight_mode="mu", tps_histories_scale=1.0,
               material_electron_response_index_file=str(index), material_electron_response_index_sha256=pin,
               material_electron_response_memory_mode="host_mapped",
               material_electron_response_host_budget_MiB=98304,
               material_electron_response_device_budget_MiB=8192,
               enable_charged_origin_voxel_scoring=False,
               dose_to_medium_name="dose", validation_output_directory=str(out),
               voxel_bins_x=nx, voxel_bins_y=ny, voxel_bins_z=nz,
               voxel_size_x_mm=sx, voxel_size_y_mm=sy, voxel_size_z_mm=sz,
               phantom_length_mm=nz*sz, depth_bin_width_mm=sz, scorer_area_mm2=nx*sx*ny*sy)
    for key in ("output_file", "voxel_dose_output_file", "charged_origin_voxel_output_file",
                "charged_origin_voxel_dose_Gy_output_file", "neutral_origin_voxel_output_file"):
        cfg[key] = ""
    report = dict(histories=64, original_spots_unchanged=True, allocation="normalized weights, fixed total",
                  ct=cfg["ct_grid_file"], grid=[nx, ny, nz], demand=demand,
                  selected_tables=[tables[i]["tag"] for i in sorted(selected)], raw_GiB=raw_bytes/1024**3,
                  candidate=True, clinical_acceptance=False)
    (out / "preflight.json").write_text(json.dumps(report, indent=2)+"\n")
    config_path = out / "material_patient_20022516_64.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None).replace(": ''\n", ":\n"))
    print(json.dumps(report, indent=2))
    print(config_path)


if __name__ == "__main__":
    main()
