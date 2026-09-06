"""Read-only, absolute-normalization analysis of the frozen 175 MeV/u pilot.
Reads 3-D scorers only. Prints JSON; does not fit parameters or overwrite evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compile_schneider_delta_longitudinal import lateral_integral

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def gpu_curve(case):
    path = case / "dose.mhd"
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    if fields["DimSize"] != "100 100 440":
        raise ValueError("Unexpected GPU dimensions")
    if list(map(float, fields["ElementSpacing"].split())) != [2, 2, 0.5]:
        raise ValueError("Unexpected GPU spacing")
    dtype = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8"}[fields["ElementType"]]
    raw = path.parent / fields["ElementDataFile"]
    data = np.fromfile(raw, dtype=dtype)
    if data.size != 4400000 or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Invalid GPU payload")
    return data.reshape(440, 100, 100).sum(axis=(1, 2), dtype=np.float64)

def analyze(root):
    contract = json.loads((root/"campaign.json").read_text())
    if contract["energy_MeV_u"] != 175 or contract["scale"] != 1:
        raise ValueError("Not the frozen holdout")
    n = contract["histories_per_run"]
    paths = [root/name/"dose.csv" for name in ("topas_s1", "topas_s2")]
    curves = [lateral_integral(p)/n for p in paths]
    reference = (curves[0]+curves[1])/2
    gpu = {name: gpu_curve(root/name)/n for name in ("gpu_base", "gpu_long")}
    z = (np.arange(440)+0.5)*0.5
    windows = []
    for lo, hi in contract["windows_mm"]:
        take = (z >= lo) & (z < hi)
        ref = reference[take].sum()
        if ref <= 0:
            raise ValueError("Empty reference window")
        windows.append(dict(depth_mm=[lo, hi],
            topas_replica_relative_difference=float(abs(curves[0][take].sum()-curves[1][take].sum())/ref),
            **{k+"_relative_error": float(v[take].sum()/ref-1) for k,v in gpu.items()}))
    rebinned = []
    for first in range(4, 240, 10):
        end = min(first+10, 240)
        ref = reference[first:end].sum()
        rebinned.append(dict(depth_mm=[first*.5, end*.5],
            topas_replica_relative_difference=float(abs(curves[0][first:end].sum()-curves[1][first:end].sum())/ref),
            **{k+"_relative_error": float(v[first:end].sum()/ref-1) for k,v in gpu.items()}))
    gates = contract["diagnostic_gates"]
    noise_ok = all(w["topas_replica_relative_difference"] <= gates["window_replica_relative_difference_max"] for w in windows)
    noise_ok &= all(w["topas_replica_relative_difference"] <= gates["rebin_abs_relative_error_max"] for w in rebinned)
    errors_ok = all(abs(w["gpu_long_relative_error"]) <= gates["window_abs_relative_error_max"] for w in windows)
    errors_ok &= all(abs(w["gpu_long_relative_error"]) <= gates["rebin_abs_relative_error_max"] for w in rebinned)
    improvement = sum(abs(w["gpu_long_relative_error"]) for w in windows) < sum(abs(w["gpu_base_relative_error"]) for w in windows)
    qualities = {}
    quality_ok = True
    for name in gpu:
        q = json.loads((root/name/"out/run/quality_report.json").read_text())
        codes = [f["code"] for f in q["failures"]]
        expected = ["unvalidated_longitudinal_candidate"] if name == "gpu_long" else []
        quality_ok &= codes == expected
        qualities[name] = dict(accepted=q["accepted"], failures=codes)
    # Gy sum × shared voxel mass / (J/MeV) yields MeV per primary.
    mev_factor = contract["density_g_cm3"] * 2e-6 / 1.602176634e-13
    integrals = {"topas": float(reference.sum()*mev_factor)}
    integrals.update({k: float(v.sum()*mev_factor) for k,v in gpu.items()})
    status = ("INCONCLUSIVE" if not noise_ok or not quality_ok else
              "PASS_PILOT_ONLY" if errors_ok and improvement else "FAIL_PILOT")
    used = paths + [root/"campaign.json"]
    for name in gpu:
        used += [root/name/p for p in ("run.yaml", "dose.mhd", "dose.raw", "out/run/quality_report.json", "out/run/energy_ledger.json")]
    for name in ("topas_s1", "topas_s2"):
        used += [root/name/"run.txt"]
    return dict(status=status, normalization="Gy per primary; no fitted scale or alignment",
                windows=windows, rebinned_5mm=rebinned, integrals_MeV_per_primary=integrals,
                noise_pilot_gate=noise_ok, quality_gate=quality_ok, candidate_error_gate=errors_ok,
                candidate_improves=improvement, quality=qualities,
                artifacts={str(p): sha(p) for p in used},
                limitations=contract["limitations"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.root), indent=2))
