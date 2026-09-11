"""Lung-case single-shard sensitivity (20022516 shard_01, frozen binary).

Arms (same source/seed, only one switch each):
  S0_base      : exact frozen config (determinism check vs frozen shard)
  S1_nosec     : enable_secondary_transport=false (fragment footprint;
                 energy goes unqueued -> quality recorded, not gated)
  S2_secmcs_off: ct_secondary_mcs_off_diagnostic=true (secondary scattering)
  S3_sefaces   : ct_secondary_exact_faces_diagnostic=true (boundary precision)

Usage:
  python3 benchmark/benchmark20260909/run_lung_sensitivity.py run
  python3 benchmark/benchmark20260909/run_lung_sensitivity.py analyze
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from run_topas10x_gpu_benchmark import sha, config_write  # noqa: E402

BASE_CFG = Path("/mnt/sda/wuwei/ct_previous_full20_20260909/20022516/shard_01/config.yaml")
OUT = Path("/mnt/sda/wuwei/lung_sensitivity_20260910")
FROZEN_BIN = REPO / "build/oneapi-nvidia-water-electron-full/carbon_mc"
FROZEN_PIN = "4df8036cd26b0a8dd7e6f87f0d45820adc697bb0a5c25615f11e564c8abd2e08"
ENV = dict(__import__("os").environ, ONEAPI_DEVICE_SELECTOR="cuda:*",
           LD_LIBRARY_PATH="/home/wuwei/sycl_workspace/llvm/build/install/lib")


def base_config():
    return yaml.safe_load(BASE_CFG.read_text())


def run_case(name, cfg):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    config_write(d / "config.yaml", cfg)
    if not (d / "data").exists():
        (d / "data").symlink_to(REPO / "data", target_is_directory=True)
    qpath = d / "out/config/quality_report.json"
    if not (qpath.exists() and (d / "dose.raw").exists()):
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU occupied, retry later")
        assert sha(FROZEN_BIN) == FROZEN_PIN, "Frozen binary pin mismatch"
        subprocess.run([sys.executable, str(REPO / "tools/verify_schneider_v2_1_data.py")],
                       check=True)
        start = time.monotonic()
        with (d / "gpu.log").open("w") as log:
            p = subprocess.run([str(FROZEN_BIN), "--config", str(d / "config.yaml"),
                                "--device", "cuda", "--voxel-dose-mhd",
                                str(d / "dose.mhd")],
                               cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        (d / "timing.json").write_text(json.dumps(
            {"wall_seconds": time.monotonic() - start,
             "returncode": p.returncode}) + "\n")
    q = json.loads(qpath.read_text())
    if q["queue_overflow_count"] or q.get("queue_overflow_energy_MeV", 0):
        raise RuntimeError(f"Overflow in {name}")
    ledger = json.loads((d / "out/config/energy_ledger.json").read_text())
    if ledger["histories"] != cfg["number_of_histories"]:
        raise RuntimeError(f"History mismatch in {name}")
    return {"directory": str(d), "quality": q,
            "ledger_histories": ledger["histories"],
            "physical_residual": q.get("physical_relative_energy_residual"),
            "dose_sha256": sha(d / "dose.raw"),
            "config_sha256": sha(d / "config.yaml")}


def run_all():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    cases = {"S0_base": {},
             "S1_nosec": {"enable_secondary_transport": False},
             "S0_smoke": {"run_mode": "smoke"},
             "S2_secmcs_off": {"run_mode": "smoke",
                               "ct_secondary_mcs_off_diagnostic": True},
             "S3_sefaces": {"run_mode": "smoke",
                            "ct_secondary_exact_faces_diagnostic": True}}
    for tag, change in cases.items():
        cfg = base_config()
        cfg.update(change)
        manifest[tag] = run_case(tag, cfg)
    (OUT / "manifest.json").write_text(json.dumps(
        {"base": str(BASE_CFG), "base_sha256": sha(BASE_CFG),
         "frozen_binary_sha256": sha(FROZEN_BIN), "cases": manifest},
        indent=2) + "\n")
    print("ALL LUNG SENSITIVITY RUNS COMPLETE", flush=True)


def load_dose(name):
    return np.fromfile(OUT / name / "dose.raw", dtype=np.float32)


def analyze():
    m = json.load(open("/mnt/sda/wuwei/ct_previous_full20_20260909/20022516/manifest.json"))
    shape = tuple(m["gpu_shape_zyx"])
    ref = np.fromfile("/mnt/sda/wuwei/ct_previous_full20_20260909/20022516/topas_sum.raw",
                      dtype="<f4").reshape(m["topas_shape_zyx"]).astype(float)
    peak = float(ref.max())
    mask = ref >= 0.1 * peak
    assert mask.shape == shape, (mask.shape, shape)
    base = load_dose("S0_base").reshape(shape).astype(float)
    frozen = np.fromfile("/mnt/sda/wuwei/ct_previous_full20_20260909/20022516/shard_01/dose.raw",
                         dtype=np.float32).reshape(shape)
    print("S0 vs frozen shard_01 bit-identical:",
          bool((load_dose("S0_base").reshape(shape) == frozen).all()))
    rr = ref[mask]
    result = {}
    pairs = [(("S0_base", "S1_nosec"), "secondary_transport_OFF"),
               (("S0_smoke", "S2_secmcs_off"), "secondary_MCS_OFF"),
               (("S0_smoke", "S3_sefaces"), "secondary_exact_faces_ON"),
               (("S0_base", "S0_smoke"), "smoke_vs_research_baseline")]
    for (a_name, b_name), label in pairs:
        b = load_dose(b_name).reshape(shape).astype(float)[mask]
        a = load_dose(a_name).reshape(shape).astype(float)[mask]
        delta = (b - a) / np.maximum(a, 1e-12) * 100
        result[label] = {
            "mean_pct": float(delta.mean()),
            "rms_pct": float(np.sqrt((delta ** 2).mean())),
            "median_pct": float(np.median(delta)),
            "bands": {},
        }
        for lo, hi, band in ((0.1, 0.2, "10-20"), (0.2, 0.5, "20-50"), (0.5, 1.01, ">=50")):
            s = (rr >= lo * peak) & (rr < hi * peak)
            e = delta[s]
            result[label]["bands"][band] = {
                "mean_pct": float(e.mean()),
                "rms_pct": float(np.sqrt((e ** 2).mean())), "n": int(s.sum())}
        q = json.load(open(OUT / b_name / "out/config/quality_report.json"))
        result[label]["quality_failures"] = [f["code"] for f in q["failures"]]
        result[label]["quality_approximations"] = [(f["code"], f["value"]) for f in q["approximations"]]
        result[label]["physical_residual"] = q.get("physical_relative_energy_residual")
        print(label, json.dumps(result[label], indent=1), flush=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    {"run": run_all, "analyze": analyze}[sys.argv[1]]()
