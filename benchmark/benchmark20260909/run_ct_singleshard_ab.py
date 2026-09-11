"""Isolated single-shard A/B experiments on frozen RT06423 shard_02 source.

Pairs (same source spots, same seed, same binary within pair):
  E1 straggling : frozen binary, straggling_scale 1.2 (A) vs 1.0 (B)
  E3 stepsize   : frozen binary, maximum_step_mm 0.5 (A) vs 0.25 (B)
  E2 electron   : electron-bounds binary, joint OFF (A) vs ON (B, r3 joint)

Each pair differs in exactly one transport switch. A/B dose differences use
identical carbon histories allocation (same seed => correlated RNG), so the
difference map isolates the switch effect. No TOPAS rerun, no frozen record
is touched; outputs go to /mnt/sda/wuwei/ct_singleshard_ab_20260909/.

Usage:
  python3 benchmark/benchmark20260909/run_ct_singleshard_ab.py run   # needs free GPU
  python3 benchmark/benchmark20260909/run_ct_singleshard_ab.py analyze
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from run_topas10x_gpu_benchmark import sha, config_write  # noqa: E402

BASE_CFG = Path("/mnt/sda/wuwei/ct_previous_full20_20260909/RT06423/shard_02/config.yaml")
OUT = Path("/mnt/sda/wuwei/ct_singleshard_ab_20260909")
FROZEN_BIN = REPO / "build/oneapi-nvidia-water-electron-full/carbon_mc"
FROZEN_PIN = "4df8036cd26b0a8dd7e6f87f0d45820adc697bb0a5c25615f11e564c8abd2e08"
ELEC_BIN = REPO / "build/oneapi-nvidia-electron-bounds/carbon_mc"
ELEC_CURRENT = REPO / "build/oneapi-nvidia-electron-current/carbon_mc"
JOINT = Path("/mnt/sda/wuwei/schneider_electron_ct_runtime_r3_20260906/joint_response.csv")
ENV = dict(os.environ, ONEAPI_DEVICE_SELECTOR="cuda:*",
           LD_LIBRARY_PATH="/home/wuwei/sycl_workspace/llvm/build/install/lib")


def base_config():
    cfg = yaml.safe_load(BASE_CFG.read_text())
    return cfg


def run_case(name, cfg, binary):
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
        if binary == FROZEN_BIN and sha(binary) != FROZEN_PIN:
            raise RuntimeError("Frozen binary changed")
        subprocess.run([sys.executable, str(REPO / "tools/verify_schneider_v2_1_data.py")],
                       check=True)
        start = time.monotonic()
        with (d / "gpu.log").open("w") as log:
            p = subprocess.run([str(binary), "--config", str(d / "config.yaml"),
                                "--device", "cuda", "--voxel-dose-mhd",
                                str(d / "dose.mhd")],
                               cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        (d / "timing.json").write_text(json.dumps(
            {"wall_seconds": time.monotonic() - start,
             "returncode": p.returncode}) + "\n")
    q = json.loads(qpath.read_text())
    if q["queue_overflow_count"] or q.get("queue_overflow_energy_MeV", 0):
        raise RuntimeError(f"Overflow in {name}, needs split/retry")
    ledger = json.loads((d / "out/config/energy_ledger.json").read_text())
    if ledger["histories"] != cfg["number_of_histories"]:
        raise RuntimeError(f"History mismatch in {name}")
    return {"directory": str(d), "quality": q, "ledger_histories": ledger["histories"],
            "dose_sha256": sha(d / "dose.raw"), "config_sha256": sha(d / "config.yaml")}


def run_all():
    assert sha(FROZEN_BIN) == FROZEN_PIN, "Frozen binary pin mismatch"
    subprocess.run([sys.executable, str(REPO / "tools/verify_schneider_v2_1_data.py")],
                   check=True)
    OUT.mkdir(exist_ok=True)
    manifest = {}
    # E1: straggling scale
    for tag, scale in (("E1_strag12", 1.2), ("E1_strag10", 1.0)):
        cfg = base_config()
        cfg["straggling_scale"] = scale
        manifest[tag] = run_case(tag, cfg, FROZEN_BIN)
    # E3: step size
    for tag, step in (("E3_step05", 0.5), ("E3_step025", 0.25)):
        cfg = base_config()
        cfg["maximum_step_mm"] = step
        manifest[tag] = run_case(tag, cfg, FROZEN_BIN)
    # E4: pure noise floor (same physics as E3_step05, seed+1 => decorrelated)
    cfg = base_config()
    cfg["random_seed"] = int(cfg["random_seed"]) + 1
    manifest["E4_seedshift"] = run_case("E4_seedshift", cfg, FROZEN_BIN)
    # E2: electron joint OFF/ON with current-source electron binary
    # (same binary both arms; Sep-06 electron-bounds predates current keys).
    if not ELEC_CURRENT.exists():
        raise RuntimeError("Current electron binary not built yet")
    from run_ct_electron_gamma_probe import run as probe_run
    cfg = base_config()
    for tag, joint in (("E2_off", None), ("E2_on", JOINT)):
        if joint is None:
            cfg["ct_electron_segment_transport"] = False
        else:
            # Absent key preserves legacy file-driven activation; explicit
            # false would clear the joint files (see config.cpp).
            cfg.pop("ct_electron_segment_transport", None)
        dest = OUT / tag
        if not (dest / "run_report.json").exists():
            if dest.exists():
                import shutil as _sh
                _sh.rmtree(dest)
            if subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid",
                     "--format=csv,noheader"], text=True).strip():
                raise RuntimeError("GPU occupied, retry later")
            src = OUT / f"{tag}.source.yaml"
            c = dict(cfg)
            config_write(src, c)
            probe_run(src, dest, ELEC_CURRENT, joint)
        manifest[tag] = {"report": json.loads((dest / "run_report.json").read_text())}
    (OUT / "manifest.json").write_text(json.dumps(
        {"base": str(BASE_CFG), "base_sha256": sha(BASE_CFG),
         "frozen_binary_sha256": sha(FROZEN_BIN), "electron_binary_sha256": sha(ELEC_BIN),
         "joint_sha256": sha(JOINT), "cases": manifest}, indent=2) + "\n")
    print("ALL SINGLE-SHARD A/B RUNS COMPLETE", flush=True)


def load_dose(name):
    return np.fromfile(OUT / name / "dose.raw", dtype=np.float32)


def analyze():
    shape = tuple(json.loads(
        Path("/mnt/sda/wuwei/ct_previous_full20_20260909/RT06423/manifest.json")
        .read_text())["gpu_shape_zyx"])
    ref = np.fromfile(
        "/mnt/sda/wuwei/ct_previous_full20_20260909/RT06423/topas_sum.raw",
        dtype="<f4").reshape((34, 440, 440)).astype(float)
    peak = float(ref.max())
    mask = ref >= 0.1 * peak

    def to_eval(a):
        return np.flip(a.reshape(shape).transpose(1, 2, 0), axis=2).astype(float)

    result = {}
    pairs = [("E1_strag12", "E1_strag10", "straggling_1.2->1.0"),
               ("E3_step05", "E3_step025", "step_0.5->0.25mm"),
               ("E3_step05", "E4_seedshift", "seed_shift_noise_floor")]
    if (OUT / "E2_off" / "dose.raw").exists() and (OUT / "E2_on" / "dose.raw").exists():
        pairs.append(("E2_off", "E2_on", "electron_OFF->ON"))
    for a_name, b_name, label in pairs:
        a = to_eval(load_dose(a_name))[mask]
        b = to_eval(load_dose(b_name))[mask]
        rr = ref[mask]
        delta = (b - a) / np.maximum(a, 1e-12) * 100  # % of A
        result[label] = {
            "mean_pct": float(delta.mean()), "rms_pct": float(np.sqrt((delta ** 2).mean())),
            "median_pct": float(np.median(delta)),
            "bands": {},
        }
        for lo, hi, band in ((0.1, 0.2, "10-20"), (0.2, 0.5, "20-50"), (0.5, 1.01, ">=50")):
            s = (rr >= lo * peak) & (rr < hi * peak)
            e = delta[s]
            result[label]["bands"][band] = {
                "mean_pct": float(e.mean()),
                "rms_pct": float(np.sqrt((e ** 2).mean())),
                "n": int(s.sum())}
        print(label, json.dumps(result[label], indent=1), flush=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    {"run": run_all, "analyze": analyze}[sys.argv[1]]()
