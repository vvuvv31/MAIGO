"""Exact-faces full20 for one frozen case (same sources/seeds, smoke mode).

Reuses the 20 frozen shard configs + run_mode smoke +
ct_secondary_exact_faces_diagnostic=true, frozen binary. Overflow shards
split like the frozen runner. Mirrors frozen layout + gamma_coarse.json
(0.5 mm lattice, frozen evaluator semantics).

Usage: python3 benchmark/benchmark20260909/run_exactfaces_full20.py 20022516
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from run_topas10x_gpu_benchmark import sha, config_write, save  # noqa: E402
from evaluate_topas10x_gpu_gamma import pass_mask  # noqa: E402

FROZEN_BASE = Path("/mnt/sda/wuwei/ct_previous_full20_20260909")
OUT_BASE = Path("/mnt/sda/wuwei/exactfaces_full20_20260910")
BINARY = REPO / "build/oneapi-nvidia-water-electron-full/carbon_mc"
PIN = "4df8036cd26b0a8dd7e6f87f0d45820adc697bb0a5c25615f11e564c8abd2e08"
ENV = dict(os.environ, ONEAPI_DEVICE_SELECTOR="cuda:*",
           LD_LIBRARY_PATH="/home/wuwei/sycl_workspace/llvm/build/install/lib")


def main(case):
    assert sha(BINARY) == PIN, "Frozen binary pin mismatch"
    fz = FROZEN_BASE / case
    m = json.loads((fz / "manifest.json").read_text())
    ex = json.loads((fz / "execution.json").read_text())
    assert ex["status"] == "complete" and len(m["shards"]) == 20
    out = OUT_BASE / case
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(m["gpu_shape_zyx"])
    total = np.zeros(shape, dtype=np.float64)
    state = dict(status="running", case=case, binary_sha256=PIN,
                 completed=[], overflow_attempts=[])

    def task(d, cfg, depth=0):
        qpath = d / "out/config/quality_report.json"

        def _usable():
            try:
                if not (qpath.exists() and (d / "dose.raw").exists()):
                    return False
                if (d / "dose.raw").stat().st_size == 0:
                    return False
                json.loads(qpath.read_text())
                return True
            except (ValueError, OSError):
                return False

        if not _usable():
            if subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid",
                     "--format=csv,noheader"], text=True).strip():
                raise RuntimeError("GPU occupied, retry later")
            if sha(BINARY) != PIN:
                raise RuntimeError("Executable changed during run")
            start = time.monotonic()
            with (d / "gpu.log").open("w") as log:
                p = subprocess.run([str(BINARY), "--config", str(d / "config.yaml"),
                                    "--device", "cuda", "--voxel-dose-mhd",
                                    str(d / "dose.mhd")],
                                   cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            (d / "timing.json").write_text(json.dumps(
                {"wall_seconds": time.monotonic() - start,
                 "returncode": p.returncode}) + "\n")
        if not qpath.exists():
            raise RuntimeError("Missing quality " + str(d))
        q = json.loads(qpath.read_text())
        if q["queue_overflow_count"] or q.get("queue_overflow_energy_MeV", 0):
            state["overflow_attempts"].append(str(d))
            (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
            if depth >= 5:
                raise RuntimeError("Overflow after five splits")
            with Path(cfg["tps_spots_file"]).open() as f:
                rows = list(csv.DictReader(f))
            for half in range(2):
                selected = [dict(r, weight=int(r["weight"]) // 2 +
                                 (int(r["weight"]) % 2 if half else 0)) for r in rows]
                selected = [r for r in selected if r["weight"]]
                if not selected:
                    continue
                sub = d / f"split_{half}"
                sub.mkdir(exist_ok=True)
                with (sub / "spots.csv").open("w") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(selected)
                cc = dict(cfg, tps_spots_file=str(sub / "spots.csv"),
                          number_of_histories=sum(r["weight"] for r in selected),
                          random_seed=int(cfg["random_seed"]) + 100003 * (half + 1) + depth)
                config_write(sub / "config.yaml", cc)
                if not (sub / "data").exists():
                    (sub / "data").symlink_to(REPO / "data", target_is_directory=True)
                task(sub, cc, depth + 1)
            return
        if not q["accepted"] or q["failures"]:
            raise RuntimeError("Quality failure " + str(d) + ": " + str(q["failures"]))
        ledger = json.loads((d / "out/config/energy_ledger.json").read_text())
        if ledger["histories"] != cfg["number_of_histories"]:
            raise RuntimeError("History mismatch")
        a = np.fromfile(d / "dose.raw", "<f4").reshape(total.shape)
        if not np.isfinite(a).all() or np.any(a < 0):
            raise RuntimeError("Invalid dose")
        total[:] += a
        log = (d / "gpu.log").read_text()
        elapsed = float(re.search(r"Elapsed: ([\d.e+-]+)", log)[1])
        state["completed"].append(dict(
            directory=str(d), histories=cfg["number_of_histories"],
            elapsed_seconds=elapsed, dose_sha256=sha(d / "dose.raw"),
            quality_sha256=sha(qpath), config_sha256=sha(d / "config.yaml")))
        (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
        print(case, d.name, "done", elapsed, "s", flush=True)

    for i, t in enumerate(m["shards"], 1):
        cfg = yaml.safe_load((Path(t["directory"]) / "config.yaml").read_text())
        cfg.update(run_mode="smoke", ct_secondary_exact_faces_diagnostic=True)
        d = out / f"shard_{i:02d}"
        d.mkdir(exist_ok=True)
        try:
            existing = yaml.safe_load((d / "config.yaml").read_text()) \
                if (d / "config.yaml").exists() else None
        except (ValueError, OSError):
            existing = None
        if existing is None:
            config_write(d / "config.yaml", cfg)
        else:
            assert existing == cfg, "config changed"
        if not (d / "data").exists():
            (d / "data").symlink_to(REPO / "data", target_is_directory=True)
        task(d, cfg)
    assert sum(x["histories"] for x in state["completed"]) == m["histories"]
    total.astype("<f4").tofile(out / "gpu_sum.raw")
    state.update(status="complete", aggregate_sha256=sha(out / "gpu_sum.raw"))
    (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    g = total.reshape(shape)
    if m["mapping"] == "packed_xneg":
        g = np.flip(g.transpose(1, 2, 0), axis=2)
    elif m["mapping"] != "native":
        raise ValueError("Unknown mapping")
    r = np.fromfile(fz / "topas_sum.raw", dtype="<f4").reshape(m["topas_shape_zyx"])
    mask = r >= 0.1 * r.max()
    pts = np.argwhere(mask)
    results = {}
    for dd, dta in [(3, 3), (2, 2), (1, 1), (3, 0)]:
        for local in [False, True]:
            key = f"{'local' if local else 'global'}_{dd}pct_{dta}mm"
            p = pass_mask(g, r, pts, np.array(m["spacing_zyx"]), dd, dta, local)
            results[key] = {"full_mask_percent": 100 * float(p.mean()),
                            "passed": int(p.sum()), "evaluated": len(pts)}
            print(case, "EXACTFACES", key, results[key]["full_mask_percent"], flush=True)
    save(out / "gamma_coarse.json",
         {"case": case, "status": "EXPERIMENT_NOT_PRODUCTION",
          "histories": m["histories"],
          "physics": "frozen sources, secondary exact faces ON, smoke mode",
          "method": "frozen >=10% mask, 0.5mm lattice + trilinear; 0mm same-voxel",
          "gamma": results, "gpu_sha256": sha(out / "gpu_sum.raw"),
          "reference_sha256": sha(fz / "topas_sum.raw")})
    print("EXACTFACES FULL20 COMPLETE", case, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
