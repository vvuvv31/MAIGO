"""Lean electron-ON full20 for one frozen case (same sources/step, r3 joint).

Reuses the 20 frozen shard configs verbatim (step 0.5, same seeds) with the
current-source electron binary + r3 joint response. Each shard runs via the
validated single-shard probe (smoke mode, expects only
unvalidated_electron_joint_response). Overflow shards split like the frozen
runner. Outputs mirror the frozen layout (dose.raw per shard, gpu_sum.raw,
execution.json) plus gamma_coarse.json (0.5 mm lattice, frozen evaluator).

Usage: python3 benchmark/benchmark20260909/run_electron_full20_lean.py RT07575
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
from run_topas10x_gpu_benchmark import sha  # noqa: E402
from run_ct_electron_gamma_probe import run as probe_run  # noqa: E402
from evaluate_topas10x_gpu_gamma import pass_mask  # noqa: E402

FROZEN_BASE = Path("/mnt/sda/wuwei/ct_previous_full20_20260909")
OUT_BASE = Path("/mnt/sda/wuwei/electron_full20_lean_20260909")
BINARY = REPO / "build/oneapi-nvidia-electron-current/carbon_mc"
JOINT = Path("/mnt/sda/wuwei/schneider_electron_ct_runtime_r3_20260906/joint_response.csv")


def main(case):
    assert sha(BINARY), "missing binary"
    fz = FROZEN_BASE / case
    m = json.loads((fz / "manifest.json").read_text())
    ex = json.loads((fz / "execution.json").read_text())
    assert ex["status"] == "complete" and len(m["shards"]) == 20
    assert sum(t["histories"] for t in m["shards"]) == m["histories"]
    out = OUT_BASE / case
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(m["gpu_shape_zyx"])
    total = np.zeros(shape, dtype=np.float64)
    state = dict(status="running_experiment", case=case, binary_sha256=sha(BINARY),
                 joint_sha256=sha(JOINT), completed=[], overflow_excluded=[])
    (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    def task(src_cfg_path, dest, depth=0):
        report = dest / "run_report.json"
        if not report.exists():
            if subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid",
                     "--format=csv,noheader"], text=True).strip():
                raise RuntimeError("GPU occupied, retry later")
            try:
                probe_run(src_cfg_path, dest, BINARY, JOINT)
            except RuntimeError as e:
                if not str(e).startswith("OVERFLOW:"):
                    raise
                state["overflow_excluded"].append(str(dest))
                (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
                if depth >= 5:
                    raise ValueError("Overflow after five subdivisions")
                cfg = yaml.safe_load(src_cfg_path.read_text())
                import csv as _csv
                with Path(cfg["tps_spots_file"]).open() as f:
                    rows = list(_csv.DictReader(f))
                for half in (0, 1):
                    counts = [int(r["weight"]) // 2 + (int(r["weight"]) % 2 if half else 0)
                              for r in rows]
                    if not sum(counts):
                        continue
                    sub = dest / f"split_{half}"
                    sub.mkdir(exist_ok=True)
                    if not (sub / "source.yaml").exists():
                        with (sub / "spots.csv").open("x") as f:
                            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
                            w.writeheader()
                            for row, n in zip(rows, counts):
                                if n:
                                    w.writerow(dict(row, weight=n))
                        from run_topas10x_gpu_benchmark import config_write
                        config_write(sub / "source.yaml", dict(
                            cfg, number_of_histories=sum(counts),
                            tps_spots_file=str(sub / "spots.csv"),
                            random_seed=int(cfg["random_seed"]) + 1000000007 + half))
                    task(sub / "source.yaml", sub / "run", depth + 1)
                return
        r = json.loads(report.read_text())
        if r["binary_sha256"] != sha(BINARY) or sha(dest / "dose.raw") != r["dose_sha256"]:
            raise ValueError("Shard binary/dose pin mismatch")
        cfg = yaml.safe_load(src_cfg_path.read_text())
        if r["histories"] != cfg["number_of_histories"]:
            raise ValueError("Shard count mismatch")
        a = np.fromfile(dest / "dose.raw", dtype=np.float32).reshape(shape)
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("Invalid dose")
        total[:] += a
        state["completed"].append(dict(directory=str(dest), histories=r["histories"],
                                       dose_sha256=r["dose_sha256"]))
        (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
        print(case, dest.name, "done", len(state["completed"]), "leaves", flush=True)

    # source configs: frozen shard configs with joint key absent (file-driven ON)
    srcdir = out / "sources"
    srcdir.mkdir(exist_ok=True)
    import os as _os
    extra_faces = _os.environ.get("LEAN_EXACT_FACES", "") == "1"
    for i, t in enumerate(m["shards"], 1):
        cfg = yaml.safe_load((Path(t["directory"]) / "config.yaml").read_text())
        cfg.pop("ct_electron_segment_transport", None)
        if extra_faces:
            cfg.update(ct_secondary_exact_faces_diagnostic=True)
        from run_topas10x_gpu_benchmark import config_write
        sp = srcdir / f"config_{i:02d}.yaml"
        if sp.exists():
            assert yaml.safe_load(sp.read_text()) == cfg, "source changed"
        else:
            config_write(sp, cfg)
        task(sp, out / f"shard_{i:02d}")
    assert sum(x["histories"] for x in state["completed"]) == m["histories"], \
        "Incomplete histories"
    total.astype("<f4").tofile(out / "gpu_sum.raw")
    state.update(status="complete_experiment",
                 aggregate_sha256=sha(out / "gpu_sum.raw"))
    (out / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    # coarse gamma vs frozen TOPAS (0.5 mm lattice, frozen evaluator semantics)
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
            print(case, "ELECTRON-ON", key, results[key]["full_mask_percent"], flush=True)
    (out / "gamma_coarse.json").write_text(json.dumps(
        {"case": case, "status": "EXPERIMENT_NOT_PRODUCTION",
         "histories": m["histories"], "physics": "electron joint r3 ON, step 0.5, frozen sources",
         "method": "frozen >=10% mask, 0.5mm lattice + trilinear; 0mm same-voxel",
         "gamma": results, "gpu_sha256": sha(out / "gpu_sum.raw"),
         "reference_sha256": sha(fz / "topas_sum.raw")}, indent=2) + "\n")
    print("ELECTRON FULL20 COMPLETE", case, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        OUT_BASE = Path(sys.argv[2])
    main(sys.argv[1])
