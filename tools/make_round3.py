#!/usr/bin/env python3
"""Round-3 thin-target gap fill (Step 24 follow-up).

Problem: thick-box (100-200mm) campaigns smear event energies over ~200 MeV/u,
leaving Poisson holes (5-9 MeV/u) near the top edge where only ceiling
campaigns contribute. Fix: thin boxes pin event energies to +-3 MeV/u of the
campaign energy without changing any interaction physics (thinner = better
energy definition).

Scope: every residual inter-node gap > 5.0 in v2 EXCEPT proven-null regions
(p+H below 281, documented in pH_unfillable_evidence.md). Plus one thick-box
p+H campaign at 311 MeV/u (100k histories) for the (304,318) gap.
Writes cases/thin_*/run.txt + scripts/round3_manifest.json +
scripts/job_tasks_round3.json + sbatch_round3_{a,b}.sh.
"""
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, "/mnt/sdb/wuwei/MAIGO/tools")
from generate_v2_manifest import (TEMPLATE, PROJ_ID, PROJ_PARTICLE, TGT,
                                  CAMPAIGN_UUID)

REPO = Path("/mnt/sdb/wuwei/MAIGO")
V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")
MAX_GAP = 5.0
SEED_BASE_R3 = 3500000
TOPAS_BIN = "/home/wuwei/topas/topas-build/topas"


def load_nodes(bin_path):
    with open(bin_path, "rb") as f:
        h = f.read(136)
        cc, ic, pc = struct.unpack_from("<3Q", h, 8 + 28)
        f.seek(136 + cc * 36)
        raw = f.read(ic * 476)
    chans = defaultdict(list)
    for i in range(ic):
        b = raw[i * 476:(i + 1) * 476]
        pz = struct.unpack_from("<h", b, 36)[0]
        pa = struct.unpack_from("<h", b, 38)[0]
        epu = struct.unpack_from("<f", b, 56)[0]
        tz = struct.unpack_from("<h", b, 108)[0]
        chans[(pz, pa, tz)].append(epu)
    return {k: sorted(v) for k, v in chans.items()}


def main():
    ch = load_nodes(REPO / "data/schneider/cinel03_secondary_targets_v2.bin")
    gaps = []  # (pz,pa,tz,lo,hi)
    for k, v in sorted(ch.items()):
        if k == (1, 1, 1):
            continue  # p+H handled separately below
        for i in range(len(v) - 1):
            if v[i + 1] - v[i] > MAX_GAP + 1e-6:
                gaps.append((k[0], k[1], k[2], v[i], v[i + 1]))
    print(f"residual gaps (excl p+H): {len(gaps)}")

    tasks = []
    for (pz, pa, tz, lo, hi) in gaps:
        e = (lo + hi) / 2.0
        if e < 10:
            hist, hlz = 20000, 5.0
        elif tz == 1:
            hist, hlz = 20000, 20.0
        else:
            hist, hlz = 10000, 5.0
        tasks.append((pz, pa, tz, e, hist, hlz, "round3-thin-gapfill"))
    # p+H (304,318) gap: thick H box (spread tolerable), 100k histories.
    tasks.append((1, 1, 1, 311.0, 100000, 200.0, "round3-pH-gap"))
    print(f"round-3 tasks: {len(tasks)}")

    cases_dir = V2 / "cases"
    manifest_tasks = []
    for i, (pz, pa, tz, e, hist, hlz, kind) in enumerate(tasks):
        pid = PROJ_ID[(pz, pa)]
        sym = TGT[tz][1]
        tag = f"thin_{pid}_{sym}_E{e:.2f}".replace(".", "p")
        case_dir = cases_dir / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        _, _, mat_name, elem_name, density, _ = TGT[tz]
        content = TEMPLATE.format(
            mat_name=mat_name, hlz=hlz, elem_name=elem_name, density=density,
            part_name=PROJ_PARTICLE[(pz, pa)], beam_energy=e * pa,
            histories=hist, seed=SEED_BASE_R3 + i, pz=pz, pa=pa)
        (case_dir / "run.txt").write_text(content)
        import hashlib
        manifest_tasks.append({
            "task_index": i, "projectile_Z": pz, "projectile_A": pa,
            "target_Z": tz, "energy_MeV_per_u": round(e, 4),
            "requested_events": 30, "requested_histories": hist,
            "thin_hlz_mm": hlz, "priority": i + 1, "kind": kind,
            "gap_bracket": None,
            "source_of_demand": "v2 residual inter-node gap",
            "expected_output": f"cases/{tag}/cinel03_exposure.worker_*.compact.csv",
            "random_seed": SEED_BASE_R3 + i,
            "TOPAS_config_hash": hashlib.sha256(content.encode()).hexdigest(),
            "case_dir": f"cases/{tag}",
        })
    for t, g in zip(manifest_tasks, gaps + [None]):
        if g is not None:
            t["gap_bracket"] = [round(g[3], 2), round(g[4], 2)]
    (V2 / "scripts" / "round3_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "round": 3, "seed_base": SEED_BASE_R3,
         "topas_binary": TOPAS_BIN, "tasks": manifest_tasks}, indent=1))
    half = (len(manifest_tasks) + 1) // 2
    by_job = {"r3a": list(range(half)), "r3b": list(range(half, len(manifest_tasks)))}
    (V2 / "scripts" / "job_tasks_round3.json").write_text(json.dumps(by_job, indent=1))
    for job in ("r3a", "r3b"):
        p = V2 / "scripts" / f"sbatch_round3_{job}.sh"
        p.write_text(f"""#!/bin/bash
#SBATCH --job-name=v2r3_{job}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

python3 /mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/scripts/worker_v2.py {job} 12 round3_manifest.json job_tasks_round3.json
""")
        p.chmod(0o755)
    print("wrote round3 manifest + 2 sbatch scripts")


if __name__ == "__main__":
    sys.exit(main())
