#!/usr/bin/env python3
"""NEED-fill + cap-fill task generator (Step 28): thin-box campaigns for
XS-substantial/never-probed energies below floors, plus uniform cap to 460.

Rules (no silent choices):
- Below floor F with max-null-beam N: fill (N, F) at <=2.5 spacing (thin box),
  but ONLY grid points where XS dump > 0 AND no null campaign >= E.
  Points with XS==0 are left for rate-zeroing (no task).
- Cap: every channel extended to 460 with <=5 spacing (thin box).
- C12 channels: fills at miss-cluster energies (from Tier-C miss log).
- p+H (250,281): thin fills at 255-280 (XS>0 there, never probed).
Task kinds: needfill / capfill / c12fill / phfill.
"""
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/mnt/sdb/wuwei/MAIGO")
V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")
MAX_GAP_FILL = 2.5
CAP_EMAX = 460.0


def package_channels():
    p = REPO / "data/schneider/cinel03_secondary_targets_v2.bin"
    with open(p, "rb") as f:
        h = f.read(136)
        cc, ic, pc = struct.unpack_from("<3Q", h, 8 + 28)
        f.seek(136 + cc * 36)
        raw = f.read(ic * 476)
    ch = defaultdict(list)
    for i in range(ic):
        b = raw[i * 476:(i + 1) * 476]
        pz = struct.unpack_from("<h", b, 36)[0]
        pa = struct.unpack_from("<h", b, 38)[0]
        epu = struct.unpack_from("<f", b, 56)[0]
        tz = struct.unpack_from("<h", b, 108)[0]
        ch[(pz, pa, tz)].append(epu)
    return {k: (min(v), max(v)) for k, v in ch.items()}


PID = {(5, 11): "b11", (5, 10): "b10", (4, 9): "be9", (4, 7): "be7",
       (4, 10): "be10", (3, 7): "li7", (3, 6): "li6", (2, 4): "he4",
       (2, 3): "he3", (1, 1): "h1", (1, 2): "h2", (1, 3): "h3", (6, 11): "c11"}


def load_dump(pid):
    d = json.load(open(f"/mnt/sda/wuwei/secondary-rates-v2-1/raw/{pid}_dump.json"))
    secs = d["sections"]
    egrid = np.array([g["energy_mevu"] for g in secs[0]["grid"]])
    tz_list = [el["target_z"] for el in secs[0]["grid"][0]["elements"]]
    arr = np.zeros((len(egrid), len(tz_list)))
    for s, sec in enumerate(secs):
        for j, g in enumerate(sec["grid"]):
            for el in g["elements"]:
                ti = tz_list.index(el["target_z"])
                arr[j, ti] = max(arr[j, ti], el["mass_partial_per_mm_at_1g_cm3"])
    return egrid, tz_list, arr


def main():
    floors = package_channels()
    val = json.load(open(V2 / "raw_validation_all.json"))
    man = json.load(open(V2 / "campaign_manifest_all.json"))
    by_task = {t["task_index"]: t for t in man["tasks"]}
    maxnull = defaultdict(lambda: -1.0)
    for r in val["results"]:
        if r["pass"]:
            continue
        t = by_task[r["task_index"]]
        if not r["checks"].get("files_exist_nonempty"):
            k = (t["projectile_Z"], t["projectile_A"], t["target_Z"])
            maxnull[k] = max(maxnull[k], t["energy_MeV_per_u"])
    dumps = {pr: load_dump(pid) for pr, pid in PID.items()}
    tasks = []

    def xs_at(pr, tz, e):
        egrid, tz_list, arr = dumps[pr]
        j = int(np.clip(round((e - 0.1) / 0.5), 0, len(egrid) - 1))
        return float(arr[j, tz_list.index(tz)])

    # 1. below-floor fills: (maxnull, floor) at <=2.5 where XS>0
    for (pz, pa, tz), (flo, fhi) in sorted(floors.items()):
        nul = maxnull.get((pz, pa, tz), -1.0)
        lo = max(nul, 0.5)
        if flo - lo < 1.0:
            continue
        import math
        n = max(1, math.ceil((flo - lo) / MAX_GAP_FILL - 1e-9))
        for k in range(n + 1):
            e = lo + k * (flo - lo) / n
            if e >= flo - 1e-9:
                continue
            if xs_at((pz, pa), tz, e) == 0.0:
                continue  # rate-zero class, no task
            tasks.append((pz, pa, tz, round(e, 2), 30000, 5.0, "needfill"))
    # 2. cap fills to 460 (thin), all channels
    for (pz, pa, tz), (flo, fhi) in sorted(floors.items()):
        if fhi >= CAP_EMAX - 1e-9:
            continue
        import math
        n = max(1, math.ceil((CAP_EMAX - fhi) / 5.0 - 1e-9))
        for k in range(1, n + 1):
            e = fhi + k * (CAP_EMAX - fhi) / n
            tasks.append((pz, pa, tz, round(e, 2), 10000, 5.0, "capfill"))
    # 3. p+H (250,281): XS>0, never probed
    for e in (255.0, 260.0, 265.0, 270.0, 275.0, 280.0):
        tasks.append((1, 1, 1, e, 100000, 200.0, "phfill"))
    # 4. C12 fills at miss clusters (floors from C12 package)
    c12_tasks = [(6, 12, 1, 0.75), (6, 12, 1, 1.0), (6, 12, 8, 1.5),
                 (6, 12, 8, 2.0), (6, 12, 8, 2.5), (6, 12, 8, 3.0),
                 (6, 12, 8, 3.5), (6, 12, 7, 1.5), (6, 12, 7, 2.0),
                 (6, 12, 7, 2.5), (6, 12, 6, 1.5), (6, 12, 6, 2.0)]
    for pz, pa, tz, e in c12_tasks:
        tasks.append((pz, pa, tz, e, 30000, 5.0, "c12fill"))
    import hashlib
    import sys
    sys.path.insert(0, "/mnt/sdb/wuwei/MAIGO/tools")
    from generate_v2_manifest import TEMPLATE, PROJ_ID, PROJ_PARTICLE, TGT
    PROJ_PARTICLE[(6, 12)] = "GenericIon(6,12)"
    PROJ_ID[(6, 12)] = "c12"
    out = []
    for i, (pz, pa, tz, e, hist, hlz, kind) in enumerate(tasks):
        tag = f"nf_{PROJ_ID[(pz,pa)]}_{TGT[tz][1]}_E{e:.2f}".replace(".", "p")
        case_dir = V2 / "cases" / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        _, _, mat, elem, dens, _ = TGT[tz]
        content = TEMPLATE.format(
            mat_name=mat, hlz=hlz, elem_name=elem, density=dens,
            part_name=PROJ_PARTICLE[(pz, pa)], beam_energy=e * pa,
            histories=hist, seed=3900000 + i, pz=pz, pa=pa)
        (case_dir / "run.txt").write_text(content)
        out.append({"task_index": i, "projectile_Z": pz, "projectile_A": pa,
                    "target_Z": tz, "energy_MeV_per_u": e,
                    "requested_events": 30, "requested_histories": hist,
                    "thin_hlz_mm": hlz, "kind": kind,
                    "source_of_demand": "step28 NEED arbitration / cap / miss clusters",
                    "expected_output": f"cases/{tag}/cinel03_exposure.worker_*.compact.csv",
                    "random_seed": 3900000 + i,
                    "TOPAS_config_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "case_dir": f"cases/{tag}"})
    json.dump({"schema_version": 1, "round": 7, "seed_base": 3900000, "tasks": out},
              open(V2 / "scripts/round7_manifest.json", "w"), indent=1)
    from collections import Counter
    print("round-7 tasks:", len(out), dict(Counter(t["kind"] for t in out)))
    # sbatch: 4 jobs x 12 workers (48 tasks each-ish); 4x12=48 CPUs
    by_job, nj = {}, 4
    for i, t in enumerate(out):
        by_job.setdefault(f"r7{i % nj}", []).append(t["task_index"])
    json.dump(by_job, open(V2 / "scripts/job_tasks_round7.json", "w"), indent=1)
    for job in sorted(by_job):
        p = V2 / "scripts" / f"sbatch_round7_{job}.sh"
        p.write_text(f"""#!/bin/bash
#SBATCH --job-name=v2r7_{job}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

python3 /mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/scripts/worker_v2.py {job} 12 round7_manifest.json job_tasks_round7.json
""")
        p.chmod(0o755)
    print("jobs:", {k: len(v) for k, v in sorted(by_job.items())})


if __name__ == "__main__":
    sys_exit = main()
