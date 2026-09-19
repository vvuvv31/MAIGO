#!/usr/bin/env python3
"""Step 22: emit v2 worker + sbatch scripts from campaign_manifest.json.

14 jobs (13 projectiles + c12floors) x 12 CPUs = 168 <= 192 CPUs,
14 x 8 GiB = 112 GiB <= 160 GiB. Tasks run in manifest priority order so
high-demand data lands first. Each task verifies: TOPAS exit 0, worker CSV
exists and is non-empty. Full Step-25 validation is a separate pass.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("MAIGO_REPO_ROOT", Path(__file__).resolve().parents[3]))
V2 = Path(os.environ.get("MAIGO_V2_CAMPAIGN_ROOT", REPO / "extensions" / "work" / "schneider-secondary-v2"))
TOPAS_BIN = os.environ.get("TOPAS_BINARY", "topas")
TOPAS_G4_DATA = os.environ.get("TOPAS_G4_DATA_DIR", "")
TOPAS_LD_PATH = os.environ.get("TOPAS_LD_LIBRARY_PATH", "")
UUID = "00000000-0000-4000-8000-000000000022"

WORKER = """#!/usr/bin/env python3
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

V2 = Path({v2_root!r})

def run_task(t):
    case_dir = V2 / t["case_dir"]
    env = os.environ.copy()
    env["TOPAS_G4_DATA_DIR"] = "{g4data}"
    env["LD_LIBRARY_PATH"] = "{ldpath}:" + env.get("LD_LIBRARY_PATH", "")
    env["CARBON_CINEL02_OUTPUT_DIR"] = str(case_dir / "raw")
    env["CARBON_CINEL02_CAMPAIGN_UUID"] = "{uuid}"
    env["CARBON_CINEL02_RUN_TAG"] = case_dir.name
    env["CARBON_CINEL02_PRIMARY_ONLY"] = "true"
    env["CARBON_CINEL02_OVERWRITE_CAMPAIGN"] = "true"
    (case_dir / "raw").mkdir(parents=True, exist_ok=True)
    res = subprocess.run(["{topas}", str(case_dir / "run.txt")],
                         cwd=case_dir, env=env, capture_output=True, text=True)
    log = case_dir / "topas_stdout.log"
    log.write_text(res.stdout[-200000:] + "\\n---STDERR---\\n" + res.stderr[-50000:])
    if res.returncode != 0:
        return (t["task_index"], False, f"exit={{res.returncode}}")
    csvs = list((case_dir).glob("cinel03_exposure.worker_*.compact.csv"))
    if not csvs or max(p.stat().st_size for p in csvs) == 0:
        return (t["task_index"], False, "no-nonempty-worker-csv")
    return (t["task_index"], True, "ok")

def main():
    job = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    manifest_name = sys.argv[3] if len(sys.argv) > 3 else "campaign_manifest.json"
    jobtasks_name = sys.argv[4] if len(sys.argv) > 4 else "job_tasks.json"
    mpath = V2 / manifest_name
    if not mpath.exists():
        mpath = V2 / "scripts" / manifest_name
    manifest = json.load(open(mpath))
    pids = json.load(open(V2 / "scripts" / jobtasks_name))
    tasks = [t for t in manifest["tasks"] if t["task_index"] in pids[job]]
    print(f"[{{job}}] {{len(tasks)}} tasks, {{workers}} workers", flush=True)
    ok = fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, good, msg in ex.map(run_task, tasks):
            if good:
                ok += 1
            else:
                fail += 1
                print(f"[{{job}}] task {{idx}} FAILED: {{msg}}", flush=True)
    print(f"[{{job}}] done: {{ok}} ok, {{fail}} failed", flush=True)

if __name__ == "__main__":
    main()
"""

SBATCH = """#!/bin/bash
#SBATCH --job-name=v2_{job}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

python3 {worker} {job} 12
"""


def main():
    manifest = json.load(open(V2 / "campaign_manifest.json"))
    by_job = {}
    for t in manifest["tasks"]:
        pid = {5: {11: "b11", 10: "b10"}, 4: {9: "be9", 7: "be7", 10: "be10"},
               3: {7: "li7", 6: "li6"}, 2: {4: "he4", 3: "he3"},
               1: {1: "h1", 2: "h2", 3: "h3"}, 6: {11: "c11", 12: "c12"}}[
            t["projectile_Z"]][t["projectile_A"]]
        job = "c12floors" if (t["projectile_Z"], t["projectile_A"]) == (6, 12) else pid
        by_job.setdefault(job, []).append(t["task_index"])
    scripts = V2 / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "worker_v2.py").write_text(
        WORKER.format(g4data=TOPAS_G4_DATA, ldpath=TOPAS_LD_PATH, uuid=UUID,
                      topas=TOPAS_BIN, v2_root=str(V2)))
    (scripts / "worker_v2.py").chmod(0o755)
    (scripts / "job_tasks.json").write_text(json.dumps(by_job, indent=1))
    for job, idxs in sorted(by_job.items()):
        p = scripts / f"sbatch_{job}.sh"
        p.write_text(SBATCH.format(job=job, worker=str(scripts / "worker_v2.py")))
        p.chmod(0o755)
    total_cpu = 12 * len(by_job)
    total_mem = 8 * len(by_job)
    print(f"jobs: {len(by_job)} tasks: {sum(len(v) for v in by_job.values())} "
          f"cpus: {total_cpu} mem: {total_mem}G")
    for job, idxs in sorted(by_job.items()):
        print(f"  {job}: {len(idxs)} tasks")
    assert total_cpu <= 192 and total_mem <= 160, "resource budget exceeded"


if __name__ == "__main__":
    sys.exit(main())
