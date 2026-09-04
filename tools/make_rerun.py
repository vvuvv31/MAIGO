#!/usr/bin/env python3
"""Pinpoint rerun generator (Step 25): same (proj,target,E) only, 4x histories.

Reads campaign_manifest.json + raw_validation.json. For every FAILED task
(including expected-null floors/probes: one 4x round to give the null claim
statistical weight), writes <case>/rerun1/run.txt with histories*4 and
seed+500000, plus scripts/rerun_manifest.json + scripts/job_tasks_rerun.json.

Backs up scripts/job_tasks.json to scripts/job_tasks_wave1.json first.
"""
import json
import shutil
import sys
from pathlib import Path

V2 = Path("/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2")


def main():
    manifest = json.load(open(V2 / "campaign_manifest.json"))
    validation = json.load(open(V2 / "raw_validation.json"))
    by_task = {t["task_index"]: t for t in manifest["tasks"]}
    failed = [r for r in validation["results"] if not r["pass"]]
    print(f"failed tasks: {len(failed)}")

    scripts = V2 / "scripts"
    backup = scripts / "job_tasks_wave1.json"
    if not backup.exists():
        shutil.copy(scripts / "job_tasks.json", backup)

    rerun_tasks = []
    for r in failed:
        t = by_task[r["task_index"]]
        rd = V2 / t["case_dir"] / "rerun1"
        rd.mkdir(parents=True, exist_ok=True)
        base = (V2 / t["case_dir"] / "run.txt").read_text()
        lines = []
        for line in base.splitlines():
            if line.startswith("i:So/PrimaryBeam/NumberOfHistoriesInRun"):
                line = f"i:So/PrimaryBeam/NumberOfHistoriesInRun = {t['requested_histories'] * 4}"
            if line.startswith("i:Ts/Seed"):
                line = f"i:Ts/Seed = {t['random_seed'] + 500000}"
            lines.append(line)
        (rd / "run.txt").write_text("\n".join(lines) + "\n")
        rerun_tasks.append({**t, "rerun_of": t["task_index"],
                            "requested_histories": t["requested_histories"] * 4,
                            "random_seed": t["random_seed"] + 500000,
                            "case_dir": t["case_dir"] + "/rerun1"})
    (scripts / "rerun_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "rerun_of_manifest": "campaign_manifest.json",
         "tasks": rerun_tasks}, indent=1))
    by_job = {}
    for t in rerun_tasks:
        pid = {5: {11: "b11", 10: "b10"}, 4: {9: "be9", 7: "be7", 10: "be10"},
               3: {7: "li7", 6: "li6"}, 2: {4: "he4", 3: "he3"},
               1: {1: "h1", 2: "h2", 3: "h3"}, 6: {11: "c11", 12: "c12"}}[
            t["projectile_Z"]][t["projectile_A"]]
        job = "c12floors" if (t["projectile_Z"], t["projectile_A"]) == (6, 12) else pid
        by_job.setdefault(job, []).append(t["rerun_of"])
    # worker resolves rerun case dirs via rerun_manifest; map job -> parent idx
    (scripts / "job_tasks_rerun.json").write_text(json.dumps(by_job, indent=1))
    for job, idxs in sorted(by_job.items()):
        p = scripts / f"sbatch_rerun_{job}.sh"
        p.write_text(f"""#!/bin/bash
#SBATCH --job-name=v2r_{job}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

python3 /mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/scripts/worker_v2.py {job} 12 rerun_manifest.json job_tasks_rerun.json
""")
        p.chmod(0o755)
    print(f"rerun tasks: {len(rerun_tasks)} in {len(by_job)} jobs")
    for job, idxs in sorted(by_job.items()):
        print(f"  {job}: {len(idxs)} tasks")


if __name__ == "__main__":
    sys.exit(main())
