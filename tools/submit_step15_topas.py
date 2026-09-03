#!/usr/bin/env python3
"""
tools/submit_step15_topas.py

Submits and monitors all Step 15 TOPAS benchmark jobs on Slurm.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

MANIFEST_PATH = Path("/mnt/sda/wuwei/step15_schneider_mcs/manifest.json")

def main():
    if not MANIFEST_PATH.is_file():
        print(f"Error: missing manifest {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    submitted_jobs = {}

    for c in manifest["cases"]:
        cid = c["id"]
        slurm_script = c["slurm_script"]
        dose_csv = Path(c["topas_dose_csv"])

        if dose_csv.is_file() and dose_csv.stat().st_size > 1000:
            print(f"[{cid}] TOPAS output already exists ({dose_csv.stat().st_size} bytes), skipping submission.")
            continue

        print(f"Submitting job for {cid}: sbatch {slurm_script}")
        res = subprocess.run(["sbatch", slurm_script], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error submitting {cid}: {res.stderr}")
            continue

        out = res.stdout.strip()
        print(f"[{cid}] {out}")
        # Parse job ID from "Submitted batch job 12345"
        job_id = out.split()[-1]
        submitted_jobs[cid] = job_id

    if not submitted_jobs:
        print("All jobs already completed or none submitted.")
        return

    print(f"\nSubmitted {len(submitted_jobs)} jobs. Monitoring status...")

    while True:
        res = subprocess.run(["squeue", "-u", "wuwei", "-h", "-o", "%i %j %T"], capture_output=True, text=True)
        active_lines = [line.split() for line in res.stdout.strip().splitlines() if line.strip()]
        active_job_ids = {parts[0] for parts in active_lines if len(parts) >= 1}

        still_running = {cid: jid for cid, jid in submitted_jobs.items() if jid in active_job_ids}

        if not still_running:
            print("All submitted jobs finished!")
            break

        print(f"[{time.strftime('%X')}] Still running ({len(still_running)}/{len(submitted_jobs)}): {list(still_running.keys())}")
        time.sleep(15)

    # Check that all dose CSVs exist
    missing = []
    for c in manifest["cases"]:
        p = Path(c["topas_dose_csv"])
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(c["id"])

    if missing:
        print(f"WARNING: The following cases have missing or empty dose CSVs: {missing}")
        sys.exit(1)
    else:
        print("All 6 TOPAS dose CSVs generated successfully!")

if __name__ == "__main__":
    main()
