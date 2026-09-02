#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

MANIFEST = Path("/mnt/sda/wuwei/step13_primary_ct/manifest.json")
with open(MANIFEST) as f:
    data = json.load(f)

submitted = []
for case in data["cases"]:
    if case.get("precomputed_in_step08", False):
        continue
    script = case.get("slurm_script", "")
    if not script or not Path(script).exists():
        continue

    cmd = ["sbatch", script]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    err = res.stderr.strip()
    print(f"Submitting {case["id"]}: {out} {err}")
    # e.g. Submitted batch job 12345
    job_id = None
    if "Submitted batch job" in out:
        job_id = int(out.split()[-1])
    submitted.append({
        "id": case["id"],
        "job_id": job_id,
        "script": script,
        "status": "SUBMITTED" if job_id else "FAILED",
        "output": out,
        "error": err
    })

submission_record = Path("/mnt/sda/wuwei/step13_primary_ct/slurm_submitted_jobs.json")
with open(submission_record, "w") as f:
    json.dump(submitted, f, indent=2)

print(f"Submitted {len(submitted)} jobs. Record saved to {submission_record}")
