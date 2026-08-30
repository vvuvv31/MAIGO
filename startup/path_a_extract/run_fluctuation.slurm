#!/bin/bash
#SBATCH --job-name=c12_fluc
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

set -euo pipefail
ROOT=/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation
TOPAS=/home/wuwei/topas/topas-build/topas
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH="/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

[[ -x "$TOPAS" ]] || { echo "missing TOPAS: $TOPAS" >&2; exit 2; }
[[ -f "$ROOT/campaign_manifest.json" ]] || { echo "missing campaign" >&2; exit 2; }

python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

root = Path("/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation")
topas = "/home/wuwei/topas/topas-build/topas"
env = os.environ.copy()
campaign = json.loads((root / "campaign_manifest.json").read_text())
failed = []
for point in campaign["points"]:
    label = point["label"]
    run_dir = root / "runs" / label
    header = run_dir / "energy_loss_fluctuation.header"
    if header.is_file() and header.stat().st_size > 0:
        print(f"[skip] {label}", flush=True)
        continue
    log_path = run_dir / "topas.log"
    print(f"[run] {label}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        rc = subprocess.run(
            [topas, "run.txt"], cwd=run_dir, env=env,
            stdout=log, stderr=subprocess.STDOUT)
    if rc.returncode != 0 or not header.is_file():
        failed.append(label)
        print(f"[fail] {label} rc={rc.returncode}", flush=True)
    else:
        print(f"[ok] {label}", flush=True)
if failed:
    print("failed points:", ",".join(failed), file=sys.stderr)
    sys.exit(1)
print("all fluctuation points complete")
PY
