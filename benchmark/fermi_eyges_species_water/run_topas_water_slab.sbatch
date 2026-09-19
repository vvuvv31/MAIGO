#!/bin/bash
#SBATCH --job-name=fe_water_species
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
# Thin homogeneous slabs use no DICOM/voxel dose structures. Twelve concurrent
# 16-thread TOPAS processes stay within this allocation; retain the full CPU
# request so the calibration matrix completes in two waves.
# The completed 24-case calibration used about 1.03 GiB MaxRSS.  Keep ample
# headroom for TOPAS/runtime variation without reserving the old 48 GiB.
#SBATCH --mem=8G
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail

run_root=${RUN_ROOT:-/mnt/sda/wuwei/fe_species_water_em10gev_20260919}
topas=/home/wuwei/topas/all-ion-elastic-build-20260911/topas
max_parallel=12

# The manifest is authoritative. Do not accidentally treat analysis/output
# directories such as gpu/ as TOPAS cases.
mapfile -t cases < <(python3 - "$run_root/manifest.json" "$run_root" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
for case in manifest["cases"]:
    print(root / case["name"])
PY
)
fail=0
active=0
pids=()

wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  pids=()
  active=0
}

for case_dir in "${cases[@]}"; do
  (
    cd "$case_dir"
    "$topas" run.txt > topas.log 2>&1
    test -s output/slab_exit.phsp
    test -s output/slab_exit.header
  ) &
  pids+=("$!")
  active=$((active + 1))
  if [ "$active" -ge "$max_parallel" ]; then
    wait_batch
  fi
done

if [ "$active" -gt 0 ]; then
  wait_batch
fi

exit "$fail"
