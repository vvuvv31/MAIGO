#!/bin/bash
set -euo pipefail

# Execute this script after entering the already allocated cpu188 node, for
# example through the wuwei@10.10.10.4 login host.  The shared defaults match
# the local reference layout; override either value when cpu188 mounts differ.
run_root=${1:-/mnt/sda/wuwei/fe_species_water_em10gev_20260919}
topas=${TOPAS_BINARY:-/home/wuwei/topas/all-ion-elastic-build-20260911/topas}
max_parallel=${TOPAS_MAX_PARALLEL:-12}

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
  if [ -s "$case_dir/output/slab_exit.phsp" ] &&
     [ -s "$case_dir/output/slab_exit.header" ]; then
    continue
  fi
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

date --iso-8601=seconds > "$run_root/completed_at.txt"
exit "$fail"
