#!/usr/bin/env bash
# 10k-history scaled SOBP timing probe on NVIDIA (CUDA backend).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${SETVARS_COMPLETED:-}" ]]; then
  if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh
  fi
fi

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:gpu}"
EXE="${EXE:-build/oneapi-release/carbon_mc}"
CFG="${CFG:-out/sobp_benchmark/beam_sobp_10k_nvidia.yaml}"
LOG="${LOG:-out/sobp_benchmark/nvidia_10k_sobp_benchmark.log}"

if [[ ! -x "$EXE" ]]; then
  echo "Missing $EXE — build with scripts/build_linux_oneapi.sh first." >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG" >&2
  exit 1
fi

mkdir -p out/sobp_benchmark
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
echo "config=$CFG"
START=$(date +%s.%N)
set +e
"$EXE" --config "$CFG" --device cuda | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s.%N)
WALL=$(python3 -c "print(f'{float('$END')-float('$START'):.6f}')")
echo "WallElapsedSeconds=$WALL" | tee -a "$LOG"
echo "exit=$RC wall=${WALL}s log=$LOG"
exit "$RC"
