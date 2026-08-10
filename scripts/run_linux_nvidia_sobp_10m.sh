#!/usr/bin/env bash
# Full 10M SOBP on NVIDIA CUDA (FP32 dose build), then compare to TOPAS + Arc.
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
CFG="${CFG:-out/sobp_benchmark/beam_sobp_10m_nvidia.yaml}"
LOG="${LOG:-out/sobp_benchmark/nvidia_10m_sobp_benchmark.log}"

if [[ ! -x "$EXE" ]]; then
  echo "Missing $EXE — build with CARBON_DOSE_FP32=ON first." >&2
  exit 1
fi

mkdir -p out/sobp_benchmark
echo "=== NVIDIA 10M SOBP FP32 $(date -Is) ===" | tee "$LOG"
echo "EXE=$EXE CFG=$CFG ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR" | tee -a "$LOG"
sycl-ls 2>&1 | tee -a "$LOG" || true

START=$(date +%s)
set +e
stdbuf -oL -eL "$EXE" --config "$CFG" --device cuda 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
echo "WallElapsedSeconds=$((END-START)) exit=$RC" | tee -a "$LOG"
if [[ "$RC" -ne 0 ]]; then
  exit "$RC"
fi

# total Gy CSV → Gy/primary (10,000,000 histories)
python3 validation/scripts/sparse_dose_to_mhd.py \
  out/sobp_benchmark/nvidia_10m_sobp_voxels_Gy.csv \
  out/sobp_benchmark/nvidia_10m_sobp_dose.mhd \
  --input-format gpu --shape 50 50 50 --spacing-mm 3 3 3 \
  --origin-mm -73.5 -73.5 1.5 --scale 1e-7 --units Gy/primary

python3 validation/scripts/compare_mhd_dose.py \
  --topas out/sobp_benchmark/topas_sobp_dose.mhd \
  --gpu out/sobp_benchmark/nvidia_10m_sobp_dose.mhd \
  --output out/sobp_benchmark/nvidia10m_vs_topas.metrics.json \
  --sobp-start-mm 50 --sobp-end-mm 100

python3 validation/scripts/compare_mhd_dose.py \
  --topas out/sobp_benchmark/arc_b580/gpu_sobp_dose.mhd \
  --gpu out/sobp_benchmark/nvidia_10m_sobp_dose.mhd \
  --output out/sobp_benchmark/nvidia10m_vs_arc.metrics.json \
  --sobp-start-mm 50 --sobp-end-mm 100

echo "=== done $(date -Is) ===" | tee -a "$LOG"
python3 - <<'PY'
import json
from pathlib import Path
for name in ["nvidia10m_vs_topas", "nvidia10m_vs_arc"]:
    p = Path(f"out/sobp_benchmark/{name}.metrics.json")
    d = json.loads(p.read_text())
    print(f"\n{name}:")
    print("  integral signed%:", d["integral"]["signed_percent"])
    print("  voxel nL1%:", d["voxel"]["normalized_L1_percent"])
    print("  voxel nRMSE%:", d["voxel"]["normalized_RMSE_to_topas_max_percent"])
    print("  high-dose mean abs%:", d["voxel"]["high_dose_mean_abs_percent"])
    print("  high-dose r:", d["voxel"]["high_dose_pearson_r"])
PY
