#!/usr/bin/env bash
# Run carbon_mc on an NVIDIA GPU via the oneAPI CUDA backend.
# Build first with: scripts/build_linux_oneapi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${SETVARS_COMPLETED:-}" ]]; then
  if [[ -n "${ONEAPI_ROOT:-}" && -f "${ONEAPI_ROOT}/setvars.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ONEAPI_ROOT}/setvars.sh"
  elif [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh
  else
    echo "Intel oneAPI setvars.sh not found. Set ONEAPI_ROOT or source setvars.sh." >&2
    exit 1
  fi
fi

# Prefer the dual-target Linux build; fall back to NVIDIA-only AOT.
EXE=""
for candidate in \
  build/oneapi-release/carbon_mc \
  build/oneapi-nvidia-release/carbon_mc
do
  if [[ -x "$candidate" ]]; then
    EXE="$candidate"
    break
  fi
done

if [[ -z "$EXE" ]]; then
  echo "No Linux SYCL binary found. Run scripts/build_linux_oneapi.sh first." >&2
  exit 1
fi

# Pin CUDA so a multi-GPU host never silently picks Intel/OpenCL.
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:gpu}"
# Prefer the CUDA plugin path; reduce noisy host-side caching growth.
export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"

CONFIG="${CONFIG:-config/beam_200MeVu_attenuation.yaml}"
OUTPUT="${OUTPUT:-out/linux_nvidia_attenuation_depth_dose.csv}"
mkdir -p "$(dirname "$OUTPUT")"

echo "Executable: $EXE"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
sycl-ls || true

# Prefer --device cuda so selection does not depend on the env filter alone.
DEVICE="${DEVICE:-cuda}"
# Unbuffered-ish progress: carbon_mc also line-buffers stdout.
export PYTHONUNBUFFERED=1
exec "$EXE" --config "$CONFIG" --device "$DEVICE" --output "$OUTPUT" "$@"
