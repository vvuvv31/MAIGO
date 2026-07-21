#!/usr/bin/env bash
# Configure and build the Linux dual-target oneAPI binary
# (Intel SPIR-V + NVIDIA CUDA PTX).
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

if ! command -v icpx >/dev/null 2>&1; then
  echo "icpx not on PATH after loading oneAPI." >&2
  exit 1
fi

PRESET="${1:-oneapi-release}"
case "$PRESET" in
  oneapi-release|oneapi-nvidia-release|oneapi-intel-release) ;;
  *)
    echo "Usage: $0 [oneapi-release|oneapi-nvidia-release|oneapi-intel-release]" >&2
    exit 1
    ;;
esac

# Prefer Ninja when available; fall back to Unix Makefiles (common on minimal WSL).
GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1 || command -v ninja-build >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
else
  GENERATOR_ARGS=(-G "Unix Makefiles")
fi

# Optional SM arch for NVIDIA AOT (e.g. CARBON_CUDA_ARCH=sm_75 for TITAN RTX).
CUDA_ARCH_ARGS=()
if [[ -n "${CARBON_CUDA_ARCH:-}" ]]; then
  CUDA_ARCH_ARGS=(-DCARBON_CUDA_ARCH="${CARBON_CUDA_ARCH}")
fi

rm -rf "build/${PRESET}"
# Map preset cache vars explicitly so CMake 3.22 works without --fresh/--preset
# when the generator in the preset is unavailable.
case "$PRESET" in
  oneapi-release)
    cmake -S . -B "build/${PRESET}" "${GENERATOR_ARGS[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCARBON_ENABLE_SYCL=ON \
      -DCARBON_BUILD_TESTS=ON \
      -DCARBON_SYCL_TARGETS="spir64,nvptx64-nvidia-cuda" \
      "${CUDA_ARCH_ARGS[@]}"
    ;;
  oneapi-nvidia-release)
    cmake -S . -B "build/${PRESET}" "${GENERATOR_ARGS[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCARBON_ENABLE_SYCL=ON \
      -DCARBON_BUILD_TESTS=ON \
      -DCARBON_SYCL_TARGETS="nvptx64-nvidia-cuda" \
      "${CUDA_ARCH_ARGS[@]}"
    ;;
  oneapi-intel-release)
    cmake -S . -B "build/${PRESET}" "${GENERATOR_ARGS[@]}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCARBON_ENABLE_SYCL=ON \
      -DCARBON_BUILD_TESTS=ON \
      -DCARBON_SYCL_TARGETS="spir64"
    ;;
esac

cmake --build "build/${PRESET}" -j"$(nproc 2>/dev/null || echo 4)"
# Unset GPU-only filters so unit tests can still select a SYCL CPU if present.
env -u ONEAPI_DEVICE_SELECTOR ctest --test-dir "build/${PRESET}" --output-on-failure
echo "Built: build/${PRESET}/carbon_mc"
