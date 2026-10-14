#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${CXX:-/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx}"
compiler_root="$(dirname -- "$(dirname -- "$compiler")")"
export LD_LIBRARY_PATH="$compiler_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build_directory="${BUILD_DIR:-$project_root/build}"
if [[ "$build_directory" != /* ]]; then build_directory="$project_root/$build_directory"; fi
# RTX 2080 Ti default; pass -DCARBON_CUDA_ARCH=sm_86/sm_89 for other GPUs.
cmake -S "$project_root" -B "$build_directory" -DCMAKE_CXX_COMPILER="$compiler" -DCMAKE_BUILD_TYPE=Release -DCARBON_CUDA_ARCH=sm_75 "$@"
cmake --build "$build_directory" --target carbon_mc -j "${BUILD_JOBS:-6}"
