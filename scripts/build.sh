#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${CXX:-/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx}"
compiler_root="$(dirname -- "$(dirname -- "$compiler")")"
export LD_LIBRARY_PATH="$compiler_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S "$project_root" -B "$project_root/build" -DCMAKE_CXX_COMPILER="$compiler" -DCMAKE_BUILD_TYPE=Release
cmake --build "$project_root/build" --target carbon_mc -j "${BUILD_JOBS:-6}"
