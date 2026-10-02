#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# == 0 )); then
    echo 'Usage: scripts/run.sh PRESET [carbon_mc options]'
    echo 'Available presets:'
    for config in "$project_root"/config/*.yaml; do
        name="${config##*/}"
        echo "  ${name%.yaml}"
    done
    exit 0
fi
preset="${1%.yaml}"
shift
if [[ "$preset" == */* || ! -f "$project_root/config/$preset.yaml" ]]; then
    echo "Unknown preset: $preset" >&2
    exit 2
fi
if [[ ! -x "$project_root/build/carbon_mc" ]]; then
    echo 'Build first with scripts/build.sh' >&2
    exit 2
fi
compiler="${CXX:-/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx}"
compiler_root="$(dirname -- "$(dirname -- "$compiler")")"
export LD_LIBRARY_PATH="$compiler_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$project_root"
exec ./build/carbon_mc --config "config/$preset.yaml" "$@"
