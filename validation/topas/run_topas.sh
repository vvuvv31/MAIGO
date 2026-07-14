#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
topas_executable="${TOPAS_EXECUTABLE:-topas}"

cd "${script_dir}"
mkdir -p output
"${topas_executable}" carbon_200MeVu_water.txt

