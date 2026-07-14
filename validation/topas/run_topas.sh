#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
topas_executable="${TOPAS_EXECUTABLE:-topas}"
case_name="${1:-development}"

case "${case_name}" in
    smoke) parameter_file="carbon_200MeVu_water_smoke.txt" ;;
    development) parameter_file="carbon_200MeVu_water_development.txt" ;;
    reference) parameter_file="carbon_200MeVu_water.txt" ;;
    em-smoke) parameter_file="carbon_200MeVu_water_em_only_smoke.txt" ;;
    em-development) parameter_file="carbon_200MeVu_water_em_only_development.txt" ;;
    *)
        printf 'Usage: %s [smoke|development|reference|em-smoke|em-development]\n' "$0" >&2
        exit 2
        ;;
esac

cd "${script_dir}"
mkdir -p output
"${topas_executable}" "${parameter_file}" 2>&1 | tee "output/${case_name}_topas.log"
