#!/usr/bin/env bash
# Run multi-energy TOPAS IDD with 1% RMS BeamEnergySpread on remote host.
# Usage:
#   bash validation/topas/run_multi_energy_espread1_remote.sh all development
#   bash validation/topas/run_multi_energy_espread1_remote.sh 200 smoke
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"

energy="${1:-}"
case_name="${2:-development}"

usage() {
    printf 'Usage: %s ENERGY [smoke|development]\n' "$0" >&2
    printf '  ENERGY: 100 | 150 | 200 | 250 | 300 | 350 | 400 | all\n' >&2
    exit 2
}

[[ -n "${energy}" ]] || usage
case "${case_name}" in
    smoke|development) ;;
    *) usage ;;
esac

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/software/gate/G4DATA}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_one() {
    local e="$1"
    local parameter_file
    case "${case_name}" in
        smoke) parameter_file="carbon_${e}MeVu_water_espread1_smoke.txt" ;;
        development) parameter_file="carbon_${e}MeVu_water_espread1_development_remote.txt" ;;
    esac
    if [[ ! -f "${topas_dir}/${parameter_file}" ]]; then
        printf 'Missing parameter file: %s\n' "${topas_dir}/${parameter_file}" >&2
        exit 1
    fi
    if [[ ! -x "${install_dir}/bin/topas" ]]; then
        printf 'TOPAS executable not found: %s/bin/topas\n' "${install_dir}" >&2
        exit 1
    fi
    cd "${topas_dir}"
    mkdir -p output
    printf 'Running TOPAS espread1 e=%s case=%s file=%s\n' "${e}" "${case_name}" "${parameter_file}"
    "${install_dir}/bin/topas" "${parameter_file}" 2>&1 |
        tee "output/e${e}-espread1-${case_name}_topas.log"
}

if [[ "${energy}" == "all" ]]; then
    for e in 100 150 200 250 300 350 400; do
        run_one "${e}"
    done
else
    case "${energy}" in
        100|150|200|250|300|350|400) run_one "${energy}" ;;
        *) usage ;;
    esac
fi
