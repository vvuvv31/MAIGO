#!/usr/bin/env bash
# Run total IDD TOPAS jobs for 150/250/350 MeV/u on v@192.168.31.5.
# Usage:
#   bash validation/topas/run_mid_energy_idd_remote.sh 150 smoke
#   bash validation/topas/run_mid_energy_idd_remote.sh all development
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/geant4-v11.3.2-install}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"

energy="${1:-}"
case_name="${2:-smoke}"

usage() {
    printf 'Usage: %s ENERGY [smoke|development]\n' "$0" >&2
    printf '  ENERGY: 150 | 250 | 350 | all\n' >&2
    exit 2
}

[[ -n "${energy}" ]] || usage
case "${case_name}" in
    smoke|development) ;;
    *) usage ;;
esac

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${geant4_install}/share/Geant4/data}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_one() {
    local e="$1"
    local parameter_file
    case "${case_name}" in
        smoke) parameter_file="carbon_${e}MeVu_water_smoke.txt" ;;
        development) parameter_file="carbon_${e}MeVu_water_development_remote.txt" ;;
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
    printf 'Running TOPAS e=%s case=%s file=%s\n' "${e}" "${case_name}" "${parameter_file}"
    "${install_dir}/bin/topas" "${parameter_file}" 2>&1 |
        tee "output/e${e}-${case_name}_topas.log"
}

if [[ "${energy}" == "all" ]]; then
    for e in 150 250 350; do
        run_one "${e}"
    done
else
    case "${energy}" in
        150|250|350) run_one "${energy}" ;;
        *) usage ;;
    esac
fi
