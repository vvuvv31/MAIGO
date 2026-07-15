#!/usr/bin/env bash
# Run BiGaussian emittance 3D-dose jobs for selected energies.
# Usage: bash run_emittance_multi_remote.sh [150|200|250|350|400|all]
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"
energy="${1:-all}"

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/software/gate/G4DATA}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_one() {
    local e="$1"
    local parameter_file="carbon_${e}MeVu_water_emittance_development_remote.txt"
    cd "${topas_dir}"
    mkdir -p output
    printf 'Running TOPAS emittance e=%s file=%s\n' "${e}" "${parameter_file}"
    "${install_dir}/bin/topas" "${parameter_file}" 2>&1 |
        tee "output/emittance-e${e}-development_topas.log"
}

if [[ "${energy}" == "all" ]]; then
    for e in 150 200 250 350 400; do
        run_one "${e}"
    done
else
    case "${energy}" in
        150|200|250|350|400) run_one "${energy}" ;;
        *) printf 'Usage: %s [150|200|250|350|400|all]\n' "$0" >&2; exit 2 ;;
    esac
fi
