#!/usr/bin/env bash
# Run 400 MeV/u cascade reaction n-tuple on v@192.168.31.5
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"
case_name="${1:-smoke}"

case "${case_name}" in
    smoke) parameter_file="carbon_400MeVu_water_cascade_reactions_smoke.txt" ;;
    development) parameter_file="carbon_400MeVu_water_cascade_reactions_development_remote.txt" ;;
    *) printf 'Usage: %s [smoke|development]\n' "$0" >&2; exit 2 ;;
esac

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/software/gate/G4DATA}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${topas_dir}"
mkdir -p output
printf 'Running TOPAS cascade e400 case=%s file=%s\n' "${case_name}" "${parameter_file}"
"${install_dir}/bin/topas" "${parameter_file}" 2>&1 |
    tee "output/cascade-e400-${case_name}_topas.log"
