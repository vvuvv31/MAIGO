#!/usr/bin/env bash
# Run 200 MeV/u BiGaussian emittance 3D-dose job on remote TOPAS host.
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/geant4-v11.3.2-install}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"
case_name="${1:-smoke}"

case "${case_name}" in
    smoke) parameter_file="carbon_200MeVu_water_emittance_smoke.txt" ;;
    development) parameter_file="carbon_200MeVu_water_emittance_development_remote.txt" ;;
    *) printf 'Usage: %s [smoke|development]\n' "$0" >&2; exit 2 ;;
esac

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${geant4_install}/share/Geant4/data}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${topas_dir}"
mkdir -p output
printf 'Running TOPAS emittance e200 case=%s file=%s\n' "${case_name}" "${parameter_file}"
"${install_dir}/bin/topas" "${parameter_file}" 2>&1 |
    tee "output/emittance-e200-${case_name}_topas.log"
