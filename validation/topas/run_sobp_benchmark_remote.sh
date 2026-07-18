#!/usr/bin/env bash
# Run the 10M-history, 56-thread SOBP benchmark on the TOPAS host.
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/software/gate/G4DATA}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${topas_dir}"
mkdir -p output
"${install_dir}/bin/topas" carbon_sobp_water_3cm_5_10cm_3mm_development_remote.txt 2>&1 |
    tee output/sobp_water_3cm_5_10cm_3mm_remote.log
