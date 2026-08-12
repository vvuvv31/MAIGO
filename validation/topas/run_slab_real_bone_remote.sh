#!/usr/bin/env bash
# Usage: run_slab_real_bone_remote.sh [smoke|development]
set -euo pipefail
project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
topas_dir="${project_root}/validation/topas"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon}"
geant4_install="${GEANT4_INSTALL_DIR:-${HOME}/software/geant4-v11.3.2-install}"
gdcm_install="${GDCM_INSTALL_DIR:-${HOME}/software/topas/gdcm-install}"
mode="${1:-development}"

export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${geant4_install}/share/Geant4/data}"
export LD_LIBRARY_PATH="${geant4_install}/lib:${install_dir}/lib:${gdcm_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${topas_dir}"
mkdir -p output
case "$mode" in
  smoke)
    param=carbon_200MeVu_water_slab_real_bone_smoke.txt
    log=output/slab-real-bone-smoke_topas.log
    ;;
  development)
    param=carbon_200MeVu_water_slab_real_bone_development_remote.txt
    log=output/slab-real-bone-development_topas.log
    ;;
  *)
    echo "Usage: $0 [smoke|development]" >&2
    exit 2
    ;;
esac
echo "[$(date -Is)] START slab real bone $mode"
"${install_dir}/bin/topas" "${param}" 2>&1 | tee "${log}"
echo "[$(date -Is)] DONE slab real bone $mode"
