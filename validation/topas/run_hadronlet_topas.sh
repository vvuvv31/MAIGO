#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
topas_install="${TOPAS_HADRONLET_INSTALL:-${repo_root}/build/opentopas-hadronlet-install}"
geant4_install="${GEANT4_INSTALL:-/home/v/Applications/GEANT4/geant4-install}"
topas_data="${TOPAS_G4_DATA_DIR:-/home/v/Applications/GEANT4/G4DATA}"
parameter_file="${1:-carbon_200MeVu_water_letd.txt}"

export TOPAS_G4_DATA_DIR="${topas_data}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${topas_install}/Frameworks"
export LD_LIBRARY_PATH="${topas_install}/lib:${geant4_install}/lib:${LD_LIBRARY_PATH:-}"

cd "${repo_root}/validation/topas"
mkdir -p output
exec "${topas_install}/bin/topas" "${parameter_file}"
