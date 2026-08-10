#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
topas_install="${TOPAS_MINIBEAM_INSTALL:-/home/v/Applications/TOPAS/OpenTOPAS-install}"
geant4_install="${GEANT4_INSTALL:-/home/v/Applications/GEANT4/geant4-install}"
topas_data="${TOPAS_G4_DATA_DIR:-/home/v/Applications/GEANT4/G4DATA}"
parameter_file="${1:-run_phase_space_center.txt}"

if [[ "${parameter_file}" = /* ]]; then
    parameter_path="${parameter_file}"
else
    parameter_path="${repo_root}/ct/minibeam/${parameter_file}"
fi
if [[ ! -f "${parameter_path}" ]]; then
    echo "Missing TOPAS parameter file: ${parameter_path}" >&2
    exit 2
fi
if [[ ! -x "${topas_install}/bin/topas" ]]; then
    echo "Missing TOPAS executable: ${topas_install}/bin/topas" >&2
    exit 2
fi

export TOPAS_G4_DATA_DIR="${topas_data}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${topas_install}/Frameworks"
export LD_LIBRARY_PATH="${topas_install}/lib:${geant4_install}/lib:${LD_LIBRARY_PATH:-}"

case_dir="${repo_root}/ct/minibeam"
mkdir -p "${case_dir}/output"
stem="$(basename "${parameter_path}" .txt)"
log="${case_dir}/output/${stem}.log"

cd "${case_dir}"
echo "TOPAS executable: ${topas_install}/bin/topas"
echo "Parameter file: ${parameter_path}"
echo "Log: ${log}"
"${topas_install}/bin/topas" "${parameter_path}" 2>&1 | tee "${log}"
