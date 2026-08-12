#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
topas_install="${TOPAS_MINIBEAM_INSTALL:-${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon}"
geant4_install="${GEANT4_INSTALL:-${HOME}/software/geant4-v11.3.2-install}"
topas_data="${TOPAS_G4_DATA_DIR:-${HOME}/software/geant4-v11.3.2-install/share/Geant4/data}"
parameter_file="${1:-run_phase_space_center.txt}"

geant4_version_file="${geant4_install}/lib/cmake/Geant4/Geant4ConfigVersion.cmake"
if [[ ! -f "${geant4_version_file}" ]] ||
   ! grep -Eq 'set\(PACKAGE_VERSION "11\.3\.2"\)' "${geant4_version_file}"; then
    echo "Geant4 11.3.2 is required; refusing older or unknown installation" >&2
    exit 2
fi

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
