#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
if [[ -n "${TOPAS_EXECUTABLE:-}" ]]; then
    topas_executable="${TOPAS_EXECUTABLE}"
elif [[ -x "${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon/bin/topas" ]]; then
    topas_executable="${HOME}/software/topas/OpenTOPAS-install-v4.2.3-carbon/bin/topas"
elif [[ -x "${project_root}/build/opentopas-material-extension-install/bin/topas" ]]; then
    topas_executable="${project_root}/build/opentopas-material-extension-install/bin/topas"
else
    printf 'TOPAS 4.2.p3 executable not found; refusing PATH fallback\n' >&2
    exit 2
fi
case_name="${1:-development}"

# Make the locally built extension executable directly runnable in WSL.
extension_install="$(cd -- "$(dirname -- "${topas_executable}")/.." 2>/dev/null && pwd || true)"
if [[ "${topas_executable}" == "${project_root}"/build/*/bin/topas ]]; then
    export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/software/geant4-v11.3.2-install/share/Geant4/data}"
    geant4_lib="${HOME}/software/geant4-v11.3.2-install/lib"
    gdcm_lib="${HOME}/software/topas/gdcm-install/lib"
    export LD_LIBRARY_PATH="${geant4_lib}:${extension_install}/lib:${gdcm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ "${topas_executable}" == "${HOME}/software/topas/"* ]]; then
    geant4_install="${HOME}/software/geant4-v11.3.2-install"
else
    geant4_install="${HOME}/Applications/GEANT4/geant4-install"
fi
geant4_version_file="${geant4_install}/lib/cmake/Geant4/Geant4ConfigVersion.cmake"
if [[ ! -f "${geant4_version_file}" ]] ||
   ! grep -Eq 'set\(PACKAGE_VERSION "11\.3\.2"\)' "${geant4_version_file}"; then
    printf 'Geant4 11.3.2 is required; refusing older or unknown installation\n' >&2
    exit 2
fi
if [[ "${topas_executable}" == "${HOME}/software/topas/"* ]]; then
    export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${geant4_install}/share/Geant4/data}"
    export LD_LIBRARY_PATH="${geant4_install}/lib:${extension_install}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

case "${case_name}" in
    smoke) parameter_file="carbon_200MeVu_water_smoke.txt" ;;
    development) parameter_file="carbon_200MeVu_water_development.txt" ;;
    reference) parameter_file="carbon_200MeVu_water.txt" ;;
    em-smoke) parameter_file="carbon_200MeVu_water_em_only_smoke.txt" ;;
    em-development) parameter_file="carbon_200MeVu_water_em_only_development.txt" ;;
    species-smoke) parameter_file="carbon_200MeVu_water_species_smoke.txt" ;;
    species-development) parameter_file="carbon_200MeVu_water_species_development.txt" ;;
    species-reference) parameter_file="carbon_200MeVu_water_species.txt" ;;
    ancestor-smoke) parameter_file="carbon_200MeVu_water_ancestor_attribution_smoke.txt" ;;
    ancestor-development) parameter_file="carbon_200MeVu_water_ancestor_attribution_development.txt" ;;
    ancestor-reference) parameter_file="carbon_200MeVu_water_ancestor_attribution.txt" ;;
    fragment-smoke) parameter_file="carbon_200MeVu_water_fragment_production_smoke.txt" ;;
    fragment-development) parameter_file="carbon_200MeVu_water_fragment_production_development.txt" ;;
    fragment-reference) parameter_file="carbon_200MeVu_water_fragment_production.txt" ;;
    cross-sections) parameter_file="carbon_200MeVu_water_cross_sections.txt" ;;
    stopping-power) parameter_file="carbon_200MeVu_water_stopping_power.txt" ;;
    delta-electron) parameter_file="carbon_200MeVu_water_delta_electron.txt" ;;
    delta-electron-smoke) parameter_file="carbon_200MeVu_water_delta_electron_smoke.txt" ;;
    delta-electron-1k) parameter_file="carbon_200MeVu_water_delta_electron_1k.txt" ;;
    cascade-e400-g4-11-3-2-100k)
        parameter_file="carbon_400MeVu_water_cascade_reactions_g4_11_3_2_100k.txt"
        ;;
    *)
        printf 'Usage: %s [smoke|development|reference|em-smoke|em-development|species-smoke|species-development|species-reference|ancestor-smoke|ancestor-development|ancestor-reference|fragment-smoke|fragment-development|fragment-reference|cross-sections|stopping-power|delta-electron|delta-electron-smoke|delta-electron-1k|cascade-e400-g4-11-3-2-100k]\n' "$0" >&2
        exit 2
        ;;
esac

cd "${script_dir}"
mkdir -p output
"${topas_executable}" "${parameter_file}" 2>&1 | tee "output/${case_name}_topas.log"
