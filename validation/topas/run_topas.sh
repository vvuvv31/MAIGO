#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
topas_executable="${TOPAS_EXECUTABLE:-topas}"
case_name="${1:-development}"

# Make the locally built extension executable directly runnable in WSL.
extension_install="${project_root}/build/opentopas-extension-install"
if [[ "${topas_executable}" == "${extension_install}/bin/topas" ]]; then
    export TOPAS_G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-${HOME}/Applications/GEANT4/G4DATA}"
    geant4_lib="${HOME}/Applications/GEANT4/geant4-install/lib"
    gdcm_lib="${HOME}/Applications/TOPAS/OpenTOPAS/gdcm-install/lib"
    export LD_LIBRARY_PATH="${geant4_lib}:${extension_install}/lib:${gdcm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
    *)
        printf 'Usage: %s [smoke|development|reference|em-smoke|em-development|species-smoke|species-development|species-reference|ancestor-smoke|ancestor-development|ancestor-reference|fragment-smoke|fragment-development|fragment-reference|cross-sections]\n' "$0" >&2
        exit 2
        ;;
esac

cd "${script_dir}"
mkdir -p output
"${topas_executable}" "${parameter_file}" 2>&1 | tee "output/${case_name}_topas.log"
