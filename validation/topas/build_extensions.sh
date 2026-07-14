#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${OPENTOPAS_SOURCE_DIR:-${project_root}/build/opentopas-source}"
build_dir="${OPENTOPAS_BUILD_DIR:-${project_root}/build/opentopas-extension-build}"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_dir="${Geant4_DIR:-${HOME}/Applications/GEANT4/geant4-install/lib/cmake/Geant4}"
gdcm_dir="${GDCM_DIR:-${HOME}/Applications/TOPAS/OpenTOPAS/gdcm-install/lib/gdcm-2.6}"
tag="${OPENTOPAS_TAG:-v4.2.3}"

if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
    git clone --depth 1 --branch "${tag}" https://github.com/OpenTOPAS/OpenTOPAS.git "${source_dir}"
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DGeant4_DIR="${geant4_dir}" \
    -DGDCM_DIR="${gdcm_dir}" \
    -DTOPAS_EXTENSIONS_DIR="${project_root}/validation/topas/extensions" \
    -DTOPAS_USE_QT=ON \
    -DTOPAS_USE_QT6=ON

cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-10}"
cmake --install "${build_dir}"

printf 'Extension-enabled TOPAS: %s/bin/topas\n' "${install_dir}"
