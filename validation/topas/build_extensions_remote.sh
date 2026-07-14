#!/usr/bin/env bash
set -euo pipefail

project_root="${REMOTE_PROJECT_ROOT:-${HOME}/gpu}"
source_dir="${OPENTOPAS_SOURCE_DIR:-${HOME}/software/topas/OpenTOPAS}"
build_dir="${OPENTOPAS_BUILD_DIR:-${project_root}/build/opentopas-extension-build}"
install_dir="${OPENTOPAS_EXTENSION_INSTALL_DIR:-${project_root}/build/opentopas-extension-install}"
geant4_dir="${Geant4_DIR:-${HOME}/software/gate/GATE/geant4-v11.1.3-install-MT/lib/cmake/Geant4}"
gdcm_dir="${GDCM_DIR:-${HOME}/software/topas/gdcm-install/lib/gdcm-2.6}"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DGeant4_DIR="${geant4_dir}" \
    -DGDCM_DIR="${gdcm_dir}" \
    -DTOPAS_EXTENSIONS_DIR="${project_root}/validation/topas/extensions"

cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-56}"
cmake --install "${build_dir}"

printf 'Remote extension-enabled TOPAS: %s/bin/topas\n' "${install_dir}"
