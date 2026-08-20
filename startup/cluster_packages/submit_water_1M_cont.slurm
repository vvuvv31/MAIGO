#!/bin/bash
#SBATCH --job-name=c12pkg1Mc
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH -n 56
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail
unset MODULEPATH || true

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARE=/ShareData1/opt/cascadelake/linux-centos7-cascadelake/gcc-9.4.0
export CC="${SHARE}/gcc-11.3.0-hlrrmo2m5ycqa54vsmbhcpe64jmp5dwm/bin/gcc"
export CXX="${SHARE}/gcc-11.3.0-hlrrmo2m5ycqa54vsmbhcpe64jmp5dwm/bin/g++"
export PATH="${SHARE}/cmake-3.25.2-hn3pewmkxtlqnynxccmcy6mt3m3rcsa4/bin:${SHARE}/python-3.10.8-tzew3vzrsgbupmdbywnpblqeeygegvjc/bin:${PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${HOME}/Applications/TOPAS/OpenTOPAS-install/Frameworks"
export TOPAS_G4_DATA_DIR="${HOME}/Applications/GEANT4/G4DATA"
export Qt5Core_DIR=/usr/lib64/cmake/Qt5Core
export Qt5Gui_DIR=/usr/lib64/cmake/Qt5Gui
export Qt5Widgets_DIR=/usr/lib64/cmake/Qt5Widgets
export Qt5OpenGL_DIR=/usr/lib64/cmake/Qt5OpenGL
export CMAKE_PREFIX_PATH=/usr/lib64/cmake
export LD_LIBRARY_PATH="${ROOT}/topas-install/lib:${HOME}/Applications/TOPAS/OpenTOPAS-install/lib:${HOME}/Applications/GEANT4/geant4-install/lib64:${LD_LIBRARY_PATH:-}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} cwd=${ROOT}"
cd "${ROOT}"

TOPAS_BIN="${ROOT}/topas-install/bin/topas"
if [[ ! -x "${TOPAS_BIN}" ]]; then
    echo "ERROR: compile carbon TOPAS on the login node first" >&2
    exit 2
fi
if ! grep -aob CarbonCascadeNtuple "${TOPAS_BIN}" >/dev/null; then
    echo "ERROR: ${TOPAS_BIN} does not contain CarbonCascadeNtuple" >&2
    exit 2
fi
if ! grep -aob projectile_continuation "${TOPAS_BIN}" >/dev/null; then
    echo "ERROR: ${TOPAS_BIN} is the old ntuple (no projectile_continuation)" >&2
    exit 2
fi

WORKDIR="${ROOT}/run_water_1M_cont"
mkdir -p "${WORKDIR}"
cp -f "${ROOT}/run_water_1M_cascade.txt" "${WORKDIR}/run.txt"
echo "[$(date -Is)] TOPAS 1M cascade (continuation ntuple) start"
(
    cd "${WORKDIR}"
    time -p "${TOPAS_BIN}" run.txt
) > "${WORKDIR}/topas.log" 2>&1
echo "[$(date -Is)] TOPAS 1M cascade done"
[[ -s "${WORKDIR}/topas_400MeVu_water_inclxx_1M_cascade.header" ]]
[[ -s "${WORKDIR}/topas_400MeVu_water_inclxx_1M_cascade.phsp" ]]
grep -Fq 'Number of Original Histories: 1000000' \
    "${WORKDIR}/topas_400MeVu_water_inclxx_1M_cascade.header"

TOOLS="${ROOT}/package_tools"
PKG="${ROOT}/packages"
mkdir -p "${PKG}"
TAG=topas_400MeVu_water_inclxx_1M_cont
python3 "${TOOLS}/prepare_topas_cascade.py" \
    --case "${TAG}" --histories 1000000 --input-dir "${WORKDIR}" \
    --stem topas_400MeVu_water_inclxx_1M_cascade --log-name topas.log \
    --phantom-half-length-mm 1000 \
    --interactions-output "${WORKDIR}/${TAG}_interactions.csv.gz" \
    --products-output "${WORKDIR}/${TAG}_products.csv.gz" \
    --metadata "${PKG}/${TAG}.metadata.json"
python3 "${TOOLS}/compile_cascade_package.py" \
    --metadata "${PKG}/${TAG}.metadata.json" \
    --interactions "${WORKDIR}/${TAG}_interactions.csv.gz" \
    --products "${WORKDIR}/${TAG}_products.csv.gz" \
    --output "${PKG}/${TAG}_cascade_3d.bin" \
    --output-metadata "${PKG}/${TAG}_cascade_3d.compiled.json" \
    --material G4_WATER --physics-model "INCLXX" \
    --source-projectile-z 6 --source-projectile-a 12
python3 "${TOOLS}/prepare_primary_reactions_from_cascade.py" \
    --cascade-metadata "${PKG}/${TAG}.metadata.json" \
    --interactions "${WORKDIR}/${TAG}_interactions.csv.gz" \
    --products "${WORKDIR}/${TAG}_products.csv.gz" \
    --reactions-output "${WORKDIR}/${TAG}_primary_reactions.csv.gz" \
    --secondaries-output "${WORKDIR}/${TAG}_primary_secondaries.csv.gz" \
    --metadata-output "${PKG}/${TAG}_primary.metadata.json" \
    --projectile-z 6 --projectile-a 12
python3 "${TOOLS}/compile_reaction_package.py" \
    --metadata "${PKG}/${TAG}_primary.metadata.json" \
    --reactions "${WORKDIR}/${TAG}_primary_reactions.csv.gz" \
    --secondaries "${WORKDIR}/${TAG}_primary_secondaries.csv.gz" \
    --output "${PKG}/${TAG}_primary_3d.bin" \
    --output-metadata "${PKG}/${TAG}_primary_3d.compiled.json" \
    --energy-bin-min-mevu 0 --energy-bin-width-mevu 4 --energy-bin-count 101 \
    --material G4_WATER --physics-model "INCLXX"
echo "[$(date -Is)] packages written under ${PKG}"
ls -lh "${PKG}/${TAG}_primary_3d.bin" "${PKG}/${TAG}_cascade_3d.bin"
