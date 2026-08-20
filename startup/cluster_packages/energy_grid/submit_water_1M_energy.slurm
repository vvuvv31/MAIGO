#!/bin/bash
#SBATCH --job-name=c12egrid
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH -n 56
#SBATCH --time=2-00:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm_%x_%A_%a.out
#SBATCH --error=slurm_%x_%A_%a.err

# Dedicated 1M C-12 water cascade at 100,150,200,250,300,350,400 MeV/u.
# Override one energy with: sbatch --export=ALL,ENERGY_MEVU=200 --array=0 submit_water_1M_energy.slurm
set -euo pipefail
unset MODULEPATH || true

ENERGIES=(100 150 200 250 300 350 400)
ENERGY_MEVU="${ENERGY_MEVU:-${ENERGIES[${SLURM_ARRAY_TASK_ID:-0}]}}"
if ! [[ "${ENERGY_MEVU}" =~ ^[0-9]+$ ]]; then
    echo "ENERGY_MEVU must be an integer MeV/u, got ${ENERGY_MEVU}" >&2
    exit 2
fi
BEAM_ENERGY_MEV=$((ENERGY_MEVU * 12))
SEED=$((2026090000 + ENERGY_MEVU))
# 4 MeV/u bins covering [0, ENERGY]; floor(E/4)+1 holds E itself.
BIN_COUNT=$((ENERGY_MEVU / 4 + 1))
TAG="topas_${ENERGY_MEVU}MeVu_water_inclxx_1M"
STEM="${TAG}_cascade"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Job is submitted from energy_grid/; TOPAS lives in the parent 1M tree.
PKG_ROOT="$(cd "${ROOT}/.." && pwd)"
SHARE=/ShareData1/opt/cascadelake/linux-centos7-cascadelake/gcc-9.4.0
export PATH="${SHARE}/cmake-3.25.2-hn3pewmkxtlqnynxccmcy6mt3m3rcsa4/bin:${SHARE}/python-3.10.8-tzew3vzrsgbupmdbywnpblqeeygegvjc/bin:${PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${HOME}/Applications/TOPAS/OpenTOPAS-install/Frameworks"
export TOPAS_G4_DATA_DIR="${HOME}/Applications/GEANT4/G4DATA"
export LD_LIBRARY_PATH="${PKG_ROOT}/topas-install/lib:${HOME}/Applications/TOPAS/OpenTOPAS-install/lib:${HOME}/Applications/GEANT4/geant4-install/lib64:${LD_LIBRARY_PATH:-}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} energy=${ENERGY_MEVU} MeV/u cwd=${ROOT}"
cd "${ROOT}"

TOPAS_BIN="${PKG_ROOT}/topas-install/bin/topas"
if [[ ! -x "${TOPAS_BIN}" ]]; then
    echo "ERROR: missing ${TOPAS_BIN}" >&2
    exit 2
fi
if ! grep -aob CarbonCascadeNtuple "${TOPAS_BIN}" >/dev/null; then
    echo "ERROR: ${TOPAS_BIN} does not contain CarbonCascadeNtuple" >&2
    exit 2
fi

WORKDIR="${ROOT}/run_${ENERGY_MEVU}MeVu_1M"
mkdir -p "${WORKDIR}"
sed \
    -e "s/__ENERGY_MEVU__/${ENERGY_MEVU}/g" \
    -e "s/__BEAM_ENERGY_MEV__/${BEAM_ENERGY_MEV}/g" \
    -e "s/__SEED__/${SEED}/g" \
    -e "s/__STEM__/${STEM}/g" \
    "${ROOT}/run_water_1M_energy.txt.in" > "${WORKDIR}/run.txt"

echo "[$(date -Is)] TOPAS ${ENERGY_MEVU} MeV/u 1M start"
(
    cd "${WORKDIR}"
    time -p "${TOPAS_BIN}" run.txt
) > "${WORKDIR}/topas.log" 2>&1
echo "[$(date -Is)] TOPAS ${ENERGY_MEVU} MeV/u 1M done"
[[ -s "${WORKDIR}/${STEM}.header" ]]
[[ -s "${WORKDIR}/${STEM}.phsp" ]]
grep -Fq 'Number of Original Histories: 1000000' "${WORKDIR}/${STEM}.header"

TOOLS="${ROOT}/package_tools"
if [[ ! -d "${TOOLS}" ]]; then
    TOOLS="${PKG_ROOT}/package_tools"
fi
PKG="${ROOT}/packages"
mkdir -p "${PKG}"

python3 "${TOOLS}/prepare_topas_cascade.py" \
    --case "${TAG}" --histories 1000000 --input-dir "${WORKDIR}" \
    --stem "${STEM}" --log-name topas.log \
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
python3 "${TOOLS}/summarize_primary_bin_yields.py" \
    --reactions "${WORKDIR}/${TAG}_primary_reactions.csv.gz" \
    --secondaries "${WORKDIR}/${TAG}_primary_secondaries.csv.gz" \
    --source-energy-mevu "${ENERGY_MEVU}" \
    --energy-bin-count "${BIN_COUNT}" \
    --output "${PKG}/${TAG}_primary_bin_yields.json"
python3 "${TOOLS}/compile_reaction_package.py" \
    --metadata "${PKG}/${TAG}_primary.metadata.json" \
    --reactions "${WORKDIR}/${TAG}_primary_reactions.csv.gz" \
    --secondaries "${WORKDIR}/${TAG}_primary_secondaries.csv.gz" \
    --output "${PKG}/${TAG}_primary_3d.bin" \
    --output-metadata "${PKG}/${TAG}_primary_3d.compiled.json" \
    --energy-bin-min-mevu 0 --energy-bin-width-mevu 4 \
    --energy-bin-count "${BIN_COUNT}" \
    --fill-empty nearest \
    --material G4_WATER --physics-model "INCLXX"

echo "[$(date -Is)] packages written for ${ENERGY_MEVU} MeV/u"
ls -lh "${PKG}/${TAG}_primary_3d.bin" "${PKG}/${TAG}_cascade_3d.bin" \
    "${PKG}/${TAG}_primary_bin_yields.json"
