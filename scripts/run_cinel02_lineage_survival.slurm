#!/usr/bin/env bash
# TOPAS full-cascade isotope lineage/survival reference smoke.
# This is a local CPU TOPAS job; GPU replay remains on the local RTX 2080Ti.
#SBATCH --job-name=cinel02lineage
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-lineage-survival/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-lineage-survival/logs/%x_%j.err

set -euo pipefail

readonly energy_mevu=200
readonly histories=100000
readonly threads=${SLURM_CPUS_PER_TASK:-60}
readonly total_energy_mev=$((energy_mevu * 12))
readonly seed=2026095100
readonly campaign_uuid=00000000-0000-4000-8000-000000960221
readonly root=/mnt/sda/wuwei/cinel02-lineage-survival
readonly run_tag=e${energy_mevu}MeVu_${histories}h_lineage_survival_v2
readonly run_dir=${root}/${run_tag}
readonly raw_root=${run_dir}/raw
readonly source_template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
readonly config=${run_dir}/run.txt
readonly topas_bin=/home/wuwei/topas/topas-build/topas

test -x "${topas_bin}"
test -f "${source_template}"
mkdir -p "${run_dir}" "${raw_root}" "${root}/logs"

sed \
    -e "s|@SEED@|${seed}|g" \
    -e "s|@THREADS@|${threads}|g" \
    -e "s|@TOTAL_ENERGY@|${total_energy_mev}|g" \
    -e "s|@HISTORIES@|${histories}|g" \
    -e "s|@ENERGY_BIN_MIN@|0.0|g" \
    -e "s|@ENERGY_BIN_WIDTH@|1.0|g" \
    -e "s|@ENERGY_BIN_COUNT@|5001|g" \
    "${source_template}" > "${config}"

cat >> "${config}" <<TOPAS_LINEAGE_SCORER

# One row per Li/Be transport episode.  This scorer is diagnostic-only and is
# intentionally separate from CarbonCascadeNtuple package extraction.
s:Sc/LineageSurvival/Quantity = "CarbonLineageSurvivalNtuple"
s:Sc/LineageSurvival/Component = "Phantom"
s:Sc/LineageSurvival/OutputType = "ASCII"
s:Sc/LineageSurvival/OutputFile = "lineage_survival"
s:Sc/LineageSurvival/IfOutputFileAlreadyExists = "Overwrite"
TOPAS_LINEAGE_SCORER

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw_root}
export CARBON_CINEL02_CAMPAIGN_UUID=${campaign_uuid}
export CARBON_CINEL02_RUN_TAG=${run_tag}
export CARBON_CINEL02_PRIMARY_ONLY=false
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=false
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02
export CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default
export CARBON_RANDOM_SEED_POLICY=${seed}

cd "${run_dir}"
time "${topas_bin}" "${config}" > topas.log 2>&1

test -s lineage_survival.header
test -s lineage_survival.phsp
test -s "${raw_root}/${campaign_uuid}/cinel02.contract.json"
grep -Fq "Total number of histories: ${histories}" topas.log
