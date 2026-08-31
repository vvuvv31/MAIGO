#!/usr/bin/env bash
# TOPAS/Geant4 charged-ion cascade extraction for the 400 MeV/u validation arm.
# Run only through Slurm on wuwei@127.0.0.1; GPU replay remains local WSL.
#SBATCH --job-name=cinel02c400
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=150
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-cascade-h1o16/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-cascade-h1o16/logs/%x_%j.err

set -euo pipefail

readonly energy_mevu=400
readonly histories=5000000
readonly threads=150
readonly campaign_uuid=00000000-0000-4000-8000-000000000552
readonly root=/mnt/sda/wuwei/cinel02-cascade-h1o16
readonly topas_bin=/home/wuwei/topas/topas-build/topas
readonly template=${root}/tools/cinel02_extraction_h1o16_cascade.txt.in
readonly run_tag=e${energy_mevu}MeVu_${histories}h_bin1_cascade_5001_h1o16_cascade
readonly run_dir=${root}/${run_tag}
readonly raw_root=${run_dir}/raw
readonly config=${run_dir}/run.txt
readonly beam_energy_mev=$((energy_mevu * 12))
readonly seed=$((2026082900 + energy_mevu + histories / 100000))

test -x "${topas_bin}"
test -f "${template}"
mkdir -p "${run_dir}" "${raw_root}" "${root}/logs"
sed \
    -e "s|@SEED@|${seed}|g" \
    -e "s|@THREADS@|${threads}|g" \
    -e "s|@TOTAL_ENERGY@|${beam_energy_mev}|g" \
    -e "s|@HISTORIES@|${histories}|g" \
    -e "s|@ENERGY_BIN_MIN@|0.0|g" \
    -e "s|@ENERGY_BIN_WIDTH@|1.0|g" \
    -e "s|@ENERGY_BIN_COUNT@|5001|g" \
    "${template}" > "${config}"

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
time "${topas_bin}" "${config}" > "${run_dir}/topas.log" 2>&1
test -s "${raw_root}/${campaign_uuid}/cinel02.contract.json"
find "${raw_root}/${campaign_uuid}" -maxdepth 1 -type f -name '*.cinel02' -size +63c -print -quit | grep -q .
grep -Fq "Total number of histories: ${histories}" "${run_dir}/topas.log"
