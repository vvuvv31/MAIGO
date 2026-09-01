#!/usr/bin/env bash
#SBATCH --job-name=cinhesmoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%A_%a.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%A_%a.err
set -euo pipefail
if [[ ${SLURM_ARRAY_TASK_ID} == 0 ]]; then particle=He3; z=2; a=3; else particle=alpha; z=2; a=4; fi
root=/mnt/sda/wuwei/cinel02-hybrid-pilot/${particle}_e200_smoke; raw=${root}/raw
uuid=$(printf '00000000-0000-4000-8000-%012d' $((930000+SLURM_ARRAY_TASK_ID)))
template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
mkdir -p "${root}" "${raw}"
sed -e "s|@SEED@|$((2026093000+SLURM_ARRAY_TASK_ID))|g" -e "s|@THREADS@|${SLURM_CPUS_PER_TASK}|g" \
 -e "s|@TOTAL_ENERGY@|$((200*a))|g" -e 's|@HISTORIES@|50000|g' -e 's|@ENERGY_BIN_MIN@|0|g' \
 -e 's|@ENERGY_BIN_WIDTH@|1|g' -e 's|@ENERGY_BIN_COUNT@|501|g' -e "s|GenericIon(6,12)|${particle}|g" \
 -e "s|ProjectileZ = 0|ProjectileZ = ${z}|g" -e "s|ProjectileA = 0|ProjectileA = ${a}|g" \
 -e 's|IncludeSecondaries = "TRUE"|IncludeSecondaries = "FALSE"|g' "${template}" > "${root}/run.txt"
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw} CARBON_CINEL02_CAMPAIGN_UUID=${uuid} CARBON_CINEL02_RUN_TAG=${particle}_e200_smoke
export CARBON_CINEL02_PRIMARY_ONLY=true CARBON_CINEL02_OVERWRITE_CAMPAIGN=false CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02 CARBON_PHYSICS_LIST=FTFP_INCLXX CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default CARBON_RANDOM_SEED_POLICY=$((2026093000+SLURM_ARRAY_TASK_ID))
cd "${root}"; time /home/wuwei/topas/topas-build/topas run.txt > topas.log 2>&1
grep -Fq 'Total number of histories: 50000' topas.log
test -s "${raw}/${uuid}/cinel02.contract.json"
find "${raw}/${uuid}" -name '*.cinel02' -size +63c -print -quit | grep -q .
