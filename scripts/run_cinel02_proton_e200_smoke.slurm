#!/usr/bin/env bash
#SBATCH --job-name=cinp200s
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%j.err
set -euo pipefail
root=/mnt/sda/wuwei/cinel02-hybrid-pilot/proton_e200_smoke; raw=${root}/raw
uuid=00000000-0000-4000-8000-000000899999
template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
mkdir -p "${root}" "${raw}" /mnt/sda/wuwei/cinel02-hybrid-pilot/logs
sed -e 's|@SEED@|2026091999|g' -e "s|@THREADS@|${SLURM_CPUS_PER_TASK}|g" \
    -e 's|@TOTAL_ENERGY@|200|g' -e 's|@HISTORIES@|50000|g' \
    -e 's|@ENERGY_BIN_MIN@|0.0|g' -e 's|@ENERGY_BIN_WIDTH@|1.0|g' \
    -e 's|@ENERGY_BIN_COUNT@|501|g' \
    -e 's|GenericIon(6,12)|proton|g' \
    -e 's|ProjectileZ = 0|ProjectileZ = 1|g' -e 's|ProjectileA = 0|ProjectileA = 1|g' \
    -e 's|IncludeSecondaries = "TRUE"|IncludeSecondaries = "FALSE"|g' \
    "${template}" > "${root}/run.txt"
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw} CARBON_CINEL02_CAMPAIGN_UUID=${uuid}
export CARBON_CINEL02_RUN_TAG=proton_e200_smoke CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=false CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02 CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm CARBON_STEP_LIMITS=default CARBON_RANDOM_SEED_POLICY=2026091999
cd "${root}"
time /home/wuwei/topas/topas-build/topas run.txt > topas.log 2>&1
grep -Fq 'Total number of histories: 50000' topas.log
test -s "${raw}/${uuid}/cinel02.contract.json"
find "${raw}/${uuid}" -name '*.cinel02' -size +63c -print -quit | grep -q .
