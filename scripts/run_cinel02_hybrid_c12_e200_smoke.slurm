#!/usr/bin/env bash
#SBATCH --job-name=cinhybs200
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/%x_%j.err
set -euo pipefail
readonly z=6 a=12 energy=200 histories=50000 uuid=00000000-0000-4000-8000-000000799999
readonly root=/mnt/sda/wuwei/cinel02-hybrid-pilot/z6a12_e200_smoke
readonly raw=${root}/raw template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
mkdir -p "${root}" "${raw}" /mnt/sda/wuwei/cinel02-hybrid-pilot/logs
sed -e 's|@SEED@|2026090999|g' -e "s|@THREADS@|${SLURM_CPUS_PER_TASK}|g" \
    -e 's|@TOTAL_ENERGY@|2400|g' -e "s|@HISTORIES@|${histories}|g" \
    -e 's|@ENERGY_BIN_MIN@|0.0|g' -e 's|@ENERGY_BIN_WIDTH@|1.0|g' \
    -e 's|@ENERGY_BIN_COUNT@|501|g' \
    -e 's|i:Sc/CarbonInelasticExposure/ProjectileZ = 0|i:Sc/CarbonInelasticExposure/ProjectileZ = 6|g' \
    -e 's|i:Sc/CarbonInelasticExposure/ProjectileA = 0|i:Sc/CarbonInelasticExposure/ProjectileA = 12|g' \
    -e 's|b:Sc/CarbonInelasticExposure/IncludeSecondaries = "TRUE"|b:Sc/CarbonInelasticExposure/IncludeSecondaries = "FALSE"|g' \
    "${template}" > "${root}/run.txt"
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw} CARBON_CINEL02_CAMPAIGN_UUID=${uuid}
export CARBON_CINEL02_RUN_TAG=z6a12_e200_smoke CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=false CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02 CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm CARBON_STEP_LIMITS=default CARBON_RANDOM_SEED_POLICY=2026090999
cd "${root}"
time /home/wuwei/topas/topas-build/topas run.txt > topas.log 2>&1
test -s "${raw}/${uuid}/cinel02.contract.json"
find "${raw}/${uuid}" -name '*.cinel02' -size +63c -print -quit | grep -q .
grep -Fq "Total number of histories: ${histories}" topas.log
