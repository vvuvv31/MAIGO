#!/usr/bin/env bash
#SBATCH --job-name=cinc12prod
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/mnt/sda/wuwei/cinel02-c12-production-5m-v2/logs/%A_%a.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-c12-production-5m-v2/logs/%A_%a.err

set -euo pipefail

readonly species=("6 12")
readonly energies=(10 25 50 100 200 400)
readonly energy_count=${#energies[@]}
readonly species_index=$((SLURM_ARRAY_TASK_ID / energy_count))
readonly energy_index=$((SLURM_ARRAY_TASK_ID % energy_count))
read -r projectile_z projectile_a <<< "${species[species_index]}"
readonly energy_mevu=${energies[energy_index]}
readonly histories=5000000
readonly threads=${SLURM_CPUS_PER_TASK}
readonly total_energy_mev=$((energy_mevu * projectile_a))
readonly campaign_uuid=$(printf '00000000-0000-4000-8000-%012d' $((810000 + SLURM_ARRAY_TASK_ID)))
readonly root=/mnt/sda/wuwei/cinel02-c12-production-5m-v2
readonly run_tag=z${projectile_z}a${projectile_a}_e${energy_mevu}_h${histories}
readonly run_dir=${root}/${run_tag}
readonly raw_root=${run_dir}/raw
readonly config=${run_dir}/run.txt
readonly template=/mnt/sda/wuwei/cinel02-cascade-h1o16/tools/cinel02_extraction_h1o16_cascade.txt.in
readonly topas_bin=/home/wuwei/topas/topas-build/topas
readonly seed=$((2026096100 + SLURM_ARRAY_TASK_ID))

test -x "${topas_bin}"
test -f "${template}"
mkdir -p "${run_dir}" "${raw_root}" "${root}/logs"
sed \
    -e "s|@SEED@|${seed}|g" \
    -e "s|@THREADS@|${threads}|g" \
    -e "s|@TOTAL_ENERGY@|${total_energy_mev}|g" \
    -e "s|@HISTORIES@|${histories}|g" \
    -e "s|@ENERGY_BIN_MIN@|0.0|g" \
    -e "s|@ENERGY_BIN_WIDTH@|1.0|g" \
    -e "s|@ENERGY_BIN_COUNT@|501|g" \
    -e "s|GenericIon(6,12)|GenericIon(${projectile_z},${projectile_a})|g" \
    -e "s|i:Sc/CarbonInelasticExposure/ProjectileZ = 0|i:Sc/CarbonInelasticExposure/ProjectileZ = ${projectile_z}|g" \
    -e "s|i:Sc/CarbonInelasticExposure/ProjectileA = 0|i:Sc/CarbonInelasticExposure/ProjectileA = ${projectile_a}|g" \
    -e 's|b:Sc/CarbonInelasticExposure/IncludeSecondaries = "TRUE"|b:Sc/CarbonInelasticExposure/IncludeSecondaries = "FALSE"|g' \
    "${template}" > "${config}"

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw_root}
export CARBON_CINEL02_CAMPAIGN_UUID=${campaign_uuid}
export CARBON_CINEL02_RUN_TAG=${run_tag}
export CARBON_CINEL02_PRIMARY_ONLY=true
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
