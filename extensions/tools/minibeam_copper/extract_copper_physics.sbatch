#!/bin/bash
#SBATCH --job-name=maigo-cu-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/sda/wuwei/minibeam_copper_extract_%j.log
set -euo pipefail
repo=/mnt/sdb/wuwei/MAIGO_pristine
run_root=/mnt/sda/wuwei/minibeam_copper_extract_${SLURM_JOB_ID}
campaign=20260917-0000-4000-8000-000000000030
mkdir -p "${run_root}"
cp "${repo}/benchmark/carbonminibeam/extract_copper_physics.txt" "${run_root}/run.txt"
export CARBON_CINEL02_OUTPUT_DIR="${run_root}/raw"
export CARBON_CINEL02_CAMPAIGN_UUID="${campaign}"
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
export CARBON_CINEL02_RUN_TAG=c12_cu_inclxx
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=11.3.2
export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
export CARBON_PRODUCTION_CUTS='all=0.05mm'
export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
export CARBON_RANDOM_SEED_POLICY='Ts/Seed=240917;192 threads'
cd "${run_root}"
/home/wuwei/topas/topas-build/topas run.txt
