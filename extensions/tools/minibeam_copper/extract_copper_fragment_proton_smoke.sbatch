#!/bin/bash
#SBATCH --job-name=maigo-cu-p-bic
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail
repo=/mnt/sdb/wuwei/MAIGO_pristine
run_root=/mnt/sda/wuwei/minibeam_copper_fragment_proton_bic_smoke
campaign=20260918-0001-4000-8000-000000000041

mkdir -p "${run_root}"
cp "${repo}/benchmark/carbonminibeam/extract_copper_fragment_cascade.txt" \
   "${run_root}/run.txt"

export CARBON_CINEL02_OUTPUT_DIR="${run_root}/raw"
export CARBON_CINEL02_CAMPAIGN_UUID="${campaign}"
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
export CARBON_CINEL02_RUN_TAG=proton_cu_binary_700MeV_smoke
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=11.3.2
export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
export CARBON_PRODUCTION_CUTS='all=0.05mm'
export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
export CARBON_RANDOM_SEED_POLICY='Ts/Seed=240918;32 threads;proton=700MeV'

cd "${run_root}"
/home/wuwei/topas/topas-build/topas run.txt

mkdir -p "${run_root}/compiled"
python3 "${repo}/benchmark/carbonminibeam/compile_copper_fragment_cascades.py" \
  --input "1,1,Binary Cascade,${campaign},${run_root}" \
  --output "${run_root}/compiled/proton_copper_binary_smoke.cinpkg" \
  --metadata "${run_root}/compiled/proton_copper_binary_smoke.metadata.json" \
  > "${run_root}/compiled/compile.log"
