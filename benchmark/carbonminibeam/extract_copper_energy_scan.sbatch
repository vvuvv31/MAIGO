#!/bin/bash
#SBATCH --job-name=maigo-cu-energy
#SBATCH --partition=compute
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/sda/wuwei/minibeam_copper_extract_energy_%A_%a.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_copper_extract_energy_%A_%a.err

set -euo pipefail
energies_mevu=(150 250 300)
energies_total_mev=(1800 3000 3600)
campaigns=(
    20260917-0150-4000-8000-000000000031
    20260917-0250-4000-8000-000000000032
    20260917-0300-4000-8000-000000000033
)
energy_mevu=${energies_mevu[${SLURM_ARRAY_TASK_ID}]}
energy_total_mev=${energies_total_mev[${SLURM_ARRAY_TASK_ID}]}
campaign=${campaigns[${SLURM_ARRAY_TASK_ID}]}
repo=/mnt/sdb/wuwei/MAIGO_pristine
run_root=/mnt/sda/wuwei/minibeam_copper_extract_e${energy_mevu}

mkdir -p "${run_root}"
cp "${repo}/benchmark/carbonminibeam/extract_copper_physics.txt" "${run_root}/run.txt"
sed -i "s/d:So\/CarbonBeam\/BeamEnergy = 2150 MeV/d:So\/CarbonBeam\/BeamEnergy = ${energy_total_mev} MeV/" \
    "${run_root}/run.txt"
export CARBON_CINEL02_OUTPUT_DIR="${run_root}/raw"
export CARBON_CINEL02_CAMPAIGN_UUID="${campaign}"
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
export CARBON_CINEL02_RUN_TAG="c12_cu_inclxx_e${energy_mevu}"
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=11.3.2
export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
export CARBON_PRODUCTION_CUTS='all=0.05mm'
export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
export CARBON_RANDOM_SEED_POLICY="Ts/Seed=240917;192 threads;energy=${energy_mevu}MeV/u"

cd "${run_root}"
/home/wuwei/topas/topas-build/topas run.txt
python3 "${repo}/benchmark/carbonminibeam/compile_copper_physics.py" \
    "${run_root}" "${run_root}/compiled" --campaign "${campaign}"
