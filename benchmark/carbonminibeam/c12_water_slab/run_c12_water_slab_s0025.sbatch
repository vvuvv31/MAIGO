#!/bin/bash
#SBATCH --job-name=c12_slab_s0025
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/sda/wuwei/c12_water_slab_step0025/slab_%j.log
#SBATCH --error=/mnt/sda/wuwei/c12_water_slab_step0025/slab_%j.err

# Paired step-dependence control for the water Urban-v2 port: identical to
# c12_water_slab_20260921/c12_e250_l10 (250 MeV/u C12, 10 mm Water_75eV,
# 250k histories, same seed 26092131) except Water MaxStepSize 0.025 mm
# instead of 0.05 mm. Tests whether Geant4 Urban itself scatters less at the
# smaller step limit (same direction as the GPU s0025 full-chain flip).
set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
run_dir=/mnt/sda/wuwei/c12_water_slab_step0025/c12_e250_l10_s0025
cd "${run_dir}"
/home/wuwei/topas/topas-build/topas run.txt
