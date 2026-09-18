#!/bin/bash
#SBATCH --job-name=mb_physics_smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/mnt/sda/wuwei/minibeam_topas_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_topas_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
cd /mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
/home/wuwei/topas/topas-build/topas run_smoke.txt
