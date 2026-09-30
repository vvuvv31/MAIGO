#!/bin/bash
#SBATCH --job-name=cp_topas100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --hint=nomultithread
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/sda/wuwei/topas_carbon_proton_100threads_20260930/slurm_%j.log
set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH="/software/topas/lib:/software/geant4-11.3.2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun --cpu-bind=cores python3 /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/topas_carbon_proton_100threads_20260930/execute.py
