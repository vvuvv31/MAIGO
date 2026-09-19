#!/bin/bash
#SBATCH --job-name=elastic_all
#SBATCH --array=0-16%4
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=/mnt/sda/wuwei/all_ion_elastic_20260911/all_%A_%a.log
#SBATCH --error=/mnt/sda/wuwei/all_ion_elastic_20260911/all_%A_%a.err
set -euo pipefail
cd /mnt/sda/wuwei/all_ion_elastic_20260911
source /software/geant4-11.3.2/bin/geant4.sh
case_name=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /mnt/sda/wuwei/all_ion_elastic_20260911/cases.txt)
/home/wuwei/topas/all-ion-elastic-build-20260911/topas "/mnt/sda/wuwei/all_ion_elastic_20260911/$case_name.txt"
