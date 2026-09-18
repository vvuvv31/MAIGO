#!/bin/bash
#SBATCH --job-name=mb_1m_ens
#SBATCH --partition=compute
#SBATCH --array=2-4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/minibeam_field3cm_1m_s%a_%A_%a.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_field3cm_1m_s%a_%A_%a.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
run_dir=/mnt/sda/wuwei/minibeam_field3cm_1m_s${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
run_file=run_field3cm_1m_s${SLURM_ARRAY_TASK_ID}.txt
mkdir -p "${run_dir}"
cp "${source_dir}/run1.txt" "${source_dir}/${run_file}" \
   "${source_dir}/aperture.txt" "${source_dir}/beam_model.csv" \
   "${source_dir}/spots.csv" "${run_dir}/"
awk -F, 'BEGIN {OFS=","} NR==1 {print; next} \
    ($2 >= -15 && $2 <= 15 && $3 >= -15 && $3 <= 15) \
    {$1 = kept++; print}' "${run_dir}/spots.csv" > "${run_dir}/spots_3cm.csv"
cd "${run_dir}"
/home/wuwei/topas/topas-build/topas "${run_file}"
