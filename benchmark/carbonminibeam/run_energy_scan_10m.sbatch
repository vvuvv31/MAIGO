#!/bin/bash
#SBATCH --job-name=mb_energy_10m
#SBATCH --partition=compute
#SBATCH --array=0-14%3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=12G
#SBATCH --output=/mnt/sda/wuwei/minibeam_energy_10m_%A_%a.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_energy_10m_%A_%a.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

energies_mevu=(150 250 300)
energies_total_mev=(1800 3000 3600)
energy_index=$((SLURM_ARRAY_TASK_ID / 5))
shard=$((SLURM_ARRAY_TASK_ID % 5))
energy_mevu=${energies_mevu[${energy_index}]}
energy_total_mev=${energies_total_mev[${energy_index}]}
seed=$((2026091700 + energy_index * 100 + shard))
source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
reference_spots=/mnt/sda/wuwei/minibeam_field3cm_6037/spots_3cm.csv
run_dir=/mnt/sda/wuwei/minibeam_energy_10m_e${energy_mevu}_s${shard}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "${run_dir}"
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"
awk -v seed="${seed}" '{gsub(/SEED_VALUE/, seed); print}' \
    "${source_dir}/run_energy_scan_2m_shard.txt" \
    > "${run_dir}/run_energy_scan_2m_shard.txt"
awk -F, -v energy="${energy_total_mev}" 'BEGIN {OFS=","} NR==1 {print; next} {$4=energy; print}' \
    "${reference_spots}" > "${run_dir}/spots_3cm.csv"
awk -F, -v energy="${energy_total_mev}" 'BEGIN {OFS=","} NR==1 {print; next} {$1=energy; print}' \
    "${source_dir}/beam_model.csv" > "${run_dir}/beam_model.csv"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas run_energy_scan_2m_shard.txt
