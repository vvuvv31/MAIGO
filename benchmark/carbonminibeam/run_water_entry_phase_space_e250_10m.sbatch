#!/bin/bash
#SBATCH --job-name=mb_phsp250_10m
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
# The previous identical-history run peaked at 28.7 GB RSS.
#SBATCH --mem=40G
#SBATCH --output=/mnt/sda/wuwei/minibeam_water_phsp_e250_10m_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_water_phsp_e250_10m_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
reference_spots=/mnt/sda/wuwei/minibeam_field3cm_6037/spots_3cm.csv
run_dir=/mnt/sda/wuwei/minibeam_water_phsp_e250_10m_${SLURM_JOB_ID}

mkdir -p "${run_dir}/output"
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/run_water_entry_phase_space_e250_10m.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"
awk -F, 'BEGIN {OFS=","} NR==1 {print; next} {$4=3000; print}' \
    "${reference_spots}" > "${run_dir}/spots_3cm.csv"
awk -F, 'BEGIN {OFS=","} NR==1 {print; next} {$1=3000; print}' \
    "${source_dir}/beam_model.csv" > "${run_dir}/beam_model.csv"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas run_water_entry_phase_space_e250_10m.txt
