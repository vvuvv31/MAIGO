#!/bin/bash
#SBATCH --job-name=mb_phsp250_em12800k
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/minibeam_water_phsp_e250_em12800k_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_water_phsp_e250_em12800k_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
input_dir=/mnt/sda/wuwei/minibeam_energy_scan_inputs/e250
beam_model=/mnt/sda/wuwei/minibeam_phase_scan_e250_6255_1/beam_model.csv
run_root=/mnt/sda/wuwei/minibeam_water_phsp_e250_em12800k
run_dir=${run_root}/topas_${SLURM_JOB_ID}
mkdir -p "${run_dir}/output"
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/run_energy_scan_electron_components.txt" \
   "${source_dir}/run_water_entry_phase_space_e250_12800k_em_only.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"
cp "${input_dir}/spots_3cm.csv" "${run_dir}/"
cp "${beam_model}" "${run_dir}/beam_model.csv"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas \
    run_water_entry_phase_space_e250_12800k_em_only.txt
