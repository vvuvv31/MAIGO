#!/bin/bash
#SBATCH --job-name=mb_em250_1024k
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=160G
#SBATCH --output=/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k/topas_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k/topas_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
input_dir=/mnt/sda/wuwei/minibeam_energy_scan_inputs/e250
beam_model=/mnt/sda/wuwei/minibeam_phase_scan_e250_6255_1/beam_model.csv
run_root=/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k
run_dir=${run_root}/topas_${SLURM_JOB_ID}
mkdir -p "${run_dir}"
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/run_field3cm_em_only_e250_1024k.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"
cp "${input_dir}/spots_3cm.csv" "${run_dir}/"
cp "${beam_model}" "${run_dir}/beam_model.csv"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas run_field3cm_em_only_e250_1024k.txt
