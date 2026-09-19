#!/bin/bash
#SBATCH --job-name=mb_water_planes_em12800k
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/minibeam_water_planes_e250_em12800k_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_water_planes_e250_em12800k_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
phase_dir=/mnt/sda/wuwei/minibeam_water_phsp_e250_em12800k/topas_6455/output
input_dir=/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/input
run_root=/mnt/sda/wuwei/minibeam_water_planes_e250_em12800k
run_dir=${run_root}/topas_${SLURM_JOB_ID}
mkdir -p "${input_dir}" "${run_dir}/output"
python3 "${source_dir}/prepare_water_entry_replay.py" \
    "${phase_dir}/water_entrance.phsp" \
    --header "${phase_dir}/water_entrance.header" \
    --output-dir "${input_dir}" \
    --water-entrance-mm 60.0 \
    --selection primary-c12
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/run_energy_scan_electron_components.txt" \
   "${source_dir}/run_water_entry_primary_replay_e250_12800k_em_only.txt" \
   "${source_dir}/run_water_primary_planes_e250_12800k_em_only.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas \
    run_water_primary_planes_e250_12800k_em_only.txt
