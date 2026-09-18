#!/bin/bash
#SBATCH --job-name=mb_water_planes
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=40G
#SBATCH --output=/mnt/sda/wuwei/minibeam_water_planes_e250_10m_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_water_planes_e250_10m_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
run_dir=/mnt/sda/wuwei/minibeam_water_planes_e250_10m/topas_${SLURM_JOB_ID}
mkdir -p "${run_dir}/output"
cp "${source_dir}/run1.txt" "${source_dir}/run_energy_scan.txt" \
   "${source_dir}/run_water_entry_primary_replay_e250_10m.txt" \
   "${source_dir}/run_water_primary_planes_e250_10m.txt" \
   "${source_dir}/aperture.txt" "${run_dir}/"

cd "${run_dir}"
/home/wuwei/topas/topas-build/topas run_water_primary_planes_e250_10m.txt
