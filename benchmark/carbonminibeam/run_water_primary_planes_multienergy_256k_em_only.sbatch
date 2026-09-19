#!/bin/bash
#SBATCH --job-name=mb_water_planes_e150_e300
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=60G
#SBATCH --output=/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/job_%j.err

set -euo pipefail
export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data

source_dir=/mnt/sdb/wuwei/MAIGO_pristine/benchmark/carbonminibeam
run_root=/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919

for energy in 150 300; do
    run_dir=${run_root}/e${energy}/topas
    mkdir -p "${run_dir}/output"
    cp "${source_dir}/run1.txt" \
       "${source_dir}/run_energy_scan.txt" \
       "${source_dir}/run_energy_scan_electron_components.txt" \
       "${source_dir}/run_water_primary_planes_e${energy}_256k_em_only.txt" \
       "${source_dir}/aperture.txt" "${run_dir}/"
    cd "${run_dir}"
    /home/wuwei/topas/topas-build/topas \
        "run_water_primary_planes_e${energy}_256k_em_only.txt"
done
