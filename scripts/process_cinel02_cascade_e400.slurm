#!/usr/bin/env bash
#SBATCH --job-name=cinel02p400
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-cascade-h1o16/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-cascade-h1o16/logs/%x_%j.err

set -euo pipefail

readonly campaign_uuid=00000000-0000-4000-8000-000000000552
readonly run_dir=/mnt/sda/wuwei/cinel02-cascade-h1o16/e400MeVu_5000000h_bin1_cascade_5001_h1o16_cascade
readonly raw_dir=${run_dir}/raw/${campaign_uuid}
readonly output_dir=${run_dir}/compiled
readonly tools=/mnt/sdb/wuwei/MAIGO/startup/package_tools

mkdir -p "${output_dir}"
exposure_inputs=()
for path in "${run_dir}"/cinel02_exposure.worker_*.compact.csv; do
    exposure_inputs+=(--input "${path}")
done
python3 "${tools}/aggregate_cinel02_exposure.py" \
    "${exposure_inputs[@]}" \
    --output "${output_dir}/exposure.csv" \
    --metadata-output "${output_dir}/exposure.json" \
    --campaign-uuid "${campaign_uuid}" \
    --topas-version 4.2.p3 \
    --geant4-version geant4-11-03-patch-02 \
    --physics-list FTFP_INCLXX \
    --hadronic-process ionInelastic \
    --hadronic-model INCLXX \
    --production-cuts 0.05mm \
    --step-limits default \
    --material G4_WATER \
    --source TOPAS
# Full streaming audit retains 0.189905% as role=2 pi+/pi-/pi0/eta/e-;
# the GPU must ledger these explicitly rather than silently depositing them.
python3 "${tools}/compile_cinel02.py" \
    --metadata "${raw_dir}/cinel02.contract.json" \
    --raw "${raw_dir}"/worker_*.cinel02 \
    --output "${output_dir}/cascade_e400.cinpkg" \
    --output-metadata "${output_dir}/cascade_e400.cinpkg.json" \
    --energy-min-mevu 0 \
    --energy-bin-width-mevu 5000 \
    --minimum-events-per-bin 1 \
    --maximum-unsupported-product-energy-fraction 0.002
python3 "${tools}/compile_cinel02_rates.py" \
    --input "${output_dir}/exposure.csv" \
    --output "${output_dir}/cascade_e400_rates.csv" \
    --metadata-output "${output_dir}/cascade_e400_rates.json" \
    --minimum-events-per-bin 1 \
    --campaign-uuid "${campaign_uuid}" \
    --topas-version 4.2.p3 \
    --geant4-version geant4-11-03-patch-02 \
    --physics-list FTFP_INCLXX \
    --hadronic-process ionInelastic \
    --hadronic-model INCLXX \
    --production-cuts 0.05mm \
    --step-limits default \
    --material G4_WATER \
    --density-g-cm3 1.0 \
    --source TOPAS \
    --capture-extension-revision cinel02-v2-ground-state-isomer-normalisation
sha256sum "${output_dir}/cascade_e400.cinpkg" \
    "${output_dir}/cascade_e400_rates.csv" > "${output_dir}/SHA256SUMS"
