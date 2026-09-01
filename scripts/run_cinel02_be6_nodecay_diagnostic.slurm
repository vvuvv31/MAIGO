#!/usr/bin/env bash
#SBATCH --job-name=cinbe6nodecay
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/sda/wuwei/cinel02-be6-nodecay-diagnostic/logs/%x_%j.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-be6-nodecay-diagnostic/logs/%x_%j.err

set -euo pipefail
readonly source_run=/mnt/sda/wuwei/cinel02-be6-secondary-smoke/z4a6_e200_h50000/run.txt
readonly root=/mnt/sda/wuwei/cinel02-be6-nodecay-diagnostic
readonly run_dir=${root}/z4a6_e200_h50000
readonly raw_root=${run_dir}/raw
readonly campaign_uuid=00000000-0000-4000-8000-000000960207
readonly run_tag=z4a6_e200_h50000_nodecay
readonly topas_bin=/home/wuwei/topas/topas-build/topas
readonly seed=2026099607

test -x "${topas_bin}"
test -s "${source_run}"
mkdir -p "${run_dir}" "${raw_root}" "${root}/logs"
sed \
  -e 's|sv:Ph/Default/Modules = 8 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "CarbonInelasticCapturePhysics" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"|sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4ion-inclxx" "CarbonInelasticCapturePhysics" "g4h-elastic_HP" "g4stopping"|' \
  -e "s|i:Ts/Seed = 2026099606|i:Ts/Seed = ${seed}|" \
  "${source_run}" > "${run_dir}/run_nodecay.txt"

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH=/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CARBON_CINEL02_OUTPUT_DIR=${raw_root}
export CARBON_CINEL02_CAMPAIGN_UUID=${campaign_uuid}
export CARBON_CINEL02_RUN_TAG=${run_tag}
export CARBON_CINEL02_PRIMARY_ONLY=true
export CARBON_CINEL02_OVERWRITE_CAMPAIGN=false
export CARBON_TOPAS_VERSION=4.2.p3
export CARBON_GEANT4_VERSION=geant4-11-03-patch-02
export CARBON_PHYSICS_LIST=FTFP_INCLXX
export CARBON_PRODUCTION_CUTS=0.05mm
export CARBON_STEP_LIMITS=default
export CARBON_RANDOM_SEED_POLICY=${seed}

cd "${run_dir}"
time "${topas_bin}" run_nodecay.txt > topas.log 2>&1
test -s "${raw_root}/${campaign_uuid}/cinel02.contract.json"
find "${raw_root}/${campaign_uuid}" -maxdepth 1 -type f -name '*.cinel02' -size +63c -print -quit | grep -q .
grep -Fq "Total number of histories: 50000" topas.log
