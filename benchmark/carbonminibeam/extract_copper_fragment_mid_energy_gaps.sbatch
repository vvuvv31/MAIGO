#!/bin/bash
#SBATCH --job-name=maigo-cu-pdt-gap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail
repo=/mnt/sdb/wuwei/MAIGO_pristine
root=/mnt/sda/wuwei/minibeam_copper_fragment_mid_energy_gaps_20260919
base_metadata=/mnt/sda/wuwei/minibeam_copper_fragment_cascade_per_energy/e300/metadata.json
base_package=/mnt/sda/wuwei/minibeam_copper_fragment_cascade_per_energy/e300/copper_fragment_cascade.cinpkg
histories=10000
threads_per_run=32
parallel_runs=2

# energy values below are MeV/u. TOPAS BeamEnergy is total kinetic energy.
# End points overlap the existing low- and high-energy campaigns.
cases=()
for energy_u in $(seq 295 5 340); do
  cases+=("proton|1|1|Binary Cascade|${energy_u}")
done
for energy_u in $(seq 295 5 430); do
  cases+=("deuteron|1|2|INCL++ undefined|${energy_u}")
done
for energy_u in $(seq 295 5 380); do
  cases+=("triton|1|3|INCL++ undefined|${energy_u}")
done

run_case() {
  local spec=$1
  local case_index=$2
  local particle z a model energy_u total_energy campaign run_root
  IFS='|' read -r particle z a model energy_u <<<"${spec}"
  total_energy=$((energy_u * a))
  campaign=$(printf '20260919-%04d-4000-8000-%012d' \
    "$((1000 + case_index))" "$((1000 + case_index))")
  run_root="${root}/z${z}_a${a}_eu${energy_u}"
  mkdir -p "${run_root}"
  cp "${repo}/benchmark/carbonminibeam/extract_copper_fragment_cascade.txt" \
     "${run_root}/run.txt"
  sed -i \
    -e "s/i:Ts\/Seed = 240918/i:Ts\/Seed = $((260919 + case_index))/" \
    -e "s/i:Ts\/NumberOfThreads = 32/i:Ts\/NumberOfThreads = ${threads_per_run}/" \
    -e "s#s:So/FragmentBeam/BeamParticle = \"proton\"#s:So/FragmentBeam/BeamParticle = \"${particle}\"#" \
    -e "s/d:So\/FragmentBeam\/BeamEnergy = 700 MeV/d:So\/FragmentBeam\/BeamEnergy = ${total_energy} MeV/" \
    -e "s/i:So\/FragmentBeam\/NumberOfHistoriesInRun = 50000/i:So\/FragmentBeam\/NumberOfHistoriesInRun = ${histories}/" \
    "${run_root}/run.txt"

  export CARBON_CINEL02_OUTPUT_DIR="${run_root}/raw"
  export CARBON_CINEL02_CAMPAIGN_UUID="${campaign}"
  export CARBON_CINEL02_PRIMARY_ONLY=true
  export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
  export CARBON_CINEL02_RUN_TAG="z${z}_a${a}_eu${energy_u}_cu_gap"
  export CARBON_TOPAS_VERSION=4.2.p3
  export CARBON_GEANT4_VERSION=11.3.2
  export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
  export CARBON_PRODUCTION_CUTS='all=0.05mm'
  export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
  export CARBON_RANDOM_SEED_POLICY="Ts/Seed=$((260919 + case_index));${threads_per_run} threads;Z=${z};A=${a};energy=${energy_u}MeV/u"
  (cd "${run_root}" && /home/wuwei/topas/topas-build/topas run.txt \
    > topas.log 2>&1)
}

compiler_inputs=()
active=0
case_index=0
for spec in "${cases[@]}"; do
  case_index=$((case_index + 1))
  IFS='|' read -r particle z a model energy_u <<<"${spec}"
  campaign=$(printf '20260919-%04d-4000-8000-%012d' \
    "$((1000 + case_index))" "$((1000 + case_index))")
  run_root="${root}/z${z}_a${a}_eu${energy_u}"
  compiler_inputs+=(--input "${z},${a},${model},${campaign},${run_root}")
  run_case "${spec}" "${case_index}" &
  active=$((active + 1))
  if ((active == parallel_runs)); then
    wait
    active=0
  fi
done
wait

mkdir -p "${root}/compiled"
python3 "${repo}/benchmark/carbonminibeam/compile_copper_fragment_cascades.py" \
  --reuse-source-metadata "${base_metadata}" \
  "${compiler_inputs[@]}" \
  --preserve-projectile-package "6,12,${base_package}" \
  --package-campaign 20260919-1060-4000-8000-000000001060 \
  --output "${root}/compiled/copper_fragment_cascade_e300_gapfilled.cinpkg" \
  --metadata "${root}/compiled/copper_fragment_cascade_e300_gapfilled.metadata.json" \
  > "${root}/compiled/compile.log"

python3 "${repo}/benchmark/carbonminibeam/report_copper_fragment_coverage.py" \
  --log /mnt/sda/wuwei/minibeam_e300_256k_miss_joint_20260919/run.log \
  --package "${root}/compiled/copper_fragment_cascade_e300_gapfilled.cinpkg" \
  --json "${root}/compiled/postfill_static_coverage.json" \
  --csv "${root}/compiled/baseline_miss_joint.csv" \
  > "${root}/compiled/postfill_static_coverage.log"
