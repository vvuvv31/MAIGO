#!/bin/bash
#SBATCH --job-name=maigo-cu-light-low
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail
repo=/mnt/sdb/wuwei/MAIGO_pristine
root=/mnt/sda/wuwei/minibeam_copper_fragment_light_low_energy
histories=10000

# particle|Z|A|model|total kinetic energies (MeV)
cases=(
  'proton|1|1|Binary Cascade|50 100 200 300'
  'deuteron|1|2|INCL++ undefined|100 200 400 600'
  'triton|1|3|INCL++ undefined|150 300 600 900'
)

compiler_inputs=()
case_index=0
for spec in "${cases[@]}"; do
  IFS='|' read -r particle z a model energies <<<"${spec}"
  for energy in ${energies}; do
    case_index=$((case_index + 1))
    campaign=$(printf '20260918-%04d-4000-8000-%012d' \
      "${case_index}" "${case_index}")
    run_root="${root}/z${z}_a${a}_e${energy}"
    mkdir -p "${run_root}"
    cp "${repo}/benchmark/carbonminibeam/extract_copper_fragment_cascade.txt" \
       "${run_root}/run.txt"
    sed -i \
      -e "s#s:So/FragmentBeam/BeamParticle = \"proton\"#s:So/FragmentBeam/BeamParticle = \"${particle}\"#" \
      -e "s/d:So\/FragmentBeam\/BeamEnergy = 700 MeV/d:So\/FragmentBeam\/BeamEnergy = ${energy} MeV/" \
      -e "s/i:So\/FragmentBeam\/NumberOfHistoriesInRun = 50000/i:So\/FragmentBeam\/NumberOfHistoriesInRun = ${histories}/" \
      "${run_root}/run.txt"
    export CARBON_CINEL02_OUTPUT_DIR="${run_root}/raw"
    export CARBON_CINEL02_CAMPAIGN_UUID="${campaign}"
    export CARBON_CINEL02_PRIMARY_ONLY=true
    export CARBON_CINEL02_OVERWRITE_CAMPAIGN=true
    export CARBON_CINEL02_RUN_TAG="z${z}_a${a}_e${energy}_cu_low"
    export CARBON_TOPAS_VERSION=4.2.p3
    export CARBON_GEANT4_VERSION=11.3.2
    export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
    export CARBON_PRODUCTION_CUTS='all=0.05mm'
    export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
    export CARBON_RANDOM_SEED_POLICY="Ts/Seed=250918;32 threads;Z=${z};A=${a};energy=${energy}MeV"
    (cd "${run_root}" && /home/wuwei/topas/topas-build/topas run.txt)
    compiler_inputs+=(--input "${z},${a},${model},${campaign},${run_root}")
  done
done

# Add the already validated broad pilot campaigns, then merge the low-energy
# light-ion campaigns for continuous strict lookup coverage.
compiler_inputs+=(
  --input '1,1,Binary Cascade,20260918-0001-4000-8000-000000000041,/mnt/sda/wuwei/minibeam_copper_fragment_proton_bic_smoke'
  --input '6,12,INCL++ undefined,20260917-0250-4000-8000-000000000032,/mnt/sda/wuwei/minibeam_copper_extract_e250'
)
for entry in \
  '1,2,INCL++ undefined,20260918-0042-4000-8000-000000000042,z1_a2' \
  '1,3,INCL++ undefined,20260918-0043-4000-8000-000000000043,z1_a3' \
  '2,3,INCL++ undefined,20260918-0044-4000-8000-000000000044,z2_a3' \
  '2,4,INCL++ undefined,20260918-0045-4000-8000-000000000045,z2_a4' \
  '3,6,INCL++ undefined,20260918-0046-4000-8000-000000000046,z3_a6' \
  '3,7,INCL++ undefined,20260918-0047-4000-8000-000000000047,z3_a7' \
  '4,7,INCL++ undefined,20260918-0048-4000-8000-000000000048,z4_a7' \
  '5,10,INCL++ undefined,20260918-0049-4000-8000-000000000049,z5_a10' \
  '5,11,INCL++ undefined,20260918-0050-4000-8000-000000000050,z5_a11' \
  '6,11,INCL++ undefined,20260918-0051-4000-8000-000000000051,z6_a11'; do
  IFS=',' read -r z a model campaign directory <<<"${entry}"
  compiler_inputs+=(--input "${z},${a},${model},${campaign},/mnt/sda/wuwei/minibeam_copper_fragment_cascade_pilot/${directory}")
done

mkdir -p "${root}/compiled"
python3 "${repo}/benchmark/carbonminibeam/compile_copper_fragment_cascades.py" \
  "${compiler_inputs[@]}" \
  --output "${root}/compiled/copper_fragment_cascade_light_complete.cinpkg" \
  --metadata "${root}/compiled/copper_fragment_cascade_light_complete.metadata.json" \
  > "${root}/compiled/compile.log"
