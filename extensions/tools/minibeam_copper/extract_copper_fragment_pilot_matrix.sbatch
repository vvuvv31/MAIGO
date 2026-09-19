#!/bin/bash
#SBATCH --job-name=maigo-cu-frag-pilot
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail
repo=/mnt/sdb/wuwei/MAIGO_pristine
pilot_root=/mnt/sda/wuwei/minibeam_copper_fragment_cascade_pilot
histories=20000

# particle|Z|A|expected model|total kinetic MeV|campaign suffix
cases=(
  'deuteron|1|2|INCL++ undefined|1200|042'
  'triton|1|3|INCL++ undefined|1500|043'
  'He3|2|3|INCL++ undefined|1350|044'
  'alpha|2|4|INCL++ undefined|1600|045'
  'GenericIon(3,6,3)|3|6|INCL++ undefined|1800|046'
  'GenericIon(3,7,3)|3|7|INCL++ undefined|2100|047'
  'GenericIon(4,7,4)|4|7|INCL++ undefined|2100|048'
  'GenericIon(5,10,5)|5|10|INCL++ undefined|2800|049'
  'GenericIon(5,11,5)|5|11|INCL++ undefined|3000|050'
  'GenericIon(6,11,6)|6|11|INCL++ undefined|3000|051'
)

compiler_inputs=(
  --input '1,1,Binary Cascade,20260918-0001-4000-8000-000000000041,/mnt/sda/wuwei/minibeam_copper_fragment_proton_bic_smoke'
  --input '6,12,INCL++ undefined,20260917-0250-4000-8000-000000000032,/mnt/sda/wuwei/minibeam_copper_extract_e250'
)

for spec in "${cases[@]}"; do
  IFS='|' read -r particle z a model energy suffix <<<"${spec}"
  campaign="20260918-00${suffix: -2}-4000-8000-000000000${suffix}"
  run_root="${pilot_root}/z${z}_a${a}"
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
  export CARBON_CINEL02_RUN_TAG="z${z}_a${a}_cu_pilot"
  export CARBON_TOPAS_VERSION=4.2.p3
  export CARBON_GEANT4_VERSION=11.3.2
  export CARBON_PHYSICS_LIST='g4em-standard_opt4,g4h-phy_QGSP_BIC_HP,g4decay,g4ion-inclxx,CarbonIonElasticPhysics,g4h-elastic_HP,g4stopping'
  export CARBON_PRODUCTION_CUTS='all=0.05mm'
  export CARBON_STEP_LIMITS='G4_Cu=0.05mm'
  export CARBON_RANDOM_SEED_POLICY="Ts/Seed=240918;32 threads;Z=${z};A=${a};energy=${energy}MeV"

  (cd "${run_root}" && /home/wuwei/topas/topas-build/topas run.txt)
  compiler_inputs+=(--input "${z},${a},${model},${campaign},${run_root}")
done

mkdir -p "${pilot_root}/compiled"
python3 "${repo}/benchmark/carbonminibeam/compile_copper_fragment_cascades.py" \
  "${compiler_inputs[@]}" \
  --output "${pilot_root}/compiled/copper_fragment_cascade_pilot.cinpkg" \
  --metadata "${pilot_root}/compiled/copper_fragment_cascade_pilot.metadata.json" \
  > "${pilot_root}/compiled/compile.log"
