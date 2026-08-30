#!/bin/bash
#SBATCH --job-name=c12_el_lib
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

set -euo pipefail
ROOT=/mnt/sda/wuwei/path_a_topas_extract
TOPAS=/home/wuwei/topas/topas-build/topas
TEMPLATE="$ROOT/elastic_target_only.txt.in"
THREADS=50
HISTORIES="${HISTORIES:-500000}"
SEED_BASE="${SEED_BASE:-2026083010}"

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH="/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

[[ -x "$TOPAS" ]] || { echo "missing TOPAS" >&2; exit 2; }

# One XS query per target (energy grid 0-400 MeV/u). Event runs per energy.
declare -a TARGETS=(H1 C12 O16)
declare -a MATERIALS=(ElasticH1 ElasticC12 ElasticO16)
declare -a ENERGIES=(95 200 300 400)

run_one() {
    local label="$1" material="$2" e_u="$3" seed="$4" kind="$5" nhist="$6"
    local total
    total=$(python3 -c "print(12.0 * float('$e_u'))")
    local outdir="$ROOT/elastic/${label}/e${e_u}/${kind}"
    mkdir -p "$outdir"
    local stem="c12_${label}_${e_u}mevu_${kind}"
    sed \
        -e "s|@SEED@|$seed|g" \
        -e "s|@THREADS@|$THREADS|g" \
        -e "s|@HISTORIES@|$nhist|g" \
        -e "s|@TARGET_MATERIAL@|$material|g" \
        -e "s|@TOTAL_ENERGY_MEV@|$total|g" \
        -e "s|@OUTPUT_STEM@|$stem|g" \
        -e "s|@XS_OUTPUT_STEM@|${stem}_xs|g" \
        "$TEMPLATE" > "$outdir/run.txt"
    echo "[$(date -Is)] elastic $kind target=$label E=$e_u MeV/u hist=$nhist"
    (
        cd "$outdir"
        "$TOPAS" run.txt > topas.log 2>&1
    )
    [[ -s "$outdir/${stem}_elastic.header" ]]
}

seed=$SEED_BASE
for i in 0 1 2; do
    label="${TARGETS[$i]}"
    material="${MATERIALS[$i]}"
    # XS query once at 400 MeV/u (grid covers 0-400)
    run_one "$label" "$material" 400 "$seed" xs 1
    seed=$((seed + 1))
    for e in "${ENERGIES[@]}"; do
        run_one "$label" "$material" "$e" "$seed" event "$HISTORIES"
        seed=$((seed + 1))
    done
done
echo "[$(date -Is)] elastic extraction complete"
