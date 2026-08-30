#!/bin/bash
#SBATCH --job-name=c12_thin_frag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

set -euo pipefail
ROOT=/mnt/sda/wuwei/path_a_topas_extract
TOPAS=/home/wuwei/topas/topas-build/topas
TEMPLATE="$ROOT/thin_fragment_target.txt.in"
THREADS=50
HISTORIES="${HISTORIES:-1000000}"
SEED_BASE="${SEED_BASE:-2026083020}"

export TOPAS_G4_DATA_DIR=/software/geant4-11.3.2/share/Geant4/data
export LD_LIBRARY_PATH="/software/topas/lib:/software/geant4-11.3.2/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

declare -a TARGETS=(H1 C12 O16)
declare -a MATERIALS=(TargetH1 TargetC12 TargetO16)
declare -a ENERGIES=(95 200 300 400)

seed=$SEED_BASE
for i in 0 1 2; do
    label="${TARGETS[$i]}"
    material="${MATERIALS[$i]}"
    for e in "${ENERGIES[@]}"; do
        total=$(python3 -c "print(12.0 * float('$e'))")
        outdir="$ROOT/thin_fragment/${label}/e${e}"
        mkdir -p "$outdir"
        stem="c12_${label}_${e}mevu_thin"
        sed \
            -e "s|@SEED@|$seed|g" \
            -e "s|@THREADS@|$THREADS|g" \
            -e "s|@HISTORIES@|$HISTORIES|g" \
            -e "s|@TARGET_MATERIAL@|$material|g" \
            -e "s|@TOTAL_ENERGY_MEV@|$total|g" \
            -e "s|@OUTPUT_STEM@|$stem|g" \
            "$TEMPLATE" > "$outdir/run.txt"
        echo "[$(date -Is)] thin fragment target=$label E=$e MeV/u"
        (
            cd "$outdir"
            "$TOPAS" run.txt > topas.log 2>&1
        )
        [[ -s "$outdir/${stem}_reactions.header" ]]
        seed=$((seed + 1))
    done
done
echo "[$(date -Is)] thin fragment extraction complete"
