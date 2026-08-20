#!/usr/bin/env bash
# Render and serially run reproducible proton-water TOPAS stages on 192.168.31.5.
# The default mode is dry-run; --run is intentionally required for Monte Carlo.
set -euo pipefail

ROOT="${MAIGO_PROTON_ROOT:-/home/v/maigo_proton_water_qgsp_bic_hp_20260819}"
TOPAS_BIN="${TOPAS_BIN:-${ROOT}/topas-install/bin/topas}"
THREADS="${THREADS:-56}"
MODE=dry-run
STAGE=pilot
ENERGIES=(70 100 150 200 250)
PHYSICS_MODEL="QGSP_BIC_HP/BinaryCascade"

usage() {
    cat <<'EOF'
Usage: run_proton_water_remote.sh [--dry-run|--run] [--stage pilot|production|reference]

Run this script only after synchronizing startup/extensions, startup/package_tools,
and startup/proton_water to v@192.168.31.5. It runs one energy at a time under
flock and writes a seed manifest. production/reference use distinct seed ranges.
EOF
}
while (($#)); do
    case "$1" in
        --dry-run) MODE=dry-run; shift ;;
        --run) MODE=run; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$STAGE" in pilot) HISTORIES=100000; SEED_BASE=2026081900 ;; production) HISTORIES=1000000; SEED_BASE=2026082900 ;; reference) HISTORIES=1000000; SEED_BASE=2026083900 ;; *) echo "invalid stage: $STAGE" >&2; exit 2 ;; esac
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] && (( THREADS <= 56 )) || { echo "THREADS must be 1..56" >&2; exit 2; }

manifest="$ROOT/manifests/${STAGE}_seeds.tsv"
if [[ "$MODE" == run ]]; then
    mkdir -p "$ROOT/runs/$STAGE" "$ROOT/manifests"
    printf 'stage\tenergy_MeV\thistories\tseed\tphysics_model\n' > "$manifest"
else
    printf '[dry-run] seed manifest: %s\n' "$manifest"
fi
for energy in "${ENERGIES[@]}"; do
    seed=$((SEED_BASE + energy))
    run_dir="$ROOT/runs/$STAGE/${energy}MeV"
    [[ "$MODE" == run ]] && mkdir -p "$run_dir"
    config="$run_dir/run.txt"
    tag="proton_${energy}MeV_water_${STAGE}"
    if [[ "$MODE" == run ]]; then
        sed -e "s|@SEED@|$seed|g" -e "s|@THREADS@|$THREADS|g" \
            -e "s|@HISTORIES@|$HISTORIES|g" -e "s|@ENERGY_MEV@|$energy|g" \
            -e "s|@OUTPUT_STEM@|$tag|g" \
            "$ROOT/startup/proton_water/proton_water_cascade.txt.in" > "$config"
        printf '%s\t%s\t%s\t%s\t%s\n' "$STAGE" "$energy" "$HISTORIES" "$seed" "$PHYSICS_MODEL" >> "$manifest"
    fi
    if [[ "$MODE" == dry-run ]]; then
        printf '[dry-run] flock %q %q %q\n' "$ROOT/.proton_water.lock" "$TOPAS_BIN" "$config"
        continue
    fi
    [[ -x "$TOPAS_BIN" ]] || { echo "TOPAS not executable: $TOPAS_BIN" >&2; exit 2; }
    (
        flock -n 9 || { echo "another proton-water run is active" >&2; exit 1; }
        cd "$run_dir"
        "$TOPAS_BIN" run.txt > "${STAGE}_${energy}MeV.log" 2>&1
    ) 9>"$ROOT/.proton_water.lock"
    if [[ "$STAGE" != reference ]]; then
        tools="$ROOT/startup/package_tools"
        package_dir="$ROOT/compiled/$STAGE"
        mkdir -p "$package_dir"
        python3 "$tools/prepare_topas_cascade.py" --case "$tag" --histories "$HISTORIES" \
            --input-dir "$run_dir" --stem "${tag}_cascade" --log-name "${STAGE}_${energy}MeV.log" \
            --phantom-half-length-mm 350 --interactions-output "$run_dir/${tag}_interactions.csv.gz" \
            --products-output "$run_dir/${tag}_products.csv.gz" --metadata "$package_dir/${tag}.metadata.json"
        python3 "$tools/compile_cascade_package.py" --metadata "$package_dir/${tag}.metadata.json" \
            --interactions "$run_dir/${tag}_interactions.csv.gz" --products "$run_dir/${tag}_products.csv.gz" \
            --output "$package_dir/${tag}_cascade_3d.bin" --output-metadata "$package_dir/${tag}_cascade_3d.compiled.json" \
            --material G4_WATER --physics-model "$PHYSICS_MODEL" --source-projectile-z 1 --source-projectile-a 1
        python3 "$tools/prepare_primary_reactions_from_cascade.py" --cascade-metadata "$package_dir/${tag}.metadata.json" \
            --interactions "$run_dir/${tag}_interactions.csv.gz" --products "$run_dir/${tag}_products.csv.gz" \
            --reactions-output "$run_dir/${tag}_primary_reactions.csv.gz" --secondaries-output "$run_dir/${tag}_primary_secondaries.csv.gz" \
            --metadata-output "$package_dir/${tag}_primary.metadata.json" --projectile-z 1 --projectile-a 1
        python3 "$tools/compile_reaction_package.py" --metadata "$package_dir/${tag}_primary.metadata.json" \
            --reactions "$run_dir/${tag}_primary_reactions.csv.gz" --secondaries "$run_dir/${tag}_primary_secondaries.csv.gz" \
            --output "$package_dir/${tag}_primary_3d.bin" --output-metadata "$package_dir/${tag}_primary_3d.compiled.json" \
            --energy-bin-min-mevu 0 --energy-bin-width-mevu 1 --energy-bin-count $((energy + 1)) \
            --fill-empty nearest --material G4_WATER --physics-model "$PHYSICS_MODEL"
    fi
done
if [[ "$MODE" == run ]]; then
    printf 'seed manifest: %s\n' "$manifest"
fi
