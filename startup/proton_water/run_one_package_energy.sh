#!/usr/bin/env bash
# Run and compile one proton-water package energy in an isolated remote task root.
set -euo pipefail

if (($# != 4)); then
    echo "usage: $0 ENERGY_MEV HISTORIES SEED STAGE" >&2
    exit 2
fi
energy="$1"
histories="$2"
seed="$3"
stage="$4"
[[ "$energy" =~ ^[1-9][0-9]*$ && "$histories" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ ]] || exit 2
[[ "$stage" =~ ^[a-z0-9_-]+$ ]] || exit 2

task_root="${MAIGO_PROTON_ROOT:-/home/v/maigo_proton_water_qgsp_bic_hp_20260819}"
topas_bin="${TOPAS_BIN:-$task_root/topas-install/bin/topas}"
threads="${THREADS:-56}"
physics_model="QGSP_BIC_HP/BinaryCascade"
run_dir="$task_root/runs/$stage/${energy}MeV"
output_dir="$task_root/compiled/$stage"
tools="$task_root/startup/package_tools"
tag="proton_${energy}MeV_water_${stage}"
mkdir -p "$run_dir" "$output_dir" "$task_root/manifests"

sed -e "s|@SEED@|$seed|g" -e "s|@THREADS@|$threads|g" \
    -e "s|@HISTORIES@|$histories|g" -e "s|@ENERGY_MEV@|$energy|g" \
    -e "s|@OUTPUT_STEM@|$tag|g" \
    "$task_root/startup/proton_water/proton_water_cascade.txt.in" > "$run_dir/run.txt"

# shellcheck disable=SC1091
. /home/v/software/geant4-v11.3.2-install/bin/geant4.sh
export LD_LIBRARY_PATH="$task_root/topas-install/lib:/home/v/software/topas/OpenTOPAS-install-v4.2.3-carbon/lib:/home/v/software/geant4-v11.3.2-install/lib:${LD_LIBRARY_PATH:-}"
start_seconds="$(date +%s)"
(
    flock -n 9 || { echo "another proton-water run is active" >&2; exit 73; }
    cd "$run_dir"
    "$topas_bin" run.txt > "${stage}_${energy}MeV.log" 2>&1
) 9>"$task_root/.proton_water.lock"
elapsed_seconds="$(( $(date +%s) - start_seconds ))"
printf '%s\n' "$elapsed_seconds" > "$run_dir/elapsed_seconds.txt"
grep -Fq 'TOPAS run sequence complete.' "$run_dir/${stage}_${energy}MeV.log"
grep -Fq "Total number of histories: $histories" "$run_dir/${stage}_${energy}MeV.log"

python3 "$tools/deduplicate_primary_tables.py" --kind stopping \
    --input "$run_dir/${tag}_stopping_power.phsp" \
    --header "$run_dir/${tag}_stopping_power.header" \
    --output "$output_dir/${tag}_stopping_power.csv" \
    --metadata "$output_dir/${tag}_stopping_power.metadata.json"
python3 "$tools/deduplicate_primary_tables.py" --kind cross-section \
    --input "$run_dir/${tag}_cross_sections.phsp" \
    --header "$run_dir/${tag}_cross_sections.header" \
    --output "$output_dir/${tag}_inelastic_cross_sections.csv" \
    --metadata "$output_dir/${tag}_inelastic_cross_sections.metadata.json"

python3 "$tools/prepare_topas_cascade.py" --case "$tag" --histories "$histories" \
    --input-dir "$run_dir" --stem "${tag}_cascade" --log-name "${stage}_${energy}MeV.log" \
    --phantom-half-length-mm 350 \
    --interactions-output "$run_dir/${tag}_interactions.csv.gz" \
    --products-output "$run_dir/${tag}_products.csv.gz" \
    --metadata "$output_dir/${tag}.metadata.json"
python3 "$tools/compile_cascade_package.py" --metadata "$output_dir/${tag}.metadata.json" \
    --interactions "$run_dir/${tag}_interactions.csv.gz" \
    --products "$run_dir/${tag}_products.csv.gz" \
    --output "$output_dir/${tag}_cascade_3d.bin" \
    --output-metadata "$output_dir/${tag}_cascade_3d.compiled.json" \
    --material G4_WATER --physics-model "$physics_model" \
    --source-projectile-z 1 --source-projectile-a 1
python3 "$tools/prepare_primary_reactions_from_cascade.py" \
    --cascade-metadata "$output_dir/${tag}.metadata.json" \
    --interactions "$run_dir/${tag}_interactions.csv.gz" \
    --products "$run_dir/${tag}_products.csv.gz" \
    --reactions-output "$run_dir/${tag}_primary_reactions.csv.gz" \
    --secondaries-output "$run_dir/${tag}_primary_secondaries.csv.gz" \
    --metadata-output "$output_dir/${tag}_primary.metadata.json" \
    --projectile-z 1 --projectile-a 1
python3 "$tools/compile_reaction_package.py" \
    --metadata "$output_dir/${tag}_primary.metadata.json" \
    --reactions "$run_dir/${tag}_primary_reactions.csv.gz" \
    --secondaries "$run_dir/${tag}_primary_secondaries.csv.gz" \
    --output "$output_dir/${tag}_primary_3d.bin" \
    --output-metadata "$output_dir/${tag}_primary_3d.compiled.json" \
    --energy-bin-min-mevu 0 --energy-bin-width-mevu 1 \
    --energy-bin-count "$((energy + 1))" --fill-empty nearest \
    --material G4_WATER --physics-model "$physics_model"

for sidecar in "$output_dir/${tag}_primary_3d.compiled.json" \
               "$output_dir/${tag}_cascade_3d.compiled.json"; do
    binary="${sidecar%.compiled.json}.bin"
    expected_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["sha256"])' "$sidecar")"
    expected_bytes="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["bytes"])' "$sidecar")"
    [[ "$(sha256sum "$binary" | cut -d' ' -f1)" == "$expected_sha" ]]
    [[ "$(stat -c%s "$binary")" == "$expected_bytes" ]]
done

rm "$run_dir/${tag}_cascade.phsp"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stage" "$energy" "$histories" "$seed" "$elapsed_seconds" "$physics_model" \
    >> "$task_root/manifests/${stage}_completed.tsv"
echo "completed $stage ${energy}MeV in ${elapsed_seconds}s"
