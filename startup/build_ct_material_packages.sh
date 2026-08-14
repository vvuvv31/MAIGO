#!/usr/bin/env bash
# Generate TOPAS INCL++ material final states and compile runtime packages.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/startup/package_tools"
TEMPLATE="$ROOT/startup/templates/ct_material_cascade.txt.in"
WORK_ROOT="$ROOT/startup/work/ct_material_packages"
PACKAGE_ROOT="$ROOT/data/packages"
TOPAS_BIN="${TOPAS_BIN:-topas}"
HISTORIES=100000
THREADS=56
MATERIALS=(lung bone)
MODE=dry-run

usage() {
    cat <<'EOF'
Usage: startup/build_ct_material_packages.sh [options]

Options:
  --dry-run             Render inputs and print commands (default).
  --run                 Run TOPAS and compile packages.
  --compile-only        Rebuild packages from existing TOPAS header/phsp/log.
  --topas-bin PATH      Extension-enabled TOPAS executable.
  --histories N         Histories per material (default: 100000).
  --threads N           TOPAS threads, range 1..56 (default: 56).
  --materials LIST      Comma-separated subset: water,soft_tissue,lung,bone,
                        schneider_section07.
  --work-root PATH      Raw/intermediate output directory.
  --package-root PATH   Runtime package destination.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run) MODE=dry-run; shift ;;
        --run) MODE=run; shift ;;
        --compile-only) MODE=compile-only; shift ;;
        --topas-bin) TOPAS_BIN="$2"; shift 2 ;;
        --histories) HISTORIES="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --materials) IFS=, read -r -a MATERIALS <<<"$2"; shift 2 ;;
        --work-root) WORK_ROOT="$2"; shift 2 ;;
        --package-root) PACKAGE_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$HISTORIES" =~ ^[1-9][0-9]*$ ]] || die "histories must be positive"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "threads must be positive"
((THREADS <= 56)) || die "threads must not exceed 56"
[[ -f "$TEMPLATE" ]] || die "missing template: $TEMPLATE"
for tool in prepare_topas_cascade.py prepare_primary_reactions_from_cascade.py \
            compile_reaction_package.py compile_cascade_package.py; do
    [[ -f "$TOOLS/$tool" ]] || die "missing package tool: $TOOLS/$tool"
done
if [[ "$MODE" == run ]]; then
    [[ -x "$TOPAS_BIN" ]] || die "TOPAS binary is not executable: $TOPAS_BIN"
fi

render() {
    local material="$1" histories="$2" destination="$3" stem="$4"
    local g4_material half_length world_half source_z seed
    case "$material" in
        water)
            g4_material=G4_WATER; half_length=1000; world_half=1100
            source_z=-999.999
            seed=2026081303
            ;;
        soft_tissue)
            g4_material=G4_TISSUE_SOFT_ICRP; half_length=1000; world_half=1100
            source_z=-999.999
            seed=2026081304
            ;;
        lung)
            g4_material=G4_LUNG_ICRP; half_length=1000; world_half=1100
            source_z=-999.999
            seed=2026081300
            ;;
        bone)
            g4_material=G4_BONE_COMPACT_ICRU; half_length=350; world_half=450
            source_z=-349.999
            seed=2026081301
            ;;
        schneider_section07)
            g4_material=SchneiderSection07; half_length=1000; world_half=1100
            source_z=-999.999
            seed=2026081302
            ;;
        *) die "unsupported material: $material" ;;
    esac
    local progress=$((histories / 10))
    ((progress > 0)) || progress=1
    sed -e "s|@SEED@|$seed|g" \
        -e "s|@THREADS@|$THREADS|g" \
        -e "s|@PROGRESS_INTERVAL@|$progress|g" \
        -e "s|@MATERIAL@|$g4_material|g" \
        -e "s|@WORLD_HALF_LENGTH_MM@|$world_half|g" \
        -e "s|@PHANTOM_HALF_LENGTH_MM@|$half_length|g" \
        -e "s|@SOURCE_Z_MM@|$source_z|g" \
        -e "s|@HISTORIES@|$histories|g" \
        -e "s|@OUTPUT_STEM@|$stem|g" "$TEMPLATE" >"$destination"
    if [[ "$material" == schneider_section07 ]]; then
        sed -i 's/^#SECTION07# //' "$destination"
    fi
}

mkdir -p "$WORK_ROOT" "$PACKAGE_ROOT"
for material in "${MATERIALS[@]}"; do
    case "$material" in water|soft_tissue|lung|bone|schneider_section07) ;;
        *) die "unsupported material: $material" ;;
    esac
    case "$material" in
        water|soft_tissue) phantom_half_length=1000 ;;
        lung) phantom_half_length=1000 ;;
        bone) phantom_half_length=350 ;;
        schneider_section07) phantom_half_length=1000 ;;
    esac
    history_tag="$HISTORIES"
    if ((HISTORIES % 1000 == 0)); then
        history_tag="$((HISTORIES / 1000))k"
    fi
    tag="topas_400MeVu_${material}_inclxx_${history_tag}"
    run_dir="$WORK_ROOT/$tag"
    mkdir -p "$run_dir"
    stem="${tag}_cascade"
    config="$run_dir/run.txt"
    log="$run_dir/topas.log"
    render "$material" "$HISTORIES" "$config" "$stem"
    if [[ "$MODE" == dry-run ]]; then
        printf '[dry-run] %s: cd %q && %q run.txt > topas.log 2>&1\n' \
            "$material" "$run_dir" "$TOPAS_BIN"
        continue
    fi

    if [[ "$MODE" == run ]]; then
        echo "[$material] TOPAS: histories=$HISTORIES threads=$THREADS"
        (cd "$run_dir" && { time -p "$TOPAS_BIN" run.txt; } >topas.log 2>&1)
    else
        echo "[$material] compile-only: reusing $run_dir/${stem}.{header,phsp}"
        [[ -s "$run_dir/${stem}.header" ]] || die "missing TOPAS header"
        [[ -s "$run_dir/${stem}.phsp" ]] || die "missing TOPAS phsp"
        [[ -s "$log" ]] || die "missing TOPAS log"
    fi
    interactions="$run_dir/${tag}_interactions.csv.gz"
    products="$run_dir/${tag}_products.csv.gz"
    cascade_metadata="$PACKAGE_ROOT/${tag}.metadata.json"
    python3 "$TOOLS/prepare_topas_cascade.py" \
        --case "$tag" --histories "$HISTORIES" --input-dir "$run_dir" \
        --stem "$stem" --log-name topas.log \
        --phantom-half-length-mm "$phantom_half_length" \
        --interactions-output "$interactions" --products-output "$products" \
        --metadata "$cascade_metadata"

    python3 "$TOOLS/compile_cascade_package.py" \
        --metadata "$cascade_metadata" --interactions "$interactions" \
        --products "$products" \
        --output "$PACKAGE_ROOT/${tag}_cascade_3d.bin" \
        --output-metadata "$PACKAGE_ROOT/${tag}_cascade_3d.compiled.json"

    reactions="$run_dir/${tag}_primary_reactions.csv.gz"
    secondaries="$run_dir/${tag}_primary_secondaries.csv.gz"
    primary_metadata="$PACKAGE_ROOT/${tag}_primary.metadata.json"
    python3 "$TOOLS/prepare_primary_reactions_from_cascade.py" \
        --cascade-metadata "$cascade_metadata" --interactions "$interactions" \
        --products "$products" --reactions-output "$reactions" \
        --secondaries-output "$secondaries" --metadata-output "$primary_metadata"
    python3 "$TOOLS/compile_reaction_package.py" \
        --metadata "$primary_metadata" --reactions "$reactions" \
        --secondaries "$secondaries" \
        --output "$PACKAGE_ROOT/${tag}_primary_3d.bin" \
        --output-metadata "$PACKAGE_ROOT/${tag}_primary_3d.compiled.json" \
        --energy-bin-min-mevu 0 --energy-bin-width-mevu 4 --energy-bin-count 101
    echo "[$material] packages complete: $PACKAGE_ROOT/${tag}_{primary,cascade}_3d.bin"
done
