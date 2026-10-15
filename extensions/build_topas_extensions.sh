#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: build_topas_extensions.sh --topas-source DIR --build-dir DIR --profile NAME [options]
Profiles:
  nuclear-copper  Nuclear capture, deterministic rates and material tables
  unified-em      Native ion EM vectors and distributions (separate implementation)
  elastic-carbon  Frozen production carbon/reference all-ion elastic exporter
  recoil-carbon   Frozen production carbon/reference recoil stopping exporter
  elastic-proton  Current combined exporter for proton Water_75eV replacement rows
Options:
  --geant4-dir DIR    Directory containing Geant4Config.cmake
  --cmake-prefix DIR CMAKE_PREFIX_PATH for TOPAS dependencies
  --topas-use-qt BOOL Default ON, matching the pinned TOPAS stack
  --jobs N           Default 4
  --cmake-arg ARG    Additional CMake argument, repeatable
  --stage-only       Prepare sources without configuring/building TOPAS
Sources are staged below the build directory. TOPAS source is never modified.
EOF
}
topas_source= build_dir= profile= geant4_dir= cmake_prefix= jobs=4 topas_use_qt=ON stage_only=0
extra=()
while (($#)); do
  case "$1" in
    --stage-only) stage_only=1; shift;;
    -h|--help) usage; exit 0;;
    --topas-source|--build-dir|--profile|--geant4-dir|--cmake-prefix|--jobs|--topas-use-qt|--cmake-arg)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --topas-source) topas_source=$2;; --build-dir) build_dir=$2;;
        --profile) profile=$2;; --geant4-dir) geant4_dir=$2;;
        --cmake-prefix) cmake_prefix=$2;; --jobs) jobs=$2;;
        --topas-use-qt) topas_use_qt=$2;; --cmake-arg) extra+=("$2");;
      esac
      shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n $build_dir && -n $profile && $jobs =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
case "$profile" in nuclear-copper|unified-em|elastic-carbon|recoil-carbon|elastic-proton ) ;;
  *) echo "Unknown profile: $profile" >&2; exit 2;; esac
if (( !stage_only )); then
  [[ -f $topas_source/CMakeLists.txt ]] || { echo "Missing TOPAS source tree" >&2; exit 2; }
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd)
[[ $build_dir != "$here" && $build_dir != "$here/topas" ]] || { echo "Build outside extension sources" >&2; exit 2; }
stage=$build_dir/maigo-extension-source
marker=$build_dir/maigo-profile.txt
if [[ -f $marker && $(cat -- "$marker") != "$profile" ]]; then
  echo "Build directory belongs to another profile; use a separate directory" >&2; exit 2
fi
mkdir -p -- "$stage"
printf '%s\n' "$profile" > "$marker"
case "$profile" in
  *) cp -- "$here"/topas/common/* "$stage/";;
esac
case "$profile" in
  unified-em)
    cp -- "$here/topas/unified_em/CarbonStoppingPowerNtuple.hh" "$stage/"
    { printf '%s\n' '// Scorer for CarbonStoppingPowerNtuple'; cat -- "$here/topas/unified_em/CarbonStoppingPowerNtuple.cc"; } > "$stage/CarbonStoppingPowerNtuple.cc";;
  elastic-carbon|recoil-carbon|elastic-proton)
    cp -- "$here/topas/all_ion_elastic/AllIonElasticDump.hh" "$stage/"
    case "$profile" in
      elastic-carbon) source_file=frozen_production/AllIonElasticDump.cc;;
      recoil-carbon) source_file=frozen_production/AllIonElasticDump_recoil.cc;;
      elastic-proton) source_file=AllIonElasticDump.cc;;
    esac
    cp -- "$here/topas/all_ion_elastic/$source_file" "$stage/AllIonElasticDump.cc";;
esac
if ((stage_only)); then printf 'Staged %s: %s\n' "$profile" "$stage"; exit 0; fi
args=(-DTOPAS_EXTENSIONS_DIR="$stage" -DTOPAS_USE_QT="$topas_use_qt"
      "-DCMAKE_CXX_FLAGS=-I$here/../include")
[[ -z $geant4_dir ]] || args+=(-DGeant4_DIR="$geant4_dir")
[[ -z $cmake_prefix ]] || args+=(-DCMAKE_PREFIX_PATH="$cmake_prefix")
cmake -S "$topas_source" -B "$build_dir" "${args[@]}" "${extra[@]}"
cmake --build "$build_dir" --parallel "$jobs"
