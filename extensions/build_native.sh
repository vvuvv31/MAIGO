#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: build_native.sh --build-dir DIR --geant4-dir DIR [--jobs N] [--cmake-arg ARG]
Build the Geant4 11.3.2 material metadata, CT/Cu/air Urban and air ionization exporters.
Geant4_DIR must contain Geant4Config.cmake. Source geant4.sh before extraction.
EOF
}
build_dir= geant4_dir= jobs=4
extra=()
while (($#)); do
  case "$1" in
    --build-dir|--geant4-dir|--jobs|--cmake-arg)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --build-dir) build_dir=$2;;
        --geant4-dir) geant4_dir=$2;;
        --jobs) jobs=$2;;
        --cmake-arg) extra+=("$2");;
      esac
      shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n $build_dir && -f $geant4_dir/Geant4Config.cmake && $jobs =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cmake -S "$here/native" -B "$build_dir" -DGeant4_DIR="$geant4_dir" "${extra[@]}"
cmake --build "$build_dir" --parallel "$jobs"
