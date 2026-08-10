#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

histories="${CT_LAYER_HISTORIES:-200000}"
for energy in 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240; do
  out="out/ct/tuning/energy_layers_${histories}/${energy}MeVu/dose.mhd"
  echo "energy=${energy} MeV/u histories=${histories} output=${out}"
  build/oneapi-release/carbon_mc \
    --config config/beam_ct_full_plan_reduced.yaml \
    --device cuda \
    --histories "$histories" \
    --spot-weights "out/ct/tuning/energy_layer_weights/weights_${energy}MeVu.csv" \
    --voxel-dose-mhd "$out"
done
