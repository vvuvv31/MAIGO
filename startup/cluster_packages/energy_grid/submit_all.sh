#!/bin/bash
# Submit the 100–400 MeV/u, 50 MeV/u step, 1M array from this directory.
set -euo pipefail
cd "$(dirname "$0")"
chmod +x submit_water_1M_energy.slurm
sbatch submit_water_1M_energy.slurm
echo "Monitor: squeue -u \$USER"
echo "Logs:    slurm_c12egrid_<arrayjob>_<0-6>.out"
echo "Packages: $(pwd)/packages/topas_*MeVu_water_inclxx_1M_*.bin"
