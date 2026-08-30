#!/bin/bash
#SBATCH --job-name=fredthin95
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err

set -euo pipefail

run_dir=/mnt/sda/wuwei/fred_thin_95_topas_1M
cd "$run_dir"
/home/wuwei/topas/topas-build/topas fred_thin_95_topas.txt
