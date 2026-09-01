#!/usr/bin/env bash
#SBATCH --job-name=cinhybp3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-101%3
#SBATCH --output=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/v3_%A_%a.out
#SBATCH --error=/mnt/sda/wuwei/cinel02-hybrid-pilot/logs/v3_%A_%a.err
set -euo pipefail
exec /mnt/sda/wuwei/cinel02-hybrid-pilot/run_cinel02_hybrid_pilot_worker_v3.sh "${SLURM_ARRAY_TASK_ID}"
