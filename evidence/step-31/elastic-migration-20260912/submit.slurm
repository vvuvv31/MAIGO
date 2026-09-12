#!/bin/bash
#SBATCH --job-name=elastic_RT07575_20022516
#SBATCH --partition=cpup6
#SBATCH --nodes=5
#SBATCH --exclude=cpu195,cpu196
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --output=/LustreData6/home/wuwei/maigo_elastic_ct_20260912/batch_%j.log
#SBATCH --error=/LustreData6/home/wuwei/maigo_elastic_ct_20260912/batch_%j.err
set -euo pipefail
source /LustreData6/home/wuwei/maigo_elastic_ct_20260912/env.sh
python3 /LustreData6/home/wuwei/maigo_elastic_ct_20260912/verify_migration.py
srun --nodes=5 --ntasks=5 --ntasks-per-node=1 --cpus-per-task=56 bash /LustreData6/home/wuwei/maigo_elastic_ct_20260912/run_node.sh
