#!/bin/bash
#SBATCH --job-name=elastic_migration_probe
#SBATCH --partition=cpup6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/LustreData6/home/wuwei/maigo_elastic_ct_20260912/probe_%j.log
#SBATCH --error=/LustreData6/home/wuwei/maigo_elastic_ct_20260912/probe_%j.err
set -euo pipefail
source /LustreData6/home/wuwei/maigo_elastic_ct_20260912/env.sh
cd /LustreData6/home/wuwei/maigo_elastic_ct_20260912
/LustreData6/home/wuwei/maigo_elastic_ct_20260912/build/topas audit.txt
