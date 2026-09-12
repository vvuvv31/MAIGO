#!/bin/bash
set -euo pipefail
source /LustreData6/home/wuwei/maigo_elastic_ct_20260912/env.sh
shard=$(printf 'shard_%02d' "$((SLURM_PROCID+1))")
for case in RT07575 20022516; do
 cd "/LustreData6/home/wuwei/maigo_elastic_ct_20260912/$case/$shard"
 hostname > node.txt
 "/LustreData6/home/wuwei/maigo_elastic_ct_20260912/build/topas" run.txt > topas.log 2>topas.err
done
