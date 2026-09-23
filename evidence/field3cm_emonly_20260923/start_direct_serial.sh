#!/bin/bash
set -eo pipefail
source /software/env_topas.sh
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAIGO_SYSTEMD_UNIT=maigo-field3cm-20260923
exec /usr/bin/python3 -u /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/run_serial.py
