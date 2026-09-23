#!/bin/bash
set -eo pipefail
source /software/env_topas.sh
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAIGO_SYSTEMD_UNIT=maigo-field3cm-64m-20260923
exec /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/analysis_venv/bin/python -u /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_64M_20260923/run_64m.py
