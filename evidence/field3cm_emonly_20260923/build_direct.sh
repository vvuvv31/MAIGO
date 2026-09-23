#!/bin/bash
set -eo pipefail
source /software/env_topas.sh
cmake -S /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/OpenTOPAS-4.2.3 -B /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/topas_build -DGeant4_DIR=/software/geant4-11.3.2/lib/cmake/Geant4 -DCMAKE_PREFIX_PATH=/software/topas -DTOPAS_EXTENSIONS_DIR=/mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/extensions -DTOPAS_MT=ON -DTOPAS_USE_QT=ON -DTOPAS_INSTALL_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_AUTOGEN_PARALLEL=8
cmake --build /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/topas_build --target topas -j 32
sha256sum /mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/field3cm_emonly_20260923/topas_build/topas
