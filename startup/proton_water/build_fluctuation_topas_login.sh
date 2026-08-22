#!/usr/bin/env bash
# Run on the cluster login node, where the Qt5 CMake packages are available.
set -euo pipefail

ROOT="${FLUCTUATION_ROOT:-/LustreData6/home/wuwei/maigo_fluctuation_proton_100k}"
SHARE=/ShareData1/opt/cascadelake/linux-centos7-cascadelake/gcc-9.4.0
export CC="${SHARE}/gcc-11.3.0-hlrrmo2m5ycqa54vsmbhcpe64jmp5dwm/bin/gcc"
export CXX="${SHARE}/gcc-11.3.0-hlrrmo2m5ycqa54vsmbhcpe64jmp5dwm/bin/g++"
export PATH="${SHARE}/cmake-3.25.2-hn3pewmkxtlqnynxccmcy6mt3m3rcsa4/bin:${PATH}"
export Qt5Core_DIR=/usr/lib64/cmake/Qt5Core
export Qt5Gui_DIR=/usr/lib64/cmake/Qt5Gui
export Qt5Widgets_DIR=/usr/lib64/cmake/Qt5Widgets
export Qt5OpenGL_DIR=/usr/lib64/cmake/Qt5OpenGL
export CMAKE_PREFIX_PATH=/usr/lib64/cmake

[[ -f "$ROOT/extensions/EnergyLossFluctuationNtuple.cc" ]] || {
    echo "missing fluctuation extension sources under $ROOT/extensions" >&2
    exit 2
}

echo "[$(date -Is)] configure $ROOT/topas-build"
cmake -S /LustreData6/home/wuwei/Applications/TOPAS/OpenTOPAS \
      -B "$ROOT/topas-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_INSTALL_PREFIX="$ROOT/topas-install" \
      -DTOPAS_EXTENSIONS_DIR="$ROOT/extensions" \
      -DGeant4_DIR=/LustreData6/home/wuwei/Applications/GEANT4/geant4-install/lib64/cmake/Geant4 \
      -DGDCM_DIR=/LustreData6/home/wuwei/Applications/GDCM/gdcm-install/lib/gdcm-2.6 \
      -DQt5Core_DIR=/usr/lib64/cmake/Qt5Core \
      -DQt5Gui_DIR=/usr/lib64/cmake/Qt5Gui \
      -DQt5Widgets_DIR=/usr/lib64/cmake/Qt5Widgets \
      -DQt5OpenGL_DIR=/usr/lib64/cmake/Qt5OpenGL \
      -DTOPAS_MT=ON \
      -DTOPAS_TYPE=expanded \
      -DTOPAS_USE_QT=ON \
      -DTOPAS_USE_QT6=OFF \
      -DTOPAS_WITH_PTL_GEANT4=OFF \
      -DTOPAS_WITH_STATIC_GEANT4=OFF

echo "[$(date -Is)] build"
cmake --build "$ROOT/topas-build" --parallel 8
echo "[$(date -Is)] install"
cmake --install "$ROOT/topas-build"

BIN="$ROOT/topas-install/bin/topas"
[[ -x "$BIN" ]]
grep -a -q EnergyLossFluctuationNtuple "$BIN"
version="$($BIN --version)"
[[ "$version" == "4.2.p3" ]]
echo "[$(date -Is)] OK $BIN version=$version"
