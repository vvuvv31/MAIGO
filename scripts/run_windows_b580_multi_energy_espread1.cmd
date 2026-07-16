@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

call "%~dp0windows_oneapi_env.cmd" || exit /b 1
set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
REM Hard VRAM cap (kill process if dedicated usage exceeds this fraction of device mem).
if not defined CARBON_GPU_MAX_FRACTION set "CARBON_GPU_MAX_FRACTION=0.50"
if not defined CARBON_GPU_MIB set "CARBON_GPU_MIB=11869"

if not exist "build\oneapi-windows-release-grok\carbon_mc.exe" (
  echo Building carbon_mc...
  cmake --build build\oneapi-windows-release-grok --config Release --target carbon_mc -j 8 || exit /b 1
)

mkdir out\multi_energy_espread1 2>nul
mkdir validation\results\espread1 2>nul

echo Running espread1 suite under GPU memory guard (%CARBON_GPU_MAX_FRACTION% of %CARBON_GPU_MIB% MiB)...
python -u validation\scripts\_run_espread1_gpu_suite.py || exit /b 1

echo Done. See validation\results\espread1\
endlocal
