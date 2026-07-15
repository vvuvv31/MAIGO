@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

call "%~dp0windows_oneapi_env.cmd" || exit /b 1

if not exist "build\oneapi-windows-release-grok\carbon_mc.exe" (
  echo Building carbon_mc...
  cmake --build build\oneapi-windows-release-grok --config Release --target carbon_mc carbon_tests -j 8 || exit /b 1
)

REM Unit tests use SYCL "cpu" (OpenCL CPU); do not pin to Level Zero GPU here.
echo === CT unit tests ===
build\oneapi-windows-release-grok\carbon_tests.exe || exit /b 1
python validation\scripts\test_schneider_mass_sp.py || exit /b 1

if /I "%~1"=="--skip-gpu" (
  python validation\scripts\run_ct_baseline.py --skip-gpu || exit /b 1
  exit /b 0
)

if not exist "ct\grid\patient_ct.bin" (
  echo Preparing patient CT grid from ct\dicom + Schneider table...
  python validation\scripts\prepare_ct_grid.py --schneider-file ct\HUtoMaterialSchneider.txt || exit /b 1
)

REM Arc B580 GPU runs pin Level Zero.
set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
echo === CT GPU baseline primary + secondary + TOPAS compare ===
python validation\scripts\run_ct_baseline.py || exit /b 1
echo Done. See validation\results\ct\ct_baseline_summary.metrics.json
endlocal
