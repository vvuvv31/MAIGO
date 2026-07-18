@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
set "ROOT=%~dp0.."
set "EXE=%ROOT%\build\oneapi-windows-release\carbon_mc.exe"
set "OUT=%ROOT%\out\sobp_benchmark"
if not exist "%EXE%" (
  echo Missing %EXE% 1>&2
  exit /b 1
)
if not exist "%OUT%" mkdir "%OUT%"

pushd "%ROOT%"
echo === CarbonGPU 10M SOBP benchmark ===
python validation\scripts\run_sobp_gpu_benchmark.py ^
  --project-root "%ROOT%" ^
  --executable "%EXE%" ^
  --config config\beam_sobp_water_3cm_5_10cm_3mm.yaml ^
  --log "%OUT%\gpu_sobp_benchmark.log"
if errorlevel 1 (
  popd
  exit /b 1
)

python validation\scripts\sparse_dose_to_mhd.py ^
  "%OUT%\gpu_sobp_voxels_Gy.csv" "%OUT%\gpu_sobp_dose.mhd" ^
  --input-format gpu --shape 50 50 50 --spacing-mm 3 3 3 ^
  --origin-mm -73.5 -73.5 1.5 --units Gy/primary
if errorlevel 1 (
  popd
  exit /b 1
)
popd

echo Wrote %OUT%\gpu_sobp_dose.mhd and .raw
exit /b 0
