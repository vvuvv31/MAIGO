@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
set "EXE=%~dp0..\build\oneapi-windows-release\carbon_mc.exe"
if not exist "%EXE%" (
  echo Missing %EXE% 1>&2
  exit /b 1
)

if not exist "%~dp0..\out\multi_energy" mkdir "%~dp0..\out\multi_energy"

echo === GPU 150 MeV/u ===
"%EXE%" --config "%~dp0..\config\beam_150MeVu_multi_energy.yaml" --device gpu
if errorlevel 1 exit /b 1

echo === GPU 250 MeV/u ===
"%EXE%" --config "%~dp0..\config\beam_250MeVu_multi_energy.yaml" --device gpu
if errorlevel 1 exit /b 1

echo === GPU 350 MeV/u ===
"%EXE%" --config "%~dp0..\config\beam_350MeVu_multi_energy.yaml" --device gpu
if errorlevel 1 exit /b 1

echo Done mid-energy GPU runs.
exit /b 0
