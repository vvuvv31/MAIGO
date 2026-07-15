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

echo === GPU 300 MeV/u with 400-package final states ===
"%EXE%" --config "%~dp0..\config\beam_300MeVu_multi_energy.yaml" --device gpu
if errorlevel 1 exit /b 1

echo === GPU 400 MeV/u with 400-package final states ===
"%EXE%" --config "%~dp0..\config\beam_400MeVu_multi_energy.yaml" --device gpu
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
