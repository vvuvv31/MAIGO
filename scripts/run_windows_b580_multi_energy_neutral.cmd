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
if not exist "%~dp0..\out\multi_energy_neutral" mkdir "%~dp0..\out\multi_energy_neutral"

for %%E in (100 150 200 250 300 350 400) do (
  echo === GPU %%E MeV/u + mode-D neutral ===
  "%EXE%" --config "%~dp0..\config\beam_%%EMeVu_multi_energy_neutral.yaml" --device gpu
  if errorlevel 1 exit /b 1
)

echo Done multi-energy neutral suite.
exit /b 0
