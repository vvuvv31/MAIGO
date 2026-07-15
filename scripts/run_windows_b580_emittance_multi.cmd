@echo off
setlocal
call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1
set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
set "EXE=%~dp0..\build\oneapi-windows-release\carbon_mc.exe"
if not exist "%~dp0..\out\emittance" mkdir "%~dp0..\out\emittance"
for %%E in (150 200 250 350 400) do (
  echo === GPU emittance %%E MeV/u ===
  "%EXE%" --config "%~dp0..\config\beam_%%EMeVu_emittance_sigma.yaml" --device gpu
  if errorlevel 1 exit /b 1
)
echo Done.
exit /b 0
