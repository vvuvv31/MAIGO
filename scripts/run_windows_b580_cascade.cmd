@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
"%~dp0..\build\oneapi-windows-release\carbon_mc.exe" --config "%~dp0..\config\beam_200MeVu_fragment_cascade_aligned_100k.yaml" --device gpu %*
