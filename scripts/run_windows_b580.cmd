@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

set "ONEAPI_DEVICE_SELECTOR=level_zero:0"
"%~dp0..\build\oneapi-windows-release\carbon_mc.exe" --config "%~dp0..\config\beam_200MeVu_attenuation.yaml" --device gpu --output "%~dp0..\out\windows_b580_attenuation_depth_dose.csv" %*

