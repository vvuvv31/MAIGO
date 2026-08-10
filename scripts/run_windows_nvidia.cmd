@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

REM Prefer a dual-target Windows build when present; otherwise use the Arc
REM SPIR-V build only if CUDA kernels were not required at compile time.
set "EXE="
if exist "%~dp0..\build\oneapi-windows-dual-release\carbon_mc.exe" (
    set "EXE=%~dp0..\build\oneapi-windows-dual-release\carbon_mc.exe"
) else if exist "%~dp0..\build\oneapi-windows-release\carbon_mc.exe" (
    set "EXE=%~dp0..\build\oneapi-windows-release\carbon_mc.exe"
)

if not defined EXE (
    echo No Windows SYCL binary found. Build with: 1>&2
    echo   cmake --fresh --preset oneapi-windows-dual-release 1>&2
    echo   cmake --build --preset oneapi-windows-dual-release 1>&2
    echo ^(requires CUDA toolkit for nvptx target^) 1>&2
    exit /b 1
)

if not defined ONEAPI_DEVICE_SELECTOR set "ONEAPI_DEVICE_SELECTOR=cuda:gpu"

echo Executable: %EXE%
echo ONEAPI_DEVICE_SELECTOR=%ONEAPI_DEVICE_SELECTOR%
sycl-ls

"%EXE%" --config "%~dp0..\config\beam_200MeVu_attenuation.yaml" --device cuda --output "%~dp0..\out\windows_nvidia_attenuation_depth_dose.csv" %*
