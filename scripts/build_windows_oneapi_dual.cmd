@echo off
setlocal

REM Dual-target Windows build: Intel SPIR-V (Arc B580) + NVIDIA CUDA.
REM Requires a CUDA toolkit visible to icx-cl (nvptx64-nvidia-cuda target).

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

where nvcc >nul 2>&1
if errorlevel 1 (
    echo nvcc not found. Install CUDA toolkit or use scripts\build_windows_oneapi.cmd for Arc-only. 1>&2
    exit /b 1
)

cmake --fresh --preset oneapi-windows-dual-release
if errorlevel 1 exit /b 1
cmake --build --preset oneapi-windows-dual-release
if errorlevel 1 exit /b 1
ctest --preset oneapi-windows-dual-release
if errorlevel 1 exit /b 1
