@echo off
setlocal

call "%~dp0windows_oneapi_env.cmd"
if errorlevel 1 exit /b 1

cmake --fresh --preset oneapi-windows-release
if errorlevel 1 exit /b 1
cmake --build --preset oneapi-windows-release
if errorlevel 1 exit /b 1
ctest --preset oneapi-windows-release
if errorlevel 1 exit /b 1
