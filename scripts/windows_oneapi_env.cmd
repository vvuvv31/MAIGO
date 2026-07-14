@echo off
set "PROGRAM_FILES_X86=%ProgramFiles(x86)%"

if defined VSCMD_VER goto visual_studio_ready
set "VSWHERE=%PROGRAM_FILES_X86%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Visual Studio Installer vswhere.exe was not found. 1>&2
    exit /b 1
)
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%I"
if not defined VS_INSTALL (
    echo Visual Studio C++ build tools were not found. 1>&2
    exit /b 1
)
call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

:visual_studio_ready

if not defined ONEAPI_SETVARS (
    if exist "%PROGRAM_FILES_X86%\Intel\oneAPI\setvars.bat" set "ONEAPI_SETVARS=%PROGRAM_FILES_X86%\Intel\oneAPI\setvars.bat"
)
if not defined ONEAPI_SETVARS (
    if exist "D:\Program Files (x86)\Intel\oneAPI\setvars.bat" set "ONEAPI_SETVARS=D:\Program Files (x86)\Intel\oneAPI\setvars.bat"
)
if not defined ONEAPI_SETVARS (
    echo Set ONEAPI_SETVARS to the full path of Intel oneAPI setvars.bat. 1>&2
    exit /b 1
)

call "%ONEAPI_SETVARS%"
if errorlevel 1 exit /b 1
