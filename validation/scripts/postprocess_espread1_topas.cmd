@echo off
setlocal EnableExtensions
REM Normalize remote TOPAS 1%% energy-spread IDDs after scp into validation\topas\output\

set "PY=python"
set "OUTDIR=validation\results\espread1"
mkdir %OUTDIR% 2>nul

for %%E in (100 150 200 250 300 350 400) do (
  if exist "validation\topas\output\e%%E_espread1_development_energy_deposit.csv" (
    echo Postprocess e=%%E
    %PY% validation\scripts\prepare_topas_multi_energy_idd.py ^
      --energy-mevu %%E --histories 100000 ^
      --raw-csv validation\topas\output\e%%E_espread1_development_energy_deposit.csv ^
      --output-csv %OUTDIR%\topas_e%%E_idd.csv ^
      --metadata %OUTDIR%\topas_e%%E.metadata.json ^
      --log validation\topas\output\e%%E-espread1-development_topas.log ^
      --parameter-file validation/topas/carbon_%%EMeVu_water_espread1_development_remote.txt
    if errorlevel 1 exit /b 1
  ) else (
    echo Missing raw CSV for e=%%E
  )
)

echo.
echo Compare when GPU + TOPAS both ready:
echo python validation\scripts\compare_espread1_suite.py
endlocal
