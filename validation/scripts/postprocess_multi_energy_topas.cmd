@echo off
setlocal EnableExtensions
REM Normalize remote multi-energy TOPAS total-IDD outputs after scp/rsync.
REM Expect raw CSVs under validation\topas\output\

set "PY=python"
if not exist "validation\topas\output\e100_development_energy_deposit.csv" (
  echo Missing e100_development_energy_deposit.csv under validation\topas\output\
  exit /b 1
)

%PY% validation\scripts\prepare_topas_multi_energy_idd.py ^
  --energy-mevu 100 --histories 100000 ^
  --raw-csv validation\topas\output\e100_development_energy_deposit.csv ^
  --output-csv validation\results\topas_100MeVu_development.csv ^
  --metadata validation\results\topas_100MeVu_development.metadata.json ^
  --log validation\topas\output\e100-development_topas.log ^
  --parameter-file validation/topas/carbon_100MeVu_water_development_remote.txt
if errorlevel 1 exit /b 1

%PY% validation\scripts\prepare_topas_multi_energy_idd.py ^
  --energy-mevu 300 --histories 100000 ^
  --raw-csv validation\topas\output\e300_development_energy_deposit.csv ^
  --output-csv validation\results\topas_300MeVu_development.csv ^
  --metadata validation\results\topas_300MeVu_development.metadata.json ^
  --log validation\topas\output\e300-development_topas.log ^
  --parameter-file validation/topas/carbon_300MeVu_water_development_remote.txt
if errorlevel 1 exit /b 1

%PY% validation\scripts\prepare_topas_multi_energy_idd.py ^
  --energy-mevu 400 --histories 100000 ^
  --raw-csv validation\topas\output\e400_development_energy_deposit.csv ^
  --output-csv validation\results\topas_400MeVu_development.csv ^
  --metadata validation\results\topas_400MeVu_development.metadata.json ^
  --log validation\topas\output\e400-development_topas.log ^
  --parameter-file validation/topas/carbon_400MeVu_water_development_remote.txt
if errorlevel 1 exit /b 1

echo.
echo Re-run multi-energy GPU vs TOPAS analysis:
echo python validation\scripts\analyze_multi_energy.py --cases 100=out\multi_energy\e100_idd.csv 200=out\multi_energy\e200_idd.csv 300=out\multi_energy\e300_idd.csv 400=out\multi_energy\e400_idd.csv --topas-200 validation\results\topas_200MeVu_development.csv --output-metrics validation\results\windows_b580_multi_energy_100_400.metrics.json --output-plot validation\results\windows_b580_multi_energy_100_400.png
echo (extend analyzer for per-energy TOPAS paths after results land)
endlocal
