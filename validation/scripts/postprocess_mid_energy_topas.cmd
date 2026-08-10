@echo off
setlocal EnableExtensions
set PY=python

for %%E in (150 250 350) do (
  echo === postprocess e%%E ===
  %PY% validation\scripts\prepare_topas_multi_energy_idd.py ^
    --energy-mevu %%E --histories 100000 ^
    --raw-csv validation\topas\output\e%%E_development_energy_deposit.csv ^
    --output-csv validation\results\topas_%%EMeVu_development.csv ^
    --metadata validation\results\topas_%%EMeVu_development.metadata.json ^
    --log validation\topas\output\e%%E-development_topas.log ^
    --parameter-file validation/topas/carbon_%%EMeVu_water_development_remote.txt
  if errorlevel 1 exit /b 1
)

echo Done.
exit /b 0
