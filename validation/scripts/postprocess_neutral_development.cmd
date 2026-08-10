@echo off
setlocal EnableExtensions
set PY=python

echo === prepare_topas_neutral development ===
%PY% validation\scripts\prepare_topas_neutral.py ^
  --header validation\topas\output\neutral_development_interactions.header ^
  --phsp validation\topas\output\neutral_development_interactions.phsp ^
  --interactions-output validation\results\topas_200MeVu_neutral_development_interactions.csv.gz ^
  --products-output validation\results\topas_200MeVu_neutral_development_products.csv.gz ^
  --metadata validation\results\topas_200MeVu_neutral_development.metadata.json ^
  --case development
if errorlevel 1 exit /b 1

echo === compile_neutral_package ===
%PY% validation\scripts\compile_neutral_package.py ^
  --metadata validation\results\topas_200MeVu_neutral_development.metadata.json ^
  --interactions validation\results\topas_200MeVu_neutral_development_interactions.csv.gz ^
  --products validation\results\topas_200MeVu_neutral_development_products.csv.gz ^
  --output validation\results\topas_200MeVu_neutral_development.bin ^
  --output-metadata validation\results\topas_200MeVu_neutral_development.compiled.json
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
