@echo off
setlocal EnableExtensions
rem Post-process remote 400 MeV/u cascade n-tuple into GPU packages covering 0--400 MeV/u.
rem Prerequisites: fetched cascade_e400_development_reactions.{header,phsp} + topas log.

set PY=python
if exist "%ONEAPI_ROOT%\intelpython\python\python.exe" set PY=%ONEAPI_ROOT%\intelpython\python\python.exe

set CASE=e400_development
set HIST=100000
set STEM=cascade_e400_development_reactions
set LOG=cascade-e400-development_topas.log
set OUT=validation\results
set IN=validation\topas\output

echo === 1/4 prepare_topas_cascade ===
%PY% validation\scripts\prepare_topas_cascade.py ^
  --case %CASE% --histories %HIST% ^
  --stem %STEM% --log-name %LOG% ^
  --input-dir %IN% ^
  --interactions-output %OUT%\topas_400MeVu_cascade_100k_interactions.csv.gz ^
  --products-output %OUT%\topas_400MeVu_cascade_100k_products.csv.gz ^
  --metadata %OUT%\topas_400MeVu_cascade_100k.metadata.json
if errorlevel 1 exit /b 1

echo === 2/4 compile_cascade_package (3d / local directions) ===
%PY% validation\scripts\compile_cascade_package.py ^
  --metadata %OUT%\topas_400MeVu_cascade_100k.metadata.json ^
  --interactions %OUT%\topas_400MeVu_cascade_100k_interactions.csv.gz ^
  --products %OUT%\topas_400MeVu_cascade_100k_products.csv.gz ^
  --output %OUT%\topas_400MeVu_cascade_100k_3d.bin ^
  --output-metadata %OUT%\topas_400MeVu_cascade_100k_3d.compiled.json
if errorlevel 1 exit /b 1

echo === 3/4 prepare_primary_reactions_from_cascade ===
%PY% validation\scripts\prepare_primary_reactions_from_cascade.py ^
  --cascade-metadata %OUT%\topas_400MeVu_cascade_100k.metadata.json ^
  --runtime-reference-metadata %OUT%\topas_200MeVu_ancestor_dose_3d_development.metadata.json ^
  --interactions %OUT%\topas_400MeVu_cascade_100k_interactions.csv.gz ^
  --products %OUT%\topas_400MeVu_cascade_100k_products.csv.gz ^
  --reactions-output %OUT%\topas_400MeVu_cascade_aligned_primary_3d_reactions.csv.gz ^
  --secondaries-output %OUT%\topas_400MeVu_cascade_aligned_primary_3d_secondaries.csv.gz ^
  --metadata-output %OUT%\topas_400MeVu_cascade_aligned_primary_3d.metadata.json
if errorlevel 1 exit /b 1

echo === 4/4 compile_reaction_package (401 x 1 MeV/u bins, 0--400) ===
%PY% validation\scripts\compile_reaction_package.py ^
  --metadata %OUT%\topas_400MeVu_cascade_aligned_primary_3d.metadata.json ^
  --reactions %OUT%\topas_400MeVu_cascade_aligned_primary_3d_reactions.csv.gz ^
  --secondaries %OUT%\topas_400MeVu_cascade_aligned_primary_3d_secondaries.csv.gz ^
  --output %OUT%\topas_400MeVu_cascade_aligned_primary_3d.bin ^
  --output-metadata %OUT%\topas_400MeVu_cascade_aligned_primary_3d.compiled.json ^
  --energy-bin-min-mevu 0.0 ^
  --energy-bin-width-mevu 1.0 ^
  --energy-bin-count 401
if errorlevel 1 (
  echo.
  echo Primary 1 MeV/u grid had empty bins; retrying with 2 MeV/u width ^(201 bins, 0--402^).
  %PY% validation\scripts\compile_reaction_package.py ^
    --metadata %OUT%\topas_400MeVu_cascade_aligned_primary_3d.metadata.json ^
    --reactions %OUT%\topas_400MeVu_cascade_aligned_primary_3d_reactions.csv.gz ^
    --secondaries %OUT%\topas_400MeVu_cascade_aligned_primary_3d_secondaries.csv.gz ^
    --output %OUT%\topas_400MeVu_cascade_aligned_primary_3d.bin ^
    --output-metadata %OUT%\topas_400MeVu_cascade_aligned_primary_3d.compiled.json ^
    --energy-bin-min-mevu 0.0 ^
    --energy-bin-width-mevu 2.0 ^
    --energy-bin-count 201
  if errorlevel 1 exit /b 1
)

echo.
echo Done. Packages:
echo   cascade:  %OUT%\topas_400MeVu_cascade_100k_3d.bin
echo   primary:  %OUT%\topas_400MeVu_cascade_aligned_primary_3d.bin
echo Wire config\beam_{300,400}MeVu_multi_energy.yaml to these paths, then re-run GPU.
exit /b 0
