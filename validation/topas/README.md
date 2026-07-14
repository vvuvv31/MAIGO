# TOPAS 200 MeV/u C-12 reference

`carbon_200MeVu_water.txt` fixes the first benchmark case: 2400 MeV total kinetic energy for fully stripped C-12, a 400 mm long `Water_75eV` phantom, 0.5 mm depth bins, a monoenergetic pencil beam, and one million histories.

Run from WSL:

```bash
TOPAS_EXECUTABLE=/path/to/topas ./validation/topas/run_topas.sh development
```

Validate syntax and geometry first with only 100 histories:

```bash
cd validation/topas
./run_topas.sh smoke
```

Available cases are `smoke` (100 histories), `development` (10,000), and `reference` (1,000,000). The runner saves the full TOPAS/Geant4 console output as `output/<case>_topas.log`.

`em-smoke` and `em-development` use only `g4em-standard_opt4`. They isolate electromagnetic stopping and energy-loss fluctuations from nuclear attenuation and secondary fragments, and are the appropriate references for Level 1/2 validation.

If `topas` is already on PATH, omit `TOPAS_EXECUTABLE`. Raw files are written below `validation/topas/output/` and intentionally ignored by Git.

If TOPAS is available only through an interactive-shell alias, pass the real executable and Geant4 data directory explicitly, for example:

```bash
TOPAS_EXECUTABLE="$HOME/Applications/TOPAS/OpenTOPAS-install/bin/topas" \
TOPAS_G4_DATA_DIR="$HOME/Applications/GEANT4/G4DATA" \
./validation/topas/run_topas.sh development
```

Before accepting this as the reference, record the exact TOPAS and Geant4 versions printed by the local installation and archive the TOPAS console log. The parameter syntax follows the official TOPAS documentation for [ion particle names](https://topas.readthedocs.io/en/3.7.0/parameters/source/intro.html), [beam sources](https://topas.readthedocs.io/en/3.7.0/parameters/source/beam.html), [default ion-capable physics modules](https://topas.readthedocs.io/en/3.7.0/parameters/defaults.html), [step limits](https://topas.readthedocs.io/en/3.7.0/parameters/physics/misc.html), and [CSV scorer output](https://topas.readthedocs.io/en/3.7.0/parameters/scoring/output.html).

TOPAS CSV layout depends on the scorer report. Convert the energy-deposit `Sum` column to the project's standard schema after inspecting its header:

```bash
python3 validation/scripts/normalize_topas_csv.py \
  validation/topas/output/topas_energy_deposit.csv \
  validation/results/topas_200MeVu.csv \
  --histories 1000000 --z-column 2 --value-column 3
```

The column numbers above are zero-based examples; use the indices shown by the actual local TOPAS output header.
