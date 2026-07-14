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

Available total-IDD cases are `smoke` (100 histories), `development` (10,000), and `reference` (1,000,000). The runner saves the full TOPAS/Geant4 console output as `output/<case>_topas.log`.

`em-smoke` and `em-development` use only `g4em-standard_opt4`. They isolate electromagnetic stopping and energy-loss fluctuations from nuclear attenuation and secondary fragments, and are the appropriate references for Level 1/2 validation.

## Species-resolved fragmentation baseline

The species cases score mutually exclusive, direct track energy deposition from primary C-12, secondary carbon, boron, beryllium, lithium, helium, and protons. Additional scorers split the legacy `other` residual into electron/positron, gamma, neutron, deuteron, triton, and a remaining unclassified category. Alpha and He-3 scorers also partition the broad helium category without changing the legacy columns used by GPU comparisons. The postprocessor verifies both the legacy and detailed bin-by-bin energy closures.

Run the 100-history syntax/filter smoke test:

```bash
TOPAS_EXECUTABLE="$HOME/Applications/TOPAS/OpenTOPAS-install/bin/topas" \
TOPAS_G4_DATA_DIR="$HOME/Applications/GEANT4/G4DATA" \
./validation/topas/run_topas.sh species-smoke

python3 validation/scripts/prepare_topas_species.py \
  --case smoke --histories 100 \
  --output-csv validation/results/topas_200MeVu_species_smoke.csv \
  --metadata validation/results/topas_200MeVu_species_smoke.metadata.json \
  --plot validation/results/topas_200MeVu_species_smoke.png
```

After smoke QA, build the 100,000-history calibration baseline by replacing `species-smoke`, `smoke`, and `100` above with `species-development`, `development`, and `100000`. Use `species-reference` for the one-million-history promotion run. The random seed remains fixed at `20260714` in the base parameter file.

The species curves represent energy deposited directly on each particle track. Gamma and neutron tracks usually deposit little or no energy directly: energy transferred to recoil ions or electrons is attributed to those charged descendant tracks. The explicit gamma/neutron columns therefore do not represent ancestor-attributed neutral dose. This definition is recorded in metadata and must remain fixed when calibrating the GPU fragmentation model; ancestor-attributed neutral dose requires a separate provenance scorer.

The metadata also records SHA-256 hashes for every raw scorer and the TOPAS log, the detected Geant4 version and elapsed wall time, integrated species fractions, and species fractions in the tail beginning at 90 mm. Change the analysis boundary with `--tail-start-mm` only when a different boundary is recorded for the comparison.

## Direct cross-section and reaction-final-state extraction

The attenuation probability and the fragmentation final state are extracted separately from the same TOPAS/Geant4 physics configuration:

- `CarbonCrossSectionNtuple` directly queries `G4HadronicProcessStore` for C-12+H, C-12+O, and Water_75eV inelastic cross sections from 1 to 400 MeV/u. This table determines the reaction distance in the GPU transport.
- `CarbonReactionNtuple` records the primary C-12 inelastic reaction header and every direct secondary in the same event, including incident energy, vertex, particle identity, kinetic energy, direction, creator process, and model ID. Rows with one `reaction_id` must be sampled jointly.
- the OriginCount and species-resolved IDD scorers are independent QA observables; they must not replace correlated event-level final-state sampling.

Build the extension-enabled TOPAS executable in WSL (all generated source/build/install files remain under the ignored project `build/` directory):

```bash
bash validation/topas/build_extensions.sh
```

Extract and standardize the cross-section table:

```bash
export TOPAS_EXECUTABLE="$PWD/build/opentopas-extension-install/bin/topas"
export TOPAS_G4_DATA_DIR="$HOME/Applications/GEANT4/G4DATA"
./validation/topas/run_topas.sh cross-sections
python3 validation/scripts/prepare_topas_cross_sections.py
```

Run and validate a 100-history reaction-final-state smoke case:

```bash
./validation/topas/run_topas.sh fragment-smoke
python3 validation/scripts/prepare_topas_reactions.py \
  --case smoke --histories 100 \
  --reactions-output validation/topas/output/fragment_smoke_reactions.csv.gz \
  --secondaries-output validation/topas/output/fragment_smoke_secondaries.csv.gz \
  --metadata validation/topas/output/fragment_smoke_reaction_sampling.metadata.json
```

Use `fragment-development` with 100,000 histories for the calibration dataset and `fragment-reference` for the one-million-history promotion run. The compact outputs separate one-row-per-reaction headers from secondaries, link them through `reaction_id` plus a zero-based offset/count pair, and support deterministic gzip compression. The standardizer converts TOPAS global `z=-200...200 mm` to water depth `0...400 mm`, checks the header history/entry counts, verifies that every secondary has exactly one primary-reaction header, and validates reaction vertices, incident energies, and direction normalization. `--output-csv` remains available only when a redundant flat QA table is useful.

The completed 100,000-history development baseline contains 37,657 primary C-12 inelastic reactions and 330,659 direct secondaries (mean multiplicity 8.781). Two low-energy reaction headers have no direct visible secondary row; they are valid zero-length packages and remain in the reaction table with `secondary_count=0`. The parser rejects orphan secondaries, but it must not discard these empty packages because doing so would bias the reaction probability. The two gzip tables and their metadata are stored in `validation/results/`.

Compile the validated gzip tables into the fixed-layout, little-endian runtime table with:

```bash
python3 validation/scripts/compile_reaction_package.py \
  --metadata validation/results/topas_200MeVu_reaction_packages_development.metadata.json \
  --reactions validation/results/topas_200MeVu_reactions_development.csv.gz \
  --secondaries validation/results/topas_200MeVu_secondaries_development.csv.gz \
  --output validation/results/topas_200MeVu_reaction_packages_development.bin \
  --output-metadata validation/results/topas_200MeVu_reaction_packages_development.binary.metadata.json
```

The compiler verifies the source hashes and row counts, reaction/secondary closure and per-reaction ordering before reorganizing complete packages into 201 one-MeV/u bins. It rejects empty runtime bins instead of silently sampling a different energy. The C++ loader independently validates the magic/version, record sizes, total file size, energy-bin coverage, secondary offsets and physical value ranges.

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
