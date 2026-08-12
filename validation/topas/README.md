# TOPAS 200 MeV/u C-12 reference

`carbon_200MeVu_water.txt` fixes the first benchmark case: 2400 MeV total kinetic energy for fully stripped C-12, a 400 mm long `Water_75eV` phantom, 0.5 mm depth bins, a monoenergetic pencil beam, and one million histories.

New TOPAS jobs run on the dedicated Linux host `v@192.168.31.5`, not in WSL. The
remote working tree is `~/gpu`, and the host provides 56 logical CPUs. Windows
remains the native oneAPI/Arc B580 execution environment.

Build and run the remote extension after synchronizing the required files:

```bash
ssh v@192.168.31.5
cd ~/gpu
bash validation/topas/build_extensions_remote.sh
bash validation/topas/run_ancestor_remote.sh smoke
```

The formal 100,000-history job uses the explicit 56-thread overlay and should be
detached so that a Windows restart cannot terminate it:

```bash
cd ~/gpu
nohup bash validation/topas/run_ancestor_remote.sh development \
  > validation/topas/output/ancestor-development_nohup.log 2>&1 < /dev/null &
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

## Ancestor-attributed 3D dose baseline

`CarbonDoseOrigin` maintains an event-local `track_id -> origin` map beginning
before transport. Electron and positron dose inherits the charged parent origin;
each charged nuclear descendant is classified by its own Z/A; neutron, gamma,
and other-neutral lineages retain their neutral source. The 12
mutually exclusive categories are primary C-12, secondary carbon, B, Be, Li, He,
proton, other charged, neutron, gamma, other neutral, and unresolved.

The full 300 x 300 x 400 mm water phantom is scored on a `60 x 60 x 800` grid,
corresponding to `5 x 5 x 0.5 mm3` voxels. TOPAS writes sparse 3D CSV files. The
postprocessor packages a dense total plus sparse category arrays in NPZ, derives
an 800-bin IDD from the 3D dose, and checks both category closure and the
independent TOPAS `DoseToMedium` total.

The earlier TOPAS 4.1.p1 / Geant4 11.1.3 reference is retired. Regenerate this
case with TOPAS 4.2.p3 / Geant4 11.3.2, five seeds, 56 threads, and no more than
100,000 primaries per run:

```bash
cd ~/gpu
python3 validation/scripts/prepare_topas_ancestor_dose.py \
  --case development --histories 100000 \
  --seed 20260714 --execution-host vv \
  --output-npz validation/results/topas_200MeVu_ancestor_dose_3d_development.npz \
  --output-idd validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv \
  --metadata validation/results/topas_200MeVu_ancestor_dose_3d_development.metadata.json \
  --plot validation/results/topas_200MeVu_ancestor_dose_3d_development.png
```

The formal result closes at `6.395e-14 MeV/primary/bin`; the separately
accumulated TOPAS total differs by at most `7.882e-7 MeV/primary/bin`, unresolved
dose is zero, and no global scale is applied. Total wall time was 1022.56 s.

Compare the derived TOPAS IDD with the existing Windows B580 result using:

```bash
python3 validation/scripts/compare_ancestor_attributed_idd.py \
  validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv \
  validation/results/windows_b580_fragment_transport_species_100k.csv \
  --metrics-output validation/results/windows_b580_ancestor_attributed_100k_vs_topas.metrics.json \
  --plot validation/results/windows_b580_ancestor_attributed_100k_vs_topas.png
```

GPU `other` is compared only with TOPAS `other_charged`. Neutron/gamma lineage
dose remains separate because the current GPU model has no spatial neutral
transport. The current GPU output is IDD-only, so a GPU-vs-TOPAS 3D spatial
comparison is not yet available.

## Multi-energy total IDD (100 / 300 / 400 MeV/u)

Standard EnergyDeposit/DoseToMedium scorers only (no extension rebuild required).
Parameter overlays override `BeamEnergy` from the 200 MeV/u base water phantom.

On `v@192.168.31.5`:

```bash
bash validation/topas/run_multi_energy_idd_remote.sh all smoke
nohup bash validation/topas/run_multi_energy_idd_remote.sh 100 development \
  > validation/topas/output/e100-development_nohup.log 2>&1 < /dev/null &
# likewise for 300 and 400
```

Full procedure: `validation/topas/REMOTE_MULTI_ENERGY_IDD.md`.  
After pulling CSVs to Windows, run `validation/scripts/postprocess_multi_energy_topas.cmd`.

## Neutron/gamma interaction package

`CarbonNeutralNtuple` records every scored neutron or gamma interaction in water:
incident and continuation kinematics, local energy deposit, macroscopic total
cross section, and correlated direct products. The stable key under MT output is
`(run, thread, event, interaction_sequence)`.

The accepted package is the 100k development table generated with TOPAS 4.2.p3
and Geant4 11.3.2. It may be generated locally or on the server, but the runtime
log is mandatory and older Geant4 versions are rejected:

```bash
bash validation/topas/build_extensions_remote.sh
bash validation/topas/run_neutral_remote.sh development
```

Standardize on Windows or any host that has the raw header/phsp:

```bash
python3 validation/scripts/prepare_topas_neutral.py \
  --header validation/topas/output/neutral_development_interactions.header \
  --phsp validation/topas/output/neutral_development_interactions.phsp \
  --interactions-output validation/results/topas_200MeVu_neutral_development_interactions.csv.gz \
  --products-output validation/results/topas_200MeVu_neutral_development_products.csv.gz \
  --metadata validation/results/topas_200MeVu_neutral_development.metadata.json \
  --runtime-log validation/topas/output/neutral-development_topas.log \
  --case development
```

The postprocessor requires positive interaction macroscopic cross sections and
covers neutron elastic/inelastic/capture plus gamma photoelectric, Compton,
pair-production and Rayleigh processes. It is a sampled transport table, not a
dose map to copy or globally scale.

Compile the host/GPU binary after standardization:

```bash
python3 validation/scripts/compile_neutral_package.py \
  --metadata validation/results/topas_200MeVu_neutral_development.metadata.json \
  --interactions validation/results/topas_200MeVu_neutral_development_interactions.csv.gz \
  --products validation/results/topas_200MeVu_neutral_development_products.csv.gz \
  --output validation/results/topas_200MeVu_neutral_development.bin \
  --output-metadata validation/results/topas_200MeVu_neutral_development.compiled.json
```

The binary contains two projectiles (PDG 22 and 2112), cross-section samples,
interaction continuation kinematics, and correlated products in the incident
local frame. Load with `NeutralPackageTable::from_binary`.

## Charged-fragment reaction cascade

`CarbonCascadeNtuple` records every charged projectile inelastic interaction,
the TOPAS macroscopic inelastic cross section at that energy, and all correlated
direct products. In MT output the stable key is `(run, thread, event,
interaction_sequence)`; track ID alone is not unique because a surviving track
may interact more than once.

Run `run_cascade_remote.sh smoke` before the detached 100,000-history
`development` job. Standardize and compile with `prepare_topas_cascade.py` and
`compile_cascade_package.py`. The accepted table contains 34 projectile
isotopes, 71,089 usable interactions, 511,019 products, and 13,469 cross-section
samples. Two Z4A4 interactions have zero TOPAS inelastic cross section and are
excluded with an explicit metadata audit entry. No decay process was observed
in this reference.

The Windows B580 implementation transports at most two further generations in
a breadth-first atomic queue. Every charged nuclear descendant is assigned to
the dose category implied by its own Z/A, matching `CarbonDoseOrigin`; inheriting
the original charged ancestor category is incorrect for nuclear products. The
isotope/generation audit then found that mixed-version final-state datasets
caused the remaining species bias. Mixed packages are now rejected; both the
primary and ancestor/cascade sides must be generated with TOPAS 4.2.p3 /
Geant4 11.3.2.

`prepare_primary_reactions_from_cascade.py` extracts 37,661 primary track-1
C-12 interactions and 323,901 correlated products from the cascade reference.
The aligned 201-bin package brings the charged-origin total difference to
`-0.11%/+2.93%` over all depths/after 90 mm. Helium is `-0.20%/+2.58%`, proton
is `+1.41%/+2.84%`, and every charged category is within 7% in the tail. The
maximum per-bin species closure is `9.80e-11 MeV/primary/bin`; no global scale
is used.

### 400 MeV/u cascade package (multi-energy high-E tail fix)

For 300/400 MeV/u beams the 200 MeV/u package clamped high-energy sampling.
Run remote cascade at 400 MeV/u (covers primary reactions ~0–400 MeV/u as the
beam slows):

```bash
# on Windows helper (password via REMOTE_TOPAS_PASSWORD)
python validation/scripts/_remote_cascade_e400.py upload
python validation/scripts/_remote_cascade_e400.py smoke
python validation/scripts/_remote_cascade_e400.py development
python validation/scripts/_remote_cascade_e400.py status
python validation/scripts/_remote_cascade_e400.py fetch
validation\scripts\postprocess_cascade_e400.cmd
```

Accepted 100k table: 73,739 primary C-12 packages into **401 × 1 MeV/u bins**,
plus multi-projectile cascade binary. Wire
`config/beam_{300,400}MeVu_multi_energy.yaml` to
`topas_400MeVu_cascade_aligned_primary_3d.bin` and
`topas_400MeVu_cascade_100k_3d.bin`. After the switch, 400 MeV/u tail integral
error improved from about −38% to −6.8% and 2%/2 mm gamma from 65% to 99%.

The dependency-free comparison command is:

```bash
python3 validation/scripts/compare_ancestor_attributed_idd_portable.py \
  validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv \
  validation/results/windows_b580_fragment_cascade_aligned_100k_species.csv \
  --metrics-output validation/results/windows_b580_fragment_cascade_aligned_100k_vs_topas.metrics.json \
  --plot validation/results/windows_b580_fragment_cascade_aligned_100k_vs_topas.svg
```

## Direct cross-section and reaction-final-state extraction

The attenuation probability and the fragmentation final state are extracted separately from the same TOPAS/Geant4 physics configuration:

- `CarbonCrossSectionNtuple` directly queries `G4HadronicProcessStore` for C-12+H, C-12+O, and Water_75eV inelastic cross sections from 1 to 400 MeV/u. This table determines the reaction distance in the GPU transport.
- `CarbonReactionNtuple` records the primary C-12 inelastic reaction header and every direct secondary in the same event, including incident energy, vertex, particle identity, kinetic energy, direction, creator process, and model ID. Rows with one `reaction_id` must be sampled jointly.
- the OriginCount and species-resolved IDD scorers are independent QA observables; they must not replace correlated event-level final-state sampling.

The command below is the legacy WSL extension build used for already completed
cross-section and reaction-package datasets. Do not use it for new TOPAS jobs;
new validation runs use `build_extensions_remote.sh` on `v@192.168.31.5`.

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
