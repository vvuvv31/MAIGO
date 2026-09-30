# MAIGO TOPAS package extraction kit

This directory contains the TOPAS extensions, run generators and binary
compilers needed to regenerate the current MAIGO GPU physics packages. Large
raw campaigns and compiled `.bin`/`.cinpkg` artifacts are intentionally not
stored here.

The kit targets TOPAS 4.2.p3 and Geant4 11.3.2. A different TOPAS/Geant4 or
physics-list version is a **new package version** and must produce new metadata
and hashes; do not overwrite a pinned production package in place.

## What is included

| Runtime input | Extraction source | Runner/compiler |
|---|---|---|
| `unified_em_v1.bin` | private Geant4 EM tables through `CarbonStoppingPowerNtuple` | `tools/unified_em/` |
| `unified_em_delta_moments_v2.bin` | derived from the pinned EM package; no TOPAS rerun | `build_delta_moments.py` |
| primary/secondary CINEL03 final states | `CarbonInelasticCapturePhysics`, event writer and exposure scorer | `tools/schneider/` + `package_tools/` |
| Schneider primary/secondary reaction rates | Schneider inelastic XS dump scorers | `xs_dump_v2_1.py`, rate compilers |
| C12 and all-ion stopping tables | stopping dump scorers | Schneider stopping compilers |
| `all_ion_elastic_v1.bin` | `AllIonElasticDump` | `tools/general_ion_elastic/` |
| elastic recoil stopping | the same extension with `RecoilStoppingOnly` | `tools/general_ion_elastic/` |
| Copper stopping/rate/elastic/C12 INCL++ | common TOPAS extensions | `tools/minibeam_copper/` |
| Copper p/d/t/He/heavy cascade packages | CINEL03 capture writer | `tools/minibeam_copper/` |

`topas/common/` is the current extension source catalog used by the nuclear,
Copper and Schneider workflows. `topas/unified_em/` deliberately contains a
different implementation of `CarbonStoppingPowerNtuple`; it accesses Geant4
process tables and must replace, not coexist with, the common implementation.

`topas/all_ion_elastic/frozen_production/` preserves the exact sources used by
the accepted 2026-09-11 packages. The parent directory contains the later
combined source that can export both elastic samples and recoil stopping.

## Build an extension-enabled TOPAS

Build separate binaries for the three profiles because the common and Unified
EM exporters have the same C++ class name:

```bash
extensions/build_topas_extensions.sh \
  --topas-source /path/to/OpenTOPAS-4.2.3 \
  --build-dir /work/topas-nuclear \
  --profile nuclear-copper \
  --geant4-dir /path/to/geant4/lib/cmake/Geant4 \
  --cmake-prefix /path/to/topas/dependencies

extensions/build_topas_extensions.sh \
  --topas-source /path/to/OpenTOPAS-4.2.3 \
  --build-dir /work/topas-em \
  --profile unified-em \
  --geant4-dir /path/to/geant4/lib/cmake/Geant4

extensions/build_topas_extensions.sh \
  --topas-source /path/to/OpenTOPAS-4.2.3 \
  --build-dir /work/topas-elastic \
  --profile all-ion-elastic \
  --geant4-dir /path/to/geant4/lib/cmake/Geant4
```

Use a fresh, profile-specific build directory. To reproduce the exact accepted
General Ion Elastic sources rather than make a new package with the combined
exporter, select `all-ion-elastic-production` for
`all_ion_elastic_v1.bin` and `elastic-recoil-production` for
`elastic_recoil_stopping_v1.bin`. They stage the frozen sources with SHA256
`ad0b5b8...` and `2d34bd9...`, respectively, together with the accepted header
`1842f947...`.

Use the same compiler, Geant4, Qt and GDCM dependencies as the target TOPAS.
The dependency values can usually be copied from the `CMakeCache.txt` of a
known working TOPAS build. Extra definitions can be supplied repeatedly with
`--cmake-arg=-DNAME=VALUE` (the equals form is convenient when VALUE starts
with `-`). The pinned Geant4 build exports Qt UI definitions, so the helper
defaults `TOPAS_USE_QT` to `ON`; use `--topas-use-qt OFF` only with a Geant4
build that was itself configured without Qt.
Merely copying `.cc` files next to an existing executable does not register the
extensions.

## Common environment

The portable copies use the repository containing this directory by default.
Large campaign locations can be changed without editing source:

```bash
export MAIGO_REPO_ROOT=/path/to/MAIGO
export MAIGO_PACKAGE_WORK_ROOT=/fast/work/maigo-packages
export TOPAS_BINARY=/work/topas-nuclear/topas
export TOPAS_G4_DATA_DIR=/path/to/Geant4/data
export TOPAS_LD_LIBRARY_PATH=/path/to/topas/lib:/path/to/geant4/lib
export MAIGO_SCHNEIDER_DICOM_DIR=/path/to/dicom-that-instantiates-all-25-sections
```

Schneider-table and all-ion-elastic extraction requires a small DICOM patient
covering all 25 Schneider material sections. The DICOM is external input and
must be hashed in the campaign manifest; the repository cannot silently
replace it with a different material construction.

The Python entry points documented below are the portable interfaces. Files
named `*.sbatch`, the saved `recoil_stopping.txt`, and `cases.txt` are retained
as exact production recipes/provenance and therefore contain the original
cluster paths. Generate a new campaign with the Python preparer, or edit those
cluster paths before submitting them on another system. Reference manifests
may likewise contain absolute paths to the original raw evidence; hashes, not
those path strings, are the portable identity.

## Unified EM

Prepare material-density nodes and one TOPAS input per supported ion:

```bash
python3 extensions/tools/unified_em/prepare.py \
  --output /fast/work/unified-em
python3 extensions/tools/unified_em/run_extract.py \
  --campaign /fast/work/unified-em --topas /work/topas-em/topas
python3 extensions/tools/unified_em/run_probes.py \
  --campaign /fast/work/unified-em --topas /work/topas-em/topas
python3 extensions/tools/unified_em/run_distributions.py \
  --campaign /fast/work/unified-em --topas /work/topas-em/topas
python3 extensions/tools/unified_em/compile_unified_em_package.py \
  /fast/work/unified-em data/em/unified_em_v1.bin
python3 extensions/tools/unified_em/verify_unified_em_data.py \
  data/em/unified_em_v1.bin --core-only
python3 extensions/tools/unified_em/build_delta_moments.py
```

The accepted package uses an EM maximum of 10 GeV. A 600 MeV TOPAS table limit
is invalid for, for example, 300 MeV/u He-4 (1200 MeV total kinetic energy).

## General Ion Elastic

Prepare and run the 18-projectile campaign:

```bash
python3 extensions/tools/general_ion_elastic/prepare_campaign.py \
  --hu-file data/HUtoMaterialSchneider.txt \
  --dicom-dir /path/to/Schneider-DICOM \
  --output /fast/work/all-ion-elastic
python3 extensions/tools/general_ion_elastic/run_campaign.py \
  --topas /work/topas-elastic/topas \
  --campaign /fast/work/all-ion-elastic
python3 extensions/tools/general_ion_elastic/compile_all_ion_elastic.py \
  /fast/work/all-ion-elastic data/schneider/all_ion_elastic_v1.bin
python3 extensions/tools/general_ion_elastic/verify_all_ion_elastic.py
```

Run `prepare_campaign.py --recoil-only` and use the generated
`recoil_stopping.bin` for the recoil table. The exact accepted package used the
two sources under `frozen_production/`; use those when byte-level provenance,
rather than a new package version, is required.

## Schneider/water inelastic and stopping

The high-level campaign generators are:

- `run_step17_element_campaigns.py`: primary C12 elemental final states;
- `run_step20_secondary_campaigns.py`: secondary-ion elemental final states;
- `xs_dump_v2_1.py` / `run_step20_secondary_rates.py`: reaction rates;
- `submit_ion_section_stopping.py`: all-ion Schneider stopping;
- `generate_v2_manifest.py` and `emit_v2_scripts.py`: targeted v2.1 gap-fill
  campaign generation.

The CINEL raw codec and runtime CINPKG04 codec live in `package_tools/`. The
accepted build order is:

```text
TOPAS raw CINEL02
  -> validate_v2_raw.py
  -> compile_primary_package_v2_1.py / compile_v2_package.py
TOPAS cross-section dumps
  -> compile_primary_rates_v2_1.py / compile_rates_v2_1.py
all outputs
  -> make_physics_bundle_v2_1.py
  -> generate_schneider_v2_1_manifest.py
  -> verify_schneider_v2_1_data.py
```

The accepted scripts preserve strict target identity, per-channel energy
domains and a maximum 5 MeV/u lookup gap. Do not replace gaps by endpoint
clamping or nearest-channel aliasing.

## Minibeam Copper

`tools/minibeam_copper/` contains the actual current templates, Slurm scripts
and compilers. The extraction physics is process-faithful:

- C12 and d/t/He/heavy ions use INCL++ where TOPAS did;
- proton+Cu uses Binary Cascade;
- the compiler checks the observed model name;
- package lookup gaps are filled with new events, not endpoint clamping.

The copied historical Slurm files retain the accepted campaign composition and
paths as provenance. Before a new campaign, change `repo`, `root`, TOPAS binary
and any reused source-package paths, or translate them to the local scheduler.
The Python compilers themselves now resolve `extensions/package_tools`
relative to this repository.

## Integrity and scope

Run:

```bash
python3 extensions/verify_kit.py
# Or verify packages installed in another checkout/artifact tree:
python3 extensions/verify_kit.py --package-root /path/to/installed/MAIGO \
  --require-packages
```

This checks all bundled source snapshots and the hashes of the currently
accepted large runtime packages when those files are locally available. It
does not download or commit multi-hundred-megabyte physics binaries.

The kit is package extraction infrastructure, not a claim that every optional
research package is enabled in production. In particular, General Ion Elastic
remains a separately selectable research input in current MAIGO configurations.

## Non-minibeam proton primary (2026-09-26)

The 0.1–250 MeV proton source workflow is under `tools/primary_source/`.
See [package contents, extraction provenance and limitations](../docs/proton_fullphysics_packages.md).
The common Schneider stopping exporter now accepts `ProjectileZ`/`ProjectileA`;
legacy defaults remain C12. The current elastic/recoil exporter uses
`Water_75eV`; historical `frozen_production` sources remain unchanged.

Conditional low-energy final-state extraction can explicitly set
`MAIGO_CINEL_FINAL_STATE_XS_FACTOR` with primary-only capture. This changes
interaction frequency **only for final-state sampling**. Never use its exposure
as a physical rate or dose reference. Raw contracts record the factor; physical
rates are extracted independently with the variable unset.

### Explicit Urban reference physics

`topas/common/AllChargedUrbanPhysics.{cc,hh}` provides the TOPAS physics module
`AllChargedUrbanPhysics`. Place it **after** `g4em-standard_opt4` in the module
list. It replaces each hadron multiple-scattering process with a fresh
`G4hMultipleScattering` using `G4UrbanMscModel` up to 10 GeV, matching
`tools/urban/extract_active_urban.cc`. For each replaced process manager it
also removes the independent electromagnetic Coulomb-scattering companion
from opt4's Wentzel mixed model. Urban samples its own angular tail; the
reference is pure Urban. Hadronic elastic and nuclear stopping retain their
configured processes, as does electron transport.
The module does not activate unless explicitly named in the TOPAS input.
Comparisons in `../evidence/proton_urban_em_20260926/` used the historical
Urban-plus-Coulomb reference. The user selected pure Urban on 2026-09-27;
see `../evidence/proton_pure_urban_20260927/` for the revised reference.

## Primary electron-response extraction

`PrimaryElectronDepositNtuple` and `PrimaryIonizationResponsePhysics` support
source-specific homogeneous-water electron response extraction. The latter is
**extraction only**, not a full-physics module: massive-particle Coulomb recoil
and nuclear stopping are removed from the ionization response source.
`tools/water_electron/compile.py` preserves deposition-weighted samples and
complete path ancestry. See `../docs/minibeam_water_electron_response.md`.

### Native copper elastic tables

`PrimaryMaterialDump` supports `d:Sc/<name>/MaximumEnergyPerNucleon` (MeV;
default 251.01 MeV/u coverage) and `i:Sc/<name>/SamplesPerNode` (default 512).
Source ZA and target material come from the actual primary step. Elastic
samples call the attached Geant4 model directly, with native isotope masses;
continuous EM step loss must not be interpreted as nuclear recoil.

`tools/minibeam_copper/compile_copper_physics.py` now requires
`--native-elastic-file` and `--native-elastic-metadata`. The previous pre/post
transport-step subtraction is invalid and is no longer available for compiling
new banks. Existing pinned historical results remain evidence, not validated
elastic reference tables. C12/Cu extraction and paired validation are recorded
under `evidence/carbon_copper_elastic_20260926`.

### Independent Copper primary path tag

`PrimaryCopperTouchNtuple` records one `(run,event,track)` row for each primary
with a positive Geant4 step in a pure Copper material (Z=29). Attach it to the
Copper component and join its IDs to a separately scored exit phase space.
It records the first Cu step length/energy, not cumulative path length. The
scorer uses worker-local event state and does not modify transport or RNG.
The native TOPAS traversed-component flag is not an equivalent path tag;
independent validation is in `evidence/valley_gpu_compare_20260927`.

### Native neutron package extraction (staged)

`tools/neutron/` builds against Geant4 11.3.2 and extracts native conditional
elastic/inelastic/capture/fission rates and complete correlated products for
Water_75eV, G4_Cu and G4_AIR. Run builds/extraction with local Slurm. The compiler
accepts only completed banks and emits a checksummed draft package;
`inspect_neutron` reads it in audit mode. `SAMPLES=0` extracts rates only and is
not a collision bank accepted by the compiler. An optional `RATE_REPEATS`
argument (only with `SAMPLES=0`) records the rate mean, SEM and extrema; it
evicts native exact-energy caches between queries. A single-query SEM of zero
means unestimated uncertainty, not an exact rate. Raw CSV energies are MeV,
macroscopic rates 1/mm and times ns.

The YAML neutron interface is guarded: GPU neutron transport and photon
descendants remain incomplete. See `../docs/neutron_package_plan.md` and
`../evidence/neutron_package_20260927/`. Never promote a package based only on
serialized kinetic bookkeeping closure.


`topas/common/NeutronProtonBirthNtuple` records proton secondaries at their
generating step, including parent PDG/process and birth phase space. Attach it
separately to existing Water, ROI and Copper with propagation disabled; TOPAS
forbids World scorers. Propagating Water scoring into an already scored ROI
creates an extra parallel world and changes fixed-seed transport. It checks the generating endpoint and deduplicates child pointers
per event/parent because scoring-step views can expose older secondaries.
The associated same-seed total-dose regression is required before interpreting
its birth statistics. See `evidence/neutron_package_20260927` in the repository.
