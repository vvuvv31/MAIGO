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
