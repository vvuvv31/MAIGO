# MAIGO_clean

A standalone GPU runtime for the eight current MAIGO presets. The source and
runtime inputs were copied independently from MAIGO_minibeam; there are no
symlinks back to it. Shared electron projectile aliases are hardlinks **within
this directory**, not to the original repository. The original working tree
was not changed by this refactor.

Build with the installed SYCL compiler:

```bash
BUILD_JOBS=6 bash scripts/build.sh
```

All five transport families are enabled by default. To compile only the family
being optimized, use `BUILD_DIR` and the `CARBON_ENABLE_*` CMake switches described
in [selective compilation](docs/transport_pipelines.md#selective-compilation).

Run one scene, or continuous batches sharing one GPU context and package load:

```bash
build/carbon_mc --config config/proton_water_broadbeam.yaml --histories 10000
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 2 --histories-per-batch 10000 --batch-output out/proton_ct_minibeam_smoke
```

The eight release presets select their proton/carbon broadbeam/minibeam
implementation with `transport_pipeline`. Custom proton/C12 configurations
without that key use `generic`; a copied release preset retains its explicit
selection until the key is removed or changed. Pipeline layout and optimization
boundaries are documented in [docs/transport_pipelines.md](docs/transport_pipelines.md).

Use a new output directory. CLI options are `--config`, `--histories`, `--seed`,
`--batch-count`, `--histories-per-batch`, `--batch-output` and `--help`.
Batch histories count incident primaries. Plan weights are allocated across the
whole campaign; output dose retains the original absolute normalization.

| Preset | Source/ion path | Electron path |
|---|---|---|
| proton_water_broadbeam | Flat source → Gaussian FE → Water_75eV | Empty uniform-water selection; local deposition |
| carbon_water_broadbeam | Carbon source → Gaussian FE → Water_75eV | Same policy |
| proton_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → proton CT kernel | FP32 tracking in low density + 5 mm halo |
| carbon_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → C12 CT kernel | Same policy |
| proton_water_minibeam | Flat source → compact copper/air → Urban water | Independent FP32 group32; nuclear neutrals untracked |
| carbon_water_minibeam | TPS CSV → compact copper/air → Urban water | Independent FP32 group32 |
| proton_ct_minibeam | Flat source → compact copper/air → Urban proton CT | FP32 group32, spatial joint |
| carbon_ct_minibeam | Flat source → compact copper/air → Urban CT | FP32 group32, spatial joint; dose accuracy deferred |

Each row has the matching `config/<preset>.yaml`. Default CT minibeams are lung
benchmarks, **not RT07575 head full plans**. Broadbeam caps ion steps at0.5 mm;
minibeam at0.1 mm. All use continuous26 ion EM and strict26 electron packages.
Broadbeam tracks primary-ionization electrons only in density≤0.1 g/cm³ plus
5 mm halo; charged nuclear-secondary ionization is local. Uniform water has
no selected region and does not load electron banks. Minibeam water uses
independent electron steps; Schneider materials preserve spatial joint.

`src/main.cpp` handles startup, `cli.cpp` parses the small CLI, `run_session.cpp`
owns source preparation/context reuse/output, `batch_run.cpp` allocates the
campaign, and `io.cpp` writes dose/physical audits. `source_pose.cpp` retains
only the geometry/air-loss helpers needed by TPS CSV sources. Transport and
current package readers remain in the shared physics core and `src/detail/`.

There are no archived readers, old density banks, per-batch SHA256 checks, DICOM-series
conversion, TOPAS text-channel plan reader, sequential spot replay, old CLI
aliases, diagnostic build modes or alternative precision scorers. Unsupported
configuration keys fail before transport. Mandatory quality/energy/coverage,
completion and overflow audits remain. LET/research scorer routes are outside
this runtime. Internally required fragments remain part of the current physics.
Proton minibeams retain `g4_universal_delta`; carbon minibeams use `g4_ion_delta`
with native Cu fragment Urban and species elastic transport. See
[Cu ion extraction and limits](extensions/copper_ion_transport/README.md).

The data/inputs are local external assets and are excluded from Git. The
installed inventory is `docs/copied_inputs.json`; expanded preset dependencies
are in `docs/preset_dependencies.json`. `data/manifest.json` records installed
asset sizes, SHA256 digests and each preset's dependency list. No default data
paths point to an archive.

Data directories follow the physics modules: `materials/`, `em/`, `nuclear/`,
`elastic/`, `stopping/`, `scattering/` and `ionization/`. Particle and material
subdirectories distinguish independent calibrations. Dates, generator versions
and binary format versions belong in metadata rather than runtime filenames.
RT07575 head inputs are in `inputs/rt07575_head/{proton,carbon}/`; the separate
lung benchmark is in `inputs/lung_minibeam/`. Historical generation records are
under `docs/data_provenance/`. See `docs/data_layout_20261007.md` for the migration.

Verify installed assets before packaging:

```bash
python3 scripts/check_data_manifest.py
python3 scripts/check_electron_packages.py
```

After an intentional asset change, regenerate the manifest with
`python3 scripts/check_data_manifest.py --write` and refresh dependencies with
`python3 scripts/list_preset_dependencies.py --output docs/preset_dependencies.json`.

Validation scripts require Python 3, NumPy and PyYAML.

The release configuration requires explicit `primary_atomic_number` and
`primary_mass_number`; YAML source energies use `initial_energy_MeVu`.
TPS CSV uses total `energy_MeV`, `x_mm`, `y_mm`, and `weight`. Central sources
are converted to the same total-energy representation at creation. No retired
source aliases or CSV Urban oracle reader remain. Urban loss/range/MFP uses
current packages. Primary mass validation checks each model's explicit mass
convention, preserving calibrated C12 kinematics without a zero-mass bypass.
Energy outputs use `physical_energy_balance_error` in the ledger and the
`physical_relative_energy_residual` / `absolute_physical_energy_residual_MeV`
quality fields; duplicate accounting aliases are removed.

All four minibeams use compact scheduling. The release cleanup also removes
retired interfaces and zero-only nuclear output layouts. See
`docs/release_cleanup_20261007.md`, `docs/release_cleanup2_20261007.md`, and
`docs/release_cleanup3_20261007.md` for migrations and validation evidence.
Primary nuclear channel metadata requires explicit boolean `has_support`.
Nuclear neutrals are not transported; their energy remains in explicit sinks.
Neutron packages, birth captures, transport settings and dose outputs are removed.
Electromagnetic electron/photon transport is retained. See
`docs/no_neutral_20261007.md` for removal scope and matched regression evidence. The proton water minibeam
therefore no longer includes the former neutron-dose contribution.
TPS isocenter input uses only the supported `tps_isocenter_z_mm` key.

For a matched implementation regression, supply a frozen control executable
and its preset directory with neutron transport disabled in every control
preset (`enable_neutron_transport: false` for binaries that support that key):

```bash
python3 scripts/validate_presets.py \
  --control /path/to/frozen/carbon_mc --control-root /path/to/frozen/root \
  --out docs/validation_new
python3 scripts/check_interfaces.py --output docs/interface_check_new.json
python3 scripts/check_release_inputs.py --output docs/release_inputs_new.json
python3 scripts/check_package_contracts.py --output docs/package_contracts_new.json
```

The control is used only for scientific comparison. Small-history matched GPU
checks are not TOPAS gamma acceptance, water dose acceptance or a throughput
measurement. The earlier refactor and compact scheduling checks remain documented
in `docs/refactor.md` and `docs/compact_20261007.md`.

Water Z geometry uses only `voxel_bins_z` and `voxel_size_z_mm`; both are required.
CT geometry comes from the CCTG header, with explicit voxel fields serving as
consistency assertions. Retired `phantom_length_mm` and `depth_bin_width_mm` YAML
keys are rejected. Internal transport dimensions are derived from the grid.

See `docs/release_cleanup4_20261007.md` for the geometry migration, removal scope
and final eight-preset regression evidence.

Primary nuclear rates use only `SCHNRATE` v3. Startup companion metadata uses
`basename.metadata.json`; the retired `basename.bin.metadata.json` route and
custom magic/version reader selectors are removed.

See `docs/release_cleanup5_20261007.md` for the fixed rate-reader contract and
matched eight-preset regression evidence.

The release runtime accepts only `device: cuda` and the
`nvptx64-nvidia-cuda` SYCL build target. Generic GPU/CPU selectors and vendor
aliases are retired; no fallback to another backend occurs.

See `docs/release_cleanup6_20261007.md` for CUDA-only migration and matched
eight-preset regression evidence.

Offline TOPAS/Geant4 extraction sources and build/package instructions are in
[extensions/README.md](extensions/README.md). They are separate from the GPU runtime.
