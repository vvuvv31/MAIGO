# MAIGO_clean

论文主线为 **GPU Monte Carlo: From Broadbeam to Minibeam**。材料与方法：[中文](mm_zh.md) / [English](mm_en.md)。

A standalone GPU runtime for the eight current MAIGO presets. The source and
runtime inputs were copied independently from MAIGO_minibeam; there are no
symlinks back to it. Shared electron projectile aliases are hardlinks **within
this directory**, not to the original repository. The original working tree
was not changed by this refactor.

## Build and run

The supported release platform is Linux with an NVIDIA GPU. MAIGO uses the
Intel LLVM/DPC++ SYCL compiler with the CUDA backend
(`nvptx64-nvidia-cuda`); this release has no CPU, Intel GPU or AMD GPU runtime.
The workflow is: install the driver and CUDA Toolkit → build the CUDA-capable
SYCL toolchain → compile MAIGO → restore physics data → supply matching case
inputs → run and inspect the audits. Compilation does not require patient data.

### 1. Install system dependencies and CUDA

The following package commands assume Ubuntu 24.04 LTS (Python 3.12).
Use a host compiler supported by your chosen LLVM and CUDA versions; MAIGO
requires CMake 3.22 or newer and C++20. The upstream LLVM build may require a
newer CMake; consult its [build guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).

```bash
sudo apt update
sudo apt install build-essential git cmake ninja-build python3 python3-venv \
  libhwloc-dev libzstd-dev libz-dev curl ca-certificates
```

`hwloc` supplies host hardware-topology information to the SYCL toolchain;
it does not enable CPU particle transport. Building LLVM requires substantially
more RAM and disk space than building MAIGO. Choose build parallelism to fit
available RAM rather than automatically using every CPU thread.

Install an NVIDIA driver compatible with your GPU and CUDA version, then the
[NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
Follow NVIDIA's distribution-specific installation instructions and reboot if
required by the driver installation. The existing local toolchain uses CUDA
12.6; this is a reference setup, not a claim that every toolkit version works.

```bash
nvidia-smi
export CUDA_ROOT=/usr/local/cuda-12.6  # change to your Toolkit installation
"$CUDA_ROOT/bin/nvcc" --version
```

The CUDA version displayed by `nvidia-smi` is the driver's supported CUDA
version, not proof that the Toolkit is installed. Both checks above should work.

### 2. Build the complete Intel LLVM/SYCL toolchain

The [`libsycl` directory](https://github.com/intel/llvm/tree/sycl/libsycl) contains
part of the runtime; building it alone does not provide the SYCL compiler and
CUDA backend. Build the complete Intel LLVM repository using its official
configuration and compilation helpers:

```bash
export SYCL_WORKSPACE="$HOME/sycl_workspace"
mkdir -p "$SYCL_WORKSPACE"
git clone --branch sycl https://github.com/intel/llvm.git "$SYCL_WORKSPACE/llvm"
# For reproducible builds, check out your chosen compatible commit before configuring.
git -C "$SYCL_WORKSPACE/llvm" rev-parse HEAD

CC=gcc CXX=g++ python3 "$SYCL_WORKSPACE/llvm/buildbot/configure.py" \
  --cuda -t Release -o "$SYCL_WORKSPACE/llvm/build" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT"
python3 "$SYCL_WORKSPACE/llvm/buildbot/compile.py" \
  -o "$SYCL_WORKSPACE/llvm/build" -j 4
```

The default compilation target is `deploy-sycl-toolchain`, which deploys the
toolchain under the build directory's `install/`. Both helpers must receive the
same `-o` directory. Record the LLVM commit and CUDA version used; the `sycl`
branch changes over time. The current development compiler reports Intel LLVM
commit `ebbf0d953099709c1ab0e4d5abe64d4884620bef` with CUDA 12.6; check out that
commit before configuration if reproducing this toolchain. The official build
guide remains the reference for version-specific dependencies and options.

Configure the installed compiler and runtime in each new shell:

```bash
export SYCL_ROOT="$SYCL_WORKSPACE/llvm/build/install"
export PATH="$SYCL_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SYCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXX="$SYCL_ROOT/bin/clang++"
"$CXX" --version
ONEAPI_DEVICE_SELECTOR=cuda:gpu "$SYCL_ROOT/bin/sycl-ls"
```

`sycl-ls` must list an NVIDIA CUDA GPU before proceeding. This requires both
the CUDA runtime adapter and a working NVIDIA driver. A preinstalled `icpx`
may also be used if its CUDA backend is available; the compiler name alone
does not establish CUDA support.

### 3. Clone and compile MAIGO

```bash
git clone --branch clean --single-branch https://github.com/vvuvv31/MAIGO.git MAIGO_clean
cd MAIGO_clean
export MAIGO_ROOT="$PWD"

# Local RTX 2080 Ti example; select the GPU that will run this binary.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="--cuda-path=$CUDA_ROOT" \
  -DCARBON_CUDA_ARCH=sm_75
cmake --build build --target carbon_mc -j 4
build/carbon_mc --help
```

| GPU | `CARBON_CUDA_ARCH` |
|---|---|
| NVIDIA L4 | `sm_89` |
| RTX A6000 | `sm_86` |
| RTX 2080 Ti | `sm_75` |

An explicit architecture produces GPU-specific code ahead of time and reduces
driver JIT work on first execution. Leaving `CARBON_CUDA_ARCH` empty uses the
portable PTX path, which can cause a substantial first-launch compilation delay.
Building on a GPU-equipped machine does not automatically select its architecture:
set this option explicitly. The local RTX 2080 Ti build uses `sm_75`; use `sm_89`
for L4 and `sm_86` for RTX A6000. Use a separate build directory when changing
the compiler or GPU architecture.

All five transport families are enabled by default: proton broadbeam, proton
minibeam, carbon broadbeam, carbon minibeam and generic proton/C12. To compile
only selected families, use the `CARBON_ENABLE_*` switches described in
[selective compilation](docs/transport_pipelines.md#selective-compilation).
A configuration cannot run a pipeline omitted from the binary.

Alternatively, with `CXX` set to the absolute compiler path above, the build
wrapper accepts the same CMake options:

```bash
CXX="$CXX" BUILD_JOBS=4 BUILD_DIR=build \
  bash scripts/build.sh -DCARBON_CUDA_ARCH=sm_75 \
  -DCMAKE_CXX_FLAGS="--cuda-path=$CUDA_ROOT"
```

Always set `CXX` when using the wrapper on another machine; its fallback path
points to the original development machine.

### 4. Download and restore the physics assets

The [v2026.10.14 Release](https://github.com/vvuvv31/MAIGO/releases/tag/v2026.10.14)
contains the physics assets for the **four CT presets**, including optional
electron/photon packages. It is not an eight-preset water/CT asset bundle.
The complete gzip archive is split into two ordered parts to satisfy GitHub's
attachment size limit. Download both parts and their helper files into one
folder outside the repository:

```bash
export ASSET_DIR="$(dirname "$MAIGO_ROOT")/maigo_assets"
mkdir -p "$ASSET_DIR"
export RELEASE_URL=https://github.com/vvuvv31/MAIGO/releases/download/v2026.10.14
for asset in maigo_four_ct_assets.tar.gz.part01 maigo_four_ct_assets.tar.gz.part02 \
  archive.json manifest.json SHA256SUMS restore.py README.md verification.json; do
  curl --fail --location --retry 3 "$RELEASE_URL/$asset" --output "$ASSET_DIR/$asset"
done
(cd "$ASSET_DIR" && sha256sum -c SHA256SUMS)
python3 "$ASSET_DIR/restore.py" --dest "$MAIGO_ROOT" --data-only
```

The restore helper requires **Python 3.12 or newer**. It verifies the parts,
assembles and verifies the full archive, then verifies each restored file.
`--data-only` preserves the cloned configuration and documentation. Reserve
space for about 3 GB of parts, 3 GB of assembled archive and 6.3 GB of extracted
data, plus the toolchain, build files and outputs. Physics files must appear at
paths such as `data/em/ions/continuous26.bin`; do not put the entire cache folder
inside `data/`.

If the same cache is already on a mounted Google Drive, restore it directly:

```bash
python3 /content/drive/MyDrive/gpumc/restore.py \
  --dest /content/MAIGO_clean --data-only
```

Run from local disk after restoring. For water presets, provide their additional
assets listed in [preset dependencies](docs/preset_dependencies.json).
Electron transport is default off; package availability does not enable it.
Only explicit `enable_electron_transport: true` loads the required complete,
projectile-matched electron/photon packages.

### 5. Supply matching case inputs

Neither Git nor the Release archive includes `inputs/`: CT grids, source plans,
beam parameters and optimized spot weights must be supplied separately at the
paths referenced by the selected YAML. Inspect
[config/](config/) and [preset dependencies](docs/preset_dependencies.json)
before running. Preserve each case's CT, source, weights, placement, collimator
and dose normalization as one consistent input set.

RT07575 is a **head** case. CT broadbeam defaults use its head inputs; default
CT minibeams use a separate lung benchmark and are not RT07575 head full plans.
A CT grid uses the CCTG binary format; TPS source CSV columns are
`energy_MeV,x_mm,y_mm,weight`, with total ion energy in `energy_MeV`.
The runtime does not convert DICOM or replay TOPAS text plans.

Validation helpers require NumPy and PyYAML:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pyyaml
```

The Release restore helper validates its own asset manifest. The later
`check_data_manifest.py` inventory check covers the complete local eight-preset
installation, including inputs; it is not a prerequisite that a four-CT physics-only
restore can satisfy by itself.

### 6. Run a scene or continuous batches

Run from the repository root, keeping the SYCL runtime library path configured.
After supplying the matching inputs, start with a small execution check:

```bash
cd "$MAIGO_ROOT"
build/carbon_mc --config config/proton_ct_broadbeam.yaml --histories 10000
```

Continuous batches share one GPU context and loaded physics packages:

```bash
build/carbon_mc --config config/proton_ct_broadbeam.yaml \
  --batch-count 3 --histories-per-batch 100000 \
  --batch-output out/proton_ct_broadbeam_first_batches
```

Use a new output directory for each campaign. CLI options are `--config`,
`--histories`, `--seed`, `--batch-count`, `--histories-per-batch`, `--batch-output`
and `--help`. Counts refer to incident primary particles. Plan weights are
allocated across the whole campaign; outputs retain the original absolute dose
normalization. Do not multiply normalized plan dose by the number of batches.

Inspect the reported output directory, `quality_report.json` and
`energy_ledger.json`. Require completed histories, acceptable energy/EM and
coverage audits, and zero queue overflow before using a result. A small run
checks execution, not dose accuracy or representative throughput. Reduce batch
size if the GPU memory budget or secondary queues are exceeded. Distinguish
process startup/package loading from subsequent batch transport when measuring
performance; neither should include compiler build time.

The eight release presets select their implementation with `transport_pipeline`.
Custom proton/C12 configurations without that key use `generic`; a copied
release preset retains its selection until the key is removed or changed.
Pipeline layout is documented in [transport pipelines](docs/transport_pipelines.md).

## Release presets and physics

| Preset | Source/ion path | Default electron path |
|---|---|---|
| proton_water_broadbeam | Flat source → Gaussian FE → Water_75eV | Empty uniform-water selection; local deposition |
| carbon_water_broadbeam | Carbon source → Gaussian FE → Water_75eV | Same policy |
| proton_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → proton CT kernel | Local electron energy; tracking disabled |
| carbon_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → C12 CT kernel | Same policy |
| proton_water_minibeam | Flat source → compact copper/air → Urban water | Local deposition; nuclear neutrals untracked |
| carbon_water_minibeam | TPS CSV → compact copper/air → Urban water | Local deposition |
| proton_ct_minibeam | Flat source → compact copper/air → Urban proton CT | Local deposition |
| carbon_ct_minibeam | Flat source → compact copper/air → Urban CT | Local deposition; dose accuracy deferred |

Ion step caps are forced during configuration loading: **broadbeam 1.0 mm**,
**minibeam 0.5 mm**, and **minibeam copper 0.05 mm**, for both water and CT.
These effective caps override the corresponding YAML values.

Release presets retain primary energy-loss straggling and disable secondary
straggling by default (`enable_secondary_energy_straggling: false`). Minibeams
also disable restricted Cu fragment straggling; Cu primary fluctuations remain
on. Native carbon Cu fragments retain hard-delta sampling. Scoped RT07575 CT
validation is recorded for proton broadbeam/minibeam and carbon broadbeam;
**carbon minibeam is under verification**. This status does not establish
water/other-patient acceptance. See [validation scope and measured effects](docs/secondary_straggling.md).

The electron-off proton CT broadbeam primary uses a moment-matched Gamma for
dense restricted-ionization tails; primary fluctuations remain on. Low-count
tails and other pipelines retain explicit Poisson marks. See [scope and paired
results](docs/proton_ct_broadbeam_dense_tail_gamma_20261013.md).
For the combined CT broadbeam sampling, range-cache and recoil-loss changes,
see the [direct throughput comparison and dose limits](docs/broadbeam_fluctuation_optimization_20261013.md).
Carbon CT broadbeam also precomputes elastic totals for primary and fragment
species; see [scope, paired timings and validation limits](docs/carbon_broadbeam_elastic_totals_20261013.md).
Electron-off proton/carbon CT minibeams now reuse actual CT material/range
fields and analytic range inversion; elastic total caches cover proton primary
and carbon primary/fragments. See [matched full-plan measurements and dose limits](docs/minibeam_ct_cache_optimization_20261013.md).

Each row has the matching `config/<preset>.yaml`. Default CT minibeams are lung
benchmarks, **not RT07575 head full plans**. Broadbeam caps ion steps at 1.0 mm
and minibeam at 0.5 mm in both water and CT.
Minibeam copper retains its separate 0.05 mm cap. All use continuous26 ion EM;
Electron/photon transport is optional for all presets and custom configurations.
It defaults to off; only `enable_electron_transport: true` loads the configured
strict26 birth, charged, photon and spatial-joint packages. Package paths alone
do not enable transport, and disabled paths need not exist. When off, ionization
electron energy is deposited locally. Enabled transport still requires complete,
projectile-matched packages. CT broadbeam also uses
`broadbeam_electron_tracking: true` to select its tracking region. The optional broadbeam
electron selection uses density≤0.1 g/cm³ plus a 5 mm halo; charged nuclear-secondary
ionization is local. Uniform broadbeam water has no selected region and does not load electron banks. Enabled minibeam water uses
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
Current process implementations and differences from TOPAS/Geant4 are documented in
[TOPAS and GPU physics comparison](docs/topas_gpu_physics_comparison.md) (Chinese).

All eight release presets enable medium nuclear elastic transport for primary and
charged secondary particles using projectile-specific `all_ion_elastic_file` and
`elastic_recoil_stopping_file` packages under `data/elastic/schneider/{proton,carbon}/`.
All four minibeams also set `minibeam_copper_enable_elastic: true` for the copper collimator.
Primary and patient-medium charged secondary transport share the nuclear optical-depth
sampler: elastic and inelastic rates compete through one clock per track. Secondary
continuation preserves the remaining depth and draw counter; the inelastic generation
limit does not disable elastic scattering.

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
Optional electromagnetic electron/photon transport is retained. See
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
