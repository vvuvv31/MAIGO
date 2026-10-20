# MAIGO_clean

论文主线为 **GPU Monte Carlo: From Broadbeam to Minibeam**。材料与方法：[中文](mm_zh.md) / [English](mm.md)。

A standalone GPU runtime for the eight current MAIGO presets. The source and
runtime inputs were copied independently from MAIGO_minibeam; there are no
symlinks back to it. Ionization energy is deposited locally; explicit
electron/photon transport and its package-loading interfaces have been removed.

## Build and run

```bash
git clone --branch clean --single-branch https://github.com/vvuvv31/MAIGO.git MAIGO_clean
cd MAIGO_clean
export MAIGO_ROOT="$PWD"
```

### 1. For NVIDIA GPU

#### 1. Install system dependencies and CUDA

```bash
sudo apt update
sudo apt install build-essential git cmake ninja-build python3 python3-venv \
  libhwloc-dev libzstd-dev libz-dev curl ca-certificates
```

Install an NVIDIA driver compatible with your GPU and CUDA version and
[NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

You may also use conda with

```bash
conda create -n maigo python=3.11 -y
conda activate maigo
conda install -c nvidia/label/cuda-12.6.0 cuda
```

#### 2. Build the CUDA adapter for Intel oneAPI

Install [Intel oneAPI Base Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html)](<https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html>).

```
export CUDA_ROOT=/usr/local/cuda-12.6  # Conda CUDA using "$CONDA_PREFIX"
export SYCL_WORKSPACE="$HOME/sycl_workspace"
mkdir -p "$SYCL_WORKSPACE"
cd $SYCL_WORKSPACE

git clone --depth 1 --branch v7.1.1 \
  https://github.com/intel/llvm.git \
  $HOME/sycl_workspace/llvm-7.1.1

CC=/usr/bin/gcc CXX=/usr/bin/g++ \
cmake \
  -S $HOME/sycl_workspace/llvm-7.1.1/unified-runtime \
  -B $HOME/sycl_workspace/ur-build-7.1.1 \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUR_BUILD_ADAPTER_CUDA=ON \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DUR_USE_EXTERNAL_UMF=ON \
  -Dumf_DIR=$HOME/intel/oneapi/umf/1.1/lib/cmake/umf \
  -DUR_BUILD_TESTS=OFF \
  -DUR_BUILD_EXAMPLES=OFF \
  -DUR_BUILD_TOOLS=OFF \
  -DUR_ENABLE_SANITIZER=OFF

cmake --build $HOME/sycl_workspace/ur-build-7.1.1 \
  --target ur_adapter_cuda -j 16

source "$HOME/intel/oneapi/setvars.sh"
export CXX="$(command -v icpx)"
export UR_ADAPTERS_SEARCH_PATH="$SYCL_WORKSPACE/ur-build-7.1.1/lib"
export CUDA_DISABLE_PTX_JIT=1
unset SYCL_DEVICE_FILTER UR_ADAPTERS_FORCE_LOAD

ONEAPI_DEVICE_SELECTOR=cuda:gpu sycl-ls
```

`sycl-ls` must list an NVIDIA CUDA GPU before proceeding. This requires both
the CUDA runtime adapter and a working NVIDIA driver. A preinstalled `icpx`
may also be used if its CUDA backend is available; the compiler name alone
does not establish CUDA support.

#### 3. Compile

```bash
cd "$MAIGO_ROOT"
export CUDA_DISABLE_PTX_JIT=1
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="--cuda-path=$CUDA_ROOT" \
  -DCARBON_CUDA_ARCH=sm_75
cmake --build build --target carbon_mc -j 4
```

### 2. For Intel Arc GPU

#### 1. Install dependencies

Install the [Intel GPU compute runtime](https://github.com/intel/compute-runtime),
firmware, `ocloc`, and the [oneAPI DPC++/C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html),
plus Git, CMake 3.22+, Ninja, Python and curl. Add your account to the `render` group
and log in again.

#### 2. Compile

```bash
cd "$MAIGO_ROOT"
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CUDA_DISABLE_PTX_JIT=1
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
# bmg-g21: Arc B580; set the ocloc device identifier for your GPU.
cmake -S . -B build-intel -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fsycl-max-parallel-link-jobs=4" \
  -DCARBON_SYCL_TARGETS=spir64_gen -DCARBON_INTEL_GPU_ARCH=bmg-g21
cmake --build build-intel --target carbon_mc -j 4
build-intel/carbon_mc --help
```

For B580 minibeam runs, set the Xe preemption timeout after each reboot
(adjust `card0` to your GPU):

```bash
ARC_ENGINE=/sys/class/drm/card0/device/tile0/gt0/engines/ccs
echo 10000000 | sudo tee "$ARC_ENGINE/preempt_timeout_us"
```

### 3. Download physics assets

Download and extract the physics assets from the
[v2026.10.20 Release](https://github.com/vvuvv31/MAIGO/releases/tag/v2026.10.20):

```bash
export ASSET_DIR="$(dirname "$MAIGO_ROOT")/maigo_assets"
export RELEASE_URL=https://github.com/vvuvv31/MAIGO/releases/download/v2026.10.20
mkdir -p "$ASSET_DIR"
curl --fail --location --retry 3 "$RELEASE_URL/maigo_four_ct_assets.tar.gz" \
  --output "$ASSET_DIR/maigo_four_ct_assets.tar.gz"
tar -xzf "$ASSET_DIR/maigo_four_ct_assets.tar.gz" -C "$MAIGO_ROOT"
```

The bundle includes physics assets for all eight presets and LET scoring.
Case inputs are supplied separately.

### 4. Prepare case inputs

Place CT grids, source plans, beam parameters and spot weights at the paths
specified in [examples/](examples/). Keep each case's inputs and normalization
together. Case inputs are supplied separately from the asset bundle.

### 5. Run

Run from the repository root. Restore the toolchain environment above in each
new shell.

**NVIDIA:**

```bash
cd "$MAIGO_ROOT"
export CUDA_DISABLE_PTX_JIT=1
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
build/carbon_mc --config examples/proton_ct_broadbeam.yaml --histories 10000
```

**Intel Arc:**

```bash
cd "$MAIGO_ROOT"
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CUDA_DISABLE_PTX_JIT=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
mkdir -p tmp/arc_run
sed 's/^device: cuda$/device: intel/' examples/proton_ct_broadbeam.yaml \
  > tmp/arc_run/proton_ct_broadbeam.yaml
build-intel/carbon_mc --config tmp/arc_run/proton_ct_broadbeam.yaml --histories 10000
```

**Continuous batches** (for Arc, use `build-intel/carbon_mc` and the Intel YAML):

```bash
build/carbon_mc --config examples/proton_ct_broadbeam.yaml \
  --batch-count 3 --histories-per-batch 100000 \
  --batch-output out/proton_ct_broadbeam_batches
```

Use a new output directory for each run. Check `quality_report.json` and
`energy_ledger.json` for successful audits and zero overflow. Batch dose is
already normalized; do not multiply it by the batch count.

## Release presets and physics

| Preset | Source/ion path | Ionization deposition |
|---|---|---|
| proton_water_broadbeam | Flat source → Gaussian FE → Water_75eV | Local deposition |
| carbon_water_broadbeam | Carbon source → Gaussian FE → Water_75eV | Same policy |
| proton_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → proton CT kernel | Local deposition |
| carbon_ct_broadbeam | RT07575 head TPS CSV → Gaussian FE → C12 CT kernel | Same policy |
| proton_water_minibeam | Flat source → compact copper/air → Urban water | Local deposition; nuclear neutrals untracked |
| carbon_water_minibeam | TPS CSV → compact copper/air → Urban water | Local deposition |
| proton_ct_minibeam | Flat source → compact copper/air → Urban primary / FE charged secondaries in CT | Local deposition |
| carbon_ct_minibeam | Flat source → compact copper/air → Urban primary / FE charged secondaries in CT | Local deposition; dose accuracy deferred |

Dedicated CT minibeams use Urban for patient primaries and segmented
Fermi–Eyges (FE) for patient-medium charged secondaries, including charged
products entering from upstream. The patient step cap remains 0.5 mm; FE
subsegments follow `fermi_eyges_max_segment_mm` (release default 0.1 mm),
with the existing FE tail treatment. Copper/air workers keep their existing
scattering models; water and generic minibeam medium workers retain Urban.
The YAML `multiple_scattering_model: urban_v2` therefore describes the CT
minibeam primary model, not the secondary model.

Ion step caps are forced during configuration loading: **broadbeam 1.0 mm**,
**minibeam 0.5 mm**, and **minibeam copper 0.05 mm**, for both water and CT.
These effective caps override the corresponding YAML values.
Carbon CT minibeam slit air defaults to a separate 0.2 mm cap; copper remains
0.05 mm and the patient cap remains 0.5 mm.
The dedicated proton and carbon CT minibeam beamlines continue native steps in segments,
regrouping live tracks by current material and energy while preserving both
Urban states, RNG identity, clocks and ledgers.
The proton CT minibeam default primary pool is 1,048,576 histories; explicit
user chunk sizes remain authoritative.
Direct charged children of carbon CT primary copper reactions also continue in
64-native-step segments, regrouped by species, energy and current Cu/air region.
The original RNG steps, Urban state, shared nuclear optical depth and energy
bookkeeping are retained.

Release presets retain primary energy-loss straggling and disable secondary
straggling by default (`enable_secondary_energy_straggling: false`). Minibeams
also disable Cu/air fragment full-loss straggling; Cu/air primary fluctuations remain
on. There is no separate hard-delta draw. Scoped RT07575 CT
validation is recorded for proton broadbeam/minibeam and carbon broadbeam;
**carbon minibeam is under verification**. This status does not establish
water/other-patient acceptance.

All implemented medium and Cu/air loss paths now use unrestricted stopping;
fluctuating tracks use a single full Universal/Ion draw and
range-based stepping. Earlier dense-tail Gamma and restricted-range performance
reports describe historical implementations.
Carbon CT broadbeam also precomputes elastic totals for primary and fragment
species.
Proton/carbon CT minibeams reuse exact actual-density material fields and
elastic total caches for proton primary and carbon primary/fragments.

Each row has the matching `examples/<preset>.yaml`. Default CT minibeams are lung
benchmarks, **not RT07575 head full plans**. Broadbeam caps ion steps at 1.0 mm
and minibeam at 0.5 mm in both water and CT.
Minibeam copper retains its separate 0.05 mm cap. All use continuous26 ion EM;
ionization loss is deposited locally in the medium. Explicit electron/photon
transport, birth/joint response packages, queues and configuration keys are
removed. Water/CT registered ions now use native unrestricted electronic dE/dx
for total mean loss, LET, CSDA range and Urban loss queries. The range is derived
from the existing continuous26 full-response nodes, with continuous actual-density
corrections. Cu/air require the new full electronic response
`data/stopping/beamline/unrestricted.bin` (`BLEM0001`, 324 species). Older asset
archives need this package added; its extraction recipe is in
[extensions/beamline_em](extensions/beamline_em/README.md). One Universal/Ion draw uses the full mean, unrestricted dispersion and the
kinematic transfer limit Tmax. Medium stepping uses residual-range and mean-loss
limits, without an electron-production-cut clock or delta-moment sidecar.
Primary fluctuations remain enabled. The EMCNT001 reader validates the fixed
disk layout, then retains compact runtime nodes/records/descriptors; unused
restricted/rate fields and spline coefficients do not enter GPU loss queries.
Generic recoils retain unrestricted electronic+nuclear total stopping and use a
consistent derived CSDA. Cu/air electronic losses remain beamline energy sinks.

`src/main.cpp` handles startup, `cli.cpp` parses the small CLI, `run_session.cpp`
owns source preparation/context reuse/output, `batch_run.cpp` allocates the
campaign, and `io.cpp` writes dose/physical audits. `source_pose.cpp` retains
only the geometry/air-loss helpers needed by TPS CSV sources. Transport and
current package readers remain in the shared physics core and `src/detail/`.

There are no archived readers, old density banks, per-batch SHA256 checks, DICOM-series
conversion, TOPAS text-channel plan reader, sequential spot replay, old CLI
aliases, diagnostic build modes or alternative precision scorers. Unsupported
configuration keys fail before transport. Mandatory quality/energy/coverage,
completion and overflow audits remain. Internally required fragments remain
part of the current physics. Optional patient/phantom electronic LETd scoring
uses `letd_output_prefix: letd` and, with medium all-ion elastic,
`letd_recoil_electronic_stopping_file: data/let/schneider/recoil_electronic_stopping.bin`,
with independent additive moments and no change
to transport or dose normalization; release presets leave it unset.

All eight release presets enable medium nuclear elastic transport for primary and
charged secondary particles using projectile-specific `all_ion_elastic_file` and
`elastic_recoil_stopping_file` packages under `data/elastic/schneider/{proton,carbon}/`.
All four minibeams also set `minibeam_copper_enable_elastic: true` for the copper collimator.
Primary and patient-medium charged secondary transport share the nuclear optical-depth
sampler: elastic and inelastic rates compete through one clock per track. Secondary
continuation preserves the remaining depth and draw counter; the inelastic generation
limit does not disable elastic scattering.

All four minibeams share an unrestricted Cu/air fluctuation engine.
Particle identity and mass select Universal/Ion parameters; full native material
response supplies the mean loss, variance and CSDA. Primary fluctuations remain
on, fragment fluctuations default off. No finite-cut mean selector, separate
hard-delta sampler or air ionization clock remains. The retired model, scale,
mean and stopping-CSV keys are rejected, rather than mapped to old behavior.
`minibeam_em_package_file` selects the required full response.
`minibeam_copper_fragment_native_transport` independently selects Cu fragment
Urban/species elastic transport (carbon defaults `true`; proton `false`).
Other primary ions use the generic pipeline with matching packages and explicit
source masses; charged fragment transport remains generic.
See [Cu ion extraction and limits](extensions/copper_ion_transport/README.md).

The data/inputs are local external assets and are excluded from Git. Expanded
preset dependencies are in `examples/preset_dependencies.json`. No default data
paths point to an archive.

Data directories follow the physics modules: `materials/`, `em/`, `nuclear/`,
`elastic/`, `stopping/`, `scattering/` and `ionization/`. Particle and material
subdirectories distinguish independent calibrations. Dates, generator versions
and binary format versions belong in metadata rather than runtime filenames.
RT07575 head inputs are in `inputs/rt07575_head/{proton,carbon}/`; the separate
lung benchmark is in `inputs/lung_minibeam/`. Research notes and historical
validation reports in `docs/` are local archives and are not included in Git.

After an intentional asset change, refresh dependencies with
`python3 scripts/list_preset_dependencies.py --output examples/preset_dependencies.json`.

Validation scripts require Python 3, NumPy and PyYAML.

Public headers are directly under `include/`; `include/detail/` holds internal
header helpers. Transport implementation phases remain in `src/detail/`.

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

All four minibeams use compact scheduling. Retired interfaces and zero-only
nuclear output layouts are removed.
Nuclear channel support flags and energy domains are read directly from binary packages.
Nuclear neutrals are not transported; their energy remains in explicit sinks.
Neutron packages, birth captures, transport settings and dose outputs are removed.
Explicit electromagnetic electron/photon transport was subsequently removed.
The proton water minibeam no longer includes the former neutron-dose contribution.
TPS isocenter input uses only the supported `tps_isocenter_z_mm` key.

For a matched implementation regression, explicitly select both a frozen
control executable with compatible physics/configuration and the candidate
executable you just built. Each root must contain its corresponding presets
and assets. Load the compiler's shared-library environment before running;
the checks inherit it without substituting a machine-specific library path.

```bash
CUDA_DISABLE_PTX_JIT=1 python3 scripts/validate_presets.py \
  --control /path/to/frozen/carbon_mc --control-root /path/to/frozen/root \
  --candidate "$PWD/build/carbon_mc" --out tmp/validation_new
CUDA_DISABLE_PTX_JIT=1 python3 scripts/check_interfaces.py \
  --binary "$PWD/build/carbon_mc" --output tmp/interface_check_new.json
```

The control is used only for scientific comparison. Small-history matched GPU
checks are not TOPAS gamma acceptance, water dose acceptance or a throughput
measurement.

Water Z geometry uses only `voxel_bins_z` and `voxel_size_z_mm`; both are required.
CT geometry comes from the CCTG header, with explicit voxel fields serving as
consistency assertions. Retired `phantom_length_mm` and `depth_bin_width_mm` YAML
keys are rejected. Internal transport dimensions are derived from the grid.

Primary nuclear rates use only `SCHNRATE` v3. Packages load directly from the
configured binary files; no metadata, channels or bundle JSON is required.
Water `material.json` supplies physical material properties, rather than a
verification sidecar.

The runtime accepts `device: cuda` with `nvptx64-nvidia-cuda` or `device: intel`
with Intel Level Zero and the `spir64_gen` AOT target. Generic GPU/CPU selectors
and vendor aliases are retired; no fallback to another backend occurs.

Offline TOPAS/Geant4 extraction sources and build/package instructions are in
[extensions/README.md](extensions/README.md). They are separate from the GPU runtime.
