# MAIGO_clean

论文主线为 **GPU Monte Carlo: From Broadbeam to Minibeam**。材料与方法：[中文](mm_zh.md) / [English](mm.md)；[文献与写作分析](docs/gpu_mc_methods_literature.md)。

A standalone GPU runtime for the eight current MAIGO presets. The source and
runtime inputs were copied independently from MAIGO_minibeam; there are no
symlinks back to it. Ionization energy is deposited locally; explicit
electron/photon transport and its package-loading interfaces have been removed.

## Build and run

Supported GPU platforms are Linux with NVIDIA CUDA or Intel Arc Level Zero.
MAIGO uses Intel LLVM/DPC++ SYCL with native AOT device images
(`nvptx64-nvidia-cuda` or `spir64_gen`); CPU and AMD backends are not supported.
Choose one backend: **NVIDIA users follow steps 1–3**; **Intel Arc users go
straight to [Intel Arc build and run](#intel-arc-build-and-run)**. Both then follow
steps 4–6 for physics assets, case inputs and execution. Arc does not require CUDA
or a source build of the CUDA-capable LLVM toolchain.

The workflow is: install the GPU driver and matching SYCL toolchain → compile
MAIGO for the destination GPU → restore physics data → supply matching case
inputs → run and inspect the audits. Compilation does not require patient data.

Clone the repository first, for either backend:

```bash
git clone --branch clean --single-branch https://github.com/vvuvv31/MAIGO.git MAIGO_clean
cd MAIGO_clean
export MAIGO_ROOT="$PWD"
```

Use separate build directories for different backends or GPU architectures.
The same executable can run different cases; compilation is not per patient.

### 1. NVIDIA: install system dependencies and CUDA

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

### 2. NVIDIA: build the complete Intel LLVM/SYCL toolchain

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

### 3. NVIDIA: compile MAIGO

```bash
cd "$MAIGO_ROOT"
# -DCARBON_CUDA_ARCH=sm_75: 根据自己显卡设置 / adjust for your GPU.
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

Always specify `-DCARBON_CUDA_ARCH=sm_75` **(根据自己显卡设置 / adjust for your
GPU)** in build commands. Both CMake and the wrapper default to `sm_75`, and
CMake rejects an empty architecture. Use `sm_89` for L4 and `sm_86` for RTX
A6000. The architecture refers to the GPU that will run the binary, which may
be different from the GPU on the build machine. Use a separate build directory
when changing the compiler or GPU architecture.

These builds compile device code ahead of time. Before running, explicitly
prohibit driver PTX JIT fallback:

```bash
export CUDA_DISABLE_PTX_JIT=1
```

Keep this variable set in every execution shell or job script. With JIT disabled,
a missing compatible machine-code image causes an error instead of compiling at
runtime; rebuild for the destination GPU. This does not remove normal startup,
package loading or module loading costs.

All five transport families are enabled by default: proton broadbeam, proton
minibeam, carbon broadbeam, carbon minibeam and generic ion transport. To compile
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

### Intel Arc build and run

Start from the clone above. Install Git, CMake 3.22+, Ninja and Python, then
install the [Intel GPU compute runtime](https://github.com/intel/compute-runtime)
(Level Zero), GPU firmware, the matching `ocloc` offline compiler, and the
[Intel oneAPI DPC++/C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html).
Follow the packages and kernel requirements for your distribution and GPU.
The tested system is Debian 13, Arc B580 12 GB, oneAPI 2026.1 and Xe driver
26.22.38646.4. The user needs access to `/dev/dri/renderD*` (normally via the
`render` group; log in again after changing group membership).

Load the oneAPI environment before compiling or running. Confirm that `sycl-ls`
lists the Intel GPU under Level Zero, then build:

```bash
cd "$MAIGO_ROOT"
source /opt/intel/oneapi/setvars.sh
icpx --version
ocloc --help
# oneAPI 2026.1 + B580 driver 26.22: use the working Level Zero V1 adapter.
export SYCL_UR_USE_LEVEL_ZERO_V2=0
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
cmake -S . -B build-intel -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fsycl-max-parallel-link-jobs=4" \
  -DCARBON_SYCL_TARGETS=spir64_gen -DCARBON_INTEL_GPU_ARCH=bmg-g21
cmake --build build-intel --target carbon_mc -j 4
build-intel/carbon_mc --help
```

`bmg-g21` targets Arc B580; choose the `ocloc` device identifier for your GPU.
An explicit architecture is required. `spir64` runtime-JIT builds are rejected.
On the tested oneAPI 2026.1/B580 setup, the default Level Zero V2 adapter returns
`UR_RESULT_ERROR_UNSUPPORTED_FEATURE` when launching a native kernel; keep
`SYCL_UR_USE_LEVEL_ZERO_V2=0` set when running. This still uses Level Zero and
the precompiled GPU image.
The driver must expose FP64, atomic64 and device USM for the existing transport
and energy audits. Physics packages and source inputs are shared with CUDA.
Intel primary kernels move the immutable nuclear/EM views into device memory
to respect the smaller kernel-argument budget. Intel AOT builds also generate
separate native images per kernel to avoid incorrect CT lookup results observed
with merged images on oneAPI 2026.1/B580. Temporary upload arrays are retained
until their asynchronous transfers complete; physics and RNG remain unchanged.
Keep the eight release YAMLs as CUDA defaults. Step 6 shows how to make a
private `device: intel` copy and run it from the repository root. There is no
CPU or alternative-backend fallback.

**AOT happens during the build, including the final link.** That link compiles
GPU kernels to native machine code and embeds them in the executable; it can
therefore take much longer than an ordinary C++ link. `cmake --build -j 4`
controls build tasks, while `-fsycl-max-parallel-link-jobs=4` controls device
compilation inside the SYCL link. Adjust both to available CPU and RAM; eight
parallel link jobs were also tested on the 62 GiB B580 host. Selective pipeline
compilation can further reduce build work.

Build once and reuse the executable across cases and continuous batches.
Changing CT, spot weights, particle counts or supported YAML parameters does
not require rebuilding. Changes to C++ code, compiled pipeline selection,
toolchain or destination GPU architecture do. A custom C++ full-plan driver
also needs an AOT build once; ordinary `carbon_mc` runs need no extra driver.
Neither build time nor AOT link time belongs in cold-start throughput.

On the tested Xe/B580 system, the compute engine's default **640 ms preemption
timeout** resets longer carbon CT minibeam kernels. Set it to 10 seconds before
running these workloads (root required; this setting resets on reboot):

```bash
# Select the DRM card belonging to the Intel GPU on your machine.
ARC_ENGINE=/sys/class/drm/card0/device/tile0/gt0/engines/ccs
cat "$ARC_ENGINE/preempt_timeout_us"
echo 10000000 | sudo tee "$ARC_ENGINE/preempt_timeout_us"
```

The tested 5,000 ms `job_timeout_ms` can remain unchanged. The executable does
not modify driver settings. Do not reduce physical track limits to work around
a driver timeout. All eight electron-off presets passed two 10,000-history
batches on B580 with the native AOT build; these are runtime/audit checks, not
TOPAS dose acceptance or throughput benchmarks. Electron-on execution on Arc
has not yet been tested. See [Intel Arc validation notes](docs/intel_arc_support_20261015.md).

### 4. Download and restore the physics assets

The [v2026.10.14 Release](https://github.com/vvuvv31/MAIGO/releases/tag/v2026.10.14)
contains the physics assets for the **four CT presets**, including optional
historical electron/photon packages that the current runtime does not use. It is not an eight-preset water/CT asset bundle.
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
The current runtime needs no electron/photon packages. Older release archives
may contain them; omit `data/em/electrons/` when installing new runtime assets.

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

Run from the repository root after supplying the matching inputs. In each new
shell, restore the appropriate toolchain environment from above and set
`MAIGO_ROOT` to your checkout. Select **one** of the following launch setups.

For NVIDIA (with the CUDA-capable SYCL library path configured):

```bash
cd "$MAIGO_ROOT"
export CUDA_DISABLE_PTX_JIT=1
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export MAIGO_BINARY="$MAIGO_ROOT/build/carbon_mc"
export MAIGO_CONFIG="$MAIGO_ROOT/config/proton_ct_broadbeam.yaml"
```

For Arc (after configuring the Xe timeout where required):

```bash
cd "$MAIGO_ROOT"
source /opt/intel/oneapi/setvars.sh
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CUDA_DISABLE_PTX_JIT=1
mkdir -p tmp/arc_run
sed 's/^device: cuda$/device: intel/' config/proton_ct_broadbeam.yaml \
  > tmp/arc_run/proton_ct_broadbeam.yaml
export MAIGO_BINARY="$MAIGO_ROOT/build-intel/carbon_mc"
export MAIGO_CONFIG="$MAIGO_ROOT/tmp/arc_run/proton_ct_broadbeam.yaml"
```

This private copy changes only the requested device. Relative data/input paths
resolve from the repository root; keep running there. Use the same approach
for other presets, or supply your own complete case configuration.

Start with a small execution check, using the selected backend:

```bash
"$MAIGO_BINARY" --config "$MAIGO_CONFIG" --histories 10000
```

Continuous batches share one GPU context and loaded physics packages:

```bash
"$MAIGO_BINARY" --config "$MAIGO_CONFIG" \
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
Custom configurations without that key use `generic`; a copied
release preset retains its selection until the key is removed or changed.
Pipeline layout is documented in [transport pipelines](docs/transport_pipelines.md).

Other primary ions must select `transport_pipeline: generic` (or omit the key),
provide `primary_atomic_number`, `primary_mass_number`, and explicitly set both
`primary_kinematic_mass_MeV` and `primary_nuclear_mass_MeV` to masses from their
corresponding extraction conventions (total ion rest energies in MeV, not per
nucleon). Proton/C12 retain their built-in masses;
no changes to the eight release presets are needed. Source parsing has no species
whitelist, but all required EM, nuclear, scattering and elastic packages must
cover the chosen ion and energy range, and match its reference masses. Existing
packages are not expanded by this change; for example, the current continuous26
bank includes He4 but not O16, and He4 still needs matching primary nuclear data.

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
minibeam primary model, not the secondary model. See the
[full-plan comparisons and adoption scope](docs/ct_minibeam_secondary_fe.md).

Ion step caps are forced during configuration loading: **broadbeam 1.0 mm**,
**minibeam 0.5 mm**, and **minibeam copper 0.05 mm**, for both water and CT.
These effective caps override the corresponding YAML values.
Carbon CT minibeam slit air defaults to a separate 0.2 mm cap; copper remains
0.05 mm and the patient cap remains 0.5 mm. See the
[slit-air timing and dose comparison](docs/carbon_minibeam_slit_air_20261016.md).
The dedicated proton and carbon CT minibeam beamlines continue native steps in segments,
regrouping live tracks by current material and energy while preserving both
Urban states, RNG identity, clocks and ledgers. See the
[matched two-GPU timing and implementation checks](docs/carbon_minibeam_beamline_continuation_20261016.md).
The proton CT minibeam default primary pool is 1,048,576 histories; explicit
user chunk sizes remain authoritative. See the
[three-case proton timing and implementation checks](docs/proton_minibeam_beamline_continuation_20261016.md).
Direct charged children of carbon CT primary copper reactions also continue in
64-native-step segments, regrouped by species, energy and current Cu/air region.
The original RNG steps, Urban state, shared nuclear optical depth and energy
bookkeeping are retained. See the
[Cu-child scheduling comparison](docs/carbon_minibeam_cu_child_continuation_20261016.md).

Release presets retain primary energy-loss straggling and disable secondary
straggling by default (`enable_secondary_energy_straggling: false`). Minibeams
also disable Cu/air fragment full-loss straggling; Cu/air primary fluctuations remain
on. There is no separate hard-delta draw. Scoped RT07575 CT
validation is recorded for proton broadbeam/minibeam and carbon broadbeam;
**carbon minibeam is under verification**. This status does not establish
water/other-patient acceptance. See [validation scope and measured effects](docs/secondary_straggling.md).

All implemented medium and Cu/air loss paths now use unrestricted stopping;
fluctuating tracks use a single full Universal/Ion draw and
range-based stepping. Earlier dense-tail Gamma and restricted-range performance
reports describe historical implementations; see the [current model and checks](docs/unrestricted_dedx.md).
Carbon CT broadbeam also precomputes elastic totals for primary and fragment
species; see [scope, paired timings and validation limits](docs/carbon_broadbeam_elastic_totals_20261013.md).
Proton/carbon CT minibeams reuse exact actual-density material fields and
elastic total caches for proton primary and carbon primary/fragments. Historical
restricted-range optimization measurements are recorded in the
[matched full-plan report](docs/minibeam_ct_cache_optimization_20261013.md).

Each row has the matching `config/<preset>.yaml`. Default CT minibeams are lung
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
consistent derived CSDA. Cu/air electronic losses remain beamline energy sinks. See
[unrestricted transport](docs/unrestricted_dedx.md) and
[electron removal scope](docs/no_electron_transport_20261015.md).

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
to transport or dose normalization; release presets leave it unset. See
[LET scoring and the matched TOPAS scorer](docs/let_scoring.md).
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

All four minibeams share an [unrestricted Cu/air fluctuation engine](docs/copper_fluctuation_engine.md).
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
```

After an intentional asset change, regenerate the manifest with
`python3 scripts/check_data_manifest.py --write` and refresh dependencies with
`python3 scripts/list_preset_dependencies.py --output docs/preset_dependencies.json`.

Validation scripts require Python 3, NumPy and PyYAML.

For the runtime's include structure, scheduling states, units and dose-accounting
contracts, see the [Chinese core code review guide](docs/core_code_review_zh.md).

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
Explicit electromagnetic electron/photon transport was subsequently removed. See
`docs/no_neutral_20261007.md` for removal scope and matched regression evidence. The proton water minibeam
therefore no longer includes the former neutron-dose contribution.
TPS isocenter input uses only the supported `tps_isocenter_z_mm` key.

Current three-case full-plan GPU/TOPAS gamma, per-device throughput, statistical
uncertainty and error-distribution figures are collected in the
[2026-10-17 validation report](docs/full_plan_validation_20261017/README.md).
Carbon broadbeam TOPAS uncertainties are explicitly marked as estimates from
matched low-statistics reruns; the original reference doses are retained.

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
python3 scripts/check_release_inputs.py --output tmp/release_inputs_new.json
CUDA_DISABLE_PTX_JIT=1 python3 scripts/check_package_contracts.py \
  --binary "$PWD/build/carbon_mc" --output tmp/package_contracts_new.json
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

The runtime accepts `device: cuda` with `nvptx64-nvidia-cuda` or `device: intel`
with Intel Level Zero and the `spir64_gen` AOT target. Generic GPU/CPU selectors
and vendor aliases are retired; no fallback to another backend occurs.

See `docs/release_cleanup6_20261007.md` for CUDA-only migration and matched
eight-preset regression evidence.

Offline TOPAS/Geant4 extraction sources and build/package instructions are in
[extensions/README.md](extensions/README.md). They are separate from the GPU runtime.
