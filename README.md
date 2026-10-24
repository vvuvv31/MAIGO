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

Install [Intel oneAPI Base Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html)](<https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html>).

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
[latest Release](https://github.com/vvuvv31/MAIGO/releases/latest):

```bash
export ASSET_DIR="$(dirname "$MAIGO_ROOT")/maigo_assets"
export RELEASE_URL=https://github.com/vvuvv31/MAIGO/releases/latest/download
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

Eight configurations are provided in [examples/](examples/):

| Medium | Beam | Proton | Carbon-12 |
|---|---|---|---|
| Water | Broadbeam | [YAML](examples/proton_water_broadbeam.yaml) | [YAML](examples/carbon_water_broadbeam.yaml) |
| CT | Broadbeam | [YAML](examples/proton_ct_broadbeam.yaml) | [YAML](examples/carbon_ct_broadbeam.yaml) |
| Water | Minibeam | [YAML](examples/proton_water_minibeam.yaml) | [YAML](examples/carbon_water_minibeam.yaml) |
| CT | Minibeam | [YAML](examples/proton_ct_minibeam.yaml) | [YAML](examples/carbon_ct_minibeam.yaml) |

CT broadbeam examples use the RT07575 head plan. CT minibeam examples use a
separate lung benchmark. For patient full-plan calculations, keep the case's
CT, source, spot weights, placement, collimator and normalization together.

All presets use unrestricted electronic stopping with local ionization energy
deposition, primary energy-loss fluctuations, and nuclear elastic and inelastic
interactions. Secondary energy-loss fluctuations are disabled by default.
Water/CT short steps use midpoint-energy stopping; larger losses use CSDA
range inversion.
Charged nuclear fragments are transported; nuclear neutrals are recorded as
untracked energy. The water reference is Water_75eV (I = 75 eV).

Broadbeams use Fermi–Eyges scattering with a Gaussian core and independently
sampled rare tails. CT minibeams use Urban scattering for patient primaries
and the same segmented Fermi–Eyges model for patient charged secondaries;
water minibeams use Urban. The carbon water preset loads a joint Urban response
(`urban_water_condensed_response_file`) for covered C12 steps; other states retain
ordinary Urban. Minibeams explicitly model the copper collimator
and upstream air. Proton and carbon minibeams share species-resolved Urban
scattering and nuclear elastic transport for charged fragments in copper.

Maximum ion steps are **1.0 mm for broadbeams**, **0.5 mm for minibeams** and
**0.05 mm in copper**. These caps override YAML step-cap values. Dose to medium
is scored, with optional dose-averaged electronic LET.
For LET scoring, set `letd_output_prefix` and
`letd_recoil_electronic_stopping_file: data/let/schneider/recoil_electronic_stopping.bin`
in the run YAML.

Physics assets are organized under `data/` by module: `materials/`, `em/`,
`nuclear/`, `elastic/`, `stopping/`, `scattering/` and `let/`.
Required files are listed in [preset dependencies](examples/preset_dependencies.json).
Model details are in [Materials and Methods](mm.md) ([中文](mm_zh.md)); offline
TOPAS/Geant4 data extraction is described in [extensions/](extensions/README.md).
