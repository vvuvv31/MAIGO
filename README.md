# MAIGO

Carbon-ion condensed-history Monte Carlo for dose and dose-averaged LET
on Intel oneAPI / SYCL (Intel Level Zero or NVIDIA CUDA plugin).

> **Research software only** -- not for clinical treatment planning.

## Features

- **Condensed-history transport** -- CSDA + Vavilov energy straggling + Highland multiple Coulomb scattering; primary nuclear attenuation.
- **Nuclear interactions** -- data-driven reaction packages extracted from TOPAS / Geant4, charged-fragment queue with cascade (up to 2 generations), optional neutral (neutron / gamma) queue.
- **Geometry** -- uniform water phantom, axial slab stacks, heterogeneous bone / tissue inserts, full CT voxel grids (HU-to-material conversion).
- **Beam models** -- mono- / multi-energy pencil beams with Gaussian emittance, TOPAS spot files, TPS plan source with arbitrary gantry angles.
- **Scoring** -- 3-D dose-to-medium, dose-averaged LET (LET_d) with numerator / denominator moments, dense MHD voxel output, fragment-species fluence, per-spot PBS scoring.
- **Backends** -- serial C++ (subset) and full SYCL GPU transport; supports Intel Arc (Level Zero / SPIR-V) and NVIDIA GPUs (CUDA plugin / PTX).
- **Optional Copper minibeam** -- dedicated beamline kernel (`CARBON_ENABLE_MINIBEAM`).
- **FP32 / FP64 dose atomics** -- FP32 default for fast consumer-GPU atomics; FP64 available via CMake option.

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| CMake | 3.22 | |
| C++ standard | C++20 | |
| Ninja | any (recommended) | Falls back to Unix Makefiles |
| **Serial-only build** | GCC 12+, Clang 16+, or MSVC 2022 | No GPU required |
| **SYCL build** | Intel oneAPI DPC++ (`icpx` / IntelLLVM) | See compiler setup below |
| NVIDIA GPU | Compute capability 7.5+ (e.g. RTX 2080 Ti) | Requires the oneAPI CUDA plugin |
| Intel GPU | Arc / Data Center GPU (Level Zero) | SPIR-V JIT |

### Building Intel LLVM (DPC++) from Source

If a packaged oneAPI toolkit is unavailable or you need the latest CUDA plugin
support, build the Intel LLVM SYCL compiler from source:

```bash
# 1. Clone the Intel LLVM fork
git clone https://github.com/intel/llvm.git intel-llvm
cd intel-llvm

# 2. Configure with CUDA support
#    Adjust --cuda-sdk-root to your local CUDA toolkit path.
python buildbot/configure.py \
  --cuda \
  --cuda-sdk-root /usr/local/cuda \
  -o build

# 3. Build (takes a while)
cd build
ninja sycl-toolchain

# 4. (Optional) Install to a prefix
ninja install

# 5. Put the compiler on PATH
export PATH=$(pwd)/bin:$PATH
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

# Verify
icpx --version   # should report IntelLLVM / oneAPI DPC++
```

If you already have a packaged Intel oneAPI toolkit, source the environment
instead:

```bash
source /opt/intel/oneapi/setvars.sh
```

## Build

The project ships CMake presets for common configurations. Ninja is the default
generator.

```bash
# Serial CPU debug
cmake --preset cpu-debug && cmake --build --preset cpu-debug

# NVIDIA SYCL (FP32 dose, legacy CT / water / LET kernel)
cmake --preset oneapi-nvidia-release && cmake --build --preset oneapi-nvidia-release

# NVIDIA SYCL + Copper minibeam kernel
cmake --preset oneapi-nvidia-minibeam && cmake --build --preset oneapi-nvidia-minibeam

# Intel SPIR-V only
cmake --preset oneapi-intel-release && cmake --build --preset oneapi-intel-release

# Dual targets (Intel + NVIDIA)
cmake --preset oneapi-release && cmake --build --preset oneapi-release
```

Or use the helper script:

```bash
scripts/build_linux_oneapi.sh                       # default: oneapi-release
scripts/build_linux_oneapi.sh oneapi-nvidia-release  # NVIDIA only
```

Binaries are placed under `build/<preset>/carbon_mc`.

### Manual CMake Invocation

```bash
source /opt/intel/oneapi/setvars.sh   # or use the from-source compiler

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda \
  -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=ON

cmake --build build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CARBON_ENABLE_SYCL` | `OFF` | Enable the SYCL transport backend |
| `CARBON_DOSE_FP32` | `ON` | Use FP32 dose scorer atomics (fast on consumer NVIDIA GPUs) |
| `CARBON_DOSE_FP64` | `OFF` | Use FP64 dose scorer atomics (overrides FP32 when ON) |
| `CARBON_LET_FP64` | `OFF` | Use FP64 LET scorer atomics |
| `CARBON_ENABLE_MINIBEAM` | `OFF` | Build the Copper minibeam beamline kernel |
| `CARBON_VALIDATION_SCORERS` | `OFF` | Compile extra validation scorers (species fluence, survival, ledger) |
| `CARBON_ENABLE_TRANSPORT_PROFILE` | `OFF` | Enable low-overhead SYCL step-path profiling counters |
| `CARBON_SYCL_TARGETS` | *(empty)* | SYCL compilation targets, e.g. `nvptx64-nvidia-cuda`, `spir64`, or both |
| `CARBON_CUDA_ARCH` | *(empty)* | NVIDIA GPU architecture for AOT compilation, e.g. `sm_75` |

With `CARBON_ENABLE_MINIBEAM=OFF` only the legacy SYCL kernel is built. With
`CARBON_ENABLE_MINIBEAM=ON`, setting `minibeam: false` in the config still
uses the legacy kernel while `minibeam: true` selects the Copper beamline path.

## How to Run

```bash
# Basic water phantom (serial)
./build/cpu-debug/carbon_mc --config config/beam_200MeVu.yaml

# SYCL on NVIDIA GPU with fragment cascade
./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_200MeVu_fragment_cascade_100k.yaml \
  --device cuda

# Dose-averaged LET
./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_200MeVu_letd.yaml \
  --device cuda

# CT full-plan (PBS dose + LET)
./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_ct_fullplan_rt07575_let.yaml \
  --device cuda
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config FILE` | Configuration file (simple `key: value` format) |
| `--device DEVICE` | `serial`, `cpu`, `gpu`, `cuda` / `nvidia`, `level_zero` / `intel` / `arc`, `opencl`, `default` |
| `--histories N` | Number of histories (per spot, or total plan with spot weights) |
| `--spots FILE` | TOPAS-format spot file; repeat to concatenate |
| `--spot-weights FILE` | One optimization weight per concatenated spot |
| `--random-seed N\|auto` | Override configured RNG seed |
| `--ct-grid FILE` | Override configured CT patient grid |
| `--output FILE` | Energy-deposition scorer CSV |
| `--dose-output FILE` | Dose scorer CSV (Gy) |
| `--scorer-let` / `--no-scorer-let` | Enable / disable LET_d scoring |
| `--let-output FILE` | LET_d CSV with raw numerator / denominator |
| `--voxel-dose-mhd FILE` | Dense voxel dose MHD output path |
| `--plan-only` | Parse and allocate the plan without running transport |
| `--sequential-spots` | Disable batched SYCL plan launch |
| `--write-canonical-config FILE` | Write normalized YAML input |

Config files use simple `key: value` lines (a lightweight parser, not a full
YAML library).

## License

GPL-3.0-or-later -- see [LICENSE](LICENSE).
