# MAIGO

Carbon-ion **condensed-history Monte Carlo** for dose and dose-averaged LET  
on **Intel oneAPI / SYCL** (Intel Level Zero or NVIDIA CUDA plugin).

Research software only — **not for clinical treatment planning**.

## What this repository is

A slim core tree: engine sources, build system, physics tables, and a few example
configs. Validation suites, patient CT cases, and development notes live on
other branches (e.g. `master`), not here.

```text
include/carbon/     Public headers (config, transport, tables, geometry)
src/                Implementation (serial + SYCL backends)
  detail/           Shared SYCL device helpers (.inc)
config/             Example beam configs
data/               Stopping power, cross sections, reaction packages
scripts/            Build helpers
CMakeLists.txt
CMakePresets.json
```

## Features

| Area | Capability |
|------|------------|
| Transport | CSDA + straggling + Highland MCS; primary attenuation |
| Secondaries | Reaction packages, charged fragment queue, cascade (≤2 gen) |
| Neutrals | Optional neutron/gamma queue (package-driven) |
| Geometry | Uniform water, axial slabs, hetero inserts, CT voxel grid |
| Beam | Mono/multi energy, emittance, TOPAS spots, TPS source |
| Scoring | Depth dose, dense MHD, LET_d moments, fragment species |
| Backends | Serial CPU subset; SYCL full path |
| Optional | Copper minibeam beamline (`CARBON_ENABLE_MINIBEAM`) |

## Requirements

- CMake ≥ 3.22, C++20 compiler  
- **Serial only**: any modern C++20 toolchain  
- **SYCL**: Intel oneAPI `icpx` (IntelLLVM)  
- GPU: Intel Arc (Level Zero) and/or NVIDIA (CUDA plugin + `nvptx64-nvidia-cuda`)

## Build

```bash
# Optional: oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Serial debug
cmake --preset cpu-debug && cmake --build --preset cpu-debug

# NVIDIA SYCL (legacy CT/water/LET kernel, FP32 dose)
cmake --preset oneapi-nvidia-release && cmake --build --preset oneapi-nvidia-release

# NVIDIA + Copper minibeam dual kernel
cmake --preset oneapi-nvidia-minibeam && cmake --build --preset oneapi-nvidia-minibeam

# Or helper script (Linux dual targets)
scripts/build_linux_oneapi.sh
```

Binaries land under `build/<preset>/carbon_mc`.

### CMake options

| Option | Default | Meaning |
|--------|---------|---------|
| `CARBON_ENABLE_SYCL` | OFF | SYCL transport |
| `CARBON_ENABLE_MINIBEAM` | OFF | Compile Copper minibeam path + dispatch |
| `CARBON_DOSE_FP32` | OFF* | FP32 dose atomics (*NVIDIA presets ON) |
| `CARBON_SYCL_TARGETS` | empty | e.g. `nvptx64-nvidia-cuda` or `spir64,nvptx64-nvidia-cuda` |
| `CARBON_CUDA_ARCH` | empty | Optional AOT, e.g. `sm_75` |

With **MINIBEAM=OFF**, only the legacy SYCL kernel is built.  
With **MINIBEAM=ON**, `minibeam: false` still uses the legacy kernel;  
`minibeam: true` selects the Copper beamline path.

## Run

```bash
# Serial CSDA water phantom
./build/cpu-debug/carbon_mc --config config/beam_200MeVu.yaml

# SYCL + cascade (needs GPU / correct --device)
./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_200MeVu_cascade.yaml \
  --device cuda

# LET_d example
./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_200MeVu_letd.yaml \
  --device cuda
```

Useful CLI flags: `--histories`, `--device`, `--output`, `--physics-profile`,
`--scorer-let` / `--no-scorer-let`, `--spots`, `--ct-grid`.

Config files are simple `key: value` lines (not a full YAML library).

## Physics data

| Path | Role |
|------|------|
| `data/stopping_power_*.csv` | Material dE/dx tables |
| `data/c12_inelastic_cross_sections_*.csv` | Macroscopic nuclear XS |
| `data/ion_stopping_power_*.csv` | Per-isotope SP for LET / fragments |
| `data/packages/*.bin` | Reaction, cascade, neutral, soft-tissue packages |
| `data/copper_*.bin` | Minibeam Copper packages |

Package binaries are precompiled event samples. Regenerate them offline from
TOPAS/Geant4 if you need different energies or materials; this branch does not
ship the generation pipeline.

## Layout (engine)

```text
CLI / config
    → TransportConfig
    → load SP / XS / packages / CT / spots
    → transport_serial  or  transport_sycl
         ├─ legacy (water / CT / LET / TPS)
         └─ minibeam (optional Copper beamline)
    → TransportResult → CSV / MHD writers
```

Shared device helpers used by both SYCL TUs live in `src/detail/sycl_*.inc`
(include-only; dual-kernel isolation preserved).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
