# Materials and methods

Revision: 2026-10-05. [中文](mm_zh.md). This describes the eight-preset cleanup and CT scheduling release. See the [cleanup regression](docs/repository_cleanup_20261005.md) and [CT throughput/acceptance report](docs/ct_throughput_20261005.md). Both CT broadbeams pass the requested dose gate; lung proton minibeam still falls below it. Historical measurements apply to their recorded executable, source, geometry and statistics.

## Eight presets

The supported configuration matrix contains exactly eight entries. Presets select projectile and geometry; electron propagation is shared.

| Preset in `config/` | Ion scattering | Electron birth selection | Current verification |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Current 3D local 1%/1 mm 99.9254%; passed |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Current 3D local 1%/1 mm 99.8677%; passed |
| `proton_ct_minibeam.yaml` | Urban | All primary ionization in CT | Shared Urban/FP32 logic; lung CT does not yet pass the new >99% gate |
| `carbon_ct_minibeam.yaml` | Urban | All primary ionization in CT | Scheduling 1.946× faster; transport audits passed; dose pending |
| `proton_water_broadbeam.yaml` | Gaussian Fermi–Eyges | Empty selection in unit-density water | Two-batch smoke passed; water campaign deferred |
| `carbon_water_broadbeam.yaml` | Gaussian Fermi–Eyges | Empty selection in unit-density water | Two-batch smoke passed; water campaign deferred |
| `proton_water_minibeam.yaml` | Urban | All primary ionization in water | Shared-worker smoke passed; water dose acceptance pending |
| `carbon_water_minibeam.yaml` | Urban | All primary ionization in water | Shared-worker smoke passed; water dose acceptance pending |

The two CT broadbeam presets use RT07575 plan inputs. The CT minibeam presets use the lung CT, with 170 MeV protons and 250 MeV/u C12 respectively. These YAMLs are not the RT07575 minibeam full plan; running that plan requires its actual geometry, spots, normalization and reference. Carbon water minibeam retains its 250 MeV/u spot source and 1.2% relative energy spread.

## Physics and geometry

MAIGO is a C++20/SYCL Monte Carlo implementation using TOPAS/Geant4-derived material tables, conditional final states and nuclear event samples. The extraction/reference stack is TOPAS 4.2.p3 with Geant4 11.3.2. These data-driven models and their statistical validation do not establish equivalence to every Geant4 configuration.

Unified EM tables cover water and Schneider material/density nodes for the transported charged species. Condensed ionization loss uses extracted stopping and delta-loss moments. Primary and secondary nuclear channels retain projectile-specific rates and CINEL03 event data; proton and C12 birth distributions must not be interchanged. Copper minibeam collimation retains its own stopping, elastic, nuclear and air data.

The water stopping reference is **Water_75eV**, with mean excitation energy 75 eV. `data/water_unified/g4_water_material.json` supplies composition/radiation-length information; its G4_WATER identity does not change the stopping reference to 78 eV. No empirical stopping rescaling is introduced.

For CT, electrons use the patient material/density grid. For homogeneous water, the same worker receives a single Water_75eV cell spanning the physical water box. This cell is separate from the scoring mesh. Leaving the scoring ROI is distinct from escaping the physical phantom. Creating electron geometry does not enable Schneider CT ion physics in water.

Minibeam primary steps remain at most 0.25 mm; the water carbon preset currently retains its smaller 0.005 mm cap. Ion Urban scattering, independent electron propagation and FP32 dose accumulation are distinct parts of the algorithm. “Urban + FP32 electrons” does not mean replacing the electron elastic law with ion Urban scattering.

## Shared electron transport

The primary ionization loss is partitioned using an energy-weighted electron birth yield for the projectile, material and density. The residual is deposited locally; the electron contribution becomes a weighted task. The current minibeam path does not add explicit secondary-ion electron births or upstream copper/air electron cascades. Nuclear recoil deposits are not folded into the primary birth response.

All four minibeam presets use one independent FP32 queued worker. It samples electron birth energy and direction, propagates charged packets with extracted elastic and hard-interaction data and Urban loss fluctuations, and transports photon branches using the common photon operators. Conditional secondary branches retain energy accounting. Short joint-response advances accelerate covered Schneider states; Water_75eV has no joint operator and uses the same independent worker throughout. There is no fallback to the retired water/material spatial trajectory library.

The production selection is FP32 transport/geometry and FP32 dose atomics. CDF support retains integer draws. Minibeam defaults use group32: weighted reservoir sampling combines primary birth contributions into representative electron tasks. Grouping changes variance and must be assessed with dose statistics, not just task counts.

For broadbeam, the YAML switch is:

```yaml
broadbeam_electron_tracking: true
broadbeam_electron_density_threshold: 0.1
broadbeam_electron_halo_mm: 5.0
```

The default CT selection uses a box dilation of density ≤0.1 g/cm³ by 5 mm, evaluated at the sampled birth position. Once selected, a packet may propagate beyond the selection. Uniform unit-density water has an empty selection and remains locally deposited under this policy. Disabling the switch selects local deposition and removes the independent electron bank from that run. No proton-only tail or carbon-only joint-replay path is required.

The current CT broadbeam YAMLs also retain low-density packet survival probability 0.5 with compensating weight. Its signed weight ledger and task/energy audits remain required. This is variance reduction, not justification for dropping external-air electrons without weights or dose validation.

## Data layout

The electron registry is [docs/electron_packages_manifest.json](docs/electron_packages_manifest.json). There are three packages:

| Package | Runtime contents | Payload bytes |
|---|---|---:|
| `data/electrons/proton/` | Proton `birth_bank.bin` | 328,620,496 |
| `data/electrons/carbon/` | C12 `birth_bank.bin` | 1,033,490,328 |
| `data/electrons/shared/` | `joint.bin`, charged and photon operators/indices | 3,837,722,550 |

Total unique payload is 5,199,833,374 bytes; JSON manifests are excluded. Only the chosen projectile birth package and the shared package load for a run. Loader accounting for the 83-node package reports approximately 3.127 GB proton or 3.597 GB carbon device data, excluding other physics, queues and scorers; these are not total process memory or peak loading footprints.

Birth, charged and photon banks contain 82 Schneider density nodes plus one Water_75eV node. Electron material section25 denotes water; Schneider sections remain0–24. Water's charged/photon domain reaches0.8 MeV, while Schneider remains0.65 MeV. Water primary birth data reach250 MeV for protons and300 MeV/u for C12; Schneider birth limits are250 and255 MeV/u respectively. A source outside actual material/energy support is not validated by this table. Missing runtime support is an error, not a reason to substitute water or local deposition.

Current schemas are MAIBIR02 birth, MAIJNB02 joint, MAIECB02 charged index and MAIPHB02 photon index. The indices contain section, density and relative payload path. Validation checks dimensions, sizes, materials, projectile identity, finite values, sampling support and certified collision-rate bounds. The runtime and current extractors have no file-digest checks. Retired configuration keys are rejected; no digest-field or old-package compatibility entrypoint is provided.

Other packages remain necessary: unified EM and delta moments; projectile nuclear rates/events; elastic/recoil data where selected; Urban tables for minibeam; copper/air tables and optional neutron data. The live dependency inventory is [docs/package_manifest.json](docs/package_manifest.json). Large binaries and patient inputs remain local under ignored `data/` and `inputs/`; a Git checkout alone is not a complete runnable dataset.

## Build and run

Use a CUDA-capable SYCL compiler/plugin. For RTX2080Ti:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_ENABLE_MINIBEAM=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=ON -DCARBON_DOSE_FP64=OFF
cmake --build build -j 6
python3 scripts/check_electron_packages.py
```

Use the architecture of the actual GPU when building for another device. The production executable at `build/carbon_mc` includes the secondary elastic-guard repair (`5cc096d`); its frozen reference is `out/ct_throughput_20261005/guard_fixed_carbon_mc`. The pre-cleanup executable remains at `out/repository_cleanup_20261004/before/carbon_mc`.

Run continuous batches in one process:

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

Use a new output directory. Each batch has its own seed, dose and quality report. Packages and immutable geometry are reused in the same process. Plan weights are normalized across the total requested histories; do not multiply an already normalized plan dose by the batch count again. To run distinct configurations sequentially with a shared session, use `--plan-manifest FILE`; `--plan-preflight` checks the manifest before transport.

## Verification and throughput work

The four CT presets use these acceptance gates:

| CT preset | Dose gate | Gamma search step |
|---|---|---|
| Proton and carbon broadbeam | Local 1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local 3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | Dose gate deferred | Report separately if evaluated |

Freeze the reference, dose normalization, coordinate mapping, dose threshold and evaluation mask before comparison. Report them with the result. The current RT07575 broadbeam comparison uses the full 3D evaluation region above 10% of reference maximum, absolute normalization and no dose fitting or registration. A smaller search step is not permission to change the mask or denominator. Zero-DTA gamma remains a noise-sensitive diagnostic.

Final RTX2080Ti results, including the elastic-guard repair:

| Preset | Histories | Resident primary/s | TOPAS local gamma |
|---|---:|---:|---|
| Proton CT broadbeam | 12,531,500 | 167,692 | 1%/1 mm: **99.9254%**, pass |
| Carbon CT broadbeam | 6,481,909 | 104,144 | 1%/1 mm: **99.8677%**, pass |
| Proton CT minibeam | 20,000,000 | 51,821 | Projected 2D 3%/0.2 mm: **98.3183%**, below gate |
| Carbon CT minibeam | Two seeds, each 3 × 200,000 per arm | **10,048**, control 5,165 | Accuracy deferred |

The proton minibeam figure uses the historical PyMedPhys 0.41.0 radial search (`interp_fraction=10`) on the BODY-projected lung dose, |y|<15 mm, |x|<15 mm, reference ≥3% maximum. It is not a 3D voxel or RT07575 full-plan acceptance. A Cartesian-lattice cross-check gives 97.1013% at the same 0.02 mm spacing; neither evaluator passes >99%. Peak and valley strips give 99.9605% and 99.0052% with PyMedPhys, but do not replace the full-field gate. No new proton minibeam performance candidate was promoted.

Carbon CT minibeam now uses `history_chunk_size` and `material_electron_queue_chunk` of 131,072, with existing species grouping and 16-step secondary continuation enabled. Group32, 16,384 electron lanes, the 576 MiB queue budget, physical steps and per-history RNG stay fixed. Same-binary reverse-order confirmation gives **1.946×** resident throughput, primary kernel 22.320→7.565 s and secondary 6.917→3.498 s per 200k batch. All compared discrete transport/EM/electron fields match; maximum dose relative L1 is 5.01e-8. Queue demand peaks at 3,798,118 / 5,807,497 tasks for this scene. This validates scheduling consistency, not carbon minibeam dose accuracy.

The shared-worker count, compact failure state, C12 specialization and smaller secondary-tail candidates did not deliver a resolved useful gain and were not promoted. Actual worker profiling showed substantial local-memory traffic and about 11 active threads per issued warp instruction; reducing static registers alone did not improve runtime. See [the full report](docs/ct_throughput_20261005.md) and [machine-readable results](docs/ct_throughput_20261005.json).

Final two-batch regression covers all eight presets on the current binary and YAMLs. Electron work counters match the retained controls exactly; maximum dose relative L1 is 1.224e-7. Quality, energy, completion, coverage and zero-overflow checks pass. These small runs are not water gamma or steady-state throughput measurements. Water birth extraction used 64 native histories per projectile and needs a separate dose/statistical campaign. The earlier cleanup control-flow regression and its larger-statistics repair are recorded in the [cleanup correction](docs/repository_cleanup_20261005.md#larger-statistics-correction).

Measure cold start, first batch and resident batches separately. Report primaries/s, kernel times, task counts and peak memory with GPU model, compiler, source, seed and history counts. Compare like-for-like source histories, not electron tasks or accepted collimator primaries. `scripts/benchmark_ct_devices.py` uses native repeated batches for all four CT presets, validates each batch, and supports `--smoke-only`; the reproduction command is in the CT report.

Search [failed.md](failed.md) before each candidate; require documented new evidence before retrying a rejected route. FP32 atomic order differences alone do not reject scheduling changes, but energy, transport/EM, completion and queue-overflow audits must pass. TOPAS references and historical experimental evidence are retained. Water performance and dose campaigns remain deferred until CT validation.
