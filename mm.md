# Materials and methods

Revision: 2026-10-05. [中文](mm_zh.md). This describes the eight-preset cleanup release. [Cleanup regression](docs/repository_cleanup_20261005.md) is recorded separately from dose and performance acceptance. The subsequent CT throughput campaign is not complete. Historical measurements apply to their recorded executable, source, geometry and statistics.

## Eight presets

The supported configuration matrix contains exactly eight entries. Presets select projectile and geometry; electron propagation is shared.

| Preset in `config/` | Ion scattering | Electron birth selection | Current verification |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Historical dose acceptance; cleanup regression passed |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | Low-density CT plus 5 mm halo | Historical dose acceptance; cleanup regression passed |
| `proton_ct_minibeam.yaml` | Urban | All primary ionization in CT | Validated proton logic; cleanup regression passed |
| `carbon_ct_minibeam.yaml` | Urban | All primary ionization in CT | Transport audits passed; dose acceptance pending |
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

Use the architecture of the actual GPU when building for another device. The cleanup executable was tested at `out/repository_cleanup_20261004/build/carbon_mc` and installed as `build/carbon_mc`. The pre-cleanup executable remains at `out/repository_cleanup_20261004/before/carbon_mc`.

Run continuous batches in one process:

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

Use a new output directory. Each batch has its own seed, dose and quality report. Packages and immutable geometry are reused in the same process. Plan weights are normalized across the total requested histories; do not multiply an already normalized plan dose by the batch count again. To run distinct configurations sequentially with a shared session, use `--plan-manifest FILE`; `--plan-preflight` checks the manifest before transport.

## Verification and throughput work

Acceptance for the forthcoming four-preset CT campaign is:

| CT preset | Dose gate | Gamma search step |
|---|---|---|
| Proton and carbon broadbeam | Local1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | Dose gate deferred | Report separately if evaluated |

Freeze the reference, dose normalization, coordinate mapping, dose threshold and evaluation mask before comparison. Report them with the result. The historical RT07575 broadbeam comparison used the full3D evaluation region above10% of reference maximum, absolute normalization and no dose fitting or registration. A smaller search step is not permission to change the mask or denominator. Zero-DTA gamma remains a noise-sensitive diagnostic.

Historical shared-electron RT07575 local1%/1 mm was99.9120% for proton and99.8677% for carbon; see [the acceptance report](docs/broadbeam_shared_electrons_20261004.md). These figures do not certify every later scheduling or variance-reduction change.

The latest cleanup tests exercised all eight presets in two batches. All eight matched controls had identical transport counters, Unified EM and electron audits, with relative dose L1 at most1.45e-7. Every case passed quality, energy, completion, coverage and zero-overflow checks. These small runs are not gamma or steady-state throughput measurements. Water birth extraction used64 native histories per projectile and needs a separate dose/statistical campaign.

Measure cold start, first batch and resident batches separately. Report primaries/s, kernel times, task counts and peak memory with GPU model, compiler, source, seed and history counts. Compare like-for-like source histories, not electron tasks or accepted collimator primaries. Search [failed.md](failed.md) before each candidate; require the documented new evidence before retrying a rejected route. FP32 atomic order differences alone do not reject scheduling changes, but energy, transport/EM, completion and queue-overflow audits must pass.

TOPAS references and historical experimental evidence are retained. The next throughput campaign uses the committed cleanup as its baseline.
