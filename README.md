# MAIGO

GPU proton/carbon Monte Carlo transport using TOPAS/Geant4-derived physics data.

Eight presets share the current transport architecture; [cleanup regression](docs/repository_cleanup_20261005.md) is complete. The [CT campaign](docs/ct_throughput_20261005.md) delivers a 1.946× carbon minibeam scheduling speedup; both broadbeams pass the requested dose gate, while lung proton minibeam remains below it. Read [materials and methods](mm.md) or [中文方法说明](mm_zh.md) for the architecture, package coverage, limitations and verification status.

## Supported presets

| Geometry | Broadbeam | Minibeam |
|---|---|---|
| Proton CT | [preset](config/proton_ct_broadbeam.yaml) | [preset](config/proton_ct_minibeam.yaml) |
| Carbon CT | [preset](config/carbon_ct_broadbeam.yaml) | [preset](config/carbon_ct_minibeam.yaml) |
| Proton water | [preset](config/proton_water_broadbeam.yaml) | [preset](config/proton_water_minibeam.yaml) |
| Carbon water | [preset](config/carbon_water_broadbeam.yaml) | [preset](config/carbon_water_minibeam.yaml) |

Minibeam uses global ion Urban scattering and shared independent FP32 electron/photon transport. Proton and carbon retain separate electron births and nuclear data. Broadbeam uses Gaussian Fermi–Eyges ion scattering; `broadbeam_electron_tracking` selects the common electron worker in low-density CT regions and their5 mm neighborhood. Uniform unit-density water has an empty broadbeam selection.

Water uses TOPAS **Water_75eV**. Carbon minibeam dose accuracy is still pending. Water transport smoke tests are not dose acceptance.

## Build and run

See [build instructions](mm.md#build-and-run) for CUDA/SYCL setup. Use `CARBON_DOSE_FP32=ON` and `CARBON_DOSE_FP64=OFF`.

Large physics packages and patient inputs are local files, not included by cloning Git. Their inventories are [preset dependencies](docs/package_manifest.json) and [three electron packages](docs/electron_packages_manifest.json). The local dependency checks are:

```bash
python3 scripts/check_electron_packages.py
python3 scripts/list_preset_dependencies.py --output out/preset_dependencies.json
```

Run multiple batches in one process to reuse the packages:

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

Use a new output directory. Check each batch's quality report, coverage, energy accounting, completed task count and queue overflow before using its dose. The tested executable is installed at `build/carbon_mc` and includes the `5cc096d` secondary elastic-guard repair. Frozen binaries and per-batch evidence are retained under `out/ct_throughput_20261005/`.

## Validation and development

Current RTX2080Ti resident rates and dose results:

| CT preset | Primary/s | Local gamma |
|---|---:|---|
| Proton broadbeam | 167,692 | 1%/1 mm: 99.9254%, pass |
| Carbon broadbeam | 104,144 | 1%/1 mm: 99.8677%, pass |
| Proton minibeam | 51,821 | Projected 2D 3%/0.2 mm: 98.3183%, below gate |
| Carbon minibeam | 10,048 (previously 5,165) | Dose accuracy deferred; runtime audits pass |

The gates are local **1%/1 mm >99%** for broadbeam (search 0.1 mm) and **3%/0.2 mm >99%** for proton minibeam (search 0.02 mm). Broadbeams use RT07575 full 3D references; minibeams use lung CT. Proton minibeam's historical projected-dose result is not a 3D or RT07575 full-plan acceptance. The [report](docs/ct_throughput_20261005.md) records statistics, masks, gamma methods and rejected candidates. Water performance and dose campaigns follow CT validation.

Read [AGENTS.md](AGENTS.md) and [failed.md](failed.md) before performance experiments. Historical TOPAS references and experiment records remain evidence for their original versions. They are not additional supported preset entrypoints.
