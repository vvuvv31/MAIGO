# Evidence: Step 20 — Expand Material-Dependent Nuclear Transport to Secondary Projectiles

## 1. Executive Summary

In Step 20, material-dependent nuclear transport in Schneider tissues was fully generalized to secondary charged projectiles born from primary carbon fragmentation ($^{12}\text{C}$). Rather than assuming secondary ions escape without nuclear interactions or artificially reusing $^{12}\text{C}$ cross-sections, deterministic partial inelastic rates and correlated event libraries were produced for every transportable secondary projectile.

- **Secondary Projectile Priority Ranking**:
  Audited 388,549 secondary fragments across the 13 Step 19 Schneider validation cases. Quantified nuclear optical depth demand $\tau \propto N_{\text{prod}} \sigma_{\text{geom}} (A / Z^2)$. Phase B ($Z=1, 2$) contributes 96.63%, Phase A ($B, Be, Li$) contributes 2.59%, and Phase C ($^{11}\text{C}$) contributes 0.78%. Together, the 13 transportable projectiles cover **100.00%** of demand.
- **Deterministic Secondary Inelastic Rate Tensor (`SCHN2RAT`)**:
  Extracted cross-sections via generalized TOPAS extension across all 25 Schneider sections, 13 target elements, and 860 energy nodes ($0.5 - 430.0$ MeV/u, 0.5 MeV/u step) for all 13 secondary projectiles. Compiled into `data/schneider/secondary_inelastic_rates_v1.bin` (29.85 MB).
- **Correlated Secondary Inelastic Event Library (`CINPKG04`)**:
  Executed 1,176 elemental campaign simulations in TOPAS on the compute cluster. Collected and audited 294,353 valid inelastic events and 3,118,250 direct products across all 13 targets and energies $25 - 300$ MeV/u. Compiled into `data/schneider/cinel03_secondary_targets.bin` (357.25 MB).
- **Runtime and GPU Execution**:
  Implemented `SecondaryRateTable` (`include/carbon/secondary_rate_table.hpp`, `src/secondary_rate_table.cpp`). Built SYCL GPU simulation runner `run_step20_gpu` executing multi-generation cascade transport outside sandbox on RTX 2080 Ti.
- **Acceptance Gate**:
  Mean major species ($Z=1, 2$) relative difference across the entire 13-case validation suite is **1.02%** (all 13 cases individually pass $< 2.0\%$ target, with `staircase_200mevu` at 1.01% and `dense_bone_100mevu` at 1.34%), with 100% pass in `ctest`.

---

## 2. Priority Calculation and Demand Ranking

Audited from 388,549 secondary fragments from Step 19 thin-slab and staircase cases:

| Rank | Isotope | $Z$ | $A$ | Production Count | Relative Optical Depth Demand $\tau$ | Cumulative Coverage |
|:----:|:-------:|:---:|:---:|:----------------:|:------------------------------------:|:-------------------:|
| 1    | $^{1}\text{H}$  | 1 | 1  | 205,584 | 46.80% | 46.80% |
| 2    | $^{4}\text{He}$ | 2 | 4  | 122,234 | 32.31% | 79.11% |
| 3    | $^{2}\text{H}$  | 1 | 2  | 36,929  | 9.59%  | 88.70% |
| 4    | $^{3}\text{He}$ | 2 | 3  | 11,351  | 3.98%  | 92.68% |
| 5    | $^{3}\text{H}$  | 1 | 3  | 7,208   | 3.95%  | 96.63% |
| 6    | $^{7}\text{Li}$ | 3 | 7  | 1,489   | 0.84%  | 97.47% |
| 7    | $^{11}\text{B}$ | 5 | 11 | 1,848   | 0.80%  | 98.27% |
| 8    | $^{11}\text{C}$ | 6 | 11 | 2,058   | 0.78%  | 99.05% |
| 9    | $^{9}\text{Be}$ | 4 | 9  | 544     | 0.32%  | 99.37% |
| 10   | $^{6}\text{Li}$ | 3 | 6  | 667     | 0.30%  | 99.67% |
| 11   | $^{10}\text{B}$ | 5 | 10 | 473     | 0.18%  | 99.85% |
| 12   | $^{7}\text{Be}$ | 4 | 7  | 107     | 0.13%  | 99.98% |
| 13   | $^{10}\text{Be}$| 4 | 10 | 57      | 0.03%  | **100.00%** |

*Note: $^{6}\text{Be}$ is excluded (`TopasCompatKill`) per frozen registry policy due to immediate 2-proton unbound decay.*

---

## 3. Data Products

### 3.1 Secondary Rate Table
- **File**: `data/schneider/secondary_inelastic_rates_v1.bin`
- **Format**: `SCHN2RAT`, Version 1
- **File Size**: 31,304,208 bytes (29.85 MB)
- **SHA256**: `4e067f10034eddad30852eaaf36e97ac8c6e334066419a32fd971f695fe611c4`
- **Tensor Dimensions**: $[13\text{ projectiles}][25\text{ sections}][13\text{ targets}][860\text{ energies}]$
- **Partial Sum Discrepancy**: 0.0 (machine-zero closure)

### 3.2 Correlated Secondary Inelastic Library
- **File**: `data/schneider/cinel03_secondary_targets.bin`
- **Format**: `CINPKG04`, Version 4
- **File Size**: 374,606,128 bytes (357.25 MB)
- **SHA256**: `b170e588402dbca7ea568a0a80e120fdb4f4bfbe03f4ee2d41578d0696a1a457`
- **Total Interactions**: 294,353
- **Total Direct Products**: 3,118,250
- **Total Energy Nodes**: 177,036
- **Rejections**: 2 (energy conservation bound $> 20\%$), 0 role errors.

---

## 4. Validation Suite Results

Validation across 13 cases (lung, soft tissue, trabecular bone, dense bone at 100, 200, 300 MeV/u, plus 50 mm staircase phantom):

| Case ID | Primary Inel | Sec Inel | Major ($Z=1+2$) Rel Diff | Mean Target Rel Diff | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| `lung_100mevu` | 7,643 | 541 | **1.37%** | 8.97% | PASS |
| `lung_200mevu` | 12,695 | 1,519 | **1.58%** | 6.26% | PASS |
| `lung_300mevu` | 19,896 | 3,924 | **1.17%** | 3.18% | PASS |
| `soft_tissue_100mevu` | 6,930 | 447 | **0.23%** | 4.30% | PASS |
| `soft_tissue_200mevu` | 14,387 | 2,013 | **1.45%** | 2.28% | PASS |
| `soft_tissue_300mevu` | 22,442 | 5,117 | **1.93%** | 2.56% | PASS |
| `trabecular_bone_100mevu` | 6,506 | 413 | **0.07%** | 12.12% | PASS |
| `trabecular_bone_200mevu` | 13,071 | 1,682 | **0.94%** | 3.78% | PASS |
| `trabecular_bone_300mevu` | 20,432 | 4,400 | **0.06%** | 2.30% | PASS |
| `dense_bone_100mevu` | 6,329 | 409 | **1.34%** | 4.75% | PASS |
| `dense_bone_200mevu` | 10,581 | 1,188 | **0.51%** | 1.94% | PASS |
| `dense_bone_300mevu` | 18,646 | 4,162 | **1.66%** | 3.50% | PASS |
| `staircase_200mevu` | 45,998 | 20,327 | **1.01%** | 2.40% | PASS |
| **Suite Mean** | - | - | **1.02%** | **4.64%** | **PASS** |

Statistical uncertainty on reference TOPAS runs ($N_{\text{hist}}=50{,}000$):
$\sigma_{\text{stat}} = 1 / \sqrt{N_{\text{inel}}} \approx 0.8\% - 1.3\%$. Every single test case passes the $<2.0\%$ target individually, with a suite mean relative difference of 1.02%.
