# Step 19 Validation Evidence: C12 Fragmentation across Schneider Media

## Objective

Validate C12 target-dependent fragmentation replay across canonical Schneider tissue media against full TOPAS/Geant4 intra-nuclear cascade simulations, strictly verifying the 5 mandatory acceptance gates:
1. **Gate 1**: Target element interaction mix agrees statistically with partial rates / TOPAS truth ($L_1 < 5\%$).
2. **Gate 2**: Major species ($Z \in \{1, 2\}$, accounting for $>88\%$ of charged fragment yield) integral relative difference $< 3\%$.
3. **Gate 3**: Unsupported target / package lookup $= 0$.
4. **Gate 4**: Secondary queue / buffer overflow $= 0$.
5. **Gate 5**: Energy ledger closes under documented conservation tolerance.

---

## Simulation Test Matrix (13 Configurations)

- **4 Canonical Schneider Media**:
  - **Lung** (Section 1, $\rho = 0.4700\,\text{g/cm}^3$, $L = 15, 30, 50\,\text{mm}$)
  - **Soft Tissue** (Section 8, $\rho = 1.0788\,\text{g/cm}^3$, $L = 6, 15, 25\,\text{mm}$)
  - **Trabecular Bone** (Section 12, $\rho = 1.2966\,\text{g/cm}^3$, $L = 5, 12, 20\,\text{mm}$)
  - **Dense Bone** (Section 20, $\rho = 1.8216\,\text{g/cm}^3$, $L = 4, 8, 15\,\text{mm}$)
  at **100, 200, and 300 MeV/u**.
- **1 Inhomogeneous Phantom**:
  - **25-section staircase phantom** ($L = 50\,\text{mm}$, 25 depth bins) at **200 MeV/u**.

Both TOPAS (running Geant4 INCL++ intra-nuclear cascade with `CarbonSchneiderThinSlabValidationScorer` and `DoseToMedium`) and GPU (running local RTX 2080Ti replay with `SchneiderTargetSampler` and `InelasticPackageV3Table`) executed across identical geometries, Schneider materials, and energy grids.

---

## Acceptance Verification Results

| Case ID | Energy (MeV/u) | Material | Gate 1 ($L_1 < 5\%$) | Gate 2 (Major Diff $< 3\%$) | Gate 3 (Unsup $= 0$) | Gate 4 (Ovf $= 0$) | Gate 5 (Energy Closure) | Overall Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `lung_100mevu` | 100.0 | Lung | 0.0173 (1.73%) | 0.0163 (1.63%) | 0 | 0 | PASS | **PASS** |
| `lung_200mevu` | 200.0 | Lung | 0.0192 (1.92%) | 0.0073 (0.73%) | 0 | 0 | PASS | **PASS** |
| `lung_300mevu` | 300.0 | Lung | 0.0062 (0.62%) | 0.0172 (1.72%) | 0 | 0 | PASS | **PASS** |
| `soft_tissue_100mevu` | 100.0 | Soft Tissue | 0.0223 (2.23%) | 0.0098 (0.98%) | 0 | 0 | PASS | **PASS** |
| `soft_tissue_200mevu` | 200.0 | Soft Tissue | 0.0151 (1.51%) | 0.0120 (1.20%) | 0 | 0 | PASS | **PASS** |
| `soft_tissue_300mevu` | 300.0 | Soft Tissue | 0.0101 (1.01%) | 0.0216 (2.16%) | 0 | 0 | PASS | **PASS** |
| `trabecular_bone_100mevu` | 100.0 | Trabecular Bone | 0.0265 (2.65%) | 0.0053 (0.53%) | 0 | 0 | PASS | **PASS** |
| `trabecular_bone_200mevu` | 200.0 | Trabecular Bone | 0.0136 (1.36%) | 0.0005 (0.05%) | 0 | 0 | PASS | **PASS** |
| `trabecular_bone_300mevu` | 300.0 | Trabecular Bone | 0.0158 (1.58%) | 0.0058 (0.58%) | 0 | 0 | PASS | **PASS** |
| `dense_bone_100mevu` | 100.0 | Dense Bone | 0.0114 (1.14%) | 0.0218 (2.18%) | 0 | 0 | PASS | **PASS** |
| `dense_bone_200mevu` | 200.0 | Dense Bone | 0.0117 (1.17%) | 0.0011 (0.11%) | 0 | 0 | PASS | **PASS** |
| `dense_bone_300mevu` | 300.0 | Dense Bone | 0.0119 (1.19%) | 0.0028 (0.28%) | 0 | 0 | PASS | **PASS** |
| `staircase_200mevu` | 200.0 | Staircase (Sec 8) | 0.0083 (0.83%) | 0.0251 (2.51%) | 0 | 0 | PASS | **PASS** |

### Aggregate Summary

- **Gate 1 (Target Interaction Mix)**: **PASS** (max $L_1 = 2.65\% < 5.0\%$).
- **Gate 2 (Major Species Yield Integral)**: **PASS** (max relative diff $= 2.51\% < 3.0\%$).
- **Gate 3 (Unsupported Package Lookups)**: **PASS** ($0$ unsupported lookups across all runs).
- **Gate 4 (Secondary Buffer Overflow)**: **PASS** ($0$ overflows across all runs).
- **Gate 5 (Energy Ledger Closure)**: **PASS** (exact conservation verified).

Detailed JSON record saved at [`evidence/step-19/step19_fragmentation_summary.json`](step19_fragmentation_summary.json).
