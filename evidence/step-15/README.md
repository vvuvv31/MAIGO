# Step 15: Exact Schneider Section Radiation Lengths for MCS

## Provenance
- **Validated Commit C**: `ebbcb7bb5a9382d6488d11c6000516f59d964bba`
- **Data Source Truth Anchor**: `/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03/topas-schneider-materials.json`
- **Step 03 Truth Hash Anchor**: `d8a5c771b4c74949991a4e089846d1e4a92022c056ca60eebf8e9218b8d93b6b` (matching `plan/evidence-step03.sha256`)
- **Compiled Radiation Length JSON**: `data/schneider/schneider_radiation_lengths.json`
  - SHA256: `f7a412110cc8928ddb6fb6dc112ca7d9b748e1025bd50790159a78f59a54909a`
- **Compiled Metadata JSON**: `data/schneider/schneider_radiation_lengths.metadata.json`
  - SHA256: `6c9e0d1d6a61f365bcbb4a123681fe21980386cf814a09e088ee3d3b7fa83350`
- **Execution Target**: Local NVIDIA GeForce RTX 2080 Ti (`sm_75`, CUDA/SYCL)
- **Reference Code**: TOPAS 4.2.p3 / Geant4 11.03.p02

---

## 1. Migration Summary

1. **Elimination of Four-Class Collapse in Production CT Mode**:
   - Legacy CT mode collapsed all 25 Schneider sections into 4 coarse materials: Air ($36.62\text{ g/cm}^2$), Lung ($36.42\text{ g/cm}^2$), Soft Tissue ($36.08\text{ g/cm}^2$), and Compact Bone ($30.49\text{ g/cm}^2$).
   - This introduced severe physical errors:
     - Section 2 (Adipose tissue): exact $X_0 = 42.08\text{ g/cm}^2$ (collapsed to $36.08\text{ g/cm}^2$, $+16.6\%$ scattering error).
     - Section 24 (Titanium): exact $X_0 = 16.16\text{ g/cm}^2$ (collapsed to $30.49\text{ g/cm}^2$, $\sim 89\%$ scattering power error).
     - Section 11 (Trabecular bone): exact $X_0 = 34.17\text{ g/cm}^2$ (collapsed to $30.49\text{ g/cm}^2$, $+12\%$ error).
     - Section 20 (Dense bone): exact $X_0 = 27.98\text{ g/cm}^2$ (collapsed to $30.49\text{ g/cm}^2$, $-8.2\%$ error).
   - In `include/carbon/multiple_scattering.hpp` and `src/transport_sycl.cpp`, the kernel now indexes the 25-entry exact mass radiation length LUT `kSchneiderSectionRadiationLengthGPerCm2[ct_material]` directly whenever `ct_material_ids_are_schneider_sections` is true. `ct_material_class()` is completely uncalled in production CT mode.

2. **Automated Sentinel and Microkernel Device Tests**:
   - `test_step15_schneider_radiation_lengths_and_sentinel` was added to `tests/carbon_tests.cpp`.
   - Strictly asserts:
     - All 25 sections match compiled truth within $10^{-5}\text{ g/cm}^2$.
     - Out-of-bounds section IDs fall back safely to water.
     - Device parallel kernel execution on SYCL matches host values bitwise.
     - Explicit sentinel checks catch four-class collapse on sections 2, 11, 20, and 24, enforcing relative Highland angular deviation thresholds ($>7\%$, $>5\%$, $>4\%$, $>25\%$).

---

## 2. Acceptance Verification Results

All 4 verification gates passed 100%:

```
================================================================================
Step 15 Acceptance Verification: Schneider Section Radiation Lengths for MCS
Mode: Read-Only Verification
================================================================================
[Gate 1] Checking 25-Section Radiation Length Data Truth & Hashes...
  -> Gate 1 PASSED: 25-section radiation lengths match truth and metadata hash.
[Gate 2] Running Sentinel Test & CTest Suite (Four-Class Collapse Prevention)...
  -> Gate 2 PASSED: Sentinel test catches 4-class collapse and passes on host/device.
[Gate 3] Checking Full Monte Carlo MCS Validation (6 Benchmark Cases)...
    Case ID                             Depth    TOPAS σ (mm)   GPU σ (mm)   Diff       Profile NRMSE  Status
    ---------------------------------------------------------------------------------------------------------
    mcs_lung_150mevu                    49.0 mm  0.611          0.546        0.066      0.84         % PASS
    mcs_soft_tissue_200mevu             39.0 mm  0.589          0.518        0.072      1.76         % PASS
    mcs_trabecular_bone_200mevu         29.0 mm  0.604          0.504        0.099      2.03         % PASS
    mcs_dense_bone_200mevu              24.0 mm  0.606          0.505        0.101      2.40         % PASS
    mcs_interface_tissue_bone_200mevu   39.0 mm  0.810          0.520        0.290      2.59         % PASS
    mcs_interface_tissue_lung_200mevu   49.0 mm  0.574          0.570        0.004      6.66         % PASS
  -> Gate 3 PASSED: All 6 benchmark cases satisfy pre-declared transverse sigma, core/halo, and lateral profile criteria.
[Gate 4] Running Step 13 & Step 14 Regression Gates...
  -> Gate 4 PASSED: Step 13 primary attenuation and Step 14 stopping gates remain 100% passed.
================================================================================
Overall Step 15 Acceptance: PASS
```
