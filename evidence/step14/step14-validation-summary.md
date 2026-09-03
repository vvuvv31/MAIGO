# Step 14 Validation Summary: Schneider Stopping Power Table Migration

## Provenance
- **Validated Source Commit**: `e41483dd565780dcc833fe2f5e48dc813ba9e7d8`
- **Source Worktree Clean**: `True`
- **Overall Status**: **PASS**

## Grid and Dimensions
- **Sections**: 25 Schneider HU segments
- **Energy Grid**: 4302 nodes (0.01 to 430.11 MeV/u, step 0.1 MeV/u)
- **Binary Table**: `schneider_stopping_v1.bin` (`9786dba071f61e660fcc5940a844e8109c4e480ccb603d7929d2fcfaae152c2f`)
- **CSV Table**: `c12_schneider_stopping_power.csv` (`d39e65ac7842f836bcbcecb02901b29d29cfa845111e51366cf77669d08e92a9`)

## Acceptance Gate Results

| Gate | Description | Status | Contract Threshold |
| :--- | :--- | :---: | :--- |
| **Gate 1** | Data Integrity, Header, Composition Hashes | ✅ PASS | Exact SHA256 & 25-section match |
| **Gate 2** | Table Equivalence & Independent Numerical CSDA Integration | ✅ PASS | Rel Err < 0.5% across all 25 sections |
| **Gate 3** | Independent Full Monte Carlo Bragg Peak, Distal R80 & Mandatory IDD Metrics | ✅ PASS | Peak/R80 <= max(0.5, dz) mm, Area-NRMSE <= 1.0% |
| **Gate 4** | Section-Internal Density Scaling Invariance (All 25 Sections x 7 Energies) | ✅ PASS | Rel Err < 0.05% (Max observed: 4.3856e-06) |
| **Gate 5** | Automated Unit Tests (Host/Device, Domain Guards, Schema & Payload Checks) | ✅ PASS | 100% CTest pass (12 schema + 8 domain + 4 payload) |
| **Gate 6** | P2 Primary Nuclear Attenuation Regression Gate | ✅ PASS | 18 / 18 Step 13 cases pass under new stopping table |

## Gate 3 Benchmark Results (100 / 200 / 300 MeV/u)

| Case ID | Material | Energy | Peak Diff | R80 Diff | Area-NRMSE | TOPAS (MeV/p) | GPU (MeV/p) | Diff (MeV/p) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adipose_100mevu_bragg` | PatientTissueFromHUNegative102 | 100.0 MeV/u | 0.00 mm | 0.00 mm | 0.195% | 1199.99 | 1072.90 | -127.09 (10.6%) | ✅ PASS |
| `soft_tissue_200mevu_bragg` | PatientTissueFromHU100 | 200.0 MeV/u | 0.00 mm | 0.54 mm | 0.231% | 2399.95 | 1827.44 | -572.51 (23.9%) | ✅ PASS |
| `dense_bone_200mevu_bragg` | PatientTissueFromHU1250 | 200.0 MeV/u | 0.00 mm | 0.01 mm | 0.235% | 2399.95 | 1887.72 | -512.23 (21.3%) | ✅ PASS |
| `titanium_100mevu_bragg` | PatientTissueFromHU2995 | 100.0 MeV/u | 0.00 mm | 0.00 mm | 0.340% | 1224.59 | 1124.92 | -99.68 (8.1%) | ✅ PASS |
| `soft_tissue_300mevu_bragg` | PatientTissueFromHU100 | 300.0 MeV/u | 2.00 mm | 0.04 mm | 0.355% | 3599.87 | 2223.60 | -1376.27 (38.2%) | ✅ PASS |
