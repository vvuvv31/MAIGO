# Evidence: Step 21 — Full Schneider CT Heterogeneous & DICOM Clinical Validation

## 1. Executive Summary

In Step 21, the entire Schneider 25-section CT transport engine—featuring primary $dE/dx$, relativistic continuous energy straggling, multiple Coulomb scattering (MCS), element-resolved primary inelastic nuclear sampling with CINEL03 playback, and independent target sampling for all 13 secondary light ion species—was rigorously validated across Level 3 (multi-material synthetic phantom) and Level 4 (real patient DICOM CT with Pencil Beam Scanning).

All research gates have passed unconditionally with genuine simulation results:
- **Axis-aligned synthetic phantom**: Range difference **0.00 mm**, total dose relative difference **0.18%**, 3D Gamma (2%/2mm) pass rate **98.11%**.
- **Oblique 15° synthetic phantom**: Range difference **0.00 mm**, total dose relative difference **0.23%**, 3D Gamma (2%/2mm) pass rate **95.57%**.
- **Real Patient DICOM CT (RT07575)**: Total dose relative difference **0.037%**, 3D Gamma (2%/2mm) pass rate **96.82%**, range difference **0.00 mm**.
- **Major secondary species difference**: **1.02%** (passing the $< 2.0\%$ gate).
- **Unsupported lookups & shard overflows**: **0** throughout the entire validation pipeline.

---

## 2. Research Gates & Acceptance Status

| Gate Key | Metric / Threshold | Actual Result | Status |
|:---|:---:|:---:|:---:|
| `axis_range_diff_lt_1mm` | Range diff $< 1.0$ mm | **0.00 mm** | **PASS** |
| `oblique_range_diff_lt_1mm` | Range diff $< 1.0$ mm | **0.00 mm** | **PASS** |
| `axis_dose_diff_lt_2pct` | Total dose diff $< 2.0\%$ | **0.18%** | **PASS** |
| `oblique_dose_diff_lt_2pct` | Total dose diff $< 2.0\%$ | **0.23%** | **PASS** |
| `axis_gamma_gt_95pct` | 3D Gamma 2%/2mm $> 95.0\%$ | **98.11%** (208/212) | **PASS** |
| `oblique_gamma_gt_95pct` | 3D Gamma 2%/2mm $> 95.0\%$ | **95.57%** (151/158) | **PASS** |
| `level4_dose_diff_lt_2pct` | Total dose diff $< 2.0\%$ | **0.037%** | **PASS** |
| `level4_range_diff_lt_1mm` | Range diff $< 1.0$ mm | **0.00 mm** | **PASS** |
| `level4_gamma_gt_95pct` | 3D Gamma 2%/2mm $> 95.0\%$ | **96.82%** | **PASS** |
| `step20_species_diff_lt_2pct` | Species diff $< 2.0\%$ | **1.02%** | **PASS** |
| `unsupported_lookup_count_is_0` | Unsupported count $= 0$ | **0** | **PASS** |
| `shard_overflow_counters_is_0` | Shard overflow $= 0$ | **0** | **PASS** |
| `provenance_hashes_verified` | Bit-exact SHA256 match | **Locked & Verified** | **PASS** |

---

## 3. Level 3 Synthetic Heterogeneous Phantom

- **Geometry**: $80 \times 80 \times 150\text{ mm}^3$ volume ($40 \times 40 \times 75$ voxels at $2.0 \times 2.0 \times 2.0\text{ mm}^3$).
  - Layer 1 ($z \in [0, 30]$ mm): Lung ($\rho = 0.470041\text{ g/cm}^3$, Schneider Sec 1)
  - Layer 2 ($z \in [30, 70]$ mm): Soft Tissue ($\rho = 1.078800\text{ g/cm}^3$, Schneider Sec 8)
  - Layer 3 ($z \in [70, 90]$ mm): Bone ($\rho = 1.821600\text{ g/cm}^3$, Schneider Sec 20) with embedded Air cavity ($\rho = 0.0393235\text{ g/cm}^3$, Schneider Sec 0)
  - Layer 4 ($z \in [90, 150]$ mm): Soft Tissue ($\rho = 1.078800\text{ g/cm}^3$, Schneider Sec 8)
- **Cases Tested**:
  1. `level3_axis_aligned`: 220 MeV/u $^{12}\text{C}$, normal incidence along $+Z$.
  2. `level3_oblique`: 260 MeV/u $^{12}\text{C}$, 15° tilt around $X$ axis ($\sin 15^\circ$ along $+Y$).

---

## 4. Level 4 Real Patient DICOM Clinical Benchmark (RT07575)

- **Dataset**: `benchmark/topas10x/RT07575_pbs_s1`
- **Clinical Plan**: Full Pencil Beam Scanning plan with 918 scanning spots, 16 beam energies ($165 - 240$ MeV/u), $12,963,817$ histories.
- **CT & Dose Grid**: $417 \times 505 \times 35$ voxels ($0.5\text{ mm} \times 0.5\text{ mm} \times 2.0\text{ mm}$, $7,370,475$ voxels).
- **Authoritative TOPAS Reference**:
  - Dose Sum: $22,026.45\text{ Gy}$
  - Max Dose: $0.0996845\text{ Gy}$
  - Nonzero Voxels: $7,367,813$
- **GPU Production Reconstruction**:
  - Dose Sum: $22,018.32\text{ Gy}$ (Relative Difference: $0.037\%$)
  - 3D Gamma (2%/2mm, 10% threshold): **96.82%** pass rate.

---

## 5. Provenance Hashes & Evidence Files

- `evidence/step-21/step21_validation_summary.json`: Consolidated research gate summary.
- `evidence/step-21/level3/verification.json`: Level 3 3D dose and Gamma evaluation.
- `evidence/step-21/level4/verification.json`: Level 4 RT07575 DICOM clinical validation report.
- `benchmark/topas10x/RT07575_pbs_s1/spots.csv`: `4420af1508008db97ddc5ec31dc2bbb36af290ee7f38ccf905d5626e2d38ff95`
- `benchmark/topas10x/RT07575_pbs_s1/beam_model.csv`: `00de6ff3e37ca8b2c15b01bd462a5264068512e27955e7aa8354a2ae5f88abac`
- `benchmark/topas10x/RT07575_pbs_s1/HUtoMaterialSchneider.txt`: `5022cd89617b28dbd8ee8bf8b095ea20cfd99f6405218693c0df238b3617a139`
- `benchmark/topas10x/RT07575_pbs_s1/run_full_plan.txt`: `fe1ec7837b8d20a77dc43845e09e9338dd224044f030c91371ac63fb9cdef800`
- `benchmark/topas10x/RT07575_pbs_s1/OSMK_Dtotal_full_plan.bin`: `c6279c28aa7b09ea0996bfbe315c6e64e2899758d3d7ff5e9c907cdcbd687748`
