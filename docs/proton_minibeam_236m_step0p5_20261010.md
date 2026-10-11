# RT07575 head proton minibeam: 0.5 mm patient cap, 236M histories

User-authorized isolated step-cap experiment on 2026-10-10. Run on RTX A6000 at `v@172.17.152.19`, directly without Slurm. Only the patient CT maximum step changes from the completed matched 0.2 mm campaign to 0.5 mm. Release presets and production sources are unchanged.

## Results

| Metric | 0.2 mm control | 0.5 mm candidate |
|---|---:|---:|
| Incident histories | 236,086,432 | 236,086,432 |
| TOPAS local3%/0.2 mm, search0.02 mm | 98.993851% | 98.941577% |
| Passed / evaluated voxels | 183,692 / 185,559 | 183,595 / 185,559 |
| Strict >99% | No | No |
| Warm incident histories/s | 275,318 | 304,240 |
| Campaign wall time | 857.668 s | 776.156 s |
| GTV mean dose | 4.922194 Gy | 4.919262 Gy |
| GTV mean bias vs TOPAS | +0.073385% | +0.013774% |
| GTV D95 | 3.987790 Gy | 4.007426 Gy |
| Evaluation-mask dose sum bias | +0.152173% | +0.048684% |
| Local RMS difference vs TOPAS | 1.830199% | 1.697536% |
| Descriptive GPU voxel SE RMS | 1.100728% | 1.101600% |
| Patient primary seconds/M | 1.053411 | 0.742816 |
| Upstream seconds/M | 2.154087 | 2.151561 |

Gamma decreases by 0.052274 percentage points (97 fewer passed voxels). Observed warm throughput increases by 10.504842%; patient-primary cost falls by approximately 29.5%, while upstream cost is nearly unchanged. These are sequential campaigns with unlocked thermal clocks, not an interleaved precise speed measurement. Candidate wall time is 12.94 minutes. The candidate remains below the strict >99% threshold and is not promoted to defaults. The relative gamma decrease is within the 1 percentage point screening allowance, but this allowance does not establish absolute TOPAS acceptance.

GTV mean changes by -0.059567% relative to the matched 0.2 mm GPU result. Local RMS improves despite the slightly lower gamma; these metrics measure different aspects of the dose distribution. The remaining differences cannot be attributed entirely to Monte Carlo noise or entirely to step bias. Historical TOPAS voxel variance is unavailable. No matched high-stat 0.1 mm arm or fresh TOPAS simulation was performed.

## Preserved case and normalization

Original RT07575 **head** CT full plan: 487 active spots among 870 candidates; original source optics/emittance, placement, collimator, weights and normalization. Cu cap 0.05 mm; air-slit cap 0.2 mm; native segmentation clock; all primary/secondary and copper primary/fragment fluctuations enabled. Explicit electron transport remains disabled in this private research experiment, matching the control. Urban, continuous26 packages/rates/moments/calibration and nuclear energy sinks remain unchanged. History chunk 262144, original secondary grouping/chunking disabled.

Global largest-remainder allocation is performed once across 236,086,432 histories and split into 24 balanced whole-plan batches (16 × 9,836,935 and 8 × 9,836,934). All 487 spots occur in each batch. Every batch source CSV is byte-identical to the control, including per-spot seeds. The same binary is used: SHA256 `eaa47107014517fd7f8766f67a7ae4fc6ba0c25e0050ce0d00e5583840826bc9`. YAML changes only `maximum_step_mm: 0.2` to `0.5`.

One GPU context/packages serves all batches. Original incident weight sum 236,086,432,317.40372, dose scale 1000.0000013444387. Sum the 24 normalized dose arrays once; do not multiply by batch count or average. Patient mapping is the original packed_xneg transpose/reversal; patient grid remains 0.5 × 0.5 × 2 mm. The step cap is not the voxel resolution.

## Reference and audits

Historical AllChargedUrbanPhysics TOPAS weighted DIJ, 1M histories per spot; original 870M simulated histories, active weighted columns correspond to 487M histories. Reference SHA256 `fcd60f21187c8305ee5a1bf36430b7133143081d99382f473638eb192fbc2645`. Gamma mask is BODY union GTV above 10% of whole-volume reference peak, 185,559 fixed voxels. No fitted scaling, shift or registration.

All 24 quality, energy, native EM, coverage and completion audits pass; zero queue overflow; maximum relative energy residual 1.0041265e-7. Cu primary child tasks all complete; no electron queue. Air ionization-clock counters, copper cascade counters and copper child task/completion/overflow counts exactly match control across all 24 batches. Downstream random trajectories change with the patient step cap. GPU ends at 1800 MHz, 84°C; clocks were not locked.

## Artifacts

`out/proton_minibeam_236m_step0p5_20261010/results/`: `report.json`, `matched_comparison.json`, `upstream_counter_comparison.json`, `dose_4Gy.mhd/raw`, `TOPAS_local3pct_0p2mm_passed.npy`, `dose_comparison.png`, `analysis_complete.json`. Individual batch doses/audits and runtime log are in `remote_results/case/gpu/` and `remote_results/case/gpu.log`. Remote directory is `/home/v/MAIGO_clean_a6000_20261010/proton_minibeam_236m_step0p5_20261010`.

Recompute with `python3 out/proton_minibeam_236m_step0p5_20261010/analyze.py` then `python3 out/proton_minibeam_236m_step0p5_20261010/compare.py`.
