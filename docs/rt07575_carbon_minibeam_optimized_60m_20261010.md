# RT07575 head carbon minibeam optimized weights,60M GPU histories

Completed 2026-10-10 on RTX A6000 (`v@172.17.152.19`), direct execution. Uses the user-selected throughput research candidate `upstream_chunk_262144`: patient cap1mm, CT primary segmentation clock1/16, Cu cap0.05mm, air slit0.2mm, primary patient/Cu fluctuationsON, patient-secondary and Cu-fragment restricted fluctuationsOFF, fragment hard deltaON, explicit electron transportOFF. These are existing research approximations, not release defaults.

## Plan and normalization

Original RT07575 **head** physical-dose optimization,400 active spots,90°/270° crossfiring with270° collimator slit phase1.5mm. Original source optics/emittance/energy/placement, Cu100mm thick/0.7mm slit/3mm pitch, and separate field CT mappings preserved. Optimized weight/details/source CSVs in inputs match `/mnt/sdb/wuwei/ct/rt07575/code/output_physical/` bytewise. No weights were reoptimized or dose-fit.

Incident weights sum13,872,495,687.640352 carbon ions per4Gy fraction. Allocate60,000,000 simulated histories globally using largest remainder across all400spots once; split evenly into24balanced batches of2,500,000, preserving every active spot in each batch. New per-spot/batch seeds give independent histories. Dose scale231.20826146067253=W/N. Sum the24batches and both fields exactly once, no batch averaging or multiplication. 90° patient mapping transpose then x reversal;270° transpose then y reversal. Original patient dose grid0.5×0.5×2mm unchanged. One GPU context/packages across all batches.

Reference is frozen optimized `physical_dose.raw` (per fraction), SHA256 `5525ee46223b73c8a036482c7f1ba43a204419ee81a445e670bc9360fcbc9aea`, from the original100k-history/column DIJ. It is not the16-fraction physical_dose_total file or the later high-stat TOPAS campaign. No new TOPAS, fitted scaling or registration.

## Local gamma and dose

All gamma calculations use **local3%/0.2mm, spherical search0.02mm**.

| Scope | Passed/evaluated | Pass rate |
|---|---:|---:|
| BODY union GTV, reference≥0.4Gy (10% prescription) |315425/393145|80.231212%|
| BODY union GTV, reference≥10%whole-volume peak |19764/22505|87.820484%|
| GTV, all voxels |21972/31184|70.459210%|

Reference peak55.13174Gy means the conventional10%peak threshold is5.513174Gy; this **excludes the entire GTV**, whose reference maximum is5.213205Gy. Therefore report prescription-threshold patient gamma and GTV separately; do not present87.82% as target matching. No global gamma was calculated.

GTV mean GPU4.01220844Gy vs reference3.97994011Gy, +0.810774%. GTV D95 GPU3.48135185 vs reference3.46110582Gy. BODY mean+2.332956%. These are observed physical-dose differences, not proof that the clock/step approximation is validated.

## GPU statistical noise

Use24 nearly equal balanced whole-plan strata. Let each batch's normalized dose be d_b; a full-plan sample is24d_b. Descriptive full60M voxel SE is std(24d_b,ddof=1)/sqrt24. This includes correlations between two fields within each whole-plan batch. Reference DIJ voxel variance is not available in the optimization output.

| Metric | Patient reference≥0.4Gy | GTV |
|---|---:|---:|
| GPU voxel SE RMS /local reference |3.326473%|2.196067%|
| Median voxel SE /local reference |2.679489%|2.127995%|
| 90th-percentile voxel SE |4.954116%|2.684449%|
| 95th-percentile voxel SE |5.750568%|2.865563%|
| GPU voxel SE RMS in Gy |0.064103|0.086826|
| Observed local RMS difference vs reference |8.688404%|3.915499%|
| Alternating two30M halves difference RMS |6.647722%|4.382340%|

Two independent30M halves should differ by approximately2×the60M SE; observed half differences agree closely with this, supporting the batch noise estimate. This is a descriptive consistency check, not an uncertainty estimate for historical DIJ.

GTV **mean dose** has GPU SE0.003024Gy (0.07538% ofmean), approximate95%interval4.012208±0.006257Gy (t23=2.069). Per-voxel noise and ROI-mean noise are different quantities. The+0.032268Gy reference mean difference is large relative to GPU mean noise, but reference uncertainty remains unknown; it does not alone prove a model bias.

Subtracting GPU variance in quadrature from the observed local difference yields RMS3.241670% in GTV and8.026390% in patient≥0.4Gy. This residual combines original DIJ noise, model differences and other comparison error, and must not be labelled pure systematic bias. GPU Monte Carlo noise alone does not explain the observed full difference. Doubling simulated histories reduces GPU SE bysqrt2;4×histories to240M would give approximately1.10%GTV and1.66%patient voxelSE, while leaving reference noise/model bias unchanged. Assuming the same regime, approximately289M histories would be needed for1%GTV voxelSE RMS. These estimates do not predict gamma acceptance.

## Runtime audits and performance

All48field-batch quality/energy/nativeEM/coverage/completion audits pass; zero overflow; all Cu child tasks complete; maximum relative energy residual8.54962e-7. Cold2.5M batch43.112s (57988/s). Warm57.5M throughput73924/s; total821.224s (13.69min). Timing differs from the earlier3×1M screen because batch histories and startup amortization differ; do not interpret this as a new code gain. No locked-clock/general-preset performance claim.

Only the offline numeric-source driver's batch/count constants changed; linked the existing carbon research static library. Production runtime/default configurations unchanged. Build/provenance/source hashes recorded. BinarySHA256 `6ea7a326086bc3cd474cf32d5815232c9a37161c30edf6fe43ee9465ea8aad13`.

## Artifacts

`out/rt07575_carbon_minibeam_optimized_60m_20261010/results/`: report.json, noise_report.json, dose_GPU_4Gy.mhd/raw, per-field dose_GPU_rot90/270_4Gy.mhd/raw, three local-gamma pass masks, dose_comparison.png and analysis_complete.json. Frozen reference/weights are in reference/, global spot allocation in spot_allocation.json, individual batch doses/audits in remote_results/case/gpu/. Remote directory mirrors the experiment name under `/home/v/MAIGO_clean_a6000_20261010/`; run complete. Carbon minibeam independent dose validation remains deferred; this run reports disagreement rather than accepting the optimized physics.

## Lateral1D profiles

Generated `results/profiles/combined_lateral_profiles.png/pdf`, `field_90_lateral_profiles.png/pdf`, `field_270_lateral_profiles.png/pdf`, per-field numeric CSVs and metadata.json using plot_profiles.py. Single patient axial slice z=-782.19mm chosen nearest GTV centroid, y window±30mm about124.7mm. Lateral sample0.5mm, no smoothing/averaging or normalization fitting. Depths20/40/60/100/120mm plus target centroid plane:79.5mm for90°,49mm for270°. Depth is physical patientX distance from the central-ray outermost BODY voxel center (90°x=67mm,270°x=-61.5mm), not WEPL or a separate contour-following entrance per lateral voxel. Field plots show GPU±1sigma bands from24 balanced batches; reference uncertainty unavailable. Combined profiles mostly overlay near peak positions; individual-field distal tails are visibly higher in GPU (90°120mm,270°100/120mm), while the opposite field can obscure these low-dose differences in combined dose. This is an observed discrepancy on these profiles, not a causal diagnosis or global accuracy acceptance.
