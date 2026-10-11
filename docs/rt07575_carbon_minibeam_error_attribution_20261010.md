# RT07575 head carbon minibeam: error attribution

## Conclusion

The original optimized physical DIJ is an unsuitable dense-dose reference in the low-dose tails: its per-spot sparsification removes real dose before optimized weights are applied. Independent TOPAS reproduces most of the apparent GPU–DIJ discrepancy. Against dense independent TOPAS, the remaining GTV voxel differences are largely consistent with the combined Monte Carlo noise. This does not establish new carbon-minibeam physics acceptance.

## Fixed experiment

RT07575 **head**, original optimized 400 active spots, 90°/270° crossfiring, original CT/source/optics/placement/collimators and weights. Weight sum 13,872,495,687.640352 ions/fraction. All doses are per fraction, without fitted scaling or reoptimization. Patient scoring grid is 0.5×0.5×2 mm; transport step cap is a separate setting.

GPU 240M uses 96 continuous 2.5M batches, chunk262144, patient1mm, CT primary clock1/16, Cu0.05mm, slit air0.2mm, primary fluctuations on, patient secondary/Cu fragment restricted fluctuations off, Cu fragment hard delta on, explicit electrons off. All 192 field audits pass with zero overflow. These are user-selected research settings; release defaults are unchanged.

## Three-way local gamma

Local3%/0.2mm, search0.02mm, fixed original-DIJ masks. Patient mask is BODY∪GTV with original dose≥0.4Gy.

| Comparison | Patient pass | GTV pass | GTV local dose RMS |
|---|---:|---:|---:|
| GPU240M vs original DIJ |81.502245%|75.269369%|3.309862%|
| GPU240M vs dense independent TOPAS150.7M |96.060995%|94.574141%|1.879443%|
| Dense independent TOPAS vs original DIJ |81.384731%|73.313238%|3.545936%|

GPU–DIJ and TOPAS–DIJ residual correlations are 0.93808 in the patient mask and 0.83521 in GTV. These correlations identify shared reference structure, not causal percentages. GPU GTV mean is 4.01184549Gy, −0.248212% versus dense independent TOPAS, but +0.801655% versus original DIJ.

The historical TOPAS replay has the same 400 optimized weights, source and original TOPAS physics module list; both score DoseToMedium on Patient. It lacks GenericIon nuclear elastic whereas GPU includes it, so this comparison is a reference diagnostic, not a matched-physics acceptance test. A 10%-of-global-maximum mask excludes the entire GTV (global max55.13Gy; GTV max5.21Gy) and should not be used to characterize target agreement.

## Causal reproduction of the DIJ threshold

`/mnt/sdb/wuwei/ct/rt07575/bio_dij_osmk_sparse_c.mat` stores physical totalDose with dose threshold **2e−6Gy per original100,000-history spot**, before applying optimized weights. Its metadata explicitly states entries below thresholds are stored as zero. Sampled columns confirm a minimum retained value near2e−6. DroppedBySpot counts are voxel counts, not lost energy. Unrelated unreadable OSMK objects do not imply corruption of the readable physical dose matrix.

Existing native local TOPAS files for all400spots, totaling **41,617,488 histories**, were postprocessed twice. Dense dose uses `sumDose * weight/Nspot`. Threshold dose keeps `sumDose >= 2e−6 * Nspot/100000` before weighting. Both versions use exactly the same samples, so their difference isolates threshold loss with no new MC, smoothing or dose fitting. Original DIJ uses different samples, so pixelwise equality is not expected.

| ROI | Dense mean Gy | Threshold mean Gy | Original DIJ mean Gy | Threshold loss |
|---|---:|---:|---:|---:|
| GTV |4.025935|3.980805|3.979940|1.120970%|
| Patient≥0.4Gy |2.593952|2.562882|2.566813|1.197778%|
| BODY |0.340809|0.331682|0.332065|2.678215%|

GTV mean loss is0.0451295Gy (90°0.0214020Gy;270°0.0237275Gy). The original DIJ deficit relative to this dense replay is1.142461%, closely reproduced by the1.120970% threshold loss.

At the central GTV axial slice z=−782.19mm, profiles span patienty±30mm without smoothing. Depth is measured from the central BODY entry along each field, not a uniform water depth.

| Field/depth | Dense profile mean Gy | Threshold mean Gy | Original DIJ mean Gy | Dose removed |
|---|---:|---:|---:|---:|
|90°/120mm|0.114245|0.091106|0.090863|20.2544%|
|270°/100mm|0.035702|0.026091|0.025831|26.9216%|
|270°/120mm|0.021939|0.013195|0.012348|39.8558%|

This directly explains the direction and substantial size of the apparent tail excess: many small per-spot fragment/tail contributions disappear from the sparse DIJ, although their weighted full-plan sum matters. Independent dense TOPAS and GPU agree much more closely there (profile-integrated GPU/TOPAS +1.49% at90°120mm, +1.69% at270°100mm, +2.86% at270°120mm).

## Residual statistical budget

Exact GTV voxelwise GPU SE from96 batches and native independent TOPAS variance give:

| Quantity | Value |
|---|---:|
|GPU voxel SE RMS relative TOPAS|1.091268%|
|TOPAS voxel SE RMS|1.383058%|
|Combined independent noise RMS|1.761736%|
|Observed GPU–TOPAS local RMS|1.879443%|
|Standardized residual RMS|1.064344|
|Absolute residual >2σ / >3σ|6.03194% /0.47781%|

The standardized RMS is near the noise-only expectation1. Quadrature excess is0.65467%, descriptive rather than established model bias. Spatial residuals are correlated; do not derive ROI-mean significance by treating voxels as independent. Original DIJ voxel variance is unavailable, and threshold introduces both bias and additional sample dependence.

Approximately40M GPU histories are useful for causal screening. Scaling the current GPU SE suggests GTV voxel SE about2.7% at40M; this is not a claim that40M establishes stringent gamma acceptance. No further histories were run to force a pass.

## Next action

Prioritize a dense/untruncated physical reference, or validate a much lower per-spot sparsification threshold on full-plan dose. Do not tune the GPU model to reproduce the thresholded tails. Keep original optimized weights and outputs frozen; regenerating a dense reference does not automatically authorize reoptimization. If remaining systematic model error needs resolution, use matched physics controls for electron transport, CT clock/step approximation and secondary fluctuations, with independent repeated samples. No evidence here justifies changing those settings to fit this DIJ.

## Evidence

- `out/rt07575_carbon_minibeam_optimized_240m_20261010/results/attribution/report.json`: three-way gamma, fields, layers and profiles.
- `.../reference_provenance.json`: DIJ threshold and replay provenance.
- `.../noise_budget.json`: combined voxel noise.
- `.../threshold_causal_report.json`: paired400spot threshold reproduction.
- `.../threshold_profile_checks.json`, `.../threshold_paired_profiles.png`: depth-localized causal effect.
- `.../field_90_three_way.png`, `.../field_270_three_way.png`: GPU/DIJ/dense TOPAS curves.
- `out/rt07575_carbon_minibeam_optimized_240m_20261010/check_dij_threshold.py`, `plot_threshold_check.py`: reproducible analysis scripts.

No runtime source/defaults, original DIJ, spot weights or output_physical files were changed by this analysis.
