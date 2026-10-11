# RT07575 proton/carbon minibeam statistical noise, 2026-10-11

Latest proton patient0.5mm236086432histories (24balanced batches) and carbon queue240Mhistories (96balanced batches), same normalized full-plan arrays used for gamma. Proton uses prescription10%=0.4Gy; carbon prescription3%=0.12Gy. Local3%/0.1mm and0.2mm, search0.02mm; patient scoring gridXYZ0.5/0.5/2mm. Carbon queue and inline are implementation-equivalent at the assessed precision; inline96batch samples are used to estimate its variance. No new Monte Carlo or default change.

## Voxelwise dose noise

Relative SE_RMS = sqrt(mean((SE_voxel/TOPAS_dose_voxel)^2)); one standard error of final plan dose, not uncertainty of an ROI mean. GPU SE estimated as SD of balanced whole-plan batch estimates divided by sqrt(batch_count); count differences between proton batches are <1e-7relative. Carbon TOPAS native per-history standard error is available and independent GPU/TOPAS variances add. Proton historical weighted DIJ has no usable voxel variance.

| Species / region | GPU SE RMS | TOPAS SE RMS | Combined noise RMS | Observed difference RMS |
|---|---:|---:|---:|---:|
| proton BODY_union_GTV | 1.7619% | unavailable | unavailable | 2.7113% |
| proton GTV | 1.0129% | unavailable | unavailable | 1.4654% |
| carbon BODY_union_GTV | 2.3808% | 2.9990% | 3.8291% | 3.8815% |
| carbon GTV | 1.0913% | 1.3831% | 1.7617% | 1.8794% |

## GPU-only sensitivity of gamma

64multinomial bootstrap samples of whole-plan batches, retaining the complete spatial covariance within each batch. Perturb original normalized GPU plan by bootstrap batch-mean minus original batch-mean; TOPAS held fixed. Report SD of resulting pass percentages, in percentage points. This is conditional sampling sensitivity, not an interval centered on observed gamma, not combined TOPAS/GPU gamma uncertainty, and not a noiseless dose estimate. The added perturbation lowers the bootstrap mean gamma; do not interpret that mean as a corrected gamma or attach these SD values to baseline as a validated confidence interval.64replicates give an approximate SD, with about9% relative sampling uncertainty even for a Gaussian statistic.

| Species / region | Actual gamma0.1mm | Perturbation SD0.1mm /pp | Actual gamma0.2mm | Perturbation SD0.2mm /pp |
|---|---:|---:|---:|---:|
| proton BODY_union_GTV | 92.3911% | 0.8836 | 97.4401% | 0.3892 |
| proton GTV | 96.6650% | 1.0750 | 98.3806% | 0.5942 |
| carbon BODY_union_GTV | 87.2792% | 0.3629 | 93.4423% | 0.2089 |
| carbon GTV | 91.3257% | 0.8085 | 94.5741% | 0.5453 |

## Interpretation

Changing DTA does not change dose statistical error;0.1mm provides less spatial tolerance for noisy dose than0.2mm. Carbon combined SE RMS is3.8291% in the patient mask and1.7617% in GTV, against local3% dose tolerance. Carbon standardized residual RMS1.0495/1.0643 and2sigma exceedance5.50%/6.03% are broadly compatible with appreciable MC noise. They do not prove all failures are stochastic or eliminate model differences. Combined relative SE P95 is7.19% in the carbon patient mask; low-dose contributions strongly affect this mask.

Under an illustrative equal-mean Gaussian zero-DTA pointwise dose test, carbon GTV noise alone predicts91.18% within3%. Actual GTV gamma0.1mm is91.33%, but coincidence cannot prove causation: gamma searches neighbors and interpolates spatially, correlations/physical bias matter, and no TOPAS spatial covariance model is available. Do not interpret this as a predicted gamma pass percentage.

About61% of estimated carbon MC variance is TOPAS variance (not61% of gamma failures). Increasing only GPU histories4x projects combined noise3.83→3.23% patient and1.76→1.49% GTV. Increasing both4x projects1.91%/.88%. These are1/sqrt(N) variance projections, not gamma predictions. Proton combined uncertainty remains unavailable; its observed1.465% GTV difference versus1.013% GPU SE leaves room for TOPAS noise and/or bias.

Different sources, masks, thresholds and reference generations forbid a direct species accuracy ranking. Historical carbon TOPAS lacks GenericIon nuclear elastic; proton reference noise is unknown. Whole-volume gamma covariance cannot be recovered from individual voxel error bars. No eight-preset acceptance or new default change.

Artifacts: `out/rt07575_minibeam_noise_20261011/report.json` includes64replicate rates, variance scaling and limitations; analyze.py and analysis.log preserve reproduction. Existing reference hashes verified.

## Approximate proton TOPAS noise from plan-weight allocation

User points out similar histories should give comparable MC noise. This is a reasonable order-of-magnitude approximation when scoring estimators and per-history variances are comparable. A direct native TOPAS measurement is unavailable, but an explicit approximate estimate can be given.

Proton TOPAS has487active columns, each1M histories=487M; GPU uses236.086432M allocated proportional to optimized weights. Weights haveCV1.1318. Under equal per-history spot-variance assumption, the source-weight effective history count is N_eff=(sum W)^2/sum(W^2/n_i)=213.515M, making estimated TOPAS SE/GPU SE=sqrt(236.086/213.515)=1.0515. This global source-weight proxy is not an actual voxel/ROI-specific effective history count: local dose variance differs strongly with spot energy, range, location and transport estimator. It supplies a plausible scale, not a precise measurement or bound.

| Region | Measured GPU SE RMS | Assume same TOPAS noise: combined | Weight-only TOPAS proxy | Weight-only combined proxy |
|---|---:|---:|---:|---:|
| Proton BODY∪GTV,threshold0.4Gy | 1.7619% | 2.4917% | 1.8527% | 2.5567% |
| Proton GTV | 1.0129% | 1.4325% | 1.0651% | 1.4699% |

Thus approximate combined proton dose noise is about2.5–2.6% in BODY∪GTV and1.4–1.5% in GTV under these alternative assumptions, not a confidence interval. Observed GTV differenceRMS1.4654% is close to this estimated scale; this supports a substantial statistical contribution without proving all gamma failures are noise. Individual-voxel SE estimates still do not establish TOPAS spatial covariance or total gamma-rate uncertainty for0.1/0.2mm. The approximation is recorded separately from measured errors in report.json.

## Zero-DTA local3% same-voxel dose test

At user request, DTA0mm is defined as gamma0=abs(GPU-TOPAS)/(0.03*TOPAS), pass gamma0<=1 at the same scoring voxel. No division by zero DTA, spatial search/interpolation, registration, fitted normalization or fresh Monte Carlo. Keep prescription thresholds proton0.4Gy/carbon0.12Gy and BODY∪GTV/GTV masks unchanged.

| GPU arm | BODY∪GTV local3%/0mm | GTV local3%/0mm |
|---|---:|---:|
| proton_patient0p2mm | 78.532339% | 95.356593% |
| proton_patient0p5mm | 80.171356% | 95.382247% |
| carbon_inline | 68.871947% | 88.936634% |
| carbon_Cu_child_queue | 68.871947% | 88.936634% |

Carbon Cu queue and inline have identical zero-DTA pass/fail masks. This is a same-voxel dose tolerance test on the current0.5/0.5/2mm scoring grid;0mm does not imply a finer grid. Artifacts: `out/rt07575_minibeam_noise_20261011/local3pct_0mm_report.json`, zero_dta.py and pass masks.

## Carbon threshold corrected to 0.4Gy (10% of prescription)

User sets carbon threshold to0.4Gy, matching the proton threshold, superseding the earlier0.12Gy choice for subsequent comparisons. Preserve historical0.12Gy results above for provenance. Same240M GPU/150711884TOPAS histories, normalization and frozen anatomical masks; new mask additionally requires TOPAS>=0.4Gy. BODY∪GTV399451voxels; GTV31184voxels unchanged.

| Carbon region | Local3%/0mm | Local3%/0.1mm | Local3%/0.2mm |
|---|---:|---:|---:|
| BODY∪GTV | 78.009067% | 92.389555% | 96.013779% |
| GTV | 88.936634% | 91.325680% | 94.574141% |

GTV gamma is unchanged because all GTV voxels are above0.4Gy. The earlier whole-patient carbon noise estimate3.83% pertains to0.12Gy and must not be labeled as the0.4Gy mask noise. Artifacts: `carbon_prescription10pct_gamma_report.json`, carbon_prescription10pct.py and masks in the same output directory. This is threshold postprocessing only.

## Histories needed for 1% noise under the unified0.4Gy BODY mask

Recomputed carbon GPU batch variance and native TOPAS variance after threshold update: GPU1.643966%, TOPAS2.068622%, independent combined2.642313% RMS over399451voxels. Earlier carbon3.8291% belongs to the0.12Gy mask and is superseded for current BODY comparisons. Proton GPU1.761915% measured, TOPAS1.852705% approximate, combined2.556728% approximate.

Target here means1sigma voxelwise relative-SE RMS<=1% in BODY atTOPAS>=0.4Gy, not every voxel,95%quantile, ROI mean or confidence interval. Use SE proportional1/sqrt(N), preserve source, weights and spot allocation. Counts are total incident histories, not added counts.

| Species | GPU histories for GPU-only1% | TOPAS histories for TOPAS-only1% | GPU histories for combined1%, scale both equally | TOPAS histories for combined1%, scale both equally |
|---|---:|---:|---:|---:|
| Proton | 732.89M | 1671.63M | 1543.26M | 3183.45M |
| Carbon | 648.63M | 644.93M | 1675.64M | 1052.24M |

Proton TOPAS counts refer to487active weighted columns, not unusedzero-weight columns. The proton estimates retain the explicit equal-per-history spot-variance/source-weight proxy assumption. Equal scaling factors6.5369proton and6.9818carbon meet combined1% under these assumptions. Practical rounded goals GPU1.6B/TOPAS3.3Bproton andGPU1.8B/TOPAS1.2Bcarbon provide modest margin. Neither these counts nor reducedMCnoise guarantee gamma>99% or correct existing model bias. Increasing GPU alone cannot reduce combinednoise to1% because fixed TOPAS alone exceeds1%. Per-voxel P95GPUSE is3.17%proton and2.76%carbon; stronger per-voxel coverage targets need more histories.

Artifacts: `out/rt07575_minibeam_noise_20261011/history_target_report.json`, history_target.py and history_target.log.
