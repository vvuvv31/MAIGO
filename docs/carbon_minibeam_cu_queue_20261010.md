# RT07575 carbon minibeam: primary-born copper children in a separate queue

RTX A6000, original head full plan,400spots,90/270degree crossfiring. Enable the existing proton primary-born copper charged-child queue for C12 in an isolated research source snapshot. Reuse the byte-identical charged-child transport body and original history/spot/product RNG identities. Carbon primary threads previously performed this work inline; the candidate records children for a persistent worker and subtracts their outgoing energy from the parent ledger after completion. Existing copper cascade generations and downstream secondary processing remain in place. All queues must drain without overflow.

Preserve the current user-selected research model: patient1mm/CT segmentation clock1/16, Cu0.05mm, air0.2mm, historychunk262144, primary fluctuationsON, patient-secondary/Cu-fragment restricted fluctuationsOFF, hard delta unchanged, explicit electronsOFF. No additional physical approximation, source restriction, history pruning or changed collimator. No production/default edits.

## Matched timing

Order A0/B0/B1/A1. Each process has3continuous1M batches covering the full plan, one shared context/package set. Exclude first batch from warm timing;4M timed histories per arm,6M simulated per arm,12M overall. Same source CSVs and per-spot seeds across all four runs; repetition is for performance and implementation matching, not independent dose statistics. Cold timing is process startup through first whole-plan dose output; filesystem cache not cleared. No concurrent GPU jobs. Clocks unlocked; control warm rates drift67,839→64,844/s, candidate108,770→107,828/s. This supports a large scheduling gain in this case, not a precise cross-device/general-source gain claim.

| Metric | Inline control | Separate child queue |
|---|---:|---:|
| Pooled cold histories/s | 38,168 | 48,249 |
| Pooled warm histories/s | 66,308 | 108,297 |
| Warm gain | — | +63.32% |
| Beamline primary kernel s/M | 9.078 | 2.404 |
| Primary-born Cu child worker s/M | included inline | 0.905 |
| Sum of these two stages s/M | 9.078 | 3.309 |
| Patient primary s/M | 1.045 | 1.037 |

The two-stage sum excludes the existing later copper-cascade processing and other runtime work. Primary-child worker tasks per field repeat exactly:801481/512309,804692/513185,805587/514490 for the three batches. All12candidate field queues complete with zero overflow. Separating children both changes scheduling and specializes kernel code generation; no hardware profiler measurements were made, so do not attribute the gain to occupancy/registers/warp divergence as an established causal breakdown.

## Validation

24field-batch quality/energy/coverage/completion reports accepted, zero queue overflow. Twelve matched field pairs: nuclear/elastic event counts, live ion terminal-reason counts, Schneider integer work counters and unified EM audits exactly match. Native copper ion loss/transport audit lines also exactly match; no AIR_IONIZATION_CLOCK line in either arm. Maximum dose relative L1=4.81081714e-8; max absolute dose change/control peak=5.39824789e-7, consistent with FP32 scoring order. Max relative energy residual8.59442208e-7.

This is a matched GPU implementation check against the existing research model, not new TOPAS acceptance. Keep the successful candidate isolated pending broader implementation integration. The 240M earlier warm73,657/s used2.5M batches and is not the matched control for this gain.

Artifacts: `out/carbon_minibeam_cu_queue_20261010/`: source snapshot, build logs, candidate/control binaries, source hashes and provenance, run_abba.py, analyze.py, and all raw matched results under remote_results/. Native audit additions are in remote_results/matched_audit.json. Remote mirror: `/home/v/MAIGO_clean_a6000_20261010/carbon_minibeam_cu_queue_20261010/`.

Control SHA256 `3363d1d7ade01f2a6272808be7506a16cd26b10de0ca323ec0b627037a4a692e`.
Candidate SHA256 `371601b07617754f8b3d84bf279a6681896424e37fd470f73e426e09300a17d4`.

## High-statistics TOPAS comparison and default adoption

At the user's request, validate the candidate against the existing unthresholded optimized-weight TOPAS replay (150,711,884 incident histories), rather than the thresholded optimization DIJ. Run 240,000,000 GPU histories in 96 continuous 2.5M whole-plan batches, with the exact source CSVs, weights, seeds, collimator and normalization from the original 240M GPU baseline. This is an additional GPU run using the archived high-statistics TOPAS reference; no new TOPAS histories were generated.

Use local3%/0.2mm with search0.02mm, the frozen BODY∪GTV/GTV masks and TOPAS >=0.12Gy (3% of the 4Gy prescription). The threshold comes from prescription; the local dose tolerance still comes from each reference voxel.

| Region | Original inline GPU | Queued GPU | Change | Pass/fail flips |
|---|---:|---:|---:|---:|
| BODY∪GTV,519692voxels | 93.442269652% | 93.442269652% | 0 percentage points | 0 |
| GTV,31184voxels | 94.574140585% | 94.574140585% | 0 percentage points | 0 |

All192matched field pairs have exactly matching nuclear/elastic/live-ion/Schneiderinteger/unifiedEM and native Cu/air counters. Runtime quality/energy/coverage/completion audits pass with zero overflow. Total plan dose relative L1 is6.76448607e-9; maximum absolute dose difference/peak is6.73535244e-8. Maximum energy residual is8.69427888e-7. Campaign wall1776.885s (29.61min), warm135901.94histories/s, cold86426.92histories/s. These use2.5M batches; the earlier interleaved1M ABBA experiment provides the matched +63.32% performance evidence, rather than comparing this throughput directly with an older noninterleaved campaign.

The user authorized default adoption if the gamma change is negligible. With zero changed pass/fail voxels, enable the shared Cu-child queue by default for C12 CT minibeams in `src/detail/sycl_pipeline_setup.inc`. Preserve proton eligibility/capacity and keep carbon water excluded. Do not promote the research patient1mm/clock1/16, electronOFF or secondaryrestrictedfluctOFF settings: the release preset physics remains unchanged.

Matched default-configuration implementation check: carbon CT and carbon water minibeams,2x100k histories per arm, current full electron/photon transport, primary/secondary/Cu fluctuations and0.1mm ion cap. All source/discrete/EM/energy/coverage/completion/zero-overflow checks pass. CT per-batch doseL1 <=2.727e-8; water <=2.935e-8. The initial inherited proton task budget2*histories rejected the lungCT batch because231070tasks exceed200000capacity. C12 now budgets16*histories capped by configured max_secondaries; retain explicit rejection on any overflow. Both lung batches231070/226629tasks complete; carbonwater has no Cu-child queue. This is an implementation check, not a throughput experiment.

Absolute carbon dose physics acceptance remains deferred: the archived TOPAS replay lacks GenericIon nuclear elastic. Unchanged gamma validates this scheduling change; it does not establish a new absolute TOPAS acceptance or remove reference statistical uncertainty.

Artifacts: `out/carbon_minibeam_cu_queue_20261010/highstat/results/gamma_report.json`, normalized GPU dose, full192pair audits, source manifests and original reference hash; `highstat/remote_results/report.json` for timing/runtime; `production_regression_v2/integration_audit.json` for release-preset checks and queue completion; isolated pre/post source/binary snapshots under `production_control/` and `production_candidate/`.
