# CT minibeam secondary Fermi–Eyges transport

On 2026-10-19 the patient-medium charged-secondary FE implementation from
`minibeam-secondary-fe` was selectively adopted as the clean default. The
research CMake switch, test drivers and campaign scripts were not imported.

## Scope

Dedicated `proton_minibeam` and `carbon_minibeam` pipelines with CT enabled
use Urban for patient primaries and the existing shared segmented
Fermi–Eyges operator for patient-medium charged secondaries. This includes
charged copper-born products after they enter the patient. Generic charged
fragment species remain supported. Copper/air workers retain their existing
models: in particular, native carbon copper fragments retain Urban and proton
upstream fragments retain their existing Highland update. Water and generic
minibeam medium workers retain Urban; broadbeams retain FE.

The secondary FE branch explicitly enforces the same outer patient step cap
of 0.5 mm. The scattering subsegment limit is independent, controlled by
`fermi_eyges_max_segment_mm` (0.1 mm in CT minibeam release presets). The
existing FE tail configuration is preserved. `multiple_scattering_model:
urban_v2` continues to select the primary model; no secondary model selector
or experimental build switch is added to the release runtime.

The shared policy is named in `include/carbon/scattering_policy.hpp`.
Startup `[medium-mcs]` and the `medium_scattering` object in
`quality_report.json` report the effective primary and secondary medium
models independently. They do not describe copper/air scattering.

The switch changes secondary trajectories and their resulting step and
interaction counts. It is a scattering approximation, not an algebraic
rewrite of Urban. It retains unrestricted stopping/ranges, primary
fluctuations, default secondary mean loss, nuclear elastic/inelastic
transport, recoil stopping, neutral energy sinks and FP32 scoring. Source,
CT placement, plan weights, per-spot seeds and whole-campaign normalization
are preserved. No new physics package is required.

## Full-plan dose comparisons

All comparisons below use a fixed mask of BODY and TOPAS dose >0.4 Gy,
local 3% gamma, and 0.02 mm search spacing. There is no fitted dose scaling
or registration. Each GPU arm uses the stated number of incident histories.

| Case and ion | Histories per GPU arm | Urban vs TOPAS: 0.2 / 0.1 / 0 mm (%) | FE vs TOPAS: 0.2 / 0.1 / 0 mm (%) |
|---|---:|---|---|
| RT07575 head, proton | 1,000,000,000 | 99.8131 / 98.5999 / 93.8965 | 99.8167 / 98.6027 / 93.9167 |
| RT06423 head, proton | 1,000,000,000 | 99.9632 / 99.0567 / 96.4237 | 99.9626 / 99.0512 / 96.3953 |
| 20022516 lung, proton | 3,000,000,000 | 99.5238 / 98.1395 / 94.9951 | 99.5235 / 98.1414 / 94.9748 |
| RT07575 head, carbon | 1,000,000,000 | 99.3934 / 98.7121 / 95.3858 | 99.4071 / 98.7161 / 95.3648 |

The largest decrease across the reported criteria is 0.0284 percentage points
for proton plans and 0.0210 for the carbon head plan. This is scoped evidence,
not acceptance for all cases or all individual voxels. No LET acceptance is
inferred from these physical-dose comparisons.

Both proton arms were freshly run with the current unrestricted implementation.
For carbon, the completed unrestricted all-Urban 1B result was reused from
L4; FE was run on RTX 2080 Ti and A6000. Required package hashes and original
400-batch counts/seeds were matched. Carbon FE−Urban relative-dose RMS in
the mask was 0.2116%, with BODY integral change +0.00014%. The respective
GPU dose standard-error RMS values were 0.8061% and 0.8061%; TOPAS was
0.8030%. Device assignment differs, so no paired-difference standard error
is claimed for the reused carbon baseline. This baseline is unrestricted;
older finite-cut Urban gamma values are not substituted.

Every batch passed runtime energy, EM/coverage/completion and live-species
audits, with zero queue overflow. Secondary counts are not required to match
Urban because the scattering operator changes; primary transport is preserved.

## Timing and provenance

Matched RT07575 31.2M-history-per-arm ABBA measurements, with the first
batch of each resident process excluded, gave the following hot throughput:

| Ion | GPU | Urban secondaries (k/s) | FE secondaries (k/s) | Whole-run increase |
|---|---|---:|---:|---:|
| Proton | RTX 2080 Ti | 364.9 | 391.1 | 7.18% |
| Proton | RTX A6000 | 533.0 | 552.4 | 3.65% |
| Carbon | RTX 2080 Ti | 111.6 | 123.5 | 10.70% |
| Carbon | RTX A6000 | 189.1 | 199.5 | 5.49% |

These measurements use that experiment's batch sizes and driver; they do
not replace large-batch throughput tables or establish a gain for every case.
The carbon 1B FE campaign measured 126.1 k/s and 195.1 k/s respectively,
without rerunning its Urban timing baseline. CUDA AOT targets were sm_75 and
sm_86, with `CUDA_DISABLE_PTX_JIT=1` for every execution.

Local research evidence is under
`/mnt/sdb/wuwei/MAIGO_mb_secondary_fe/tmp/`:

- `secondary_fe_current_20261019/`: frozen source/build scope, native images,
  matched ABBA timing, source and primary audits.
- `proton_mb_secondary_fe_fullplan_20261019/RESULTS.json`: three-case
  high-statistics comparisons (10B newly simulated histories across both arms).
- `carbon_mb_secondary_fe_RT07575_1B_20261019/RESULTS.json`: carbon 1B FE
  comparison and retrieved unrestricted Urban baseline provenance.

Release integration checks are recorded separately in
`MAIGO_clean/tmp/secondary_fe_merge_20261019/`. They verify integration and
unaffected paths; they are not new TOPAS acceptance or performance measurements.

The native sm_75 all-five-pipeline release build passed two-batch checks for
all eight presets and four generic configurations (960,000 histories across
reference and candidate). CT minibeams were compared with the validated
FE core using the same CLI; the six other presets and generic paths were
compared with pre-adoption clean. Source allocations and 3,344 release
discrete audit values matched. All runtime quality/coverage/completion and
zero-overflow checks passed; the maximum relative dose L1 difference across
the integration checks was 2.662e-7. The integrated secondary transport was
also checked against the research source with its experimental guard removed.
The local `build/carbon_mc` contains the new release default.
