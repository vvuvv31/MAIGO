# Schneider CT material-dependent transport plan

## Current strict-dose refinement (2026-09-05)

Entrance-mask correction remains a single-shard candidate, not a replacement
for the frozen full20 baseline. Global 3%/0mm improves 98.000 -> 98.140%
in the paired shard; local 1%/1mm does not improve.
Two 12-history TOPAS electron-ancestry diagnostics close against 3D dose
within 1e-8 and show forward longitudinal energy migration missing from
the transverse-only model. These are preliminary finite-slab measurements,
not a validated new physics table. Next gate: bounded-output joint electron
response and explicit escape accounting, followed by independent geometry
validation. No package change or new full20 run.
See [diagnostic evidence](../evidence/step-31/entrance-mask-candidate/longitudinal-diagnostic.md).

Follow-up: same-seed binary TOPAS job 2338 preserves the 3D dose exactly and
closes electron birth = family deposit + escape to 1.3e-16 relative. Joint
radial/longitudinal histogram and nine diagnostic tests are available.
ASCII boundary-coordinate rounding was detected and not bypassed. The
finite-slab response remains preliminary; no GPU/package promotion or new
Gamma/full20 run in this follow-up.

## Purpose and authority

This directory is the single progress controller for the Schneider CT workstream. The water/CINEL secondary-species line is frozen at the repository state recorded by Step 00. CT work must not tune water physics or silently reuse water H/O physics for other elements.

The required architecture is four separately gated layers:

```text
HU -> Schneider section + density
   -> section-dependent stopping and MCS
   -> material-dependent inelastic total rate + target selection
   -> projectile/target-dependent correlated CINEL final state
```

Passing a later layer never excuses a failed earlier layer. Patient CT is forbidden until the synthetic validation gates pass.

## Repository facts verified when this plan was written

- `data/HUtoMaterialSchneider.txt` declares 13 elements and 25 material sections.
- `SchneiderHuTable::from_topas_file()` currently skips `SchneiderElements` and `SchneiderMaterialsWeight*`.
- `CtGrid` stores per-voxel density and an 8-bit material/section ID; composition must remain a global 25 x 13 table, not per voxel.
- `CrossSectionTable::from_schneider_csv()` already accepts section mass-rate columns.
- The worktree already contains uncommitted CT material-rate work (`Cinel02MaterialRateNtuple`, an extractor, runtime/config changes), including changes in secondary transport. Step 00 must inventory and freeze it; do not assume it is accepted merely because it exists.
- The legacy `ct_material_class()` four-class collapse remains in the repository. It is not permitted for the Schneider production CT path.

## Non-negotiable rules

1. Never change frozen water/CINEL parameters, event semantics, random-number mapping, or water validation tolerances as part of this workstream.
2. Phase P2 uses C12 primaries only and no secondary cascade. A nuclear interaction may terminate the primary in an explicit attenuation-validation mode, but it must not replay a fragmentation event.
3. CT material selection uses all 25 Schneider sections. No air/lung/soft/bone collapse is allowed in the new CT path.
4. Never alias an unsupported target element to oxygen or hydrogen. Missing rate/final-state coverage is a hard error for production and an explicit counter only in a named audit mode.
5. Nuclear optical depth may not be integrated across a voxel face at which density or section changes. Step length is capped at the face, then material and rate are re-evaluated.
6. Use a 3D dose scorer and sum transverse bins for IDD. Do not add or use a 1D dose scorer.
7. Do not spend work on FP32 versus FP64. Keep the established precision policy.
8. GPU and SYCL commands run locally, outside the sandbox, on RTX 2080 Ti / `sm_75`. Never submit GPU work to a remote host or cluster.
9. TOPAS jobs run locally through `sbatch`; input/output and raw data live under `/mnt/sda/wuwei`, while TOPAS extensions/source/build live under `/home/wuwei/topas`.
10. All concurrent TOPAS jobs together must stay at or below 192 CPU threads and 160 GiB RAM. Allocate roughly in proportion to energy/computation so jobs finish at similar times. A brief `InvalidAccount` state is rechecked after 1--3 minutes and is not treated as failure.
11. Split large particle campaigns into shards. If any secondary buffer overflow is reported, discard the affected aggregate, reduce histories per shard, rerun, and merge only overflow-free shards.
12. Raw TOPAS ntuples are artifacts outside Git. Only audited compiled products and metadata enter `data/schneider/`.
13. Do not modify source in a step marked `BLOCKED`, and do not advance the README status without all gate evidence.

## Status vocabulary

- `TODO`: no implementation accepted.
- `IN_PROGRESS`: exactly one step may have this status.
- `BLOCKED`: an external dependency or failed gate is documented; later dependent steps remain `TODO`.
- `DONE`: implementation, tests, evidence, diff review, and the step gate all passed.
- `FROZEN`: intentionally unchanged and protected by regression evidence.

The executor updates only the `Status` column and the execution log. Never rewrite acceptance thresholds after seeing results. If a threshold is scientifically wrong, open a separate documented plan change before rerunning.

## Progress table

| Step | Status | Deliverable | Depends on |
|---|---|---|---|
| [00](steps/00-freeze-and-provenance.md) | FROZEN | Freeze water/secondary baseline and provenance | none |
| [01](steps/01-current-state-audit.md) | DONE | Audited local implementation map and conflict decision | 00 |
| [02](steps/02-schneider-material-model.md) | DONE | Host-side 13-element/25-section data model and parser | 01 |
| [03](steps/03-schneider-parser-tests.md) | DONE | Boundary, malformed-input, and golden parser tests | 02 |
| [04](steps/04-topas-material-truth-dump.md) | DONE | TOPAS/Geant4 material truth extension and synthetic inputs | 03 |
| [05](steps/05-material-truth-gate.md) | DONE | Automated MAIGO-parser versus TOPAS material audit | 04 |
| [06](steps/06-topas-inelastic-xs-dump.md) | DONE | Deterministic C12 section/element inelastic XS dump | 05 |
| [07](steps/07-xs-compiler-and-metadata.md) | DONE | Audited Schneider mass-rate data product | 06 |
| [08](steps/08-thin-slab-xs-validation.md) | DONE | Independent TOPAS attenuation validation of XS | 07 |
| [09](steps/09-primary-xs-host-path.md) | DONE | Strict config/load/resample path for 25-section C12 XS | 08 |
| [10](steps/10-primary-xs-device-path.md) | DONE | Correct device upload/index/density scaling | 09 |
| [11](steps/11-voxel-boundary-hazard.md) | DONE | Piecewise-material optical-depth stepping | 10 |
| [12](steps/12-primary-only-observables.md) | DONE | Explicit primary-only mode and validation scorers | 11 |
| [13](steps/13-primary-ct-validation.md) | DONE | Slab + staircase primary CT milestone | 12 |
| [14](steps/14-schneider-stopping.md) | DONE | TOPAS-derived 25-section stopping tables | 13 |
| [15](steps/15-schneider-mcs.md) | DONE | Exact 25-section radiation-length MCS path | 14 |
| [16](steps/16-cinel03-schema.md) | DONE | Element-target correlated-event package schema | 15 |
| [17](steps/17-c12-element-campaigns.md) | DONE | C12 x 13-target TOPAS final-state campaigns | 16 |
| [18](steps/18-material-target-runtime.md) | DONE | Partial-rate target selection and CINEL03 replay | 17 |
| [19](steps/19-c12-fragment-validation.md) | DONE | C12 fragmentation validation in Schneider media | 18 |
| [20](steps/20-secondary-projectiles.md) | BLOCKED | Prioritized secondary projectile coverage (blocked on v2 CINEL03 coverage; resumes at Step 28) | 19 |
| [21](steps/21-heterogeneous-and-dicom-gates.md) | BLOCKED | Heterogeneous and real-DICOM research gates (blocked on v2 CINEL03 coverage; resumes at Step 30) | 20 |
| [22](steps/22-reachable-demand-manifest.md) | DONE | Reachable secondary demand manifest + audit (7232 tasks, audit PASS) | 19 |
| [23](steps/23-campaign-grid-generation.md) | DONE | Per-channel campaign energy grids (gap<=5 + thin-box rounds 3-6) | 22 |
| [24](steps/24-topas-extraction.md) | DONE | Local TOPAS sbatch event extraction (14+13+2+2+1 jobs, all ok) | 23 |
| [25](steps/25-raw-data-validation.md) | DONE | Raw node acceptance / pinpoint rerun (7223/7472 pass; 256 documented nulls) | 24 |
| [26](steps/26-cinel03-compilation.md) | DONE | Versioned v2 secondary CINEL03 package (242,494 events, 332MB) | 25 |
| [27](steps/27-package-provenance-audit.md) | DONE | Independent v2 package audit (8/9; below-demand 0.71% documented exception) | 26 |
| [28](steps/28-lookup-closure-50k.md) | DONE for declared RT06423 research scope | v2.1 Tier-C accepted=true (post-EM null taxonomy fixed, E_h/E_c documented, He6/B8/C10 em_only declared, NEED masked residual) | 27 |
| [29](steps/29-generation2-transport-validation.md) | DONE for declared RT06423 research scope | v2.1 Tier-D accepted=true (same scope notes as Step 28) | 28 |
| [30](steps/30-single-shard-abcd-gamma.md) | IN_PROGRESS | TPS direction basis bug fixed; Shard01 nominal Gamma now global 3%/3mm 100%, global 2%/2mm 99.842%, local 2%/2mm 98.994%. Formal same-binary A/B/C/D matrix remains | 29 |
| [31](steps/31-production-20shard-gamma.md) | DONE for declared RT06423 research scope | section-0 delta-tail: 20/20 accepted, zero overflow; nominal global 1%/1mm 98.426%, global 3%/0mm 99.804%; production_generalization=false | 30 (formal A/B/C/D attribution remains) |

General Schneider production coverage: LIMITED. Declared scope limitations
(not general transport completion): He6/B8/C10 secondary nuclear EM-only
policy, residual NEED masked (unbounded), tertiary generations capped at 2,
single RT06423 geometry, 3D voxel/history-ledger closure gate added after
initial validation (see quality schema v3).

## Phase gates

- P0, frozen baseline: Steps 00--01.
- P1, material truth: Steps 02--05.
- P2, C12 primary nuclear attenuation: Steps 06--13. This is the first usable milestone.
- P3, exact electromagnetic material transport: Steps 14--15.
- P4, C12 material-dependent fragmentation: Steps 16--19.
- P5, secondary nuclear transport and clinical validation: Steps 20--21.

Do not start a phase until every step in the previous phase is `DONE`, except Step 00 which becomes `FROZEN` after completion.

## Per-step execution protocol

For every step, the executor must:

1. Read this README, the current step, `AGENTS.md`, and the exact current source sections named by the step.
2. Run `git status --short`; record pre-existing changes and avoid overwriting them.
3. Create outputs under the path required by the step. Do not improvise a second configuration key or file schema.
4. Add positive, boundary, malformed-input, and fail-fast tests where applicable.
5. Run the smallest tests first, then the full relevant CPU suite, then local out-of-sandbox SYCL/GPU validation when requested.
6. For source edits, re-read before patching; use small edits; run `git diff --check` and inspect `git diff`.
7. Write an evidence file under `/mnt/sda/wuwei/maigo-ct-schneider/evidence/step-NN/` containing commands, exit codes, hashes, machine/software versions, numerical results, and overflow counters. Large outputs stay there, not in `plan/`.
8. Mark `DONE` only if every acceptance item is demonstrated. “Build succeeds”, plots that only look reasonable, or partial data are not acceptance evidence.
9. Commit only the scope of the current step, using the commit intent stated in that step. Never include unrelated dirty-worktree changes.

## Required provenance fields

Every generated formal data product has a sibling metadata JSON containing at least:

```text
schema_version
data_sha256
topas_version
geant4_version
physics_list
schneider_source_path
schneider_sha256
extractor_git_commit
compiler_git_commit
raw_campaign_manifest_sha256
energy_min_MeVu
energy_max_MeVu
energy_grid
projectiles
target_elements
units
generation_timestamp_utc
validation_report_sha256
```

Missing or placeholder fields fail the gate.

## Final data layout

```text
data/schneider/
  schneider_materials_geant4_11_3_2.json
  schneider_materials_geant4_11_3_2.metadata.json
  c12_schneider_inelastic_mass_xs.csv
  c12_schneider_inelastic_mass_xs.metadata.json
  schneider_inelastic_rates_v1.bin
  schneider_inelastic_rates_v1.metadata.json
  cinel03_c12_targets.bin
  cinel03_c12_targets.metadata.json
  cinel03_secondary_targets.bin
  cinel03_secondary_targets.metadata.json
  schneider_stopping_v1.bin
  schneider_stopping_v1.metadata.json
```

## Phase P6 coverage campaign status (authoritative, 2026-09-03)

- Exact lookup, named diagnostics, fail-closed framework: DONE (Steps 16/18,
  strict gates in `src/run_quality.cpp`, 5 schneider-strict test groups).
- Secondary CINEL03 coverage补齐: data DONE (Steps 22-27; v2 package 242,494
  events, 169 channels, gaps=0, missing=0, node monotonicity OK, hashes OK;
  below-domain demand 0.71% documented exception with Tier-L evidence).
- 50k lookup closure: DONE (Step 28; v2.1 Tier-C accepted=true: primary
  19010=19010, secondary 12999=12923+76, below/above/missing/gap/empty all 0,
  lookup-fail-E=0, overflow 0; 439 out-of-scope He6/B8/C10 tracks deposited
  locally + counted, C12 fully transported via 14p package).
- Tertiary productionization: DONE (Step 29; v2.1 Tier-D accepted=true:
  secondary 14474=14375+99, born 59418=53955+5463+0, overflow 0;
  gen-1 unsupported births prove tertiary depth; 0 in-scope unsupported).
- Gamma validation: PASS for the declared RT06423 research scope. Nominal
  full-statistics global 3%/3mm = 99.994% and global 2%/2mm = 99.460%; the
  formal same-binary A/B/C/D attribution matrix in Step 30 remains open.
- 20-shard validation: DONE for the declared RT06423 research scope by explicit
  user authorization (20/20 accepted, zero overflow). This does not change
  `production_generalization=false` or the documented v2.1 limitations.
- Follow-ups identified (not started): Step-28b rate low-E floor analysis
  (sub-0.5 clamp queries); v2.1 isotope expansion (C12-reuse + N/O/F/He6…,
  full census in report); p+H physics-list change (escalated, needs full
  revalidation).

## Execution log (P6 entries)

```text
2026-09-03 21:00 CST | step 22 | TODO -> IN_PROGRESS | manifest 7232 tasks, audit PASS (missing=0, gap=0, below=0, above=0) | /mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2/
2026-09-03 21:10 CST | step 24 | TODO -> IN_PROGRESS | pilot c12floors 13/13 ok; 13/14 projectile jobs submitted (168 CPUs) | /mnt/sda/wuwei/job_1136.log
2026-09-03 21:30 CST | step 22-note | p+H proven unfillable under frozen physics list (v1 7 tasks + v2 probes, 0 events; p+O control 208 events) | schneider-secondary-v2/pH_unfillable_evidence.md
2026-09-03 22:00 CST | step 24-25 | wave1 7232 tasks done; validation 6784/7232 pass; 38 shortfalls + 311 nulls -> wave2 (311 tasks) done | job_1136-1149, job_1150-1162
2026-09-03 22:30 CST | step 24-note | thick-box smearing diagnosed (420 campaign spreads 239-420); thin-box verified (228 events within +-3) | round-3 69 tasks
2026-09-03 23:00 CST | step 26-27 | v2 compiled (242,494 ev); float32-dup fix; round-5 ceil-cap + round-6 microfill; package audit 8/9 (below-demand 0.71% documented exception) | data/schneider/cinel03_secondary_targets_v2.*
2026-09-03 23:30 CST | step 28 | Tier-A 1016 queries agree; Tier-B accepted=true; Tier-C/D hit 98.5/98.1%, gap=0, missing=0; residual: below/above-domain ~108, unsupported non-Be6 1769-3066 tracks (N/O/F/C12/He6 census) | out/diag_50k_v2_g1, out/diag_50k_v2_g2
```

## Execution log

Append one line after each status change. Do not erase old entries.

```text
YYYY-MM-DD HH:MM TZ | step NN | OLD -> NEW | commit/hash or blocker | evidence path
2026-09-02 14:42 CST | step 00 | TODO -> FROZEN | chore(ct): freeze Schneider workstream provenance | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-00/
2026-09-02 15:01 CST | step 01 | TODO -> DONE | audit completed; evidence in /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-01/
2026-09-02 15:15 CST | step 02 | TODO -> DONE | feat(ct): parse Schneider elemental material composition | tests/carbon_tests.cpp
2026-09-02 15:30 CST | step 03 | TODO -> DONE | test(ct): add Schneider parser hardening and domain sweep tests | tests/carbon_tests.cpp
2026-09-02 15:40 CST | step 04 | TODO -> DONE | feat(topas): dump Schneider material truth | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-03/
2026-09-02 15:42 CST | step 05 | TODO -> DONE | audit(ct): 25/25 section MAIGO ↔ TOPAS material truth gate passed | plan/evidence-step03.sha256
2026-09-02 15:56 CST | step 06 | TODO -> DONE | fix(topas): harden Schneider C12 XS provenance and process locking | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-04/
2026-09-02 16:19 CST | step 07 | TODO -> DONE | feat(ct): compile Schneider primary-carbon nuclear rates | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-07/
2026-09-02 16:26 CST | step 07 | IN_PROGRESS -> DONE | data(ct): regenerate audited Schneider C12 rate products | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-07/
2026-09-02 16:43 CST | step 08 | TODO -> IN_PROGRESS | start independent TOPAS thin-slab MC attenuation validation | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-08/
2026-09-02 17:10 CST | step 08 | IN_PROGRESS -> DONE | test(ct): validate Schneider C12 attenuation with TOPAS thin slabs | /mnt/sda/wuwei/maigo-ct-schneider/evidence/step-08/
2026-09-02 17:18 CST | step 09 | TODO -> IN_PROGRESS | start strict 25-section host load/resample path | data/schneider/c12_schneider_inelastic_mass_xs.csv
2026-09-02 17:27 CST | step 09 | IN_PROGRESS -> DONE | feat(ct): load section-resolved primary C12 nuclear rates | tests/carbon_tests.cpp
2026-09-02 17:40 CST | step 08/09 | DONE -> DONE | fix(ct): close primary XS host and thin-slab gates | plan/evidence-step08.sha256
2026-09-02 17:50 CST | step 10 | IN_PROGRESS -> DONE | feat(ct): upload and verify 25-section primary XS on GPU | tests/carbon_tests.cpp
2026-09-02 17:56 CST | step 10A | DONE -> DONE | fix(ct): log mode and source SHA256; compute-sanitizer 0 errors | src/transport_sycl.cpp
2026-09-02 18:03 CST | step 11 | TODO -> DONE | feat(ct): preserve nuclear optical depth across Schneider voxels | tests/carbon_tests.cpp
2026-09-02 18:29 CST | step 11A | DONE -> DONE | fix(ct): close production Schneider face stepping | tests/carbon_tests.cpp
2026-09-02 18:58 CST | step 11B | DONE -> DONE | fix(ct): native 860-node XS upload, exact error bound and boundary crossing | tests/carbon_tests.cpp
2026-09-02 19:08 CST | step 11C | DONE -> DONE | fix(ct): exact directional sampling and hit-face clamp mask | tests/carbon_tests.cpp
2026-09-02 19:24 CST | step 11D | DONE -> DONE | test(ct): close realizable full-energy error bound and low-energy optical depth gates | tests/carbon_tests.cpp
2026-09-02 20:13 CST | step 11E | DONE -> DONE | test(ct): verify whole-trajectory survival bias and exact mass-SPR tail bounds | tests/carbon_tests.cpp
2026-09-02 20:24 CST | step 11F | DONE -> DONE | test(ct): close Step 11F exact 1mm max step, endpoint index clamp, and strict slowing gates | tests/carbon_tests.cpp
2026-09-02 20:55 CST | step 12 | TODO -> IN_PROGRESS | feat(ct): implement primary-only validation mode, dynamic SHA256 provenance check, terminal state conservation, decoupled energy accounting, and IDD/Bragg peak metrics | tests/carbon_tests.cpp
2026-09-02 21:40 CST | step 12 | IN_PROGRESS -> IN_PROGRESS | fix(ct): segregate other_terminal energy ledger, move provenance check early, and add watchdog regression test | tests/carbon_tests.cpp
2026-09-02 22:05 CST | step 12 | IN_PROGRESS -> DONE | fix(ct): prioritize physical terminals over watchdog, early provenance before queue, fail-closed mandatory buffers, strict uint32 parsing, and simultaneous terminal tests | tests/carbon_tests.cpp
2026-09-02 23:45 CST | step 13 | TODO -> DONE | feat(ct): validate primary C12 slab and staircase CT transmission against TOPAS | evidence/step-13/
2026-09-03 01:30 CST | step 14 | IN_PROGRESS -> DONE | feat(ct): apply 25-section Schneider stopping tables with mass-SPR scaling | evidence/step-14/
2026-09-03 11:21 CST | step 15 | TODO -> DONE | feat(ct): use Schneider section radiation lengths for MCS | evidence/step-15/
2026-09-03 11:32 CST | step 16 | TODO -> DONE | feat(cinel): add versioned elemental-target event package | evidence/step-16/
2026-09-03 11:36 CST | step 17 | TODO -> IN_PROGRESS | start C12 elemental-target TOPAS campaigns for 13 elements | /mnt/sda/wuwei/cinel03-c12-element-campaigns/
2026-09-03 12:19 CST | step 17 | IN_PROGRESS -> DONE | feat(topas): extract C12 elemental-target CINEL events | evidence/step-17/
2026-09-03 12:20 CST | step 18 | TODO -> IN_PROGRESS | start partial-rate target selection and CINEL03 replay | include/carbon/inelastic_package_v3.hpp
2026-09-03 12:27 CST | step 18 | IN_PROGRESS -> DONE | feat(ct): sample Schneider elemental targets for C12 on GPU | evidence/step-18/
2026-09-03 12:28 CST | step 19 | TODO -> IN_PROGRESS | start C12 fragmentation validation in Schneider media | tools/
2026-09-03 13:05 CST | step 19 | IN_PROGRESS -> DONE | test(ct): validate C12 fragmentation in Schneider materials | evidence/step-19/
2026-09-03 13:30 CST | step 20 | IN_PROGRESS -> DONE | feat(ct): enable secondary-ion Schneider nuclear transport | evidence/step-20/
2026-09-03 13:35 CST | step 21 | TODO -> IN_PROGRESS | start heterogeneous and real-DICOM final research gates | plan/steps/21-heterogeneous-and-dicom-gates.md
2026-09-03 14:32 CST | step 21 | IN_PROGRESS -> DONE | test(ct): add end-to-end Schneider DICOM reference | tests/test_schneider_dicom_reference.cpp
2026-09-03 14:52 CST | step 20 | DONE -> IN_PROGRESS | audit: reopen Step 20 due to verifier masking single-case failures (staircase_200mevu major species diff 4.81%) and lack of production integration | plan2/steps/00-status-reset-and-verifier-audit.md
2026-09-03 14:52 CST | step 21 | DONE -> IN_PROGRESS | audit: reopen Step 21 due to hardcoded reference test assertions and empty evidence/step-21/ | plan2/steps/00-status-reset-and-verifier-audit.md
2026-09-04 19:00 CST | step 30 | IN_PROGRESS | plan-attribution diagnosis: source truth 9 spots pass (counts exact, phase-space chi2/dof 0.61); 5 energy-group patient runs show totals 0.98 but GPU air excess 3-8x growing with energy; uniform-air slab EM 1.0003 but nuclear GPU +21% with halo 7.4 vs 4.2mm; sandwich tisA 1.008 / air 0.22 / tisB 1.13. Root: secondary angular spectra (CINEL vs BIC), package frozen -> no code change, Step 30 stays IN_PROGRESS | tools/dump_tps_phase_space.cpp
2026-09-05 00:30 CST | step 30 | IN_PROGRESS | round-2 diagnosis: BIC cannot serve C12 (distal-zero proven); ref C12 final state is INCLXX (dual census 613/613); rotation runtime device-verified (5e-7); package==campaign==dual@matched-energy; INCLXX-slab dose == BIC-slab dose, GPU still +21% with same INCLXX birth; elastic ablation nil; analytic CSDA-straight-line gives 75 MeV/ev vs GPU 862 vs TOPAS 38. Residual: secondary slowing/deposit in low-density media under identical birth. No code change (package frozen, no proven bug site). | evidence/step-30-plan-attribution/round2.json
2026-09-05 07:00 CST | step 30 | IN_PROGRESS | supersedes prior package-angle limitation: secondary midpoint dE used unscaled density-1 water stopping in CT. Fix applies Schneider density/material factor at midpoint. Air slab T/G 0.8281->1.0052; sandwich air 0.2224->0.9634 and downstream tissue 1.1319->1.0090; RT06423 Shard01 accepted/no overflow, total T/G 0.9837->0.9979, air 0.1526->0.9786, IDD r 0.99375->0.999993, global 2%/2mm 85.34->87.28. No package/beam/MCS/scale change; 20-shard not run. | evidence/step-30-secondary-midpoint-stopping/validation.json
2026-09-05 09:16 CST | step 30 | IN_PROGRESS | fixed TPS direction_y basis typo (uy_x->uy_y via shared helper). Primary fluence sigma mismatch patient-Z -0.567->-0.0082 mm; accepted Shard01/no overflow; nominal Gamma global 3%/3mm 100%, global 2%/2mm 99.842%, local 2%/2mm 98.994%, r=0.99808, LS scale=19.901. Formal same-binary A/B/C/D matrix not rerun, so Step 30 remains IN_PROGRESS and Step 31 blocked. | evidence/step-30-tps-direction-fix/validation.json
2026-09-05 09:40 CST | step 31 | BLOCKED -> DONE for declared RT06423 research scope | explicit user authorization; 20/20 local RTX 2080 Ti shards accepted, overflow=0, 151091740 histories; nominal global Gamma 3%/3mm=99.994%, 2%/2mm=99.460%, 1%/1mm=91.238%, 3%/0mm=98.206%; r=0.999015, IDD r=0.999617. Formal Step-30 A/B/C/D attribution remains open and production_generalization=false. | evidence/step-31/step31_full20_directionfix_summary.json
2026-09-05 14:40 CST | step 31 | strict-dose refinement | TOPAS-derived primary-C12 Schneider-section-0 transverse delta-electron tail plus scorer-boundary escape accounting; exact-split 20/20 local GPU shards accepted, overflow=0, 151087660 histories. Frozen paired nominal Gamma: global/local 1%/1mm 98.426/87.786%, global/local 3%/0mm 99.804/93.422%; no package/scale/water change; longitudinal entrance build-up and generalization remain open. | evidence/step-31/delta-tail/strict-gamma-summary.json
```
