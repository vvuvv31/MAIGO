# Schneider CT material-dependent transport plan

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
| [12](steps/12-primary-only-observables.md) | TODO | Explicit primary-only mode and validation scorers | 11 |
| [13](steps/13-primary-ct-validation.md) | TODO | Slab + staircase primary CT milestone | 12 |
| [14](steps/14-schneider-stopping.md) | TODO | TOPAS-derived 25-section stopping tables | 13 |
| [15](steps/15-schneider-mcs.md) | TODO | Exact 25-section radiation-length MCS path | 14 |
| [16](steps/16-cinel03-schema.md) | TODO | Element-target correlated-event package schema | 15 |
| [17](steps/17-c12-element-campaigns.md) | TODO | C12 x 13-target TOPAS final-state campaigns | 16 |
| [18](steps/18-material-target-runtime.md) | TODO | Partial-rate target selection and CINEL03 replay | 17 |
| [19](steps/19-c12-fragment-validation.md) | TODO | C12 fragmentation validation in Schneider media | 18 |
| [20](steps/20-secondary-projectiles.md) | TODO | Prioritized secondary projectile coverage | 19 |
| [21](steps/21-heterogeneous-and-dicom-gates.md) | TODO | Heterogeneous and real-DICOM research gates | 20 |

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
```
