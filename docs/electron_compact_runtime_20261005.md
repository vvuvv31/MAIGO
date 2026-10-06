# Electron collision storage and local photon aggregation — 2026-10-05

Status: installed as the shared collision representation. Formal broadbeam and matched head checks pass; all final default smokes pass.

## Scope

This implements the first two steps of the [26-composition assessment](electron_26_composition_assessment_20261005.md), on the large charged/photon collision arrays. All 82 Schneider density nodes plus Water_75eV, native scalar tables, production cuts, hard-rate certification and transport kernels remain. Birth and joint binary formats are unchanged. This is not a 26-material electron approximation.

MAIECF01 charged and MAIPHF01 photon store collision records in the actual runtime layout: 64-bit integer CDF threshold, FP32 energy fraction and cosine, and PDG/padding integers (24 bytes instead of 32). Existing MAIECB02/MAIPHB02 index files retain83 entries. Native scalar grids remain FP64 on disk and are validated/converted on the host; GPU transport/scoring remains FP32. The loader uploads these components directly, without the intermediate FP64 device banks.

For photon pdg=0, the current sampler terminates at the interaction position and scores the packet weight W. Selected energy fraction/cosine do not change that terminal score or continuation. Within each energy/process channel, the compiler retains every nonlocal outcome and its original integer probability interval, and aggregates local outcomes into one terminal entry. It changes the random-number-to-outcome mapping, not the discrete marginal probability law. No spatial local-deposition substitution, cutoff change, nonlocal thinning, tail truncation or independent energy/angle sampling is introduced. Charged local outcomes are not merged.

## Storage and proof

Charged/photon files total2,413,799,800bytes native,1,814,252,640bytes direct FP32,1,363,383,144bytes after photon terminal aggregation. These exclude birth/joint and index metadata. The whole three-package total would fall from5,199,833,374 to4,149,416,718bytes with the merged representation.

All74,943,395 original collision outcomes and106,655channels are checked. Direct fields match the current GPU conversion exactly. Every nonlocal integer probability, energy and angle is preserved by aggregation, as is the summed local probability. This is not a claim of identical same-seed transport after reordering. Runtime energy/completion/coverage/overflow and dose comparisons remain required.

## Reproduction

Use scripts/compact_electron_runtime.py with a fresh output directory, optionally --merge-photon-local. scripts/check_compact_electron_runtime.py checks all runtime fields or all probability intervals. scripts/validate_compact_electron_runtime.py runs matched three-arm four-CT screens. scripts/validate_compact_electron_dose.py verifies formal broadbeam controls/configs and runs an independent matched complete-head-plan screen.

Raw evidence: out/electron_compact_runtime_20261005/. No digest checks or FP64 dose fallback.

## Measured screens and broadbeam qualification

Three-arm, three-batch screens use ordinary batch sizes: proton broadbeam1,253,150; carbon broadbeam925,987; lung proton minibeam500,000; lung carbon minibeam250,000. Pure-format conversion preserves every compared EM/charged/photon/joint work counter; dose relativeL1 is at most3.43e-8. Aggregation retains all runtime/energy/completion/coverage/zero-overflow gates; its four screen dose relativeL1 values range2.00e-6–1.28e-5. Primary EM audits remain identical; downstream electron/photon counts change with the reordered uniform-to-outcome mapping.

| CT case | Old resident primary/s | Direct FP32 | Aggregated | Old first setup s | Aggregated first setup s |
|---|---:|---:|---:|---:|---:|
| Proton broadbeam | 163,810 | 160,469 | 159,392 | 3.623 | 2.167 |
| Carbon broadbeam | 75,433 | 76,023 | 76,435 | 3.870 | 2.442 |
| Proton lung minibeam | 25,909 | 25,634 | 25,765 | 5.031 | 2.873 |
| Carbon lung minibeam | 8,486 | 8,597 | 8,616 | 5.471 | 4.017 |

These are unlocked-clock screens; they do not resolve small steady-state differences or establish a throughput gain. Setup includes the full transport setup, not just electron loading. Aggregation reduces mapped photon payload from1,375,344,312 to924,474,816bytes, saving450,869,496bytes; it does not claim a measured total GPU peak-memory reduction.

Formal broadbeam repeats two seeds with25,063,000proton and12,963,818carbon primary. Verified controls use identical configs/seeds/history counts except collision storage paths. TOPAS local1%/1mm remains99.883408%/99.794424%, search0.1mm, both>99%. Matched old GPU local gamma is99.999760%/99.999347%; dose relativeL1 is9.12054e-7/1.58123e-6. All34 new batches pass runtime gates. No dose fitting or registration is used.

## Head matched screen and installation

The independent RT07575 **head** screen uses6×1,983,920=11,903,520primary per arm, all487positive spots, ion step0.1mm and the unchanged full-plan geometry/normalization. New versus old local3%/0.2mm is100%, search0.02mm, BODY/reference>=10%. Dose relativeL1 is2.72989e-6; maximum local difference is0.105963%, 99th percentile0.012913%. All12batches pass runtime gates. This is a reduced-statistics complete-plan comparison, not the0.1% qualification campaign: TOPAS gamma is86.805275% old and86.804736% new. Neither passes the absolute>99% gate; the earlier0.1% baseline gate remains unresolved.

The merged representation is installed in the stable data/electrons/shared/charged and photon paths as electrons_shared_v3. Proton/carbon birth bytes and joint are unchanged; only dependency manifests point to the updated shared package. Total unique bytes are4,149,416,718, versus5,199,833,374old:20.20097% reduction. The changed native collision components and former manifests are archived in data/archive/electron_collision_native_20261005/, not a default fallback. Offline conversion/check scripts default to this explicit archive. Rate/elastic scalar diagnostics use header-only loading and remain compatible with the active collision format. Native full-trajectory/FP64 diagnostic tools must explicitly select archived records; they are not the active runtime.

The83-node/12,300-piece rate certificate passes with maximum bound/majorant ratio0.2352709; six intentionally bad certificate fixtures are rejected. No density scaling approximation or electronic cutoff change is introduced.

## Final default closeout

All eight actual default presets pass two50k-primary batches, plus two independent100k-primary head smoke batches (18batches). This includes empty-selection water broadbeam and actual electron tracking in water minibeams. Across100runtime batches (36screen,34formal broadbeam,12matched head,18default smoke), runtime/energy/coverage/completion/zero-overflow checks pass. No broader water or carbon-minibeam dose acceptance is inferred from these smokes.

Active component sizes and schemas pass scripts/check_electron_packages.py. Source and package regeneration use the explicit archived native collision records; the density registry and birth/joint still require separate work before a26-composition electron model can be adopted. This closeout accepts the compact representation and startup/memory benefit, not an established steady-state throughput gain or an absolute head TOPAS pass.
