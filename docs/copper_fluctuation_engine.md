# Shared copper energy-loss fluctuation engine

The proton and ion copper transport paths share
[`sample_copper_fluctuation`](../include/carbon/copper_fluctuation.hpp).
The engine serves existing proton/C12 primaries and the existing charged
fragment registry. It does not enable additional source species or introduce
new data readers, packages, or configuration aliases.

## Particle state and model selection

`CopperFluctuationStep` explicitly carries kinetic energy, reference mass,
Z/A, effective charge squared, mean energy loss, accepted true path, density,
mean-loss type, whether to sample restricted fluctuations, and the native-fragment
calibration context. Mass is supplied
by the caller's current kinematic convention; it is not inferred from A.
This preserves the calibrated C12 A*amu and proton mass conventions.

`CopperMeanLoss::total` includes the expected hard-delta contribution;
`CopperMeanLoss::restricted` excludes it. The proton preparation converts the
former to a restricted mean before sampling. Ion preparation requires a
restricted mean. This explicit distinction prevents counting the same delta
energy twice.

All four release minibeams use `minibeam_copper_straggling_model: g4_delta`.
The transport-facing entry selects primary preparation from explicit Z/A.
Two internal physical preparations remain:

- `universal_proton` preserves the current G4UniversalFluctuation proton
  preparation, independently extracted cut, and validated hard-delta domain.
- `ion` uses the supplied effective charge, G4IonFluctuations dispersion
  correction, and the extracted projectile spin and magnetic moment. Its
  restricted sampler also uses Universal at the existing high-energy model
  transition; the low-energy ion branch remains particle dependent.

The common backend samples restricted loss, bounded Poisson components, and
hard-delta transfers. Larger ion proposal means are split into independent
bounded Poissons. Existing form-factor null events and strict invalid-data
failures remain. Disabling restricted fluctuations retains the ion mean and
hard-delta sampling, as before. Original call-site enable conditions remain.

The separate extracted proton/ion cut values and FP32 expression order are
preserved. A shared engine does not imply identical physical parameters or a
new calibration. Native mathematical evaluation remains enabled only in the
previously measured dedicated proton-minibeam primary Cu path. Ion, fragment,
and generic Cu calls retain their preceding arithmetic.

## Independent transport controls

The primary fluctuation selector no longer chooses fragment transport, package
loading, resident-cache retention, or fragment-transport audits.
`minibeam_copper_fragment_native_transport` controls those operations explicitly:
carbon water/CT defaults are `true`, proton water/CT defaults are `false`.
`minibeam_copper_primary_native_mean_loss` retains its independent primary
restricted-mean selection and physical compatibility requirements.

Native fragments use `CopperFluctuationContext::native_ion_fragment`. This
retains the extracted ion cut, variance correction, spin/magnetic-moment and
hard-delta convention for **every native fragment, including H1**. A primary
proton uses its existing Universal calibration. Relabeling an H1 fragment as a
primary proton would change its extracted calibration; that is not part of
this refactor. Particle identity and supplied mass/effective charge continue
to select the particle-dependent corrections inside each preparation.

The old `g4_universal_delta` and `g4_ion_delta` YAML values are removed.
Custom configurations must select `g4_delta` and explicitly set the fragment
switch; no compatibility aliases infer transport from an old model name.
Existing source species, package formats, paths, calibrated masses, steps,
RNG identities and primary/secondary fluctuation enable flags are preserved.

## Extending to another heavy ion

The engine already accepts projectile Z/A, mass and effective charge as
parameters. An additional supported ion needs a matching extracted projectile
record, including spin/magnetic moment, and an actual model-derived effective
charge and restricted mean. A missing nonzero-transport projectile record
fails; it does not use a proton/C12 replacement.

This is the fluctuation layer for solid G4_Cu at the existing density. Enabling
another **source** still requires its source validation, stopping/continuous
EM, nuclear reaction/final-state, elastic/recoil data and transport coverage.
The runtime's supported source species are unchanged. Patient-material loss
continues through the existing shared continuous26/restricted sampler.

## Original engine-unification checks

The component GPU comparison covers 360 conditions with 262,144 samples each:
94,371,840 paired draws. It includes proton total/restricted means, standard and
native proton math, and registry proton/He4/B10/C12 ion calls; restricted
fluctuations both enabled and disabled; varying energy, path and effective
charge. All draws are valid, sampled losses are bitwise identical, and RNG
call counts match the separate prior implementations.

Matched runtime checks also passed on the RTX 2080 Ti, with all five pipelines
compiled ahead of time for `sm_75` and `CUDA_DISABLE_PTX_JIT=1`:

- RT06423 carbon minibeam: original optimized 90°/270° full plan, four matched
  1.2-million-primary campaigns, each using three resident batches.
- RT07575 proton minibeam: original head full plan, 3 million primaries per
  implementation in three resident batches.
- All eight release presets and generic proton CT/carbon water minibeams:
  two batches of 20,000 primaries per implementation and configuration.

Full-plan and generic source allocations/seeds and non-energy integer ledger
counters match. Preset source, discrete physics and live-ion audits match.
All runtime energy, coverage, completion and zero-overflow checks pass.
The maximum relative dose L1 difference across these comparisons is
3.31884e-07; FP32 dose atomics need not accumulate in identical order.

The exact tested executable and static library replace the preceding local
runtime artifacts. Evidence is under
`tmp/copper_fluctuation_unify_20261016/`: `fixture.jsonl`, frozen baseline
headers and fixture source, `VERIFY.json`, and `COMPLETE.json` (source and
binary provenance). These are matched implementation checks, not new TOPAS
or additional-source acceptance. This refactor makes no throughput claim.

## Transport-decoupling checks

The subsequent `g4_delta`/independent-fragment-control refactor was compared
against the frozen shared-engine implementation before decoupling. Its new
identity-driven entry passed another 94,371,840 paired component draws across
360 conditions: sampled losses were bitwise identical and RNG call counts
matched, including native H1 fragment calibration.

Native AOT runtime comparisons used `CUDA_DISABLE_PTX_JIT=1` throughout:

- RT06423 carbon minibeam optimized 90°/270° full plan: 6 million primaries
  per implementation on RTX 2080 Ti (`sm_75`, two 3-million batches) and
  12 million per implementation on RTX A6000 (`sm_86`, two 6-million batches).
- All eight release presets: two 20,000-primary batches per implementation.
- Each of the four minibeam configurations through the generic pipeline:
  two 20,000-primary batches per implementation.

Source allocations and seeds, nested non-energy integer counters, discrete
physics and live-ion audits match. The largest relative dose L1 difference
over these matched checks is `3.06199e-7`; all energy, coverage, completion and
zero-overflow checks pass. The maximum physical relative energy residual is
`5.83749e-6`.

Independent-control checks additionally exercise proton primary fluctuations
with native fragment transport enabled, Gaussian carbon primary fluctuations
with native fragment transport enabled, and both projectiles with native
fragment transport disabled and deliberately nonexistent fragment-package
paths. All four 20,000-primary runs pass their runtime audits; native fragment
packages load only when that control is enabled. Missing required fragment
paths and both removed model values are rejected before transport.

The single ordered full-plan timing pair measured warm throughput of
67.81k → 66.73k primaries/s on RTX 2080 Ti and 114.09k → 110.72k on A6000.
These observations include run-order variability and have no reversed-order
confirmation; they are recorded as refactor timing observations, not a
performance improvement or a resolved slowdown estimate.

Eight-preset package/input dependencies and manifest memberships are unchanged.
Evidence and source/binary provenance are under
`tmp/copper_transport_decouple_20261016/`: `fixture.jsonl`, `VALIDATION.json`,
`INDEPENDENCE.json`, per-device `AUDIT.json`, `VERIFY.json` and `COMPLETE.json`.
These checks compare GPU implementations; they do not constitute a new TOPAS
comparison or acceptance of additional source species.
