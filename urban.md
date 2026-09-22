# Geant4 Urban MSC port for the primary Copper collimator

## Goal

Replace the analytic `copper_fermi_eyges_tail_step` model for the primary C12
transport through the Copper minibeam collimator with a faithful port of the
Geant4 11.3.2 `G4UrbanMscModel`, to reduce the single-spot water-entrance
dose-contrast error against TOPAS.

## Why Urban is the correct model

TOPAS ran with `g4em-standard_opt4` (`run1.txt:51`). In Geant4 11.3.2 option 4
uses `G4GoudsmitSaundersonMscModel` **only for e-/e+**. For ions the MSC process
is `G4hMultipleScattering` (`G4EmStandardPhysics_option4.cc:164`), whose default
model is `G4UrbanMscModel`:

```
G4hMultipleScattering::InitialiseProcess:
  if(nullptr == EmModel(0)) { SetEmModel( new G4UrbanMscModel() ); }
```

`ConstructCharged` is called by opt4 as `ConstructCharged(hmsc, pnuc)` with
the default `isWVI = false`, so no WentzelVI override is installed on the ion
`hmsc` (the `isWVI = true` path only affects muons/light hadrons when
explicitly requested); `ConstructIonEmPhysics` does not either. And the
`G4EmModelActivator` only reconfigures explicitly registered regions, which
TOPAS does not set. So opt4 -> C12 ion MSC = Urban. The executed-installation
proof (2026-09-22, direct oracle against `/software/geant4-11.3.2`) prints
`UrbanMsc, StepLim=Minimal, Rfact=0.2, DispFlag:1, Skin=3, Llim=1 mm` for
GenericIon, matching the TOPAS reference log's EM dump.

This was confirmed empirically from the TOPAS per-step MSC ntuple
(`/mnt/sda/wuwei/minibeam_msc_steps_e250/output/msc_steps.phsp`, Copper steps).
The measured `Local Scatter Angle` divided by the Urban `ComputeTheta0`
prediction is constant across a 10x step-length range at ~200 MeV/u:

| h (mm) | median(theta)/theta_urban |
|---:|---:|
| 0.0074 | 1.2425 |
| 0.0147 | 1.2295 |
| 0.0288 | 1.2362 |
| 0.0500 | 1.2316 |

The constant ~1.24 sits between the Rayleigh median
`sqrt(2*ln2) = 1.1776` and the Rayleigh mean `sqrt(pi/2) = 1.2531` for a
per-component Gaussian of width theta0 (> 2026-09-22 correction: the original
text mislabeled `sqrt(pi/2)` as the median; the median is `sqrt(2ln2)`. The
measured 1.23-1.24 is consistent with Urban's `sqrt(h)*(c1 + c2*ln(h/X0))`
step dependence at either statistic, but the median/mean distinction matters
for moment-based fits and is now pinned: the host regression asserts the
median ratio band, see `test_angular_small_quantity`).

## Implementation

`src/detail/sycl_device_math.inc` adds device functions:

- `copper_urban_cross_section_per_atom_cm2` — faithful
  `ComputeCrossSectionPerAtom` (electron-equivalent energy, eps branches,
  charge^2 * Z^2 / (beta2*bg2), Copper Z=29 high-energy `sig0` and low-energy
  `cpositron` table, low-energy correction).
- `copper_urban_transport_mfp_mm` — `lambda0 = 1/(n*sigma)`.
- `copper_urban_coefficients` — `facz`, `coeffth1/2`, `coeffc1..4` for Z=29.
- `copper_urban_theta0` — `ComputeTheta0` (Highland + coeffth correction).
- `copper_urban_msc_step` — `SampleCosineTheta` + `SimpleScattering` +
  `SampleDisplacementNew`, returning the existing `CorrelatedScatteringStep`
  (direction + displacement). Constants `taubig=8`, `tausmall=1e-16`.

`src/transport_sycl.cpp` adds the model selector
`minibeam_copper_mcs_model == "urban"` and routes both `urban` and
`fermi_eyges_tail` through `minibeam_copper_correlated_scattering`, so the
Urban result is not overwritten by the legacy Highland branch.
`minibeam_copper_mcs_scale` scales theta0 (the caller's amplitude).

`src/config.cpp` accepts `urban` for `minibeam_copper_mcs_model`.

Configs: `config/beam_minibeam_single_center_em_only_e250_urban_2m.yaml`
(Copper step 0.25 mm) and `..._urban_step003.yaml` (Copper step 0.03 mm).

## Slab validation

Python port of the same sampling against the TOPAS 250 MeV/u Cu 1/10/15/20 mm
matrix gave thickness-independent q50/90/99/99.9 G/T ratios of ~1.03-1.04
(core), whereas the FE/tail model degraded to 0.87 at 20 mm. The Urban model
captures the correct thickness evolution.

## Collimator result

Single central spot, 250 MeV/u, EM-only, 10M histories, vs TOPAS 10M. Entry
contrast error (GPU/TOPAS - 1, pp):

| depth | FE/tail | Urban 0.25 mm | Urban 0.03 mm |
|---:|---:|---:|---:|
| 0.88 mm | +3.27 | **+2.31** | +3.15 |
| 4.88 mm | +1.95 | **+1.31** | +2.15 |
| 9.88 mm | +1.69 | **+0.40** | +1.73 |
| 19.88 mm | +0.39 | -0.55 | +2.04 |
| 119.88 mm | -1.36 | -1.79 | -0.82 |

- The faithful Urban port improves the entrance contrast by ~1 pp at 1 mm and
  ~1.3 pp at 10 mm over the FE/tail model.
- Reducing the Copper step to TOPAS's ~0.03 mm makes it **worse**, because the
  Urban total variance scales as `(c1 + c2*ln(h/X0))^2`: 0.491 at h=0.03 mm vs
  0.635 at h=0.25 mm. Step length is therefore not the residual.
- Total deposited dose is identical across all three models (lateral
  redistribution only); the peak is ~1.6% high and the valley ~0.7% low.

## Status

The residual +2.3 pp at the very entrance is **not** an MSC model mismatch and
**not** a step-length effect. Remaining candidates: dose-scoring convention
(GPU voxel `dose_to_medium=mhd` vs TOPAS `DoseToMedium`) and the source/optics.

> 2026-09-21 correction (this paragraph's exclusion was premature, do not use
> it as a premise): the "not MSC / not step" verdict is INVALID until (a) the
> 1 mm slab lateral-x ~100x discrepancy is resolved with a slab config that
> reproduces the TOPAS box geometry (it was dismissed as "geometry-confounded"
> without that control), (b) the production safety wiring defect is fixed
> (`transport_sycl.cpp` passed `safety_mm=0.0F`, forcing every urban_v2
> displacement to cancel; fixed this round to a real endpoint isotropic
> safety), and (c) a signed-distance finite-slit scan with shared entry
> records exists (the offset/translation attempts produced no valid scan).
> The dose-scoring and source/optics candidates remain hypotheses, not
> established exclusions of MSC/step effects.

Production default remains `fermi_eyges_tail`; `urban` is an opt-in, validated
candidate. Enable with:

```yaml
minibeam_copper_mcs_model: urban
```

## Water urban_v2 stage-D first look (2026-09-22)

New paired configs isolate the water MCS model with copper fixed to
`urban_v2` and the same seed (`202609250`):

- `config/beam_minibeam_single_center_em_only_e250_urban_v2_10m.yaml`:
  copper urban_v2 + water `fermi_eyges_tail` (pre-existing 10M output in
  `out/beam_minibeam_single_center_em_only_e250_urban_v2_10m/`).
- `config/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m.yaml`
  (new): copper urban_v2 + water `urban_v2` via
  `data/urban/c12_water75ev_urban_g4_11_3_2.csv`, `urban_max_step_mm: 0.05`.
  10M output in
  `out/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m/`.

Single central spot, 250 MeV/u, EM-only, RTX 2080 Ti, FP32 dose, stopping
scale 1.0, copper MCS scale 1.0 (no 0.785 double correction). Run accepted:
energy residual 1.6e-5, queue overflow 0, 3.11B urban segments, 140 s at
71k histories/s. Reference is TOPAS 10M
(`/mnt/sda/wuwei/minibeam_single_center_em_only_e250_10m/topas_7374/dose.bin`).

Entry-contrast error (pp) vs TOPAS, A = Cu-v2+W-FE, B = Cu-v2+W-Urbanv2:

| depth | A | B |
|---:|---:|---:|
| 0.88 mm | +0.55 | +0.60 |
| 4.88 mm | -1.02 | -0.92 |
| 9.88 mm | -2.30 | -2.11 |
| 19.88 mm | -4.17 | -4.76 |
| 39.88 mm | -4.00 | -4.03 |
| 79.88 mm | -6.91 | -5.65 |
| 119.88 mm | -5.67 | -2.26 |

Total dose GPU/TOPAS is 0.99904 (A) vs 0.99902 (B): pure lateral
redistribution. Water-Urban-v2 matches FE-tail shallow and improves deep
(119.88 mm valley 1.080 -> 1.055, peak 1.019 -> 1.031); the mid-depth valley
excess (~20-80 mm) is common to both water models, so it is not a water
angular-PDF effect. Single seed; EM-only secondaries are negligible, so the
legacy secondary-C12 path is not a confounder here. Next: multi-seed repeat
of the deep improvement, then fragment-water isolation for full physics
(valley/halo residual must be split by species + elastic/inelastic, not tuned
via MCS).

Multi-seed repeat (2026-09-22, seed 202609251, same config, `--random-seed`
override; dose in `out/..._cuwater_10m_s2/`): accepted, residual 1.61e-5,
overflow 0. Totals 0.99916. Contrast errors (pp): 0.88 mm +1.56, 4.88 mm
-0.95, 9.88 mm -3.62, 19.88 mm -5.22, 39.88 mm -4.17, 79.88 mm -5.78,
119.88 mm -2.50. Seed-to-seed variation is ~1pp shallow (0.88 mm: +0.60 vs
+1.56; 9.88 mm: -2.11 vs -3.62), so the entrance pp differences are noise;
the deep improvement over A (-5.67 -> -2.3 +/- 0.2) reproduces on both seeds
and is robust. Mid-depth valley excess reproduces on both seeds for both
water models.

Step-convergence check (2026-09-22,
`config/..._cuwater_10m_s0025.yaml`: water Urban max step 0.025 mm, same
seed/histories; dose in `out/..._s0025/`): accepted, residual 1.61e-5,
5.74B segments, 177 s. Totals 0.99901, identical (lateral only). Shallow
(<=10 mm) is invariant (<0.1 pp), but deep is not: 79.88 mm -5.65 -> -4.03,
119.88 mm -2.26 -> +7.67 (peak +1.8%, valley -7.6%). Cause: fixed max-step
subdivision with independent per-segment Urban sampling does not preserve
total variance -- Urban's `(c1+c2*ln(h/X0))^2` scaling makes the sum
N-dependent, and the port lacks Geant4's consecutive-step limitation logic
that restores approximate invariance. Same pathology as the copper 0.25 ->
0.03 mm result in failed.md. Consequences: (1) keep 0.05 mm matched to the
TOPAS water MaxStepSize -- it is load-bearing, not incidental; (2) the deep
match at 0.05 is therefore partly step-matched, not yet principled -- true
invariance needs the consecutive-step machinery ported; (3) do not retune
physics against the 119.88 mm point until (2) is done. Unit pin:
`test_water_urban_segment_scaling` in
`benchmark/carbonminibeam/test_urban_v2_helpers.cpp` reproduces the
N-dependence deterministically on host (100k paired proposals, real
Water_75eV table): E[th2] 1x0.05mm = 1.45e-8 vs 2x0.025mm = 4.13e-9, ratio
0.284 (band [0.20, 0.38]). Far below Highland-like ~0.94 because Urban
theta0 carries the (coeffth1 + coeffth2*ln(t/X0)) correction, steep at
t/X0 ~ 1e-4.

Geant4 two-step-limit reference (2026-09-22, Slurm 8544, paired slab:
e250_l10 rerun with Water MaxStepSize 0.025 mm, same seed 26092131;
`benchmark/carbonminibeam/c12_water_slab/topas_c12_water_slab_e250_l10_s0025.txt`,
output `/mnt/sda/wuwei/c12_water_slab_step0025/`): exit angle variance ratio
0.929 (A2 3.3721/3.6306), q68/q99/q999 ratios 0.966/0.960/0.940. So Geant4
itself scatters less at the finer limit -- same direction as the port, which
rules out "missing consecutive-step renormalization" as the sole explanation:
part of the s0025 flip is faithful step-limit dependence, and 0.05 mm must
stay matched to the TOPAS setting regardless. Magnitude caveat: the slab
0.929 is a whole-transport outcome (step-count and energy-loss evolution
included), not directly comparable to the unit-test per-path 0.284; inferring
a per-sample Geant4 ratio by halving (0.46) assumes uniform 200-vs-400
splitting that fMinimal/energy loss invalidate. The remaining gap (port
samples steeper than Geant4's theta0 scaling suggests) is an open modeling
question for the Urban sampler, explicitly NOT a tuning target.

## Absorbing-mode null-table hang (found 2026-09-22, fixed)

The water-Urban loss-range upload lived inside `if (minibeam_copper_transport)`
(= mode `copper_em`), so under `absorbing_geometry` (water-entry replays) the
device table stayed null. Every `urban_v2_propose_and_sample` then returned a
zero geom path, and the subdivision loop (`traversed_mm +=
final_geom_path_mm`, no progress guard) spun forever: 2000-history mini and
1.6M replay hung at 100% GPU while the FE mate finished in ~5 s. Found by
bisecting FE-vs-Urban on identical entrance/seeds, confirmed via nsys (zero
kernels launched) and host unit probes (proposals healthy once loaded).

Fix in `src/transport_sycl.cpp`: the water-Urban upload moved out of the
copper-gated block (own `enable_minibeam` gate; shared `upload_float_pair`
lambda hoisted to function scope), plus fail-fast throw if `urban_v2` is
selected with < 2 table nodes. A 1M-iteration trip cap with slot-135 counter
(`minibeam_water_urban_subdiv_cap_slot`) guards the subdivision loop as
defense-in-depth. After the fix the 2000-history mini finishes in 4.7 s
(trips=0) and the full 1.6M replay in 26.6 s, accepted, residual 1.35e-10.
Reproducible fast smoke: `benchmark/carbonminibeam/water_entrance_primary_c12_smoke2k.csv`
(first 2000 rows of the real entrance file, E >= 100 MeV by construction)
with `config/beam_minibeam_water_replay_e250_em12800k_urban_v2_mini.yaml`
(~5 s; guards init/upload regressions only, not physics).

## Paired replay intervals: Urban vs FE-tail (2026-09-22)

Same 12.8M-field entrance (1,622,795 C12), same seed, EM-only; only the water
model differs.
`config/beam_minibeam_water_replay_e250_em12800k_urban_v2.yaml` (+ planes in
`/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/gpu_urban_v2/`).
Per-interval pairing (histories crossing both planes); N matches to <0.01%,
so no survival-selection confound. Replay dose totals agree to 0.99998.

| interval | angVar U/F | disVar U/F | q68 U/F | q99 U/F | q999 U/F |
|---|---|---|---|---|---|
| 40->60 | 1.257 | 3.76 | 1.000 | 1.027 | 1.077 |
| 60->80 | 1.117 | 2.04 | 0.993 | 1.038 | 1.079 |
| 80->100 | 1.072 | 1.29 | 0.998 | 1.038 | 1.110 |
| 100->120 | 1.058 | 1.04 | 1.010 | 1.051 | 1.194 |

(angle increments mrad^2, displacement mm^2 vs straight line, quantiles mrad)

Urban scatters more than FE everywhere with the SAME core (q68 ~1.00): the
excess is tail (q99/q999 higher) plus larger correlated displacement, strongest
shallow and converging deep. Against the TOPAS-paired FE ratios (angle var
~0.99-1.01, q999 ~0.90-0.95 low), Urban overshoots shallow variance while
repairing the far tail -- i.e. Urban and FE bracket TOPAS from opposite sides.
Together with the full-chain dose (Urban deep improvement despite shallow
variance overshoot), the dose match involves cross-depth compensation, not a
uniformly better kernel. Do not tune Urban against dose; the next discriminant
is the same interval table against the TOPAS paired planes (job 6529 data
already on disk).

Absolute TOPAS anchor (2026-09-22, job 6529 parent-0 planes at
40/60/80/100/120 mm, same interval metrics; `/tmp/topas_replay_intervals.py`):

| interval | TOPAS angVar | FE/T | Urb/T | TOPAS disVar | FE/T | Urb/T | TOPAS q999 | FE/T | Urb/T |
|---|---|---|---|---|---|---|---|---|---|
| 40->60 | 37.89 | 0.987 | 1.241 | 0.00404 | 1.009 | 3.80 | 32.29 | 0.935 | 1.007 |
| 60->80 | 47.70 | 1.003 | 1.120 | 0.00505 | 1.011 | 2.07 | 33.70 | 0.947 | 1.022 |
| 80->100 | 69.35 | 0.991 | 1.062 | 0.00711 | 1.007 | 1.30 | 36.62 | 0.917 | 1.018 |
| 100->120 | 169.51 | 0.975 | 1.032 | 0.01378 | 0.991 | 1.03 | 49.76 | 0.862 | 1.029 |

(q68 matches both models to ~1.5%; q99: FE 0.97-0.99, Urb 1.01-1.02.)
FE matches variance/core/q99 but misses the far tail (-6..-14%); Urban matches
core/q99/far tail (+1..+3%) but overshoots variance and displacement shallow
(+24%/+276% at 40->60, converging deep) -- excess weight beyond q999. Survival
N matches to 0.02%, so no selection confound. Verdict stands: bracket from
opposite sides; no dose tuning of either model.

## Secondary-C12 Urban-v2 (2026-09-22)

`minibeam_water_secondary_c12_mcs_model` now accepts `urban_v2`
(`src/config.cpp`, `transport_config.hpp`; default stays `legacy_highland`;
requires the shared Water_75eV loss-range file). The secondary kernel branch
(`src/transport_sycl.cpp`, after the FE-tail branch) reuses
`water_urban_v2_propose_and_sample` with the same 0.05 mm subdivision,
endpoint-projection plane records, subdivision cap (shared slot 135) and a
new step counter (slot 136, counter size 136->137). Differences vs primary,
all documented in code: fresh per-step track state with `at_boundary=true`
every step (fragments carry no persistent MSC state; tlimit >> step except
near range end), RNG dims 110+ (FE 100+, primary Urban 70+), C12-only gate,
no production default change. Full-chain traffic is ~zero (secondary C12 is
~4/57k of water-entry fragments), so validation uses the diagnostic
secondary-C12 replay, not full chain.

Paired secondary replay (same 1.6M entrance, seed 202609250, EM-only;
`config/beam_minibeam_water_secondary_c12_replay_e250_1622795_urban_v2.yaml`,
planes in `.../same_source_e250_20260919/urban_v2/`): 1.99B branch entries,
accepted, residual ~2e-9, trips 0, 27.9 s. Dose totals U/F 0.999981, voxel L1
2.3%. Interval moments show the same signature as the primary queue (core
q68 ~1.00; angVar +14%/+14%/+9%/+8%, disVar +134%/+97%/+32%/+5%,
q999 +9%/+11%/+13%/+22% shallow-to-deep; N matched to 0.02%): the unified
kernel behaves consistently across queues. Full-chain paired A/B
(legacy-vs-urban secondary, 10M) differs at 2.7e-8 voxel L1 -- nil, as
expected from ~zero traffic, proving the branch is inert unless exercised.

Build compatibility (2026-09-22): the minibeam-ON preset builds clean with
all of the above. The minibeam-OFF preset (`oneapi-nvidia-minibeam-off`) does
NOT build, but the breakage is pre-existing owner WIP unrelated to Urban
(`minibeam_isotropic_safety_at_point`, delta-response ledger, generic-FE
region, urban segment counter); the single OFF error attributable to these
edits (startup print using a minibeam-gated variable) was fixed by guarding
it, verified by error-list diff. OFF-preset repair belongs to the WIP owner.

## Highland→Urban refactor map (2026-09-22)

Consolidated index of the `Highland theta sampler -> MscTransportModel`
upgrade (all research paths opt-in; production defaults unchanged:
copper `fermi_eyges_tail`, water primary `fermi_eyges_tail`, water secondary
`legacy_highland`).

| Piece | Location |
|---|---|
| `MscModel {Highland, Urban}`, `MscStepResult{direction, displacement_mm}` | `src/detail/sycl_device_math.inc:178` |
| Highland primary (Box-Muller, dims 3,4,5,6), zero displacement | `.inc:185` called at `src/transport_sycl.cpp:7741` |
| Highland secondary (Rayleigh+phi, dims 0,1), zero displacement | `.inc:215` called at `transport_sycl.cpp:12259` |
| Urban core (`propose_and_sample`, fMinimal limiter, Alg96 displacement) | `.inc:1964` + `copper_urban_v2_limit_step`, `copper_urban_v2_sample_full` |
| Water loss-range table (mode-independent upload + fail-fast) | `transport_sycl.cpp:2931` |
| Primary Urban branch (0.05 mm subdivision, endpoint planes, 1M cap slot 135) | `transport_sycl.cpp:7465` |
| Secondary Urban branch (same kernel, dims 110+, step slot 136, planes) | `transport_sycl.cpp:11980` |
| Config selection + validation | `transport_config.hpp`, `src/config.cpp:1574,1582,1621` |
| Unit pins (scaling 0.284, subdivision robustness, low-E proposals) | `benchmark/carbonminibeam/test_urban_v2_helpers.cpp` |

Lateral displacement placement: every MCS branch writes a
`displacement_mm` alongside the scattered direction; callers advance
`position += seg_dir * step + displacement` (primary: `segment_offset`
minus straight-line; secondary: `segment_x/y/z` accumulation). Highland
returns exactly zero (endpoint scattering documented, not approximated).
RNG A/B: fixed per-model dimension blocks (primary Highland 3-6 + tail 7,
primary Urban 58/59/70+, secondary FE 100+, secondary Urban 110+, secondary
Highland 0/1), deterministic counter streams, paired seeds across all A/Bs.

Staged scope as delivered: (1) Highland + explicit zero-displacement API
(bit-identical); (2) Urban-compatible transport with correlated displacement
for primary C12 (copper + water) and secondary C12, each validated by
paired slab/interval/dose benchmarks against TOPAS in the sections above.

## Correctness fixes B1-B6 + direct-Geant4 oracle (2026-09-22)

Prompt task: fix the Urban port against the EXECUTED Geant4 11.3.2 (not the
tag text alone) and validate carbon minibeam EM-only/full-physics plus
broadbeam before any default migration. No default changed; production
defaults all bit-identical (Highland/FE/legacy paths untouched).

Direct oracle (`evidence/urban_9618eb0_20260922/g4_oracle/g4_urban_oracle.cc`,
links `/software/geant4-11.3.2`, version `11-03-patch-02`): model dump,
per-atom C12 cross-section grid (installed lib == v11.3.2 tag to 6e-5;
earlier "2.09x patch difference" was a probe bug — GenericIon mass used
instead of real C12, recorded in `failed.md`), thin-slab exit/step
aggregates with a TOPAS-like MaxStepSize limiter.

Fixes (all in `src/detail/sycl_device_math.inc` /
`src/transport_sycl.cpp` / `src/config.cpp` / `include/carbon/rng.hpp`;
regression: `benchmark/carbonminibeam/test_urban_localize.cpp`, all PASS):

- B1: `rng::urban_unit_strict` (Urban-only; global `uniform01` untouched):
  the `bits=2^24-1 -> exactly 1.0f` endpoint (reached preimage on file)
  spuriously took the isotropic branch at q>=1 with rate 2^-24/draw.
- B2: mixture gate + both Bernoulli trials in double (q rounds to 1 in
  FP32; ~4e-4 drift at Bragg-end E through the `1-d` denominator);
  `|cth|>=1` returns unscattered on the stable `omcth` measure (a float
  clamp would kill 89% of small-angle updates); postSafety caps safety at
  dispR per `G4SafetyHelper::ComputeSafety` (far-field displacements scale
  by 0.99, R2 test updated).
- B3: tlimitmin frozen at the StartTracking value 1e-7 mm on ALL limiter
  paths (was recomputed per step at ~3e-3 mm, 3e4x; Stepmin deleted —
  reference fMinimal never calls ComputeStepmin/ComputeTlimitmin).
- B4: branch-consistent `(t, g, delta)` — `true_to_geom` and `finalize`
  return the delta of their taken branch (double); the par1<0-only
  re-inversion and the dead `t_final > t_ext` clamp deleted; the
  `min()-commutes` ordering claim removed.
- B5: invalid/zero-progress/capped proposals raise slot-137 fatal and the
  host throws (hard gate); copper invalid included.
- B6: secondary `at_boundary=false` (newborn-track semantics) + segment-
  START Epre (was midpoint, double-predicted); RNG `steps*1024+segment`
  proven collision-free for seg<1024 (config guard:
  `maximum_step_mm <= 1024*max_segment_mm`) and <4.19M steps/history
  (partners at >=12.6M); naive `(step<<32)|seg` rejected (cross-dim
  collision at steps 0 vs 3, see `failed.md`).

Single-step validation (0.05 mm water, 250 MeV/u, 200k, direct libG4):
q50-q999 ratios 1.0009/1.0035/1.0006/1.0001/1.0066/0.9980, E[th^2] 0.983 —
IDENTICAL before and after B1-B6 (fixes inert at this point by design).
Full branch map: `evidence/urban_9618eb0_20260922/branch_map.md`.

Status of older inferences in this file: the 0.284 band is a legacy
characterization artifact (E2: the `2*(1-dz)` metric reads 2.44x low;
stable moment is `2*one_minus_cth`); "missing consecutive-step
renormalization" is NOT an established Geant4 mechanism (no such logic in
the fetched limiter/sampler; slab 0.929 shows reference step dependence is
real) — kept as open modeling question, not a premise. The interval/dose
gaps (angVar +24% shallow etc.) are therefore NOT sampler-numerics issues;
transport-level re-validation with the fixed binary is next (Phase D).

## B-candidate validation results (2026-09-22, fixed binary)

Binary `build/oneapi-nvidia-minibeam/carbon_mc`
(`0f0f7060343a9cc5b04ad7769b4488e8270d03546b81aa6b02bb2ae3f06cab5f`
post-B2c; host tests ALL PASS). Candidate outputs (history-safe):
`/mnt/sda/wuwei/urban_B_validation/` (planes/dose/logs) + `out/*_B_*/`
(stem-distinct; historical `out/` untouched except one incident, see
`failed.md`). All runs: subdiv cap trips 0, fatal proposals 0, queue
overflow 0, FP32 dose.

1. Water replay 1.6M EM-only, same entrance+seed (`..._urban_v2_B.yaml`):
   per-interval pairing vs TOPAS job 6529 (N matched):

   | interval | angVar B/T | disVar B/T | q999 B/T |
   |---|---|---|---|
   | 40->60 | 1.014 (was 1.241) | 1.011 (was 3.80) | 1.005 |
   | 60->80 | 1.026 (was 1.120) | 1.019 (was 2.07) | 1.020 |
   | 80->100 | 1.021 (was 1.062) | 1.017 (was 1.30) | 1.016 |
   | 100->120 | 1.032 (was 1.032) | 1.028 (was 1.03) | 1.029 |

   Far-tail triple check (40->60, |dth|>0.1/1.34M): TOPAS 24, B 35,
   pristine 51 (>0.3: 3/2/19; max 0.60/0.39/0.97). B matches the reference
   tail; pristine carried ~16-25 spurious radian kicks. Root cause: the
   `uniform01` 2^-24 upper endpoint (E1 preimage on file) took the
   isotropic branch at float-q == 1.0 (reference: q slightly >1, mixture
   always). My interim "debunk" (u0=1<q-double) was wrong — the code
   compares against float-q == 1.0. B1 strict uniform removes it; the 51
   extra plane-0 histories in B are the previously ejected ones.
   Deep intervals converge (kicked histories eject shallow; survivors match).
2. Full-chain EM-only 10M x2 seeds (`..._cuwater_10m_B_s1/s2.yaml`):
   totals 0.99904/0.99916; B_s2 vs pristine OLD_s2 (same seed) identical to
   <=0.07pp at all depths (bulk dose insensitive to dozens of kicks, as
   expected); seed spread ~1pp shallow. Deep improvement over FE retained
   (119.88 mm -2.2/-2.5); mid-depth valley excess (-4..-6pp at 20-80mm)
   UNCHANGED — documented non-MCS residual, not an MCS tuning target.
3. Secondary-C12 replay: B/OLD angVar 0.893/0.907/0.958/0.994 shallow->deep
   (same endpoint-fix shape); absolute moments track primary B closely.
   Full-chain secondary traffic still ~nil (2.7e-8 voxel L1 A/B earlier).
4. Host multi-step composition vs direct-G4 slabs (0.05 maxstep):
   1mm: E[th^2] 1.052, q50/q99/q999 1.012/1.002/1.052, varX 0.994,
   exit-E +0.04%; 10mm: E[th^2] 0.988, q 0.997/0.976/0.867, varX 0.970,
   exit-E +0.6% (CSDA-linear vs G4 straggling; noted, not tuned).
   G4's own step dependence (0.025/0.05): E[th^2] 0.960 (1mm) / 0.924
   (10mm) — same direction as the port; 0.05 stays matched to TOPAS.
5. CT RT07575 production, B vs pristine binary (same config+seed): dose
   sums identical (5490.062), max abs diff 1.1e-8 (FP32 atomic noise) —
   Urban code inactive in CT, non-interference proven.
6. Full-physics 10M production (scale1_B): accepted, fatal 0, overflow 0;
   dose sums identical to stored reference (207.4604, ratio 1.0), max abs
   3.7e-9 — full-physics production paths unperturbed.
7. Perf: B vs pristine full-replay wall 51s vs 38s / 122s vs 38s across runs
   is CLOCK variance (SM 300 MHz P8 idle observed; no root to lock clocks),
   not code: unconditional-double was measured 3-4x and REMOVED after the
   reachable-domain audit justified float; hot-path delta vs pristine is a
   few ALU ops. Controlled perf measurement (warmup + repeats + locked
   clocks) is PENDING — no perf claim is made.

Promotion verdict: SCOPED_URBAN_DEFAULT NOT_PROMOTED, GLOBAL NOT_PROMOTED.
Gates status: NUMERICAL_ORACLE PASS (model/xsec/single-step/slab-composition
vs executed G4); MSC_TRANSPORT PASS (branch map + determinism + failure
propagation + RNG proof); CARBON_EM_ONLY PASS (intervals 1-3%, dose bulk +
deep improvement, mid-depth residual documented non-MCS);
CARBON_FULL_PHYSICS BLOCKED (species-split/fragment-slab/held-out campaigns
not run; full-physics smoke only); BROADBEAM_REGRESSION PARTIAL (CT
non-interference proven; water/broadbeam absolute references + Gamma gates
not run — no frozen user baseline found in evidence); GLOBAL coverage
(Cu/Water C12 only; CT/proton/fragment routing unverified) BLOCKED.
Defaults untouched. See `evidence/urban_9618eb0_20260922/` for all
artifacts, manifests, and scripts.

## C-round: C1–C8 close-out (2026-09-23, anchor 0f2c0ca)

Evidence: `evidence/urban_after_0f2c0ca_20260922/` (new dir; old dir
untouched). Binary `build/oneapi-nvidia-minibeam/carbon_mc`
(`6d79ecf7ad83192eb1fcb43b767449557f50407842fdb17e7c8c137f0b9e3508`
final; earlier C-stage SHAs in `binsha_C2/C4.txt`). B1–B6 preserved
(strict uniform, frozen tlimitmin, stable one_minus_cth, fatal
propagation); C-round changed: conversion math, RNG domain, finalize
inversion, secondary state, MSC range. Legacy streams bit-identical
(helpers ALL PASS).

### Errata to the B-round record (§1.2清算)

- B2 "double chain": the unconditional-double sampler chain was measured
  3-4x wall time and REMOVED; final state is float-only with series/expm1/
  log1p stable forms (C1 extends this to every conversion branch). Any
  "B2 keeps double" wording above is superseded.
- B6 "collision-free addressing": WRONG in the dangerous direction. Real
  collision found at (outer=0, seg=0, dim 70/71, primary key): all-ion
  elastic `uniform01(steps=0)` shares raw Philox bits with old-Urban
  `urban_unit_strict(0*1024+0)` whenever elastic runs (full physics, never
  EM-only — hence never observed). Fixed by the MSC domain tag (C2), not
  by the old 12.6M-partner argument (retired).
- Binary SHAs: `binsha_FINAL.txt` (post-B2c) vs validation binary
  (`0f0f7060` post-B2c? and `4d46c910` in §B-validation) are DIFFERENT
  STAGE binaries (intermediate vs final-B), not a record error; C-round
  binaries are hashed per stage in the new manifest.
- `dose_compare_B.py` "pp": misnomer; it computed 100*(ratio-1), i.e.
  RELATIVE percent. Renamed in `dose_gate.py`; old numbers unchanged.
- B_s1/B_s2 dose "2 seeds": FALSE — same seed 202609250 (output dirs only).
  C-round dose uses 3 independent seeds (202609251/2/3).
- B verdict "CARBON_EM_ONLY PASS" used study criteria (bulk + deep);
  the C-round froze prompt-§7.3 gates BEFORE measuring and the valley/PVDR
  residual fails them. The C verdict below SUPERSEDES for any promotion
  question; the B numbers stand as measurements.

### C1 conversion (true<->geom)

Executed triple oracle `g4_conv_oracle.cc` (real G4UrbanMscModel:
Limit→Geom→TrueStepLength, 8 energies × 19 steps, Water_75eV):
t≤1nm → g==t exactly; round-trip exact; t_lim caps at CSDA-R (→C4).
Fixes: expm1 z + series/expm1 delta (killed the false "≥100 ulp" claim);
log1p/expm1 par>=0 inversion (was t=0 on boundary slivers → 49 GPU guard
trips → 0); keep-delta repairs; 8-ulp representability switch; finite
guards. Bounds (independent long-double + G4): g ≤2.3e-7, small-t delta
exact, range-direct inside 4-ulp envelope, fallback rmax ≤8.4%, par-branch
z 2e-7, truncated inversion 3e-7, sub-nm exact. No g>t, no pseudo-zero.
GPU helper probe: host/GPU spread ≤1 ulp (g), bitwise RNG/fix/trip.

### C2 RNG + safety

MSC domain tag (counter word-3 bit 30; disjoint sets + Philox bijectivity
= structural proof; legacy bit-identical). Guards (outer<2^22, seg<1024)
before any draw, host-injected. Full accept contract (reasons 1-8) +
FP32-progress rule + sub-ulp completion + first-failure record (ticket
election; PTX has no 64-bit CAS) + run-quality JSON + nonzero exit + no
accepted dose. GPU fault injection (degenerate table): seg==1024 guard
fires, 1 fault/history then stopped, exit 1, no dose. Normal smoke:
all-zero. E2 ensemble (8×200k) confirms the domain change is
distribution-neutral (old center inside new CI).

### C3 q-audit

Weight export + long-double recomputation: q/prob/x_mean1 ~1e-7,
d ~6e-6, tail weight <1e-7; gate margin ≥7.3e-4; straddles exact by search
(mapping merges at b≥2^23 documented, 2^-24 effect); q≥1→mixture pinned;
fuzzy-zone error model replaces "identical". Domain: q∈[0.933,1.001].

### C4 transport contract

Secondary tlimit lifecycle fixed (was pinned 1e10 → never limited):
per-track persistent state + born flag + chunk save/restore
(`SecondaryResumeState.sec_urban_tlimit_mm/born`). MSC currentRange
mirror (executed, cuts-independent, exact-linear): water E/7.2, Cu E/64.5
— neither CSDA nor restricted. Old restricted range clamped low-E steps
and killed scatter at end-of-range. Loss side keeps restricted tables
(current_range_mm has no readers — verified). Geometry readout (§5.1) +
remaining approximations A1–A6 in
`evidence/urban_after_0f2c0ca_20260922/transport_contract.md`.
Determinism + save/restore + newborn tests ALL PASS.

### C5 hygiene

E7 relabeled diagnostic + hang-guard (slab reference BLOCKED); test
outputs to $CARBON_TEST_OUT_DIR/temp (frozen evidence never rewritten);
host-vs-GPU probed (above); OFF build still broken for PRE-EXISTING
missing MINIBEAM-off definitions (not Urban code — verified untouched).

### C6 dose/phase-space (frozen acceptance.yaml)

- Phase space vs real TOPAS .phsp (matched histories, exact space angle,
  batch SEs, replay_gate_C.json): angVar 0.996–1.025, disVar 1.003–1.016,
  q999 1.008–1.027, survival identical. PASS. (Two loader bugs caught and
  fixed during the work: column shift, beam-axis mixup.)
- Row fluence matched ~1% → residual is NOT water-MCS (C6.2 discriminator
  with evidence). Stable B→C (+6.0%→+6.3%).
- Dose (3 seeds) vs TOPAS dose.bin: totals +0.4–0.8%, peak ±0.7%, valley
  +4–6% FAIL, PVDR −4–6% FAIL, FWHM PASS, R80 −0.500mm marginal FAIL.
- Cu slab (10mm, 60k G4): ang 0.956, q999 0.90, dE +3.5% (loss-model side).
  Cu tail is weak-side: wrong sign for the valley excess. Electrons
  wrong-sign too. Scoring/sources match. → RESIDUAL_UNRESOLVED (C6_verdict.md).
- Peak −0.26mm (C4 range fix improved B −0.50mm), height matched.

### C7/C8 assessment

- Active-Urban full line (Cu+water+secondary C12 → urban_v2, 10M):
  2.3G urban segments, all-zero quality counters, routing logged
  (config template in evidence; nuclear scope effectively off per ledger
  — true full-physics validation needs nuclear-on scope + held-out).
  Dose acceptance BLOCKED (C6 must pass first).
- Broadbeam: CT non-interference RE-PROVEN with C binary (sums identical
  5490.062012, maxabs 1.2e-8). Reference accuracy BLOCKED (no frozen
  broadbeam reference). Perf NOT_RUN.

Promotion verdict: SCOPED_URBAN_DEFAULT NOT_PROMOTED,
GLOBAL_URBAN_DEFAULT NOT_PROMOTED. Defaults untouched. No commit/push
(this round unauthorized).
