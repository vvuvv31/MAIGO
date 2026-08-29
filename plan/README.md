# FRED inelastic repair — load one file at a time

Source: repo `plan.md`. Do not load the full plan every turn.
After each step: write status here so later turns skip GPU re-runs.

| Order | File | Topic | Status |
|---|---|---|---|
| 0 | `steps/00-overview.md` | Freeze EM/elastic | **done** — generator-only work |
| 1 | `steps/p0-1-eq13-first-fragment.md` | First fragment not 0.6×P | **done** — `sample_projectile_fragment_Eu`; `test_eq13_first_fragment_not_scaled_down` |
| 2 | `steps/p0-2-eq12-mix.md` | Gauss/exp by species | **done** — `eq12_sample_gaussian`; 35 MeV/u and 90° caps removed |
| 3 | `steps/p0-3-xs-h-o.md` | CSV H/O | **done** — `target_h_fraction` on device; Kox selector gone from GPU sampler |
| 4 | `steps/p0-4-no-runtime-newton.md` | No GPU invert | **done** — production copies `kFredProbH/O`; invert not at GPU startup |
| 5 | `steps/p0-5-ledger.md` | Unassigned ≠ untracked | **done** — `inelastic_numerical_residual_MeV`; fail leftover not double-counted |
| 6 | `steps/p0-6-secondary-steplimit.md` | Step cap no dump | **done** — 30000 steps; StepLimit → escaped, not local dose |
| 7 | `steps/s0-baseline.md` | Repro baseline | **done** — same configs/seeds; 100k smoke |
| 8 | `steps/s1-diagnostics.md` | Split ledger | **done** — Q/neutron/remnant/unassigned fields; capacity overflow=0 on 100k |
| 9 | `steps/s2-xs-tables.md` | Water inelastic table | **done** — CSV loads macro_h, macro_o, target_h_fraction; `pH` interpolated at Σ energy |
| 10 | `steps/s3-channels.md` | Offline channels | **done (stage 1)** — Table 1 + A/Z resample; joint CDF packing deferred by this step file |
| 11 | `steps/s4-eq12.md` | Mixture weights | **done** — hydrogen mix; truncated exponential 0–180° |
| 12 | `steps/s5-eq13.md` | Implicit E_i | **done** — same helper as P0-1 |
| 13 | `steps/s6-q-remnant.md` | Q=0 phase A | **done** — `base_q_MeV=0`; residual not stuffed into untracked |
| 14 | `steps/s7-optical-depth.md` | In-step collision | **later / non-gating** — quote: “Not a gating item for this goal if P0s pass.” |
| 15 | `steps/s8-secondary-range.md` | CSDA local-stop | **done** — per-species A=1 CSDA from ion SP LUT; `remnant_local_stop_from_csda`; StepLimit still no dump; no secondary fragmentation |
| 16 | `steps/s9-layered-validation.md` | 100 MeV/u first | **done** — 100k: shift 0 mm, peak 2.48%, ROI −0.62%, full −0.75%, E-res 1.38e-8. 400 peak 16% not a gate. |

## Evidence (do not re-run unless code after this date)

- Unit: `./build/carbon_tests` → All carbon_tests passed.
- 100 MeV/u 100k: `config/beam_100MeVu_inelastic.yaml` (shift/peak/ROI/full pass).
- 400 MeV/u peak is **not** a must-pass gate (s9).
