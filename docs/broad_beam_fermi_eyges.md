# Broad-beam C12 Fermi--Eyges integration

## Runtime selection

Broad-beam and minibeam C12 transport now share the YAML selector:

```yaml
enable_multiple_scattering: true
multiple_scattering_model: fermi_eyges  # or highland
fermi_eyges_species: all_charged        # staged species scope
fermi_eyges_max_segment_mm: 0.1
```

`highland` preserves the historical per-step Gaussian angular kick.
`fermi_eyges` uses the common correlated angle/displacement core plus the
path-length Poisson tail. The staged species selector accepts:

- `c12`;
- `c12_he4`;
- `c12_he4_pdt`;
- `all_charged` (the alias `all` is also accepted).

Ions outside the selected scope continue to use Highland. The default remains
`highland`, so old YAML files do not change silently. The older
`minibeam_water_*_mcs_model` keys remain accepted for reproducibility.
The short YAML value `fe` is accepted as an alias for `fermi_eyges`.
When `multiple_scattering_model` is explicitly present it is authoritative:
selecting `highland` also disables the older minibeam-only FE paths, making the
common selector a genuine one-key rollback.

The common selector is compiled and runnable with `CARBON_ENABLE_MINIBEAM` both
ON and OFF. C12 retains the model constrained by water minibeam phase space.
Protons, deuterons, tritons and He-4 now use independent pure-water TOPAS fits
at 50/150/300 MeV/u with linear interpolation. He-3 and heavier fragments still
fall back to the C12 constants. The fitted interval is 50--300 MeV/u; endpoint
values are held outside it. Selecting scopes beyond `c12` remains experimental,
especially for the uncalibrated ions and Schneider materials.

## First local checks (2026-09-19)

All runs used an RTX 2080 Ti and FP32 dose scoring.

The paired 200 MeV/u, 10k-history homogeneous-water pure-EM smoke used the same
seed and Unified EM. Highland and FE both completed without overflow. FE had
energy residual `7.68e-9`; FE/Highland total dose was `0.999999996`, IDD L1 was
`0.0235%`, lateral-integral L1 was `1.46%`, and voxel L1 was `2.89%`. The
minibeam-disabled build independently ran the FE YAML with the same energy
residual.

The corresponding 10k full-physics pair passed production quality checks.
Highland/FE energy residuals were `4.59e-6/3.97e-6`, with no queue overflow.
FE/Highland total dose was `0.999912`, IDD L1 `0.0945%`, lateral-integral L1
`0.557%`, voxel L1 `6.58%`, and both Bragg maxima were in the same depth bin.
The voxel difference confirms that the selector changes transport rather than
only configuration text.

The rebuilt minibeam-disabled binary also parsed the short `fe` alias and ran
the full-physics water configuration. Its canonical YAML retained the spelling
`fe`, while transport selected `fermi_eyges`; the 100-history integration run
passed quality with an energy residual of `2.94e-8` and zero overflow.

### Staged secondary-ion integration

The same minibeam-disabled binary ran a 200 MeV/u, 10k-history full-physics
case at every species scope. All four runs passed production quality with zero
queue overflow. Direct routing counters in `energy_ledger.json` gave:

| scope | C12 FE steps | He-4 FE steps | p/d/t FE steps | other FE steps | Highland steps | energy residual |
|---|---:|---:|---:|---:|---:|---:|
| `c12` | 31,582 | 0 | 0 | 0 | 3,882,684 | `3.97e-6` |
| `c12_he4` | 31,504 | 1,361,865 | 0 | 0 | 2,809,671 | `3.74e-6` |
| `c12_he4_pdt` | 31,498 | 1,361,732 | 3,164,238 | 0 | 414,033 | `3.15e-6` |
| `all_charged` | 31,790 | 1,363,806 | 3,160,107 | 476,537 | 0 | `2.92e-6` |

The total number of secondary steps changes between scopes because altered
trajectories change stopping, escape and subsequent nuclear histories. The
route counters therefore verify dispatch, not physical equivalence. The
dedicated FE YAML now selects `all_charged`; the production YAML remains
explicitly `highland`, so integration does not silently change established
runs.

### Initial energy and internal-segment checks

A 2k-history pure-EM screen at 150, 250 and 300 MeV/u completed for both
Highland and FE with finite FP32 dose, energy residual below `9e-9`, and zero
overflow. FE/Highland total dose agreed to better than `2e-8`. The IDD L1 values
were `0.0361%`, `0.0505%` and `0.0582%`, respectively. The corresponding voxel
L1 values (`4.79%`, `4.87%`, `6.63%`) are dominated by low-statistics spatial
redistribution and are not accuracy claims against TOPAS.

At 200 MeV/u, a 100k-history FE-only `0.10` versus `0.05 mm` internal-segment
comparison gave a total-dose ratio of `0.99999995`, IDD L1 `0.0051%`,
one-dimensional lateral-integral L1 values of `0.308%/0.023%`, and voxel L1
`0.455%`. Changing the segmentation changes random-number assignment, so this
is an ensemble-level convergence screen rather than a history-paired identity
test. Both runs had energy residual below `7e-10` and zero overflow.

### Initial heterogeneous-CT integration check

The existing complete RT07575 shard (6,481,909 histories) was run with the same
seed and binary in Highland and FE modes. Both runs passed production quality,
had zero queue overflow and finite dose, and had energy residuals of
`3.65e-6/3.67e-6`. FE/Highland total dose was `0.999981`; voxel L1 was `1.94%`.
This is a transport/integration A/B, not a TOPAS clinical acceptance result.
On the local RTX 2080 Ti, elapsed time changed from `48.72 s` to `54.22 s`
(about 11% slower). The configured TPS weights determine this plan's history
count, so the attempted `--histories 5000` smoke override still executed the
complete shard.

The same-shard TOPAS dose was then used without new Monte Carlo. TOPAS dose was
scaled only by the exact GPU/TOPAS history ratio, and a frozen 50k subset of
the reference >=10%-of-maximum mask was evaluated with the project's existing
0.5 mm trilinear Gamma lattice. Results below are Highland/FE percentages:

| case | Global 1%/1 mm | Local 1%/1 mm | Global 3%/0 mm | Local 3%/0 mm |
|---|---:|---:|---:|---:|
| RT06423 | 92.744 / 92.756 | 74.930 / 75.242 | 95.014 / 95.076 | 65.032 / 65.398 |
| RT07575 | 94.770 / 94.754 | 75.748 / 75.646 | 95.938 / 95.712 | 70.246 / 69.754 |
| 20022516 | 96.184 / 96.326 | 76.916 / 76.932 | 96.800 / 96.806 | 63.972 / 63.912 |

All three FE runs passed quality and overflow checks. These single-shard
results show no broad clinical regression, but they are deliberately reported
as a paired integration screen rather than a final multi-shard acceptance.
RT06423 and 20022516 slightly improve several strict metrics; RT07575 is mixed.

An attempted CT-boundary remedy made the FE `0.10 mm` internal segment cap a
real outer transport-step cap for all CT C12 tracks. On the RT07575 shard it
increased steps from `2.069B` to `7.709B` and elapsed time from `54.22 s` to
`113.73 s`. Strict sampled Gamma changes were small and mixed: Global/Local
1%/1 mm became `94.814%/75.710%`, while Global 3%/3 mm changed from `99.968%`
to `99.942%`. The unconditional clamp was therefore reverted and recorded in
`failed.md`. A future boundary fix must identify actual stochastic transverse
material crossings and split only those paths; globally shortening every CT
step is not justified by this result.

Reproducible homogeneous-water YAML files are:

- `config/unified_water_highland_smoke.yaml` for the isolated pure-EM baseline;
- `config/unified_water_production.yaml` for full-physics Highland;
- `config/unified_water_fermi_eyges.yaml` for full-physics FE.

## Species-water calibration and CT scaling (2026-09-19)

The corrected 24-case pure-EM water matrix and its reproduction commands are
documented in `benchmark/fermi_eyges_species_water/README.md`. The original
matrix used TOPAS's default 600 MeV EM table ceiling, which is below the
1200 MeV total kinetic energy of the 300 MeV/u He-4 case. It is superseded by
job 7026 with a 10 GeV ceiling. In the corrected He-4 300 MeV/u, 10 mm case,
TOPAS/GPU mean losses are 3.50323/3.52421 MeV/u and exit spectrum widths are
0.174162/0.174193 MeV/u. Thus the apparent 45% He-4 stopping deficit was a
reference-generation error. Secondary Unified EM, not an empirical He-4
scale, restores the non-C12 straggling width.

Across the corrected matrix, the worst mean-loss error is 0.765%, the worst
exit-width error is 4.302%, and the median width ratio is 1.00314. The exact
water-exit scorer now records all 250,000 tracks per case before the next
transport draw; TOPAS's 0.010 mm vacuum drift is projected back to the water
face offline. At 1 mm the single-core/single-tail representation retains a
visible proton shoulder limitation. This is a model-shape limitation, not a
reason to tune full-dose compensation factors.

For CT materials the core uses the current material radiation length. The tail
event rate uses the water fit scaled as
`lambda_material = lambda_water * X0_water / X0_material`, with
`X0_water = 360.8297746 mm`. No bone/lung dose was used to fit parameters and
the tail angular scale remains the water value. B1--B4 all passed quality,
energy and overflow checks. B3 lateral core/halo fits improved slightly, while
B4 lung/bone changes were mixed by depth; CT scaling is therefore an integrated
candidate rather than a completed material validation.

## Frozen follow-up scope

- CT fitting and a new FE shoulder component are intentionally paused until
  the minibeam component-dose result gives evidence that either dominates the
  target ROI.
- Calibrate He-3 and heavier fragments. p/d/t/He-4 are now independently fitted
  in water and remain selectable through staged FE scopes; Highland remains the
  production default and per-run fallback.
- The minibeam-only energy-band ROI diagnostic shows that p/d/t/He-4 Bragg ROI
  dose above 300 MeV/u is below 0.7%, while the below-50 MeV/u band contributes
  about 43--53%. If a later species-isolated A/B exposes a stable residual,
  low-energy held-out slabs take priority over additional high-energy nodes.
