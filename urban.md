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

`ConstructCharged`'s `isWVI = true` default only overrides muons and light
hadrons; it does not set WVI on the ion `hmsc`, and `ConstructIonEmPhysics`
does not either. So opt4 -> C12 ion MSC = Urban.

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

The constant ~1.24 ~= sqrt(pi/2) = 1.253 is the Rayleigh median for a
per-component Gaussian of width theta0, i.e. exactly the Urban
`sqrt(h)*(c1 + c2*ln(h/X0))` step dependence.

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
