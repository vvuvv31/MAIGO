# Fixed mass-thickness hypothesis — 2026-09-06

Status: PASS_OFFLINE_HYPOTHESIS_ONLY. No GPU density rule or physical package changed.

## Frozen prediction

At 175 MeV/u, linear interpolation of the unchanged 150/200 nodes gives
fraction=0.0846558834 and lambda_ref=23.81780485 mm.
No parameter was fit to the three validation cases; scale=1.

For a homogeneous medium, test lambda=lambda_ref*rho_ref/rho:
HU=-1000: 23.817805 mm; HU=-975: 6.854022 mm; HU=-951: 4.070732 mm.
This is a homogeneous mass-thickness hypothesis, NOT authorization to use
source-density scaling through heterogeneous CT.

R is exponentially distributed; deposit is uniform from 0 to R.
The displacement CDF is F(s)=1-exp(-x)+x E1(x), x=s/lambda.
Finite-volume source/destination-bin weights are obtained from its analytic
integral. The baseline 3-D dose lateral sum supplies the approximate uniform
per-bin source. No Monte Carlo resampling, coordinate shift or dose scale fit.

First prerequisite: reference-density offline versus actual GPU candidate
5 mm-bin disagreement <=0.5%. Observed maximum 0.0212%.
Then frozen error gates: windows <=1%, 5 mm bins <=2%.

## Results — OFFLINE predicted error relative to TOPAS

| HU | Baseline 2–20 mm | Predicted 2–20 mm | Max predicted 5 mm-bin error |
|---|---:|---:|---:|
| -1000 | +3.944% | +0.544% | 0.582% |
| -975 | +1.284% | +0.318% | 0.452% |
| -951 | +0.563% | +0.126% | 0.418% |

The two higher-density GPU candidates are still inactive and byte-identical to
their baselines. The improved numbers in this table are offline predictions,
NOT newly generated GPU doses or Gamma scores.

## Numerical checks and limits

Four new unit tests pass: analytic integral vs quadrature; probability/energy
accounting; cell refinement and mass scaling; invalid inputs.
All 11 longitudinal diagnostic tests pass; diff check clean.
Halving 0.5 mm to 0.25 mm changes projected sums only at roundoff in the
uniform-per-bin assumption. This is NOT a real transport step-convergence test.
Energy closure is a convolution identity, not independent physical validation.

Still approximated: fixed 175 MeV/u along depth; angular transport omitted;
source from already transversely redistributed voxel dose rather than actual
local step energy; no fixed-birth joint-response or material interface test.
Initial 0–2 mm remains excluded as prespecified. Two TOPAS replicas are only
pilot statistics. These restrictions prevent promotion to patient physics.

## Next bounded step

Test the prediction in an explicitly unvalidated, homogeneous section-0 GPU
diagnostic path at the three known densities, retaining exact target-cell
integration and strict output gates. Such a test must remain isolated from
heterogeneous patient routing and must be compared against both this offline
prediction and TOPAS, not just against the old GPU.

Before patient use, density-varying and air/tissue interfaces require actual
path material/density handling; fixed-birth and joint-response gates remain.
Do not reinstate the old source-density-only patient rule on this evidence.

Reproduce:
python3 tools/analyze_longitudinal_mass_thickness.py \
  --reference /mnt/sda/wuwei/longitudinal_holdout_175_20260906 \
  --density /mnt/sda/wuwei/longitudinal_density_175_20260906

No TOPAS/GPU jobs, patient Gamma, commit or push this turn. Output analysis is
read-only except for this separately saved evidence.
