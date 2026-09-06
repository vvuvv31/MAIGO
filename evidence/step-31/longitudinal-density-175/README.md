# Section-0 density reference pilot — 2026-09-06

Status: REFERENCE_COLLECTED, NOT a validated density correction.

Independent 175 MeV/u EM-only C12, same 1% spread/source/geometrical grid as
the earlier held-out energy pilot. Each density has two TOPAS seeds (20k each)
and a paired GPU base/candidate (20k each, same seed). Scale remains 1.
Only the CCTG density payload changed; header, section-0 IDs and material LUT
are byte-preserved. Formula-derived float densities:
HU=-975: 0.03932345286011696; HU=-951: 0.06621015816926956 g/cm³.
No package, transport code or physics parameter changed this turn.

## Absolute reference comparison

3-D DoseToMedium, lateral integration, absolute per-primary normalization.
No normalization fit, no 1-D reference scorer.

| Depth mm | HU=-975 GPU/TOPAS−1 | HU=-951 GPU/TOPAS−1 |
|---|---:|---:|
| 2–7 | +2.806% | +1.700% |
| 2–20 | +1.284% | +0.563% |
| 20–50 | −0.007% | −0.032% |
| 50–100 | −0.005% | −0.039% |
| 100–120 | −0.030% | −0.016% |

TOPAS replica window differences: ≤0.0133% / ≤0.0550%, respectively.
Total deposition MeV/primary:
HU=-975 TOPAS 137.68755, GPU 137.69449;
HU=-951 TOPAS 235.88127, GPU 235.70558.

Candidate is intentionally inactive outside its reference-density scope.
For each density, base/candidate raw SHA match exactly; longitudinal moved and
escaped energy are zero. GPU baseline accepted; candidate only has the
mandatory unvalidated_longitudinal_candidate failure. Strict writer completed.
This is an inactivity/safety check, NOT evidence of successful density correction.
The reused holdout error gates are marked not applicable to this inactive model.

## Execution / provenance

Local Slurm jobs 2355–2358 all COMPLETED 0:0, 353–373 seconds.
Combined allocation 96 CPUs / 40 GB; each job ~5.56 GB RSS.
GPU stayed local. No outstanding jobs.
Data/configs/logs/CCTGs:
 /mnt/sda/wuwei/longitudinal_density_175_20260906

manifest.json freezes all prepared configs/CCTGs before execution.
analysis.json pins generated doses/quality and executable/tool hashes.
The preload DICOM and unchanged input stack are inherited from the
longitudinal-holdout-175 evidence; no extraction binary rebuild occurred.
Minimum stack verifier 16/16, diagnostic unit tests 7/7, diff check clean.

Reproduce into a NEW output directory with:
tools/prepare_longitudinal_density_pilot.py --template (previous 175 pilot)
--out (new directory) --schneider data/HUtoMaterialSchneider.txt
--grid /mnt/sda/wuwei/delta_section0/hu_n1000_200mevu.cctg
This prepares only; submission must follow live Slurm aggregate resource checks.
Read-only analysis: tools/analyze_longitudinal_density_pilot.py (campaign root).

## Decision

The entrance discrepancy becomes shorter at increased density. This motivates
testing a fixed-parameter mass-thickness prediction OFFLINE, not enabling source
density scaling in GPU. A projection test alone cannot validate the joint electron
response, fixed-birth conditioning, or air/tissue interfaces. Those plan gates
remain incomplete; do not mark the physical density model or plan2 complete.
No density-specific fitting, new LUT/package, patient Gamma, commit or push.
