# 175 MeV/u independent energy holdout — 2026-09-06

Status: PASS_PILOT_ONLY; longitudinal candidate remains unvalidated.

175 MeV/u was not one of the 150/200/225 MeV/u fitted nodes.
No LUT, scale, transport, beam-model or physics parameter was changed this turn.
Both GPU variants use the same 20k histories/seed/source; scale stays exactly 1.
Two TOPAS 20k replicas were run on local Slurm (2353/2354), both COMPLETED 0:0.
Total request: 48 CPU, 20 GB; elapsed ~3 min per replica.
Source z=0.01 mm and 1% energy spread matched; EM-only C12, uniform HU=-1000.
Scorer is 3-D DoseToMedium, 100×100×440, 2×2×0.5 mm. Only lateral integration
is used, never a 1-D reference scorer. No fitted dose normalization/alignment.

## Absolute window errors relative to TOPAS

| Depth mm | Baseline | Candidate |
|---|---:|---:|
| 2–20 | +3.944% | +0.547% |
| 20–50 | +1.158% | +0.411% |
| 50–100 | +0.151% | +0.051% |
| 100–120 | +0.003% | −0.008% |

Candidate maximum absolute 5 mm rebin error (last group 3 mm): 0.590%.
TOPAS replica maximum window difference: 0.0715%; rebin difference: 0.1812%.
Prespecified window 1% / rebin 2% pilot error gates passed.
Total energy deposition: TOPAS 38.87254, baseline 38.98914,
candidate 38.81149 MeV/primary. Thin air slab: most beam energy exits.
Baseline quality accepted; candidate has only the mandatory unvalidated marker.
All 3-D outputs and strict writer checks completed.

## Limits and next gate

This checks ONE held-out interpolation energy at the reference density only.
Two replicas are a pilot noise check, not a comprehensive uncertainty estimate.
The test does not validate energies outside [150,225], other densities, air/tissue
interfaces, transverse/longitudinal correlation, or patient Gamma.
The source-to-endpoint density model has NOT been re-enabled or tuned.

Next: independent section-0 density response (HU=-975, HU=-951), with source,
geometry, stopping/composition and absolute normalization verified first.
The current candidate intentionally declines those densities; first collect
reference evidence, do not interpret its no-op there as a physical prediction.
Interface and fixed-birth joint-response validation still follow the plan gates.
Do not generate new nominal package values from this holdout or promote v2.1.

## Reproducibility

All configs/jobs/logs/doses: /mnt/sda/wuwei/longitudinal_holdout_175_20260906
campaign.json records gates fixed before execution.
analysis.json includes output hashes and normalized metrics.
input_hashes.txt pins executable/source/data and all 38 preload DICOM files;
these hashes were captured during execution, not a clean commit snapshot.
The existing authoritative v2.1 verifier passed 16/16 before GPU execution.

Read-only analysis:
python3 tools/analyze_longitudinal_holdout.py /mnt/sda/wuwei/longitudinal_holdout_175_20260906

Analyzer tests: 4/4; git diff --check clean. No patient run, commit or push.
