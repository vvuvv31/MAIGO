# Depth mirror fix — 2026-09-06

Supersedes the shared controlled 200 MeV/u writer blocker in the previous report.

Root cause: transverse delta-tail scored its endpoint into a destination 3-D voxel
but retained all depth/in-FOV energy at the step-start bin. The midpoint can cross
a depth face even for a straight pencil beam; oblique transverse offsets can also
change z. Escaped energy was likewise still present in the in-FOV source tally.

Fix: credit relocated energy to both destination tallies, remove it from both
source tallies, and remove transverse escape from the in-FOV source tally.
The unrestricted-depth legacy escape convention is retained. No voxel writes,
RNG, package, stopping, MCS, or physical energy ledger were changed this turn.
The strict writer tolerance is unchanged.

## Verification

Six sequential local RTX 2080 Ti runs, 1000 histories each, completed all output
checks: base/longitudinal at 200 and 120 MeV/u, oblique200, edge200.
Candidate runs still exit nonzero solely for unvalidated_longitudinal_candidate;
their successful output checks do not constitute physical acceptance.

- baseline cases accepted; candidate cases intentionally not accepted.
- 120 MeV/u on/off and previous baseline raw SHA are identical:
  4f72a3db32448a6100ac134f1aae9ccd938aa8eec66152d527472f1ce929cc1b.
- oblique transverse moved/escape: 2589.37801 / 169.88999 MeV.
- edge transverse moved/escape: 1828.316408 / 1802.556592 MeV.
- Final CUDA unit test passed; minimum stack verifier 16/16; diff check clean.
- These controls disable spread/straggling/MCS for isolation. The earlier realistic
  small-sample physical-residual failure is not claimed fixed.

Artifacts (configs/logs/quality/ledgers/3-D outputs):
/mnt/sda/wuwei/schneider_depth_mirror_20260906_KOQgiH

Reproduce locally with tests/run_delta_depth_mirror_gpu.py, the built carbon_mc,
and the archived base200/run.yaml as --config; choose a NEW --out directory.
The runner deliberately retains the strict sparse 3-D writer check.
No TOPAS, new patient Gamma, full legacy regression, commit or push this turn.
Independent energy/density/interface/joint-response validation remains blocked.
