# Short-range electron shortcut: first GPU smoke comparison

Same patient 20022516, 64 histories, seed 9186201, chunk 64, identical binary
and material-response data. Threshold defaults to zero. This is a single-trial
correctness/performance smoke test, not a production throughput or Gamma gate.

| Threshold (mm) | Shortcut packets | Primary kernel (s) | Change vs off | Dose difference |
|---|---:|---:|---:|---|
| 0 | 0 | 61.262500 | baseline | baseline |
| 0.1 | 27941 | 61.588422 | +0.532% time | raw dose bitwise equal |
| 0.25 | 29257 | 61.594563 | +0.542% time | raw dose bitwise equal |

All three have zero queue overflow, voxel/in-grid ratio 1.00000000012 and
relative energy residual below 2.1e-7. All remain **accepted=false** with exactly
`unvalidated_material_electron_response`; no production gate was weakened.
The two enabled runs' raw dose arrays equal the disabled run, so any IDD derived
from these arrays is also identical. This does not establish patient accuracy
at useful statistics or against TOPAS. No new Local Gamma claim is made.

## Interpretation

The implementation proves complete childless stopping inside the current
scoring voxel, bounded by the threshold and distance to every voxel face.
It scans up to 64 raw steps before replacing replay by local deposition.
This is substantially stricter than merely requiring range below a threshold
and distance from a material interface. Raising 0.1 to 0.25 mm increases hits
only 4.71%. The single-trial timing shows no measurable benefit; it does not
prove the exact source of the overhead or establish a significant slowdown.

Do not enable by default or advance to 1 mm on this evidence. The next candidate
is an optional precomputed per-source/row terminal-tail length index, built from
the same verified raw steps at load time. Preserve childless/terminal/material
checks, conservative path-length bound, voxel containment and fallback for
unknown tails. Allocate it only when the shortcut is enabled; account for its
memory and construction cost. Compare decisions against the scanning reference
before repeating this GPU test. Long/divergent lanes may still dominate runtime,
so even an O(1) predicate is not guaranteed to improve throughput.

## Reproduction and provenance

`execution.json` records binary/config hashes and wall times; `comparison.json`
contains dose and closure comparisons. Each case has its config, run log and
`out/gpu` results. The disabled output was copied from the original repository
`out/gpu` after the first runner used repository cwd; later runs use isolated
cwd with a data-directory link. The first 0.1-mm startup failed on a relative
data path before transport; its log and record are retained separately.

Run `python3 benchmark/benchmark20260909/summarize_short_range.py` to reproduce
the numerical comparison. No physics data or package was changed by this test.
