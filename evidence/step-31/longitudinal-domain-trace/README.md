# Longitudinal low-energy domain diagnosis — 2026-09-06

Status: diagnosis complete; zero-domain-query coverage gate remains BLOCKED.

Added a bounded 4096-entry domain log only for the explicit homogeneous diagnostic.
The existing atomic counter reserves slots; no RNG or dose/transport updates added.
Record: global history, step, actual sampled initial energy, query energy,
step-start z, step length and locally retained energy. Host sorts records.
Truncation is explicit; analysis refuses truncation, missing/duplicate records,
nonfinite values, covered energies, and failed microMeV reconciliation.

## Direct runtime evidence

HU=-951, same 20k histories/seed/config as the preceding ON run:
- 521/521 records, no truncation, 45 distinct primary histories.
- Every query BELOW domain, none above.
- Initial energies 168.266830–170.515076 MeV/u: all inside [150,225].
- Query energies 148.202393–149.997345 MeV/u.
- Step-start depth 200.000015–219.500015 mm.
- 49 queries in [200,210) mm; 472 in [210,220) mm.
- Record energy sum 298.67516284 MeV; existing fixed-point counter
  298.674904 MeV; difference 0.00025884 MeV is within the per-record
  truncation bound 521 microMeV.
- 3-D raw SHA matches the pre-instrumentation run exactly:
  c09d252f972e4bc888cee6b764db7ade4dbeb04003a16de4ac5de8b20322a59b.

This directly identifies slowdown below the data floor late in the slab,
not an initially out-of-domain source, interpolation leak, interface error,
or a reason to weaken the gate. All observed events lie beyond the 2–120 mm
window used for the entrance pilot; that window passing did not prove full-path
energy coverage.

## Next step

Acquire independently sourced low-energy response evidence below 150 MeV/u.
Use a fresh diagnostic dataset (e.g. 140 MeV/u lower anchor and a separate
145/147.5 MeV/u holdout), with the same 3-D scoring/source/normalization protocol.
Do not choose a floor only a few keV below this seed's observed minimum.
No assurance of universal coverage follows from this single 20k-history census.

Only after low-energy response validation may a new explicitly unvalidated
candidate table/contract be introduced and full-length three-density tests rerun.
Keep current immutable CSV/metadata and minimum v2.1 stack unchanged; no clamp,
patient-specific scale, shortened phantom, source spread removal, or gate waiver.
Joint/fixed-birth/interface validation remains separate.

## Verification / artifacts

Local CUDA test passed, Python diagnostics 13/13, minimum-stack 16/16,
git diff --check clean. One local 20k GPU rerun, no TOPAS/patient Gamma.
Candidate quality remains rejected by its explicit unvalidated marker.
No commit/push.

Outputs and full domain log:
 /mnt/sda/wuwei/longitudinal_domain_trace_20260906
Read-only checker:
python3 tools/analyze_longitudinal_domain_log.py (trace directory) \
 --before /mnt/sda/wuwei/longitudinal_uniform_gpu_20260906/hu951/on

validation.json pins source/executable/config/raw and full ledger. The worktree
was not clean; this is not a clean-commit physics validation.
