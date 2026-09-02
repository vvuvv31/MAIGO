# Step 01 — Audit the local CT implementation before editing

## Objective

Replace assumptions from public HEAD with an exact map of the current dirty worktree, and decide which existing CT material-rate work can be reused without extending frozen secondary physics.

## Inspect

- `include/carbon/ct_grid.hpp`, `src/ct_grid.cpp`
- `include/carbon/cross_section.hpp`, `src/cross_section.cpp`
- `include/carbon/transport_config.hpp`, `src/config.cpp`
- `src/transport_sycl.cpp` and `src/detail/sycl_*`
- `include/carbon/inelastic_package_v2.hpp`, `src/inelastic_package_v2.cpp`
- `startup/extensions/Cinel02MaterialRateNtuple.*`
- `startup/package_tools/extract_ct_cinel02_rates.py`
- all CT/CINEL tests and CT configs

## Required audit table

For each existing path, record: config key, host loader, validation, resampling, allocation, upload, kernel pointer, indexing order, units, density normalization, primary use, secondary use, fallback behavior, tests, and known gaps. Trace at least:

```text
ct_schneider_file
ct_schneider_cross_section_file
cinel02 material-rate input (if present)
CtGrid material_id
ct_cross_section_material_index
ct_material_class
CT face clamp
primary and secondary nuclear hazards
```

## Reuse decision

Classify each uncommitted CT change as `reuse unchanged`, `supersede later`, or `out of scope/frozen`. Existing code is not validated evidence. Any code that routes Schneider sections through four-class tables is legacy fallback, never the production Schneider path. Any code that adds CT behavior to secondary transport is frozen until Step 20 and must not be used to claim P1/P2 completion.

## Acceptance

- The audit names functions and line anchors from the current tree, not GitHub URLs.
- Every config-to-kernel chain has a unit declaration.
- All silent fallback paths are listed.
- The P2 minimal edit surface is explicit and excludes final-state replay and secondary transport.

## Evidence

Write `current-state-audit.md` and `path-matrix.csv` under the Step 01 evidence directory.
