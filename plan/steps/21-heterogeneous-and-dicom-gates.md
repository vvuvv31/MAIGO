# Step 21 — Heterogeneous phantom and real-DICOM final gates

## Objective

Validate the complete four-layer system first on controlled heterogeneity, then on real patient DICOM.

## Level 3 synthetic heterogeneous CT

Use `lung -> soft tissue -> bone -> soft tissue` with axis-aligned and oblique beams, plus thin inclusions and section boundaries. Validate DDA/face handling, range, target mix, fragments crossing materials, MCS, energy ledger, and 3D dose. Derive IDD only by transverse summation.

## Level 4 real DICOM

Freeze CT DICOM hash, RT plan/source hash, Schneider hash, coordinate transform, TOPAS/Geant4 versions, physics list, dose grid, normalization, and random seeds. Start with C12 primary-only diagnostic, then full C12 final states, then full secondary coverage. Do not jump directly to the final run.

## Final research gates

```text
total dose integral relative difference < 2%
range difference < 1 mm
3D gamma 2%/2mm pass rate > 95% (document threshold, normalization, dose cutoff)
primary survival relative difference < 2%
target interaction mix agrees statistically
major species integral relative difference < 2%
unsupported target/package lookup = 0
all shard overflow counters = 0
```

Specify the gamma implementation and dose threshold before running. Report statistical uncertainty, voxel size, and whether criteria are global or per beam/field.

## Final audit

- Verify formal data hashes against metadata.
- Re-run frozen water baseline and all phase gate reports.
- Confirm Schneider production mode never uses four-class nuclear/MCS paths.
- Confirm no 1D scorer was introduced.
- Confirm GPU jobs were local and TOPAS jobs were local sbatch within resource policy.
- Inspect `git diff --check`, full diff, test logs, data licenses/size, and repository cleanliness for intended files.

## Acceptance

Every fixed gate passes for the declared validation set, all provenance is complete, and reports are reproducible from manifests. Any failed case keeps the step `BLOCKED`/`IN_PROGRESS`; averages cannot hide a failing patient/energy/material.

## Commit intent

`test(ct): add end-to-end Schneider DICOM reference`
