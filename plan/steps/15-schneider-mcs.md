# Step 15 — Use exact Schneider section radiation length for MCS

## Objective

Remove the four-class radiation-length collapse from production Schneider CT transport.

## Data source and implementation

Take `radiation_length_g_cm2` from the validated TOPAS material truth product, one entry per section. Upload a 25-entry LUT. Convert to local length using voxel density with documented units. Index by Schneider section directly.

Legacy `ct_material_class()` may remain for old file versions/configs but must be unreachable in logged `schneider-25` production mode. Add a runtime assertion or mode-level test proving this.

## Validation

Use narrow pencil beams through lung-like, soft-tissue, trabecular bone, and dense bone, including heterogeneous interfaces. Compare transverse sigma/core/halo metrics from 3D dose or fluence; never introduce a 1D dose scorer. Fix metrics and tolerances before viewing final results; minimum required reporting includes sigma at multiple depths and endpoint lateral profile.

## Acceptance

- All 25 exact values match Step 05 truth.
- Sentinel device-index test catches four-class collapse.
- Homogeneous and interface MCS report passes its predeclared statistical criteria.
- Stopping and primary attenuation gates remain passed.

## Commit intent

`feat(ct): use Schneider section radiation lengths for MCS`
