# Step 11 — Make nuclear optical depth piecewise correct at CT faces

## Objective

Guarantee that no sampled hazard assumes constant density/composition across a material boundary.

## Correct algorithm

Maintain a dimensionless remaining optical depth `tau = -log(U)` per pending interaction. For each segment:

1. Sample current voxel density/section.
2. Compute current `Sigma(E, section, density)`.
3. Compute condensed-history limit and exact positive distance to the next x/y/z CT face using direction-aware DDA.
4. Choose the smallest transport segment allowed by energy-loss/MCS/scoring constraints and voxel face.
5. Consume optical depth consistently over that segment (`tau -= integral Sigma(E(s)) ds`). A constant-rate approximation is allowed only with a documented step-energy error bound; otherwise use the established step hazard convention.
6. If `tau` reaches zero inside the segment, place the interaction at the solved distance.
7. If a face is reached first, move with robust nextafter/nudge semantics, re-sample the new voxel, and continue with remaining `tau`.

Simply drawing a new exponential distance after every face without preserving optical depth is forbidden unless memorylessness is rigorously implemented with a fresh independent draw at a completed no-collision segment and tested; prefer remaining optical depth for clarity.

## Homogeneous-face optimization

Skipping a face clamp is allowed only when both adjacent voxels have identical section and density within an exact documented criterion and scorer boundaries do not require the clamp. A same four-class label is not sufficient.

## Tests

- Axis-aligned and oblique rays; positive/negative/zero direction components.
- Start on/near a face, corner and edge crossings, anisotropic voxel spacing.
- Two voxels with same density/different section and same section/different density.
- Analytic two-layer survival `exp(-(Sigma1 L1 + Sigma2 L2))` and sampled first-interaction distribution.
- No infinite loop; minimum-step nudge does not skip a thin voxel.

## Acceptance

Analytic survival and interaction CDF agree within statistical uncertainty with systematic error <0.2% in high-statistics tests; boundary counters show no stuck tracks; no step spans a changed density/section.

## Commit intent

`feat(ct): preserve nuclear optical depth across Schneider voxels`
