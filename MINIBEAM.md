# Minibeam research baseline — 2026-09-23

This branch records the repaired minibeam implementation used for the matched
3 cm × 3 cm TOPAS/GPU comparison. Its parent is d51e599; the changes were
developed in the separate MAIGO_review_fix_20260923 worktree.

## Implementation

- Extract and validate C12 restricted loss/range tables inside the active Geant4
  step context. Check the material, production cut and source metadata.
- Preserve the Urban range-law true/geometrical path conversion, terminal range
  handling and one scattering sample per accepted transport step.
- Use direction-owned physical voxel boundaries and voxel safety in the
  homogeneous water comparison when voxel_scorer_clamps_transport is enabled.
- Correct experimental delta-response geometry, escape accounting and component
  scoring. The principal comparison retains local electron deposition.
- Enable content-hash checks by default. GPU dose scoring remains FP32.

## Completed field comparison

Both engines used 12.8 million original source histories (4 independent batches
of 3.2 million), 256 spots, C12 at 250 MeV/u, EM-only physics, Water_75eV and
0.05 mm Cu/water step ceilings. The scoring grid is 0.1 × 100 × 0.25 mm;
the 100 mm vertical dimension is averaged. There is no fitted normalisation.

| Gamma mode | 3% / 0 mm | 3% / 0.2 mm |
|---|---:|---:|
| Global | 99.6528% | 99.9995% |
| Local | 58.5448% | 97.4983% |

TOPAS is the reference; the reference cutoff is 3% Dmax in both modes.
The 0.2 mm search uses PyMedPhys with a 0.02 mm interpolation step in the
transverse/depth plane. These rates do not establish a 1% pointwise dose bound:
valley voxel noise remains appreciable, and an entrance-layer discrepancy remains.

- [Field results and limitations](evidence/field3cm_emonly_20260923/results/FINAL_REVIEW_zh.md)
- [Gamma analysis](evidence/field3cm_emonly_20260923/results/gamma_0p2mm_3pct_local_global_cutoff3/GAMMA_RESULT_zh.md)
- [Monte Carlo uncertainty](evidence/field3cm_emonly_20260923/results/mc_noise_12p8M/MC_NOISE_zh.md)
- [Fresh Urban numerical regression](evidence/minibeam_publish_20260923/urban_localize.txt)

The 64 million history extension is running at this snapshot: 20 batches per
engine, reusing the first 4 and adding 16. Its preparation, serial runner and
analysis are under evidence/field3cm_emonly_64M_20260923. It runs directly with
a 30 GB memory cap and no swap; the scripts retain the original server paths.
The baseline_validation directory on the server contains the 12.8M analysis
regression, not completed 64M results.

## Reproduction and provenance

GPU executable SHA256:
003249623e409d386c85c233c27286d57a8ffca15271acfd20aeede3050ab4f5

TOPAS executable SHA256:
fcbcd71377ec6e5f35ae903bbc3b2765acadf871cff9dc1eb19adefa38ab5594

TOPAS 4.2.3 / Geant4 11.3.2 uses the scheduling extension at
[vvuvv31/topas_tps_source_extension, 4654e68](https://github.com/vvuvv31/topas_tps_source_extension/commit/4654e68a57cee4ab3737797a2707c5e54ba39d47).

The commit includes source, numerical tests, reference-table generators, the
active C12 reference CSVs, study configurations, analysis scripts and selected
completed reports. Large physics packages use the project's existing release
asset workflow. Simulation binaries, raw dose/phase-space arrays, environments,
third-party source/build trees and result archives remain server-side.

Research configurations preserve their original absolute paths and external
data dependencies. To reproduce elsewhere, first supply the pinned physics
packages and TOPAS extension, then rebase paths in copied configurations/scripts;
do not edit inputs of an active run. Manifests record the reference inputs,
random seeds, executable hashes and original raw-dose hashes.
