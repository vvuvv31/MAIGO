# Active physics data after 2026-09-05 cleanup

The cleanup keeps the exact Schneider v2.1 14-projectile stack used by the
three-case full-statistics GPU benchmark, plus the current CINEL02/basic-water
runtime inputs. It does not change physics, replace packages, or downgrade data.

- CT authority: `schneider/v2_1_data_manifest.json` and
  `schneider/schneider_physics_bundle_v2_1.json`.
- Water: retain Geant4 primary/ion stopping and water inelastic XS; retain
  `packages/fred_3_76_mcs_2gr.*` and
  `packages/c12_G4_WATER_fluctuation_topas_4_2_p3_g4_11_3_2_50k.*`
  for CINEL02 water configurations. Their CINEL02 event/rate files under
  `/mnt/sda/wuwei/` were not moved. Those external inputs are still required;
  this repository alone is not a self-contained water package installation.
- Keep CT upstream-air stopping, frozen benchmark input pins, sidecar
  provenance, and compiled-X0 source records.
- Existing uncommitted longitudinal delta-response candidate data and spot-plan
  fixtures are preserved; they are not promoted to the validated stack.

## Archived data

60 files (865.31 MiB) were selected for recoverable relocation
to `../trash/data_archive_20260905/data/`, preserving original relative paths.
The archive is ignored by Git and is not included in future clones.
Nothing was permanently deleted. Moving within the repository filesystem does
not free disk space.

[data_archive_20260905.json](data_archive_20260905.json) records original paths,
archive paths, sizes and SHA256, as well as the retained pre-cleanup input set.
Archived items include old Schneider event/rate packages, old FRED event
libraries, old fluctuation data, copper/four-material experiments, and obsolete
upstream CSV/phantom inputs.

Old Step 08/14/19/20/21 and v2-specific tests or historical FRED/insert
configurations may require these archived inputs. Their paths and tests were
NOT redirected, disabled, or changed to use newer data. Current-data integrity
checks are distinct from those legacy regression suites. The historical
`README_physics_tables.md` describes older paths too; this file documents
the current cleanup boundary.

## Restore a historical input

Read its exact `source`, `destination`, and SHA256 in the archive manifest.
Check that the original source path is absent and that the archived SHA256
matches, then move that one file back to its original path. Restore all required
sidecars with the data. Never overwrite a newer file or use a runtime fallback
that silently searches `trash/`. Restore the full relevant fixture set before
running historical regression suites.

Verify current CT inputs with:

```sh
python3 tools/verify_schneider_v2_1_data.py
```
