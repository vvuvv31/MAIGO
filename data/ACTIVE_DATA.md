# Current CT / water benchmark physics data

Updated 2026-09-13. Paths below are relative to the repository root.

## Required data

- Unified EM production entries now also require `data/em/unified_em_v1.bin`
  (1,386,682,652 bytes) and `data/em/unified_em_v1.json`. SHA256:
  `8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`.
  This one package covers water, 25 Schneider sections and 18 charged ions.
  The binary is distributed separately from Git; this change does not upload
  it to GitHub Releases. Older release archives do not supply this new package.
  Production activation uses the explicit AGENTS exception; density cut-onset
  and patient Gamma accuracy validation remain pending.

- Schneider CT and unified-water nuclear transport use the exact v2.1 stack.
  `data/schneider/v2_1_data_manifest.json` and
  `data/schneider/schneider_physics_bundle_v2_1.json` remain authoritative;
  all manifest inputs and provenance sidecars are retained unchanged.
- Final CT secondary material-specific stopping uses
  `data/schneider/schneider_ion_section_stopping_v1.bin` and its
  `.metadata.json` sidecar: 25 Schneider sections, 18 ion species,
  0.01–6000.11 MeV/u. Binary SHA256:
  `ef87ad19bd0b0552d846a6c4e23eb635746e375432a40f40fb79b202235dec20`;
  metadata SHA256:
  `c815808fd22e56e9db26475eeddb99aa596a9bc56aa01d0b8800c2e5e1ec12b8`.
  `tools/run_final_stopping_validation.py` now defaults to this location.
  Primary midpoint stopping and secondary exact-face transport remain enabled
  by that runner. This additional table does not replace the pinned primary
  Schneider stopping table or modify the v2.1 bundle.
- `config/unified_water_production.yaml` retains the water primary/ion stopping
  CSV inputs and selects the unified EM package above, `data/water_unified/g4_water_material.json`, and the same v2.1 nuclear
  packages. It does not require the historical CINEL02/FRED packages.
- Water/air stopping, HU stopping LUT, water cross-section defaults, LET
  electron fractions, HU material definitions, Schneider radiation-length
  records and their available sidecars are retained for current configuration
  inputs, shared loaders and provenance.

## Verification and transfer

From the repository root:

```sh
python3 tools/verify_schneider_v2_1_data.py
python3 tools/verify_unified_em_data.py
python3 tools/verify_schneider_ion_stopping.py data/schneider/schneider_ion_section_stopping_v1.bin
```

Copy the actual data files to a new machine, including the large nuclear
binaries and the new ion stopping binary; a Git clone alone does not provide
all local packages. Patient CT grids, spot plans and beam models remain
separate benchmark inputs outside this folder. The ion-table verification
script additionally audits extraction CSV/JSON and the TOPAS executable at
paths recorded in metadata; these provenance artifacts are not runtime inputs.

## Recoverable cleanup

13 files (6,420,000 bytes) were moved into
`trash/data_cleanup_20260911/data/`: optional FRED 2GR and fluctuation packages,
water78 candidates, the longitudinal electron-response candidate, and outdated
physics/archive documentation. Historical presets using these files require
restoration before use. No runtime fallback to trash was added.

[data_cleanup_20260911.json](data_cleanup_20260911.json) records source and
archive paths, sizes, SHA256 hashes and the new stopping-package relocation.
A second copy is in `trash/data_cleanup_20260911/manifest.json`. Trash is ignored
by Git. The earlier 2026-09-05 archive manifest is preserved in this archive;
its original destinations still refer to the earlier sibling trash directory.

The new stopping binary and metadata were moved from
`/mnt/sda/wuwei/ion_section_stopping_20260911_extended/compiled/`.
That location now contains compatibility symlinks to the files in `data/`,
so frozen benchmark configurations and metadata pins remain valid without
rewriting historical records. Extraction raw data remain at their original
location. To restore an archived input, verify its recorded SHA256, ensure
its original path is absent, and move it back together with its sidecars.

## Additional elastic research data (not in the 11.3.2 Release)

`schneider/all_ion_elastic_v1.bin` and `schneider/elastic_recoil_stopping_v1.bin`,
with their metadata sidecars, support the new all-ion elastic research presets.
They are additive packages; the pinned v2.1 stack remains unchanged.
Verify with `python3 tools/verify_all_ion_elastic.py`.
See [elastic scope, configuration and validation](../docs/all_ion_elastic.md).
The 11.3.2 ZIP predates these files and has not been replaced.
