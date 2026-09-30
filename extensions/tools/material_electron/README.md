# Material electron response compiler

`compile.py` compiles pinned homogeneous TOPAS V4 histories into a material response
(schema 6). `export_state_refs.py` binds samples and ancestry nodes to the unchanged
raw rows. No benchmark dose is a compiler input.

Required arguments: `--manifest`, `--output`, `--energy-max` (5 MeV/u multiple),
`--projectile-z`, `--projectile-a`, `--material-response`, `--transport-state-source`.
The manifest supplies explicit Schneider section composition, physical density,
source histories, recording bounds, and SHA256 pins. A runtime index (schema 2)
must repeat the projectile ZA and pin each response and state-reference file.

The present GPU integration is research CT primary EM only. Packets redistribute a
measured birth-energy fraction of the existing total energy loss, with an independent
RNG stream. They do not add a second loss. Legacy C12 indexes retain their original
species restriction. Missing electron continuation or geometry is a hard failure;
uncovered photon continuation is separately accounted, never scored locally.

Birth and path statistics and interpolation in energy/density require convergence
studies. Successful compilation or execution does not validate patient dose.
The initial real-CT campaign is documented in
`evidence/real_ct_electron_diagnostic_20260928/REPORT_zh.md`.

For a large bank, `material_electron_response_memory_mode: host_mapped` can be
combined with `material_electron_response_hot_device_budget_MiB` (default 0).
A positive explicit budget places raw states, birth samples and small metadata in
device memory; larger continuation indexes remain mapped. It preserves all bytes
and the sampling law, and rejects budget overflow without dropping tables.
