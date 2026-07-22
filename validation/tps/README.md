# TPS source

The TPS source is opt-in. Existing configurations remain on the legacy source
path unless they explicitly contain:

```yaml
tpsSource: true
```

`tps_source` is accepted as a snake-case alias. Setting both aliases to
different values is an error.

## YAML beam fields

```yaml
tps_spots_file: validation/tps/spots_example.csv  # optional; empty = one spot
tps_particle_type: carbon                         # only supported primary today
tps_patient_position: HFS                         # HFS, HFP, FFS, FFP
tps_gantry_angle_deg: 90
tps_couch_angle_deg: 0
tps_collimator_angle_deg: 0
tps_isocenter_mm: [0, 0, 100]
tps_sad_mm: 450
```

The simulation coordinate system is the patient coordinate system used by the
voxel volume. Local beam axes start as `u=+X`, `v=+Y`, `w=-Z`. Positive gantry
rotates around patient `+Y`, couch around patient `+Z`, and collimator around
the local beam axis. The central source is `isocenter - SAD*w`; spot offsets
are `x*u + y*v`. Particles travel through vacuum to the first intersection with
the finite voxel scorer box, so TPS mode requires `enable_voxel_scoring: true`.
The component keys `tps_isocenter_x_mm`, `tps_isocenter_y_mm`, and
`tps_isocenter_z_mm` remain available as an alternative to the vector form.

## Spot CSV

Required columns are:

```text
energy_MeVu,x_mm,y_mm,mu_weight
```

Optional columns are:

```text
spot_id,energy_spread_percent,sigma_x_mm,sigma_y_mm,
sigma_x_prime,sigma_y_prime,correlation_x,correlation_y
```

Blank optional values inherit the corresponding YAML source value. Positive MU
is converted to the exact requested `number_of_histories` with a deterministic
Hamilton/largest-remainder allocation. Zero-MU spots are omitted. This is
importance sampling by fluence; the current transport still models carbon
primaries and does not silently reinterpret a proton plan.

Run the example with:

```bash
./build/oneapi-release/carbon_mc \
  --config config/beam_tps_source_example.yaml
```

Use `--plan-only` to inspect transformed source bounds and the central ray
without transporting particles.

## Legacy CT regression

Leaving `tpsSource` absent (or setting it to `false`) keeps the existing
TOPAS/CT spot path. After adding this module, the calibrated 10M full-plan run
against `ct/code/physical_dose.mhd` produced 95.4294% global and 93.6898% local
3%/3 mm gamma at the 10% threshold. The pre-module values were 95.4332% and
93.69%, respectively. The legacy configuration and source construction were
not changed.
