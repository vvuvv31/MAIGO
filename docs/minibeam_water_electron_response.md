# Minibeam primary electron response

`minibeam_water_delta_response_model: water_response_v2` enables a **research**
spatial response for primary-ion EM energy loss in homogeneous `Water_75eV`.
The full nuclear transport stays enabled. A matched Z/A table and SHA256 pins
are required; a carbon table is rejected for a proton source. Defaults remain
`local`; `water_response_v1` retains its C12 EM-only scope.

Configurations:

- `config/proton_minibeam_fullphysics_water_response.yaml`: 170 MeV proton.
- `config/carbon_minibeam_fullphysics_water_response.yaml`: 250 MeV/u C12 spot.

Build with `CARBON_ENABLE_MINIBEAM=ON`, `CARBON_DOSE_FP32=ON`, and
`CARBON_DOSE_FP64=OFF`. Example executable: `build/entrance-response/carbon_mc`.
Configurations reference local physics packages. Run in an isolated working
directory with `data` linked to the repository's data directory; the default
output is `out/gpu`. No dose scaling or reference-dose fitting is applied.

## Data and extraction

`data/water_electron_response_20260926/manifest.json` records copied packages
and hashes. Proton schema 5 covers 0–250 MeV in 5 MeV channels; the response
includes 221,843 primary-electron families and 1,443,614 energy-weighted
segments from 50 source energies × 64 histories. Lower channels also receive
slowing-down tracks from higher source energies. Carbon reuses the existing
0–300 MeV/u response with exact retention of its far-tail samples and ancestry.

`extensions/topas/common/PrimaryElectronDepositNtuple` records the actual
parent step for direct ionization electron births (`hIoni` and `ionIoni`).
`PrimaryIonizationResponsePhysics` is **extraction only**: it removes massive
charged particles' Coulomb scattering and nuclear stopping so nuclear recoil
cannot contaminate an electron-energy partition. It does not change electron
or photon transport. Do not add this module to a full-physics benchmark.

`extensions/tools/water_electron/compile.py` audits source identity, complete
step genealogy, parent-child energy/position links, per-family/event closure,
physical escape and payload hashes. A charged particle escaping the extraction
volume rejects compilation. The rejected initial opt4-only dataset is retained;
the accepted ionization-only extraction reran every energy with the same seed.
All TOPAS runs used local Slurm (17915 rejected dataset, 17972 accepted dataset).
Preparation and exact inputs are under `evidence/entrance_response_20260926/`
and `/mnt/sda/wuwei/entrance_response_20260926/`.

## Runtime semantics and limits

Only the primary's continuous EM step loss is partitioned. Nuclear collision
energy and secondary-ion energy are not mixed into the response. Dedicated RNG
dimensions 80–83 leave transport RNG unchanged. The deposit-weighted response
has channel-averaged energy fractions; it is not an explicit electron spectrum
sample for each individual ionization event.

The shared path graph distinguishes physical slab escape from a deposit outside
the scoring ROI. Deposits are credited to their destination in both dose and
grid energy accounting. Invalid/out-of-domain packets reject the run rather
than silently falling back to local scoring. Electron response requires voxel
scoring, unit-density homogeneous water, LET off, and research mode.

Remaining limits: secondary-ion electron relocation, copper/air-born electrons,
air-to-water return after electron escape, finite response-library uncertainty,
and 5 MeV channel averaging. This feature does not establish complete entrance
peak/valley agreement. The 1M paired GPU comparison and unchanged TOPAS reference
are reported in `evidence/entrance_response_20260926/results/REPORT.md`.

## Primary air transport follow-up (2026-09-26)

The explicit research configurations
`config/proton_minibeam_fullphysics_air_response.yaml` and
`config/carbon_minibeam_fullphysics_air_response.yaml` additionally enable
`minibeam_air_mcs_model: urban_v2` with an independent source-ZA air bank.
Water response and air transport are separately selectable. This follow-up
also fixes beamline air loss counted twice in total-difference copper removal
branches, and uses analytic finite-slit intersections for long air steps.

See `evidence/minibeam_air_20260926/README.md` for provenance, configuration,
rejected initial navigation candidate, and model limits. Its paired dose
comparison keeps the electron response enabled in both modes; it must not be
interpreted as a new response-on versus local-deposition comparison. Defaults
remain unchanged pending species-specific dose and entrance validation.

## Native carbon copper elastic and higher statistics (2026-09-26)

Use `config/carbon_minibeam_fullphysics_native_elastic_response.yaml` for the
latest carbon research comparison. It replaces the legacy copper elastic table,
which incorrectly treated the full transport-step energy loss as nuclear recoil,
with native `ApplyYourself` samples. Proton already used the native export path.
The new optional export ceiling/sample count retain byte-identical default
proton output. See `evidence/carbon_copper_elastic_20260926/REPORT_zh.md`.

The follow-up 5M-per-species dose comparison is recorded in
`evidence/minibeam_response_5m_20260926/REPORT_zh.md`. TOPAS birth-volume and
neutron-ancestor diagnostics are in
`evidence/entrance_components_20260926/REPORT_zh.md`. Outside-water-born electron
deposits explain only about 0.045% of integrated entrance IDD (0–20 mm); this
response does not replace neutral nuclear transport or its charged descendants.
No measured TOPAS component fraction is added as a GPU dose correction.

## Copper proton energy-loss correction (2026-09-27)

The latest proton air+response research YAML additionally selects
`minibeam_copper_proton_straggling_model: g4_universal_delta`. The old
zero-clipped Gaussian biases short-step loss upward. The replacement uses
native Universal restricted fluctuations plus compound-Poisson delta energy
transfers at the G4_Cu 0.05 mm cut. It applies to Z=1,A=1 copper tracks only;
it does not transport the emitted electrons. The copper density/cut and
unit fluctuation scales are constrained, and invalid draws reject the run.

Independent TOPAS thin-slab moments and 1M/5M same-seed minibeam comparisons
are in `evidence/copper_loss_20260927/REPORT_zh.md`. The corrected 5M proton
IDD0–20 remains -0.916 ± 0.109%: this loss correction does not resolve the
missing-neutral contribution or establish complete peak/valley agreement.
Historical results retain their exact per-batch YAML snapshots.

### Primary air ionization clock (2026-09-27)

The proton air-response and copper-fluctuation research configurations now pin
`minibeam_air_ionization_rate_file` plus its SHA256. This table supplies the native
G4_AIR hIoni proposal clock (0.05 mm production cut) to primary beamline air
transport. Native proposal limits compete with Urban and geometry on TRUE path;
accepted true path consumes optical depth and the clock resets on its own
proposal. The existing aggregate air stopping law remains the only air-energy
subtraction. No MCS amplitude, fixed air step, or dose scale is adjusted.

The table validates source ZA, material, density, cut, hash and energy coverage.
An omitted table preserves the previous air path; the selector is research/smoke
only. Existing carbon configurations do not select this proton table. The added
`AIR_IONIZATION_CLOCK` log records step/proposal counts, while invalid transport
still fails the Urban quality gates. See `evidence/proton_edge_fix_20260927` for
native step traces, offset-grid rate validation and the 5M paired comparison.

## Proton entrance priority and neutron scope (2026-09-27)

The current user-selected entrance comparison uses **0–20 mm peak and valley**
as its main metrics. The first 0–0.5 mm layer is supplementary diagnostic data;
IDD is not the current optimization target. Preserve the positive central valley
and report symmetric/pooled windows alongside it.

`evidence/proton_entrance20_20260927/` records a pure-Urban TOPAS neutron-lineage
partition with the same seeds as the full reference, plus GPU component-score
regression. The reference excluding neutron descendants is diagnostic only:
it does not replace the full-physics reference or supply an additive GPU dose
correction. Current copper and water nuclear-product paths skip neutral
transport, so generating neutron entries in an event bank does not implement
their later interactions or charged descendants.

The copper primary native-mean candidate remains opt-in through
`minibeam_copper_primary_native_mean_loss: true`. Its 10M paired comparison
(`evidence/proton_entrance_followup_20260927/`) did not establish a significant
0–20 mm improvement; research configurations retain the default `false`.
