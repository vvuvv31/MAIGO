# Unified water / Schneider charged-ion EM package

Production integration authorized 2026-09-13 under the explicit exception in
`AGENTS.md`. `em_model: g4_material_joint_v1` selects one binary and a shared
primary/secondary transport algorithm. The water production entry and new RT07575
production entry select this model; historical configs retain legacy behaviour.
The validated v2.1 nuclear stack remains mandatory. Production execution does not
assert completion of density cut-onset or patient Gamma accuracy validation.

## Coverage and data

The package covers water (`Water_75eV`) and all 25 Schneider sections, with 175 explicit
material density nodes (3,150 material/ion records; binary about 1.29 GiB). Each node contains 18 projectiles:
p, d, t, He3, He4, He6, Li6, Li7, Be6, Be7, Be9, Be10, B8, B10, B11, C10, C11, C12.
Additional heavy elastic recoil isotopes retain their existing recoil stopping/MCS
path; they are outside this registry.

Data are extracted from TOPAS/Geant4 11.3.2, `g4em-standard_opt4`, with the existing
10 GeV EM table setting. The extraction instrument reads native process tables
and species parameters; it does not tune the default TOPAS dose reference.
The package energy ceiling is 6000 MeV **total kinetic energy per ion**, i.e.
6000/A MeV/u. Queries outside the package material/density/energy domain reject
the dose; there is no silent water or older-package fallback.

`data/em/unified_em_v1.json` records the binary SHA256, extraction binary SHA256,
source table hashes, density nodes and projectile registry. `EMJOINT1` v1 stores
native restricted stopping, range, inverse range and delta-rate spline intervals,
plus charge, along-step correction and fluctuation quantities on a dense energy
mesh. The particle records include rest mass, native mass scaling, electron cut,
excitation energy, StepFunction parameters, linear-loss threshold, spin,
form factor, magnetic moment and the native low-energy stopping threshold.

## One step

Production uses the two-moment Gamma delta aggregate and analytic Poisson-partition correction accepted on 2026-09-15. See [Methods, Section 3](../../mm.md) for equations and limitations. Native restricted stopping/range, ion corrections and restricted fluctuations remain active. No discrete delta clock or collision step limit remains. An additional 1% mean-loss guard applies only where delta stopping is positive. Nuclear optical depth and MCS remain separate; delta energy is deposited locally.

The required companion `unified_em_delta_moments_v2.bin` is reproducibly derived from the pinned core EM package, including accepted-spectrum moments for all material/ion nodes. Run `python3 tools/build_delta_moments.py` after installing the core package; verification pins both SHA256 hashes. No old/water-table fallback is allowed. This table is not bundled in older releases.

## Configuration

Production entries: [`config/unified_water_production.yaml`](../../config/unified_water_production.yaml)
and [`config/rt07575_unified_em_production.yaml`](../../config/rt07575_unified_em_production.yaml).
The two earlier research configs remain available for diagnostics and for
all-ion nuclear elastic, which has its own existing research-only guard.
The new CT production entry does not enable that elastic bank; this EM exception
does not authorize a separate nuclear-elastic promotion.
The patient example retains the existing machine-specific CT/spot paths and is
not a completed patient validation. Apply these fields to other complete configs:

```yaml
run_mode: production
em_model: g4_material_joint_v1
em_package_file: /absolute/path/to/MAIGO/data/em/unified_em_v1.bin
em_delta_moments_file: /absolute/path/to/MAIGO/data/em/unified_em_delta_moments_v2.bin
em_package_sha256: 8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855
primary_em_model: legacy
enable_energy_straggling: true
enable_secondary_energy_straggling: true
straggling_scale: 1.0
ct_secondary_exact_faces: true
```

Remove `primary_joint_em_data_directory` and electron-response/delta-tail options.
Keep the required v2.1 nuclear/stopping inputs and recoil inputs. One joint EM core plus its derived moment companion
serves both geometries; this is not a claim that the entire simulation now
needs only one physics file. GPU execution is local only. Before a Schneider run:

```sh
python3 tools/verify_schneider_v2_1_data.py
python3 tools/verify_unified_em_data.py
```

The binary loader checks SHA, schema, sizes, all sections and the exact ion registry.
A nonzero first counter in `[unified-em-audit]` rejects the output. Counters are:
failure, primary steps, secondary steps, delta proposals (zero), accepted deltas (zero),
continuous micro-MeV, delta micro-MeV, secondary sampling failures.
Any secondary queue overflow also invalidates a validation run.

## Validation and remaining gates

See `evidence/unified-em-20260913/README.md` for measured results and exact hashes.
The initial package passed native lookup, CPU/GPU finite-step mean, all-ion
fluctuation and delta distribution checks and seven 50k transport checks.
The refined package passed 94,500 native finite-step queries on both CPU and GPU
(maximum relative difference 0.0130385%), 700,000 lookup queries, 432 fluctuation
combinations and 162 delta-distribution checks. Seven 50k full transport cases
also passed with zero overflow and zero EM failures using this final package.
Density refinement from 77 to 175
nodes reduced the withheld-node mean-loss diagnostic from 4.92% to 0.787%.
The remaining maximum is near the electron production-cut floor in low-density
material. The delta-rate onset is particularly sensitive there: relative rate
errors alone are unsuitable at a zero reference rate. This region still requires
an independent density-specific closure gate; it is not validated by the
explicit-density b3/b4 cases.

The user authorized production integration before the remaining accuracy gates.
Arbitrary-density accuracy, patient one-shard BODY-masked Gamma and a full patient
comparison remain outstanding. The quality report always includes
`unified_material_em_accuracy_pending`; `status: pass` only means the runtime
checks passed. Production configs reject a different package hash or disabled
primary/secondary fluctuations. Scope, SHA, energy and overflow checks remain active.
The 1.29 GiB binary is not stored in Git and has not been added to an existing
GitHub Release in this change; copy it separately and run the verifier.

## Reproduction

Use `tools/topas/unified_em/prepare.py` to prepare material/ion configs;
`build_remote.py` builds an isolated diagnostic extension, and `run_extract.py`
exports one killed seed track per ion. `compile_unified_em_package.py` requires
all material/ion exports and a common extraction executable hash.
`run_probes.py` samples actual `G4VEnergyLossProcess::AlongStepDoIt` with fluctuations
disabled only for deterministic component checks. `run_distributions.py` calls the
native fluctuation sampler; neither changes full-dose reference physics.

CMake executables `unified_em_lookup`, `unified_em_mean`, `unified_em_gpu`,
`unified_em_distribution`, `unified_em_delta` and `unified_em_density` provide the
component checks. The mean/GPU/distribution tools take package, SHA, and export-root;
lookup/delta/density take package and SHA. Density is a diagnostic, not a passing
production gate by itself.
