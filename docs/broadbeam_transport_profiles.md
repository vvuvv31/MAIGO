# Broad-beam and minibeam transport profiles

Research-only explicit selection; the default `reference` keeps the existing dispatch.

For non-minibeam YAML configurations, omitted MCS settings now default to
`enable_multiple_scattering: true`, `multiple_scattering_model: fermi_eyges`
and `fermi_eyges_species: all_charged`. Explicit settings, including disabling
MCS or selecting Urban/Highland, take precedence under `reference`.
Minibeam defaults are unchanged. This default does not select `broadbeam_fast`
or remove electron/neutron physics. The explicit `broadbeam_fast` preset below
still applies its documented physics overrides. FE uses segmented core+tail
unless Gaussian FE is explicitly selected or the fast preset is used.

```yaml
run_mode: research
transport_profile: broadbeam_fast
minibeam: false
em_model: g4_material_joint_v1
```

Add these settings to a valid proton broad-beam configuration with its CT/source,
Unified EM and nuclear packages. The profile is not a standalone physics
package and does not infer source geometry from field size.

## Effective physics

`broadbeam_fast` compiles separate primary and secondary kernels from shared
transport/scoring helpers. It selects Gaussian Fermi–Eyges by default for all charged
species (explicit `fermi_eyges_gaussian_only: false` retains the FE core and
Poisson tail in a separate specialized kernel), using material radiation length and density, and deposits electron
energy locally unless the pinned C12 CT joint response below is enabled. It disables
neutron transport and removes Urban, other electron-response and neutron package
paths before configuration validation/loading. Joint response paths and opt-ins
are preserved. Parallel hash prefetch retains direct effective input files and excludes removed
packages and speculative nested-manifest traversal; effective package loaders
retain their integrity checks. Canonical configuration records effective values.

Nuclear reactions, charged secondary transport, stopping/fluctuations and energy
accounting remain controlled by the original configuration. Maximum step and
relative energy loss are unchanged. FE internal segment limit is also unchanged.
This is not a pure electromagnetic approximation. Neutron energy is accounted as
a sink, not deposited as neutron dose.

For full-physics `broadbeam_fast` runs with inelastic reactions, charged secondary
transport enabled, omitted scheduling settings default to:

```yaml
secondary_species_grouping: true
secondary_step_chunking: true
```

Explicit values take precedence. Existing files with these flags set to `false`
must change them explicitly to use this scheduling. The continuation budget stays
at the compiled value (16 steps in the measured build). EM-only configurations
do not acquire these defaults. Neither physical step size nor RNG identity changes.

The controlled proton CT comparison measured 61.6k/s -> 74.3k/s process throughput
with these existing switches alone. Carbon CT and water benchmarks already used
them. Persistent-secondary workers, a runtime segment budget and compile-time
sampling isolation did not add sufficient throughput benefit and were reverted;
their candidate-only keys are not supported by the final code. Details:
`evidence/fe_scheduler_20260929/REPORT_zh.md`.

The Gaussian scattering power includes the previous FE core and Poisson-tail
second moments: `T = core_sigma_squared_per_mm + tail_rate * tail_sigma_squared`.
Per transverse axis its moments are `A0=T*L`, `A1=T*L^2/2`, `A2=T*L^3/3`.
This preserves local moments, not the non-Gaussian angular tails or equality to
Urban. Existing species fits are used; coefficients are not fitted to CT dose.

## RT07575 proton throughput profile (2026-09-30)

The measured optimized proton CT profile explicitly keeps core+tail FE:

```yaml
fermi_eyges_gaussian_only: false
fermi_eyges_max_segment_mm: 0.5
broadbeam_maximum_mean_loss_fraction: 0.02
enable_energy_straggling: true
enable_secondary_energy_straggling: true
```

This changes FE internal segmentation and the combined mean-loss guard. CT and
scoring faces, native EM proposals, nuclear processes, source weights and dose
normalization retain their existing rules. The `maximum_relative_energy_loss`
legacy setting does not select the actual Unified EM primary step. These values
have been evaluated on the RT07575 optimized proton plan; they are not changed
global defaults for other patients, beam energies or minibeam configurations.
Gaussian FE did not give a stable throughput benefit in the proton comparison.

For broadbeam primary protons without joint response, the host verifies every
matching record's native `is_ion` and fluctuation flags in the already validated
Unified EM package. If those records use the non-ion mean and universal
fluctuation model, a separate primary kernel compiles out the unused general-ion
correction branches. This flag is Geant4's **model selection**, not a statement
that the proton is uncharged. Native fluctuations, RNG identities, table values,
target competition, secondary transport and FP32 scoring remain in place. A
package with incompatible flags uses the existing kernel. Carbon, minibeam and
joint-response configurations use their existing dispatch. Set
`CARBON_PROTON_NONION_PRIMARY=0` for an otherwise identical same-binary control.

The standalone one-shard configuration is
`config/proton_rt07575_fe_fast.yaml`; its explicit source CSV contains 12,531,500
histories. Full-plan experiments and dose analysis scripts are in
`evidence/proton_broadbeam_opt_20260930/`, with frozen original plan inputs under
`benchmark/protonCT/RT07575_proton_GPU/`. See the experiment report for matched
timings, gamma definition, dose changes and statistical limits.

## Detailed and diagnostic selections

```yaml
transport_profile: minibeam_detailed
minibeam: true
```

This selects the existing detailed path and retains configured Urban/electron/
neutron options. It does not automatically supply missing physics packages or
enable every process. Broad-beam configurations can still use `reference` for
Urban and electron-response comparisons.

For generic-versus-specialized diagnostics use `reference` with the same effective
settings, including `fermi_eyges_gaussian_only: true`, `fermi_eyges_species:
all_charged`, `fermi_eyges_parameter_set: species_water`, and CT material MCS.
Do not compare only the profile name while leaving different physics enabled.

## Optional C12 CT joint electron response

The FE profile supports the historical Schneider electron-joint response in a
dedicated `broadbeam_fe_joint` kernel. The OFF path remains a separate
`broadbeam_fe` instantiation with joint replay removed at compile time. Both the
Gaussian FE and core+tail FE variants support replay. The same response code,
sampling streams, FP32 dose atomics, package verification and quality diagnostics
are shared with the reference implementation; enabling this option does not
dispatch to the reference transport kernel.

Add to a complete C12 CT research configuration:

```yaml
transport_profile: broadbeam_fast
ct_electron_segment_transport: true
ct_electron_joint_patient_experiment: true
ct_electron_joint_allow_research: true
ct_electron_joint_response_diagnostic_file: /path/to/joint_response.csv
ct_electron_joint_response_sha256: <64-character SHA256>
ct_electron_joint_response_metadata_sha256: <64-character SHA256>
ct_schneider_delta_tail_file: /path/to/schneider_section0_c12_delta_tail_v1.csv
```

The required Schneider physics bundle must also be supplied. Joint replay
replaces the transverse tail; the two are not applied together. Setting
`ct_electron_segment_transport: false` on the same FE configuration disables
replay and removes its companion tail table from the effective configuration.
Package paths without that switch retain the existing file-driven activation
semantics. Missing pins/opt-ins, non-C12 projectiles and non-CT geometries remain
invalid for this package.

`config/carbon_rt07575_fe_joint.yaml` is a complete machine-local RT07575 example
with the historical r3 package pins, 6,481,909 histories, and core+tail FE, matching
the preceding generic-path ablation physics. Run it from a directory with the
usual `data` inputs using the FE-joint build. It intentionally retains
`unvalidated_electron_joint_response`: dose files can be written while the run
returns quality failure for this experimental status. Do not interpret that
status as a missing response; inspect `electron_joint_response.queries`, domain
misses, invalid marches and energy accounting in the outputs.

The response redistributes primary-loop ionization only. Secondary electron
losses remain local; neutron transport remains disabled. The earlier generic
ablation results are in `evidence/carbon_fe_electron_ablation_20260929/`; they do
not constitute a dose or throughput measurement of this new dedicated kernel.

### Optional throughput settings

Two additional controls are restricted to research `broadbeam_fast` runs:

```yaml
# Default 0.01. Allowed range 0.01–0.05; applies to primary and charged
# secondary unified-EM combined mean loss. Geometry/nuclear limits still apply.
broadbeam_maximum_mean_loss_fraction: 0.05

# Default 1.0. Allowed range 0.05–1.0; active joint package required.
ct_electron_joint_sample_probability: 0.05
```

For packet sampling, actual probability is `p=max(configured_probability,f)`,
where `f` is the package's nonlocal energy fraction. A selected step moves
`deposited_energy*f/p` through the original response; an unselected step deposits
locally. The expected nonlocal energy is unchanged. The floor ensures a packet
never exceeds the step's energy and avoids negative local deposits. Independent
RNG dimension 903 selects the packet; transport identities and response draws
remain unchanged. This reduces replay work but increases dose sampling noise.
Disabling `ct_electron_segment_transport` also resets its sampling probability
to 1 in the effective configuration.

Machine-local examples (all retain nuclear reactions and charged secondaries):

| YAML | FE segment | Mean-loss cap | Energy straggling | Response sampling floor |
|---|---:|---:|---|---:|
| `carbon_rt07575_fe_joint.yaml` | 0.1 mm, core+tail | 1% | primary + secondary | 100% |
| `carbon_rt07575_fe_joint_straggling.yaml` | 0.5 mm, Gaussian | 5% | primary + secondary | 12.5% |
| `carbon_rt07575_fe_joint_fast.yaml` | 0.5 mm, Gaussian | 5% | disabled; mean loss | 5% |

Use `build/fe-joint/carbon_mc` with these inputs. The fast example is an explicit
physics approximation, not a replacement for native straggling validation.
All experimental response quality markers remain active. RT07575 throughput,
independent-seed gamma and limitations are recorded in
`evidence/fe_joint_200k_20260929/REPORT_zh.md`; passing that case's local1%/1mm
criterion does not establish other patients, projectiles or minibeam accuracy.

## Limitations and evidence

Gaussian tails and local electron deposition require validation at field edges,
lung/bone interfaces and distal falloff. Runtime quality acceptance does not
certify the agreed dose budget (additional bias <=1%, gamma decrease <=1 pp).
Current CT GPU neutron dose and secondary electron transport remain incomplete.

Implementation and pilot evidence: `evidence/broadbeam_kernel_20260929/`.
All GPU dose development uses FP32 scoring and atomics, never FP64 scoring.

## FE sampling and specialization (2026-09-29)

Both choices below use the specialized broad-beam primary/secondary kernels:

```yaml
transport_profile: broadbeam_fast
fermi_eyges_gaussian_only: false  # original segmented core + Poisson tail
fermi_eyges_sampling: segmented  # default
```

Set `fermi_eyges_gaussian_only: true` for Gaussian FE. The specialized secondary
resume state omits only Urban MCS state (224 versus 240 bytes in the measured
build); it retains the Unified EM optical-depth clock, RNG identity and audits.
The generic `reference` path remains available for same-physics comparisons.

An optional research approximation is integrated but is **not accuracy certified**:

```yaml
run_mode: research
transport_profile: broadbeam_fast
minibeam: false
fermi_eyges_gaussian_only: true
fermi_eyges_sampling: moments
```

This integrates A0/A1/A2 at the existing energy-midpoint segment nodes, then
samples one correlated angular/displacement pair per transverse axis. It keeps
the original scattering-power parameters and energy-loss process. It changes the
finite-step rotating-frame approximation and the subsequent random trajectory;
it is not an exactly equivalent scheduling change.

Merging requires multiple internal segments, no interior observation, angular
RMS <=0.05 rad and conservative pre-sampling geometry clearance. CT clearance
uses the physical voxel, including density boundaries; inserts/layer geometries
fall back to segmented transport. Gaussian displacement is unbounded, so the
six-sigma clearance is a conservative probabilistic guard, not a proof of no
boundary crossing. Failed eligibility uses segmented FE; invalid covariance
increments an always-active quality-failure counter. No sampled tail is truncated.

`CARBON_FE_DIAGNOSTICS=1` records counts in `energy_ledger.json` in this order:
calls, estimated internal segments, merged calls, single-segment fallback,
observation fallback, angular fallback, geometry fallback, invalid moments.
Diagnostic atomics affect performance; leave diagnostics off for timings.

See `evidence/fe_optimization_20260929/REPORT_zh.md` for measured results and limits.

## RT07575 FE segment pilot (2026-09-30)

`config/carbon_rt07575_fe_joint_fast_segment1.yaml` retains the prior research
fast physics settings and changes only `fermi_eyges_max_segment_mm` from 0.5 to
1.0 mm. This changes the internal FE scattering integration/sampling interval,
not CT face ownership or dose grid spacing. Three 6,481,909-history pairs
(including two holdout seeds with reversed execution order) give aggregate
127.54 -> 132.87k histories/s (+4.18%). Local 1%/1 mm, DTA/10 search, is
99.790/99.875/99.829%, above the user's 98% criterion; dose relative L1 against
paired 0.5 mm runs is approximately 0.534%. Queue overflow and response domain
misses/invalid marches remain zero. The experimental joint response quality
failure remains; this is not a production profile promotion or a proton/minibeam
validation. The 200k/s target has not been reached.

A same-physics candidate deferring endpoint CDF sampling until after packet
roulette was benchmarked in two reversed-order pairs and reverted for no
throughput benefit. Evidence and limitations:
`evidence/fe_joint_step_20260930/REPORT_zh.md`.

## Separate FE joint-response replay (2026-09-30)

The optional queue architecture compiles the primary transport and ordered
electron response into separate kernels. Select it in an otherwise valid
research C12 CT joint-response configuration:

```yaml
ct_electron_joint_execution_mode: queued
```

The default is `inline`. Both modes are available in the current
`build/fe-joint/carbon_mc`. Rebuilding queue support requires
`-DCARBON_FE_JOINT_QUEUE=ON`, with `CARBON_DOSE_FP32=ON` and
`CARBON_DOSE_FP64=OFF`. The CMake queue option defaults OFF; requesting queued
execution without that build support raises an error.

`config/carbon_rt07575_fe_joint_fast_queued.yaml` uses the validated RT07575
1 mm FE research settings and adds this execution selection. The queue requires
the pinned ordered path package, CT voxel dose, and no LET/component scoring.
It does not change response sampling, source/RNG identities, FE, EM, nuclear
events, or charged secondary transport. It retains the existing
`unvalidated_electron_joint_response` failure and exit=1.

The primary kernel checks the domain and response fraction for every original
query. Selected responses produce 48-byte tasks containing FP32 birth/direction,
the original CDF/azimuth random values, packet energy, and source scorer indices.
A separate kernel chooses the original CDF endpoint, replays the same path,
and scores FP32 dose. Tasks are drained after each primary batch; overflow or
incomplete drain rejects the result. Scorer escape energy remains accounted.

Two reversed-order pairs using the final binary, each arm 6,481,909 histories,
give **133.33k/s inline -> 167.19k/s queued (+25.39%)**. Local 1%/1 mm remains
99.8847% / 99.8449%, exact integer transport/EM audits agree, and dose relative
L1 is <=1.108e-7. GPU runs were direct and serial; builds used local Slurm.
This validation covers the carbon RT07575 Gaussian FE configuration. Core+tail
queue code is compiled, but no separate core+tail, other CT, proton or minibeam
performance claim is made.

Primary register usage falls from 246 to 226, without crossing its 25%
theoretical occupancy tier. The standalone replay kernel uses 82 registers;
its resource occupancy limit is 62.5% for block128, not a measured active-warp
percentage. The default 17,408-worker launch has only 136 blocks across 68 SMs,
so the actual worker grid also limits occupancy. Queue memory defaults to
256 MiB and is freed before the later secondary peak; measured overall tracked
memory remains about 8.36 GiB for this case.

The 200k/s target remains unmet. Charged-secondary transport now takes about
22–23 s and is the largest measured stage. Detailed timings, unchanged quality
limitations, source patch, checksums, and raw output locations are recorded in
`evidence/fe_joint_queue_arch_20260930/REPORT_zh.md`.

## Water/CT EM cleanup (2026-09-30)

Water and Schneider CT charged transport now has one energy-loss implementation:
`g4_material_joint_v1`. Primary and all 18 covered charged species use the pinned
Unified EM package. There is no secondary-CSDA or standalone fluctuation fallback,
and no runtime `CARBON_EM_SPECIALIZE` dispatch. FE/Highland/Urban remain MCS choices.

Removed legacy stopping tables are no longer loaded or SHA-prefetched. Nuclear
bundle rate/event pins and Unified EM/delta pins remain required. Old bundle JSON
may still contain an archived `stopping_table` section; it is no longer a runtime
input. The existing `primary_stopping_power_file` interface supplies only the
reference energy grid for independent nuclear/recoil interpolation. Its stopping
values do not drive primary water/CT loss. Unsupported elastic recoils retain
the independent reference table and prior interpolation without new calibration.
Copper/air, neutron recoil, and the current research electron-response modules
remain independent.

`em_model: legacy`, explicit `enable_secondary_unified_em: false`, enabled old
CSDA/fixed-block/Schneider stopping diagnostics, non-unit legacy CT scale and
standalone fluctuation packages now fail with a descriptive error. Old Unified
inputs with `primary_em_model: legacy`, `enable_secondary_unified_em: true`,
`ct_primary_midpoint_stopping` and obsolete stopping-table paths/pins are consumed
as compatibility aliases and omitted from the canonical effective configuration.
Unit `straggling_scale` and the former C12-only secondary EM switch are also
consumed as inert aliases; non-unit scales or nonempty scale tables are rejected.
Current Unified YAML presets no longer carry those aliases. `device: serial` is
unsupported because its removed solver implemented only legacy EM; SYCL device
selection remains available.

This cleanup was compiled in FP32 with local Slurm. It has not received new dose,
quality-audit, gamma or throughput validation; previous benchmark numbers above
belong to the pre-cleanup binaries. The YAML primary cutoff now controls its
terminal deposition directly, rather than being raised by the removed Schneider
stopping-table minimum (previously 0.12 MeV for C12). No throughput improvement
is claimed. See `evidence/unified_em_cleanup_20260930/REPORT_zh.md`.
