# T4 throughput optimization — 2026-10-09

The accepted electron change updates the joint-transport packet in place. This
removes the large packet value argument/return and its associated register
spills. The source, packages, FP32 arithmetic, RNG calls, fluctuations, step
caps, electron/photon coverage, and FP32 dose atomics are preserved.

The initial Colab allocation is a Tesla T4 (15,360 MiB), Ubuntu24.04, driver
580.82.07. Matched control/candidates were compiled together on the local CPU
with Clang24 and CUDA12.6 and executed on the T4. The local ongoing carbon HU0
campaign was preserved. Colab reported1.07 CU/hour. The first two startup
attempts failed before transport because an overwritten library path hid the
Colab NVIDIA driver libraries; registering `/usr/lib64-nvidia` with ldconfig
resolved this. Failed startups contribute no timing or dose evidence.

## Electron optimization

The compiler screen reports joint-pilot spills48/68 B (four instantiations)
and64/92 B (two) in the control; the in-place candidate has zero joint-pilot
spill stores/loads. This does not mean all electron functions or the worker
have zero local-memory traffic. No retired expected-value scorer is restored.

Every ABBA process runs four batches using one resident GPU context and
packages. The first batch is excluded from resident timing; remaining seeds
are matched. Carbon HU0 uses200 MeV/u C12,121 equal spots,3cm field,
0.5mm scoring and121,000 particles/batch. Proton CT uses the default lung
benchmark and100,000/batch; it is not RT07575 head validation.

| Workload | Control particles/s | In-place particles/s | Pooled ratio | Paired ratios |
|---|---:|---:|---:|---|
| Carbon HU0 | 7,072.5 | 7,091.3 | 1.00266 | 0.98564 / 1.01941 |
| Proton CT minibeam (lung) | 9,001.8 | 9,581.8 | 1.06444 | 1.07918 / 1.04968 |

The carbon whole-run difference is unresolved against thermal/run-order drift;
no useful carbon speedup is claimed. Carbon electron mean0.34512→0.27678s,
proton electron mean5.98283→5.32117s. Carbon remains dominated by Cu primary
transport. Carbon timing is transport elapsed time including result transfer;
proton native batch-wall timing additionally includes output writes. Do not
compare their absolute rates as a species-only hardware effect.

An additional selected-view/model-reference candidate also has zero joint
spills. Its observed rates6,883/9,583 particles/s provide no useful advantage
over the simpler in-place implementation; it is not adopted. References alone
still spill8/24B, and references plus unreachable-branch cleanup0/8B; those
failed the zero-joint-spill screen and have no accepted throughput claim.

## Matched verification

All eight presets pass the matched two-batch20,000-history/arm regression.
Source/seed manifests, discrete EM/ion/electron/photon audits, ion terminal
counts, and runtime energy/coverage/completion/zero-overflow checks match/pass.
Maximum dose relative L1 is3.6419745e-7 across the sixteen batch pairs. These
small regressions verify implementation behavior and are not timing evidence.

The representative ABBA runs additionally match discrete audits and ion
ledgers. Their >=10%-of-control-maximum voxel masks have100% local3% pass at
zero distance, a sufficient lower bound on matched local3%/0.2mm gamma. This is
neither searched gamma nor new absolute TOPAS acceptance. Carbon dose
acceptance remains deferred. Continuous26 hard-rate/conditional-terminal
calibration approximations remain unchanged.

Evidence: `out/t4_throughput_20261009/electron_result.json`,
`local_spills/*/{spills.json,ptxas.log}`,
`remote/benchmark/`, and `remote/validation/inplace/summary.json`.
Source baseline is `source.tgz`; candidates and the reproducible orchestration
scripts are retained in the same experiment directory.

## Copper scheduling follow-up

A separate candidate extends the existing Cu/open-slit lane prediction from
flat sources to TPS/emittance sources. The prediction changes only execution
order; actual source sampling and every original history/spot seed remain in
the existing transport kernel. The predictor's nominal direction is a work
estimate, not a physical source or a particle rejection decision.

Its matched ABBA campaign uses a1350MHz locked T4 after the electron regression.
The nominal-direction predictor gives pooled ratios1.00631 for carbon HU0
and0.99013 for proton CT lung minibeam. All matched audits pass, but there is
no useful general throughput gain, so it is not adopted. A second candidate
reads angular spreads and position/angle correlations from each actual spot;
the carbon ratio is0.99941 (pairs1.00491/0.99395) and proton ratio1.00214
(pairs0.98930/1.01500). All matched audits pass, but neither species has a
useful stable benefit. Neither ordering candidate is adopted. No fixed emittance or optical parameters are added to
the runtime.

The accepted electron implementation is also checked with narrow/parallel,
wide/divergent and asymmetric/oppositely correlated source ensembles. All three ensembles pass: maximum dose relative L1=4.0581551e-9,
exact discrete audits,100% local3% zero-distance pass on the stated mask.
These are matched implementation checks, not throughput certification.
Evidence: remote/optics/done.json.

## Primary transport follow-up

The compiler reports255 registers and approximately1.6KiB stack for the
primary beamline kernel on sm75. This motivates isolated trials of64/256
thread primary workgroups and moving fixed primary-species/Cu-material index
selection to host startup. Additional candidates reuse the exact pre-step
range already carried by accepted MSC and accumulate integer fluctuation
counters per history before atomic merge. Dynamic energy-dependent Urban contexts, nuclear
competition, geometry, fluctuations, FP32 arithmetic and history identities
remain unchanged. All five candidates retain255 primary beamline registers and4/4B spills.
Workgroup/range candidates retain1640B stack; fixed-index and integer-audit
candidates use1648B. These compiler results alone establish no speedup.
The64-thread carbon pooled ratio is0.99635 (pairs0.99540/0.99731);
proton ratio0.99493 (pairs1.01191/0.97804). Neither provides stable benefit.
The256-thread carbon pooled ratio is0.95673 (pairs0.95682/0.95665);
proton ratio0.99738 (pairs0.99157/1.00332). Both workgroup changes are rejected.
In the first256-thread carbon pair, downstream primary time increases from
about4.25s to4.96s while Cu time changes much less. Fixed primary Cu indices give carbon ratio1.00437 (pairs1.00370/1.00503),
insufficient to justify a useful throughput claim or adoption. Its proton
trial and the range/audit/Urban candidates were not run on T4: the user
requested switching to L4. The current carbon index bracket was allowed to
complete, and the remaining T4 campaign was stopped before migration.
The L4 campaign uses the accepted electron implementation as its baseline;
all primary candidates are rebuilt with the same electron implementation. The separate Urban result-in-place candidate
retains255 registers/4B spills and increases primary stack to1880B; this
compiler result is not grounds for assuming the electron gain transfers to
primary.

The main CUDA target rebuilt successfully after the electron change. All129
files in src/include match the verified in-place candidate;83 retired-interface
checks pass. Evidence: main_build.log, main_candidate_equivalence.json and
main_interfaces.json in the experiment directory.
