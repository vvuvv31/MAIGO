# Eight-preset production runtime refactor

Location: `/mnt/sdb/wuwei/MAIGO_clean`. Source baseline: the current working tree
of `/mnt/sdb/wuwei/MAIGO_minibeam`, with the frozen round-4 executable as the
implementation-regression control. The original working tree was not edited.

## Scope and structure

This is a standalone source/data copy, not a linked worktree. Assets were copied
with reflink where supported; there are no symlinks to the original directory.
Internal hardlinks preserve shared electron projectile aliases only within the
new directory. Normal compilation and execution do not require the original.

The eight YAMLs are the supported runtime configuration surface. The CLI,
source preparation, continuous batches and output handling are separated into
`cli.cpp`, `run_session.cpp`, `batch_run.cpp`, `source_pose.cpp` and `io.cpp`.
CMake explicitly lists production sources and builds one SYCL GPU executable.
The production transport, fragments, nuclear sampling, Urban/FE equations,
continuous26 EM, strict26 electrons/photons and mandatory audits are preserved.

## Removed code and inputs

- Archived EM/CSV readers, density-node loaders, old interfaces and automatic
  migration/fallback handling.
- Debug/research compile branches, CPU stub and alternative scoring precision.
- TOPAS text-channel/native plan replay, sequential spot replay, DICOM conversion,
  generic primary-source species and old CLI aliases. Current TPS CSV geometry
  and internally generated nuclear fragments remain.
- Unused LET/CSV/research scorers, stale ion manifest helpers, inactive native
  PBS branches, loss-query probes and confirmed uncalled device helpers.
- Old archived packages and the unused LET delta-electron fraction input.

Runtime package indices and physical/numerical coverage checks are retained;
there is no SHA256 implementation. Required physics is not removed merely
because a helper was previously named Candidate or legacy: the active nuclear
rate cache is now named HadronicRateCache, and the current copper fluctuation
law remains unchanged.

## Preset identity and physics limits

RT07575 CT broadbeam is a head case. The default CT minibeam YAMLs describe a
separate lung benchmark; this refactor does not include the native TOPAS-text
head-minibeam full-plan runner. All four minibeams now use compact scheduling;
carbon water retains its TPS CSV source, emittance and generic Urban water
medium kernel. Ion step caps remain
0.5 mm broadbeam and 0.1 mm minibeam. All ion fluctuations remain enabled.

Uniform-water broadbeam has an empty low-density electron selection and uses
local deposition. CT broadbeam retains density <=0.1 g/cm3 plus 5 mm halo
tracking for primary ionization. Minibeams retain independent FP32 electron
transport with group32, and Schneider spatial joint. Water is Water_75eV.
Continuous26 hard-rate/conditional-terminal calibration approximations are
unchanged. Carbon-minibeam dose accuracy is still deferred.

## Verification

The final production binary builds successfully (`build_single_entry_fix.log`).
All eight presets passed the final matched two-batch GPU regression against the
frozen round-4 executable: **20,000 primaries per scene per arm**, matching
transport audit rows and byte-identical primary batch/source files. Maximum
relative dose L1 was **2.9834027289e-07**. Quality, energy/EM/coverage/completion checks
passed with zero queue overflows and finite nonnegative scored doses.

All **23** retired CLI/YAML interface rejection checks passed. Three additional
10,000-primary single-run smoke cases passed: proton water broadbeam, proton CT
broadbeam TPS CSV and carbon water minibeam TPS CSV. The single-run TPS smoke
exposed an entry-only total-energy versus per-nucleon-energy field error; the
entry now obtains energy from the generated primary batch, as continuous
batches already did. The final regression above was rerun after that fix.

Evidence:
- `validation_closeout/summary.json`: final eight-preset paired regression.
- `single_smoke/summary.json`: single-run source/quality smoke.
- `interface_check.json`: unsupported-interface rejection checks.
- `package_check.json`: strict26 electron bundle verification.

Package and input inventories: `package_check.json`, `preset_dependencies.json`,
`copied_inputs.json`, and `copied_source.json`. The source inventory describes
the final live tree, not an unmodified historical copy.

Small-history matched GPU tests establish implementation consistency only.
They are not new TOPAS gamma, full-plan dose acceptance or throughput benchmarks.

The subsequent compact-only minibeam change is documented in
[compact_20261007.md](compact_20261007.md), with a fresh eight-preset matched
GPU regression against the pre-change executable.
