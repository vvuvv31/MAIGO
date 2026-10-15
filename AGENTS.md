# MAIGO_clean

- This is a production-only runtime with eight proton/carbon ×
  water/CT × broadbeam/minibeam release presets in `config/`. Custom proton/C12
  configurations may use the generic pipeline with the same validated packages. Do not reintroduce old
  package readers, diagnostic build modes, CPU transport stubs, generic source
  species, TOPAS text-plan replay or DICOM conversion into this runtime.
- RT07575 is a **head** case. Its CT, source, weights, placement, collimator and
  normalization must stay together. Default CT minibeams use a separate lung
  benchmark; do not report them as RT07575 head full-plan validation.
- Electron transport is optional in every configuration, default off. Only
  explicit `enable_electron_transport: true` loads electron/photon packages.
  Preserve EMCNT001 continuous26 ion EM; when enabled, strict26 electron/photon packages;
  proton/carbon-specific birth parameters; current spatial joint; Urban and
  FP32 electron geometry/loss/scoring. No density-node fallback or missing-data
  local-deposition substitution. Water reference is Water_75eV (I=75 eV).
- All eight defaults enable primary ion fluctuations and disable secondary
  energy-loss straggling. Minibeams retain Cu primary fluctuations and disable
  Cu fragment restricted fluctuations; native carbon Cu hard-delta sampling
  remains. RT07575 CT proton broadbeam/minibeam and carbon broadbeam have
  scoped paired validation; carbon minibeam is under verification. Ion step caps
  are forced to 1.0 mm broadbeam and 0.5 mm minibeam in water and CT.
  Minibeam copper is forced to 0.05 mm; these override YAML step-cap values.
  GPU dose atomics stay FP32.
- All eight release presets enable medium nuclear elastic transport with
  projectile-specific all-ion elastic and recoil stopping packages. All four
  minibeams also enable copper-collimator nuclear elastic scattering.
- All four minibeams use compact scheduling. Carbon water retains its TPS CSV
  source, emittance and generic Urban water medium kernel. Preserve physical/
  source behavior in refactors. Generic charged fragment species are required.
- Continuous batches reuse one GPU context/packages. Allocate plan weights
  across the whole campaign using largest remainder, preserve per-spot seeds,
  and never multiply normalized plan dose by the number of batches.
- Before performance experiments read `failed.md`. No performance claim from
  smoke tests. Preserve energy/EM/coverage/completion and zero-overflow audits.
- Broadbeam dose gate: local1%/1mm >99%, search0.1mm. Proton-minibeam package
  replacement: local3%/0.2mm, search0.02mm, decrease ≤1 percentage point against
  matched old baseline. Absolute TOPAS gamma is reported separately. Carbon
  minibeam dose accuracy remains deferred; runtime audits remain required.
- The clean-repository regression is a matched GPU implementation check,
  not new TOPAS acceptance. Explicit calibration approximations of continuous26
  hard rates/conditional terminals remain unchanged.
- Release configurations use `max_secondary_inelastic_generations` and
  current Cu fluctuations (`g4_universal_delta` for proton minibeams and
  `g4_ion_delta` for carbon minibeams); do not restore old aliases or retired scorer keys.
  YAML source identity requires explicit Z/A, energies use `initial_energy_MeVu`,
  and TPS CSV uses `energy_MeV,x_mm,y_mm,weight`. Urban loss/range/MFP uses only
  current packages; no CSV oracle route. Primary reference masses are checked
  against each model's explicit kinematic or nuclear convention. Preserve the
  calibrated C12 A*amu kinematics and FE/copper FP32 arithmetic. Keep live
  `ion_species_*` audits.
- Nuclear primary channel sidecars require explicit boolean `has_support`.
  Nuclear neutrals remain explicit untracked energy sinks; do not restore
  neutron transport/packages or replace their energy with local deposition.
  Enabled electron transport requires electromagnetic photon transport.
  TPS isocenter configuration uses only the supported Z component key.
- Water Z geometry requires explicit `voxel_bins_z` and `voxel_size_z_mm`.
  CT geometry comes from the CCTG header. Do not restore independent YAML
  phantom-length/depth-bin aliases or v1 nuclear-rate audit branches.
- Primary rate readers are fixed to `SCHNRATE` v3. Startup rate metadata uses
  `basename.metadata.json`; do not restore appended `.bin.metadata.json` naming
  or configurable binary magic/version selectors.
- Supported GPU backends are CUDA (`device: cuda`, `nvptx64-nvidia-cuda`)
  and Intel Level Zero (`device: intel`, `spir64_gen`). No vendor aliases or
  CPU/backend fallback. Preserve CUDA queue/device ordering and physics.
  Intel builds require an explicit `CARBON_INTEL_GPU_ARCH` (Arc B580: `bmg-g21`)
  and native AOT device images; do not introduce SPIR-V runtime JIT builds.
  Intel primary kernels pass immutable nuclear/EM views through one device
  context pointer to respect their kernel-argument budget; CUDA keeps by-value captures.
  Temporary host upload buffers must remain alive until asynchronous copies complete.
  The tested Xe/B580 needs `preempt_timeout_us=10000000` for long minibeam
  kernels (driver default 640000). Configure the driver separately; do not
  truncate trajectories or change physics to bypass its preemption timeout.
- CUDA device code must be compiled ahead of time for the GPU that will run it.
  Every CUDA build command must explicitly specify `-DCARBON_CUDA_ARCH=sm_75`
  (adjust for the destination GPU: RTX 2080 Ti `sm_75`, RTX A6000 `sm_86`,
  L4 `sm_89`); direct SYCL compiler commands must specify the corresponding
  `--cuda-gpu-arch`. Do not rely on automatic architecture selection, an empty
  architecture, or portable PTX-only builds. Every GPU execution, smoke test,
  regression and performance experiment must set `CUDA_DISABLE_PTX_JIT=1`.
  Runtime PTX JIT compilation and JIT warm-up runs are prohibited. If a compatible
  native image is unavailable, stop that run and rebuild ahead of time for the
  destination GPU; never enable JIT as a fallback. Build time is excluded from
  cold-start and steady-state throughput measurements.
- Release assets use the physics-module layout documented in
  `docs/data_layout_20261007.md`; keep `data/manifest.json` and preset dependency
  inventories current. Do not restore dated directory aliases or old-path
  lookups. Preserve independent projectile calibrations and package formats.
