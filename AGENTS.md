# MAIGO_clean

- This is a production-only runtime for exactly the eight proton/carbon ×
  water/CT × broadbeam/minibeam presets in `config/`. Do not reintroduce old
  package readers, diagnostic build modes, CPU transport stubs, generic source
  species, TOPAS text-plan replay or DICOM conversion into this runtime.
- RT07575 is a **head** case. Its CT, source, weights, placement, collimator and
  normalization must stay together. Default CT minibeams use a separate lung
  benchmark; do not report them as RT07575 head full-plan validation.
- Preserve EMCNT001 continuous26 ion EM; strict26 electron/photon packages;
  proton/carbon-specific birth parameters; current spatial joint; Urban and
  FP32 electron geometry/loss/scoring. No density-node fallback or missing-data
  local-deposition substitution. Water reference is Water_75eV (I=75 eV).
- All eight defaults enable primary and secondary ion fluctuations. Copper
  primary/fragment fluctuations remain enabled in minibeams. Ion step caps
  are 0.5 mm broadbeam and 0.1 mm minibeam. GPU dose atomics stay FP32.
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
  `clipped_gaussian`; do not restore old aliases or retired scorer keys.
  YAML source identity requires explicit Z/A, energies use `initial_energy_MeVu`,
  and TPS CSV uses `energy_MeV,x_mm,y_mm,weight`. Urban loss/range/MFP uses only
  current packages; no CSV oracle route. Primary reference masses are checked
  against each model's explicit kinematic or nuclear convention. Preserve the
  calibrated C12 A*amu kinematics and FE/copper FP32 arithmetic. Keep live
  `ion_species_*` audits.
- Nuclear primary channel sidecars require explicit boolean `has_support`.
  Nuclear neutrals remain explicit untracked energy sinks; do not restore
  neutron transport/packages or replace their energy with local deposition.
  Electron electromagnetic photon transport remains required.
  TPS isocenter configuration uses only the supported Z component key.
- Water Z geometry requires explicit `voxel_bins_z` and `voxel_size_z_mm`.
  CT geometry comes from the CCTG header. Do not restore independent YAML
  phantom-length/depth-bin aliases or v1 nuclear-rate audit branches.
- Primary rate readers are fixed to `SCHNRATE` v3. Startup rate metadata uses
  `basename.metadata.json`; do not restore appended `.bin.metadata.json` naming
  or configurable binary magic/version selectors.
- The release runtime and build target are CUDA-only (`device: cuda`,
  `nvptx64-nvidia-cuda`). Do not restore vendor aliases, CPU fallback or Intel
  kernel-argument pointer adaptation. Preserve CUDA queue/device ordering.
- Release assets use the physics-module layout documented in
  `docs/data_layout_20261007.md`; keep `data/manifest.json` and preset dependency
  inventories current. Do not restore dated directory aliases or old-path
  lookups. Preserve independent projectile calibrations and package formats.
