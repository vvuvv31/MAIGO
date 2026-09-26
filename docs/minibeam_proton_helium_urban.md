# Minibeam water proton / helium Urban candidate

Opt in with `minibeam_water_proton_helium_mcs_model: urban_v2`.
The default `inherit` preserves all existing routing. The selected species
are proton (Z=1,A=1) and transported helium isotopes (Z=2); deuterons,
tritons and heavier ions retain their configured model. The dedicated primary
and secondary C12 Urban paths and all copper transport remain unchanged.

Requires research CUDA minibeam homogeneous water, secondary unified EM,
unscaled MCS, a SHA-pinned Urban package with the exact water/cut couple,
and `urban_water_half_width_mm` matching the physical water box. The current
single-spot reference uses 50 mm; this value is not a scattering fit.
The current package production cut is 0.05 mm.

This uses the existing all-ion Urban step implementation: active per-isotope
range/transport-MFP data, persistent track limit state, true/geometrical path
conversion, native ionization proposal clock, voxel/boundary clipping, nuclear
competition, and source-voxel continuous scoring. The configured maximum
true step now bounds these selected secondary tracks. This is a full transport
route comparison, not an angular-kick-only substitution. A 300,000-step
watchdog rejects a selected track that exhausts the budget; the historical
30,000-step secondary cap would be insufficient for >150 mm at 0.005 mm.
Missing material/ion/energy records reject the dose instead of falling back.

`[urban-v2-steps]` reports actual per-isotope steps. In this mode only proton
and helium may have nonzero counts in that table; C12 retains its separate
minibeam counters. FP32 dose scoring is required.

The existing TOPAS full-physics reference uses `g4em-standard_opt4`:
its runtime log reports WentzelVIUni for proton and UrbanMsc for alpha and
GenericIon. Reusing that reference measures the effect of the requested GPU
substitution; it does not make the proton MCS models identical. No TOPAS
physics settings are changed for this candidate.
