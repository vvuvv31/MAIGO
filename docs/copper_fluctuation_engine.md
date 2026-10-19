# Shared unrestricted Cu/air energy-loss engine

[`sample_beamline_fluctuation`](../include/carbon/beamline_fluctuation.hpp)
uses the same Universal/Ion sampling backend as water/CT. The caller provides
kinetic energy, its calibrated rest mass, Z/A, full electronic mean loss,
accepted true path, the full material response and the straggling flag.
There is one full draw with `tcut=Tmax`; no restricted mean or explicit delta
transfer is added. A range-ending step stops without fluctuating.

All four minibeams require `minibeam_em_package_file`, normally
`data/stopping/beamline/unrestricted.bin`. The package covers native Cu and air
for all 324 transported Cu fragment species. Missing species, invalid data or
out-of-domain energies reject the run; no charge-squared stopping fallback is
used. Other primary ions use the generic pipeline with explicit source masses
and matching coverage in every required package, including primary nuclear data.

The package contains native `ComputeElectronicDEDX` and native Ion/Universal
`Dispersion(tcut=Tmax)` per unit true path. Material I/e0 and the active
particle mass/ZA select the sampler parameters. Proton H1 uses Universal;
other ions use the shared Ion/Universal transition. Independently extracted
MSC and elastic fragment tables remain unchanged, including H1 scattering.

All defaults retain primary straggling and disable fragment straggling in both
Cu and air. With fragment straggling off, the entire electronic loss is the
full mean; there is no residual stochastic hard-delta component.
`minibeam_copper_fragment_native_transport` selects native Urban/species
elastic transport independently (carbon on, proton off). The old finite-cut
model/mean/scale keys, stopping-ratio CSV reader and air ionization clock are
removed, without compatibility aliases.

All implemented Cu/air energy-loss queries and range limits use the full
response. Urban supplies scattering MFP, angular/lateral motion and geometric
conversion. Geometry, shared nuclear optical depth, nuclear final states,
source weights and seeds, forced caps and FP32 dose scoring retain their own
roles. Beamline electronic loss and untracked nuclear energy are explicit sinks,
not patient local dose.

See [all-region model and verification](unrestricted_dedx.md) and
[offline extraction](../extensions/beamline_em/README.md). Earlier Cu
refactor/performance reports describe their frozen finite-cut implementations.
