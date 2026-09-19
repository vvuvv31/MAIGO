# Carbon minibeam reference

The GPU geometry now uses the same cylindrical Copper outer body as TOPAS.
The 15 finite rectangular air slits are evaluated inside that cylinder; this
matters for scattered primaries and charged fragments that leave the 50 mm
slit length but remain inside the 60 mm Copper radius.

| parameter | value |
|---|---:|
| collimator centre upstream of isocentre | 30 mm |
| cylindrical radius × thickness | 60 × 60 mm |
| slit width × length × thickness | 0.5 × 50 × 60 mm |
| slit centre spacing | 3.6 mm |
| slit centres | -7 … +7 |
| water entrance downstream of isocentre | 60 mm |

`run1.txt` uses INCLXX for ion inelastic reactions and the local
`CarbonIonElasticPhysics` TOPAS extension for general-ion elastic scattering.
The extension is the thin `G4IonElasticPhysics` wrapper installed in
`/home/wuwei/topas/extensions`; no project-specific elastic model is substituted.
`g4h-elastic_HP` remains for the nucleon/HP part of the list.

The matching GPU build is optional:

```sh
cmake -S . -B build/minibeam -G Ninja \
  -DCARBON_ENABLE_SYCL=ON -DCARBON_ENABLE_MINIBEAM=ON \
  -DCARBON_DOSE_FP32=ON -DCARBON_DOSE_FP64=OFF
cmake --build build/minibeam -j
```

`config/beam_minibeam_center_absorbing_10k.yaml` is the small geometry/dose
smoke case.  Copper is a black absorber in this first isolated path: histories
that touch Copper are included in `beamline_removed_energy_MeV`, while rays
that remain in one slit enter the unchanged water/CT transport.  INCLXX and
general-ion elastic extraction data are therefore reference inputs for the
subsequent Copper-interaction path, not silently approximated by this smoke
mode.

`run_field3cm.sbatch` is the initial TOPAS dose reference requested for this
case.  It crops the supplied 2 mm spot grid to -15 .. +15 mm in x/y (256
spots), runs 1000 histories per spot (256,000 total), and uses 192 threads.
All generated data and logs are written below `/mnt/sda/wuwei`.

The first local reference completed as Slurm job 6037: 256,000 histories in
61.49 s execution time.  Its output is
`/mnt/sda/wuwei/minibeam_field3cm_6037/dose.bin`.  The matching optional GPU
absorber run uses `config/beam_minibeam_field3cm_absorbing_256k.yaml`; this is
a geometry/normalization comparison, not yet a replacement for Copper EM,
elastic, and INCLXX product transport.

The optional Copper database extraction is separate from the dose run:

- `extract_copper_physics.txt` exports C-12 stopping/rate tables and captures
  authoritative INCL++ final states in homogeneous `G4_Cu`.
- `extract_copper_physics.sbatch` runs 256k histories with 192 threads and
  writes raw CINEL02 records below `/mnt/sda/wuwei`; normal dose runs never
  enable the capture wrapper.

Local extraction job 6039 produced 39,284 `NNDiffuseElastic` events and
41,721 C-12+Cu INCLXX interactions (754,838 products). The CINEL03 energy
domain is 2.92--179.17 MeV/u with a 0.336 MeV/u maximum node gap. The compiled
tables are under `/mnt/sda/wuwei/minibeam_copper_extract_6039/compiled`.

The initial water-nuclear-off diagnostic completed at 70.7k histories/s but
exposed a configuration error: it produced no water nuclear interactions or
secondary tracks, overestimated the 65--72 mm dose by 28.0%, and had no dose
beyond 80 mm.  It is retained as
`config/beam_minibeam_field3cm_copper_water_off_256k.yaml` and must not be used
as the reference result.

The corrected 256k GPU Copper run enables water nuclear and secondary
transport.  On the RTX 2080 Ti it completed at 61.7k histories/s with queue
overflow zero and relative energy residual 1.09e-7.  Against TOPAS job 6037,
its dose sum ratio is 1.0144, 2-D Pearson correlation is 0.98046, IDD Pearson
is 0.99880, IDD L1/TOPAS is 1.76%, and the depth-peak positions are 70.45 mm
(GPU) versus 70.25 mm (TOPAS).  This remains a research candidate: neutral
INCLXX products are currently retained in the explicit beamline energy sink
rather than transported into the phantom, and 1.62e6 MeV of water nuclear
product energy remains in the audited out-of-scope bucket.

The absolute-dose comparison plots and machine-readable peak/valley results
can be reproduced without a fitted dose scale:

```sh
MPLCONFIGDIR=/tmp/maigo-minibeam-mpl \
python3 benchmark/carbonminibeam/compare_gpu_topas_dose.py \
  --topas /mnt/sda/wuwei/minibeam_field3cm_6037/dose.bin \
  --gpu out/beam_minibeam_field3cm_copper_256k/dose.raw \
  --output-dir out/beam_minibeam_field3cm_copper_256k/comparison_topas
```

For Copper-fragment diagnostics, set
`minibeam_fragment_phase_space_output_file` to write every charged INCL++
product accepted into the water secondary queue.  The output is opt-in and
contains source history, isotope, kinetic energy, transverse position, and
direction.  Compare it with the mixed-species TOPAS water-entrance scorer via:

```bash
python3 benchmark/carbonminibeam/compare_fragment_phase_space.py \
  --gpu out/minibeam_fragment_phase_diag_e250_256k/fragment_phase.csv \
  --topas /mnt/sda/wuwei/minibeam_phase_scan_e250_6255_1/output/water_entrance.phsp \
  --output out/minibeam_fragment_phase_diag_e250_256k/comparison_topas.json
```

The comparison defaults match the TOPAS scorer: GPU tracks are projected
0.02 mm upstream from the water entrance and clipped to ±70 mm in both
transverse coordinates. Radial angles use
`atan2(p_transverse,p_longitudinal)` rather than treating direction cosines
as small angles.

Primary and fragment Copper MCS scales are intentionally separate.  The
slab-constrained Fermi--Eyges/tail primary model uses
`minibeam_copper_mcs_scale`, while fragments retain the independently
controlled `minibeam_copper_fragment_mcs_scale`.  Fragment straggling is also
separately gated and remains disabled until a per-species spectrum requires
it.

Fragment nuclear packages must preserve the actual reference-model selection.
For the current TOPAS list, proton uses Binary Cascade from 0--6 GeV, while
d/t/He and GenericIon use INCL++ below 3 GeV/u, with overlapping high-energy
FTFP/QGSP transitions. `compile_copper_fragment_cascades.py` rejects raw
campaigns whose recorded Geant4 `model_name` differs from the declared model
before combining their projectile-indexed CINPKG04 nodes.

Set `minibeam_copper_fragment_cascade_generations` to 1--3 and point
`minibeam_copper_inclxx_file` at the combined package to replay real
fragment+Cu final-state generations. Zero keeps terminal absorption. Products
at the configured cap receive full EM transport through the remaining
collimator but no further nuclear reaction. Runtime output reports both
terminal integrated optical depth and the directly interpretable per-track
sum `1-exp(-tau)` for generation-convergence checks.
Copper queue generations are independent of water generations; survivors
enter water at generation zero.  Reaching the configured water nuclear cap
suppresses only later nuclear hazards and never converts an above-cutoff
charged product into local dose: terminal EM transport remains active.
Runtime output reports strict lookup misses in the order
`projectile/target/below/above/gap/empty`; do not replace a missing domain with
endpoint clamping.  `extract_copper_fragment_light_low_energy.sbatch` is the
small p/d/t coverage job used after the broad pilot exposed low-energy gaps.

Peak and valley curves use a 1 mm moving depth slab and the median of the
central nine peaks/eight valleys. At 0.5, 35, and 70.25 mm, respectively, the
corrected GPU/TOPAS peak ratios are 1.007, 0.923, and 0.936; valley ratios are
1.095, 1.076, and 1.088; and PVDR ratios are 0.920, 0.858, and 0.860. At 80 mm
the peak ratio is 0.997 and the valley ratio is 1.101.  The remaining spatial
error is therefore predominantly excess Copper-history dose in minibeam
shoulders/valleys, not range or missing water nuclear transport.

The matched 1,024,000-history convergence run uses
`run_field3cm_1m.txt`/`run_field3cm_1m.sbatch` and
`config/beam_minibeam_field3cm_copper_1m.yaml`. TOPAS job 6040 used 192
threads and completed in 237.28 s; the RTX 2080 Ti GPU completed in 6.21 s
(164.8k histories/s). Increasing statistics reduced 2-D L1/TOPAS from 17.95%
to 10.30% and lateral-integral L1 from 7.69% to 5.58%, while 2-D Pearson rose
from 0.98046 to 0.99127. The remaining valley excess did not converge away:
the all-depth peak/shoulder/valley ratios are 0.991/1.068/1.212, and the
0.5/35/70.25 mm PVDR ratios are 0.841/0.750/0.666. Thus the high-resolution
valley result is a systematic Copper-history transport difference rather than
the visible 256k Monte Carlo noise.

A four-seed ensemble was then run at 1,024,000 histories per seed
(4,096,000 histories per engine). Three additional TOPAS jobs ran concurrently
as Slurm array 6041; no cluster nodes were needed. The summed comparison gives
2-D Pearson 0.99426, 2-D L1/TOPAS 6.59%, lateral-integral Pearson 0.99932,
lateral-integral L1 4.45%, IDD L1 2.66%, and a total-dose ratio of 1.0253.
Across the four independent seed pairs, the total-dose-ratio 95% confidence
interval is 1.0213--1.0294 and lateral L1 is 5.10--5.72%. At 70.25 mm the
valley-ratio 95% interval is 1.397--1.481 and the PVDR-ratio interval is
0.627--0.694; at 35 mm they are 1.154--1.269 and 0.716--0.908. Additional
histories would narrow these intervals but cannot explain the systematic
valley excess. The next diagnostic should separate Copper-scattered primary
dose from Copper-INCLXX charged-product dose at the water entrance.

That diagnostic exposed three implementation/configuration gaps.  The 3 cm
configs had overridden the independently calibrated Copper Highland scale
(`0.785`) with `1.0`; the parsed low-energy primary-C12 MCS correction was not
wired into the water kernel; and Copper INCLXX charged products used only a
`C-12 stopping * Z^2/36` fallback with no Copper MCS.  The corrected path uses
the existing Geant4 11.3.2 isotope-specific Copper stopping table, transports
those products with Copper MCS, applies the frozen primary low-energy MCS
response, and samples general-ion-elastic outcomes from a local energy window
instead of making transfer a deterministic function of collision energy.

With the same four TOPAS seeds and 4,096,000 histories per engine, the summed
corrected result is in
`out/beam_minibeam_field3cm_copper_4m_physics_complete`: total GPU/TOPAS dose
is 1.00593, 2-D Pearson is 0.99485, 2-D L1/TOPAS is 5.13%, IDD L1 is 1.62%,
and lateral-integral L1 is 2.13%.  At 0.5/35/70.25 mm the peak ratios are
1.035/1.031/0.999 and valley ratios are 1.047/0.911/1.120.  The Bragg-depth
difference remains +0.20 mm.  The four paired-seed total-dose-ratio interval is
1.00357--1.00832; the remaining local valley spread is still statistics
sensitive, but the merged distal valley excess remains about 12%.  Copper
product stopping/MCS costs roughly 8--10% throughput on the RTX 2080 Ti
(about 150k histories/s versus 163k histories/s before that transport).

The remaining Copper-fragment over-survival was then fixed by applying the
existing Geant4 11.3.2 isotope-specific inelastic cross sections while each
INCLXX charged product traverses the rest of the collimator.  A sampled second
reaction terminates that product because tertiary final states are not yet in
the minibeam package.  Charged survivors per 1.024M histories fell from about
89k to 67k.  A range-derived primary stopping correction of 1.0029 (the prior
GPU peak was 0.20 mm too deep over a 70 mm range) aligns both Bragg peaks at
70.35 mm.

The final four-seed result is in
`out/beam_minibeam_field3cm_copper_4m_final`: GPU/TOPAS total dose is 0.99799,
2-D Pearson is 0.99522, 2-D L1/TOPAS is 4.80%, IDD Pearson is 0.99979, IDD L1
is 0.66%, and lateral-integral L1 is 2.12%.  At 0.5/35/70.25/80 mm the valley
ratios are 1.013/0.890/1.120/1.035 and the peak ratios are
1.035/1.031/1.007/1.001.  Scanning the Highland transition at 80, 100, 120,
and 180 MeV/u, low-energy scales 0.05, 0.10, and 0.20, a smoothstep ramp, and
a paired global-1.05/low-0.05 correction found no further joint improvement:
changes that fill the 35 mm valley propagate downstream and worsen the Bragg
valley.  The residual therefore requires a non-Gaussian/correlated scattering
model or component-resolved phase-space data, rather than another scalar
Highland fit.

The fixed-optics energy scan keeps the 179.17 MeV/u beam-model widths,
divergences, correlations, and 1.2% energy spread, changing only the C-12 total
kinetic energy to 1800, 3000, and 3600 MeV.  It is therefore a transport
isolation test, not a claim about measured accelerator optics at 150, 250, or
300 MeV/u.  `run_energy_scan.sbatch` uses a common 250 mm water depth with
0.25 mm depth bins and 0.1 mm lateral bins.  Each energy has a separately
extracted Copper elastic/INCLXX package so no event lookup is made above its
sampled energy domain.

Slurm arrays 6239 and 6240 completed the extraction and 256,000-history TOPAS
runs.  For 150/250/300 MeV/u, respectively, GPU/TOPAS total-dose ratios are
0.99382/1.01585/1.01762, IDD L1 errors are 2.18%/2.09%/3.54%, and
lateral-integral L1 errors are 7.57%/5.97%/5.06%.  The 150 and 250 MeV/u Bragg
depths agree exactly at the 0.25 mm grid resolution (51.625 and 124.375 mm).
At 300 MeV/u the raw distal maxima are 169.375 mm in TOPAS and 167.375 mm in
GPU; 1 mm smoothing reduces the difference to 1.5 mm and a local curve
cross-correlation gives an approximately 0.75 mm upstream GPU shift, so more
histories are required before fitting another range correction.  The combined
summary is in `out/beam_minibeam_energy_scan_256k`, with per-energy overview,
depth-dose, peak/valley, PVDR, and lateral-profile products below the matching
`out/beam_minibeam_field3cm_copper_e*_256k/comparison_topas` directories.

The corrected multi-energy validation uses `EMRangeMax=6 GeV`, transports
the upstream and post-collimator air path, terminates Copper tracks that reach
the kinetic-energy cutoff instead of leaking cutoff-energy primaries into the
water, and applies a 3 x 3 cm field phase-space calibration of the surviving
C-12 Copper loss.  At the water entrance the 150/250/300 MeV/u GPU primary
counts are 26,399/31,146/34,591 versus TOPAS
26,425/30,878/34,557.  The Copper-touched primary mean energies are
1331/2083/2395 MeV versus 1333/2084/2396 MeV.

The final 1.024M-history comparison is under
`out/beam_minibeam_energy_scan_1m_gpu/e*/comparison_topas`.  For
150/250/300 MeV/u, GPU/TOPAS total-dose ratios are
0.99366/1.00688/1.00218, IDD L1 errors are 1.08%/1.80%/2.04%,
lateral-integral L1 errors are 3.86%/3.68%/3.71%, and absolute Bragg-depth
differences are 0.25/0.25/0.50 mm.  At the Bragg depth the
peak/valley/PVDR ratios are respectively 1.032/1.022/1.010,
0.968/1.213/0.798, and 0.997/1.113/0.896.  The remaining high-energy error is
localized mainly to the valley shape; a Copper INCLXX +/-0.25 MeV/u local
event sampler did not improve it and was reverted.  Secondary transport uses
the shared Unified EM kernel in all these runs, with zero missing-domain audit
events and zero queue overflow.

The converged comparison uses five independent 2M-history TOPAS shards per
energy (10M total) and 10,000,128 GPU histories.  The combined TOPAS dose is
under `/mnt/sda/wuwei/minibeam_energy_10m_combined`; plots and metrics are
under `out/beam_minibeam_energy_scan_10m_gpu/e*/comparison_topas`, with the
cross-energy regional analysis in
`out/beam_minibeam_energy_scan_10m_gpu/analysis_10m.json`.  For
150/250/300 MeV/u, respectively, GPU/TOPAS total-dose ratios are
1.00155/1.00475/1.00433, 2-D L1 errors are 3.64%/5.33%/4.99%, IDD L1 errors
are 0.70%/1.69%/2.18%, and lateral-integral L1 errors are
1.85%/2.67%/2.54%.  The smoothed R80 differences are -0.014/-0.044/+0.018 mm,
so the residual is not a range error.

Higher statistics resolves two systematic components.  At the TOPAS Bragg
depth, the 150/250/300 peak ratios are 1.004/0.942/0.940 and valley ratios are
1.098/1.235/1.154, giving PVDR ratios 0.915/0.763/0.814.  The sign changes
with depth: entrance valleys are low by 18%/12%/10%, whereas the high-energy
Bragg valleys are high.  Together with a roughly 10% outer-field deficit at
250--300 MeV/u, this is consistent with a too-Gaussian, incorrectly
energy-dependent scattering kernel rather than a single Copper Highland
normalization.  In particular, the configured water correction transitions
at 180 MeV/u and linearly scales the per-step Highland core toward 0.20 at
zero energy, matching the depth at which the 250/300 MeV/u discrepancy grows.
The existing failed scalar scans must not be repeated; the next scattering
work needs differential angular/lateral-displacement data or a non-Gaussian
correlated kernel.

Separately, dose more than 20 mm beyond the Bragg peak is high by
3.8%/13.8%/12.8%, while dose outside |x|=25 mm is low by
14.3%/10.1%/10.7%.  These regional biases are far above the five-shard TOPAS
standard errors and point to nuclear-fragment final-state/secondary transport
and wide-angle-tail modeling, not missing secondary Unified EM (its audit is
already clean).  From 1.024M to 10M, 2-D L1 falls by roughly one half, but the
250/300 IDD and Bragg peak/valley biases persist; they are therefore no longer
plausibly attributable to Monte Carlo noise.

The follow-up component-resolved diagnostic used 1,024,000 histories at 250
and 300 MeV/u.  TOPAS scored primary C, secondary C, B, Be, Li, He, Z=1,
other charged, neutral-origin, and unclassified dose in one run; the GPU used
its optional charged-origin voxel scorer.  The category sums close to total
dose on both engines.  Primary dose already agrees in normalization and IDD
(GPU/TOPAS total 1.005/1.003 and IDD L1 0.70%/0.79%), but its 2-D L1 remains
10.6%/11.2%; at the Bragg depth primary peak is 0.959/0.992 while primary
valley is 1.288/1.245.  Charged fragments have a separate distal excess of
19.2%/17.0%.  TOPAS neutral-origin dose is only 0.86%/1.00% of total, so it
cannot explain that excess.  Reproducible analysis and plots are under
`out/beam_minibeam_component_diagnostics`, generated by
`analyze_phase_space.py` and `compare_origin_components.py`.

### Water-entry primary replay

`prepare_water_entry_replay.py` defaults to filtering parent-0 C12 tracks from
a TOPAS water entrance phase space and writes matched TOPAS and GPU inputs.
`--selection charged-ions` instead emits one independently runnable GPU/TOPAS
source pair per `(Z,A)` plus a normalization manifest; this is required because
one GPU source run has a fixed projectile species. These isotope-split GPU runs
currently use the primary kernel and are therefore diagnostic inputs, not yet
a production-equivalent fragment replay through the secondary kernel. The GPU CSV uses
the explicit per-row source pose supported by `TpsSourcePlan`; TOPAS and GPU
are compared per surviving C12 history.  Do not ask multithreaded TOPAS to
append the upstream empty histories: worker-local phase-space cycling can
replay the non-empty records repeatedly.  Apply the common survival fraction
only after both downstream runs if incident-history normalization is needed.

The first 250 MeV/u replay used 30,878 C12 tracks from 256,000 original
histories.  With the production water corrections, GPU/TOPAS total dose is
0.9973, IDD L1 is 0.645%, and entrance peak/valley/PVDR ratios are
1.037/0.998/1.039.  At the TOPAS Bragg depth they are
1.017/0.808/1.258.  Keeping the rare-wide MCS tail but disabling the empirical
low-energy Highland-core scaling changes the Bragg ratios to
0.965/0.917/1.052 and distal PVDR to 1.005.  The same change worsens the full
Copper-plus-water 256k Bragg valley from 1.141 to 1.232, proving that the
current Copper input and water low-energy correction compensate each other.
The production multi-energy configurations therefore remain unchanged until
the Copper joint phase space is repaired.

The paired collimator-exit replay uses 30,895 TOPAS parent-0 C12 tracks and
transports the 59.97 mm air gap before the same water calculation. TOPAS loses
17 C12 tracks in air; GPU writes 30,890 tracks at the water scoring plane.
For the 30,878 event IDs present in both outputs, GPU-minus-TOPAS mean energy
is only 0.0019 MeV (RMS 0.163 MeV), while the transverse position RMS
differences are 0.0089/0.0144 mm.  Direction-cosine RMS differences are
0.000277/0.000388, small compared with the 9.6/11.2 mrad total angular widths.
Thus straight propagation and mean air loss agree; GPU omits the very small
air nuclear-loss/MCS component, but it is not the source of the entrance
minibeam discrepancy.

The exit-replay GPU/TOPAS dose ratio is 0.9960 and IDD L1 is 0.802%, compared
with 0.9973 and 0.645% for water-entry replay.  Exit-versus-water replay maps
within each engine differ by 17% in raw 2-D L1 and about 4.1% laterally at this
30.9k-primary statistic, demonstrating that individual selected-depth PVDR
changes are noise dominated.  The stable global/IDD result shows that the air
gap adds no material dose discrepancy.  Consequently, the full-run water
entrance mismatch is upstream in the Copper transport, while the independently
observed Bragg-region replay mismatch remains downstream in the water MCS
model.

The initial accepted water-primary correction was a step-size-aware Gaussian
core/rare-tail split.  Tail probability scales as `strength * step/X0`, tail
width as `width/sqrt(step/X0)`, and the core variance is reduced so the
projected second moment is unchanged.  It is default-off; the first 250/300
MeV/u validation used strength 0.5 and width 0.85, while the 150 MeV/u baseline
was already matched and remains unchanged.  Against the
10M TOPAS reference, 250 MeV/u 2-D/lateral L1 improve from 5.33%/2.67% to
4.53%/2.00%, and Bragg peak/valley/PVDR improve from
0.942/1.235/0.763 to 0.994/1.075/0.924.  At 300 MeV/u, 2-D/lateral L1 improve
from 4.99%/2.54% to 4.37%/1.98%, and Bragg peak/valley/PVDR improve from
0.940/1.154/0.814 to 0.983/1.058/0.929.  Total dose and IDD are unchanged;
all final runs have zero Unified-EM missing-domain events and zero queue
overflow.  The remaining main error is species-dependent nuclear production:
B/Li/He distal dose is high while Z=1 dose is low, so it must not be fitted by
another MCS scalar.

A later 1,202,572-primary water-entry replay corrected the earlier phase-space
filter so that every C12 charge state was retained.  It showed that the 0.85
model matches primary fluence modulation at 40 mm but over-smears it from
80 mm onward: the 3.6 mm fundamental GPU/TOPAS ratios at
40/80/100/115/124 mm were 0.997/0.968/0.948/0.947/0.953.  A single
Geant4-moment-constrained refinement to width 0.90 changed those ratios to
0.997/0.972/0.959/0.965/0.978.  In the same high-statistics dose replay,
Bragg peak/valley/PVDR changed from 0.978/1.034/0.946 to
0.982/1.009/0.973; 2-D and lateral-integral L1 changed from 3.36%/1.17% to
3.24%/1.00%.  Same-seed 256k complete Copper-plus-water checks improved both
250 and 300 MeV/u 2-D/lateral L1 and Bragg PVDR, with zero Unified-EM misses,
no queue overflow, and energy residuals below 1.8e-5.  The formal 250/300
validation configs therefore use width 0.90; 150 remains tail-off.

The nuclear follow-up found that the minibeam energy configs had omitted
`cinel02_max_secondary_inelastic_generations`, whose default zero disables all
fragment reinteractions.  The documented production value 2 is now explicit
in the 150/250/300 MeV/u configs.  In the 1.024M component comparison it moves
250/300 MeV/u total distal ratios from 1.146/1.131 to 0.992/0.989 and charged
fragment distal ratios from 1.192/1.170 to 1.032/1.023.  Species-integrated
ratios become C 0.874/0.894, B 0.950/1.002, Be 1.043/0.915, Li 1.042/0.947,
He 1.026/1.012, and Z=1 0.991/0.983.  Total 2-D L1 also improves modestly from
10.28%/9.72% to 9.69%/9.37%.  Results and plots are under
`out/beam_minibeam_secondary_generation2`; the single-generation cap is not a
valid substitute because it locally deposits charged products at the terminal
generation.  A 1.024M 150 MeV/u regression retains 1.30% IDD L1 and a Bragg
PVDR ratio of 0.992, with zero queue overflow and zero Unified-EM domain misses.

The final 10,000,128-history generation-2 validation is under
`out/beam_minibeam_secondary_generation2_10m`.  At 150/250/300 MeV/u the
GPU/TOPAS total-dose ratios are 0.9955/0.9902/0.9887, 2-D L1 errors are
3.66%/3.92%/3.80%, IDD L1 errors are 0.95%/1.00%/1.14%, and lateral-integral
L1 errors are 1.80%/1.80%/1.63%.  Bragg peak/valley/PVDR ratios are
1.003/1.102/0.910, 0.988/1.056/0.936, and 0.966/1.031/0.938.  Compared with
the accepted primary-MCS-only result, the high-energy 2-D errors fall from
4.53%/4.37% and the distal nuclear excess is removed.  RTX 2080 Ti throughput
is 314.3k/149.7k/95.9k histories/s; the lower high-energy throughput is the
cost of transporting the previously omitted secondary cascade.  All three
runs have zero queue overflow, zero Unified-EM missing-domain events, and
normal energy residuals.

The water-entry follow-up records optional per-primary GPU phase space and
compares the joint slit position, energy, and direction distribution with
TOPAS.  Direct primaries agree at about the one-percent level, while the old
GPU discrete-Copper-elastic path produced an excessive angular tail.  More
importantly, GPU valley primaries were too energetic: at 150/250/300 MeV/u the
mean valley-energy ratios were 1.180/1.146/1.078.  Folding the phase space with
the Geant4 water stopping-power table changes the valley/peak proxy ratios to
0.945/0.850/0.906, explaining most of the entrance dose deficit despite the
similar or excessive GPU valley fluence.  This diagnostic is implemented in
`analyze_phase_space.py`; raw and summarized results are under
`out/minibeam_match_fix_phase_compare_joint`.

A separate TOPAS 250 MeV/u phase-space run removed only
`CarbonIonElasticPhysics`.  Relative to the full list, water-entry q99 changed
by 0.002%, valley mean energy by 0.20%, and the stopping-weighted valley proxy
by 1.7%, so General Ion Elastic has negligible influence at the precision of
this 256k process-isolation run.  GPU discrete elastic is consequently
default-on for backward compatibility but disabled in the 150/250/300
validation configs.  At 1.024M histories this leaves 2-D L1 essentially
unchanged (9.406/9.722/9.461%) while reducing IDD L1 to
1.289/0.571/0.971%.  After correcting the analysis grid to the actual
0.1 mm lateral and 0.25 mm depth spacing, the 150/250/300 MeV/u Bragg
peak/valley/PVDR ratios are 1.030/1.029/1.001,
1.042/1.036/1.005, and 1.046/0.995/1.051.  The carrier-dose TOPAS diagnostic
also shows that electron transport cannot justify a fixed lateral delta-dose
smearing: electron PVDR divided by total PVDR is 0.602/0.898/0.916 at the
0.375 mm entrance bin, rises to 1.701/1.385/1.269 at 4.875 mm, and falls to
0.063/0.448/0.573 at the Bragg depth for 150/250/300 MeV/u.  Reproducible
inputs and analysis are `run_energy_scan_electron_components.txt`,
`run_phase_space_no_general_ion_elastic.txt`, and
`analyze_electron_components.py`.

Dose analysis now reads and cross-checks the GPU `DimSize`, `ElementSpacing`,
and `Offset` from the adjacent MHD header against the TOPAS X/Y bin metadata
in `.binheader`.  Missing metadata requires explicit lateral and depth
spacing, and a conflicting override is rejected.  This prevents the former
silent 0.1 mm depth default from relabelling a 0.25 mm grid and changing the
physical width of the peak/valley smoothing window.  Metrics generated before
this check must be trusted for selected-depth quantities only when their
stored grid reports `depth_spacing_mm: 0.25`.

## Structural Copper transport follow-up

The next transport revision removes three compensating approximations without
retuning the water model:

- the outer Copper body is the TOPAS cylinder (`R=60 mm`) rather than a
  `120 x 50 mm` transverse box;
- the primary nuclear clock is a persistent optical depth and a reaction is
  placed inside the current step, so INCL++ products traverse the actual
  remaining Copper thickness;
- Copper stopping is applied on the energy history used by MCS and the nuclear
  rate.  The old post-exit survivor energy rewrite has been removed.

The historical survivor-only loss factors cannot be reused as local stopping
factors.  Doing so at 150/250/300 MeV/u increased the Copper-touched survivor
ratios to `1.030/1.101/1.095` relative to TOPAS by restoring too many very-low
energy tracks.  With the extracted Geant4 table unscaled (`s(E)=1`), those
ratios are `0.955/1.009/0.988`; at 250 MeV/u the touched angular-RMS, valley
fluence, and valley stopping-proxy ratios are `1.015/0.983/1.024`.  The exact
cylinder also brings the 250 MeV/u direct-primary count to `18785` versus
TOPAS `18760`.

The remaining touched-primary mean-energy ratios are `0.955/0.971/0.981`.
The reduced homogeneous slab matrix (150/250/300 MeV/u at 1/10/60 mm) then
showed that the unscaled table is already correct: where primary C12 exits,
TOPAS mean energy differs from direct table integration by only
`0.013--0.341 MeV`.  Therefore no local `s(E)` is retained.  The full-slit
conditional-energy residual must be resolved through path length, scattering,
and survival correlations rather than another stopping correction.

The matrix was deliberately reduced to 48 CPUs/24 GB, with one failed case
rerun alone on 16 CPUs/8 GB.  A low-cost RTX 2080 Ti solid-Cu check then used
256k histories at 250 MeV/u.  For 1/10 mm Copper, primary survival was
`0.99970/0.99933` and mean exit energy was `1.00006/1.00130`, while the
GPU/TOPAS Fermi--Eyges moment ratios `(A0,A1,A2)` were
`(0.491,0.555,0.730)/(0.700,0.706,0.724)`.  Radial-angle quantile ratios
`q68/q95/q99/q99.9` were `0.910/0.854/0.765/0.538` at 1 mm and
`0.874/0.853/0.824/0.688` at 10 mm.  Thus nuclear survival and stopping are
already constrained, but the Copper MCS core is too narrow, its tail is too
light, and the thin-slab displacement--angle correlation is missing.  A
single rescale of the current `0.785` Highland width cannot repair all three
moments and the tail simultaneously; the next candidate must use a
step-invariant scattering power with correlated displacement and an
independently constrained tail.

The optional `fermi_eyges_tail` Copper model now implements that follow-up.
At 250 MeV/u its GPU/TOPAS `(A0,A1,A2)` ratios are
`(1.087,1.065,1.023)` for 1 mm and `(1.019,1.013,1.003)` for 10 mm; all
q68--q99 ratios are within about 2%, and q99.9 is `0.997/1.000`.
Changing the maximum step from 0.25 to 0.05 mm changes the 10 mm observables
by at most 1.3%.  The legacy `highland` path remains the default rollback.

Copper primary straggling is also wired into the kernel, using the Copper
`Z/A` condensed-loss variance and midpoint stopping integration.  At
250 MeV/u the 1/10 mm GPU exit mean/std values are
`2913.873/1.744` and `2051.947/6.142 MeV`, versus TOPAS
`2913.891/1.708` and `2052.384/6.001 MeV`.  The three formal 256k energy
configs explicitly enable both new Copper options; fragment straggling is not
yet implemented.

## Identity-preserving water-entry replay

`prepare_water_entry_replay.py --selection charged-ions` now advances every
forward charged ion from the actual TOPAS scoring plane to the configured
water boundary before changing coordinates.  In particular, the 250 MeV/u
reference scorer is at world `Y=59.980 mm`; both transverse coordinates are
drifted along the recorded direction to `Y=60.000 mm`.  TOPAS `+Y` becomes GPU
depth `+Z`, while the signed third TOPAS direction cosine becomes GPU `+Y`.

The converter writes a canonical identity CSV containing origin, run/event/
track/parent IDs, PDG, `(Z,A)`, energy, weight, position, and direction.  Only
parent-0 C12 is tagged `primary`; every other charged ion, including non-parent
C12, is tagged `fragment`.  It also writes a fragment-only identity file for
`minibeam_water_entry_secondary_replay_file`.  The 250 MeV/u input contains
`30,878` primary C12 plus `57,685` fragments, including four fragment C12.

Fragment replay is deliberately a separate component run.  It skips the
primary kernel, injects records at water generation zero into the normal
secondary queue, keeps one independent energy ledger per injected particle,
and uses `number_of_histories` as the original-history normalization
denominator.  Unit weights are currently required.  The primary-only replay
continues to use the primary path and its output must be multiplied by the
survival fraction before it is added to the already incident-normalized
fragment dose.

`config/beam_minibeam_water_fragment_replay_e250_256k.yaml` is the 250 MeV/u
fragment input.  With nuclear transport enabled, all `57,685` input fragments
ran on the RTX 2080 Ti with zero queue overflow and relative energy residual
`1.28e-5`.  Setting `enable_inelastic: false` now remains runnable: the queue
does not grow, both nuclear inelastic and elastic counts stay zero, and the
relative residual is `3.12e-7`.  The latter run leaves `482.64 MeV` in the
explicit untracked ledger, exactly the entrance kinetic energy of the few
isotopes outside the current 18-ion EM table.

These checks validate plane propagation, routing, normalization, pure-EM
execution, and energy bookkeeping.  They do not yet constitute the required
GPU-export replay closure or a GPU/TOPAS mixed-ion dose match.  The next
comparison must use the same charged-ion list in TOPAS, exclude entrance
electrons/photons/neutrons on both sides, and compare primary, fragment, and
summed dose components at multiple depths.

## Full-chain EM-only 1.024M reference

`run_field3cm_em_only_e250_1024k.txt` is the upstream-field isolation run:
the original 256 PBS spots each receive 4,000 histories, for exactly 1,024,000
histories.  This is not a resampled water-entry replay.  The complete Copper,
air-gap, and water geometry is retained, while TOPAS loads only
`g4em-standard_opt4` and `g4decay`.  The matching GPU run disables Copper
nuclear attenuation and water inelastic transport and uses FP32 dose scoring.

The successful TOPAS run was Slurm job 6441 with 192 threads (307.157 s,
about 2.92 GB MaxRSS). Jobs 6439 and 6440 failed before simulation because of
source-file setup errors and are not physics trials. The RTX 2080 Ti GPU run
took 9.504 s (107,745 histories/s), with zero nuclear events, zero queue
overflow, and relative energy residual `1.63e-5`.

The absolute-Gy comparison uses no fitted normalization and a 0.1 mm lateral
by 0.25 mm depth grid. GPU/TOPAS total dose is `1.007995`; 2-D Pearson/L1 is
`0.995510/8.8566%`, IDD Pearson/L1 is `0.999968/0.9257%`, and lateral-integral
Pearson/L1 is `0.999281/3.2545%`. TOPAS/GPU Bragg depths are
`124.375/124.625 mm`. Entrance peak/valley/PVDR ratios are
`1.037/1.030/1.006`; at 99.875 mm they are `1.033/1.107/0.933`, and at the
TOPAS Bragg depth they are `1.062/1.097/0.968`.

The good IDD and entrance PVDR do not imply a complete dose match. From about
80 to 100 mm, GPU valley dose is 8--11% high and PVDR is 4--7% low. Combined
with the larger 2-D than IDD error, this EM-only result isolates the leading
remaining discrepancy to lateral water EM/MCS phase-space evolution rather
than a nuclear-cascade contribution. The output, metrics, and plots are under
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k/`.

The follow-up increases the same original source to 50,000 histories per spot,
12.8M total. TOPAS job 6442 completed on 192 threads in 3822.14 s; the matching
FP32 RTX 2080 Ti run took 55.80 s (229,387 histories/s), a 68.5x wall-time
speedup. Total-dose ratio is `1.003424`; 2-D Pearson/L1 is
`0.999564/2.8091%`, IDD L1 is `0.4887%`, and lateral-integral L1 is `1.1785%`.

This higher-statistics result supersedes the 1.024M local-feature estimates.
At depths 0.375/19.875/39.875/59.875/79.875/99.875/124.375 mm, the
peak ratios are `1.036/1.012/1.010/0.997/1.005/0.995/1.004`, valley ratios
are `0.950/0.988/1.018/1.028/1.109/1.167/1.033`, and PVDR ratios are
`1.090/1.024/0.992/0.970/0.907/0.853/0.972`. Thus peak dose is close through
most of the water path, while valley filling evolves too quickly in GPU water.
The discrepancy changes sign with depth and cannot be corrected by one dose
scale. Outputs are under
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_12800k/`; the comparison now
also emits `dose_ratios_vs_depth.png`.

## High-statistics same-source water replay

The 12.8M pure-EM full-chain run also produced 1,622,795 parent-0 C12 states
at the water boundary plus total/electron/non-electron dose in one pass. A
single TOPAS replay (job 6529) scored dose and all 40/60/80/100/120 mm C12
planes; the matching FP32 GPU replay took 10.153 s. Water-only total-dose ratio
is `1.000019`, IDD L1 is `0.1719%`, and fixed-region valley ratios reach
`1.028/1.039` at 80/100 mm. Thus the mid-depth valley excess survives an
identical entrance and is primarily downstream water transport/scoring.

`compare_water_primary_phase_space.py` intentionally does not map downstream
TOPAS EventID to an entrance row: MT phase-space replay does not preserve that
mapping across worker blocks. It reports only identity-free crossing, energy,
angle, pitch-folded position/covariance, harmonic, and fixed-region metrics.
At 100 mm the GPU angular core q68 is `0.937` of TOPAS but q99/q99.9 are
`1.230/1.270`; fixed-valley C12 fluence is `1.046`. The present water model is
therefore too leptokurtic rather than uniformly too broad. Corrected outputs
are under
`/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/phase_comparison_aggregate/`.

The analyzer now also joins adjacent downstream planes by stable
`(RunID,EventID,TrackID)` and writes interval angle/displacement/covariance,
energy-group survival, quantile curves, stopping-weighted crossing proxies,
and `water_interval_scattering_ratios.png`. Fixed regions are restricted to
`|x|<=18 mm`, matching dose. Corrected legacy 80/100 mm valley-fluence ratios
are `1.0486/1.0497`.

An optional C12-water `fermi_eyges_tail` candidate uses local scattering power,
correlated displacement, and an untruncated Poisson tail. It replaces all
legacy water-MCS corrections when selected and is off by default. Same-source
results are under
`/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/phase_comparison_fe_candidate/`
and `dose_comparison_fe_candidate/`. Fixed 40--100 mm valley dose is within
about 1%, while 2-D/lateral L1 are `2.2834%/0.6793%`. Propagation-interval joint
moments are close, but aggregate q99.9 remains low and energy-group uncertainty
is nonuniform, so the model is not promoted to formal multi-energy configs.

The plane scorer for this candidate uses an integrated-Brownian conditional
bridge rather than linearly interpolating the full-step angle and displacement.
Tail events enter a plane record only if their sampled path location is upstream
of that plane; requesting a bridge does not change the transported endpoint.
Both minibeam-enabled and minibeam-disabled SYCL builds are checked. Run the
independent fixed-energy sampler regression with:

```bash
python3 benchmark/carbonminibeam/validate_water_mcs_sampler.py \
  --output /tmp/water_mcs_sampler_validation.json
```

Interval tail uncertainty can be generated without another TOPAS run by adding
`--bootstrap-replicates 64 --bootstrap-blocks 256` to
`compare_water_primary_phase_space.py`. Stable identity blocks and common
weights across planes preserve within-engine track correlation. The resulting
intervals still share tracks and must not be described as statistically
independent holdout samples. Frozen parameters and input hashes are recorded in
`docs/minibeam_water_mcs_candidate_20260919.md`.

## Optional minibeam source/species dose decomposition

With `minibeam: true`, enabling `enable_charged_origin_voxel_scoring` and
setting `charged_origin_voxel_mhd_output_prefix` also writes 17 minibeam
component maps. They separate primary C12 and eight secondary-ion classes by
immediate Copper/water birth material. The switch is diagnostic-only and is
disabled in the formal configurations.

Analyze a completed run with, for example:

```bash
python3 benchmark/carbonminibeam/analyze_minibeam_source_components.py \
  --gpu-dir /path/to/gpu/run --gpu-histories 10000128 \
  --topas-total /path/to/topas/dose.bin \
  --topas-total-histories 10000128 \
  --topas-component-dir /path/to/topas/component/run \
  --topas-component-histories 1024000 \
  --output-dir /path/to/gpu/run/analysis
```

The analyzer checks voxel/global closure, reports fixed peak/shoulder/valley
fractions at Bragg, and compares only taxonomy-compatible TOPAS aggregates.
The older TOPAS `secondary carbon` map contains every carbon isotope, so it is
not presented as a C12-only comparison.

The fixed-region masks are shared with `compare_gpu_topas_dose.py` and use the
coordinates and spacing stored in the GPU MHD header. Peak owns
`|folded x| < 0.25 mm`, shoulder owns `[0.25, 0.9) mm`, and valley owns
`[0.9, 1.8] mm`, within `|x| <= 18 mm`. For TOPAS all-helium comparisons the
analyzer uses the legacy all-Z=2 GPU map when it is available; the 17-way map
keeps only He-3/He-4 explicit and places rarer helium isotopes in `other`.

Low-cutoff charged nuclear products are deposited locally as before, but their
diagnostic component dose is assigned to the child's `(Z,A)` and immediate
water birth material rather than to the parent track. This changes labels
only; a 300 MeV/u, 256k same-seed scorer A/B gave voxel-dose L1 `1.09e-8`,
while the 17 component maps reconstructed the total with voxel L1 `2.48e-7`
and global relative difference `1.10e-9`.

`minibeam_water_secondary_c12_mcs_model: fermi_eyges_tail` is a default-off
research switch. It lets secondary C12 reuse the validated C12-water
correlated angle/displacement kernel in homogeneous water; p/d/t/He/heavier
ions remain on legacy Highland. The switch has passed compile and 256k
full-chain smoke checks.

The same-source accuracy comparison is now available. It sends the exact same
1,622,795 parent-0 C12 water-entry states through the primary path, secondary
legacy Highland, and secondary FE with 0.10/0.05 mm internal segment caps. The
input is a code-path diagnostic and is not described as a physical secondary
birth sample. Nuclear inelastic and elastic transport are disabled.

Across the four paired 20 mm water intervals, secondary legacy gives
GPU/TOPAS angle-variance ratios `1.356--1.477`, displacement-variance ratios
`1.379--1.445`, and q99 ratios `1.163--1.204`. Secondary FE reduces these to
`0.984--1.013`, `1.015--1.018`, and `0.992--1.007`, respectively. Reducing the
internal cap to 0.05 mm doubles the actual segment count and leaves the moments
stable. Plane scoring on/off changes the FP32 voxel dose by only `6.25e-8` L1.

The validation also exposes a separate longitudinal path difference: the
primary minibeam path applies the frozen `0.9958` stopping scale after Unified
EM, while the secondary path does not. Consequently the secondary replay Bragg
peak is 0.75 mm upstream and its IDD L1 is about 1.89%, independent of whether
legacy or FE scattering is selected. Do not compensate this with MCS tuning.
The FE result is accepted for secondary-C12 scattering only and remains off in
formal configs until true secondary-C12 birth spectra and common C12
stopping/straggling semantics are isolated.

Diagnostic configs are named
`beam_minibeam_water_secondary_c12_replay_e*_*.yaml`; reproducible commands are
in `run_secondary_c12_fe_validation.txt`. The combined 250 MeV/u plot and JSON
are under
`/mnt/sda/wuwei/minibeam_secondary_c12_same_source_e250_20260919/summary/`.
Use `summarize_secondary_c12_fe_validation.py` to regenerate them. Existing
150/300 MeV/u references also show near-unity FE angle/position moments, but
their 26k/35k statistics are screening data rather than q99.9 acceptance data.

The controlled longitudinal closure supersedes the earlier unscaled-secondary
interpretation. `prepare_secondary_c12_energy_path_cases.py` explicitly routes
both code paths through Unified EM and generates no-MCS/no-straggling,
straggling, and matched post-sampling-scale cases. With the same 1,622,795
entrance C12 states, primary versus secondary IDD L1 is `0.00827%` without
straggling, `0.02993%` with straggling at scale 1, and `0.03137%` with both
paths at scale `0.9958`. The secondary loss audit measures
`scaled/raw=0.99579945`, confirming that this knob scales each sampled total
loss, including its fluctuation, rather than only the mean stopping power.

After that closure, the matched-scale secondary FE result gives interval
angle-variance, displacement-variance, covariance, and q99 ratios of
`0.969--0.988`, `0.995--1.017`, `0.990--1.012`, and `0.983--0.994` relative to
the existing TOPAS reference. Legacy Highland gives `1.471--1.512`,
`1.535--1.552`, `1.516--1.537`, and `1.202--1.221`. The FE 0.10/0.05 mm IDD
L1 is `0.03095%`; fixed ROIs and interval moments are stable. A skipped
positive z-boundary remainder below `1e-5 mm`, not intrinsic FE instability,
caused the older periodic dose spikes. Formal secondary FE settings remain
unchanged and default-off.

True queued C12 births can now be exported by setting
`fragment_birth_spectrum_output_file`; the additional
`<prefix>_c12_joint.csv` retains unique replay-particle/source-history IDs,
RNG stream, generation, immediate birth region, energy, weight, true position,
and direction. Convert and replay real water births with
`prepare_secondary_c12_birth_replay.py` and
`prepare_secondary_c12_birth_validation.py`. Internal-birth replay is guarded
by the default-off
`minibeam_water_entry_secondary_replay_allow_internal_births` option and accepts
positions anywhere inside water plus backward directions. It assigns a unique
energy-ledger identity to every injected particle while keeping the configured
incident-history normalization independent of replay record count.

The 250 MeV/u 256k diagnostic exported 4,511 water-born C12 states, including
644 backward states and 221 records sharing an incident history. Pure-EM
replay completed all records with no queue growth, nuclear/elastic event,
Unified-EM miss, or overflow. After preserving the exported RNG stream,
generation and birth region in replay, the relative energy residual is
`6.33e-8`.
The separate Copper-exit file contains 58,116 charged particles; this sample
contains 17 other carbon isotopes and no C12, so none were mixed into the
water-born replay.
This establishes implementation and coverage closure only: no matching TOPAS
true-birth sample exists, so it is not a new physics-accuracy result. Outputs
are under
`/mnt/sda/wuwei/minibeam_secondary_c12_birth_validation_e250_20260919/`.

### Directed secondary depth boundaries and full-chain C12 A/B

Secondary depth-bin ownership is direction-aware. A particle exactly on a
depth face belongs to the bin it is about to enter: the downstream bin while
moving forward and the upstream bin while moving backward. The legacy path
also preserves a real boundary-limited step below `1e-5 mm` instead of
enlarging it with the generic minimum-step guard. The deterministic regression
in `validate_secondary_depth_boundaries.py` covers before/on/after a depth
face, both directions, and both legacy and C12-only Unified-EM paths. All 12
start-bin checks pass; the legacy/Unified total-dose ratio is `0.999999921` and
both energy residuals are below `7e-8`.

`minibeam_water_secondary_c12_enable_unified_em` is a default-off diagnostic
switch. It changes only C12 already travelling in the water secondary queue;
other secondary species retain their formal EM paths. Together with the
independent post-sampling loss scale and secondary-C12 MCS switch, it supports
the staged full-chain comparison:

| case | secondary-C12 EM | loss scale | MCS |
|:---:|:---|---:|:---|
| A | formal legacy | 1 | Highland |
| B | Unified EM | 1 | Highland |
| C | Unified EM | 0.9958 | Highland |
| D | Unified EM | 0.9958 | FE |

The 300 MeV/u, 256k same-seed screen found no resolvable benefit from A to B
or B to C, so those comparisons were not expanded to high statistics. C and D
were repeated with seeds `202609300` and `202609301`, each with `10,000,128`
histories. FE changes the secondary-C12 component strongly (about `31%` voxel
L1), but that component is only `0.55%` of total dose and about `0.97%` in the
Bragg +/-2 mm slab. Consequently D/C total-dose ratios are
`0.9999945/1.0000064`, and 2-D, IDD, lateral and fixed-region dose metrics show
no consistent improvement across seeds. The 150/250 MeV/u 256k C/D screens
show no large regression but are not promotion evidence.

Therefore C12-only Unified EM, the secondary `0.9958` scale, and secondary FE
remain disabled in all formal energy configurations. The validated FE path is
retained for controlled diagnostics. Full reproduction commands are in
`run_secondary_c12_fe_validation.txt`; results are under
`/mnt/sda/wuwei/minibeam_secondary_c12_fullchain_ab_e300_256k_20260919/` and
`/mnt/sda/wuwei/minibeam_secondary_c12_fullchain_cd_e300_10m_seed*_20260919/`.

The true-birth replay CSV additionally preserves `rng_stream`, water
`generation`, and `birth_region`. Accumulated spot/shard results merge birth
records with offset source histories and globally unique replay-particle IDs.
This preserves replay state but does not make independently replayed ancestors
and descendants additive as a full-chain dose.
