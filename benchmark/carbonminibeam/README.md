# Carbon minibeam reference

The GPU geometry uses a rectangular Copper block.  Its outer shape is not
intended to reproduce the visible TOPAS cylinder; the 15 air slits are the
shared reference geometry.

| parameter | value |
|---|---:|
| collimator centre upstream of isocentre | 30 mm |
| rectangular width × length × thickness | 120 × 50 × 60 mm |
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

`prepare_water_entry_replay.py` filters parent-0 C12 tracks from a TOPAS water
entrance phase space and writes matched TOPAS and GPU inputs.  The GPU CSV uses
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
