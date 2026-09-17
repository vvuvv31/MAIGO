# δ collision sampling optimization screening — 2026-09-14

The reference is the previously optimized unified EM build, not the original EM
implementation. All runs use 200,000 primary histories and the same unified EM
package, seeds and physical configuration. GPU runs are local, sequential under
the existing GPU lock. RT07575 uses the validated Schneider v2.1 stack and the
production configuration without independent research elastic. No production
physics settings were changed.

## Exact interval-cache candidates

The shared-cache candidate stores the interpolation interval for each density
endpoint and reuses it across candidate-rate and post-loss acceptance queries.
The separate-cache candidate uses different indices for these two query energies.
Both validate the interval bounds on every reuse and fall back to the original
binary-search convention. Polynomial arithmetic, rates, energy spectra, RNG
consumption, form-factor veto, continuous fluctuations, MCS and collision clocks
are unchanged. The lookup test covered 3,150 package records, traversed knots in
both directions with neighboring representable energies and varying factors:
4,712,376 GPU queries, zero bit mismatches.

|Candidate|Case|Paired rounds|Baseline histories/s|Candidate histories/s|Change|
|---|---|---:|---:|---:|---:|
|cache|b3_layers|3|11202.28|11081.29|-1.08%|
|cache|b4_soft_lung|3|8934.63|8835.29|-1.11%|
|cache|b4_soft_bone|3|12665.39|12571.04|-0.74%|
|cache|RT07575|3|11410.46|11242.93|-1.47%|
|separate|b3_layers|3|11202.28|11103.64|-0.88%|
|separate|b4_soft_lung|3|8934.63|8850.55|-0.94%|
|separate|b4_soft_bone|3|12665.39|12553.15|-0.89%|
|separate|RT07575|3|11410.46|11338.62|-0.63%|

Throughput entries are medians. Scheduling of the two screening drivers was
serialized by the GPU lock; runs were not simultaneous. This is a screening,
not a randomized thermal-controlled performance study. Some early runs overlapped
CPU compilation; the small measured differences should not be over-interpreted. All compared runs pass
quality and have zero queue overflow, identical EM audit counters and primary
steps, identical nuclear interaction counts, and identical configuration hashes.
The b3/b4 mass-weighted IDD comparisons also pass the 0.0001% peak-normalized
difference gate. Maximum voxel dose difference is
normalized to the reference global peak; it is not a local Gamma measurement.

Maximum observed voxel difference/global peak: 0.000396408215%.

Neither interval-cache variant is promoted. Added live state, bounds checks and
GPU code generation may offset saved searches; no hardware-counter profiling
was available to establish the precise cause. Candidate source snapshots,
binaries, the GPU lookup test and result comparison scripts are retained under
`scratch/delta_rate_cache_20260914/`. Source and CMake files were restored to their
pre-screening contents, preserving the earlier uncommitted optimizations.

## Revised direction for larger gains

The current implementation already caches material state and accumulates primary
and secondary dose within a voxel before updating the main dose array. These
cannot be counted as new optimization opportunities. Each accepted delta event
still shortens the EM step. Merely optimizing the rejection sampler cannot
remove that dominant event count (RT07575: approximately 2,251 proposals and
1,984 accepted events per primary, including descendants).

A multi-collision macro step needs a separate controlled research implementation:

1. Keep the existing material-specific production threshold Tc. Introduce a
   computational split Ts >= Tc. Explicitly sample hard transfers T >= Ts;
   treat Tc <= T < Ts as a compound random energy loss. This split is not a
   change to the TOPAS production cut.
2. Derive accepted differential rate nu(T; E, material) from the same native
   spectrum, including spin, magnetic moment and form-factor acceptance. Keep
   proposal nulls separate from accepted collisions. In a frozen local state,
   soft-event count has mean h * integral(nu dT), total-loss mean is
   h * integral(T nu dT), and variance is h * integral(T^2 nu dT).
3. Add this soft component to the existing loss below Tc exactly once. Preserve
   its non-Gaussian tail when counts are small; do not replace all delta energy
   by its expectation. A Gaussian approximation is not assumed adequate.
4. Bound energy/rate variation over the macro step. Maintain CT and dose-grid
   boundaries, remaining nuclear optical depths and independent MCS spatial
   resolution. Near Bragg peak, production-cut onset, material interfaces or
   sparse soft collisions, fall back to explicit transport. Frozen-rate Poisson
   batches are an approximation to energy-dependent transport, not exact.
5. Validate loss distributions and energy balance for water and Schneider
   materials across all 18 ions before transport performance claims. Then use
   100/200/300 MeV/u IDD/R80 and b3/b4, followed by RT07575 BODY-masked Gamma
   with DTA/10 search spacing. Acceptance gates: Gamma loss <=0.2 percentage
   points, peak error increase <=0.3 percentage points, R80 shift <=0.1 mm,
   zero overflow. Report statistical uncertainty from changed RNG streams.

This larger algorithm is a design, not an implemented or validated production
feature. No 100k histories/s claim follows from the interval-cache experiment.
