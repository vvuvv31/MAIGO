# FP32 hard-rate certificate experiment

Status: component proof, trajectory equivalence, RT07575 ABBA and default-preset
regression passed. The build preset now enables the certificate. Dose scoring remains FP32. Baseline is the
accepted same-step parameter-reuse binary `b3039e0d…`.

## What is removed

The pre-step material query recomputes both hard-process rates at both density
nodes solely to check the precomputed global majorant. The candidate computes
elastic rate, stopping power and range exactly as before, and reuses these
parameters in loss sampling. It omits the four pre-step hard-rate searches and
cubic evaluations **only after a load-time certificate succeeds**.

The collision flight/majorant itself, endpoint hard-rate evaluation, finite-rate
and actual-rate≤majorant check, null test and conditional outcome sampling are
unchanged. No rate, energy, step limit, RNG law or material approximation changes.

## Certificate and rounding scope

The original SHA/content-validated source tables are retained. The host validates
the values after conversion to FP32, checks contiguous noncollapsed rate intervals
and coverage of the parameter energy domain. Binary search therefore selects an
interval containing E; monotone FP32 subtraction and division give normalized
input t∈[0,1].

Each interval is subdivided into 64 exact dyadic subintervals. Interval arithmetic
evaluates the existing four-term Lagrange expression over the **whole continuum**
of each subinterval, rounding every endpoint outward beyond FP32 rounding. It
encloses both exact and rounded intermediate values, including contraction into
FMA. This is not a finite grid probe of the polynomial maximum.

The resulting positive upper bound must be ≤0.99 of the unchanged stored node
majorant. For physical density between the bracketing nodes, all mixing weights
are nonnegative. Thus exact density mixing retains the same bound ratio. The
different FP32 evaluation orders of mixed rate and mixed majorant are covered
by the 1% reserve, much larger than gamma_32<2e-6. Validated density [1e-6,1e3]
and node-majorant [1e-15,1e15] bounds prevent overflow and relevant underflow;
any subnormal term is negligible relative to the strictly positive mixed bound.
Builds do not use fast-math. Unsupported certificates fail before transport.

All 82 current nodes, 12,169 pieces pass. The largest conservative bound/majorant
ratio is **0.2352708943**, leaving substantially more reserve than required.
Certificate overhead occurs only at first bank loading, not each batch. Endpoint
checks still independently reject a violated bound at an actual hard proposal.

## Component validation

Separate baseline/candidate GPU executables cover 82 native density nodes plus
57 intermediate densities, 284,672 electron histories and 28,555,782 steps.
Each step hashes every physical result field, flags and RNG state. Per-history
hash files are identical, as are final statuses and all step counts. Zero invalid
or missing-coverage cases; every history terminates or hands off to a photon.

Evidence under `out/electron_rate_certificate_20261003/`:

- `bound.log`: interval-certificate result.
- `step_hash_report.json`: exact trajectory/RNG comparison.
- `source/fingerprint.json`: frozen candidate source.
- `screen/report.json`: ABBA results as they finish.

`CARBON_ELECTRON_CERTIFIED_RATES` is an isolated build option. It is OFF in the
original baseline; the accepted build preset now explicitly enables it.

## Full-runtime result and adoption

RT07575/group32/chunk262144/lanes16384, ABBA with the same two 1,983,920-primary
shards per process. Distinct histories per arm: 3,967,840; repeated timing runs
do not increase dose statistics.

| Measure | Parameter-reuse baseline | Certified rates |
|---|---:|---:|
| Resident batch mean |20.881494 s|19.783516 s|
| Resident throughput |95,009 primary/s|100,281 primary/s|
| Electron stage mean per batch |10.952777 s|9.747400 s|

Throughput +5.550% (pairs +5.358%/+5.739%); electron time −11.005%;
whole-process throughput +4.735%, including load-time proof overhead.
All eight batches pass research-quality, completion, energy and zero-overflow
checks. All primary/EM and electron/joint/photon work counts are exactly equal.
BODY L1≤6.015e-8, pooled peak/valley change≤1.010e-6%, and local difference
at ≥1% BODY maximum≤0.0001447%. Maximum relative energy residual4.942e-8.

Default lung CT/170 MeV2×500k regression also passes, with exact work, one
certificate for the two batches and bank cache reuse. Its timing is excluded
because it overlapped CPU compilation; no performance claim uses that run.

`build/carbon_mc` is now the accepted binary, SHA256
`63255a237a44dd4d88656ba5b8196ea1539baf7d391ddfd8f5297e0b9bd0980a`.
The direct kernel remains241 registers/1280B static stack; its speedup is not
attributed to an occupancy increase. All default electron arithmetic remains
FP32; the independent FP64 loss-gate experiment is OFF.
