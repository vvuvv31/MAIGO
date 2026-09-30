# Native neutron package tools

These tools extract Geant4 11.3.2 neutron data. An experimental GPU neutron
consumer is now available in research mode (see docs/neutron_package_plan.md). Builds and extraction run through local Slurm, with
`/software/env_topas.sh` loaded. The water material is Water_75eV.

## Commands

```text
extract_neutron OUTPUT MATERIAL GRID SAMPLES [RATE_REPEATS [SEED]]
python3 compile.py RAW_ROOT OUTPUT_PACKAGE
inspect_neutron OUTPUT_PACKAGE/package.json
python3 adaptive_rates.py EXTRACTOR OUTPUT MATERIAL --guide DENSE_RATES_CSV
python3 assemble.py BANK_PACKAGE OUTPUT_PACKAGE --water WATER_RATES --copper CU_RATES --air AIR_RATES
```

Materials: Water_75eV, G4_Cu, G4_AIR. Grid energies are **total neutron kinetic
MeV**, not MeV/u. Outputs never overwrite an existing directory. `COMPLETE`
marks complete extraction, not physical acceptance. `compile.py` expects all
three material directories and emits a checksummed draft collision bank.

For event banks, use SAMPLES>0 and RATE_REPEATS=1. For repeated rate estimates,
use SAMPLES=0 and RATE_REPEATS>1. Each repeated query evicts the exact-energy HP
cache with a different energy before querying the unchanged target energy.
Rates are 1/mm. `rate_sem_per_mm=0` with one query means uncertainty was not
estimated. The recorded variance includes the native thermal Monte Carlo
integrator's query variation; it does not measure integrator bias.

## Adaptive rate extraction

`adaptive_rates.py` uses independently seeded queries at endpoints and
log-energy midpoints. It starts from a log grid, known model boundaries
(19.5, 19.9, 20 MeV), and brackets strong extrema from an optional dense guide.
Guide values are never accepted as the rate values. All nodes are representable
FP32 energies; adjacent floats on either side of a model seam are retained.

The default per-process midpoint tolerance is 2%. Residuals exceeding that
threshold plus three combined standard errors cause interval subdivision.
An interval is classified separately as:

- `pointwise_bound_met`: residual + 3 SEM meets the pointwise tolerance;
- `statistics_limited`: inconclusive given query noise;
- `refine`: significant residual remains when the refinement budget ends;
- `zero_process`: zero native estimates and variance at this probe;
- `adjacent_float32`: no intermediate FP32 energy exists.

Use `--resume PREVIOUS_OUTPUT` to continue significant unresolved intervals.
Resume checks the previous rates/audit checksums, material, extractor and
tolerance; it retains resolved intervals and uses a new seed range. Use `--refresh-statistics` with a larger query count to resample noisy endpoints;
all intervals sharing those endpoints are audited again. `--refine-inconclusive`
allows subdivision near the tolerance boundary when the measured curvature
exceeds 3 SEM. Neither option silently certifies unresolved intervals.

These classifications concern sampled midpoints. They do not certify absence
of narrower resonances, conditional-event interpolation, or dose accuracy.
`runtime_validated` remains false for every output. Complete drafts are useful
for further extraction and audits; they are not accepted by the runtime loader.

## Runtime integration status

`assemble.py` produces a v2 draft with two independent grids: the unchanged
`collisions.bin` contains sparse conditional event nodes, and `macroscopic.bin`
contains the refined macroscopic rates. Collision-bank rate fields are retained
for provenance and event indexing; they are not the transport rate table.
Each macroscopic record stores material, process subtype, native query count,
outgoing interval status, FP32 energy, rate and SEM. Rates and SEM are in 1/mm.
The binary starts with `NEUTR001`, then four little-endian uint32 fields
(version=1, flags=0, record count, reserved=0), followed by 28-byte records.
Records are ordered by material, process, then energy. Terminal interval status
is zero; other codes are included in the manifest.

Assembly checks source hashes, complete process/interval coverage, common rate
extractor, FP32 representability and energy coverage of the event bank. It
preserves unresolved statistical/refinement statuses. The C++ audit loader
checks the independent binary and keeps supporting v1 banks. Neither assembling
nor loading implements event interpolation, neutron flight or descendants;
the validated loader still rejects v2 for production. Research transport uses
explicit draft loading and reports its unresolved approximations.

`neutron_birth_output_file` is an independent optional diagnostic. In SYCL
minibeam/homogeneous-water runs it records CINEL03 neutrons at the actual GPU
birth position, including copper cascades and water secondary reactions. It
uses existing event rotation, does not consume random draws, and does not
apply the charged-particle energy cutoff. `neutron_birth_capacity` defaults to
1,000,000 records; overflow rejects the run. Shard/spot aggregation offsets
source-history IDs. CSV rows are in GPU queue order, not history order.

With transport disabled, capture does not alter dose or accounting. With
`enable_neutron_transport: true` and `run_mode: research`, captured births feed
the new experimental GPU pass, which adds water dose and keeps a separate
terminal/Q ledger. This is an incomplete physical model, not a validated package.

## Recoil stopping extension

`extract_neutron_recoil OUTPUT MATERIAL SPECIES_TXT` uses the native neutron
physics/material initialization and extracts electronic plus nuclear stopping
for each supplied charged ZA. Initialize physics tables before calculator
queries and destroy calculator caches before the run manager. Run extraction
through Slurm. `compile_recoil.py RAW_ROOT` emits the checksummed three-material
manifest consumed by `neutron_recoil_stopping_manifest`. The companion
`neutron_elastic_two_body` switch enables eligible >=100 keV stationary-target
elastic reconstruction. Both remain experimental; they do not add photon or
electron transport.
