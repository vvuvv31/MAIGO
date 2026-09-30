# Throughput: host pipeline, neutron scheduling and beamline lane order (2026-09-28)

RTX 2080 Ti (sm_75), FP32 dose, `build/entrance-response`. Benchmarks are the
two throughput presets at 1M histories unless stated:

- open field: `config/proton_water_throughput.yaml` (170 MeV proton, uniform water, Urban)
- minibeam: `config/proton_minibeam_neutron_throughput.yaml` (Copper collimator, air Urban,
  experimental neutron transport)

Wall time is the full process (`bench.py`: `perf_counter` around the executable),
including setup and output.

## What changed

None of the changes alters per-history physics. Every history keeps its index,
RNG stream and output slots; integer transport/EM/neutron audits are identical to
the baseline for the same seed. Remaining dose differences are FP32 atomic
accumulation order (relative L1 ~1e-7), which `AGENTS.md` accepts for scheduling
work.

| Change | Files | Scope |
|---|---|---|
| Process-local SHA256 memo + parallel prefetch of pinned inputs | `include/carbon/sha256.hpp`, `src/main.cpp` | both |
| Background parsing of CINPKG04 packages (Schneider primary/secondary, Copper cascade), unified EM + delta moments, neutron bank + recoil stopping tables | `include/carbon/host_prefetch.hpp`, `src/transport_sycl.cpp`, `src/neutron_transport_sycl.cpp`, `include/carbon/neutron_transport.hpp` | both (neutron part: minibeam) |
| Unified EM package parsed before its pin is checked (pin still enforced, mismatch reported first) | `src/unified_em_package.cpp` | both |
| Sparse charged-origin CSVs formatted with `std::to_chars` (byte-identical to `%.12g`) | `src/io.cpp` | both |
| Neutron recoils moved to a queue + persistent-lane kernel; neutron flights on persistent lanes; per-flight rate reuse across boundary crossings | `src/neutron_transport_sycl.cpp` | minibeam |
| Beamline lane order: straight-ray copper prediction groups lanes | `src/transport_sycl.cpp` | minibeam (split/compact pipeline, flat source) |
| Bracketed nearest-record search for the per-step Copper elastic rate; skip the Gaussian loss that native proton fluctuation overwrites | `src/transport_sycl.cpp` | minibeam |

### Host pipeline

Each run used to read ~10 GB for ~3.6 GB of unique inputs: config validation,
every loader and the output provenance report re-hashed the same packages
(`cinel03_secondary` 5x, `proton_cinel03` 5x, neutron `collisions.bin` 8x,
`unified_em_v1.bin` 2x). `compute_file_sha256_hex()` now memoizes digests for the
process. A digest is reused only while the file keeps the device/inode/size/mtime
seen before hashing; waiters re-stat after the digest is ready. `main()` hashes
every large file named by the shard configs (and by JSON manifests they name) on
8 background threads before configs are validated.

The large packages are then parsed on background threads from the start of
transport setup (`HostPrefetch<T>`). Each consumer calls `take()`, which returns
the result of the exact loader it would have called (rethrowing its exception) or
runs it synchronously when nothing was prefetched. Nothing is prefetched when a
plan-level physics cache will serve the package.

### Neutron transport

Measured before the change: one lane per neutron birth, 32.5M flights and 360M
recoil steps per 1M histories, warp efficiency 0.22 (per-birth work p50 660,
p99 4909, max 39269 iterations). Recoils are now written to a queue (with their
ordinal, weight and RNG stream, so Urban seeds and proton capture records are
unchanged) and transported by a second kernel whose lanes pull the next track
when theirs ends. Flights also run on persistent lanes that pull births; each
birth keeps its xorshift stream, stack order and ledger slot. Per-track ledgers
are summed per lane in double. If the queue is full a recoil runs inline, as
before.

### Beamline lane order

About 15% of flat-source histories pass straight through a slit; the rest spend
~590 Copper steps each. A small pre-pass samples the same source draws (34/35),
tests the straight ray against the collimator and partitions lanes
(copper-bound first). Only the lane -> history map changes. The water stage also
gains because compact water entries now arrive grouped.

## Switches (same binary)

| Variable | Default | Effect when 0 |
|---|---|---|
| `CARBON_SHA256_MEMO` | 1 | every check re-reads the file |
| `CARBON_SHA256_PREFETCH_THREADS` | 8 | no startup hashing threads |
| `CARBON_NEUTRON_CHARGED_QUEUE` | 1 | recoils transported inline in the neutron lane |
| `CARBON_MINIBEAM_BEAMLINE_ORDER` | 1 | beamline lanes in history order |

## Results

Final binary vs the pre-session baseline, same seed per pair, run order
alternated between pairs (the 2080 Ti throttles 1860 -> ~1600 MHz when hot, so a
fixed order biases kernel times by 2-5%). Raw runs:
`/mnt/sda/wuwei/claude_perf_20260928/runs`, summary: `final_summary.json`.

| Scene | Histories | Baseline wall | New wall | histories/s | Gain |
|---|---:|---:|---:|---:|---:|
| open field | 1M (3 seeds) | 19.53 / 19.85 / 20.00 s | 15.09 / 15.28 / 15.45 s | 50.5k -> 65.5k | x1.30 |
| open field | 4M | 61.94 s | 56.71 s | 64.6k -> 70.5k | x1.09 |
| minibeam | 1M (3 seeds) | 22.13 / 22.35 / 22.23 s | 12.58 / 12.61 / 12.57 s | 45.0k -> 79.5k | x1.77 |
| minibeam | 4M | 61.63 s | 42.05 s | 64.9k -> 95.1k | x1.47 |

Minibeam GPU time per 1M: beamline 6.46 -> ~5.45 s, water 3.3 -> ~3.1 s,
neutron kernels 2.76 -> ~0.65 s. The open-field primary kernel is unchanged
(order-alternated means equal within noise); its gain is host time.

Checks on every pair: quality accepted, queue overflow 0, energy balance
<1e-6, dose relative L1 5.6e-8 (open field) / 1.9e-7 (minibeam). All integer
audit lines are identical. The only differing line is the neutron summary's
6-significant-digit `water_MeV` (e.g. 1.18342e+06 vs 1.18343e+06): recoil
deposits are now tallied per track in double instead of per birth in float;
births, collisions and all other neutron tallies are identical.

Integrity: wrong pins (EM, Urban) and tampered copies (a flipped byte in the
Urban package; a corrupted EM header) fail with the same message and exit code
as the baseline. Early config errors exit 1 as before (see `failed.md` for the
OpenSSL initialisation fix found during this check).

## Not done / next candidates

- Open-field primary kernel: warp efficiency measured from per-history step
  counts is 0.858 (nuclear interactions end histories early). A persistent-lane
  (history refill) water loop could recover up to ~10% of primary time, but the
  loop body is ~3000 lines with many exits; not attempted here.
- Minibeam water stage: live entries have warp efficiency ~0.79 in history
  order; an energy-bucketed entry order might add a few percent.
- Larger open-field chunks were measured and rejected (see `failed.md`).
