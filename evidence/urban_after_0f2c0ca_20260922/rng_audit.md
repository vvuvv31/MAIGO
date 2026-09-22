# RNG address audit (fix C2) — anchor 0f2c0ca + C2 work

Source: `include/carbon/rng.hpp`, consumers in `src/transport_sycl.cpp` and
`src/detail/sycl_device_math.inc`. Key = (seed, history) pair; counter =
{hist_lo, hist_hi, ii_lo, ii_hi ^ block[^ MSC_tag]}; lane = dim % 4.

## 1. Key spaces (disjoint by construction)

| key | users |
|---|---|
| (spot_seed, rng_history) | primary source, EM, MSC, elastic, inelastic, diagnostics |
| (2026, frag.rng_stream) | secondary C12 + fragments (child_stream of parent; never equals a primary key in practice — distinct derivation; not part of the formal proof) |
| (2026, ...) fixed seeds | legacy diagnostic draws with fixed ii |

## 2. Legacy (uniform01 / random_u32) consumers — all ii < 2^32

| family | ii form | bound | dims (lanes) |
|---|---|---|---|
| source | 0 | exact | 30–41, 34/35 |
| nuclear/ion | steps (u32) | ≤ maximum_primary_steps (u32) | 0–24, 44, 80/81 |
| all-ion elastic (primary) | steps (u32) | same | 10/11, 14, 17, 19, **70/71** |
| electron/photon channels | steps (u32) | same | 20–24 and others |
| unified primary | unified_primary_counter++ (u64) | draws/history ≪ 2^32 | 120 |
| secondary legacy | sec_steps (u32) | ≤ kSecondaryMaxSteps = 30000 | 1, 2, 13–16, 70/71, ... (all < 100) |
| FE observation | steps/sec_steps composed (legacy 1024 scheme, untouched) | < 2^32 in practice | 40, 100 |

Legacy counter[3] = (ii>>32) ^ (dim/4). With ii < 2^32, ii>>32 = 0; with dim
any u32, dim/4 ≤ 0x3FFFFFFF. Bit 30 is ALWAYS clear. (steps is uint32_t, so
ii>>32 = 0 structurally; the unified counter would need > 4G draws in one
history to violate — excluded by step caps × draws/step ≪ 2^32.)

## 3. MSC-domain (msc_unit_strict) consumers — all ii < 2^32 by guard

| user | ii | dims |
|---|---|---|
| Cu primary Urban | beamline_step·1024 + 0 (segment 0: one MSC step per outer iteration) | 50 + k |
| limiter Randomizetlimit | outer·1024 + segment | 58, 59 |
| water primary Urban | steps·1024 + segment | 70 + k |
| secondary Urban | sec_steps·1024 + segment | 110 + k |

Guards (propose + limiter, before any draw): segment < 1024, outer < 2^22,
hence ii < 2^32. MSC counter[3] = (ii>>32) ^ (dim/4) ^ 0x40000000: bit 30 is
ALWAYS set.

## 4. Separation proof

Legacy word3 ∈ [0, 0x3FFFFFFF] (bit30 = 0). MSC word3 has bit30 = 1. The
counter SETS are disjoint; Philox-4x32-10 at fixed key is a bijection over
counters, so no MSC draw can equal a legacy draw. This replaces the old
"partner steps ≥ 12.6M macro steps" probabilistic argument (which was wrong
in the dangerous direction: the actual collision was at outer = segment = 0,
dim 70/71, primary key — elastic vs old-Urban sharing identical bits on the
first step whenever elastic runs; EM-only never runs elastic, which is why
no effect was ever seen there).

Legacy streams are bit-identical (no consumer touched). Urban draws move to
the new domain: physics comparisons across the change are valid only as
distribution/statistical equivalence (see E2 ensemble note below).

## 5. Verification

- `test_msc_domain_separation` (urban_localize): word3 bit30 clear/set over
  the legacy ii×dim envelope and MSC ii×dim envelope; the documented
  (0,0,70) collision point now draws differently; determinism +
  segment sensitivity + (0,1) openness on the new domain.
- `test_rng_index_guards`: envelope top (2^22−1, 1023) valid; segment 1024
  rejected; outer 2^22 rejected; limiter rejects segment 1024. No draws are
  consumed on rejection (guard precedes all draws).
- E2 ensemble (8×200k, new domain): mean 3.4636e-08, SE 2.6e-10; old-domain
  frozen center 3.4131e-08 inside the CI → distribution unchanged; frozen
  band kept (covers the ensemble mean at ~2.5 SE of run-to-run scatter).
- GPU: normal smoke shows fatal = 0, guard = 0 (see manifest); degenerate
  table injection fires reason-6 stall fault with first-failure record,
  nonzero exit, and no accepted dose (see below).
