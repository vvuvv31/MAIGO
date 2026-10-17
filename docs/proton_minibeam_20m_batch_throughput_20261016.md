# Proton CT minibeam: 20-million-history resident batches

Date: 2026-10-16.

This follow-up measures the retained shared Cu/air continuation and 1,048,576-history primary pool using larger resident batches. The earlier controlled tests used 1M histories/batch on RTX 2080 Ti and 2M on A6000. Historical full-plan throughput records used approximately 20M histories/batch. Those absolute rates cannot be interchanged.

## Matched protocol

Each of the three original optimized full plans (RT07575 Head1, RT06423 Head2, 20022516 Lung) runs three consecutive batches of exactly 20,000,000 incident histories per arm and device, reusing one context and loaded packages. Both GPUs run concurrently. Each device executes 360M histories in total; the two devices execute 720M. Matched histories are performance/implementation comparisons and are not combined into extra independent dose statistics.

The control is the frozen pre-upstream-change implementation with a 262,144-history primary pool. The candidate is the current installed implementation, enabling shared proton Cu/air continuation with a 1,048,576-history pool. The source/native binaries are the exact already-tested artifacts from the [upstream continuation experiment](proton_minibeam_beamline_continuation_20261016.md). RTX 2080 Ti uses the full five-pipeline native sm_75 builds, and A6000 uses matched proton-minibeam-only native sm_86 builds. There is no compilation during timing. All GPU execution sets CUDA_DISABLE_PTX_JIT=1.

The 60M campaign allocation uses largest remainder from original optimized full-plan counts, then balances each batch to exactly 20M. Source float strings, spot optics, CT, placement and collimator are preserved. Unique reproducible per-spot/batch seeds match between arms. Normalization is W/60M, and each normalized batch contributes once. The same campaign on different devices is not treated as independent additional dose statistics.

Patient/Cu step caps remain 0.5/0.05 mm. Proton slit-air settings, primary fluctuations, native ionization/nuclear clocks, loss slopes, nuclear processes, secondary settings and FP32 scoring are unchanged. No electron transport is introduced.

Each arm is one three-batch campaign. Case execution order is control/candidate for Head1 and Lung and candidate/control for Head2. There is no independent reverse-order repetition in this follow-up. Clocks are unlocked, so small changes and precise hardware ratios require caution.

## Hot throughput

Batch 0 is excluded. Hot throughput is 40,000,000 divided by the sum of elapsed whole-call times of batches 1 and 2, including scheduling, scoring and output. It is not an arithmetic average of batch rates.

| Case | GPU | Control k histories/s | Current k histories/s | Matched gain |
|---|---|---:|---:|---:|
| RT07575 | RTX 2080 Ti | 271.48 | 319.29 | 17.61% |
| RT07575 | RTX A6000 | 455.90 | 547.89 | 20.18% |
| RT06423 | RTX 2080 Ti | 289.25 | 350.55 | 21.19% |
| RT06423 | RTX A6000 | 476.97 | 598.15 | 25.41% |
| 20022516 | RTX 2080 Ti | 290.58 | 347.23 | 19.50% |
| 20022516 | RTX A6000 | 470.03 | 583.74 | 24.19% |

## Individual batch rates

Batch 0 includes first-use package loading inside the measured call. Context and stopping-table construction precede the call, so this column is not a complete process cold-start benchmark.

| Case | GPU | Arm | Batch 0 k/s | Batch 1 k/s | Batch 2 k/s |
|---|---|---|---:|---:|---:|
| RT07575 | RTX 2080 Ti | control | 280.47 | 275.16 | 267.90 |
| RT07575 | RTX 2080 Ti | candidate | 316.11 | 319.88 | 318.70 |
| RT07575 | RTX A6000 | control | 453.83 | 465.91 | 446.31 |
| RT07575 | RTX A6000 | candidate | 522.95 | 547.82 | 547.95 |
| RT06423 | RTX 2080 Ti | control | 279.92 | 289.08 | 289.42 |
| RT06423 | RTX 2080 Ti | candidate | 337.64 | 350.46 | 350.64 |
| RT06423 | RTX A6000 | control | 450.30 | 476.72 | 477.21 |
| RT06423 | RTX A6000 | candidate | 558.52 | 597.36 | 598.94 |
| 20022516 | RTX 2080 Ti | control | 287.66 | 290.12 | 291.04 |
| 20022516 | RTX 2080 Ti | candidate | 342.49 | 346.76 | 347.71 |
| 20022516 | RTX A6000 | control | 447.20 | 469.33 | 470.74 |
| 20022516 | RTX A6000 | candidate | 550.70 | 582.89 | 584.60 |

## Implementation checks

All batches pass energy, coverage, completion and zero-overflow gates. Per-spot inputs, nested non-energy integer ledgers, live-ion and discrete EM/Urban/nuclear/elastic audits match between arms. Maximum relative dose L1 is 3.00020704e-07; maximum relative energy residual is 5.74965648e-08. These checks establish matched GPU implementation consistency; no new TOPAS gamma is calculated here. The preceding eight-preset and generic regression remains documented in the upstream report.

Raw inputs, exact batch times, binary/source provenance, per-batch quality reports and matched checks are retained under `tmp/proton_mb_20m_batches_20261016/`, including `inputs.json`, `provenance.json`, `SUMMARY.json`, `local/runs/` and fetched A6000 audit records. Remote dense dose outputs remain on A6000. No production code, physical parameter or default changes are made by this timing follow-up.
