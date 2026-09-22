# C6 EM-only verdict (2026-09-23): CARBON_EM_ONLY_DOSE = FAIL

Gate runner: dose_gate.py vs acceptance.yaml (frozen BEFORE comparisons).
Reference: TOPAS 4.2.p3 job 7374 dose.bin (10M, single central spot).
GPU: 3 independent seeds (202609251/2/3), current binary 6d79ecf7,
10M histories each, all URBAN_RUN_QUALITY zeros.

## Dose gate (single central spot, 250 MeV/u)

| depth | total | peak | valley | PVDR | verdict |
|---|---|---|---|---|---|
| 20 | +0.62% | +0.67% | +4.5% | −3.7% | FAIL (valley/PVDR) |
| 40 | +0.40% | +0.15% | +4.3% | −4.0% | FAIL |
| 60 | +0.58% | −0.19% | +6.3% | −6.1% | FAIL |
| 80 | +0.72% | −0.02% | +6.3% | −5.9% | FAIL |
| 100 | +0.84% | +0.38% | +5.8% | −5.2% | FAIL |
| 120 | +3.50% | +2.97% | +5.8% | −2.7% | FAIL |
| FWHM 127.75 vs 128.25 (−0.4%) | PASS |
| R80 126.12 vs 126.62 (−0.500 mm, gate <0.5) | marginal FAIL |

Gates: total 0.5%, peak 2%, valley/PVDR 3%, FWHM 5%, R80 0.5mm.
Equivalence rule: |rel| + 2-SE < gate (TOPAS SE by equal-N parity).

## What is exonerated (with evidence, not assumption)

1. Water MCS transport: replay phase-space gate PASSES on the real TOPAS
   .phsp (angVar 0.996–1.025, disVar 1.003–1.016, q999 1.008–1.027,
   survival identical to 4 decimals; replay_gate_C.json).
2. Primary fluence per row matches to ~1% peak and +0.4–1.0% valley at
   40–120 mm (1.0–1.4M matched histories). Fluence matched + dose not →
   the residual is NOT water-MCS (prompt C6.2 discriminator, now with
   evidence).
3. Stability: valley residual identical B (+6.0%) → C (+6.3%) across ALL
   C-round changes (RNG domain, MSC range, secondary state, conversion
   math, par>=0 inversion). An MCS-driven residual would have moved.
4. Cu tail is weak-side (Cu slab: ang 0.956, q999 0.90): cannot produce a
   valley EXCESS. Electron-transport difference is wrong-sign (local
   deposit would LOWER our valley, observed HIGHER). Scoring grids match
   (0.1 x, 0.25 depth, 100 transverse) and sources are byte-identical.
5. Energy spectra at planes match (mean 0.1–0.4%, rms 0.2%): no missing
   straggling in transported energy.

## Residual characterization (RESIDUAL_UNRESOLVED)

- Single phenomenon: ~0.6% of total energy sits in valley rows in GPU vs
  TOPAS (totals excess == valley excess quantitatively). Peak matched.
- Longitudinal: peak −0.26 mm (C4 range fix improved B −0.50 mm toward
  TOPAS), height matched, narrower distally; R80 −0.500 mm (marginal).
- Narrowed suspects (not proven): per-particle deposit in valley rows
  (water dE/dx? delta-production rate? low-E secondary handling?), distal
  tail integral (stoppers/range-straggling population, not covered by
  plane survivors), air-gap/slit-edge penumbra transport.
- Explicitly NOT tuned: stopping scale 0.9958 left as-is (0.4% effect,
  wrong direction for the excess, and the no-tuning rule); no MCS/tail/
  stopping/normalization changes to chase the residual.

## Required follow-ups (precise blockers for C7)

F1. Cu-transport isolation: TOPAS single-spot collimator exit does not
    exist (only array exit); GPU array exit not run. Need one of the two
    for a clean Cu verdict (unit-level Cu slab: ang 0.956, q999 0.90,
    dE +3.5% recorded in cu_slab_gate.json).
F2. Electron/loss isolation: needs carrier-separated dose or a TOPAS
    local-deposit diagnostic run (no local TOPAS binary; cluster only).
F3. Held-out energy/slit for full-physics (C7.3) still requires C6 pass.
