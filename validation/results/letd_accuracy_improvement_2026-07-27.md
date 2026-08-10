# LET_d accuracy improvement (2026-07-27)

## Root cause

CUDA LET runs used `secondary_queue_capacity: 0` → auto ≈ **4 × histories**.  
At 100k histories / 400 MeV/u this capped the queue at ~400k slots and produced:

- secondary overflow ~56k  
- **cascade overflow ~366k**

Overflow cascade products never transported; their energy was residual-localized.  
Effect: **all-hadron LET peak too high, fragment tail too low** vs TOPAS.

## Fixes

1. **Queue capacity**: LET configs and `run_gpu_letd_energy_sweep.py` now use  
   `max(2.5e6, 25 × histories)` so overflow = 0 at 100k.  
2. **Cascade sampling**: tight energy band (≤15%, abs cap 25 MeV/u), no 40–80%  
   expansion; scale clamp `[0.85, 1.18]`; unmatched package → local residual only.  
3. Default LET config enables **particle-specific stopping power**.

## Monoenergetic all-hadron LET_d (median |rel| vs TOPAS, dose≥1% mask)

| E (MeV/u) | Before | After | Δ |
|----------:|-------:|------:|--:|
| 100 | 1.21% | 1.24% | +0.03 |
| 200 | 3.72% | 3.77% | +0.05 |
| 300 | **9.93%** | **7.04%** | **−2.9** |
| 400 | **7.25%** | **4.12%** | **−3.1** |

400 MeV post-peak mean GPU/TOPAS: **0.70 → 0.88**; peak ratio **~1.16 → 1.08**.

Primary C-12 LET unchanged (already excellent).

## SOBP 50–100 mm all-hadron

| | Before | After |
|--|-------:|------:|
| median \|rel\| | 0.79% | 0.82% |
| mean bias | −0.23% | −0.17% |

Still well matched; small changes within statistical noise.

## Plots

- `out/letd_summary_view/gpu_vs_topas_letd_improved.png`  
- `out/letd_energy_sweep/comparison_large_queue_v1/`  
- `out/letd_sobp/comparison_large_queue_v1/`

## Remaining

High-E all-hadron still has residual error (300 MeV ~7%, 400 MeV ~4%).  
Next: parent-species–conditioned cascade final states / rates (see birth-spectrum work).
