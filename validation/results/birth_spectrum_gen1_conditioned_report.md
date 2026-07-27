# Generation-split birth spectra and conditioned gen1 diagnosis

Date: 2026-07-27  
GPU: 10k histories, 400 MeV/u, TITAN RTX  
Commit series: birth scorer with `species × generation` histograms + joint parent×product MeV/u  

## What was implemented

1. **Generation-split** birth histograms (`_mevu`, `_depth`, `_costheta`, `_parent_mevu`, `_parent_z`)  
2. **Joint** `_parent_product_mevu.csv` for conditioned spectrum comparison  
3. TOPAS prep filters: `--projectile-z/a`, `--parent-mevu-min/max`, `--parent-z-min/max`  
4. Scripts:  
   - `validation/scripts/run_gen1_conditioned_compare.py`  
   - `validation/scripts/compare_birth_joint_parent_product.py`  

## gen1 He4 parent composition (GPU)

| parent Z | count | fraction |
|---------:|------:|---------:|
| 1 | 3165 | 39% |
| 2 | 2975 | 37% |
| 5 | 768 | 10% |
| 6 | 598 | 7% |
| other | ~500 | 7% |

Mean parent MeV/u for gen1 He4 ≈ **213**.  
**Light projectiles (p, He) dominate gen1 He4 births**, not secondary C-12.

## Joint parent-E spectrum (He4 gen1)

Comparing GPU mean product MeV/u in each parent-energy bin to TOPAS package:

| Reference slice | Behavior |
|-----------------|----------|
| TOPAS **all** projectiles | GPU much softer at high parent E (ratio → 0.03 at 390–400) — **misleading** (package high-E bin is C12-dominated hard He4) |
| TOPAS parents **Z≤2** | GPU **harder** than residual-like He4 (ratio often 5–10) |
| TOPAS parents **C12** | GPU **softer** than fragmentation He4 (ratio ~0.15–0.35) |

Proton gen1 by parent energy vs all-TOPAS package: ratio **≈0.85–1.05** across most bins → light-ion product sampling is comparatively healthy.

## Interpretation

1. The earlier “gen1 He4 mean KE ratio 0.31 vs full cascade package” is largely a **parent-species / parent-energy mixture** effect, not a single scalar sampling bug.  
2. Remaining physics issues to chase separately:  
   - **C12 secondary cascade**: under-hard He4 relative to package (ratio ~0.25)  
   - **Light-parent cascade**: over-hard He4 relative to residual package (ratio ≫1) — possible over-scaling of soft residual topologies  
   - **Parent mix**: high cascade rate on Z=1/2 vs Z=6 may bias all-hadron LET  
3. Primary (gen0) vs primary package remains closed at 400 MeV/u.

## Next engineering steps

1. Joint hist (or filter) by **parent Z/A**, not only parent MeV/u.  
2. A/B: disable or retune `cascade_event_energy_scale` for residual-like events.  
3. Cascade **XS / interaction rate** by projectile species vs TOPAS.  
4. Only after 1–3, change binary cascade package layout if residual remains.
