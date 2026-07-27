# Birth-spectrum energy suite report

Date: 2026-07-27  
Histories: 10,000 GPU (seed 20260727)  
Device: NVIDIA TITAN RTX (oneAPI CUDA)  
Packages: Geant4 11.3.2 / TOPAS 4.2.p3 400 MeV cascade + aligned primary  

Full CSV/JSON suite is generated locally under  
`validation/results/birth_spectrum_energy_suite/` (gitignored bulk results).  
Re-run:

```bash
ONEAPI_DEVICE_SELECTOR=cuda:gpu \
python3 validation/scripts/run_birth_spectrum_energy_suite.py \
  --carbon-mc build/oneapi-release/carbon_mc \
  --device cuda --histories 10000
```

## Cascade sampling change

Replaced fixed 8-event index window with energy-bandwidth selection:

- relative half-widths 5% → 10% → 20% → 40% → 80% (floor 2 MeV/u);
- fallback to nearest event;
- event KE scale clamped to `[0.25, 4]`.

## gen0 vs 400 MeV primary package

| E (MeV/u) | proton yield | proton meanKE | he4 yield | he4 meanKE |
|----------:|-------------:|--------------:|----------:|-----------:|
| 100 | 0.135 | 0.255 | 0.225 | 0.219 |
| 200 | 0.433 | 0.477 | 0.515 | 0.466 |
| 300 | 0.729 | 0.715 | 0.789 | 0.710 |
| 400 | **1.002** | **0.992** | **1.013** | **1.002** |

Interpretation: only 400 MeV is a fair package match. Lower energies under-produce relative to a 400 MeV reaction library because fewer primaries interact and products are softer — not a gen0 sampling regression.

## gen1 vs full cascade package (coarse)

| E | proton yield | he4 yield | he4 meanKE |
|--:|-------------:|----------:|-----------:|
| 100 | 0.013 | 0.020 | 0.095 |
| 200 | 0.101 | 0.107 | 0.129 |
| 300 | 0.271 | 0.252 | 0.222 |
| 400 | 0.435 | 0.363 | **0.317** |

These ratios mix different denominators (GPU first-generation cascade births vs all TOPAS cascade products of all generations/projectiles). They motivate **parent-energy and projectile-filtered** references, not immediate case-specific LET scaling.

## Package parent-energy structure (400 MeV cascade CSV)

He-4 product mean MeV/u rises strongly with parent MeV/u (e.g. ~5 at 20–30 MeV/u parent, ~240 at 390–400 MeV/u). Soft high-parent bins (E>400 for light projectiles) are low-energy residuals.

## Next diagnostics

1. Split GPU birth histograms by generation for parent/product MeV/u.  
2. Filter TOPAS cascade products by projectile Z/A and incident MeV/u before compare.  
3. Audit cascade macroscopic XS / interaction rate vs TOPAS.  
4. Only then change binary cascade package layout if residual remains.
