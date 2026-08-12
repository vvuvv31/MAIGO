# Physics tables

Runtime inputs used by MAIGO. Tables are Geant4 11.3.2-derived CSVs unless noted.

| Pattern | Use |
|---------|-----|
| `stopping_power_*.csv` | Continuous energy loss (MeV/mm vs MeV/u) |
| `c12_inelastic_cross_sections_*.csv` | C-12 macroscopic nuclear XS (1/mm) |
| `ion_stopping_power_*.csv` | Per-isotope SP for fragment / LET scoring |
| `let_delta_electron_fraction_*.csv` | Optional delta-electron fraction for LET_d |
| `packages/*.bin` | Reaction, cascade, neutral, and soft-tissue final-state packages |
| `copper_*.bin` | Minibeam Copper reaction / neutral packages |

`stopping_power_water.csv` is the default light-weight water table for the
example configs. Prefer `*_geant4_11_3_2.csv` for production-grade runs.

Runtime nuclear packages live under `data/packages/` (not `validation/results/`).
`validation/` retains generation intermediates, ablation variants, and benchmark
outputs only.

Small reusable TOPAS/TPS spot-plan fixtures live under `data/plans/`. These are
runtime/test inputs; larger patient and dose-validation plans remain in the
local ignored `benchmark/` and `validation/` workspaces.
