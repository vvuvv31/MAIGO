# CT validation artifacts (7c)

Absolute MeV/primary IDD only — **no global dose scale**.

## One-shot baseline

From repo root (oneAPI env + built `carbon_mc`):

```bat
scripts\run_windows_b580_ct_baseline.cmd
python validation\scripts\run_ct_baseline.py
python validation\scripts\run_ct_baseline.py --skip-gpu
```

After CT physics or grid prep changes:

1. `python validation/scripts/prepare_ct_grid.py --schneider-file ct/HUtoMaterialSchneider.txt`  
   (writes **CCTG v3**: density + section + za_rel + I_eV)
2. Re-run baseline (not `--skip-gpu`)

SP model: `SP_water(E) * f_E(za_rel, I, E) * density` (absolute MeV/primary).

## Key files

| File | Role |
|------|------|
| `gpu_e150_patient_multimat_idd.csv` | primary-only multimat |
| `gpu_e150_patient_multimat_secondary_idd.csv` | +secondary/cascade |
| `topas_e150_patient_idd.csv` | TOPAS reference (from development CSV) |
| `compare_patient_primary.metrics.json` | primary vs TOPAS |
| `compare_patient_secondary.metrics.json` / `compare_patient_mass_sp.metrics.json` | secondary vs TOPAS |
| `ct_baseline_summary.metrics.json` | three-curve summary |

Metrics always include at least: `delta_R80_mm`, `integral_rel_diff`, `nrmse`.
