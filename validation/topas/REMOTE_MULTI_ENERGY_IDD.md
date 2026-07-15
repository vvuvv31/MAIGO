# Remote TOPAS total IDD: 100 / 300 / 400 MeV/u

Target host: `v@192.168.31.5` working tree `~/gpu`  
Max threads: **56**  
Do **not** run TOPAS in WSL.

## 1. Sync parameter files from Windows

From the project root (adjust host if needed):

```bat
scp validation/topas/carbon_100MeVu_water*.txt ^
    validation/topas/carbon_300MeVu_water*.txt ^
    validation/topas/carbon_400MeVu_water*.txt ^
    validation/topas/run_multi_energy_idd_remote.sh ^
    v@192.168.31.5:~/gpu/validation/topas/
```

Or `rsync -av` the whole `validation/topas/` directory (extensions only needed if rebuilding).

## 2. Smoke (100 histories each)

```bash
ssh v@192.168.31.5
cd ~/gpu
bash validation/topas/run_multi_energy_idd_remote.sh all smoke
```

Expect logs:

- `validation/topas/output/e100-smoke_topas.log`
- `validation/topas/output/e300-smoke_topas.log`
- `validation/topas/output/e400-smoke_topas.log`

## 3. Development (100000 histories, 56 threads)

Detach so a Windows reboot cannot kill the jobs:

```bash
cd ~/gpu
nohup bash validation/topas/run_multi_energy_idd_remote.sh 100 development \
  > validation/topas/output/e100-development_nohup.log 2>&1 < /dev/null &
nohup bash validation/topas/run_multi_energy_idd_remote.sh 300 development \
  > validation/topas/output/e300-development_nohup.log 2>&1 < /dev/null &
nohup bash validation/topas/run_multi_energy_idd_remote.sh 400 development \
  > validation/topas/output/e400-development_nohup.log 2>&1 < /dev/null &
```

Raw EnergyDeposit CSVs:

- `output/e100_development_energy_deposit.csv`
- `output/e300_development_energy_deposit.csv`
- `output/e400_development_energy_deposit.csv`

## 4. Pull results to Windows

```bat
scp v@192.168.31.5:~/gpu/validation/topas/output/e100_development_energy_deposit.csv validation\topas\output\
scp v@192.168.31.5:~/gpu/validation/topas/output/e300_development_energy_deposit.csv validation\topas\output\
scp v@192.168.31.5:~/gpu/validation/topas/output/e400_development_energy_deposit.csv validation\topas\output\
scp v@192.168.31.5:~/gpu/validation/topas/output/e*-development_topas.log validation\topas\output\
```

## 5. Normalize locally

```bat
validation\scripts\postprocess_multi_energy_topas.cmd
```

Writes:

- `validation/results/topas_100MeVu_development.csv` (+ metadata)
- `validation/results/topas_300MeVu_development.csv` (+ metadata)
- `validation/results/topas_400MeVu_development.csv` (+ metadata)

## 6. Compare with GPU multi-energy suite

GPU baselines already exist at 100k:

- `validation/results/windows_b580_multi_energy_{100,200,300,400}MeVu_idd_100k.csv`

After TOPAS normalization, re-run or extend:

```bat
python validation\scripts\analyze_multi_energy.py ^
  --cases 100=validation\results\windows_b580_multi_energy_100MeVu_idd_100k.csv ^
         200=validation\results\windows_b580_multi_energy_200MeVu_idd_100k.csv ^
         300=validation\results\windows_b580_multi_energy_300MeVu_idd_100k.csv ^
         400=validation\results\windows_b580_multi_energy_400MeVu_idd_100k.csv ^
  --topas-200 validation\results\topas_200MeVu_development.csv ^
  --output-metrics validation\results\windows_b580_multi_energy_100_400.metrics.json ^
  --output-plot validation\results\windows_b580_multi_energy_100_400.png
```

Per-energy TOPAS comparison can use `compare_depth_dose.py` for each pair.

## Notes

- Seed remains `20260714` from the base water parameter file.
- Histories = **100000** to match GPU multi-energy runs (legacy 200 MeV development was 10k).
- Standard scorers only (EnergyDeposit / DoseToMedium / primary C-12); no custom extension required.
