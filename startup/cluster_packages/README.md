# Cluster 1M water primary + cascade packages

One TOPAS `CarbonCascadeNtuple` job at 400 MeV/u, 1M histories, then
`prepare_*` / `compile_*` write both CRPKG and CCAS binaries.

Submitted on `wuwei@10.10.10.4`:

```
/LustreData6/home/wuwei/carbonGPU/maigo_packages_1M
sbatch submit_water_1M_cascade.slurm
```

Job first builds a carbon-enabled TOPAS (cluster stock binary has no
`CarbonCascadeNtuple`), then runs 1M, then compiles:

- `packages/topas_400MeVu_water_inclxx_1M_primary_3d.bin`
- `packages/topas_400MeVu_water_inclxx_1M_cascade_3d.bin`

Continuation ntuple (records surviving projectile after ionInelastic):
rebuild TOPAS on the login node, then

```
sbatch submit_water_1M_cont.slurm
```

writes `run_water_1M_cont/` and `packages/topas_400MeVu_water_inclxx_1M_cont_*.bin`
without overwriting the first 1M packages.

## Energy-grid 1M (100–400 MeV/u, step 50)

Dedicated first-inelastic packages so a 400 MeV/u long beam is not sliced
into empty/biased low-energy bins. Submit on the same tree:

```
cd /LustreData6/home/wuwei/carbonGPU/maigo_packages_1M/energy_grid
bash submit_all.sh
```

Array tasks 0–6 → 100, 150, 200, 250, 300, 350, 400 MeV/u. Each writes
`packages/topas_<E>MeVu_water_inclxx_1M_{primary,cascade}_3d.bin` and
`packages/topas_<E>MeVu_water_inclxx_1M_primary_bin_yields.json`.

Coverage analysis (which bins of each dedicated package are trustworthy)
is described in `energy_grid/GEMINI_COVERAGE.md`.
