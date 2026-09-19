# Water calibration for the staged ion Fermi--Eyges model

This directory calibrates the experimental common Fermi--Eyges (FE) transport
for `p`, `d`, `t`, and He-4.  It does not alter the retained YAML choice between
`multiple_scattering_model: highland` and `fermi_eyges`.

## Reference matrix

The TOPAS matrix contains 24 pure-EM `G4_WATER` slabs:

- species: proton, deuteron, triton, He-4;
- energy: 50, 150, and 300 MeV/u;
- thickness: 1 and 10 mm;
- 250,000 histories per case;
- `g4em-standard_opt4 + g4decay`, 0.05 mm production cut and 0.05 mm
  water maximum step.

The authoritative run is local Slurm job 7026. It sets
`Ph/Default/EMRangeMax = 10 GeV` and supersedes job 7024, whose default
600 MeV EM table ceiling is invalid for the 1200 MeV He-4 case. Job 7026
completed all 24 cases in 15 seconds using 192 CPUs. The batch MaxRSS was
approximately 0.96 GiB; the
reproduction script now requests 8 GiB rather than 48/160 GiB.  Reference and
GPU outputs are under:

```text
/mnt/sda/wuwei/fe_species_water_em10gev_20260919
```

The reduced request was also exercised directly with a fresh 300 MeV proton,
10 mm water, 250k-history run: local job 7025 completed in 7 seconds with
16 CPUs, an 8 GiB request, approximately 42 MiB MaxRSS, exit code 0, and both
phase-space files present.

Run `prepare_topas_water_slab.py`, submit `run_topas_water_slab.sbatch`, fit
with `fit_water_species_fe.py`, prepare/run the GPU replay, and finish with
`compare_water_species_fe.py`.  All GPU validation uses FP32 dose scoring.
If local scheduling is unavailable, stage the same directory on the allocated
`cpu188` node through `wuwei@10.10.10.4` and execute
`run_topas_water_slab_remote.sh` there.  The retired personal-machine route is
not used.

## Model and fitted nodes

Each species uses a linearly interpolated 50/150/300 MeV/u table for the FE
core scattering energy, Poisson tail rate per water millimetre, and tail
scattering energy.  Values outside that interval currently hold the nearest
endpoint.  C12 and uncalibrated ions retain the existing `9.9 / 0.0025 / 2.4`
candidate.

| species | core energy at 50/150/300 | tail rate at 50/150/300 (mm^-1) | tail energy at 50/150/300 |
|---|---|---|---|
| p | 12.12504 / 11.86204 / 11.72358 | 0.0024267 / 0.0022461 / 0.0019987 | 6.78092 / 7.80957 / 8.80336 |
| d | 10.72516 / 10.58235 / 10.44330 | 0.0030699 / 0.0031040 / 0.0032738 | 3.21950 / 3.06519 / 2.78422 |
| t | 10.67358 / 10.45643 / 10.30587 | 0.0030336 / 0.0031511 / 0.0033817 | 3.28221 / 2.92718 / 2.60765 |
| He-4 | 10.37869 / 10.34273 / 10.24029 | 0.0031598 / 0.0034918 / 0.0034359 | 3.09083 / 2.65185 / 2.57552 |

For CT materials the core uses the material radiation length directly.  The
water-fitted tail rate is scaled without fitting CT dose:

```text
lambda_material = lambda_water * X0_water_mm / X0_material_mm
X0_water_mm = 360.8297746
```

The tail angular scale remains the water-fitted species value.  This is a
first material extrapolation, not an independent bone/lung calibration.

## Validated range and limitation

The exact-endpoint audit is recorded in `calibration_manifest.json` and by
`audit_water_species_em.py`. Across all 24 cases the worst mean-loss error is
0.765%, the worst exit-spectrum-width error is 4.302%, and the median spectrum
width ratio is 1.00314. The worst angle-variance, displacement-variance and
covariance errors are 6.328%, 17.25% and 12.71%; the latter two occur in thin
slabs where the lateral displacement is very small.

For He-4 at 300 MeV/u through 10 mm, TOPAS/GPU mean losses are
3.50323/3.52421 MeV/u and exit-energy standard deviations are
0.174162/0.174193 MeV/u. This closes the apparent 45% stopping deficit in the
retired reference: it was a TOPAS table-range error, not a GPU stopping-scale
error. No empirical He-4 stopping scale is used.

The 1 mm slabs expose a structural limitation: median q68 is 1.0749 while
median q99 is 0.9158, even though q99.9 is 0.9987.  A single Gaussian core plus
one Gaussian Poisson-tail family cannot simultaneously reproduce the UrbanMsc
core, shoulder, and far tail at this thickness.  Do not compensate this by
retuning the three parameters against full-dose results.  A future correction
requires an additional shoulder/tail shape component and new held-out checks.

The B1--B4 regression passed all quality, energy, and overflow checks.  Changes
were small because those cases are primary-C12 dominated.  B3 core/halo median
fit errors improved from 0.512/14.274% to 0.453/12.925%.  B4 lung and bone were
mixed by depth, so the radiation-length CT scaling remains experimental.

`calibration_manifest.json` pins the authoritative paths, hashes, physics,
case matrix and fitted nodes without committing multi-gigabyte result files.
The batch and remote launchers enumerate only `manifest.json["cases"]`; they do
not traverse analysis or `gpu/` directories. `audit_water_species_em.py`
produces the per-case energy/scattering report and confidence intervals.

The downstream full-chain screen is reproduced with
`run_species_fe_minibeam_ab.py`, summarized with
`summarize_species_fe_minibeam_ab.py`, and its optional low-memory transport
energy diagnostic is analyzed by `analyze_minibeam_energy_band_roi.py`. The
diagnostic defaults off and does not change formal three-energy YAML files.
