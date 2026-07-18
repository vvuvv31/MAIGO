# 3 cm field, 5–10 cm SOBP benchmark

This case runs the same 21-layer carbon-ion plan in TOPAS and CarbonGPU:

- water phantom: `150 × 150 × 150 mm³`;
- flat square field: `30 × 30 mm²`;
- SOBP: `50–100 mm` water depth;
- dose grid: `50 × 50 × 50`, `3 mm` isotropic;
- histories: `10,000,000` total, using the per-layer counts in
  `spots_sobp_water_5_10cm.txt`;
- source energy spread: `1%` RMS per layer;
- seed: `20260716`.

## TOPAS reference

The accepted run used TOPAS 4.1.p1, Geant4 11.1.3, and 56 threads on `vv`.
It completed in `52348.2 s` (`14.5412 h`), or `191.03 histories/s`.
Metadata and source hashes are recorded in
`validation/results/sobp_water_3cm_5_10cm_3mm_topas.metadata.json`.

Run again on the TOPAS host:

```bash
cd ~/gpu
bash validation/topas/run_sobp_benchmark_remote.sh
```

Fetch the existing result:

```bash
scp v@10.10.10.216:~/gpu/validation/topas/output/sobp_water_3cm_5_10cm_3mm_dose3d_development.csv \
  validation/topas/output/
scp v@10.10.10.216:~/gpu/validation/topas/output/sobp_water_3cm_5_10cm_3mm_remote.log \
  validation/topas/output/
```

Convert TOPAS `DoseToMedium Sum` to `Gy/primary` MHD. The scale is `1/10,000,000`:

```bash
python3 validation/scripts/sparse_dose_to_mhd.py \
  validation/topas/output/sobp_water_3cm_5_10cm_3mm_dose3d_development.csv \
  out/sobp_benchmark/topas_sobp_dose.mhd \
  --input-format topas --shape 50 50 50 --spacing-mm 3 3 3 \
  --origin-mm -73.5 -73.5 1.5 --scale 1e-7 --units Gy/primary
```

## CarbonGPU

The GPU source uses the same TOPAS-format energy-layer file and samples a flat
`±15 mm` rectangle in the local beam frame. Build the oneAPI Windows release,
then run:

```bat
scripts\run_windows_b580_sobp_benchmark.cmd
```

The runner writes:

- `out/sobp_benchmark/gpu_sobp_benchmark.log`;
- `out/sobp_benchmark/gpu_sobp_voxels_Gy.csv`;
- `out/sobp_benchmark/gpu_sobp_dose.mhd`;
- `out/sobp_benchmark/gpu_sobp_dose.raw`.

The CarbonGPU log reports summed transport time across all 21 energy layers and
the resulting histories/s. Queue overflow counters and energy balance must be
checked before accepting the timing.

### 100k Arc B580 speed check

A proportionally scaled 100,000-primary plan was run through the OpenCL SYCL
backend on the Arc B580. The Level Zero backend on this Linux installation
returned `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`; this is a runtime/backend issue,
not a transport validation result.

- transport elapsed: `3.410379 s`;
- end-to-end wall time: `3.792798 s`;
- transport throughput: `29,322.25 histories/s`;
- wall throughput: `26,365.76 histories/s`;
- energy-balance error: `2.14e-5`;
- secondary/cascade queue overflows: zero.

Relative to the accepted 56-thread TOPAS run (`191.03 histories/s`), this is a
`153.5×` transport-throughput or `138.0×` end-to-end speedup. Linear projection
to 10 million primaries is about `341 s` transport / `379 s` wall, but the formal
10M run should be timed directly because queue sizes and fixed per-layer startup
costs differ.

The 100k MHD result is under `out/sobp_benchmark_100k/`. Against the 10M TOPAS
reference it gives `-0.683%` integral difference, `3.87%` normalized voxel L1,
`8.31%` high-dose mean absolute difference, and Pearson `r=0.99859` above 1% of
the TOPAS maximum.

## Dose comparison

After both MHD files exist:

```bash
python3 validation/scripts/compare_mhd_dose.py \
  --topas out/sobp_benchmark/topas_sobp_dose.mhd \
  --gpu out/sobp_benchmark/gpu_sobp_dose.mhd \
  --output out/sobp_benchmark/gpu_vs_topas.metrics.json
```

The comparison checks grid identity and reports integral difference, normalized
voxel L1/RMSE, high-dose mean absolute error/correlation, and central-axis SOBP
flatness over `50–100 mm`.
