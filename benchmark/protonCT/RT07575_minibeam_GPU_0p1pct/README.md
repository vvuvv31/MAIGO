# RT07575 proton CT minibeam: 0.1% of the 4 Gy plan

Requested total: 236,086,432 incident primary protons (before copper), 487 active spots, 119 shards of at most 2M histories. This is an independent new simulation; it does not include or reuse the previous 23,608,643 histories. The optimized incident weights are allocated with the largest-remainder method and exact integer totals.

The current GPU binary SHA, FP32 ON / FP64 OFF build, source, CT geometry, source optics, group32, and extended material electron response bank match the validated 0.01% campaign. Only histories, seeds, source count channels and output paths differ. Inputs link to the frozen sibling campaign. All existing transport/EM, energy and overflow gates remain active.

Run or resume from the repository with the same group size:

```bash
python3 benchmark/protonCT/RT07575_minibeam_GPU_0p1pct/run_gpu.py --group 16
```

Sixteen shards share a process and retained response data. Each shard has separate scoring and quality outputs; all shards in a group must pass before that group is accepted. `runner.lock` prevents concurrent runners. `launch.json` records the detached launch; `campaign_status.json`, `campaign.log`, `progress.py` and per-group logs show progress. Completed group records verify saved dose checksums on resume.

The runner produces `results/dose_4Gy.mhd` and `results/summary.json`, then automatically computes the same 3D local/global gamma definitions, independent-shard uncertainty, geometric peak/valley means, plots and `REPORT_zh.md`. `results/convergence.json` directly compares this independent campaign to the unchanged 0.01% reference campaign. Normalization uses the requested optimized proton count, without dose fitting or registration. GPU dose atomics are FP32; offline dose summation is FP64 on CPU.

All reported uncertainty excludes finite TOPAS DIJ noise and finite electron response-bank error. The same research approximations remain, including disabled CT neutron transport. The reference voxel spacing is 0.5 x 0.5 x 2 mm; subvoxel gamma uses trilinear interpolation.

Initial directory setup failed before transport because the default cross-section loader requires the repository `data/` link. Logs are preserved in `archive/missing_data_link`; the link was restored before the accepted campaign. No failed dose is included.
