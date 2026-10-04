# RT07575 joint electron pilot

See [REPORT_zh.md](REPORT_zh.md) for the completed 0.1% full plan and scope limitations.

Only two configuration keys select the pilot:

```yaml
material_electron_joint_pilot_file: /mnt/sdb/wuwei/MAIGO_minibeam/out/proton_joint_pilot_20261003/package/joint_pilot.bin
material_electron_joint_pilot_sha256: a20883e9f73e9bba5b09bf88654bcd2b9c54a6e8fac76990188d006b52751a73
```

The original RT07575 82-table packed bank, plan sources, per-batch counts/seeds and physics settings are retained. All original 63 index entries were verified unchanged after path resolution (`index_compatibility.json`); the loader accepts this exact index SHA only. Every package/binary/config/source is pinned. The default preset is unchanged.

`launch.py` froze the successfully built FP32 binary and ran `run_gpu.py --group 119`. `compare.py` uses the unchanged historical TOPAS dose and gamma method; `compare_joint.py` compares the completed packed control and the new hybrid. Existing output is preserved, and completed checkpoint groups are reused; a fresh simulation requires an isolated new output directory. Initial failed launch before any histories is retained under `rejected_missing_data_link/`.

The `preview_12batches/` results are an earlier 23,807,040-primary preview; use `results/` for the final 236,086,432-primary dose and comparison. The new method covers one lung response-density node, retains native electron spatial spread and exact photon-family graphs, and falls back to the original bank at unsupported material/energy/interface states. It is not a complete replacement or formal patient acceptance.
