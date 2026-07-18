---
name: verify
summary: Exercise CarbonGPU SOBP through the real CLI and inspect MHD output.
---

1. Build the SYCL CLI:
   `cmake -S . -B /tmp/maigo-verify-build -DCARBON_ENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=/home/intel/oneapi/compiler/2026.1/bin/icpx -DCMAKE_BUILD_TYPE=Release`
   then `cmake --build /tmp/maigo-verify-build --parallel 4`.
2. For a quick run, copy the SOBP plan/config to `/tmp`, reduce the L4 history counts while preserving 21 layers, and redirect `out/sobp_benchmark/` paths to `/tmp/verify_sobp/`.
3. On this Linux Arc B580 host use `ONEAPI_DEVICE_SELECTOR=opencl:gpu`; Level Zero may return `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`.
4. Run `/tmp/maigo-verify-build/carbon_mc --config <temp-config> --device gpu` and require finite energy balance plus zero secondary/cascade queue overflow.
5. Convert the emitted voxel Gy CSV with `validation/scripts/sparse_dose_to_mhd.py`; verify `DimSize = 50 50 50`, `ElementSpacing = 3 3 3`, and a 500000-byte float RAW file.
6. Probe source validation by enabling both flat and emittance sources; the CLI should exit nonzero with the mutual-exclusion error.
