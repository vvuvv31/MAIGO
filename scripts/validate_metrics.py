#!/usr/bin/env python3
import os, sys, math, json
import numpy as np

configs = [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw", "out/beam_100MeVu_inelastic/quality_report.json", 40.0),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw", "out/beam_200MeVu_inelastic/quality_report.json", 120.0),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e300/topas_emittance_inelastic_e300.bin", "out/gpu_inelastic_e300/voxel_dose.raw", "out/beam_300MeVu_inelastic/quality_report.json", 220.0),
    (400, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e400/topas_emittance_inelastic_e400.bin", "out/gpu_inelastic_e400/voxel_dose.raw", "out/beam_400MeVu_inelastic/quality_report.json", 340.0)
]

nx, ny, nz = 400, 400, 800
dx, dy, dz = 0.2, 0.2, 0.5

print("="*130)
print(f"{'Energy (MeV/u)':<14} | {'Peak Shift (mm)':<16} | {'Peak Diff (%)':<15} | {'ROI Int Diff (%)':<18} | {'Full Int Diff (%)':<18} | {'Energy Residual':<16}")
print("="*130)

all_passed = True
errors = []

for E, tpath, gpath, qpath, max_z in configs:
    if not os.path.exists(tpath):
        err = f"FAIL: Missing TOPAS reference binary for E={E}: {tpath}"
        print(err)
        errors.append(err)
        all_passed = False
        continue

    if not os.path.exists(gpath):
        err = f"FAIL: Missing GPU output binary for E={E}: {gpath}"
        print(err)
        errors.append(err)
        all_passed = False
        continue

    # 1. Quality report verification
    if not os.path.exists(qpath):
        err = f"FAIL: Missing quality report for E={E}: {qpath}"
        print(err)
        errors.append(err)
        all_passed = False
        rel_residual_str = "MISSING"
    else:
        with open(qpath, "r") as f:
            try:
                rep = json.load(f)
                if not rep.get("accepted", False):
                    err = f"FAIL: Quality report not accepted for E={E}"
                    print(err)
                    errors.append(err)
                    all_passed = False
                
                rel_res = rep.get("relative_energy_residual", None)
                if rel_res is None or not math.isfinite(rel_res):
                    err = f"FAIL: Invalid relative_energy_residual for E={E}: {rel_res}"
                    print(err)
                    errors.append(err)
                    all_passed = False
                    rel_residual_str = "INVALID"
                else:
                    rel_residual_str = f"{rel_res:.2e}"
                    if rel_res > 1.0e-4:
                        err = f"FAIL: relative_energy_residual {rel_res:.2e} > 1.0e-4 for E={E}"
                        print(err)
                        errors.append(err)
                        all_passed = False
                
                failures = rep.get("failures", [])
                if len(failures) > 0:
                    err = f"FAIL: Quality report contains failures for E={E}: {failures}"
                    print(err)
                    errors.append(err)
                    all_passed = False
            except Exception as e:
                err = f"FAIL: Corrupted quality report JSON for E={E}: {e}"
                print(err)
                errors.append(err)
                all_passed = False
                rel_residual_str = "ERROR"

    # 2. Dose profile comparisons
    t_vol = np.fromfile(tpath, dtype=np.float64).reshape((nz, ny, nx))
    g_vol = np.fromfile(gpath, dtype=np.float32).reshape((nz, ny, nx))
    
    t_idd_full = np.sum(t_vol, axis=(1, 2))
    g_idd_full = np.sum(g_vol, axis=(1, 2))
    
    # Full volume integral
    t_full_int = np.sum(t_idd_full) * dz
    g_full_int = np.sum(g_idd_full) * dz
    full_int_diff_pct = (g_full_int - t_full_int) / t_full_int * 100.0

    # ROI volume
    z_all = (np.arange(nz) + 0.5) * dz
    valid = z_all <= max_z
    t_idd_roi = t_idd_full[valid]
    g_idd_roi = g_idd_full[valid]
    z_roi = z_all[valid]
    
    t_pk_idx = np.argmax(t_idd_roi)
    g_pk_idx = np.argmax(g_idd_roi)
    
    t_pk_z = z_roi[t_pk_idx]
    g_pk_z = z_roi[g_pk_idx]
    shift_mm = abs(g_pk_z - t_pk_z)
    
    peak_diff_pct = (g_idd_roi[t_pk_idx] - t_idd_roi[t_pk_idx]) / t_idd_roi[t_pk_idx] * 100.0
    
    t_roi_int = np.sum(t_idd_roi) * dz
    g_roi_int = np.sum(g_idd_roi) * dz
    roi_int_diff_pct = (g_roi_int - t_roi_int) / t_roi_int * 100.0

    # Physical Assertions
    if shift_mm > 0.5:
        err = f"FAIL: Peak shift {shift_mm:.2f} mm > 0.5 mm at E={E}"
        print(err)
        errors.append(err)
        all_passed = False

    if abs(peak_diff_pct) > 6.0:
        err = f"FAIL: Peak difference {peak_diff_pct:.2f}% > 6.0% at E={E}"
        print(err)
        errors.append(err)
        all_passed = False

    if abs(roi_int_diff_pct) > 8.0:
        err = f"FAIL: ROI integral difference {roi_int_diff_pct:.2f}% > 8.0% at E={E}"
        print(err)
        errors.append(err)
        all_passed = False

    if abs(full_int_diff_pct) > 10.0:
        err = f"FAIL: Full volume integral difference {full_int_diff_pct:.2f}% > 10.0% at E={E}"
        print(err)
        errors.append(err)
        all_passed = False

    print(f"{E:<14} | {shift_mm:<16.2f} | {peak_diff_pct:<15.2f} | {roi_int_diff_pct:<18.2f} | {full_int_diff_pct:<18.2f} | {rel_residual_str:<16}")


print("="*130)
if all_passed:
    print("ALL ACCEPTANCE CRITERIA PASSED SUCCESSFULLY [x]")
    sys.exit(0)
else:
    print(f"VALIDATION FAILED with {len(errors)} errors:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
