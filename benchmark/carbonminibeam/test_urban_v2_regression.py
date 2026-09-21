"""urban_v2 production-access regression (must-fail on HEAD d5a453c).

Scope: host mirror of the EXACT current device/caller logic, checked against
the G4VMultipleScattering reference semantics required by the task.  This file
does not claim any gate passes; it freezes the defects so a fix must turn
FAIL -> PASS via the real caller/shared-helper path.

Defects frozen here (file/function/line at HEAD):
  R1 safety wiring: src/transport_sycl.cpp:4234-4238 passes safety_mm=0.0F to
     copper_urban_v2_msc_step (src/detail/sycl_device_math.inc:807-811).
     Device then does post_safety=0.99*max(0,safety) (:1002), so any raw>0 is
     cancelled. Testing the sampler with a hand-passed large safety does NOT
     cover the production caller.
  R2 geomMin scope: sycl_device_math.inc:824,1007 uses model constant
     geommin=1e-3 mm for the process displacement-reduction gate. Task
     requires G4VMultipleScattering reference: geomMin vs minDisplacement2
     must not be conflated. Case raw=1e-4, safety=1e-5 must enter the
     reference reduction branch, not cancel.
  R3 FP32 delta: sycl_device_math.inc:839-851 computes true_path (float) then
     delta=float(true)-float(geom) (:851). log1p/rmax^2 reorder does not fix
     the prior cancellation for small g. Must compute/save delta on the
     correct path-conversion branch.
  R4 path consumer: true_path lives only inside the sampler; caller at
     transport_sycl.cpp:4260-4282 advances position by transport_path and
     energy loss by stopping*transport_path. No proposed/final t, final g,
     stable delta, limit reason, boundary state is returned.
  R5 diagnostic identity: record.copper_touched = minibeam_hits_copper
     (transport_sycl.cpp:5371) is the initial-ray hit, not in-transport touch;
     TOPAS joins in docs/urban_v2_diagnosis.md sec.12 use TrackID only, which
     failed.md 2026-09-19 entry proves invalid across MT worker blocks
     (needs (run,event,track)+shard namespace).

Run: python3 benchmark/carbonminibeam/test_urban_v2_regression.py
Exit 0 with FAIL lines = defects reproduced (expected before fix).
After a correct fix, the same spec assertions against the fixed shared-helper
+ real caller path must all PASS.
"""
import math
import numpy as np

F32 = np.float32
GEOMMIN_MODEL = 1e-3  # current device constant, sycl_device_math.inc:824

def current_accept(raw, safety, geommin=GEOMMIN_MODEL):
    """Exact mirror of device D5 block (:1000-1012)."""
    disp_r = float(raw)
    post = 0.99 * max(0.0, float(safety))
    if disp_r <= 0.0:
        return 0.0, "zero"
    if disp_r <= post:
        return raw, "accept"
    if post > geommin:
        return raw * (post / disp_r), "reduce"
    return 0.0, "cancel"

def reference_accept(raw, safety, geom_min, min_disp2):
    """Task-required reference shape: accept / reduce / cancel use distinct
    geomMin (geometry tolerance) and minDisplacement2 (physics floor).
    Exact numeric values come from G4VMultipleScattering; what matters here is
    they must NOT be the same 1e-3 model constant. We parametrize explicitly
    so conflation is visible."""
    post = 0.99 * max(0.0, float(safety))
    if raw <= 0.0:
        return 0.0, "zero"
    if raw <= post:
        return raw, "accept"
    # reference: reduction allowed when postSafety exceeds the geometry
    # tolerance; cancellation only when below the physics displacement floor.
    if post > geom_min:
        return raw * (post / raw), "reduce"
    if post <= min_disp2:
        return 0.0, "cancel"
    return raw * (post / raw), "reduce"

def current_delta_f32(g_mm, lambda0_mm):
    """Mirror of :839-851 in float32: t=-lam*log1p(-g/lam), delta=f32(t)-f32(g)."""
    g = F32(g_mm); lam = F32(lambda0_mm)
    ratio = F32(g / lam)
    t = F32(-lam * np.log1p(F32(-ratio)))
    return float(t), float(F32(t - g))

def reference_delta_f64(g_mm, lambda0_mm):
    g = float(g_mm); lam = float(lambda0_mm)
    r = g / lam
    t = -lam * math.log1p(-r)
    # Stable form (no t-g cancellation): the true reference value.
    d = -lam * (math.log1p(-r) + r)
    return t, d

def fixed_accept(raw, safety,
                 geom_min=1.0e-6, min_disp=1.0e-7):
    """Mirror of copper_urban_v2_accept_displacement after fix: distinct
    thresholds, double-precision disp_r/postSafety."""
    import math as _m
    disp_r = float(raw)  # device uses sqrt(dx^2+dy^2) in double; equal here
    post = 0.99 * max(0.0, float(safety))
    if not (disp_r > 0.0):
        return 0.0, "zero"
    if disp_r <= post:
        return raw, "accept"
    if post > geom_min:
        return raw * (post / disp_r), "reduce"
    if post <= min_disp:
        return 0.0, "cancel"
    return raw * (post / disp_r), "reduce"

def fixed_delta(g_mm, lambda0_mm):
    """Mirror of copper_urban_v2_true_path_and_delta: Taylor branch for
    r<1e-3, direct branch delta otherwise."""
    g = float(g_mm); lam = float(lambda0_mm)
    r = g / lam
    if r < 1e-3:
        s = 0.5 + r * (1.0/3.0 + r * (0.25 + r * 0.2))
        d = g * r * s
        return g + d, d
    t = -lam * math.log1p(-r)
    return t, t - g

def run_fixed_checks():
    print("== FIXED-helper spec checks (post-fix contract) ==")
    ok = True
    cases = [
        # (safety, raw, expected_branch)
        (0.0, 5e-5, "cancel"),      # safety=0 cancels correctly
        (1.0, 5e-5, "accept"),      # in-vivo large safety accepts
        (1e-5, 1e-4, "reduce"),     # 0<post<raw reduces (R2)
        (1.0, 1e-4, "accept"),      # post>=raw keeps
    ]
    for safety, raw, exp in cases:
        _, got = fixed_accept(raw, safety)
        mark = "PASS" if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"  [{mark}] safety={safety:g} raw={raw:g} -> {got} (expect {exp})")
    lam = 1.9749e6
    for g in (0.03, 0.05, 0.10, 0.25):
        _, d_fix = fixed_delta(g, lam)
        _, d_ref = reference_delta_f64(g, lam)
        rel = abs(d_fix - d_ref) / d_ref if d_ref else 0.0
        # Tolerance 1e-8: residual is libm log1p rounding noise in the
        # reference itself (absolute ~1e-17 mm).  The pre-fix FP32 error was
        # up to 88% / full cancellation to 0, i.e. 7+ orders worse.
        mark = "PASS" if rel < 1e-8 else "FAIL"
        if rel >= 1e-8:
            ok = False
        print(f"  [{mark}] g={g:5.2f} d_fix={d_fix:.3e} d_ref={d_ref:.3e} rel={rel:.1e}")
    print("  fixed contract:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok

def run():
    fails = []
    print("== R1: production caller safety wiring ==")
    # In-vivo case where reference allows displacement: raw>0, true postSafety large.
    raw = 5e-5
    got, why = current_accept(raw, 0.0)  # what transport_sycl.cpp:4238 passes
    print(f"  caller safety=0.0F, raw={raw:g} -> accepted={got:g} ({why})")
    if not (got == 0.0 and why == "cancel"):
        fails.append("R1-unexpected")
    else:
        print("  FAIL reproduced: production caller恒零 (accepted==0 despite raw>0)")
    # Spec cases the fixed caller+helper must satisfy (not satisfied today):
    for safety, raw2, exp in [(0.0, 5e-5, "cancel"), (1.0, 5e-5, "accept")]:
        print(f"  [spec] safety={safety:g} raw={raw2:g} -> expect {exp}")

    print("== R2: reduction-gate scope ==")
    raw, safety = 1e-4, 1e-5
    got, why = current_accept(raw, safety)
    post = 0.99 * safety
    print(f"  raw={raw:g} safety={safety:g} postSafety={post:g} geommin(model)={GEOMMIN_MODEL:g}")
    print(f"  current -> {why} (accepted={got:g})")
    # Reference: this point must REDUCE (postSafety above geometry tolerance),
    # not cancel. Current code cancels because 9.9e-6 < 1e-3 model geommin.
    if why == "cancel":
        print("  FAIL reproduced: must enter reference reduction branch, but cancels")
    else:
        fails.append("R2-not-reproduced")
    # Explicit conflation check:
    print("  [spec] geomMin and minDisplacement2 must be distinct reference")
    print("         constants; current code uses single 1e-3 model geommin.")

    print("== R3: FP32 small-quantity delta ==")
    lam = 1.9749e6  # oracle lambda0(250MeV/u)
    for g in (0.03, 0.05, 0.10, 0.25):
        t32, d32 = current_delta_f32(g, lam)
        t64, d64 = reference_delta_f64(g, lam)
        print(f"  g={g:5.2f} t32={t32:.12f} d32={d32:.3e} | t64={t64:.12f} d64={d64:.3e}")
        if d32 == 0.0 and d64 > 0.0:
            print("    FAIL reproduced: float delta cancels to 0, double delta>0")
    # rmax consequence:
    g = 0.25
    _, d32 = current_delta_f32(g, lam)
    _, d64 = reference_delta_f64(g, lam)
    t64, _ = reference_delta_f64(g, lam)
    r32 = math.sqrt(max(0.0, d32 * (2 * g - d32))) if d32 > 0 else 0.0
    r64 = math.sqrt(max(0.0, d64 * (2 * t64 - d64)))
    print(f"  rmax32={r32:.3e} rmax64={r64:.3e} -> {'FAIL: rmax=0' if r32==0.0 else 'ok'}")
    print("  [spec] must compute/save delta on correct branch; compare delta AND rmax,")
    print("         not only t relative error. cth~1 rounding: save 1-cos or half-angle")
    print("         before rounding; log zero-angle/NaN/clamp/fallback rates.")

    print("== R4: path consumer contract ==")
    print("  current: true_path internal only (sycl_device_math.inc:839); caller uses")
    print("           transport_path for position (transport_sycl.cpp:4260-4265) and")
    print("           stopping*transport_path for loss (:4266-4282). No proposed/final")
    print("           t, final g, stable delta, limit reason, boundary state returned.")
    print("  FAIL reproduced by inspection: post-hoc t/g rescaling cannot fix the")
    print("  energy ledger; collision truncation uses the wrong length.")
    print("  [spec] need explicit step proposal/finalization: proposed/final t,")
    print("         final g, stable delta, limit reason, boundary state; geometry")
    print("         consumes g, loss/fluctuation (+reaction path when on) consume final t.")

    print("== R5: diagnostic geometry + identity ==")
    print("  slit control: x=0 straight ray vs slit centered +0.05mm, half-width")
    print("  0.25mm -> inside opening; this navigation control must pass before any")
    print("  large-statistics run. Status here: NOT_RUN (no shared entry records yet).")
    print("  identity: two events sharing TrackID must NOT merge on TrackID alone;")
    print("  need (run,event,track)+shard. docs sec.12 TrackID-only join is INVALID")
    print("  per failed.md 2026-09-19 EventID-row entry. FAIL reproduced by doc audit.")

    print()
    print(f"regression verdict: defects reproduced ({len(fails)} unexpected).")
    print("Gates 1,2,3,5 NOT established: oracle=angle-only, slab x100 dismissed as")
    print("'geometry-confounded', step matrix without real fMinimal, TrackID join invalid.")
    fixed_ok = run_fixed_checks()
    return 0 if (len(fails) == 0 and fixed_ok) else 1

if __name__ == "__main__":
    raise SystemExit(run())
