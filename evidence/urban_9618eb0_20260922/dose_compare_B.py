#!/usr/bin/env python3
"""Urban-B full-chain EM-only 10M dose validation (2 seeds) vs TOPAS and vs
the surviving pre-B seed-2 run. Fixed-ROI peak/valley/PVDR, no registration,
no renormalization (prompt Phase D/9.4)."""
import numpy as np

TOPAS_BIN = "/mnt/sda/wuwei/minibeam_single_center_em_only_e250_10m/topas_7374/dose.bin"
B_S1 = "/mnt/sdb/wuwei/MAIGO_pristine/out/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m_B_s1/dose.raw"
B_S2 = "/mnt/sdb/wuwei/MAIGO_pristine/out/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m_B_s2/dose.raw"
OLD_S2 = "/mnt/sdb/wuwei/MAIGO_pristine/out/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m_s2/dose.raw"

NX, NZ = 1000, 1000
DX, DZ = 0.1, 0.25
X0, Z0 = -50.0, 0.0

topas = np.fromfile(TOPAS_BIN, dtype="<f8").reshape((NZ, NX))
b1 = np.fromfile(B_S1, dtype="<f4").reshape((NZ, NX))
b2 = np.fromfile(B_S2, dtype="<f4").reshape((NZ, NX))
old = np.fromfile(OLD_S2, dtype="<f4").reshape((NZ, NX))

x = X0 + (np.arange(NX) + 0.5) * DX
peak = (np.abs(x) < 0.25)
valley = (np.abs(np.abs(x) - 1.8) < 0.45)

print(f"totals: TOPAS={topas.sum():.6g} B_s1={b1.sum():.6g} "
      f"({b1.sum()/topas.sum():.5f}) B_s2={b2.sum():.6g} "
      f"({b2.sum()/topas.sum():.5f}) OLD_s2={old.sum():.6g} "
      f"({old.sum()/topas.sum():.5f})", flush=True)
print(f"{'depth':>8} {'B1-pp':>8} {'B2-pp':>8} {'OLD-pp':>8} "
      f"{'B1Peak':>8} {'B1Val':>8} {'B1PVDR':>8} "
      f"{'B2Peak':>8} {'B2Val':>8} {'B2PVDR':>8} {'topPVDR':>9}", flush=True)
for d in [0.88, 4.88, 9.88, 19.88, 39.88, 79.88, 119.88]:
    iz = int(round((d - Z0 - DZ / 2) / DZ))
    T = topas[max(0, iz - 2):iz + 3].mean(axis=0)
    A = b1[max(0, iz - 2):iz + 3].mean(axis=0)
    C = b2[max(0, iz - 2):iz + 3].mean(axis=0)
    O = old[max(0, iz - 2):iz + 3].mean(axis=0)

    def contrast(a):
        return a[peak].mean() / a[valley].mean()
    ct, ca, cc, co = contrast(T), contrast(A), contrast(C), contrast(O)
    print(f"{d:8.2f} {100*(ca/ct-1):8.2f} {100*(cc/ct-1):8.2f} "
          f"{100*(co/ct-1):8.2f} "
          f"{A[peak].mean()/T[peak].mean():8.4f} "
          f"{A[valley].mean()/T[valley].mean():8.4f} "
          f"{ca/ct:8.4f} "
          f"{C[peak].mean()/T[peak].mean():8.4f} "
          f"{C[valley].mean()/T[valley].mean():8.4f} "
          f"{cc/ct:8.4f} {ct:9.4f}", flush=True)
