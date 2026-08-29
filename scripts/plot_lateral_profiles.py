import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
os.makedirs("/home/wuwei/.gemini/antigravity-cli/brain/b6e8050e-cd93-476b-bd8a-039915838a5f/plots", exist_ok=True)

configs = [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw", [5, 15, 25.8, 30]),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw", [20, 50, 85.5, 95]),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e300/topas_emittance_inelastic_e300.bin", "out/gpu_inelastic_e300/voxel_dose.raw", [40, 100, 169.5, 185]),
    (400, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e400/topas_emittance_inelastic_e400.bin", "out/gpu_inelastic_e400/voxel_dose.raw", [60, 150, 269.0, 290])
]

nx, ny, nz = 400, 400, 800
dx, dy, dz = 0.2, 0.2, 0.5
x = (np.arange(nx) - nx / 2.0 + 0.5) * dx

for E, tpath, gpath, depths in configs:
    if not (os.path.exists(tpath) and os.path.exists(gpath)):
        continue
    print(f"Plotting lateral profiles for {E} MeV/u at depths {depths} mm...")
    t_vol = np.fromfile(tpath, dtype=np.float64).reshape((nz, ny, nx))
    g_vol = np.fromfile(gpath, dtype=np.float32).reshape((nz, ny, nx))
    
    # 1. Linear profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle(f"Lateral Dose Profiles @ {E} MeV/u (Linear Scale)\nTOPAS vs GPU MAIGO", fontsize=14, fontweight="bold", y=0.98)
    axes = axes.flatten()
    
    for i, d in enumerate(depths):
        iz = int(round(d / dz))
        iz = min(iz, nz - 1)
        z_actual = (iz + 0.5) * dz
        t_x = np.sum(t_vol[iz], axis=0)
        g_x = np.sum(g_vol[iz], axis=0)
        
        ax = axes[i]
        ax.plot(x, t_x, "b-", linewidth=2.0, label="TOPAS")
        ax.plot(x, g_x, "r--", linewidth=1.8, label="GPU MAIGO")
        ax.set_title(f"Depth z = {z_actual:.1f} mm", fontsize=11, fontweight="bold")
        ax.set_xlabel("Lateral x (mm)", fontsize=10)
        ax.set_ylabel("Dose (Gy)", fontsize=10)
        ax.set_xlim(-30, 30)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        
    plt.tight_layout()
    out_linear = f"plots/lateral_profiles_{E}MeVu.png"
    plt.savefig(out_linear, dpi=300)
    plt.close()
    
    # 2. Logarithmic profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle(f"Lateral Dose Profiles @ {E} MeV/u (Log Scale)\nTOPAS vs GPU MAIGO", fontsize=14, fontweight="bold", y=0.98)
    axes = axes.flatten()
    
    for i, d in enumerate(depths):
        iz = int(round(d / dz))
        iz = min(iz, nz - 1)
        z_actual = (iz + 0.5) * dz
        t_x = np.sum(t_vol[iz], axis=0)
        g_x = np.sum(g_vol[iz], axis=0)
        
        ax = axes[i]
        ax.semilogy(x, np.maximum(t_x, 1e-7), "b-", linewidth=2.0, label="TOPAS")
        ax.semilogy(x, np.maximum(g_x, 1e-7), "r--", linewidth=1.8, label="GPU MAIGO")
        ax.set_title(f"Depth z = {z_actual:.1f} mm (Log)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Lateral x (mm)", fontsize=10)
        ax.set_ylabel("Dose (Gy, log)", fontsize=10)
        ax.set_xlim(-40, 40)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        
    plt.tight_layout()
    out_log = f"plots/lateral_profiles_{E}MeVu_log.png"
    plt.savefig(out_log, dpi=300)
    plt.close()

# Copy to artifact directory
import shutil
for f in os.listdir("plots"):
    if f.endswith(".png"):
        shutil.copy(os.path.join("plots", f), os.path.join("/home/wuwei/.gemini/antigravity-cli/brain/b6e8050e-cd93-476b-bd8a-039915838a5f/plots", f))

print("All lateral profiles generated successfully!")
