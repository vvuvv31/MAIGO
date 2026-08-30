import os, sys, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_lateral_profiles(E_mevu, topas_path, gpu_path, out_png, depths_mm):
    nx, ny, nz = 400, 400, 800
    dx, dy, dz = 0.2, 0.2, 0.5
    x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
    
    t_vol = np.fromfile(topas_path, dtype=np.float64).reshape((nz, ny, nx))
    g_vol = np.fromfile(gpu_path, dtype=np.float32).reshape((nz, ny, nx))
    
    t_idd = np.sum(t_vol, axis=(1, 2))
    peak_iz = np.argmax(t_idd)
    z_peak = (peak_iz + 0.5) * dz
    
    n_depths = len(depths_mm)
    cols = 3
    rows = (n_depths + cols - 1) // cols
    
    # 1. Linear Scale Figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), dpi=300)
    fig.suptitle(f"Carbon-12 Lateral Dose Profiles D(x) @ {E_mevu} MeV/u (Linear Scale)\nTOPAS Full Physics vs GPU MAIGO", fontsize=15, fontweight="bold", y=0.99)
    axes = axes.flatten()
    
    for i, z_target in enumerate(depths_mm):
        iz = int(round((z_target - 0.5 * dz) / dz))
        iz = max(0, min(nz - 1, iz))
        actual_z = (iz + 0.5) * dz
        
        t_prof = np.sum(t_vol[iz], axis=0) # 1D profile
        g_prof = np.sum(g_vol[iz], axis=0)
        
        # Region label
        if actual_z < z_peak - 2.0:
            reg = "Plateau"
        elif abs(actual_z - z_peak) <= 2.0:
            reg = "Bragg Peak"
        else:
            reg = "Tail (Fragments)"
            
        ax = axes[i]
        ax.plot(x, t_prof, "b-", linewidth=2.0, label="TOPAS (INCL++)")
        ax.plot(x, g_prof, "r--", linewidth=1.8, label="GPU MAIGO")
        ax.set_title(f"z = {actual_z:.1f} mm ({reg})", fontsize=12, fontweight="bold")
        ax.set_xlabel("x (mm)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Linear Dose (Gy)", fontsize=10, fontweight="bold")
        ax.set_xlim(-20, 20)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Generated Linear: {out_png}")
    
    # 2. Logarithmic Scale Figure (to see the wide halo 3 orders of magnitude down)
    out_log_png = out_png.replace(".png", "_log.png")
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), dpi=300)
    fig.suptitle(f"Carbon-12 Lateral Dose Profiles D(x) @ {E_mevu} MeV/u (Log Scale - Core & Halo)\nTOPAS Full Physics vs GPU MAIGO", fontsize=15, fontweight="bold", y=0.99)
    axes = axes.flatten()
    
    for i, z_target in enumerate(depths_mm):
        iz = int(round((z_target - 0.5 * dz) / dz))
        iz = max(0, min(nz - 1, iz))
        actual_z = (iz + 0.5) * dz
        
        t_prof = np.sum(t_vol[iz], axis=0)
        g_prof = np.sum(g_vol[iz], axis=0)
        
        if actual_z < z_peak - 2.0:
            reg = "Plateau"
        elif abs(actual_z - z_peak) <= 2.0:
            reg = "Bragg Peak"
        else:
            reg = "Tail (Fragments)"
            
        ax = axes[i]
        # Avoid log(0)
        t_plot = np.maximum(t_prof, 1e-6)
        g_plot = np.maximum(g_prof, 1e-6)
        
        ax.semilogy(x, t_plot, "b-", linewidth=2.0, label="TOPAS (INCL++)")
        ax.semilogy(x, g_plot, "r--", linewidth=1.8, label="GPU MAIGO")
        ax.set_title(f"z = {actual_z:.1f} mm ({reg})", fontsize=12, fontweight="bold")
        ax.set_xlabel("x (mm)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Log Dose (Gy)", fontsize=10, fontweight="bold")
        ax.set_xlim(-30, 30)
        ax.set_ylim(bottom=1e-5, top=max(np.max(t_plot), np.max(g_plot)) * 3.0)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(out_log_png, dpi=300)
    plt.close()
    print(f"Generated Log: {out_log_png}")

os.makedirs("plots", exist_ok=True)

# Six physical depth locations: entrance, approximately 1/4, 1/2 and 3/4
# of the TOPAS peak depth, the Bragg peak, and 5 mm beyond the peak.
configs = [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e100/topas_emittance_inelastic_e100.bin",
     "out/fred_topas_1M_idd_corrected_independent/e100/voxel_dose.raw",
     [1.25, 6.25, 12.75, 19.25, 25.75, 30.75]),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e200/topas_emittance_inelastic_e200.bin",
     "out/fred_topas_1M_idd_corrected_independent/e200/voxel_dose.raw",
     [5.25, 21.75, 43.25, 65.25, 86.75, 91.75]),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e300/topas_emittance_inelastic_e300.bin",
     "out/fred_topas_1M_idd_corrected_independent/e300/voxel_dose.raw",
     [5.25, 42.75, 85.75, 128.75, 171.75, 176.75]),
    (400, "/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e400/topas_emittance_inelastic_e400.bin",
     "out/fred_topas_1M_idd_corrected_independent/e400/voxel_dose.raw",
     [5.25, 68.75, 137.25, 205.75, 274.25, 279.25]),
]

for energy, topas_path, gpu_path, depths in configs:
    output = f"plots/lateral_profiles_{energy}MeVu.png"
    plot_lateral_profiles(energy, topas_path, gpu_path, output, depths)
