import os, sys, math
import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fit_profile_scipy(x, y):
    """
    Fits double Gaussian with exact constraint:
    sh = 1.8 * sc + exp(delta)
    and Poisson / relative weighting: w_i = 1 / sqrt(y_i + 1e-4 * max(y))
    """
    total = np.sum(y)
    if total <= 1e-4:
        return 0.0, np.nan, 0.0
        
    mean_x = np.sum(x * y) / total
    var_x = np.sum(((x - mean_x)**2) * y) / total
    sig0 = math.sqrt(max(0.1, var_x))
    peak_y = np.max(y)
    weights = 1.0 / np.sqrt(y + 1e-4 * peak_y)
    
    # 1. Single Gaussian
    def single_gauss(p):
        A, s = p
        diff = (y - A * np.exp(-0.5 * ((x - mean_x)/s)**2)) * weights
        return diff
    res_s = opt.least_squares(single_gauss, [peak_y, sig0], bounds=([0, 0.2], [np.inf, np.inf]))
    loss_s = np.sum(res_s.fun**2)
    
    # 2. Constrained Double Gaussian: sh = 1.8 * sc + exp(delta)
    def double_gauss(p):
        Ac, sc, Ah, delta = p
        sh = 1.8 * sc + math.exp(delta)
        diff = (y - (Ac * np.exp(-0.5 * ((x - mean_x)/sc)**2) + Ah * np.exp(-0.5 * ((x - mean_x)/sh)**2))) * weights
        return diff
        
    p0 = [peak_y * 0.90, sig0 * 0.90, peak_y * 0.05, math.log(max(0.1, sig0 * 1.5))]
    bounds = ([0, 0.2, 0, -5.0], [np.inf, np.inf, np.inf, 5.0])
    res_d = opt.least_squares(double_gauss, p0, bounds=bounds)
    loss_d = np.sum(res_d.fun**2)
    
    Ac, sc, Ah, delta = res_d.x
    sh = 1.8 * sc + math.exp(delta)
        
    area_c = Ac * math.sqrt(2 * math.pi) * sc
    area_h = Ah * math.sqrt(2 * math.pi) * sh
    w_h = area_h / (area_c + area_h) if (area_c + area_h) > 0 else 0.0
    
    if w_h < 0.010 or (loss_s - loss_d) / loss_s < 0.008:
        return res_s.x[1], np.nan, 0.0
        
    return sc, sh, w_h

def load_and_analyze(topas_path, gpu_path, max_z_mm=None):
    nx, ny, nz = 400, 400, 800
    dx, dy, dz = 0.2, 0.2, 0.5
    x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
    
    t_vol = np.fromfile(topas_path, dtype=np.float64).reshape((nz, ny, nx))
    g_vol = np.fromfile(gpu_path, dtype=np.float32).reshape((nz, ny, nx))
    
    z_all = (np.arange(nz) + 0.5) * dz
    if max_z_mm:
        valid = z_all <= max_z_mm
        t_vol = t_vol[valid]
        g_vol = g_vol[valid]
        z_all = z_all[valid]
        
    # Compute IDD (sum over xy)
    t_idd = np.sum(t_vol, axis=(1, 2))
    g_idd = np.sum(g_vol, axis=(1, 2))
    
    # Step sampling for lateral profiles
    step = 2 # every 1.0 mm
    sample_indices = np.arange(0, len(z_all), step)
    z_samples = z_all[sample_indices]
    
    t_sc, t_sh, t_wh = [], [], []
    g_sc, g_sh, g_wh = [], [], []
    
    for iz in sample_indices:
        t_x = np.sum(t_vol[iz], axis=0)
        g_x = np.sum(g_vol[iz], axis=0)
        
        sc_t, sh_t, wh_t = fit_profile_scipy(x, t_x)
        sc_g, sh_g, wh_g = fit_profile_scipy(x, g_x)
        
        t_sc.append(sc_t)
        t_sh.append(sh_t)
        t_wh.append(wh_t)
        g_sc.append(sc_g)
        g_sh.append(sh_g)
        g_wh.append(wh_g)
        
    return z_all, t_idd, g_idd, z_samples, np.array(t_sc), np.array(t_sh), np.array(g_sc), np.array(g_sh)

def plot_benchmark_png(E_mevu, z_all, t_idd, g_idd, z_samples, t_sc, t_sh, g_sc, g_sh, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    fig.suptitle(f"Carbon-12 Inelastic Fragmentation Benchmark @ {E_mevu} MeV/u\nTOPAS Full Physics vs GPU MAIGO", fontsize=15, fontweight="bold", y=0.98)
    
    peak_iz = np.argmax(t_idd)
    z_peak = z_all[peak_iz]
    
    # 1. IDD Absolute
    ax1 = axes[0, 0]
    ax1.plot(z_all, t_idd, "b-", linewidth=2.2, label="TOPAS (INCL++ Full Physics)")
    ax1.plot(z_all, g_idd, "r--", linewidth=2.0, label="GPU MAIGO (Literature Data-Driven)")
    ax1.axvline(z_peak, color="gray", linestyle=":", label=f"Bragg Peak ({z_peak:.2f} mm)")
    ax1.set_title("Integrated Depth Dose (IDD)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Depth z (mm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Absorbed Dose (Gy / 100k)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")
    
    # 2. IDD Relative Error (Unclipped)
    ax2 = axes[0, 1]
    valid_mask = t_idd > (0.005 * np.max(t_idd))
    idd_diff = (g_idd - t_idd) / t_idd * 100.0
    ax2.plot(z_all[valid_mask], idd_diff[valid_mask], color="purple", linewidth=2.0, label="IDD Diff: (GPU - TOPAS)/TOPAS (%)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.axvline(z_peak, color="gray", linestyle=":")
    ax2.set_title("IDD Relative Error (%) [Unclipped]", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Depth z (mm)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Relative Error (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10, loc="upper right")
    
    # 3. Lateral Sigma Absolute (Core & Halo)
    ax3 = axes[1, 0]
    ax3.plot(z_samples, t_sc, "g-", linewidth=2.2, label=r"TOPAS Core $\sigma_{core}$")
    ax3.plot(z_samples, g_sc, "g--", linewidth=2.0, label=r"GPU Core $\sigma_{core}$")
    
    mask_th = ~np.isnan(t_sh)
    mask_gh = ~np.isnan(g_sh)
    if np.any(mask_th):
        ax3.plot(z_samples[mask_th], t_sh[mask_th], color="darkorange", linestyle="-", marker="o", markersize=3.5, label=r"TOPAS Halo $\sigma_{halo}$")
    if np.any(mask_gh):
        ax3.plot(z_samples[mask_gh], g_sh[mask_gh], color="red", linestyle="--", marker="x", markersize=3.5, label=r"GPU Halo $\sigma_{halo}$")
        
    ax3.axvline(z_peak, color="gray", linestyle=":")
    ax3.set_title(r"Lateral Beam Profile: Core & Halo $\sigma$", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Depth z (mm)", fontsize=11, fontweight="bold")
    ax3.set_ylabel(r"Lateral $\sigma$ (mm)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(fontsize=10, loc="upper left")
    
    # 4. Lateral Sigma Relative Error (Unclipped)
    ax4 = axes[1, 1]
    sig_c_diff = (g_sc - t_sc) / t_sc * 100.0
    ax4.plot(z_samples, sig_c_diff, "g-", linewidth=2.0, label=r"Core $\sigma_{core}$ Diff (%)")
    
    both_h = mask_th & mask_gh
    if np.any(both_h):
        sig_h_diff = (g_sh[both_h] - t_sh[both_h]) / t_sh[both_h] * 100.0
        ax4.plot(z_samples[both_h], sig_h_diff, color="darkorange", linestyle="-", marker="s", markersize=4, label=r"Halo $\sigma_{halo}$ Diff (%)")
        
    ax4.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax4.axvline(z_peak, color="gray", linestyle=":")
    ax4.set_title(r"Lateral $\sigma$ Relative Error (%) [Unclipped]", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Depth z (mm)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Relative Error (%)", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(fontsize=10, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Successfully generated {out_png}")

os.makedirs("plots", exist_ok=True)
os.makedirs("/home/wuwei/.gemini/antigravity-cli/brain/b6e8050e-cd93-476b-bd8a-039915838a5f/plots", exist_ok=True)

for E, tpath, gpath, max_z in [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw", 40.0),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw", 120.0),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e300/topas_emittance_inelastic_e300.bin", "out/gpu_inelastic_e300/voxel_dose.raw", 220.0),
    (400, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e400/topas_emittance_inelastic_e400.bin", "out/gpu_inelastic_e400/voxel_dose.raw", 340.0)
]:
    if os.path.exists(tpath) and os.path.exists(gpath):
        print(f"Processing {E} MeV/u...")
        z_all, t_idd, g_idd, z_samples, t_sc, t_sh, g_sc, g_sh = load_and_analyze(tpath, gpath, max_z)
        out_png = f"plots/benchmark_{E}MeVu_idd_sigma.png"
        plot_benchmark_png(E, z_all, t_idd, g_idd, z_samples, t_sc, t_sh, g_sc, g_sh, out_png)
