import array, math, os, sys

def nelder_mead(f, x0, step=0.1, no_improve_thr=1e-9, no_improv_break=300, max_iter=4000,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    dim = len(x0)
    prev_best = f(x0)
    no_improv = 0
    res = [[x0, prev_best]]

    for i in range(dim):
        x = list(x0)
        x[i] = x[i] + step if x[i] != 0 else step
        score = f(x)
        res.append([x, score])

    iters = 0
    while True:
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        if max_iter and iters >= max_iter:
            return res[0][0], res[0][1]
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0][0], res[0][1]

        x0_c = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0_c[i] += c / (len(res)-1)

        xr = [x0_c[i] + alpha * (x0_c[i] - res[-1][0][i]) for i in range(dim)]
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        if rscore < res[0][1]:
            xe = [x0_c[i] + gamma * (x0_c[i] - res[-1][0][i]) for i in range(dim)]
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        xc = [x0_c[i] + rho * (x0_c[i] - res[-1][0][i]) for i in range(dim)]
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = [x1[i] + sigma * (tup[0][i] - x1[i]) for i in range(dim)]
            score = f(redx)
            nres.append([redx, score])
        res = nres

def fit_regularized_profile(xs, ys):
    peak_y = max(ys)
    total = sum(ys)
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_x = sum(x * y for x, y in zip(xs, ys)) / total
    var_x = sum(((x - mean_x)**2) * y for x, y in zip(xs, ys)) / total
    sig0 = math.sqrt(max(0.1, var_x))
    
    def single_loss(p):
        A, s = p
        if s < 0.5 or A < 0: return 1e18
        res = 0.0
        for x, y in zip(xs, ys):
            diff = y - A * math.exp(-0.5 * ((x - mean_x)/s)**2)
            res += diff * diff
        return res
        
    (best_A_s, best_s_s), s_loss = nelder_mead(single_loss, [peak_y, sig0], step=0.1)
    
    def double_loss(p):
        Ac, sc, Ah, sh = p
        if sc < 0.5 or sh < sc * 1.80 or Ac < 0 or Ah < 0:
            return 1e18
        res = 0.0
        for x, y in zip(xs, ys):
            dx = x - mean_x
            pred = Ac * math.exp(-0.5 * (dx / sc)**2) + Ah * math.exp(-0.5 * (dx / sh)**2)
            diff = y - pred
            res += diff * diff
        return res

    best_d_loss = 1e20
    best_p = None
    starts = [
        [peak_y * 0.95, sig0 * 0.98, peak_y * 0.05, sig0 * 2.2],
        [peak_y * 0.90, sig0 * 0.92, peak_y * 0.10, sig0 * 2.8],
        [peak_y * 0.80, sig0 * 0.88, peak_y * 0.20, sig0 * 2.0],
    ]
    for x0 in starts:
        p, score = nelder_mead(double_loss, x0, step=0.1, max_iter=2500)
        if score < best_d_loss:
            best_d_loss = score
            best_p = p

    Ac, sc, Ah, sh = best_p
    area_c = Ac * math.sqrt(2.0 * math.pi) * sc
    area_h = Ah * math.sqrt(2.0 * math.pi) * sh
    w_halo = area_h / (area_c + area_h) if (area_c + area_h) > 0 else 0.0
    
    if w_halo < 0.010 or (s_loss - best_d_loss) / (s_loss + 1e-12) < 0.005:
        return best_s_s, 0.0, 0.0, best_A_s, 0.0
        
    return sc, sh, w_halo, Ac, Ah

def load_data(topas_path, gpu_path, max_z_mm=None):
    nx, ny, nz = 400, 400, 800
    dx, dy, dz = 0.2, 0.2, 0.5
    x_coords = [(ix - nx / 2.0 + 0.5) * dx for ix in range(nx)]
    
    t_idd, g_idd = [], []
    t_sig_c, t_sig_h = [], []
    g_sig_c, g_sig_h = [], []
    z_vals = []
    
    bytes_t = nx * ny * 8
    bytes_g = nx * ny * 4
    
    with open(topas_path, "rb") as ft, open(gpu_path, "rb") as fg:
        for iz in range(nz):
            z_mm = (iz + 0.5) * dz
            if max_z_mm and z_mm > max_z_mm:
                break
                
            buft = ft.read(bytes_t)
            bufg = fg.read(bytes_g)
            if not buft or not bufg: break
            
            arrt = array.array("d", buft)
            arrg = array.array("f", bufg)
            
            t_tot = sum(arrt)
            g_tot = sum(arrg)
            
            if t_tot < 1e-4 and g_tot < 1e-4 and z_mm > 50:
                break
                
            z_vals.append(z_mm)
            t_idd.append(t_tot)
            g_idd.append(g_tot)
            
            # Extract x projection
            t_x = [0.0] * nx
            g_x = [0.0] * nx
            for iy in range(ny):
                off = iy * nx
                for ix in range(nx):
                    t_x[ix] += arrt[off + ix]
                    g_x[ix] += arrg[off + ix]
                    
            sc_t, sh_t, _, _, _ = fit_regularized_profile(x_coords, t_x)
            sc_g, sh_g, _, _, _ = fit_regularized_profile(x_coords, g_x)
            
            t_sig_c.append(sc_t)
            t_sig_h.append(sh_t if sh_t > 0 else float("nan"))
            g_sig_c.append(sc_g)
            g_sig_h.append(sh_g if sh_g > 0 else float("nan"))
            
    return z_vals, t_idd, g_idd, t_sig_c, t_sig_h, g_sig_c, g_sig_h

def draw_svg(filename, E_mevu, z_vals, t_idd, g_idd, t_sig_c, t_sig_h, g_sig_c, g_sig_h):
    w, h = 1100, 850
    margin_l, margin_r, margin_t, margin_b = 90, 50, 70, 60
    
    # 4 subplots layout:
    # Top-Left: IDD Absolute [x: 90..530, y: 70..400]
    # Top-Right: IDD Relative Diff [x: 620..1050, y: 70..400]
    # Bottom-Left: Sigma Core & Halo Absolute [x: 90..530, y: 480..790]
    # Bottom-Right: Sigma Relative Diff [x: 620..1050, y: 480..790]
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="35" font-family="Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle" fill="#1e293b">Carbon-12 Inelastic Fragmentation Benchmark @ {E_mevu} MeV/u (TOPAS vs GPU MAIGO)</text>')
    
    # Helper to draw a plot panel
    def render_panel(px, py, pw, ph, x_data, y_series, title, xlabel, ylabel, y_min_val=None, y_max_val=None):
        # Background box
        svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="4"/>')
        svg.append(f'<text x="{px + pw/2}" y="{py - 10}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#334155">{title}</text>')
        
        # X and Y limits
        xmin, xmax = min(x_data), max(x_data)
        all_y = []
        for _, ys, _, _, _ in y_series:
            all_y.extend([y for y in ys if not math.isnan(y) and not math.isinf(y)])
            
        if not all_y: all_y = [0, 1]
        ymin = y_min_val if y_min_val is not None else min(all_y)
        ymax = y_max_val if y_max_val is not None else max(all_y)
        if ymin == ymax: ymax += 1.0
        
        # Grid lines (5 ticks)
        for i in range(6):
            # Y grid
            y_val = ymin + i * (ymax - ymin) / 5.0
            y_pos = py + ph - i * ph / 5.0
            svg.append(f'<line x1="{px}" y1="{y_pos}" x2="{px+pw}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')
            svg.append(f'<text x="{px - 8}" y="{y_pos + 4}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" fill="#64748b">{y_val:.2f}</text>')
            
            # X grid
            x_val = xmin + i * (xmax - xmin) / 5.0
            x_pos = px + i * pw / 5.0
            svg.append(f'<line x1="{x_pos}" y1="{py}" x2="{x_pos}" y2="{py+ph}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')
            svg.append(f'<text x="{x_pos}" y="{py + ph + 18}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#64748b">{x_val:.1f}</text>')
            
        # Axis labels
        svg.append(f'<text x="{px + pw/2}" y="{py + ph + 38}" font-family="Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#475569">{xlabel}</text>')
        svg.append(f'<text x="{px - 45}" y="{py + ph/2}" font-family="Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" transform="rotate(-90 {px - 45} {py + ph/2})" fill="#475569">{ylabel}</text>')
        
        # Plot curves
        for label, ys, color, style, width in y_series:
            points = []
            for x, y in zip(x_data, ys):
                if math.isnan(y) or math.isinf(y):
                    if points:
                        pts_str = " ".join([f"{xp:.1f},{yp:.1f}" for xp, yp in points])
                        dash = 'stroke-dasharray="5,4"' if style == "dashed" else ""
                        svg.append(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="{width}" {dash}/>')
                        points = []
                    continue
                xp = px + (x - xmin) / (xmax - xmin) * pw
                yp = py + ph - (y - ymin) / (ymax - ymin) * ph
                points.append((xp, yp))
                
            if points:
                pts_str = " ".join([f"{xp:.1f},{yp:.1f}" for xp, yp in points])
                dash = 'stroke-dasharray="5,4"' if style == "dashed" else ""
                svg.append(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="{width}" {dash}/>')
                
        # Legend
        leg_x = px + 15
        leg_y = py + 20
        for i, (label, _, color, style, width) in enumerate(y_series):
            ly = leg_y + i * 18
            dash = 'stroke-dasharray="4,3"' if style == "dashed" else ""
            svg.append(f'<line x1="{leg_x}" y1="{ly}" x2="{leg_x+25}" y2="{ly}" stroke="{color}" stroke-width="{width}" {dash}/>')
            svg.append(f'<text x="{leg_x + 32}" y="{ly + 4}" font-family="Arial, sans-serif" font-size="10" font-weight="500" fill="#1e293b">{label}</text>')

    # 1. IDD Panel
    render_panel(90, 70, 440, 310, z_vals, [
        ("TOPAS Full Physics (Gy)", t_idd, "#0284c7", "solid", 2.5),
        ("GPU MAIGO (Gy)", g_idd, "#dc2626", "dashed", 2.0),
    ], "Integrated Depth Dose (IDD)", "Depth z (mm)", "Dose (Gy)")
    
    # 2. IDD Relative Diff Panel
    idd_diff = [(g - t) / t * 100.0 if t > 0 else 0.0 for t, g in zip(t_idd, g_idd)]
    render_panel(610, 70, 440, 310, z_vals, [
        ("IDD Diff: (GPU - TOPAS)/TOPAS (%)", idd_diff, "#7c3aed", "solid", 2.0),
    ], "IDD Relative Error (%)", "Depth z (mm)", "Relative Error (%)", -15.0, 15.0)
    
    # 3. Sigma Panel
    render_panel(90, 470, 440, 310, z_vals, [
        ("TOPAS Core σ (mm)", t_sig_c, "#059669", "solid", 2.2),
        ("GPU Core σ (mm)", g_sig_c, "#16a34a", "dashed", 2.0),
        ("TOPAS Halo σ (mm)", t_sig_h, "#d97706", "solid", 2.2),
        ("GPU Halo σ (mm)", g_sig_h, "#ea580c", "dashed", 2.0),
    ], "Lateral Beam Profile: Core & Halo Sigma", "Depth z (mm)", "Lateral Sigma (mm)")
    
    # 4. Sigma Relative Diff Panel
    sig_c_diff = [(gc - tc) / tc * 100.0 if tc > 0 else 0.0 for tc, gc in zip(t_sig_c, g_sig_c)]
    sig_h_diff = [(gh - th) / th * 100.0 if not math.isnan(th) and not math.isnan(gh) and th > 0 else float("nan") for th, gh in zip(t_sig_h, g_sig_h)]
    render_panel(610, 470, 440, 310, z_vals, [
        ("Core σ Diff (%)", sig_c_diff, "#059669", "solid", 2.0),
        ("Halo σ Diff (%)", sig_h_diff, "#d97706", "solid", 2.0),
    ], "Lateral Sigma Relative Error (%)", "Depth z (mm)", "Relative Error (%)", -15.0, 15.0)

    svg.append('</svg>')
    
    with open(filename, "w") as f:
        f.write("\n".join(svg))
    print(f"Generated {filename}")

os.makedirs("plots", exist_ok=True)
os.makedirs("/home/wuwei/.gemini/antigravity-cli/brain/b6e8050e-cd93-476b-bd8a-039915838a5f/plots", exist_ok=True)

for E, tpath, gpath, max_z in [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw", 45.0),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw", 120.0),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e300/topas_emittance_inelastic_e300.bin", "out/gpu_inelastic_e300/voxel_dose.raw", 210.0),
]:
    print(f"Processing {E} MeV/u curves...")
    z_vals, t_idd, g_idd, t_sc, t_sh, g_sc, g_sh = load_data(tpath, gpath, max_z)
    
    svg_file = f"plots/benchmark_{E}MeVu_idd_sigma.svg"
    draw_svg(svg_file, E, z_vals, t_idd, g_idd, t_sc, t_sh, g_sc, g_sh)
    
    # Copy to artifact directory
    art_svg = f"/home/wuwei/.gemini/antigravity-cli/brain/b6e8050e-cd93-476b-bd8a-039915838a5f/plots/benchmark_{E}MeVu_idd_sigma.svg"
    with open(svg_file, "r") as fin, open(art_svg, "w") as fout:
        fout.write(fin.read())
