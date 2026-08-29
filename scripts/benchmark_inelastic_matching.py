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

def fit_double_gaussian(xs, ys):
    peak_y = max(ys)
    total = sum(ys)
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_x = sum(x * y for x, y in zip(xs, ys)) / total
    var_x = sum(((x - mean_x)**2) * y for x, y in zip(xs, ys)) / total
    sig0 = math.sqrt(max(0.1, var_x))
    
    def loss(p):
        Ac, sc, Ah, sh = p
        if sc < 0.5 or sh < sc * 1.20 or Ac < 0 or Ah < 0:
            return 1e18
        if Ah > Ac * 1.2:
            return 1e18
        res = 0.0
        for x, y in zip(xs, ys):
            dx = x - mean_x
            pred = Ac * math.exp(-0.5 * (dx / sc)**2) + Ah * math.exp(-0.5 * (dx / sh)**2)
            diff = y - pred
            res += diff * diff
        return res

    best_loss = 1e20
    best_p = None
    starts = [
        [peak_y * 0.95, sig0 * 0.98, peak_y * 0.05, sig0 * 2.0],
        [peak_y * 0.90, sig0 * 0.90, peak_y * 0.10, sig0 * 3.0],
        [peak_y * 0.80, sig0 * 0.85, peak_y * 0.20, sig0 * 1.8],
    ]
    for x0 in starts:
        p, score = nelder_mead(loss, x0, step=0.1, max_iter=2000)
        if score < best_loss:
            best_loss = score
            best_p = p

    Ac, sc, Ah, sh = best_p
    if sc > sh:
        sc, sh = sh, sc
        Ac, Ah = Ah, Ac

    area_c = Ac * math.sqrt(2.0 * math.pi) * sc
    area_h = Ah * math.sqrt(2.0 * math.pi) * sh
    w_halo = area_h / (area_c + area_h) if (area_c + area_h) > 0 else 0.0
    return sc, sh, w_halo, Ac, Ah

def load_voxel_grid(path, is_double=True):
    nx, ny, nz = 400, 400, 800
    dx, dy, dz = 0.2, 0.2, 0.5
    x_coords = [(ix - nx / 2.0 + 0.5) * dx for ix in range(nx)]
    
    if not os.path.exists(path) or os.path.getsize(path) < nx*ny*nz*(8 if is_double else 4):
        return None, None, None, None

    idd = []
    x_projs = {}
    sig_xs = []
    
    fmt = "d" if is_double else "f"
    bytes_per_slice = nx * ny * (8 if is_double else 4)
    
    with open(path, "rb") as f:
        for iz in range(nz):
            buf = f.read(bytes_per_slice)
            if not buf: break
            arr = array.array(fmt, buf)
            tot = sum(arr)
            idd.append(tot)
            if tot > 1e-4:
                x_proj = [0.0] * nx
                for iy in range(ny):
                    off = iy * nx
                    for ix in range(nx):
                        x_proj[ix] += arr[off + ix]
                mean_x = sum(x * p for x, p in zip(x_coords, x_proj)) / tot
                var_x = sum(((x - mean_x)**2) * p for x, p in zip(x_coords, x_proj)) / tot
                sig_xs.append(math.sqrt(var_x))
                x_projs[iz] = x_proj
            else:
                sig_xs.append(0.0)
                
    return x_coords, idd, sig_xs, x_projs

def benchmark_energy(E, tpath, gpath):
    print("=" * 115)
    print(">>> BENCHMARKING %d MeV/u (TOPAS INCL++ vs GPU MAIGO)" % E)
    print("=" * 115)
    
    x_coords, t_idd, t_sig, t_xprojs = load_voxel_grid(tpath, is_double=True)
    _, g_idd, g_sig, g_xprojs = load_voxel_grid(gpath, is_double=False)
    
    if t_idd is None:
        print("[!] TOPAS dataset not yet available or incomplete: %s" % tpath)
        return
    if g_idd is None:
        print("[!] GPU dataset not yet available or incomplete: %s" % gpath)
        return
        
    dz = 0.5
    peak_t = max(range(len(t_idd)), key=lambda i: t_idd[i])
    peak_g = max(range(len(g_idd)), key=lambda i: g_idd[i])
    
    pos_diff = (peak_g - peak_t) * dz
    val_diff = (g_idd[peak_g] - t_idd[peak_t]) / t_idd[peak_t] * 100
    
    print("Bragg Peak Range : TOPAS = %6.2f mm (iz=%d) | GPU = %6.2f mm (iz=%d) | Diff = %+6.2f mm" % (
        (peak_t+0.5)*dz, peak_t, (peak_g+0.5)*dz, peak_g, pos_diff))
    print("Bragg Peak Dose  : TOPAS = %6.4f Gy      | GPU = %6.4f Gy      | Diff = %+6.2f%%" % (
        t_idd[peak_t], g_idd[peak_g], val_diff))
    print("-" * 115)
    
    step = max(1, peak_t // 6)
    sample_slices = [0, step, step*2, step*3, step*4, step*5, peak_t, peak_t + 8, peak_t + 16]
    sample_slices = sorted(list(set(sample_slices)))
    
    print("%7s | %10s | %10s | %8s | %11s | %11s | %8s | %11s | %11s | %8s" % (
        "z (mm)", "TOPAS IDD", "GPU IDD", "IDD Diff", "TOPAS sig_c", "TOPAS sig_h", "TOPAS w_h", "GPU sig_c", "GPU sig_h", "GPU w_h"))
    print("-" * 115)
    
    for iz in sample_slices:
        if iz >= len(t_idd) or iz >= len(g_idd): continue
        z_mm = (iz + 0.5) * dz
        t_val = t_idd[iz]
        g_val = g_idd[iz]
        idd_d = (g_val - t_val) / t_val * 100 if t_val > 0 else 0
        
        t_sc, t_sh, t_wh = 0, 0, 0
        g_sc, g_sh, g_wh = 0, 0, 0
        if iz in t_xprojs and sum(t_xprojs[iz]) > 1e-3:
            t_sc, t_sh, t_wh, _, _ = fit_double_gaussian(x_coords, t_xprojs[iz])
        if iz in g_xprojs and sum(g_xprojs[iz]) > 1e-3:
            g_sc, g_sh, g_wh, _, _ = fit_double_gaussian(x_coords, g_xprojs[iz])
            
        print("%7.1f | %10.4f | %10.4f | %+7.2f%% | %11.4f | %11.4f | %7.2f%% | %11.4f | %11.4f | %7.2f%%" % (
            z_mm, t_val, g_val, idd_d, t_sc, t_sh, t_wh*100, g_sc, g_sh, g_wh*100))
    print()

for E, tfile, gfile in [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw"),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw"),
    (300, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e300/topas_emittance_inelastic_e300.bin", "out/gpu_inelastic_e300/voxel_dose.raw"),
    (400, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e400/topas_emittance_inelastic_e400.bin", "out/gpu_inelastic_e400/voxel_dose.raw"),
]:
    benchmark_energy(E, tfile, gfile)
