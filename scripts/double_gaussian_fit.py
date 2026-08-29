import array, math

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
    
    # Model: f(x) = A_core * exp(-x^2 / (2*sc^2)) + A_halo * exp(-x^2 / (2*sh^2))
    # Constraints: sc in [0.5, sig0*1.5], sh >= 1.25 * sc, Ac >= Ah
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

    # Multi-start optimization
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
    # Ensure sc is the core (narrower) and sh is the halo (wider)
    if sc > sh:
        sc, sh = sh, sc
        Ac, Ah = Ah, Ac

    area_c = Ac * math.sqrt(2.0 * math.pi) * sc
    area_h = Ah * math.sqrt(2.0 * math.pi) * sh
    w_halo = area_h / (area_c + area_h) if (area_c + area_h) > 0 else 0.0
    return sc, sh, w_halo, Ac, Ah

def extract_profiles(topas_path, gpu_path, nz_slices):
    nx, ny, nz = 400, 400, 800
    dx, dy, dz = 0.2, 0.2, 0.5
    x_coords = [(ix - nx / 2.0 + 0.5) * dx for ix in range(nx)]

    topas_data = {}
    with open(topas_path, "rb") as f:
        slice_floats = nx * ny
        for iz in range(nz):
            buf = f.read(slice_floats * 8)
            if not buf: break
            if iz in nz_slices:
                arr = array.array("d", buf)
                x_proj = [0.0] * nx
                for iy in range(ny):
                    offset = iy * nx
                    for ix in range(nx):
                        x_proj[ix] += arr[offset + ix]
                topas_data[iz] = x_proj

    gpu_data = {}
    with open(gpu_path, "rb") as f:
        slice_floats = nx * ny
        for iz in range(nz):
            buf = f.read(slice_floats * 4)
            if not buf: break
            if iz in nz_slices:
                arr = array.array("f", buf)
                x_proj = [0.0] * nx
                for iy in range(ny):
                    offset = iy * nx
                    for ix in range(nx):
                        x_proj[ix] += arr[offset + ix]
                gpu_data[iz] = x_proj

    return x_coords, topas_data, gpu_data

for E, tpath, gpath, peak_iz in [
    (100, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100/topas_emittance_inelastic_e100.bin", "out/gpu_inelastic_e100/voxel_dose.raw", 51),
    (200, "/mnt/sda/wuwei/carbon_emittance_inelastic_100k/e200/topas_emittance_inelastic_e200.bin", "out/gpu_inelastic_e200/voxel_dose.raw", 173)
]:
    step = max(1, peak_iz // 5)
    slices = [0, step, step*2, step*3, step*4, peak_iz, peak_iz + 8, peak_iz + 16]
    x_coords, t_profs, g_profs = extract_profiles(tpath, gpath, slices)

    print("=" * 125)
    print("=== %d MeV/u Double-Gaussian Fitting (TOPAS vs GPU) ===" % E)
    print("%7s | %11s | %11s | %9s | %11s | %11s | %9s | %10s | %10s" % (
        "z (mm)", "TOPAS sig_c", "TOPAS sig_h", "TOPAS w_h", "GPU sig_c", "GPU sig_h", "GPU w_h", "sig_c Diff", "sig_h Diff"))
    print("-" * 125)
    for iz in sorted(slices):
        z_mm = (iz + 0.5) * 0.5
        t_ys = t_profs.get(iz, [])
        g_ys = g_profs.get(iz, [])
        if not t_ys or not g_ys or sum(t_ys) < 1e-3 or sum(g_ys) < 1e-3:
            continue
        t_sc, t_sh, t_wh, _, _ = fit_double_gaussian(x_coords, t_ys)
        g_sc, g_sh, g_wh, _, _ = fit_double_gaussian(x_coords, g_ys)
        sc_diff = (g_sc - t_sc) / t_sc * 100 if t_sc > 0 else 0
        sh_diff = (g_sh - t_sh) / t_sh * 100 if t_sh > 0 else 0
        print("%7.1f | %11.4f | %11.4f | %8.2f%% | %11.4f | %11.4f | %8.2f%% | %+9.2f%% | %+9.2f%%" % (
            z_mm, t_sc, t_sh, t_wh * 100, g_sc, g_sh, g_wh * 100, sc_diff, sh_diff))
