# Double transcription of ComputeCrossSectionPerAtom (v11.3.2 tag).
# Validated: matches the executed local libG4 to 6e-5 over the C12 grid
# 1-3600 MeV (see evidence manifest). Diagnostic use only.
import math

ME = 0.51099895
SIG0 = [0.2672, 0.5922, 2.653, 6.235, 11.69, 13.24, 16.12, 23.0, 35.13,
        39.95, 50.85, 67.19, 91.15, 104.4, 113.1]
HEC = [120.7, 117.5, 105.0, 92.92, 79.23, 74.51, 68.29, 57.39, 41.97,
       36.14, 24.53, 10.21, -7.855, -16.84, -22.3]
ZDAT = [4, 6, 13, 20, 26, 29, 32, 38, 47, 50, 56, 64, 74, 79, 82]
BARN = 1e-24
CPOS = {
    0: [2.589, 2.044, 1.658, 1.446, 1.347, 1.217, 1.144, 1.110, 1.097,
        1.083, 1.080, 1.086, 1.092, 1.108, 1.123, 1.131, 1.131, 1.126,
        1.117, 1.108, 1.103, 1.100],
    1: [3.904, 2.794, 2.079, 1.710, 1.543, 1.325, 1.202, 1.145, 1.122,
        1.096, 1.089, 1.092, 1.098, 1.114, 1.130, 1.137, 1.138, 1.132,
        1.122, 1.113, 1.108, 1.102],
    2: [7.970, 6.080, 4.442, 3.398, 2.872, 2.127, 1.672, 1.451, 1.357,
        1.246, 1.194, 1.179, 1.178, 1.188, 1.201, 1.205, 1.203, 1.190,
        1.173, 1.159, 1.151, 1.145],
}
TDAT = [100e-6, 200e-6, 400e-6, 700e-6, 1e-3, 2e-3, 4e-3, 7e-3, 1e-2,
        2e-2, 4e-2, 7e-2, 1e-1, 2e-1, 4e-1, 7e-1, 1.0, 2.0, 4.0, 7.0,
        10.0, 20.0]


def tag_c12_sigma(E_total_MeV, Z, charge=6, A=12):
    m = A * 931.49410242
    TAU = E_total_MeV / m
    c = m * TAU * (TAU + 2) / (ME * (TAU + 1))
    w = c - 2
    ekin = ME * 0.5 * (w + math.sqrt(w * w + 4 * c))
    etot = ekin + ME
    b2 = ekin * (etot + ME) / (etot * etot)
    bg2 = ekin * (etot + ME) / (ME * ME)
    z23 = Z ** (2 / 3)
    epsf = 2 * ME * ME * (5.29177210903e-9) ** 2 / (1.973269804e-11) ** 2
    eps = epsf * bg2 / z23
    if eps < 1e-4:
        s = 2 * eps * eps
    elif eps < 1e10:
        s = math.log(1 + 2 * eps) - 2 * eps / (1 + 2 * eps)
    else:
        s = math.log(2 * eps) - 1 + 1 / eps
    s *= charge * charge * Z * Z / (b2 * bg2)
    iz = 14
    while iz >= 0 and ZDAT[iz] >= Z:
        iz -= 1
    iz = max(0, min(13, iz))
    z1, z2 = ZDAT[iz], ZDAT[iz + 1]
    ratz = (Z - z1) * (Z + z1) / ((z2 - z1) * (z2 + z1))
    Tlim = 10.0
    b2lim = Tlim * (Tlim + 2 * ME) / ((Tlim + ME) ** 2)
    bg2lim = Tlim * (Tlim + 2 * ME) / (ME * ME)
    if ekin <= Tlim:
        it = 21
        while it >= 0 and TDAT[it] >= ekin:
            it -= 1
        it = max(0, min(20, it))
        t0 = TDAT[it]
        e0 = t0 + ME
        b2s = t0 * (e0 + ME) / (e0 * e0)
        t1 = TDAT[it + 1]
        e1 = t1 + ME
        b2b = t1 * (e1 + ME) / (e1 * e1)
        rb2 = (b2 - b2s) / (b2b - b2s)
        lo = CPOS[0 if iz == 0 else 1]
        hi = CPOS[1 if iz == 0 else 2]
        cc1 = lo[it] + ratz * (hi[it] - lo[it])
        cc2 = lo[it + 1] + ratz * (hi[it + 1] - lo[it + 1])
        s *= 2 * math.pi * (2.8179403262e-13) ** 2 / (cc1 + rb2 * (cc2 - cc1))
    else:
        c1 = bg2lim * SIG0[iz] * BARN * (1 + HEC[iz] * (b2 - b2lim)) / bg2
        c2 = bg2lim * SIG0[iz + 1] * BARN * (1 + HEC[iz + 1] * (b2 - b2lim)) / bg2
        if z1 <= Z <= z2:
            s = c1 + ratz * (c2 - c1)
        elif Z < z1:
            s = Z * Z * c1 / (z1 * z1)
        else:
            s = Z * Z * c2 / (z2 * z2)
    s *= 1 + 0.30 / (1 + math.sqrt(1000 * ekin))
    return s
