# Double-precision transcription of G4UrbanMscModel::SampleCosineTheta
# (v11.3.2, mechanical port of the fetched source; electron branch validated
# to 8 digits against the executed libG4). DIAGNOSTIC ONLY: decides whether
# the FP32 mixture trial in MAIGO needs double precision (prompt B2).
# It is not a correctness proof of the port (see direct-G4 single-step
# comparison instead).
import math
import sys

MSK = 0xFFFFFFFF
M0, M1 = 0xD2511F53, 0xCD9E8D57
K0, K1 = 0x9E3779B9, 0xBB67AE85


def philox(c0, c1, c2, c3, k0, k1):
    c = [c0, c1, c2, c3]
    for _ in range(10):
        p0, p1 = M0 * c[0], M1 * c[2]
        c = [(p1 >> 32) & MSK ^ c[1] ^ k0, p1 & MSK,
             (p0 >> 32) & MSK ^ c[3] ^ k1, p0 & MSK]
        k0 = (k0 + K0) & MSK
        k1 = (k1 + K1) & MSK
    return c


def random_u32(seed, hist, ii, dim):
    b = dim // 4
    c = philox(hist & MSK, (hist >> 32) & MSK, ii & MSK,
               ((ii >> 32) & MSK) ^ b, seed & MSK, (seed >> 32) & MSK)
    return c[dim % 4]


INV24 = 2.0 ** -24


def uniform01(seed, hist, ii, dim):
    return ((random_u32(seed, hist, ii, dim) >> 8) + 0.5) * INV24


# Water_75eV couple constants (from MAIGO extraction metadata).
ZEFF = 3.3334
RADLEN = 360.829  # mm
MASS = 12.0 * 931.49410242  # C12 total MeV
LAMBDALIMIT = 1.0  # mm
TAUBIG, TAUSMALL = 8.0, 1e-16


def coeff_water():
    w = math.exp(math.log(ZEFF) / 6.0)
    facz = 0.990395 + w * (-0.168386 + w * 0.093286)
    z13 = w * w
    return {
        'th1': facz * (1.0 - 8.7780e-2 / ZEFF),
        'th2': facz * (4.0780e-2 + 1.7315e-4 * ZEFF),
        'c1': 2.3785 - z13 * (4.1981e-1 - z13 * 6.3100e-2),
        'c2': 4.7526e-1 + z13 * (1.7694 - z13 * 3.3885e-1),
        'c3': 2.3683e-1 - z13 * (1.8111 - z13 * 3.2774e-1),
        'c4': 1.7888e-2 + z13 * (1.9659e-2 - z13 * 2.6664e-3),
    }


COEF = coeff_water()


def lambda0_of_E(epre):
    # Transport MFP from the validated per-atom transcription (diagnostic).
    # Values pinned to MAIGO's table path: use the mfp printed by the
    # helpers test (250 MeV/u -> 5.8410692e7 mm) scaled by sigma ratios.
    # Simpler: Bragg-sum via the tag transcription in xsec_audit (import).
    from xsec_tag import tag_c12_sigma
    NA = 6.02214076e23
    n_h = NA * 1.0 * 0.111894 / 1.008
    n_o = NA * 1.0 * 0.888106 / 16.00
    s_h = tag_c12_sigma(epre, 1)
    s_o = tag_c12_sigma(epre, 8)
    return 10.0 / (n_h * s_h + n_o * s_o)


def mixture_audit(t_mm, epre, n, seed=777001):
    """Return (min_1mq, frac_q_le_1m7, frac_q_ge_1, flips_vs_float)."""
    lam0 = lambda0_of_E(epre)
    tau = t_mm / lam0
    # xmeanth (tau << 0.01 in scope -> series; keep general form)
    if tau < 0.01:
        xmeanth = 1.0 - tau * (1.0 - 0.5 * tau)
        x2meanth = 1.0 - tau * (5.0 - 6.25 * tau) / 3.0
    else:
        xmeanth = math.exp(-tau)
        x2meanth = (1.0 + 2.0 * math.exp(-2.5 * tau)) / 3.0
    kin = epre  # no loss within one audit step
    inv = (kin + MASS) / (kin * (kin + 2.0 * MASS))
    y = t_mm / RADLEN
    th0 = 13.6 * 6.0 * math.sqrt(y) * inv
    th0 *= COEF['th1'] + COEF['th2'] * math.log(y)
    th2 = th0 * th0
    x = th2 * (1.0 - th2 / 12.0)
    if th2 > 0.01:
        x = (2.0 * math.sin(0.5 * th0)) ** 2
    ltau = math.log(tau)
    u = math.exp(ltau / 6.0)
    lambeff = lam0
    xx = math.log(lambeff / RADLEN)
    xsi = COEF['c1'] + u * (COEF['c2'] + COEF['c3'] * u) + COEF['c4'] * xx
    xsi = max(xsi, 1.9)
    c = xsi
    if abs(c - 3.0) < 0.001:
        c = 3.001
    elif abs(c - 2.0) < 0.001:
        c = 2.001
    c1 = c - 1.0
    ea = math.exp(-xsi)
    eaa = 1.0 - ea
    xmean1 = 1.0 - (1.0 - (1.0 + xsi) * ea) * x / eaa
    x0 = 1.0 - xsi * x
    b = 1.0 + (c - xsi) * x
    b1 = b + 1.0
    bx = c * x
    eb1 = math.exp(math.log(b1) * c1)
    ebx = math.exp(math.log(bx) * c1)
    d = ebx / eb1
    xmean2 = (x0 + d - (bx - b1 * d) / (c - 2.0)) / (1.0 - d)
    f1x0 = ea / eaa
    f2x0 = c1 / (c * (1.0 - d))
    prob = f2x0 / (f1x0 + f2x0)
    q = xmeanth / (prob * xmean1 + (1.0 - prob) * xmean2)
    # float32 shadow of the same algebra
    import struct

    def f32(x):
        return struct.unpack('f', struct.pack('f', x))[0]

    xmeanth_f = f32(1.0 - f32(tau) * (1.0 - 0.5 * f32(tau)))
    q_f = f32(xmeanth_f / f32(f32(prob) * f32(xmean1) +
                              f32(1.0 - prob) * f32(xmean2)))
    one_m_q = 1.0 - q
    # decision-flip probe: same u0 stream, float-q vs double-q trial
    flips = 0
    for i in range(n):
        u0 = uniform01(seed, i, 0, 70)
        take_f = u0 < float(q_f)
        take_d = u0 < q
        flips += (take_f != take_d)
    return {'q': q, '1-q': one_m_q, 'q_f32': q_f, 'prob': prob,
            'flips': flips, 'n': n, 'tau': tau, 'xsi': xsi}


if __name__ == '__main__':
    sys.path.insert(0, '.')
    for (t, E) in [(0.05, 3000.0), (0.05, 1000.0), (0.05, 300.0),
                   (0.05, 100.0), (0.025, 3000.0), (0.10, 3000.0)]:
        r = mixture_audit(t, E, 20000)
        print('t=%.3f E=%7.1f q=%.10f 1-q=%.3e q_f32=%a flips=%d/%d '
              'tau=%.2e xsi=%.4f' % (t, E, r['q'], r['1-q'], r['q_f32'],
                                     r['flips'], r['n'], r['tau'], r['xsi']))
