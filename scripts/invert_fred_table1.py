#!/usr/bin/env python3
"""Newton inversion of independent Table-1 sampling probabilities.

Sequential nucleon-conserving draws do not reproduce inclusive Table 1 yields
when the raw percentages are used as the independent vector. This script
inverts p so that projectile-fragment inclusive fractions match Table 1.
"""
from __future__ import annotations

import numpy as np

ZA = [
    (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 6), (3, 7),
    (4, 7), (4, 9), (4, 10), (5, 8), (5, 10), (5, 11), (6, 10), (6, 11), (6, 12),
]
PROB_H = np.array(
    [14.0, 38.0, 5.0, 1.3, 3.0, 26.0, 0.036, 2.3, 0.93, 1.6, 0.25, 0.0001,
     0.15, 1.3, 2.1, 0.19, 3.9, 0.59], dtype=np.float64)
PROB_O = np.array(
    [60.0, 16.0, 8.8, 5.1, 1.7, 6.3, 1.0, 0.28, 0.39, 0.12, 0.079, 0.10,
     0.014, 0.086, 0.18, 0.016, 0.071, 0.079], dtype=np.float64)


def sample_iso(p: np.ndarray, a_rem: int, z_rem: int, u: float) -> int:
    n_rem = a_rem - z_rem
    tot = 0.0
    for i, (z, a) in enumerate(ZA):
        ni = a - z
        if z <= z_rem and ni <= n_rem and a > 0:
            tot += p[i]
    if tot <= 0.0:
        if z_rem > 0 and n_rem > 0:
            return 2
        if z_rem > 0:
            return 1
        return 0
    target = u * tot
    cum = 0.0
    last = 0
    for i, (z, a) in enumerate(ZA):
        ni = a - z
        if z <= z_rem and ni <= n_rem and a > 0:
            cum += p[i]
            last = i
            if target <= cum:
                return i
    return last


def event_projectile(p: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    a, z = 12, 6
    counts = np.zeros(18, dtype=np.float64)
    steps = 0
    while (a > 0 or z > 0) and steps < 16:
        steps += 1
        i = sample_iso(p, a, z, float(rng.random()))
        zz, aa = ZA[i]
        if aa <= a and zz <= z and aa > 0:
            counts[i] += 1.0
            a -= aa
            z -= zz
        elif z > 0 and a > 0:
            counts[1] += 1.0
            a -= 1
            z -= 1
        elif a > 0:
            counts[0] += 1.0
            a -= 1
        else:
            break
    return counts, a, z


def simulate(p: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tot = np.zeros(18, dtype=np.float64)
    for _ in range(n):
        c, _, _ = event_projectile(p, rng)
        tot += c
    return tot / float(n)


def invert(table: np.ndarray, n: int = 4000, iters: int = 8) -> np.ndarray:
    p = np.maximum(table.copy(), 1e-8)
    fexp = table / table.sum()
    for it in range(iters):
        f = simulate(p, n, seed=1 + it)
        fn = f / f.sum()
        err = fexp - fn
        print(f"  iter {it:2d}  max|err|={np.max(np.abs(err)):.4f}  l2={np.linalg.norm(err):.4f}")
        J = np.zeros((18, 18), dtype=np.float64)
        for j in range(18):
            dp = p.copy()
            dp[j] = dp[j] * 1.20 + 1e-4
            fj = simulate(dp, max(n // 2, 2000), seed=2000 + it * 37 + j)
            fjn = fj / fj.sum()
            J[:, j] = (fjn - fn) / (dp[j] - p[j])
        A = J.T @ J + 3e-4 * np.eye(18)
        b = J.T @ err
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        p = np.clip(p + 0.35 * d * (p + 1e-3), 1e-8, None)
    return p


def fmt_array(name: str, p: np.ndarray) -> str:
    body = ", ".join(f"{x:.6g}F" for x in p)
    return f"inline constexpr std::array<float, 18> {name} = {{\n    {body}\n}};\n"


def main() -> None:
    print("Inverting H target")
    p_h = invert(PROB_H)
    print("Inverting O target")
    p_o = invert(PROB_O)
    print("\n// --- C++ ---")
    print(fmt_array("kFredSampleProbH", p_h))
    print(fmt_array("kFredSampleProbO", p_o))
    for label, table, p in (("H", PROB_H, p_h), ("O", PROB_O, p_o)):
        f = simulate(p, 40000, seed=99)
        fn = f / f.sum()
        fe = table / table.sum()
        print(label, "max abs frac err", np.max(np.abs(fn - fe)))
        print(" p=", np.array2string(p, precision=5, separator=", "))


if __name__ == "__main__":
    main()
