#include <cmath>

#include "carbon/fred_table1.hpp"

#include <algorithm>
#include <vector>

namespace carbon {
namespace {

std::uint32_t xorshift32(std::uint32_t& state) noexcept {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

float u01(std::uint32_t& state) noexcept {
    return (xorshift32(state) >> 8U) * (1.0F / 16777216.0F);
}

void gauss_solve_18(double a[18][18], double b[18], double x[18]) {
    double m[18][19]{};
    for (int i = 0; i < 18; ++i) {
        for (int j = 0; j < 18; ++j) {
            m[i][j] = a[i][j];
        }
        m[i][18] = b[i];
    }
    for (int col = 0; col < 18; ++col) {
        int piv = col;
        double best = std::fabs(m[col][col]);
        for (int r = col + 1; r < 18; ++r) {
            const double v = std::fabs(m[r][col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1.0e-18) {
            continue;
        }
        if (piv != col) {
            for (int j = col; j < 19; ++j) {
                std::swap(m[col][j], m[piv][j]);
            }
        }
        const double diag = m[col][col];
        for (int j = col; j < 19; ++j) {
            m[col][j] /= diag;
        }
        for (int r = 0; r < 18; ++r) {
            if (r == col) {
                continue;
            }
            const double f = m[r][col];
            if (f == 0.0) {
                continue;
            }
            for (int j = col; j < 19; ++j) {
                m[r][j] -= f * m[col][j];
            }
        }
    }
    for (int i = 0; i < 18; ++i) {
        x[i] = m[i][18];
        if (!std::isfinite(x[i])) {
            x[i] = 0.0;
        }
    }
}

void one_conserving_event(const float* p, int a0, int z0, std::uint32_t& rng,
                          double counts[18], bool* closed, int max_frags,
                          bool stop_on_heavy) {
    int a_rem = a0;
    int z_rem = z0;
    int steps = 0;
    int num_frags = 0;
    while ((a_rem > 0 || z_rem > 0) && steps < 16 && num_frags < max_frags) {
        ++steps;
        const int iso_idx = sample_table1_isotope(p, a_rem, z_rem, u01(rng));
        const auto iso = kFredIsotopes[static_cast<std::size_t>(iso_idx)];
        if (iso.a <= a_rem && iso.z <= z_rem && iso.a > 0) {
            counts[iso_idx] += 1.0;
            a_rem -= iso.a;
            z_rem -= iso.z;
            ++num_frags;
            if (stop_on_heavy && (iso.z >= 3 || iso.a >= 6)) {
                break;
            }
        } else if (z_rem > 0 && a_rem > 0) {
            counts[1] += 1.0;
            a_rem -= 1;
            z_rem -= 1;
            ++num_frags;
        } else if (a_rem > 0) {
            counts[0] += 1.0;
            a_rem -= 1;
            ++num_frags;
        } else {
            break;
        }
    }
    *closed = (a_rem == 0 && z_rem == 0) || stop_on_heavy;
}

}  // namespace

void simulate_nucleon_conserving_inclusive(const float* sample_prob,
                                           int a0,
                                           int z0,
                                           unsigned events,
                                           std::uint32_t seed,
                                           double counts_out[18],
                                           unsigned* closed_events,
                                           int max_frags,
                                           bool stop_on_heavy) {
    for (int i = 0; i < 18; ++i) {
        counts_out[i] = 0.0;
    }
    unsigned closed = 0;
    const std::uint32_t base = seed == 0U ? 1U : seed;
    for (unsigned e = 0; e < events; ++e) {
        std::uint32_t rng = base ^ (e * 747796405U + 2891336453U);
        if (rng == 0U) {
            rng = 1U;
        }
        double local[18]{};
        bool ok = false;
        one_conserving_event(sample_prob, a0, z0, rng, local, &ok, max_frags, stop_on_heavy);
        for (int i = 0; i < 18; ++i) {
            counts_out[i] += local[i];
        }
        if (ok) {
            ++closed;
        }
    }
    if (closed_events != nullptr) {
        *closed_events = closed;
    }
}

void simulate_projectile_inclusive(const float* sample_prob,
                                   unsigned events,
                                   std::uint32_t seed,
                                   double counts_out[18],
                                   unsigned* closed_events) {
    simulate_nucleon_conserving_inclusive(sample_prob, 12, 6, events, seed, counts_out,
                                          closed_events);
}

Table1InvertResult invert_table1_independent_probs(const float* table_percent,
                                                   int a0,
                                                   int z0,
                                                   unsigned events,
                                                   unsigned iterations,
                                                   std::uint32_t seed) {
    Table1InvertResult result;
    result.events = events;
    double table_sum = 0.0;
    for (int i = 0; i < 18; ++i) {
        table_sum += static_cast<double>(table_percent[i]);
        result.sample_prob[static_cast<std::size_t>(i)] =
            std::max(table_percent[i], 1.0e-8F);
    }
    for (int i = 0; i < 18; ++i) {
        result.table_fraction[static_cast<std::size_t>(i)] =
            static_cast<double>(table_percent[i]) / table_sum;
    }

    std::array<float, 18> p = result.sample_prob;
    for (unsigned it = 0; it < iterations; ++it) {
        double f[18]{};
        unsigned closed = 0;
        simulate_nucleon_conserving_inclusive(p.data(), a0, z0, events, seed, f, &closed);
        result.closed_events = closed;
        double fsum = 0.0;
        for (int i = 0; i < 18; ++i) {
            fsum += f[i];
        }
        if (fsum <= 0.0) {
            break;
        }
        double fn[18]{};
        double err[18]{};
        float max_err = 0.0F;
        for (int i = 0; i < 18; ++i) {
            fn[i] = f[i] / fsum;
            err[i] = result.table_fraction[static_cast<std::size_t>(i)] - fn[i];
            max_err = std::max(max_err, static_cast<float>(std::fabs(err[i])));
        }
        result.max_abs_fraction_error = max_err;
        result.iterations = static_cast<int>(it + 1);
        for (int i = 0; i < 18; ++i) {
            result.inclusive_fraction[static_cast<std::size_t>(i)] = fn[i];
        }

        double J[18][18]{};
        const unsigned n_pert = std::max(events / 2U, 1500U);
        for (int j = 0; j < 18; ++j) {
            auto dp = p;
            dp[static_cast<std::size_t>(j)] =
                dp[static_cast<std::size_t>(j)] * 1.25F + 1.0e-4F;
            double fj[18]{};
            simulate_nucleon_conserving_inclusive(dp.data(), a0, z0, n_pert, seed, fj, nullptr);
            double fjsum = 0.0;
            for (int i = 0; i < 18; ++i) {
                fjsum += fj[i];
            }
            const double denom =
                static_cast<double>(dp[static_cast<std::size_t>(j)] - p[static_cast<std::size_t>(j)]);
            for (int i = 0; i < 18; ++i) {
                const double fjn = (fjsum > 0.0) ? fj[i] / fjsum : 0.0;
                J[i][j] = (fjn - fn[i]) / (denom == 0.0 ? 1.0 : denom);
            }
        }
        double A[18][18]{};
        double b[18]{};
        for (int r = 0; r < 18; ++r) {
            for (int c = 0; c < 18; ++c) {
                double s = (r == c) ? 4.0e-4 : 0.0;
                for (int k = 0; k < 18; ++k) {
                    s += J[k][r] * J[k][c];
                }
                A[r][c] = s;
            }
            double sb = 0.0;
            for (int k = 0; k < 18; ++k) {
                sb += J[k][r] * err[k];
            }
            b[r] = sb;
        }
        double d[18]{};
        gauss_solve_18(A, b, d);
        for (int i = 0; i < 18; ++i) {
            const double updated =
                static_cast<double>(p[static_cast<std::size_t>(i)]) +
                0.30 * d[i] * (static_cast<double>(p[static_cast<std::size_t>(i)]) + 1.0e-3);
            p[static_cast<std::size_t>(i)] =
                static_cast<float>(std::clamp(updated, 1.0e-8, 1.0e6));
        }
        result.sample_prob = p;
    }
    return result;
}

}  // namespace carbon
