#pragma once

#include <cstdint>
#include <array>
#include <cmath>

namespace carbon {

// Fixed 95 MeV/u reference support; residual tail is below 0.7% at the
// smallest fitted exponential slope.
inline constexpr float kFredReferenceEnergyMaxMeVu = 1000.0F;

struct FredIsotope {
    int16_t z;
    int16_t a;
    const char* name;
};

inline constexpr std::array<FredIsotope, 18> kFredIsotopes = {{
    {0, 1, "n"},
    {1, 1, "1H"},
    {1, 2, "2H"},
    {1, 3, "3H"},
    {2, 3, "3He"},
    {2, 4, "4He"},
    {2, 6, "6He"},
    {3, 6, "6Li"},
    {3, 7, "7Li"},
    {4, 7, "7Be"},
    {4, 9, "9Be"},
    {4, 10, "10Be"},
    {5, 8, "8B"},
    {5, 10, "10B"},
    {5, 11, "11B"},
    {6, 10, "10C"},
    {6, 11, "11C"},
    {6, 12, "12C"}
}};

// Parameters of Eq. 12 in FRED paper (A1, A2, <E>, sE, <theta>, stheta, aE, atheta)
struct Fred2DParams {
    float a1;
    float a2;
    float mean_e;
    float sigma_e;
    float mean_th;
    float sigma_th;
    float a_e;
    float a_th;
};

// Table 1: Production probabilities [%]
inline constexpr std::array<float, 18> kFredProbH = {
    14.0F, 38.0F, 5.0F, 1.3F, 3.0F, 26.0F, 0.036F, 2.3F, 0.93F, 1.6F, 0.25F, 0.0001F, 0.15F, 1.3F, 2.1F, 0.19F, 3.9F, 0.59F
};

inline constexpr std::array<float, 18> kFredProbC = {
    65.0F, 10.0F, 7.5F, 6.1F, 1.2F, 6.4F, 1.7F, 0.25F, 0.41F, 0.083F, 0.11F, 0.24F, 0.013F, 0.089F, 0.20F, 0.017F, 0.055F, 0.043F
};

inline constexpr std::array<float, 18> kFredProbO = {
    60.0F, 16.0F, 8.8F, 5.1F, 1.7F, 6.3F, 1.0F, 0.28F, 0.39F, 0.12F, 0.079F, 0.10F, 0.014F, 0.086F, 0.18F, 0.016F, 0.071F, 0.079F
};

inline constexpr std::array<float, 18> kFredCdfH = {
    0.139101F, 0.516662F, 0.566341F, 0.579257F, 0.609065F, 0.867396F, 0.867753F, 0.890606F, 0.899846F, 0.915743F, 0.918227F, 0.918228F, 0.919719F, 0.932635F, 0.953500F, 0.955388F, 0.994138F, 1.000000F
};

inline constexpr std::array<float, 18> kFredCdfC = {
    0.653858F, 0.754451F, 0.829896F, 0.891258F, 0.903330F, 0.967709F, 0.984810F, 0.987325F, 0.991450F, 0.992284F, 0.993391F, 0.995805F, 0.995936F, 0.996831F, 0.998843F, 0.999014F, 0.999567F, 1.000000F
};

inline constexpr std::array<float, 18> kFredCdfO = {
    0.598116F, 0.757614F, 0.845337F, 0.896177F, 0.913124F, 0.975926F, 0.985894F, 0.988686F, 0.992573F, 0.993770F, 0.994557F, 0.995554F, 0.995694F, 0.996551F, 0.998345F, 0.998505F, 0.999212F, 1.000000F
};

// Table 2: 1H Target parameters
inline constexpr std::array<Fred2DParams, 18> kFredParamsH = {{
    {0.54F, 49.0F, 93.0F, 33.0F, 0.0F, 9.7F, 0.010F, 0.025F},
    {0.54F, 49.0F, 93.0F, 33.0F, 0.0F, 9.7F, 0.012F, 0.025F},
    {0.20F, 7.5F, 81.0F, 23.0F, 0.0F, 8.2F, 0.022F, 0.065F},
    {0.078F, 2.7F, 76.0F, 22.0F, 0.0F, 6.8F, 0.019F, 0.18F},
    {0.022F, 7.6F, 96.0F, 27.0F, 0.0F, 6.2F, 0.018F, 0.10F},
    {0.0098F, 56.0F, 84.0F, 12.0F, 0.0F, 4.3F, 0.015F, 0.18F},
    {0.030F, 3.1F, 78.0F, 14.0F, 0.0F, 4.0F, 0.026F, 0.27F},
    {0.0080F, 4.5F, 84.0F, 11.0F, 0.0F, 3.4F, 0.019F, 0.20F},
    {0.019F, 3.3F, 79.0F, 8.6F, 0.0F, 3.2F, 0.020F, 0.27F},
    {0.0036F, 6.6F, 85.0F, 11.0F, 0.0F, 3.1F, 0.017F, 0.21F},
    {0.020F, 1.3F, 82.0F, 7.3F, 0.0F, 3.0F, 0.023F, 0.27F},
    {0.071F, 0.38F, 79.0F, 5.3F, 0.0F, 3.1F, 0.024F, 0.32F},
    {0.63F, 0.0083F, 89.0F, 13.0F, 0.0F, 3.2F, 0.022F, 0.16F},
    {0.0035F, 13.0F, 83.0F, 6.5F, 0.0F, 2.5F, 0.019F, 0.37F},
    {0.011F, 1.4F, 83.0F, 4.6F, 0.0F, 2.2F, 0.018F, 0.58F},
    {0.0012F, 3.7F, 88.0F, 7.2F, 0.0F, 2.3F, 0.018F, 0.21F},
    {0.00050F, 39.0F, 84.0F, 4.8F, 0.0F, 2.1F, 0.017F, 0.30F},
    {0.00050F, 93.0F, 83.0F, 3.6F, 0.0F, 0.92F, 0.010F, 0.20F}
}};

// Table 3: 12C Target parameters
inline constexpr std::array<Fred2DParams, 18> kFredParamsC = {{
    {0.28F, 100.0F, 93.0F, 37.0F, 0.0F, 10.0F, 0.012F, 0.025F},
    {0.28F, 100.0F, 93.0F, 37.0F, 0.0F, 10.0F, 0.0013F, 0.026F},
    {0.27F, 54.0F, 81.0F, 26.0F, 0.0F, 8.9F, 0.026F, 0.031F},
    {0.26F, 25.0F, 73.0F, 18.0F, 0.0F, 7.6F, 0.032F, 0.057F},
    {0.15F, 36.0F, 92.0F, 29.0F, 0.0F, 7.0F, 0.031F, 0.047F},
    {0.074F, 190.0F, 83.0F, 15.0F, 0.0F, 5.2F, 0.029F, 0.080F},
    {0.081F, 10.0F, 78.0F, 17.0F, 0.0F, 5.4F, 0.030F, 0.15F},
    {0.070F, 13.0F, 84.0F, 14.0F, 0.0F, 4.4F, 0.027F, 0.11F},
    {0.061F, 13.0F, 79.0F, 13.0F, 0.0F, 4.2F, 0.031F, 0.12F},
    {0.037F, 12.0F, 83.0F, 16.0F, 0.0F, 4.2F, 0.026F, 0.12F},
    {0.036F, 5.7F, 83.0F, 11.0F, 0.0F, 3.7F, 0.025F, 0.22F},
    {0.048F, 3.0F, 82.0F, 9.3F, 0.0F, 3.6F, 0.024F, 0.27F},
    {0.018F, 1.9F, 88.0F, 17.0F, 0.0F, 4.0F, 0.028F, 0.15F},
    {0.0077F, 19.0F, 86.0F, 9.3F, 0.0F, 3.2F, 0.023F, 0.21F},
    {0.0084F, 39.0F, 84.0F, 7.3F, 0.0F, 2.9F, 0.20F, 0.30F},
    {0.0059F, 3.5F, 88.0F, 9.5F, 0.0F, 3.1F, 0.019F, 0.21F},
    {0.0034F, 3.0F, 86.0F, 7.2F, 0.0F, 2.7F, 0.019F, 0.26F},
    {0.0035F, 65.0F, 88.0F, 4.9F, 0.0F, 2.3F, 0.016F, 0.28F}
}};

// Table 4: 16O Target parameters
inline constexpr std::array<Fred2DParams, 18> kFredParamsO = {{
    {0.30F, 130.0F, 93.0F, 37.0F, 0.0F, 10.0F, 0.013F, 0.024F},
    {0.30F, 130.0F, 93.0F, 37.0F, 0.0F, 10.0F, 0.013F, 0.024F},
    {0.30F, 63.0F, 82.0F, 26.0F, 0.0F, 9.2F, 0.026F, 0.030F},
    {0.26F, 28.0F, 73.0F, 18.0F, 0.0F, 7.9F, 0.032F, 0.056F},
    {0.15F, 42.0F, 91.0F, 29.0F, 0.0F, 7.3F, 0.029F, 0.043F},
    {0.083F, 210.0F, 83.0F, 15.0F, 0.0F, 5.3F, 0.029F, 0.078F},
    {0.081F, 11.0F, 79.0F, 17.0F, 0.0F, 5.5F, 0.029F, 0.14F},
    {0.079F, 15.0F, 84.0F, 14.0F, 0.0F, 4.5F, 0.028F, 0.10F},
    {0.063F, 14.0F, 79.0F, 13.0F, 0.0F, 4.4F, 0.031F, 0.11F},
    {0.039F, 13.0F, 83.0F, 16.0F, 0.0F, 4.3F, 0.027F, 0.11F},
    {0.031F, 6.0F, 83.0F, 12.0F, 0.0F, 3.9F, 0.027F, 0.18F},
    {0.055F, 3.2F, 82.0F, 8.8F, 0.0F, 3.7F, 0.024F, 0.24F},
    {0.022F, 2.1F, 89.0F, 17.0F, 0.0F, 4.1F, 0.029F, 0.14F},
    {0.0077F, 19.0F, 84.0F, 9.3F, 0.0F, 3.3F, 0.026F, 0.17F},
    {0.0080F, 31.0F, 85.0F, 7.1F, 0.0F, 2.9F, 0.021F, 0.25F},
    {0.0088F, 3.5F, 88.0F, 9.2F, 0.0F, 3.2F, 0.023F, 0.20F},
    {0.0034F, 30.0F, 86.0F, 7.1F, 0.0F, 2.8F, 0.020F, 0.23F},
    {0.0034F, 62.0F, 87.0F, 4.7F, 0.0F, 2.4F, 0.017F, 0.30F}
}};

// Map (Z, A) to charged species index 0..16 for stopping power table lookup
inline int get_charged_species_idx(int z, int a) noexcept {
    if (z == 1) {
        if (a == 1) return 0;  // 1H (p)
        if (a == 2) return 1;  // 2H (d)
        if (a == 3) return 2;  // 3H (t)
        return 0;
    }
    if (z == 2) {
        if (a == 3) return 3;  // 3He
        if (a == 4) return 4;  // 4He (alpha)
        if (a == 6) return 5;  // 6He
        return 4;
    }
    if (z == 3) {
        if (a == 6) return 6;  // 6Li
        if (a == 7) return 7;  // 7Li
        return 7;
    }
    if (z == 4) {
        if (a == 7) return 8;  // 7Be
        if (a == 9) return 9;  // 9Be
        if (a == 10) return 10; // 10Be
        return 9;
    }
    if (z == 5) {
        if (a == 8) return 11;  // 8B
        if (a == 10) return 12; // 10B
        if (a == 11) return 13; // 11B
        return 13;
    }
    if (z == 6) {
        if (a == 10) return 14; // 10C
        if (a == 11) return 15; // 11C
        if (a == 12) return 16; // 12C
        return 16;
    }
    return 0;
}

// 1. ICRU Data Fit for sigma_H(E) (Figure 2 in PMC8990885)
inline float calculate_icru_sigma_H(float e_per_u) noexcept {
    if (e_per_u <= 0.0F) return 250.0F;
    if (e_per_u >= 250.0F) return 250.0F;
    return 250.0F * (1.0F + 0.80F * std::exp(-e_per_u / 60.0F));
}

// 2. Kox Reaction Cross Section for 12C + 16O (Kox et al., Phys. Rev. C 35, 1678)
inline float calculate_kox_sigma_O(float e_per_u, float incident_energy_MeV) noexcept {
    if (e_per_u <= 0.0F || incident_energy_MeV <= 0.0F) return 0.0F;
    constexpr float a_kox = 1.85F;
    constexpr float ap_c12 = 12.0F;
    constexpr float at_o16 = 16.0F;
    constexpr float zp_c12 = 6.0F;
    constexpr float zt_o16 = 8.0F;
    constexpr float ap13 = 2.289428F; // 12^(1/3)
    constexpr float at13 = 2.519842F; // 16^(1/3)
    const float c_e = 0.80F * (1.0F - std::exp(-e_per_u / 40.0F));
    const float r_vol = ap13 + at13 + a_kox * (ap13 * at13) / (ap13 + at13) - c_e;
    const float b_c = (1.44F * zp_c12 * zt_o16) / (1.3F * (ap13 + at13));
    const float e_cm = (at_o16 / (ap_c12 + at_o16)) * incident_energy_MeV;
    const float coulomb_barrier_factor = (e_cm > b_c) ? (1.0F - b_c / e_cm) : 0.0F;
    // Conversion fm^2 to mb: 1 fm^2 = 10 mb, pi * r0^2 * 10 = 38.013 mb (r0 = 1.1 fm)
    return 38.013F * r_vol * r_vol * coulomb_barrier_factor;
}

inline float calculate_target_prob_H(float sigma_H, float sigma_O) noexcept {
    const float denom = 2.0F * sigma_H + ((sigma_O > 1.0F) ? sigma_O : 1.0F);
    return (2.0F * sigma_H) / denom;
}

// Sample isotope from Table 1 conditional on remaining protons (z_rem) and neutrons (n_rem = a_rem - z_rem)
inline int sample_table1_isotope(const float* prob_table, int a_rem, int z_rem, float u) noexcept {
    const int n_rem = a_rem - z_rem;
    float total_prob = 0.0F;
    for (int i = 0; i < 18; ++i) {
        const int zi = kFredIsotopes[i].z;
        const int ni = kFredIsotopes[i].a - zi;
        if (zi <= z_rem && ni <= n_rem && kFredIsotopes[i].a > 0) {
            total_prob += prob_table[i];
        }
    }
    if (total_prob <= 0.0F) {
        if (z_rem > 0 && n_rem > 0) return 2; // 2H (d: Z=1, N=1)
        if (z_rem > 0) return 1;              // 1H (p: Z=1, N=0)
        return 0;                             // n  (n: Z=0, N=1)
    }
    const float target_p = u * total_prob;
    float cum_p = 0.0F;
    for (int i = 0; i < 18; ++i) {
        const int zi = kFredIsotopes[i].z;
        const int ni = kFredIsotopes[i].a - zi;
        if (zi <= z_rem && ni <= n_rem && kFredIsotopes[i].a > 0) {
            cum_p += prob_table[i];
            if (target_p <= cum_p) {
                return i;
            }
        }
    }
    if (z_rem > 0 && n_rem > 0) return 2;
    if (z_rem > 0) return 1;
    return 0;
}

// Paper Eq. 13–16 with current fragment in R (implicit E_i). First fragment (T=M=0) gives E/A = P, not 0.6 P.
inline float sample_projectile_fragment_Eu(float e95, float projectile_Eu, int fragment_A,
                                           float previous_total_energy,
                                           float previous_total_A) noexcept {
    constexpr float c = 0.40F;
    if (!(projectile_Eu > 0.0F) || fragment_A <= 0 || !(e95 > 0.0F)) {
        return -1.0F;
    }
    const float x = e95 * (projectile_Eu / 95.0F);
    const float total_A = previous_total_A + static_cast<float>(fragment_A);
    const float denom =
        1.0F - c * x * static_cast<float>(fragment_A) / (total_A * projectile_Eu);
    if (!(denom > 1.0e-6F)) {
        return -1.0F;
    }
    return x *
           ((1.0F - c) + c * previous_total_energy / (total_A * projectile_Eu)) / denom;
}

// Paper Eq. 12: H isotopes mix; other projectile Gaussian; other target exponential.
inline int nearest_fred_isotope(int z, int a) noexcept {
    int best = (z > 0) ? 1 : 0;
    int best_d = 1000;
    for (int i = 0; i < 18; ++i) {
        const int dz = kFredIsotopes[i].z - z;
        const int da = kFredIsotopes[i].a - a;
        const int d = (dz < 0 ? -dz : dz) + (da < 0 ? -da : da);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

// Nearest Table-1 charged isotope that does not overshoot leftover (A,Z).
inline int nearest_fred_isotope_fitting(int z, int a) noexcept {
    int best = -1;
    int best_d = 1000;
    for (int i = 0; i < 18; ++i) {
        const int iz = kFredIsotopes[i].z;
        const int ia = kFredIsotopes[i].a;
        if (iz < 1 || ia < 1 || iz > z || ia > a) {
            continue;
        }
        const int dz = iz - z;
        const int da = ia - a;
        const int d = (dz < 0 ? -dz : dz) + (da < 0 ? -da : da);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

// Sequential Table 1 without n/p evaporation dump. Leftover (A,Z) with Z>=1 is one remnant.
// Returns fragment count; leftover neutrons encoded as out_neutron_a (mass only).
inline int fill_projectile_table1_fragments(const float* prob, const float* uniforms, int n_u,
                                            uint8_t* idx_out, int max_out,
                                            int* leftover_n) noexcept {
    int a_rem = 12;
    int z_rem = 6;
    int nfrag = 0;
    const int cap = (max_out > 1) ? max_out - 1 : max_out;
    for (int k = 0; k < n_u && (a_rem > 0 || z_rem > 0) && nfrag < cap; ++k) {
        const int iso_idx = sample_table1_isotope(prob, a_rem, z_rem, uniforms[k]);
        const auto iso = kFredIsotopes[static_cast<std::size_t>(iso_idx)];
        if (iso.a <= a_rem && iso.z <= z_rem && iso.a > 0) {
            idx_out[nfrag++] = static_cast<uint8_t>(iso_idx);
            a_rem -= iso.a;
            z_rem -= iso.z;
        } else {
            break;
        }
    }
    if (z_rem > 0 && a_rem > 0 && nfrag < max_out) {
        const int ridx = nearest_fred_isotope_fitting(z_rem, a_rem);
        if (ridx >= 0) {
            const auto r = kFredIsotopes[static_cast<std::size_t>(ridx)];
            idx_out[nfrag++] = static_cast<uint8_t>(ridx);
            a_rem -= r.a;
            z_rem -= r.z;
        }
    }
    if (leftover_n != nullptr) {
        *leftover_n = (z_rem == 0 && a_rem > 0) ? a_rem : 0;
    }
    return nfrag;
}

// Inclusive production knots. 95 MeV/u is GANIL/FRED Table 1. 200/300/400 MeV/u
// keep intra-Z isotope ratios from Table 1 and rescale Z-groups toward Toshito
// et al. PRC 75, 054606 (2007) water partial charge-changing (ΔZ=1,2,3 nearly
// flat 200–400 MeV/u): more n/p/He/Li, fewer projectile-like C/B on H.
inline constexpr int kFredYieldNKnots = 4;
inline constexpr std::array<float, 4> kFredYieldKnotMeVu = {95.0F, 200.0F, 300.0F, 400.0F};

inline constexpr std::array<std::array<float, 18>, 4> kFredProbHByE = {{
    {14.0F, 38.0F, 5.0F, 1.3F, 3.0F, 26.0F, 0.036F, 2.3F, 0.93F, 1.6F, 0.25F, 0.0001F, 0.15F, 1.3F, 2.1F, 0.19F, 3.9F, 0.59F},
    {14.7F, 39.52F, 5.2F, 1.352F, 3.24F, 28.08F, 0.03888F, 2.438F, 0.9858F, 1.68F, 0.2625F, 0.000105F, 0.1425F, 1.235F, 1.995F, 0.1748F, 3.588F, 0.5428F},
    {15.12F, 40.66F, 5.35F, 1.391F, 3.36F, 29.12F, 0.04032F, 2.53F, 1.023F, 1.728F, 0.27F, 0.000108F, 0.135F, 1.17F, 1.89F, 0.1615F, 3.315F, 0.5015F},
    {15.4F, 41.42F, 5.45F, 1.417F, 3.45F, 29.9F, 0.0414F, 2.576F, 1.0416F, 1.76F, 0.275F, 0.00011F, 0.129F, 1.118F, 1.806F, 0.152F, 3.12F, 0.472F},
}};

inline constexpr std::array<std::array<float, 18>, 4> kFredProbCByE = {{
    {65.0F, 10.0F, 7.5F, 6.1F, 1.2F, 6.4F, 1.7F, 0.25F, 0.41F, 0.083F, 0.11F, 0.24F, 0.013F, 0.089F, 0.20F, 0.017F, 0.055F, 0.043F},
    {69.55F, 10.8F, 8.1F, 6.588F, 1.344F, 7.168F, 1.904F, 0.285F, 0.4674F, 0.08964F, 0.1188F, 0.2592F, 0.013F, 0.089F, 0.20F, 0.01547F, 0.05005F, 0.03913F},
    {71.5F, 11.1F, 8.325F, 6.771F, 1.416F, 7.552F, 2.006F, 0.3F, 0.492F, 0.09296F, 0.1232F, 0.2688F, 0.01248F, 0.08544F, 0.192F, 0.01428F, 0.0462F, 0.03612F},
    {73.45F, 11.3F, 8.475F, 6.893F, 1.464F, 7.808F, 2.074F, 0.31F, 0.5084F, 0.09462F, 0.1254F, 0.2736F, 0.01209F, 0.08277F, 0.186F, 0.01326F, 0.0429F, 0.03354F},
}};

inline constexpr std::array<std::array<float, 18>, 4> kFredProbOByE = {{
    {60.0F, 16.0F, 8.8F, 5.1F, 1.7F, 6.3F, 1.0F, 0.28F, 0.39F, 0.12F, 0.079F, 0.10F, 0.014F, 0.086F, 0.18F, 0.016F, 0.071F, 0.079F},
    {64.8F, 17.6F, 9.68F, 5.61F, 1.955F, 7.245F, 1.15F, 0.336F, 0.468F, 0.132F, 0.0869F, 0.11F, 0.0147F, 0.0903F, 0.189F, 0.0144F, 0.0639F, 0.0711F},
    {67.2F, 18.24F, 10.032F, 5.814F, 2.074F, 7.686F, 1.22F, 0.3584F, 0.4992F, 0.138F, 0.09085F, 0.115F, 0.01428F, 0.08772F, 0.1836F, 0.01312F, 0.05822F, 0.06478F},
    {69.0F, 18.56F, 10.208F, 5.916F, 2.176F, 8.064F, 1.28F, 0.3696F, 0.5148F, 0.1416F, 0.09322F, 0.118F, 0.014F, 0.086F, 0.18F, 0.012F, 0.05325F, 0.05925F},
}};

// Identify H/C/O by Table-1 neutron % so GPU copies of the 95 MeV/u table match.
inline const std::array<std::array<float, 18>, 4>* fred_yield_family(const float* table95) noexcept {
    const float n0 = table95[0];
    if (n0 > 62.0F) {
        return &kFredProbCByE;
    }
    if (n0 > 50.0F) {
        return &kFredProbOByE;
    }
    return &kFredProbHByE;
}

inline void interpolate_fred_yield_table(float e_per_u,
                                         const std::array<std::array<float, 18>, 4>& family,
                                         float* out18) noexcept {
    const float e = (e_per_u > 0.0F) ? e_per_u : kFredYieldKnotMeVu[0];
    int ihi = 1;
    while (ihi < kFredYieldNKnots - 1 && e > kFredYieldKnotMeVu[static_cast<std::size_t>(ihi)]) {
        ++ihi;
    }
    const int ilo = ihi - 1;
    const float e0 = kFredYieldKnotMeVu[static_cast<std::size_t>(ilo)];
    const float e1 = kFredYieldKnotMeVu[static_cast<std::size_t>(ihi)];
    float t = (e - e0) / (e1 - e0);
    if (t < 0.0F) {
        t = 0.0F;
    }
    if (t > 1.0F) {
        t = 1.0F;
    }
    const auto& a = family[static_cast<std::size_t>(ilo)];
    const auto& b = family[static_cast<std::size_t>(ihi)];
    for (int i = 0; i < 18; ++i) {
        const float w = a[static_cast<std::size_t>(i)] * (1.0F - t) +
                        b[static_cast<std::size_t>(i)] * t;
        out18[i] = (w > 0.0F && w < 1.0e30F) ? w : 0.0F;
    }
}

inline void fill_energy_dependent_inclusive_weights(float e_per_u, const float* table95,
                                                    float* out18) noexcept {
    (void)e_per_u;
    for (int i = 0; i < 18; ++i) {
        out18[i] = table95[i];
    }
}

inline float inclusive_element_weight(const float* w, int z) noexcept {
    float s = 0.0F;
    for (int i = 0; i < 18; ++i) {
        if (kFredIsotopes[static_cast<std::size_t>(i)].z == z) {
            s += w[i];
        }
    }
    return s;
}

// Sequential Table-1 approximation with exact projectile A/Z closure.
// Returns -1 when the sampled partial channel cannot be completed.
inline int fill_projectile_constrained_channel(
    const float* prob, const float* uniforms, int n_uniforms,
    uint8_t* idx_out, int max_out, int* leftover_n) noexcept {
    int a_rem = 12;
    int z_rem = 6;
    int neutrons = 0;
    int nfrag = 0;
    for (int k = 0; k < n_uniforms && a_rem > 0 && z_rem > 0; ++k) {
        const int idx = sample_table1_isotope(prob, a_rem, z_rem, uniforms[k]);
        const auto iso = kFredIsotopes[static_cast<std::size_t>(idx)];
        const int next_a = a_rem - iso.a;
        const int next_z = z_rem - iso.z;
        if (iso.a <= 0 || next_a < 0 || next_z < 0 || next_a < next_z) {
            continue;
        }
        if (iso.z == 0) {
            neutrons += iso.a;
        } else {
            if (nfrag >= max_out) {
                return -1;
            }
            idx_out[nfrag++] = static_cast<uint8_t>(idx);
        }
        a_rem = next_a;
        z_rem = next_z;
    }
    if (z_rem > 0) {
        int exact_idx = -1;
        for (int i = 1; i < 18; ++i) {
            if (kFredIsotopes[static_cast<std::size_t>(i)].a == a_rem &&
                kFredIsotopes[static_cast<std::size_t>(i)].z == z_rem) {
                exact_idx = i;
                break;
            }
        }
        if (exact_idx >= 0) {
            if (nfrag >= max_out) {
                return -1;
            }
            idx_out[nfrag++] = static_cast<uint8_t>(exact_idx);
            a_rem = 0;
            z_rem = 0;
        } else {
            if (a_rem < z_rem || nfrag + z_rem > max_out) {
                return -1;
            }
            for (int i = 0; i < z_rem; ++i) {
                idx_out[nfrag++] = 1;
            }
            a_rem -= z_rem;
            z_rem = 0;
        }
    }
    neutrons += a_rem;
    if (z_rem != 0) {
        return -1;
    }
    if (leftover_n != nullptr) {
        *leftover_n = neutrons;
    }
    return nfrag;
}

// One leading Table-1 fragment plus at most one A/Z-fitting remnant. No n/p dump.
inline int fill_projectile_joint_channel(const float* prob, float u_lead, uint8_t* idx_out,
                                         int max_out, int* leftover_n) noexcept {
    int a_rem = 12;
    int z_rem = 6;
    int nfrag = 0;
    int leftover = 0;
    const int lead = sample_table1_isotope(prob, a_rem, z_rem, u_lead);
    const auto iso = kFredIsotopes[static_cast<std::size_t>(lead)];
    if (iso.z == 0 && iso.a > 0 && iso.a <= a_rem) {
        leftover += iso.a;
        a_rem -= iso.a;
    } else if (iso.a > 0 && iso.a <= a_rem && iso.z <= z_rem && nfrag < max_out) {
        idx_out[nfrag++] = static_cast<uint8_t>(lead);
        a_rem -= iso.a;
        z_rem -= iso.z;
    }
    if (z_rem > 0 && a_rem > 0 && nfrag < max_out) {
        const int ridx = nearest_fred_isotope_fitting(z_rem, a_rem);
        if (ridx >= 0) {
            const auto r = kFredIsotopes[static_cast<std::size_t>(ridx)];
            idx_out[nfrag++] = static_cast<uint8_t>(ridx);
            a_rem -= r.a;
            z_rem -= r.z;
        }
    }
    if (z_rem == 0 && a_rem > 0) {
        leftover += a_rem;
        a_rem = 0;
    }
    if (nfrag == 0 && max_out > 0) {
        idx_out[nfrag++] = 17;
        leftover = 0;
    }
    if (leftover_n != nullptr) {
        *leftover_n = leftover;
    }
    return nfrag;
}

// Exponential free path. If u < 1-exp(-Sigma L) the collision sits inside the step.
// Paper mode has no fixed local neutron kerma at the production vertex.
inline constexpr float kInelasticNeutronKermaFraction = 0.0F;

inline float inelastic_neutron_kerma_MeV(float neutron_ke_MeV) noexcept {
    (void)neutron_ke_MeV;
    return 0.0F;
}

inline bool inelastic_collision_in_step(float macro_xs_per_mm, float step_mm, float u,
                                        float* collision_s_mm) noexcept {
    if (collision_s_mm == nullptr) {
        return false;
    }
    if (!(macro_xs_per_mm > 0.0F) || !(step_mm > 0.0F)) {
        *collision_s_mm = step_mm;
        return false;
    }
    const float p = 1.0F - std::exp(-macro_xs_per_mm * step_mm);
    if (!(u < p)) {
        *collision_s_mm = step_mm;
        return false;
    }
    const float s = -std::log(std::fmax(1.0F - u, 1.0e-12F)) / macro_xs_per_mm;
    *collision_s_mm = std::fmin(std::fmax(s, 1.0e-5F), step_mm);
    return true;
}

inline bool eq12_sample_gaussian(int z, int a, bool is_target, float u_mix,
                                 float p_gauss) noexcept {
    const bool hydrogen = (z == 1 && a >= 1 && a <= 3);
    if (hydrogen) {
        return u_mix < p_gauss;
    }
    if (!is_target && z >= 1) {
        return true;
    }
    return false;
}

} // namespace carbon


