#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace carbon {

constexpr double nucleon_rest_mass_MeV = 931.49410242;
constexpr double water_radiation_length_g_per_cm2 = 36.08;
constexpr double highland_energy_constant_MeV = 13.6;
constexpr double fred_2gr_max_energy_MeVu = 236.0;

template <typename Scalar>
inline Scalar beta_momentum_per_u_MeV(const Scalar energy_MeVu) noexcept {
    const auto mass = static_cast<Scalar>(nucleon_rest_mass_MeV);
    if (!(energy_MeVu > Scalar{0})) return Scalar{0};
    return energy_MeVu * (energy_MeVu + Scalar{2} * mass) /
           (energy_MeVu + mass);
}

template <typename Scalar>
inline Scalar fred_2gr_high_energy_angle_scale(const Scalar energy_MeVu) noexcept {
    if (energy_MeVu <= static_cast<Scalar>(fred_2gr_max_energy_MeVu)) return Scalar{1};
    const auto reference = beta_momentum_per_u_MeV(
        static_cast<Scalar>(fred_2gr_max_energy_MeVu));
    const auto current = beta_momentum_per_u_MeV(energy_MeVu);
    return current > Scalar{0} ? reference / current : Scalar{0};
}

struct Fred2GrMcsTable {
    static constexpr std::size_t energy_bins = 48;
    static constexpr std::size_t thickness_bins = 51;
    static constexpr std::size_t parameter_count = 6;
    std::vector<float> values;
    static Fred2GrMcsTable from_binary(const std::filesystem::path& path);
};

template <typename Scalar>
inline Scalar fred_2gr_parameter(const Scalar* table, int parameter,
                                 Scalar energy_MeVu,
                                 Scalar areal_density_g_per_cm2) noexcept {
    if (table == nullptr || parameter < 0 || parameter >= 6 ||
        energy_MeVu < Scalar{1} || energy_MeVu > Scalar{236} ||
        areal_density_g_per_cm2 < Scalar{1.0e-4} ||
        areal_density_g_per_cm2 > Scalar{10}) return Scalar{-1};
    Scalar x = (energy_MeVu - Scalar{1}) / Scalar{5};
    Scalar y = (static_cast<Scalar>(std::log10(
                    static_cast<double>(areal_density_g_per_cm2))) + Scalar{4}) /
               Scalar{0.1};
    int ix = std::max(0, std::min(46, static_cast<int>(std::floor(x))));
    int iy = std::max(0, std::min(49, static_cast<int>(std::floor(y))));
    const Scalar fx = x - static_cast<Scalar>(ix);
    const Scalar fy = y - static_cast<Scalar>(iy);
    const auto at = [&](int yy, int xx) {
        return static_cast<Scalar>(table[(parameter * 51 + yy) * 48 + xx]);
    };
    const auto a = at(iy, ix) + fx * (at(iy, ix + 1) - at(iy, ix));
    const auto b = at(iy + 1, ix) + fx * (at(iy + 1, ix + 1) - at(iy + 1, ix));
    return a + fy * (b - a);
}

// Geant4 material mass radiation lengths (g/cm2), measured with the
// CarbonMaterialPropertiesNtuple TOPAS extension.  Keep these in mass units:
// the local CT density below converts the transported step to areal density.
constexpr double ct_material_radiation_length_g_per_cm2(const unsigned material_class) {
    switch (material_class) {
        case 0U: return 36.6161;  // G4_AIR
        case 1U: return 36.4162;  // G4_LUNG_ICRP
        case 3U: return 30.4866;  // G4_BONE_COMPACT_ICRU
        default: return 36.0830;  // G4_WATER / soft tissue class
    }
}

inline double highland_projected_rms_angle_material_rad(
    double kinetic_energy_MeV,
    int atomic_number,
    int mass_number,
    double path_length_mm,
    double density_g_per_cm3,
    double radiation_length_g_per_cm2) noexcept {
    if (kinetic_energy_MeV <= 0.0 || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0 || density_g_per_cm3 <= 0.0 ||
        radiation_length_g_per_cm2 <= 0.0) {
        return 0.0;
    }

    const auto energy_MeV_per_u = kinetic_energy_MeV / static_cast<double>(mass_number);
    const auto total_energy_MeV_per_u = energy_MeV_per_u + nucleon_rest_mass_MeV;
    const auto momentum_MeV_per_c_per_u =
        std::sqrt(energy_MeV_per_u *
                  (energy_MeV_per_u + 2.0 * nucleon_rest_mass_MeV));
    const auto beta = momentum_MeV_per_c_per_u / total_energy_MeV_per_u;
    const auto momentum_MeV_per_c =
        static_cast<double>(mass_number) * momentum_MeV_per_c_per_u;
    const auto radiation_lengths =
        density_g_per_cm3 * (path_length_mm / 10.0) / radiation_length_g_per_cm2;
    if (beta <= 0.0 || momentum_MeV_per_c <= 0.0 || radiation_lengths <= 0.0) {
        return 0.0;
    }

    const auto charge = static_cast<double>(atomic_number);
    const auto logarithm_argument =
        radiation_lengths * charge * charge / (beta * beta);
    const auto correction =
        std::max(0.0, 1.0 + 0.038 * std::log(logarithm_argument));
    return highland_energy_constant_MeV * charge /
           (beta * momentum_MeV_per_c) * std::sqrt(radiation_lengths) * correction;
}

inline double highland_projected_rms_angle_rad(double kinetic_energy_MeV,
                                                int atomic_number,
                                                int mass_number,
                                                double path_length_mm,
                                                double density_g_per_cm3) noexcept {
    return highland_projected_rms_angle_material_rad(
        kinetic_energy_MeV, atomic_number, mass_number, path_length_mm,
        density_g_per_cm3, water_radiation_length_g_per_cm2);
}

}  // namespace carbon
