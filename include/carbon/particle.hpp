#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

namespace carbon {

inline constexpr double ion_nucleon_rest_mass_MeV = 931.49410242;

enum class PrimaryMassConvention { Kinematic, Nuclear };

// Production kinematic convention, also used by primary CINEL03 events.
// Preserve the calibrated C12 A*amu convention separately from nuclear masses.
inline double primary_kinematic_mass_MeV(int z, int a) noexcept {
    if (z == 1 && a == 1) return 938.27208816;
    if (z == 6 && a == 12) return 12.0 * ion_nucleon_rest_mass_MeV;
    return std::numeric_limits<double>::quiet_NaN();
}

// Nuclear mass contract for current Urban, continuous26 EM and elastic banks.
inline double primary_nuclear_mass_MeV(int z, int a) noexcept {
    if (z == 1 && a == 1) return 938.27208816;
    if (z == 6 && a == 12) return 11174.86338798439;
    return std::numeric_limits<double>::quiet_NaN();
}

inline std::string primary_component_label(int z, int a) {
    if (z == 6 && a == 12) return "primary_c12";
    if (z == 1 && a == 1) return "primary_proton";
    return "primary_z" + std::to_string(z) + "_a" + std::to_string(a);
}

inline constexpr std::size_t primary_energy_band(float kinetic_MeV, int a) noexcept {
    if (a <= 0) return 3; // invalid/sentinel, never divide by zero
    const float energy_per_u = kinetic_MeV / static_cast<float>(a);
    return energy_per_u < 50.0F ? 0U : (energy_per_u <= 300.0F ? 1U : 2U);
}

struct PrimaryIonDefinition {
    int atomic_number{6};
    int mass_number{12};
    double rest_mass_MeV{12.0 * ion_nucleon_rest_mass_MeV};
    double inverse_mass_number{1.0 / 12.0};
    double charge_power{0.3028534321386899};
};

inline PrimaryIonDefinition make_primary_ion_definition(
    const int atomic_number, const int mass_number) {
    const double mass = primary_kinematic_mass_MeV(atomic_number, mass_number);
    if (!(mass > 0.0) || !std::isfinite(mass))
        throw std::invalid_argument("Production primary must be proton (Z=1,A=1) or C12 (Z=6,A=12)");
    return PrimaryIonDefinition{atomic_number, mass_number, mass,
        1.0 / static_cast<double>(mass_number),
        std::pow(static_cast<double>(atomic_number), -2.0 / 3.0)};
}

inline constexpr std::size_t charged_origin_category_count = 8;
inline constexpr std::size_t primary_charged_origin_category = 0;
inline constexpr std::size_t be_isotope_origin_category_count = 4;
inline constexpr std::size_t he_isotope_origin_category_count = 3;

inline constexpr std::uint8_t minibeam_birth_region_copper = 1;
inline constexpr std::uint8_t minibeam_birth_region_water = 2;
inline constexpr std::size_t minibeam_component_species_count = 8;
inline constexpr std::size_t minibeam_component_category_count =
    1 + 2 * minibeam_component_species_count;
inline constexpr std::size_t minibeam_primary_component_category = 0;
inline constexpr std::size_t minibeam_energy_band_species_count = 4;
inline constexpr std::size_t minibeam_energy_band_count = 3;
inline constexpr std::size_t minibeam_fixed_region_count = 3;
inline constexpr std::size_t minibeam_primary_roi_kind_count = 6;
inline constexpr std::size_t minibeam_primary_roi_local_total_deposit = 0;
inline constexpr std::size_t minibeam_primary_roi_continuous_sampled = 1;
inline constexpr std::size_t minibeam_primary_roi_delta_sampled = 2;
inline constexpr std::size_t minibeam_primary_roi_continuous_after_scale = 3;
inline constexpr std::size_t minibeam_primary_roi_delta_after_scale = 4;
inline constexpr std::size_t minibeam_primary_roi_fluence = 5;
inline constexpr std::size_t minibeam_spatial_audit_slot_count = 8;

// Diagnostic order: p, d, t, He-4. The sentinel count means unsupported.
constexpr std::size_t minibeam_energy_band_species_category(
    const int z, const int a) noexcept {
    if (z == 1 && a == 1) return 0;
    if (z == 1 && a == 2) return 1;
    if (z == 1 && a == 3) return 2;
    if (z == 2 && a == 4) return 3;
    return minibeam_energy_band_species_count;
}

// Secondary component order: C12, p, d, t, He3, He4, heavier (Z>=3),
// and all remaining charged ions. Category zero is reserved for the primary;
// Copper-born and water-born copies of these eight classes follow.
constexpr std::size_t minibeam_component_species_category(
    const int z, const int a) noexcept {
    if (z == 6 && a == 12) return 0;
    if (z == 1 && a == 1) return 1;
    if (z == 1 && a == 2) return 2;
    if (z == 1 && a == 3) return 3;
    if (z == 2 && a == 3) return 4;
    if (z == 2 && a == 4) return 5;
    if (z >= 3) return 6;
    return 7;
}

constexpr std::size_t minibeam_component_category(
    const int z, const int a, const std::uint8_t birth_region) noexcept {
    const auto source_offset = birth_region == minibeam_birth_region_copper
        ? 0U : minibeam_component_species_count;
    return 1U + source_offset + minibeam_component_species_category(z, a);
}

constexpr std::size_t he_isotope_origin_category(const int z, const int a) noexcept {
    if (z != 2) return he_isotope_origin_category_count;
    return a == 3 ? 0 : (a == 4 ? 1 : 2);
}

constexpr std::size_t be_isotope_origin_category(const int atomic_number,
                                                 const int mass_number) noexcept {
    if (atomic_number != 4) return be_isotope_origin_category_count;
    if (mass_number == 6) return 0;
    if (mass_number == 7) return 1;
    if (mass_number == 9) return 2;
    if (mass_number == 10) return 3;
    return be_isotope_origin_category_count;
}
inline constexpr std::size_t light_isotope_category_count = 8;

// Optional LET diagnostics: p, d, t, He-3, He-4, N, O, F.  The three target
// recoil elements expose the dominant Z > 6 contribution in CT without
// adding atomics to normal production runs. Other ions return the sentinel.
constexpr std::size_t light_isotope_category(const int atomic_number,
                                             const int mass_number) noexcept {
    if (atomic_number == 1 && mass_number >= 1 && mass_number <= 3) {
        return static_cast<std::size_t>(mass_number - 1);
    }
    if (atomic_number == 2 && (mass_number == 3 || mass_number == 4)) {
        return static_cast<std::size_t>(mass_number);
    }
    if (atomic_number >= 7 && atomic_number <= 9) {
        return static_cast<std::size_t>(atomic_number - 2);
    }
    return light_isotope_category_count;
}

// Fragment birth-spectrum diagnostics for the same optional categories.
// Generation bins: 0 = primary ion direct product, 1 = first cascade
// generation, 2 = generation >= 2.
inline constexpr std::size_t birth_generation_bin_count = 3;
inline constexpr std::size_t birth_mevu_bin_count = 200;  // 0–400 MeV/u @ 2 MeV/u
inline constexpr double birth_mevu_bin_width = 2.0;
inline constexpr double birth_mevu_max = 400.0;
inline constexpr std::size_t birth_cos_bin_count = 20;  // cosθ ∈ [-1, 1]
inline constexpr std::size_t birth_parent_mevu_bin_count = 40;  // 0–400 @ 10 MeV/u
inline constexpr double birth_parent_mevu_bin_width = 10.0;
inline constexpr std::size_t birth_parent_z_bin_count = 9;  // index = clamp(Z, 0..8)

constexpr std::size_t birth_generation_bin(const std::uint8_t generation) noexcept {
    return generation >= 2 ? 2 : static_cast<std::size_t>(generation);
}

constexpr std::size_t birth_mevu_bin(const double kinetic_energy_MeV,
                                    const int mass_number) noexcept {
    if (mass_number <= 0 || !(kinetic_energy_MeV > 0.0)) {
        return 0;
    }
    const double mevu = kinetic_energy_MeV / static_cast<double>(mass_number);
    if (!(mevu > 0.0)) {
        return 0;
    }
    auto bin = static_cast<std::size_t>(mevu / birth_mevu_bin_width);
    if (bin >= birth_mevu_bin_count) {
        bin = birth_mevu_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_cos_bin(const double direction_z) noexcept {
    double c = direction_z;
    if (c < -1.0) {
        c = -1.0;
    }
    if (c > 1.0) {
        c = 1.0;
    }
    auto bin = static_cast<std::size_t>(((c + 1.0) * 0.5) * birth_cos_bin_count);
    if (bin >= birth_cos_bin_count) {
        bin = birth_cos_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_parent_mevu_bin(const double parent_kinetic_energy_MeV,
                                           const int parent_mass_number) noexcept {
    if (parent_mass_number <= 0 || !(parent_kinetic_energy_MeV > 0.0)) {
        return 0;
    }
    const double mevu =
        parent_kinetic_energy_MeV / static_cast<double>(parent_mass_number);
    auto bin = static_cast<std::size_t>(mevu / birth_parent_mevu_bin_width);
    if (bin >= birth_parent_mevu_bin_count) {
        bin = birth_parent_mevu_bin_count - 1;
    }
    return bin;
}

constexpr std::size_t birth_parent_z_bin(const int parent_atomic_number) noexcept {
    if (parent_atomic_number <= 0) {
        return 0;
    }
    if (parent_atomic_number >= static_cast<int>(birth_parent_z_bin_count)) {
        return birth_parent_z_bin_count - 1;
    }
    return static_cast<std::size_t>(parent_atomic_number);
}

// Histogram layout helpers: species-major, then generation, then bin.
// index = ((category * birth_generation_bin_count) + gen) * n_bins + bin
constexpr std::size_t birth_hist_index(const std::size_t category,
                                       const std::size_t generation,
                                       const std::size_t bin,
                                       const std::size_t n_bins) noexcept {
    return (category * birth_generation_bin_count + generation) * n_bins + bin;
}

constexpr std::size_t birth_hist_plane_size(const std::size_t n_bins) noexcept {
    return light_isotope_category_count * birth_generation_bin_count * n_bins;
}

// Joint parent MeV/u × product MeV/u histogram (species × generation).
// index = birth_hist_index(cat, gen, 0, parent_bins*product_bins)
//       + parent_bin * product_bins + product_bin
constexpr std::size_t birth_joint_bins() noexcept {
    return birth_parent_mevu_bin_count * birth_mevu_bin_count;
}

constexpr std::size_t birth_joint_index(const std::size_t category,
                                        const std::size_t generation,
                                        const std::size_t parent_bin,
                                        const std::size_t product_bin) noexcept {
    return birth_hist_index(category, generation, 0, birth_joint_bins()) +
           parent_bin * birth_mevu_bin_count + product_bin;
}

constexpr std::size_t birth_joint_plane_size() noexcept {
    return birth_hist_plane_size(birth_joint_bins());
}

constexpr std::uint8_t charged_dose_category(const int atomic_number,
                                             const int mass_number) noexcept {
    (void)mass_number;
    if (atomic_number == 1) {
        return 5;
    }
    if (atomic_number >= 2 && atomic_number <= 6) {
        return static_cast<std::uint8_t>(6 - atomic_number);
    }
    return 6;
}

constexpr std::size_t charged_origin_category_from_fragment(
    const std::uint8_t fragment_category) noexcept {
    return static_cast<std::size_t>(fragment_category) + 1;
}

struct Particle1D {
    double position_mm{0.0};
    double kinetic_energy_MeV{0.0};
    int particle_type{6012};
    bool alive{true};
};

struct SecondaryParticle3D {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{0.0F};
    std::int32_t pdg_id{0};
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    std::uint8_t origin_category{0};
    std::uint8_t generation{0};
    std::uint16_t reserved{0};  // Padding.
    // Deterministic Philox stream id (not the atomic queue slot). Primary children
    // derive from history; cascade children derive from parent stream.
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(SecondaryParticle3D) == 48);

struct ElectronParticle3D {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float kinetic_energy_MeV{0.0F};
    float direction_x{0.0F};
    float direction_y{0.0F};
    float direction_z{0.0F};
    std::int32_t pdg_id{11};  // 11 = e-, -11 = e+
    std::uint8_t origin_category{0};
    std::uint8_t generation{0};
    std::uint16_t reserved{0};
    std::uint64_t rng_stream{0};
};

static_assert(sizeof(ElectronParticle3D) == 48);
inline constexpr float positron_annihilation_reserve_MeV = 1.0219979F;

struct ElectronTransportSummary {
    std::uint32_t step_count{0};
    std::uint32_t is_positron{0};
    std::uint32_t queued_gamma_count{0};
    std::uint32_t gamma_overflow_count{0};
    float deposited_energy_MeV{0.0F};
    float escaped_energy_MeV{0.0F};
    float radiative_energy_MeV{0.0F};
    float annihilation_reserve_MeV{0.0F};
    float queued_brems_gamma_energy_MeV{0.0F};
    float queued_annihilation_gamma_energy_MeV{0.0F};
    float gamma_overflow_energy_MeV{0.0F};
};

static_assert(sizeof(ElectronTransportSummary) == 44);

}  // namespace carbon
