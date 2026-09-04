#pragma once

#include "carbon/transport_profile.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

struct TransportConfig;

inline constexpr std::array<std::string_view, 13> schneider_canonical_element_names = {
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen",
    "Magnesium", "Phosphorus", "Sulfur", "Chlorine",
    "Argon", "Calcium", "Sodium", "Potassium", "Titanium"
};

struct SchneiderElement {
    std::uint8_t z{0};
    double atomic_mass_g_mol{0.0};
    std::string name{};
};

struct SchneiderMaterialSection {
    int hu_min_inclusive{0};
    int hu_max_exclusive{0};
    std::array<double, 13> mass_fraction{};
};

struct SchneiderMaterialTable {
    static constexpr std::size_t element_count = 13;
    static constexpr std::size_t section_count = 25;

    std::array<SchneiderElement, element_count> elements{};
    std::array<SchneiderMaterialSection, section_count> sections{};

    static SchneiderMaterialTable builtin();
    static SchneiderMaterialTable from_topas_file(const std::filesystem::path& path);
    [[nodiscard]] std::uint8_t section_id(int hu) const noexcept;
};

// Patient CT volume stored in conventional medical-image coordinates:
//   x: patient left/right within an axial slice
//   y: patient posterior/anterior within an axial slice
//   z: patient inferior/superior, normal to the axial slices
// Beam direction is independent of these axes. Clinical transport must rotate
// the source, not repack the CT so that a beam happens to point along +z.
// Legacy angle-specific CCTG files may still contain a beam-repacked volume;
// their metadata identifies the axis mapping and they are compatibility input.
// File magic "CCTG" little-endian.
// v1: density float[] + material_id uint8[] (legacy 4-class 0..3)
// v2: + n u32 + mass_sp_factor float[n]  (energy-independent Z/A factor)
// v3: + n u32 + za_rel float[n] + I_eV float[n]
//     SP = SP_water(E) * mass_sp_energy_factor(za,I,E) * density
struct CtGrid {
    static constexpr std::uint32_t magic_value = 0x47544343U;  // 'CCTG'
    static constexpr std::uint32_t version_value = 3U;
    static constexpr std::uint32_t version_v2 = 2U;
    static constexpr std::uint32_t version_legacy = 1U;

    std::uint32_t nx{0};
    std::uint32_t ny{0};
    std::uint32_t nz{0};
    float origin_x_mm{0.0F};
    float origin_y_mm{0.0F};
    float origin_z_mm{0.0F};
    float spacing_x_mm{1.0F};
    float spacing_y_mm{1.0F};
    float spacing_z_mm{1.0F};
    std::vector<float> density_g_per_cm3{};
    // v1: 0=air..3=bone; v2/v3: Schneider section index
    std::vector<std::uint8_t> material_id{};
    // (Z/A)_section / (Z/A)_water  — high-energy limit mass-SP factor
    std::vector<float> mass_sp_za_rel{};
    // Bragg mean excitation energy I [eV] per Schneider section
    std::vector<float> mass_sp_I_eV{};
    // Legacy constant factors (filled for v2 / diagnostics)
    std::vector<float> mass_sp_factor{};
    std::uint32_t file_version{version_value};

    [[nodiscard]] std::size_t number_of_voxels() const noexcept {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz);
    }

    [[nodiscard]] float extent_x_mm() const noexcept {
        return static_cast<float>(nx) * spacing_x_mm;
    }

    [[nodiscard]] float extent_y_mm() const noexcept {
        return static_cast<float>(ny) * spacing_y_mm;
    }

    [[nodiscard]] float extent_z_mm() const noexcept {
        return static_cast<float>(nz) * spacing_z_mm;
    }

    [[nodiscard]] bool has_mass_sp_factors() const noexcept {
        return !mass_sp_za_rel.empty() || !mass_sp_factor.empty();
    }

    [[nodiscard]] bool uses_schneider_mass_sp() const noexcept {
        return file_version != version_legacy && has_mass_sp_factors();
    }

    static CtGrid from_binary(const std::filesystem::path& path);
    static CtGrid from_config(const TransportConfig& config);
    // File: CCTG binary. Directory or .dcm: CT Image series + Schneider HU map.
    static CtGrid load(const std::filesystem::path& path,
                       const std::filesystem::path& schneider_file = {},
                       std::string_view origin_mode = "centered");
    static CtGrid from_dicom_directory(const std::filesystem::path& directory,
                                       const std::filesystem::path& schneider_file = {},
                                       std::string_view origin_mode = "centered");
    void write_binary(const std::filesystem::path& path) const;
};

struct SchneiderHuTable {
    std::vector<int> density_hu_edges;
    std::vector<double> density_offset;
    std::vector<double> density_factor;
    std::vector<double> density_factor_offset;
    std::vector<double> density_correction;
    int density_correction_hu0{-1000};
    std::vector<int> material_hu_edges;
    std::vector<float> za_rel;
    std::vector<float> I_eV;

    static SchneiderHuTable builtin();
    static SchneiderHuTable from_topas_file(const std::filesystem::path& path);
    [[nodiscard]] float density_g_per_cm3(float hu) const noexcept;
    [[nodiscard]] std::uint8_t section_id(float hu) const noexcept;
};

float hu_to_density_g_per_cm3(float hu) noexcept;
std::uint8_t density_to_material_id(float density_g_per_cm3) noexcept;

// Collapse TOPAS/Schneider sections to the legacy material tables used by the
// current CT XS/SP implementation. Schneider bounds are:
//   0 air, 1 lung, 2..8 soft tissue/water-like, 9+ bone-like.
// Section 8 is HU 80..120 in the TOPAS Schneider table and retains a
// soft-tissue composition. Section 9 (HU 120..200) is the first material with
// a substantial calcium fraction and is the bone-family boundary.
// CCTG v1 already stores these four class IDs directly.
inline std::uint8_t ct_material_class(const std::uint8_t material_id,
                                      const bool schneider_section_id) noexcept {
    if (!schneider_section_id) {
        return material_id < 4U ? material_id : 3U;
    }
    if (material_id == 0U) {
        return 0U;
    }
    if (material_id == 1U) {
        return 1U;
    }
    return material_id < 9U ? 2U : 3U;
}

// Reference densities of the Geant4 materials used to generate the optional
// four-class macroscopic cross-section tables. The transport converts those
// tables to a mass cross section before applying the CT voxel density.
inline constexpr float ct_material_reference_density_g_per_cm3(
    const std::uint8_t material_class) noexcept {
    switch (material_class) {
        case 0U:
            return 0.00120479F;  // G4_AIR
        case 1U:
            return 1.04F;        // G4_LUNG_ICRP
        case 3U:
            return 1.85F;        // G4_BONE_COMPACT_ICRU
        default:
            return 1.0F;         // Water_75eV
    }
}

inline std::uint8_t ct_cross_section_material_index(
    const std::uint8_t material_id,
    const bool schneider_section_id,
    const bool use_schneider_cross_sections) noexcept {
    return schneider_section_id && use_schneider_cross_sections
               ? material_id
               : ct_material_class(material_id, schneider_section_id);
}

inline std::size_t ct_linear_index(const std::uint32_t ix,
                                   const std::uint32_t iy,
                                   const std::uint32_t iz,
                                   const std::uint32_t nx,
                                   const std::uint32_t ny) noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(nx) *
               (static_cast<std::size_t>(iy) +
                static_cast<std::size_t>(ny) * static_cast<std::size_t>(iz));
}

inline bool ct_sample(const float x_mm,
                      const float y_mm,
                      const float z_mm,
                      const float origin_x,
                      const float origin_y,
                      const float origin_z,
                      const float spacing_x,
                      const float spacing_y,
                      const float spacing_z,
                      const std::uint32_t nx,
                      const std::uint32_t ny,
                      const std::uint32_t nz,
                      const float* densities,
                      const std::uint8_t* materials,
                      float& density_out,
                      std::uint8_t& material_out,
                      const float dir_x = 0.0F,
                      const float dir_y = 0.0F,
                      const float dir_z = 0.0F) noexcept {
    if (densities == nullptr || nx == 0 || ny == 0 || nz == 0) {
        return false;
    }
    const auto fx = (x_mm - origin_x) / spacing_x;
    const auto fy = (y_mm - origin_y) / spacing_y;
    const auto fz = (z_mm - origin_z) / spacing_z;

    auto resolve_cell = [](float f, float dir) noexcept -> int {
        return dir < 0.0F ? static_cast<int>(std::ceil(f)) - 1 : static_cast<int>(std::floor(f));
    };

    const auto ix_int = resolve_cell(fx, dir_x);
    const auto iy_int = resolve_cell(fy, dir_y);
    const auto iz_int = resolve_cell(fz, dir_z);
    if (ix_int < 0 || iy_int < 0 || iz_int < 0) {
        return false;
    }
    const auto ix = static_cast<std::uint32_t>(ix_int);
    const auto iy = static_cast<std::uint32_t>(iy_int);
    const auto iz = static_cast<std::uint32_t>(iz_int);
    if (ix >= nx || iy >= ny || iz >= nz) {
        return false;
    }
    const auto index = ct_linear_index(ix, iy, iz, nx, ny);
    density_out = densities[index];
    material_out = materials != nullptr ? materials[index] : static_cast<std::uint8_t>(2);
    return true;
}

// Energy-dependent mass-SP factor vs Water_75eV (I_w = 75 eV).
// za_rel = (Z/A)_s / (Z/A)_w. Uses a simplified Bethe stopping-number ratio.
template <typename LogFn>
inline float ct_mass_sp_energy_factor_impl(const float za_rel,
                                          const float I_eV,
                                          const float energy_MeVu,
                                          LogFn&& log_fn) noexcept {
    constexpr float nucleon_mass_MeV = 931.49410242F;
    constexpr float two_me_c2_MeV = 1.0219979F;
    // The reference table is generated from the explicit TOPAS Water_75eV
    // material, not Geant4's stock G4_WATER (78 eV).
    constexpr float I_water_eV = 75.0F;
    const auto e = energy_MeVu > 0.5F ? energy_MeVu : 0.5F;
    const auto gamma = 1.0F + e / nucleon_mass_MeV;
    const auto beta2 = 1.0F - 1.0F / (gamma * gamma);
    const auto beta2_clamped = beta2 > 1.0e-8F ? beta2 : 1.0e-8F;
    const auto bg2 = beta2_clamped * gamma * gamma;
    auto stopping_number = [&](const float Iev) {
        const auto I_MeV = (Iev > 5.0F ? Iev : 5.0F) * 1.0e-6F;
        return log_fn(two_me_c2_MeV * bg2 / I_MeV) - beta2_clamped;
    };
    const auto Lw = stopping_number(I_water_eV);
    const auto Ls = stopping_number(I_eV);
    if (Lw < 0.05F) {
        return za_rel;
    }
    auto ratio = za_rel * (Ls / Lw);
    if (ratio < 0.5F) {
        ratio = 0.5F;
    }
    if (ratio > 1.5F) {
        ratio = 1.5F;
    }
    return ratio;
}

// Host/CPU path (std::log).
inline float ct_mass_sp_energy_factor(const float za_rel,
                                    const float I_eV,
                                    const float energy_MeVu) noexcept {
    return ct_mass_sp_energy_factor_impl(za_rel, I_eV, energy_MeVu,
                                        [](float x) { return std::log(x); });
}

// Mass-SP scaled water table: SP = SP_water * mass_factor * density.
inline float ct_mass_scaled_stopping_power(const float water_sp_MeV_per_mm,
                                           const float density_g_per_cm3,
                                           const float mass_sp_factor) noexcept {
    return water_sp_MeV_per_mm * mass_sp_factor *
           (density_g_per_cm3 > 1.0e-6F ? density_g_per_cm3 : 1.0e-6F);
}

// Apply the material scaling used by charged-secondary transport at every
// energy evaluation (step start and midpoint). Keeping this operation in one
// helper prevents a raw density-1 water value from leaking into the midpoint
// dE calculation in CT. Outside CT the ion table is already an absolute water
// stopping power and must remain unchanged.
inline float secondary_material_stopping_power(
    const float water_sp_MeV_per_mm,
    const bool in_ct,
    const float density_g_per_cm3,
    const bool use_mass_sp_factor,
    const float mass_sp_factor) noexcept {
    if (!in_ct) {
        return water_sp_MeV_per_mm;
    }
    if (use_mass_sp_factor) {
        return ct_mass_scaled_stopping_power(
            water_sp_MeV_per_mm, density_g_per_cm3, mass_sp_factor);
    }
    return water_sp_MeV_per_mm *
           (density_g_per_cm3 > 1.0e-6F ? density_g_per_cm3 : 1.0e-6F);
}

// Row of the mass-SP LUT is either a Schneider section or a log-density bin.
template <typename LogFn>
inline float ct_lookup_mass_sp_factor(const float* lut,
                                      const std::uint32_t n_rows,
                                      const std::size_t table_size,
                                      const bool density_mode,
                                      const float log_rho_min,
                                      const float inv_dlog,
                                      const std::uint32_t section,
                                      const float density,
                                      const std::size_t energy_index,
                                      const float energy_fraction,
                                      LogFn&& log_fn) noexcept {
    if (lut == nullptr || n_rows == 0 || table_size < 2) {
        return 1.0F;
    }
    std::uint32_t row0 = 0;
    std::uint32_t row1 = 0;
    float row_frac = 0.0F;
    if (density_mode && n_rows > 1U) {
        const auto rho = density > 1.0e-6F ? density : 1.0e-6F;
        auto t = (log_fn(rho) - log_rho_min) * inv_dlog;
        if (t < 0.0F) {
            t = 0.0F;
        }
        const auto last = static_cast<float>(n_rows - 1U);
        if (t > last) {
            t = last;
        }
        row0 = static_cast<std::uint32_t>(t);
        if (row0 >= n_rows - 1U) {
            row0 = n_rows - 2U;
            row_frac = 1.0F;
        } else {
            row_frac = t - static_cast<float>(row0);
        }
        row1 = row0 + 1U;
    } else {
        row0 = section < n_rows ? section : (n_rows - 1U);
        row1 = row0;
    }
    const auto lerp_energy = [&](const std::uint32_t row) {
        const auto base = static_cast<std::size_t>(row) * table_size + energy_index;
        return lut[base] + energy_fraction * (lut[base + 1U] - lut[base]);
    };
    const auto a = lerp_energy(row0);
    if (row_frac <= 0.0F || row0 == row1) {
        return a;
    }
    return a + row_frac * (lerp_energy(row1) - a);
}

inline float distance_to_next_ct_face_1d(const float position_mm,
                                        const float origin_mm,
                                        const float spacing_mm,
                                        const float direction) noexcept {
    if (direction > -1.0e-6F && direction < 1.0e-6F) {
        return 1.0e30F;
    }
    const auto f = (position_mm - origin_mm) / spacing_mm;
    const auto truncated = static_cast<int>(f);
    const auto cell =
        (f >= 0.0F || f == static_cast<float>(truncated)) ? truncated : truncated - 1;
    if (direction > 0.0F) {
        auto boundary = origin_mm + static_cast<float>(cell + 1) * spacing_mm;
        auto t = (boundary - position_mm) / direction;
        if (t <= 0.0F) {
            boundary += spacing_mm;
            t = (boundary - position_mm) / direction;
        }
        return t > 0.0F ? t : 1.0e30F;
    }
    auto boundary = origin_mm + static_cast<float>(cell) * spacing_mm;
    auto t = (boundary - position_mm) / direction;
    if (t <= 0.0F) {
        boundary -= spacing_mm;
        t = (boundary - position_mm) / direction;
    }
    return t > 0.0F ? t : 1.0e30F;
}

// Amanatides–Woo style 3D DDA state for CT voxel traversal.
// t_max_* is the parametric distance along the ray to the next face on each axis;
// t_delta_* is the distance between successive faces on that axis.
struct CtDdaState {
    int ix{0};
    int iy{0};
    int iz{0};
    int step_x{0};
    int step_y{0};
    int step_z{0};
    float t_max_x{0.0F};
    float t_max_y{0.0F};
    float t_max_z{0.0F};
    float t_delta_x{0.0F};
    float t_delta_y{0.0F};
    float t_delta_z{0.0F};
};

inline int ct_floor_div(const float value) noexcept {
    const auto truncated = static_cast<int>(value);
    return (value >= 0.0F || value == static_cast<float>(truncated)) ? truncated
                                                                   : truncated - 1;
}

// Initialize DDA at a point known to be inside the CT grid. Returns false if the
// point is outside. When the point lies on a face in the travel direction with
// near-zero remaining distance, the state is advanced into the next cell so
// callers never get a zero-length step (avoids boundary thrash).
inline bool ct_dda_init(const float x_mm,
                        const float y_mm,
                        const float z_mm,
                        const float dx,
                        const float dy,
                        const float dz,
                        const float origin_x,
                        const float origin_y,
                        const float origin_z,
                        const float spacing_x,
                        const float spacing_y,
                        const float spacing_z,
                        const std::uint32_t nx,
                        const std::uint32_t ny,
                        const std::uint32_t nz,
                        CtDdaState& state) noexcept {
    constexpr float huge = 1.0e30F;
    constexpr float dir_eps = 1.0e-6F;
    if (nx == 0 || ny == 0 || nz == 0 || spacing_x <= 0.0F || spacing_y <= 0.0F ||
        spacing_z <= 0.0F) {
        return false;
    }
    const auto fx = (x_mm - origin_x) / spacing_x;
    const auto fy = (y_mm - origin_y) / spacing_y;
    const auto fz = (z_mm - origin_z) / spacing_z;
    if (fx < 0.0F || fy < 0.0F || fz < 0.0F) {
        return false;
    }
    state.ix = ct_floor_div(fx);
    state.iy = ct_floor_div(fy);
    state.iz = ct_floor_div(fz);
    if (state.ix < 0 || state.iy < 0 || state.iz < 0 ||
        state.ix >= static_cast<int>(nx) || state.iy >= static_cast<int>(ny) ||
        state.iz >= static_cast<int>(nz)) {
        return false;
    }

    state.step_x = dx > dir_eps ? 1 : (dx < -dir_eps ? -1 : 0);
    state.step_y = dy > dir_eps ? 1 : (dy < -dir_eps ? -1 : 0);
    state.step_z = dz > dir_eps ? 1 : (dz < -dir_eps ? -1 : 0);

    state.t_delta_x =
        state.step_x != 0 ? spacing_x / (dx > 0.0F ? dx : -dx) : huge;
    state.t_delta_y =
        state.step_y != 0 ? spacing_y / (dy > 0.0F ? dy : -dy) : huge;
    state.t_delta_z =
        state.step_z != 0 ? spacing_z / (dz > 0.0F ? dz : -dz) : huge;

    if (state.step_x > 0) {
        state.t_max_x =
            (origin_x + static_cast<float>(state.ix + 1) * spacing_x - x_mm) / dx;
    } else if (state.step_x < 0) {
        state.t_max_x =
            (origin_x + static_cast<float>(state.ix) * spacing_x - x_mm) / dx;
    } else {
        state.t_max_x = huge;
    }
    if (state.step_y > 0) {
        state.t_max_y =
            (origin_y + static_cast<float>(state.iy + 1) * spacing_y - y_mm) / dy;
    } else if (state.step_y < 0) {
        state.t_max_y =
            (origin_y + static_cast<float>(state.iy) * spacing_y - y_mm) / dy;
    } else {
        state.t_max_y = huge;
    }
    if (state.step_z > 0) {
        state.t_max_z =
            (origin_z + static_cast<float>(state.iz + 1) * spacing_z - z_mm) / dz;
    } else if (state.step_z < 0) {
        state.t_max_z =
            (origin_z + static_cast<float>(state.iz) * spacing_z - z_mm) / dz;
    } else {
        state.t_max_z = huge;
    }

    // Snap off an already-crossed face so the first distance is strictly positive.
    if (state.t_max_x <= 0.0F && state.step_x != 0) {
        state.ix += state.step_x;
        state.t_max_x += state.t_delta_x;
    }
    if (state.t_max_y <= 0.0F && state.step_y != 0) {
        state.iy += state.step_y;
        state.t_max_y += state.t_delta_y;
    }
    if (state.t_max_z <= 0.0F && state.step_z != 0) {
        state.iz += state.step_z;
        state.t_max_z += state.t_delta_z;
    }
    if (state.ix < 0 || state.iy < 0 || state.iz < 0 ||
        state.ix >= static_cast<int>(nx) || state.iy >= static_cast<int>(ny) ||
        state.iz >= static_cast<int>(nz)) {
        return false;
    }
    if (state.t_max_x < 0.0F) {
        state.t_max_x = huge;
    }
    if (state.t_max_y < 0.0F) {
        state.t_max_y = huge;
    }
    if (state.t_max_z < 0.0F) {
        state.t_max_z = huge;
    }
    return true;
}

inline float ct_dda_distance_to_next_face(const CtDdaState& state) noexcept {
    auto t = state.t_max_x;
    if (state.t_max_y < t) {
        t = state.t_max_y;
    }
    if (state.t_max_z < t) {
        t = state.t_max_z;
    }
    return t;
}

struct CtFaceClampResult {
    float step_mm{0.0F};
    bool hit_face{false};
    std::uint8_t axis_mask{0}; // 1 = x, 2 = y, 4 = z
};

inline CtFaceClampResult clamp_step_to_ct_faces_exact(
    const float step_mm,
    const float x_mm,
    const float y_mm,
    const float z_mm,
    const float dx,
    const float dy,
    const float dz,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const float spacing_x,
    const float spacing_y,
    const float spacing_z,
    const std::uint32_t nx,
    const std::uint32_t ny,
    const std::uint32_t nz) noexcept {
    CtFaceClampResult result{step_mm, false, 0};
    if (nx == 0 || ny == 0 || nz == 0) {
        return result;
    }
    CtDdaState state{};
    if (!ct_dda_init(x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
                     spacing_x, spacing_y, spacing_z, nx, ny, nz, state)) {
        return result;
    }
    const float t_face = ct_dda_distance_to_next_face(state);
    if (t_face > 0.0F && t_face <= step_mm) {
        result.step_mm = t_face;
        result.hit_face = true;
        result.axis_mask = 0;
        constexpr float kEps = 1.0e-6F;
        if (state.t_max_x <= t_face + kEps) result.axis_mask |= 1;
        if (state.t_max_y <= t_face + kEps) result.axis_mask |= 2;
        if (state.t_max_z <= t_face + kEps) result.axis_mask |= 4;
    }
    return result;
}

// Advance exactly one face (the nearest). Returns false if the next cell is
// outside the grid.
inline bool ct_dda_step(CtDdaState& state,
                        const std::uint32_t nx,
                        const std::uint32_t ny,
                        const std::uint32_t nz) noexcept {
    if (state.t_max_x <= state.t_max_y && state.t_max_x <= state.t_max_z) {
        state.ix += state.step_x;
        state.t_max_x += state.t_delta_x;
    } else if (state.t_max_y <= state.t_max_z) {
        state.iy += state.step_y;
        state.t_max_y += state.t_delta_y;
    } else {
        state.iz += state.step_z;
        state.t_max_z += state.t_delta_z;
    }
    return state.ix >= 0 && state.iy >= 0 && state.iz >= 0 &&
           state.ix < static_cast<int>(nx) && state.iy < static_cast<int>(ny) &&
           state.iz < static_cast<int>(nz);
}

// Max free path up to max_step that stays inside the same material/density span.
// Stops at the first face before entering a different material or density.
// Does not cross heterogeneous CT voxels.
inline float ct_dda_homogeneous_span(CtDdaState state,
                                     const float max_step,
                                     const std::uint32_t nx,
                                     const std::uint32_t ny,
                                     const std::uint32_t nz,
                                     const float* densities,
                                     const std::uint8_t* materials,
                                     const float density_here,
                                     const std::uint8_t material_here) noexcept {
    if (max_step <= 0.0F || densities == nullptr) {
        return 0.0F;
    }
    constexpr float dens_rel_tol = 0.02F;
    auto t_exit = 0.0F;
    // Cap span length to avoid long device loops in rare aligned rays.
    constexpr int max_voxels = 64;
    for (int hop = 0; hop < max_voxels; ++hop) {
        const auto t_face = ct_dda_distance_to_next_face(state);
        if (!(t_face > 0.0F) || t_face >= 1.0e29F) {
            return max_step;
        }
        if (t_exit + t_face >= max_step) {
            return max_step;
        }
        // Distance to the face that may enter a different voxel.
        const auto t_to_face = t_face;
        if (!ct_dda_step(state, nx, ny, nz)) {
            // Leaving the CT grid: allow travel up to the exit face.
            return t_exit + t_to_face < max_step ? t_exit + t_to_face : max_step;
        }
        const auto index = ct_linear_index(static_cast<std::uint32_t>(state.ix),
                                           static_cast<std::uint32_t>(state.iy),
                                           static_cast<std::uint32_t>(state.iz), nx, ny);
        const auto dens = densities[index];
        const auto mat =
            materials != nullptr ? materials[index] : static_cast<std::uint8_t>(2);
        const auto rel =
            (dens > density_here ? dens - density_here : density_here - dens) /
            (density_here > 1.0e-3F ? density_here : 1.0e-3F);
        if (mat != material_here || rel >= dens_rel_tol) {
            // Stop at the heterogeneous face (do not enter the new voxel).
            return t_exit + t_to_face;
        }
        t_exit += t_to_face;
    }
    return t_exit > 0.0F ? t_exit : max_step;
}

// Optional path_out records which face-clamp branch ran (for profiling only).
// Pass nullptr in production paths; the null check is free after inlining when
// constant-propagated.
//
// Uses Amanatides–Woo DDA so a particle sitting on a CT face advances into the
// next voxel instead of returning a near-zero step (eliminates boundary thrash).
inline float clamp_step_to_ct_faces(const float step_mm,
                                    const float x_mm,
                                    const float y_mm,
                                    const float z_mm,
                                    const float dx,
                                    const float dy,
                                    const float dz,
                                    const float origin_x,
                                    const float origin_y,
                                    const float origin_z,
                                    const float spacing_x,
                                    const float spacing_y,
                                    const float spacing_z,
                                    CtClampPath* path_out = nullptr,
                                    const std::uint32_t nx = 0,
                                    const std::uint32_t ny = 0,
                                    const std::uint32_t nz = 0) noexcept {
    if (path_out != nullptr) {
        *path_out = CtClampPath::three_axis;
    }
    if (nx > 0 && ny > 0 && nz > 0) {
        CtDdaState state{};
        if (ct_dda_init(x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
                        spacing_x, spacing_y, spacing_z, nx, ny, nz, state)) {
            const auto t_face = ct_dda_distance_to_next_face(state);
            if (t_face > 0.0F && t_face < step_mm) {
                return t_face;
            }
            return step_mm;
        }
    }
    // Fallback without grid dimensions (legacy / unit tests).
    auto step = step_mm;
    const auto tx = distance_to_next_ct_face_1d(x_mm, origin_x, spacing_x, dx);
    const auto ty = distance_to_next_ct_face_1d(y_mm, origin_y, spacing_y, dy);
    const auto tz = distance_to_next_ct_face_1d(z_mm, origin_z, spacing_z, dz);
    if (tx < step) {
        step = tx;
    }
    if (ty < step) {
        step = ty;
    }
    if (tz < step) {
        step = tz;
    }
    return step;
}

// Homogeneous DDA span + short-step skip (energy-limited steps thrash faces at Bragg peak).
inline float clamp_step_to_ct_faces_if_needed(const float step_mm,
                                              const float x_mm,
                                              const float y_mm,
                                              const float z_mm,
                                              const float dx,
                                              const float dy,
                                              const float dz,
                                              const float origin_x,
                                              const float origin_y,
                                              const float origin_z,
                                              const float spacing_x,
                                              const float spacing_y,
                                              const float spacing_z,
                                              const std::uint32_t nx,
                                              const std::uint32_t ny,
                                              const std::uint32_t nz,
                                              const float* densities,
                                              const std::uint8_t* materials,
                                              const float density_here,
                                              const std::uint8_t material_here,
                                              const bool skip_homogeneous = true,
                                              CtClampPath* path_out = nullptr) noexcept {
    if (!skip_homogeneous || densities == nullptr) {
        return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                      origin_y, origin_z, spacing_x, spacing_y, spacing_z,
                                      path_out, nx, ny, nz);
    }
    if (step_mm <= 1.0e-6F) {
        // Keep energy-limited micro-steps unchanged; DDA only prevents zero face
        // clamps (handled below when step is finite).
        if (path_out != nullptr) {
            *path_out = CtClampPath::short_step_skip;
        }
        return step_mm;
    }
    // Energy-limited steps much smaller than a voxel: skip face calc.
    const auto min_sp =
        spacing_x < spacing_y
            ? (spacing_x < spacing_z ? spacing_x : spacing_z)
            : (spacing_y < spacing_z ? spacing_y : spacing_z);
    if (step_mm < 0.2F * min_sp) {
        if (path_out != nullptr) {
            *path_out = CtClampPath::short_step_skip;
        }
        return step_mm;
    }

    CtDdaState state{};
    if (!ct_dda_init(x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
                     spacing_x, spacing_y, spacing_z, nx, ny, nz, state)) {
        return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                      origin_y, origin_z, spacing_x, spacing_y, spacing_z,
                                      path_out, nx, ny, nz);
    }

    // Fast path: endpoint still inside same homogeneous region → free step.
    const auto x1 = x_mm + dx * step_mm;
    const auto y1 = y_mm + dy * step_mm;
    const auto z1 = z_mm + dz * step_mm;
    float dens1 = density_here;
    std::uint8_t mat1 = material_here;
    if (ct_sample(x1, y1, z1, origin_x, origin_y, origin_z, spacing_x, spacing_y,
                  spacing_z, nx, ny, nz, densities, materials, dens1, mat1)) {
        const auto rel =
            (dens1 > density_here ? dens1 - density_here : density_here - dens1) /
            (density_here > 1.0e-3F ? density_here : 1.0e-3F);
        if (mat1 == material_here && rel < 0.02F) {
            if (path_out != nullptr) {
                *path_out = CtClampPath::homogeneous_skip;
            }
            return step_mm;
        }
    }

    // Walk DDA until material/density changes; stop at that face.
    const auto span = ct_dda_homogeneous_span(state, step_mm, nx, ny, nz, densities,
                                              materials, density_here, material_here);
    if (path_out != nullptr) {
        *path_out = (span >= step_mm) ? CtClampPath::homogeneous_skip
                                      : CtClampPath::three_axis;
    }
    return span < step_mm ? span : step_mm;
}

// Exact fast path for primary tracks that are nearly axial. If the proposed
// segment provably remains in the same CT x/y column, only the next z face can
// be the limiting CT face. Any lateral-cell change falls back to the general
// three-axis DDA implementation.
inline float clamp_step_to_ct_faces_near_z_if_needed(
    const float step_mm,
    const float x_mm,
    const float y_mm,
    const float z_mm,
    const float dx,
    const float dy,
    const float dz,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const float spacing_x,
    const float spacing_y,
    const float spacing_z,
    const std::uint32_t nx,
    const std::uint32_t ny,
    const std::uint32_t nz,
    const float* densities,
    const std::uint8_t* materials,
    const float density_here,
    const std::uint8_t material_here,
    const bool skip_homogeneous = true,
    CtClampPath* path_out = nullptr) noexcept {
    if (!skip_homogeneous || std::fabs(dz) < 0.999F || densities == nullptr) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous, path_out);
    }
    const auto min_sp =
        spacing_x < spacing_y
            ? (spacing_x < spacing_z ? spacing_x : spacing_z)
            : (spacing_y < spacing_z ? spacing_y : spacing_z);
    if (step_mm > 1.0e-6F && step_mm < 0.2F * min_sp) {
        if (path_out != nullptr) {
            *path_out = CtClampPath::short_step_skip;
        }
        return step_mm;
    }

    CtDdaState state{};
    if (!ct_dda_init(x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
                     spacing_x, spacing_y, spacing_z, nx, ny, nz, state)) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous, path_out);
    }

    // Lateral motion within this step would leave the x/y column → general DDA.
    const auto t_face = ct_dda_distance_to_next_face(state);
    const auto t_limit = step_mm < t_face ? step_mm : t_face;
    if (state.step_x != 0 && state.t_max_x <= t_limit + 1.0e-6F) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous, path_out);
    }
    if (state.step_y != 0 && state.t_max_y <= t_limit + 1.0e-6F) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous, path_out);
    }

    if (skip_homogeneous) {
        const auto x1 = x_mm + dx * step_mm;
        const auto y1 = y_mm + dy * step_mm;
        const auto z1 = z_mm + dz * step_mm;
        float density_end = density_here;
        std::uint8_t material_end = material_here;
        if (ct_sample(x1, y1, z1, origin_x, origin_y, origin_z, spacing_x, spacing_y,
                      spacing_z, nx, ny, nz, densities, materials, density_end,
                      material_end)) {
            const auto rel =
                std::fabs(density_end - density_here) /
                (density_here > 1.0e-3F ? density_here : 1.0e-3F);
            if (material_end == material_here && rel < 0.02F) {
                if (path_out != nullptr) {
                    *path_out = CtClampPath::near_z_homogeneous;
                }
                return step_mm;
            }
        }
        // Walk only along +z (column-locked) for the homogeneous span.
        const auto span = ct_dda_homogeneous_span(state, step_mm, nx, ny, nz, densities,
                                                  materials, density_here, material_here);
        if (path_out != nullptr) {
            *path_out = (span >= step_mm) ? CtClampPath::near_z_homogeneous
                                          : CtClampPath::near_z_face;
        }
        return span < step_mm ? span : step_mm;
    }

    if (path_out != nullptr) {
        *path_out = CtClampPath::near_z_face;
    }
    return state.t_max_z < step_mm ? state.t_max_z : step_mm;
}

}  // namespace carbon
