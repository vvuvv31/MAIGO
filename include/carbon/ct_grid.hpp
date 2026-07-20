#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

// Binary CT grid for GPU transport (7c).
// Coordinates: transport frame with beam along +z, entrance near z=0.
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

    [[nodiscard]] bool has_mass_sp_factors() const noexcept {
        return !mass_sp_za_rel.empty() || !mass_sp_factor.empty();
    }

    static CtGrid from_binary(const std::filesystem::path& path);
    void write_binary(const std::filesystem::path& path) const;
};

float hu_to_density_g_per_cm3(float hu) noexcept;
std::uint8_t density_to_material_id(float density_g_per_cm3) noexcept;

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
                      std::uint8_t& material_out) noexcept {
    if (densities == nullptr || nx == 0 || ny == 0 || nz == 0) {
        return false;
    }
    const auto fx = (x_mm - origin_x) / spacing_x;
    const auto fy = (y_mm - origin_y) / spacing_y;
    const auto fz = (z_mm - origin_z) / spacing_z;
    if (fx < 0.0F || fy < 0.0F || fz < 0.0F) {
        return false;
    }
    const auto ix = static_cast<std::uint32_t>(fx);
    const auto iy = static_cast<std::uint32_t>(fy);
    const auto iz = static_cast<std::uint32_t>(fz);
    if (ix >= nx || iy >= ny || iz >= nz) {
        return false;
    }
    const auto index = ct_linear_index(ix, iy, iz, nx, ny);
    density_out = densities[index];
    material_out = materials != nullptr ? materials[index] : static_cast<std::uint8_t>(2);
    return true;
}

// Energy-dependent mass-SP factor vs liquid water (I_w = 75 eV).
// za_rel = (Z/A)_s / (Z/A)_w. Uses a simplified Bethe stopping-number ratio.
template <typename LogFn>
inline float ct_mass_sp_energy_factor_impl(const float za_rel,
                                          const float I_eV,
                                          const float energy_MeVu,
                                          LogFn&& log_fn) noexcept {
    constexpr float nucleon_mass_MeV = 931.49410242F;
    constexpr float two_me_c2_MeV = 1.0219979F;
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

inline float distance_to_next_ct_face_1d(const float position_mm,
                                        const float origin_mm,
                                        const float spacing_mm,
                                        const float direction) noexcept {
    constexpr float eps = 1.0e-4F;
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
        if (t <= eps) {
            boundary += spacing_mm;
            t = (boundary - position_mm) / direction;
        }
        return t > 0.0F ? t : 1.0e30F;
    }
    auto boundary = origin_mm + static_cast<float>(cell) * spacing_mm;
    auto t = (boundary - position_mm) / direction;
    if (t <= eps) {
        boundary -= spacing_mm;
        t = (boundary - position_mm) / direction;
    }
    return t > 0.0F ? t : 1.0e30F;
}

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
                                    const float spacing_z) noexcept {
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

// Homogeneous skip + short-step skip (energy-limited steps thrash faces at Bragg peak).
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
                                              const bool skip_homogeneous = true) noexcept {
    if (!skip_homogeneous || densities == nullptr || step_mm <= 1.0e-6F) {
        return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                      origin_y, origin_z, spacing_x, spacing_y, spacing_z);
    }
    // Energy-limited steps much smaller than a voxel: skip face calc (C).
    const auto min_sp =
        spacing_x < spacing_y
            ? (spacing_x < spacing_z ? spacing_x : spacing_z)
            : (spacing_y < spacing_z ? spacing_y : spacing_z);
    if (step_mm < 0.2F * min_sp) {
        return step_mm;
    }
    const auto x1 = x_mm + dx * step_mm;
    const auto y1 = y_mm + dy * step_mm;
    const auto z1 = z_mm + dz * step_mm;
    float dens1 = density_here;
    std::uint8_t mat1 = material_here;
    if (!ct_sample(x1, y1, z1, origin_x, origin_y, origin_z, spacing_x, spacing_y,
                   spacing_z, nx, ny, nz, densities, materials, dens1, mat1)) {
        return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                      origin_y, origin_z, spacing_x, spacing_y, spacing_z);
    }
    const auto rel =
        (dens1 > density_here ? dens1 - density_here : density_here - dens1) /
        (density_here > 1.0e-3F ? density_here : 1.0e-3F);
    if (mat1 == material_here && rel < 0.02F) {
        return step_mm;
    }
    return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                  origin_y, origin_z, spacing_x, spacing_y, spacing_z);
}

// Exact fast path for primary tracks that are nearly axial. If the proposed
// segment provably remains in the same CT x/y column, only the next z face can
// be the limiting CT face. Any lateral-cell change falls back to the general
// three-axis implementation.
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
    const bool skip_homogeneous = true) noexcept {
    if (std::fabs(dz) < 0.999F || !skip_homogeneous || densities == nullptr ||
        step_mm <= 1.0e-6F) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous);
    }
    const auto min_sp =
        spacing_x < spacing_y
            ? (spacing_x < spacing_z ? spacing_x : spacing_z)
            : (spacing_y < spacing_z ? spacing_y : spacing_z);
    if (step_mm < 0.2F * min_sp) {
        return step_mm;
    }
    const auto x1 = x_mm + dx * step_mm;
    const auto y1 = y_mm + dy * step_mm;
    const auto z1 = z_mm + dz * step_mm;
    const auto fx0 = (x_mm - origin_x) / spacing_x;
    const auto fy0 = (y_mm - origin_y) / spacing_y;
    const auto fx1 = (x1 - origin_x) / spacing_x;
    const auto fy1 = (y1 - origin_y) / spacing_y;
    if (fx1 < 0.0F || fy1 < 0.0F || fx1 >= static_cast<float>(nx) ||
        fy1 >= static_cast<float>(ny)) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous);
    }
    const auto ix0 = static_cast<int>(fx0);
    const auto iy0 = static_cast<int>(fy0);
    const auto ix1 = static_cast<int>(fx1);
    const auto iy1 = static_cast<int>(fy1);
    if (ix0 != ix1 || iy0 != iy1) {
        return clamp_step_to_ct_faces_if_needed(
            step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x, origin_y, origin_z,
            spacing_x, spacing_y, spacing_z, nx, ny, nz, densities, materials,
            density_here, material_here, skip_homogeneous);
    }
    float density_end = density_here;
    std::uint8_t material_end = material_here;
    if (ct_sample(x1, y1, z1, origin_x, origin_y, origin_z, spacing_x, spacing_y,
                  spacing_z, nx, ny, nz, densities, materials, density_end,
                  material_end)) {
        const auto rel =
            std::fabs(density_end - density_here) /
            (density_here > 1.0e-3F ? density_here : 1.0e-3F);
        if (material_end == material_here && rel < 0.02F) {
            return step_mm;
        }
    }
    const auto z_face =
        distance_to_next_ct_face_1d(z_mm, origin_z, spacing_z, dz);
    return z_face < step_mm ? z_face : step_mm;
}

}  // namespace carbon
