#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace carbon {

// Binary CT grid for GPU transport (7c).
// Coordinates: transport frame with beam along +z, entrance near z=0.
// File magic "CCTG" little-endian.
// v1: density float[] + material_id uint8[] (legacy 4-class 0..3)
// v2: + n_mass_sp_factors u32 + mass_sp_factor float[n]
//     material_id = Schneider material section index; SP uses
//     water_table(E) * mass_sp_factor[section] * density
struct CtGrid {
    static constexpr std::uint32_t magic_value = 0x47544343U;  // 'CCTG'
    static constexpr std::uint32_t version_value = 2U;
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
    // v1: 0=air,1=lung,2=water,3=bone; v2: Schneider section index
    std::vector<std::uint8_t> material_id{};
    // Relative mass stopping-power factors (water = 1). Empty → treat as 1.
    std::vector<float> mass_sp_factor{};
    std::uint32_t file_version{version_value};

    [[nodiscard]] std::size_t number_of_voxels() const noexcept {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz);
    }

    [[nodiscard]] bool has_mass_sp_factors() const noexcept {
        return !mass_sp_factor.empty();
    }

    static CtGrid from_binary(const std::filesystem::path& path);
    void write_binary(const std::filesystem::path& path) const;
};

// Legacy piecewise HU→density (tests / fallback only).
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

// Skip full 3-axis face clamp when density is nearly constant over the energy
// step (homogeneous region). Still clamps when material index would change.
// When skip_homogeneous is false, always face-clamps (for isolated perf A/B).
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
    // Probe endpoint of the unconstrained step.
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
        // Homogeneous enough: skip voxel-face thrashing.
        return step_mm;
    }
    return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                  origin_y, origin_z, spacing_x, spacing_y, spacing_z);
}

}  // namespace carbon
