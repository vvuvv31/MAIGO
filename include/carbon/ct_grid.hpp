#pragma once

#include "carbon/transport_profile.hpp"

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

// Collapse TOPAS/Schneider sections to the legacy material tables used by the
// current CT XS/SP implementation. Schneider bounds are:
//   0 air, 1 lung, 2..7 soft tissue/water-like, 8+ bone-like.
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
    return material_id < 8U ? 2U : 3U;
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
    // Match G4_WATER / ICRU mean excitation energy used by the Geant4 water SP
    // tables under data/stopping_power_water_*.csv (not the older 75 eV value).
    constexpr float I_water_eV = 78.0F;
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
    constexpr float eps = 1.0e-4F;
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

    // Snap off an already-crossed face so the first distance is positive.
    if (state.t_max_x <= eps && state.step_x != 0) {
        state.ix += state.step_x;
        state.t_max_x += state.t_delta_x;
    }
    if (state.t_max_y <= eps && state.step_y != 0) {
        state.iy += state.step_y;
        state.t_max_y += state.t_delta_y;
    }
    if (state.t_max_z <= eps && state.step_z != 0) {
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
    if (!skip_homogeneous || densities == nullptr) {
        return clamp_step_to_ct_faces(step_mm, x_mm, y_mm, z_mm, dx, dy, dz, origin_x,
                                      origin_y, origin_z, spacing_x, spacing_y, spacing_z,
                                      path_out, nx, ny, nz);
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
    if (std::fabs(dz) < 0.999F || densities == nullptr) {
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
