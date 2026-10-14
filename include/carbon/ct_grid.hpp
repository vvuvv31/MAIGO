#pragma once

#include "carbon/transport_profile.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

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

    static SchneiderMaterialTable from_topas_file(const std::filesystem::path& path);
    [[nodiscard]] std::uint8_t section_id(int hu) const noexcept;
};

// Patient CT volume stored in conventional medical-image coordinates:
//   x: patient left/right within an axial slice
//   y: patient posterior/anterior within an axial slice
//   z: patient inferior/superior, normal to the axial slices
// Beam direction is independent of these axes. Clinical transport must rotate
// the source, not repack the CT so that a beam happens to point along +z.
// Production input requires CCTG v3 with all 25 Schneider compositions.
// File magic "CCTG" little-endian.
// v3: + n u32 + za_rel float[n] + I_eV float[n]
//     SP = SP_water(E) * mass_sp_energy_factor(za,I,E) * density
struct CtGrid {
    static constexpr std::uint32_t magic_value = 0x47544343U;  // 'CCTG'
    static constexpr std::uint32_t version_value = 3U;

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
    std::vector<std::uint8_t> material_id{};
    // (Z/A)_section / (Z/A)_water  — high-energy limit mass-SP factor
    std::vector<float> mass_sp_za_rel{};
    // Bragg mean excitation energy I [eV] per Schneider section
    std::vector<float> mass_sp_I_eV{};
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
        return mass_sp_za_rel.size() == 25 && mass_sp_I_eV.size() == 25;
    }

    [[nodiscard]] bool uses_schneider_mass_sp() const noexcept {
        return file_version == version_value && has_mass_sp_factors();
    }

    static CtGrid from_binary(const std::filesystem::path& path);
    // Current CCTG binary, decoded once into an immutable shared grid.
    static std::shared_ptr<const CtGrid> load(const std::filesystem::path& path);

    void write_binary(const std::filesystem::path& path) const;
};

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
                      const float dir_z = 0.0F,
                      std::size_t* index_out = nullptr) noexcept {
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
    if(index_out) *index_out = index;
    density_out = densities[index];
    material_out = materials != nullptr ? materials[index] : static_cast<std::uint8_t>(2);
    return true;
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

// Snap only a fully traversed face-limited segment, never a nuclear-truncated
// segment. nextafter assigns both positive and negative crossings to the
// destination cell; merely flooring a negative-going face leaves it upstream.
inline std::array<float,3> ct_finish_exact_face_step(
    const CtFaceClampResult& clamp, const float used_step,
    const std::array<float,3>& start, const std::array<float,3>& direction,
    const std::array<float,3>& origin, const std::array<float,3>& spacing) noexcept {
    std::array<float,3> result{};
    for (int axis=0;axis<3;++axis) {
        result[axis]=start[axis]+used_step*direction[axis];
        if (clamp.hit_face && used_step==clamp.step_mm &&
            (clamp.axis_mask & (1U<<axis)) && std::fabs(direction[axis])>1e-6F) {
            const float face=origin[axis]+std::round((result[axis]-origin[axis])/spacing[axis])*spacing[axis];
            result[axis]=std::nextafter(face,direction[axis]>0 ? 1e30F : -1e30F);
        }
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
                                    CtClampPath* path_out,
                                    const std::uint32_t nx,
                                    const std::uint32_t ny,
                                    const std::uint32_t nz) noexcept {
    if (path_out != nullptr) {
        *path_out = CtClampPath::three_axis;
    }
    if (nx == 0 || ny == 0 || nz == 0)
        return std::numeric_limits<float>::quiet_NaN();
    {
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
    // Outside the grid, limit to the next physical voxel-face plane.
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
