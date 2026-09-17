#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace carbon {

// Axial slab layers for density-heterogeneous water-equivalent phantoms.
// Layer i covers [z_start_i, z_end_i) where z_start_0 = 0 and
// z_start_{i} = z_end_{i-1}. Tables (SP/XS) are for water at 1 g/cm3;
// local density scales dE/dx, macroscopic nuclear XS, straggling, and MCS.
struct SlabLayer {
    double z_end_mm{0.0};
    double density_g_per_cm3{1.0};
};

// Axis-aligned insert for lateral heterogeneity (7b): bone strip / air cavity.
// Half-open box [x_min,x_max) x [y_min,y_max) x [z_min,z_max) in water-depth coords.
struct HeteroInsert {
    double x_min_mm{-10.0};
    double x_max_mm{10.0};
    double y_min_mm{-10.0};
    double y_max_mm{10.0};
    double z_min_mm{50.0};
    double z_max_mm{70.0};
    double density_g_per_cm3{1.85};
};

// Host-side validation: layers sorted, last end matches phantom, densities > 0.
void validate_slab_layers(const std::vector<SlabLayer>& layers,
                          double phantom_length_mm);
void validate_hetero_insert(const HeteroInsert& insert, double phantom_length_mm);

// Device/host helpers (header-only, SYCL-safe arithmetic).
inline std::uint32_t slab_layer_index(const float position_z_mm,
                                      const float* z_ends_mm,
                                      const std::uint32_t layer_count) noexcept {
    if (layer_count == 0) {
        return 0;
    }
    for (std::uint32_t index = 0; index < layer_count; ++index) {
        if (position_z_mm < z_ends_mm[index]) {
            return index;
        }
    }
    return layer_count - 1U;
}

inline float slab_density_g_per_cm3(const float position_z_mm,
                                    const float* z_ends_mm,
                                    const float* densities,
                                    const std::uint32_t layer_count,
                                    const float fallback_density) noexcept {
    if (layer_count == 0 || z_ends_mm == nullptr || densities == nullptr) {
        return fallback_density;
    }
    return densities[slab_layer_index(position_z_mm, z_ends_mm, layer_count)];
}

// Path length (mm) from z along dir_z to the next slab interface (or 0 if none).
inline float distance_to_slab_interface_mm(const float position_z_mm,
                                           const float direction_z,
                                           const float* z_ends_mm,
                                           const std::uint32_t layer_count,
                                           const float phantom_length_mm) noexcept {
    if (layer_count == 0 || z_ends_mm == nullptr) {
        return phantom_length_mm;  // large; caller also clips to bins
    }
    constexpr float eps = 1.0e-6F;
    const auto abs_uz = direction_z < 0.0F ? -direction_z : direction_z;
    if (abs_uz < eps) {
        return phantom_length_mm;
    }
    const auto index = slab_layer_index(position_z_mm, z_ends_mm, layer_count);
    if (direction_z > 0.0F) {
        const auto z_end = z_ends_mm[index];
        const auto delta = z_end - position_z_mm;
        return delta > 0.0F ? delta / direction_z : 0.0F;
    }
    const auto z_start = index == 0U ? 0.0F : z_ends_mm[index - 1U];
    const auto delta = position_z_mm - z_start;
    return delta > 0.0F ? delta / abs_uz : 0.0F;
}

inline bool inside_hetero_insert(const float x_mm,
                                 const float y_mm,
                                 const float z_mm,
                                 const float x_min,
                                 const float x_max,
                                 const float y_min,
                                 const float y_max,
                                 const float z_min,
                                 const float z_max) noexcept {
    return x_mm >= x_min && x_mm < x_max && y_mm >= y_min && y_mm < y_max &&
           z_mm >= z_min && z_mm < z_max;
}

// Ray vs AABB: distance (mm) along direction to next material interface of the insert.
// If inside: distance to exit. If outside: distance to entry (or large_value if miss).
inline float distance_to_insert_interface_mm(const float x_mm,
                                             const float y_mm,
                                             const float z_mm,
                                             const float dx,
                                             const float dy,
                                             const float dz,
                                             const float x_min,
                                             const float x_max,
                                             const float y_min,
                                             const float y_max,
                                             const float z_min,
                                             const float z_max,
                                             const float large_value) noexcept {
    constexpr float eps = 1.0e-6F;
    const auto inside = inside_hetero_insert(x_mm, y_mm, z_mm, x_min, x_max, y_min,
                                             y_max, z_min, z_max);
    auto t_enter = 0.0F;
    auto t_exit = large_value;
    // X slabs
    if (dx > eps || dx < -eps) {
        const auto inv = 1.0F / dx;
        auto t0 = (x_min - x_mm) * inv;
        auto t1 = (x_max - x_mm) * inv;
        if (t0 > t1) {
            const auto tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
    } else if (x_mm < x_min || x_mm >= x_max) {
        return large_value;
    }
    // Y slabs
    if (dy > eps || dy < -eps) {
        const auto inv = 1.0F / dy;
        auto t0 = (y_min - y_mm) * inv;
        auto t1 = (y_max - y_mm) * inv;
        if (t0 > t1) {
            const auto tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
    } else if (y_mm < y_min || y_mm >= y_max) {
        return large_value;
    }
    // Z slabs
    if (dz > eps || dz < -eps) {
        const auto inv = 1.0F / dz;
        auto t0 = (z_min - z_mm) * inv;
        auto t1 = (z_max - z_mm) * inv;
        if (t0 > t1) {
            const auto tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
    } else if (z_mm < z_min || z_mm >= z_max) {
        return large_value;
    }
    if (t_enter > t_exit || t_exit < 0.0F) {
        return large_value;
    }
    // Keep a finite positive step so surface FP jitter cannot stall transport.
    constexpr float min_interface_step = 1.0e-4F;
    if (inside) {
        return t_exit > min_interface_step ? t_exit : min_interface_step;
    }
    if (t_enter <= 0.0F) {
        return min_interface_step;
    }
    return t_enter;
}

}  // namespace carbon
