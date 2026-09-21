#pragma once

#include <cmath>

namespace carbon {

// Device-safe exact geometry for a finite parallel-slit collimator in the
// canonical beam frame. The phantom entrance is z=0 and the beam travels +z.
inline int nearest_minibeam_slit(const float u_mm, const float pitch_mm) noexcept {
    const auto scaled = u_mm / pitch_mm;
    return scaled >= 0.0F ? static_cast<int>(scaled + 0.5F)
                          : static_cast<int>(scaled - 0.5F);
}

inline bool minibeam_point_in_slit(
    const float x_mm,
    const float y_mm,
    const float cosine_angle,
    const float sine_angle,
    const float radius_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm,
    const float slit_offset_mm,
    int& slit_index) noexcept {
    const auto u_mm = cosine_angle * x_mm + sine_angle * y_mm;
    const auto v_mm = -sine_angle * x_mm + cosine_angle * y_mm;
    if (u_mm * u_mm + v_mm * v_mm >= radius_mm * radius_mm ||
        (v_mm <= -slit_half_length_mm || v_mm >= slit_half_length_mm)) {
        return false;
    }
    const auto centered_u_mm = u_mm - slit_offset_mm;
    slit_index = nearest_minibeam_slit(centered_u_mm, slit_pitch_mm);
    const auto half_count = slit_count / 2;
    if (slit_index < -half_count || slit_index > half_count) {
        return false;
    }
    const auto delta = centered_u_mm - static_cast<float>(slit_index) * slit_pitch_mm;
    return delta > -0.5F * slit_width_mm && delta < 0.5F * slit_width_mm;
}

// Rectangular equivalent used by the GPU minibeam path.  The outer body is a
// box; the slit array is unchanged from TOPAS.  Strict boundary comparisons
// deliberately assign shared solid boundaries to Copper.
inline bool minibeam_point_in_rectangular_slit(
    const float x_mm, const float y_mm,
    const float cosine_angle, const float sine_angle,
    const float block_width_mm, const float block_length_mm,
    const int slit_count, const float slit_width_mm,
    const float slit_pitch_mm, const float slit_length_mm,
    const float slit_offset_mm, int& slit_index) noexcept {
    const auto u_mm = cosine_angle * x_mm + sine_angle * y_mm;
    const auto v_mm = -sine_angle * x_mm + cosine_angle * y_mm;
    if (!(u_mm > -0.5F * block_width_mm && u_mm < 0.5F * block_width_mm &&
          v_mm > -0.5F * block_length_mm && v_mm < 0.5F * block_length_mm &&
          v_mm > -0.5F * slit_length_mm && v_mm < 0.5F * slit_length_mm)) {
        return false;
    }
    const auto centered_u_mm = u_mm - slit_offset_mm;
    slit_index = nearest_minibeam_slit(centered_u_mm, slit_pitch_mm);
    const auto half_count = slit_count / 2;
    if (slit_index < -half_count || slit_index > half_count) return false;
    const auto slit_center_mm =
        slit_offset_mm + static_cast<float>(slit_index) * slit_pitch_mm;
    return u_mm > slit_center_mm - 0.5F * slit_width_mm &&
           u_mm < slit_center_mm + 0.5F * slit_width_mm;
}

inline bool minibeam_point_in_rectangular_copper(
    const float x_mm, const float y_mm,
    const float cosine_angle, const float sine_angle,
    const float block_width_mm, const float block_length_mm,
    const int slit_count, const float slit_width_mm,
    const float slit_pitch_mm, const float slit_length_mm,
    const float slit_offset_mm) noexcept {
    const auto u_mm = cosine_angle * x_mm + sine_angle * y_mm;
    const auto v_mm = -sine_angle * x_mm + cosine_angle * y_mm;
    if (!(u_mm > -0.5F * block_width_mm && u_mm < 0.5F * block_width_mm &&
          v_mm > -0.5F * block_length_mm && v_mm < 0.5F * block_length_mm)) {
        return false;
    }
    int slit = 0;
    return !minibeam_point_in_rectangular_slit(
        x_mm, y_mm, cosine_angle, sine_angle, block_width_mm, block_length_mm,
        slit_count, slit_width_mm, slit_pitch_mm, slit_length_mm,
        slit_offset_mm, slit);
}

// Returns true when a forward ray intersects the rectangular body and is not
// wholly contained by one slit across the requested slit thickness.
inline bool minibeam_ray_hits_rectangular_copper(
    const float position_x_mm, const float position_y_mm,
    const float position_z_mm, const float direction_x,
    const float direction_y, const float direction_z,
    const float cosine_angle, const float sine_angle,
    const float block_width_mm, const float block_length_mm,
    const float block_center_z_mm, const float block_thickness_mm,
    const int slit_count, const float slit_width_mm,
    const float slit_pitch_mm, const float slit_length_mm,
    const float slit_thickness_mm, const float slit_offset_mm) noexcept {
    if (direction_z <= 1.0e-8F) return true;
    const auto slit_entrance_z = block_center_z_mm - 0.5F * slit_thickness_mm;
    const auto slit_exit_z = block_center_z_mm + 0.5F * slit_thickness_mm;
    const auto block_entrance_z = block_center_z_mm - 0.5F * block_thickness_mm;
    const auto block_exit_z = block_center_z_mm + 0.5F * block_thickness_mm;
    const auto block_t0 = (block_entrance_z - position_z_mm) / direction_z;
    const auto block_t1 = (block_exit_z - position_z_mm) / direction_z;
    if (block_t1 <= 0.0F || block_t1 <= block_t0) return false;

    const auto slit_t0 = (slit_entrance_z - position_z_mm) / direction_z;
    const auto slit_t1 = (slit_exit_z - position_z_mm) / direction_z;
    if (slit_t0 < 0.0F || slit_t1 <= slit_t0) return true;
    const auto x0 = position_x_mm + slit_t0 * direction_x;
    const auto y0 = position_y_mm + slit_t0 * direction_y;
    const auto x1 = position_x_mm + slit_t1 * direction_x;
    const auto y1 = position_y_mm + slit_t1 * direction_y;
    int slit0 = 0, slit1 = 0;
    const auto through_one_slit =
        minibeam_point_in_rectangular_slit(
            x0, y0, cosine_angle, sine_angle, block_width_mm,
            block_length_mm, slit_count, slit_width_mm, slit_pitch_mm,
            slit_length_mm, slit_offset_mm, slit0) &&
        minibeam_point_in_rectangular_slit(
            x1, y1, cosine_angle, sine_angle, block_width_mm,
            block_length_mm, slit_count, slit_width_mm, slit_pitch_mm,
            slit_length_mm, slit_offset_mm, slit1) && slit0 == slit1;
    if (through_one_slit && slit_thickness_mm >= block_thickness_mm) return false;

    // Exact forward ray/box overlap, including rays that enter through a side
    // even though both z-face points lie outside the box.
    const auto origin_u = cosine_angle * position_x_mm + sine_angle * position_y_mm;
    const auto origin_v = -sine_angle * position_x_mm + cosine_angle * position_y_mm;
    const auto direction_u = cosine_angle * direction_x + sine_angle * direction_y;
    const auto direction_v = -sine_angle * direction_x + cosine_angle * direction_y;
    auto enter = block_t0 > 0.0F ? block_t0 : 0.0F;
    auto leave = block_t1;
    auto intersect_axis = [&](const float origin, const float direction,
                              const float half_extent) {
        if (direction > -1.0e-8F && direction < 1.0e-8F) {
            if (origin <= -half_extent || origin >= half_extent) leave = -1.0F;
            return;
        }
        auto first = (-half_extent - origin) / direction;
        auto second = (half_extent - origin) / direction;
        if (first > second) {
            const auto temporary = first;
            first = second;
            second = temporary;
        }
        if (first > enter) enter = first;
        if (second < leave) leave = second;
    };
    intersect_axis(origin_u, direction_u, 0.5F * block_width_mm);
    intersect_axis(origin_v, direction_v, 0.5F * block_length_mm);
    return leave > enter;
}

inline bool minibeam_point_in_copper(
    const float x_mm,
    const float y_mm,
    const float cosine_angle,
    const float sine_angle,
    const float radius_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm,
    const float slit_offset_mm) noexcept {
    if (x_mm * x_mm + y_mm * y_mm >= radius_mm * radius_mm) {
        return false;
    }
    auto slit_index = 0;
    return !minibeam_point_in_slit(
        x_mm, y_mm, cosine_angle, sine_angle, radius_mm, slit_count,
        slit_width_mm, slit_pitch_mm, slit_half_length_mm, slit_offset_mm,
        slit_index);
}

// Distance along a straight ray to the first Copper/air material boundary,
// capped by max_path_mm.  Candidate surfaces include the cylindrical body and
// every finite slit side/end.  A before/after material probe rejects surfaces
// that do not actually change material (for example a slit side beyond its
// finite length).  Condensed scattering displacement is sampled only after
// this straight-path navigation segment has been selected.
inline float minibeam_path_to_material_boundary(
    const float x_mm, const float y_mm,
    const float direction_x, const float direction_y,
    const float cosine_angle, const float sine_angle,
    const float radius_mm, const int slit_count,
    const float slit_width_mm, const float slit_pitch_mm,
    const float slit_half_length_mm, const float slit_offset_mm,
    const float max_path_mm) noexcept {
    if (!(max_path_mm > 0.0F)) return 0.0F;
    const auto origin_u = cosine_angle * x_mm + sine_angle * y_mm;
    const auto origin_v = -sine_angle * x_mm + cosine_angle * y_mm;
    const auto direction_u =
        cosine_angle * direction_x + sine_angle * direction_y;
    const auto direction_v =
        -sine_angle * direction_x + cosine_angle * direction_y;
    auto best = max_path_mm;
    constexpr float minimum_path = 1.0e-6F;
    constexpr float probe_path = 1.0e-5F;

    auto consider = [&](const float candidate) {
        if (!(candidate > minimum_path && candidate < best)) return;
        const auto before_path = candidate > probe_path
            ? candidate - probe_path : 0.5F * candidate;
        const auto after_path = candidate + probe_path < max_path_mm
            ? candidate + probe_path
            : 0.5F * (candidate + max_path_mm);
        if (!(after_path > candidate)) return;
        const auto before_copper = minibeam_point_in_copper(
            x_mm + before_path * direction_x,
            y_mm + before_path * direction_y,
            cosine_angle, sine_angle, radius_mm, slit_count,
            slit_width_mm, slit_pitch_mm, slit_half_length_mm,
            slit_offset_mm);
        const auto after_copper = minibeam_point_in_copper(
            x_mm + after_path * direction_x,
            y_mm + after_path * direction_y,
            cosine_angle, sine_angle, radius_mm, slit_count,
            slit_width_mm, slit_pitch_mm, slit_half_length_mm,
            slit_offset_mm);
        if (before_copper != after_copper) best = candidate;
    };

    const auto transverse_a =
        direction_x * direction_x + direction_y * direction_y;
    if (transverse_a > 1.0e-16F) {
        const auto transverse_b = x_mm * direction_x + y_mm * direction_y;
        const auto transverse_c =
            x_mm * x_mm + y_mm * y_mm - radius_mm * radius_mm;
        const auto discriminant =
            transverse_b * transverse_b - transverse_a * transverse_c;
        if (discriminant >= 0.0F) {
            const auto root = std::sqrt(discriminant);
            consider((-transverse_b - root) / transverse_a);
            consider((-transverse_b + root) / transverse_a);
        }
    }

    const auto half_count = slit_count / 2;
    const auto nearest_slit = nearest_minibeam_slit(
        origin_u - slit_offset_mm, slit_pitch_mm);
    for (int local_offset = -1; local_offset <= 1; ++local_offset) {
        const auto slit = nearest_slit + local_offset;
        if (slit < -half_count || slit > half_count) continue;
        const auto center_u =
            slit_offset_mm + static_cast<float>(slit) * slit_pitch_mm;
        if (std::fabs(direction_u) > 1.0e-8F) {
            consider((center_u - 0.5F * slit_width_mm - origin_u) /
                     direction_u);
            consider((center_u + 0.5F * slit_width_mm - origin_u) /
                     direction_u);
        }
        if (std::fabs(direction_v) > 1.0e-8F) {
            consider((-slit_half_length_mm - origin_v) / direction_v);
            consider((slit_half_length_mm - origin_v) / direction_v);
        }
    }
    return best;
}

// Straight-ahead Copper length before a downstream axial plane.  This is a
// diagnostic geometry integral: it subtracts finite slit air and stops after
// a cylindrical side exit, but intentionally does not predict later MCS.
inline float minibeam_straight_copper_path_to_plane(
    float x_mm, float y_mm, const float direction_x, const float direction_y,
    const float cosine_angle, const float sine_angle, const float radius_mm,
    const int slit_count, const float slit_width_mm,
    const float slit_pitch_mm, const float slit_half_length_mm,
    const float slit_offset_mm, float path_to_plane_mm) noexcept {
    auto copper_path_mm = 0.0F;
    constexpr unsigned maximum_segments = 64U;
    constexpr float minimum_remaining_path = 1.0e-6F;
    for (unsigned segment = 0;
         segment < maximum_segments &&
         path_to_plane_mm > minimum_remaining_path;
         ++segment) {
        const auto path = minibeam_path_to_material_boundary(
            x_mm, y_mm, direction_x, direction_y, cosine_angle, sine_angle,
            radius_mm, slit_count, slit_width_mm, slit_pitch_mm,
            slit_half_length_mm, slit_offset_mm, path_to_plane_mm);
        if (!(path > 0.0F)) break;
        const auto midpoint_x = x_mm + 0.5F * path * direction_x;
        const auto midpoint_y = y_mm + 0.5F * path * direction_y;
        if (minibeam_point_in_copper(
                midpoint_x, midpoint_y, cosine_angle, sine_angle, radius_mm,
                slit_count, slit_width_mm, slit_pitch_mm,
                slit_half_length_mm, slit_offset_mm)) {
            copper_path_mm += path;
        }
        x_mm += path * direction_x;
        y_mm += path * direction_y;
        path_to_plane_mm -= path;
    }
    return copper_path_mm;
}

// Returns true when a forward ray overlaps the finite cylindrical body and is
// not wholly contained by one air slit.  The cylinder and each slit are convex,
// so two probes just inside the ray/body interval are sufficient.
inline bool minibeam_ray_hits_cylindrical_copper(
    const float position_x_mm,
    const float position_y_mm,
    const float position_z_mm,
    const float direction_x,
    const float direction_y,
    const float direction_z,
    const float cosine_angle,
    const float sine_angle,
    const float radius_mm,
    const float block_center_z_mm,
    const float block_thickness_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm,
    const float slit_thickness_mm,
    const float slit_offset_mm) noexcept {
    if (direction_z <= 1.0e-8F) {
        return true;
    }
    const auto entrance_z_mm = block_center_z_mm - 0.5F * block_thickness_mm;
    const auto exit_z_mm = block_center_z_mm + 0.5F * block_thickness_mm;
    auto enter = (entrance_z_mm - position_z_mm) / direction_z;
    auto leave = (exit_z_mm - position_z_mm) / direction_z;
    if (leave <= 0.0F || leave <= enter) {
        return false;
    }
    if (enter < 0.0F) enter = 0.0F;

    const auto transverse_a = direction_x * direction_x + direction_y * direction_y;
    const auto transverse_b = position_x_mm * direction_x +
                              position_y_mm * direction_y;
    const auto transverse_c = position_x_mm * position_x_mm +
                              position_y_mm * position_y_mm - radius_mm * radius_mm;
    if (transverse_a <= 1.0e-16F) {
        if (transverse_c >= 0.0F) return false;
    } else {
        const auto discriminant = transverse_b * transverse_b -
                                  transverse_a * transverse_c;
        if (discriminant <= 0.0F) return false;
        const auto root = __builtin_sqrtf(discriminant);
        auto radial_enter = (-transverse_b - root) / transverse_a;
        auto radial_leave = (-transverse_b + root) / transverse_a;
        if (radial_enter > enter) enter = radial_enter;
        if (radial_leave < leave) leave = radial_leave;
        if (leave <= enter || leave <= 0.0F) return false;
        if (enter < 0.0F) enter = 0.0F;
    }

    if (slit_thickness_mm + 1.0e-6F < block_thickness_mm) return true;
    const auto inset = (leave - enter) > 2.0e-5F ? 1.0e-5F : 0.0F;
    const auto entrance_distance = enter + inset;
    const auto exit_distance = leave - inset;

    const auto entrance_x = position_x_mm + entrance_distance * direction_x;
    const auto entrance_y = position_y_mm + entrance_distance * direction_y;
    const auto exit_x = position_x_mm + exit_distance * direction_x;
    const auto exit_y = position_y_mm + exit_distance * direction_y;
    auto entrance_slit = 0;
    auto exit_slit = 0;
    const auto through_one_slit = minibeam_point_in_slit(
               entrance_x, entrance_y, cosine_angle, sine_angle, radius_mm,
               slit_count, slit_width_mm, slit_pitch_mm, slit_half_length_mm,
               slit_offset_mm, entrance_slit) &&
           minibeam_point_in_slit(
               exit_x, exit_y, cosine_angle, sine_angle, radius_mm, slit_count,
               slit_width_mm, slit_pitch_mm, slit_half_length_mm,
               slit_offset_mm, exit_slit) &&
           entrance_slit == exit_slit;
    return !through_one_slit;
}

// Isotropic safety at a point inside/near the collimator block: minimum
// distance to any Copper/air material boundary (R1 fix supporting the real
// production caller).  This is NOT the along-ray distance to the next
// boundary and NOT the step-start safety.  The caller evaluates it at the
// geometric endpoint (position + g*direction) before displacement.
// Transverse part: outer circle + nearest finite-slit walls/ends.
// Axial part: block entrance/exit planes.  Returns >= 0.
inline float minibeam_isotropic_safety_at_point(
    const float x_mm, const float y_mm, const float z_mm,
    const float cosine_angle, const float sine_angle,
    const float radius_mm, const int slit_count,
    const float slit_width_mm, const float slit_pitch_mm,
    const float slit_half_length_mm, const float slit_offset_mm,
    const float block_entrance_z_mm, const float block_exit_z_mm) noexcept {
    float safety = 1.0e30F;
    const auto r = std::sqrt(x_mm * x_mm + y_mm * y_mm);
    // Outer cylindrical wall (extruded along z while inside the block z range).
    if (z_mm >= block_entrance_z_mm && z_mm <= block_exit_z_mm) {
        const auto d_outer = radius_mm - r;
        if (d_outer < safety) safety = d_outer;
    }
    // Axial planes.
    {
        const auto d_enter = z_mm - block_entrance_z_mm;
        const auto d_exit = block_exit_z_mm - z_mm;
        auto d_axial = d_enter < d_exit ? d_enter : d_exit;
        if (d_axial < safety) safety = d_axial;
    }
    // Finite slit walls/ends around the nearest slits.
    const auto u_mm = cosine_angle * x_mm + sine_angle * y_mm;
    const auto v_mm = -sine_angle * x_mm + cosine_angle * y_mm;
    const auto half_count = slit_count / 2;
    const auto nearest_slit = nearest_minibeam_slit(
        u_mm - slit_offset_mm, slit_pitch_mm);
    for (int local_offset = -1; local_offset <= 1; ++local_offset) {
        const auto slit = nearest_slit + local_offset;
        if (slit < -half_count || slit > half_count) continue;
        const auto center_u =
            slit_offset_mm + static_cast<float>(slit) * slit_pitch_mm;
        // Distance to the two side walls of this slit.
        const auto d_wall = std::fabs(u_mm - center_u) - 0.5F * slit_width_mm;
        // Inside-slit (air) points: distance to the nearest wall is -d_wall
        // when d_wall < 0; inside-copper points: distance to wall is +d_wall
        // when the v coordinate is within the finite slit length.
        const bool v_inside = v_mm > -slit_half_length_mm &&
                              v_mm < slit_half_length_mm;
        if (v_inside) {
            const auto candidate = std::fabs(d_wall);
            if (candidate < safety) safety = candidate;
        }
        // Distance to the finite slit ends (only relevant near this slit in u).
        if (std::fabs(u_mm - center_u) < 0.5F * slit_width_mm) {
            const auto d_end =
                std::fabs(std::fabs(v_mm) - slit_half_length_mm);
            if (d_end < safety) safety = d_end;
        }
    }
    if (!(safety >= 0.0F)) safety = 0.0F;
    if (safety > 1.0e29F) safety = 0.0F;
    return safety;
}

}  // namespace carbon
