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

struct MinibeamSlitInterval {
    float lower_mm, upper_mm;
};

// Navigation and material ownership must use the same represented walls.
// Subtracting the offset first gives slit zero different FP32 walls: a ray
// already on a wall can then propose a positive step that cannot move x/y.
inline MinibeamSlitInterval minibeam_slit_interval(
    const int slit, const float pitch, const float width,
    const float offset) noexcept {
    const float center = offset + static_cast<float>(slit) * pitch;
    return {center - 0.5F * width, center + 0.5F * width};
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
    const auto walls = minibeam_slit_interval(
        slit_index, slit_pitch_mm, slit_width_mm, slit_offset_mm);
    return u_mm > walls.lower_mm && u_mm < walls.upper_mm;
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
        const auto walls = minibeam_slit_interval(
            slit, slit_pitch_mm, slit_width_mm, slit_offset_mm);
        if (std::fabs(direction_u) > 1.0e-8F) {
            consider((walls.lower_mm - origin_u) /
                     direction_u);
            consider((walls.upper_mm - origin_u) /
                     direction_u);
        }
        if (std::fabs(direction_v) > 1.0e-8F) {
            consider((-slit_half_length_mm - origin_v) / direction_v);
            consider((slit_half_length_mm - origin_v) / direction_v);
        }
    }
    return best;
}

// Long air steps need analytic interval intersections. Fixed 1e-5 mm
// before/after probes can round to the same FP32 point at grazing slit walls
// and silently miss a crossing. Scan every finite slit, without spatial probes.
inline float minibeam_path_to_surface_interval(
    const float x, const float y, const float dx, const float dy,
    const float ca, const float sa, const float radius, const int slit_count,
    const float width, const float pitch, const float half_length,
    const float offset, const float max_path) noexcept {
    float best=max_path;
    auto consider=[&](float t) { if(t>1.0e-6F && t<best) best=t; };
    const float aa=dx*dx+dy*dy, bb=x*dx+y*dy, cc=x*x+y*y-radius*radius;
    float c0=-1.0e30F,c1=1.0e30F;
    if(aa>1.0e-16F) {
        const float disc=bb*bb-aa*cc;
        if(disc<0.0F) return max_path;
        const float root=std::sqrt(disc);
        c0=(-bb-root)/aa; c1=(-bb+root)/aa;
        consider(c0); consider(c1);
    } else if(cc>=0.0F) return max_path;
    const float u=ca*x+sa*y, v=-sa*x+ca*y;
    const float du=ca*dx+sa*dy, dv=-sa*dx+ca*dy;
    for(int slit=-slit_count/2;slit<=slit_count/2;++slit) {
        float enter=-1.0e30F,leave=1.0e30F;
        auto slab=[&](float p,float d,float lo,float hi) {
            if(std::fabs(d)<1.0e-12F) { if(p<lo || p>hi) leave=-1.0e30F; }
            else {
                float a=(lo-p)/d,b=(hi-p)/d;
                if(a>b) {const float tmp=a;a=b;b=tmp;}
                if(a>enter)enter=a; if(b<leave)leave=b;
            }
        };
        const auto walls = minibeam_slit_interval(slit,pitch,width,offset);
        slab(u,du,walls.lower_mm,walls.upper_mm);
        slab(v,dv,-half_length,half_length);
        if(enter<leave) {
            if(enter>=c0 && enter<=c1)consider(enter);
            if(leave>=c0 && leave<=c1)consider(leave);
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
        const auto walls = minibeam_slit_interval(
            slit, slit_pitch_mm, slit_width_mm, slit_offset_mm);
        // Distance to the represented side walls, in either material,
        // while within the finite slit length.
        const bool v_inside = v_mm > -slit_half_length_mm &&
                              v_mm < slit_half_length_mm;
        if (v_inside) {
            const auto candidate = std::fmin(std::fabs(u_mm - walls.lower_mm),
                                             std::fabs(u_mm - walls.upper_mm));
            if (candidate < safety) safety = candidate;
        }
        // Distance to the finite slit ends (only relevant near this slit in u).
        if (u_mm > walls.lower_mm && u_mm < walls.upper_mm) {
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
