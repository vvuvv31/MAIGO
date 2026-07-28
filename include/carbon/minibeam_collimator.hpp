#pragma once

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
    int& slit_index) noexcept {
    const auto u_mm = cosine_angle * x_mm + sine_angle * y_mm;
    const auto v_mm = -sine_angle * x_mm + cosine_angle * y_mm;
    if (u_mm * u_mm + v_mm * v_mm >= radius_mm * radius_mm ||
        (v_mm <= -slit_half_length_mm || v_mm >= slit_half_length_mm)) {
        return false;
    }
    slit_index = nearest_minibeam_slit(u_mm, slit_pitch_mm);
    const auto half_count = slit_count / 2;
    if (slit_index < -half_count || slit_index > half_count) {
        return false;
    }
    const auto delta = u_mm - static_cast<float>(slit_index) * slit_pitch_mm;
    return delta > -0.5F * slit_width_mm && delta < 0.5F * slit_width_mm;
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
    const float slit_half_length_mm) noexcept {
    if (x_mm * x_mm + y_mm * y_mm >= radius_mm * radius_mm) {
        return false;
    }
    auto slit_index = 0;
    return !minibeam_point_in_slit(
        x_mm, y_mm, cosine_angle, sine_angle, radius_mm, slit_count,
        slit_width_mm, slit_pitch_mm, slit_half_length_mm, slit_index);
}

// True only when the unscattered ray stays inside the same air slit over the
// complete copper thickness. The slit and cylinder cross-sections are convex,
// so checking both endpoints is sufficient for a straight ray.
inline bool minibeam_straight_through_air_slit(
    const float position_x_mm,
    const float position_y_mm,
    const float position_z_mm,
    const float direction_x,
    const float direction_y,
    const float direction_z,
    const float cosine_angle,
    const float sine_angle,
    const float radius_mm,
    const float thickness_mm,
    const float exit_to_phantom_mm,
    const int slit_count,
    const float slit_width_mm,
    const float slit_pitch_mm,
    const float slit_half_length_mm) noexcept {
    if (direction_z <= 1.0e-8F) {
        return false;
    }
    const auto entrance_z_mm = -(exit_to_phantom_mm + thickness_mm);
    const auto exit_z_mm = -exit_to_phantom_mm;
    const auto entrance_distance = (entrance_z_mm - position_z_mm) / direction_z;
    const auto exit_distance = (exit_z_mm - position_z_mm) / direction_z;
    if (entrance_distance < 0.0F || exit_distance <= entrance_distance) {
        return false;
    }

    const auto entrance_x = position_x_mm + entrance_distance * direction_x;
    const auto entrance_y = position_y_mm + entrance_distance * direction_y;
    const auto exit_x = position_x_mm + exit_distance * direction_x;
    const auto exit_y = position_y_mm + exit_distance * direction_y;
    auto entrance_slit = 0;
    auto exit_slit = 0;
    return minibeam_point_in_slit(
               entrance_x, entrance_y, cosine_angle, sine_angle, radius_mm,
               slit_count, slit_width_mm, slit_pitch_mm, slit_half_length_mm,
               entrance_slit) &&
           minibeam_point_in_slit(
               exit_x, exit_y, cosine_angle, sine_angle, radius_mm, slit_count,
               slit_width_mm, slit_pitch_mm, slit_half_length_mm, exit_slit) &&
           entrance_slit == exit_slit;
}

}  // namespace carbon
