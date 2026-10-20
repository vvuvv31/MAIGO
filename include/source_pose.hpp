#pragma once
namespace carbon {
class StoppingPowerTable;
struct SpotSourcePose {
    double origin_x_mm{0.0};
    double origin_y_mm{0.0};
    double origin_z_mm{0.0};
    // Orthonormal beam frame: particles leave along +uz; emittance in ux, uy.
    double ux_x{1.0}, ux_y{0.0}, ux_z{0.0};
    double uy_x{0.0}, uy_y{1.0}, uy_z{0.0};
    double uz_x{0.0}, uz_y{0.0}, uz_z{1.0};
};
[[nodiscard]] double propagate_total_kinetic_energy_through_stopping_power(
    double initial_total_energy_MeV,
    int mass_number,
    double distance_mm,
    const StoppingPowerTable& stopping_power);
[[nodiscard]] double spot_entry_total_energy_after_optional_upstream_loss(
    double initial_total_energy_MeV,
    int mass_number,
    double distance_mm,
    const StoppingPowerTable* stopping_power);
[[nodiscard]] SpotSourcePose transform_world_pose_to_ct(
    const SpotSourcePose& world_pose,
    double patient_trans_x_mm,
    double patient_trans_y_mm,
    double patient_trans_z_mm,
    double patient_rot_z_deg,
    double ct_axis_min_mm) noexcept;
}
