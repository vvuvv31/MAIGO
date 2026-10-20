#include "source_pose.hpp"
#include "stopping_power.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
namespace carbon {
double propagate_total_kinetic_energy_through_stopping_power(
    const double initial_total_energy_MeV,
    const int mass_number,
    const double distance_mm,
    const StoppingPowerTable& stopping_power) {
    if (!(initial_total_energy_MeV > 0.0) || mass_number <= 0 ||
        !std::isfinite(distance_mm) || distance_mm < 0.0) {
        throw std::invalid_argument(
            "upstream stopping-power propagation requires positive energy/mass and finite nonnegative distance");
    }
    constexpr double maximum_step_mm = 0.1;
    auto energy = initial_total_energy_MeV;
    auto remaining = distance_mm;
    const auto min_energy = stopping_power.minimum_energy_MeVu();
    const auto max_energy = stopping_power.maximum_energy_MeVu();
    const auto derivative = [&](const double total_energy) {
        const auto energy_MeVu = total_energy / static_cast<double>(mass_number);
        if (energy_MeVu < min_energy || energy_MeVu > max_energy) {
            throw std::invalid_argument(
                "upstream stopping-power propagation energy is outside the supplied table");
        }
        return -stopping_power.interpolate(energy_MeVu);
    };
    while (remaining > 1.0e-12) {
        const auto step = std::min(maximum_step_mm, remaining);
        const auto k1 = derivative(energy);
        const auto k2 = derivative(energy + 0.5 * step * k1);
        const auto k3 = derivative(energy + 0.5 * step * k2);
        const auto k4 = derivative(energy + step * k3);
        energy += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        if (!(energy > 0.0) || !std::isfinite(energy)) {
            throw std::invalid_argument(
                "upstream stopping-power propagation exhausted the spot kinetic energy");
        }
        remaining -= step;
    }
    return energy;
}
double spot_entry_total_energy_after_optional_upstream_loss(
    const double initial_total_energy_MeV,
    const int mass_number,
    const double distance_mm,
    const StoppingPowerTable* const stopping_power) {
    return stopping_power == nullptr
               ? initial_total_energy_MeV
               : propagate_total_kinetic_energy_through_stopping_power(
                     initial_total_energy_MeV, mass_number, distance_mm,
                     *stopping_power);
}
SpotSourcePose transform_world_pose_to_ct(
    const SpotSourcePose& world_pose,
    const double patient_trans_x_mm,
    const double patient_trans_y_mm,
    const double patient_trans_z_mm,
    const double patient_rot_z_deg,
    const double ct_axis_min_mm) noexcept {
    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
    // TOPAS component rotations are passive: Patient local→world is R(-RotZ).
    // Therefore its world→patient inverse is R(+RotZ), not R(-RotZ).
    const auto angle = patient_rot_z_deg * deg2rad;
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);

    auto to_patient_point = [&](double x, double y, double z) {
        x -= patient_trans_x_mm;
        y -= patient_trans_y_mm;
        z -= patient_trans_z_mm;
        const auto patient_x = c * x - s * y;
        const auto patient_y = s * x + c * y;
        return std::tuple{patient_x, patient_y, z};
    };
    auto to_patient_vector = [&](const double x, const double y, const double z) {
        const auto patient_x = c * x - s * y;
        const auto patient_y = s * x + c * y;
        return std::tuple{patient_x, patient_y, z};
    };

    double ox = 0.0, oy = 0.0, oz = 0.0;
    double uxx = 0.0, uxy = 0.0, uxz = 0.0;
    double uyx = 0.0, uyy = 0.0, uyz = 0.0;
    double uzx = 0.0, uzy = 0.0, uzz = 0.0;
    std::tie(ox, oy, oz) = to_patient_point(
        world_pose.origin_x_mm, world_pose.origin_y_mm, world_pose.origin_z_mm);
    std::tie(uxx, uxy, uxz) =
        to_patient_vector(world_pose.ux_x, world_pose.ux_y, world_pose.ux_z);
    std::tie(uyx, uyy, uyz) =
        to_patient_vector(world_pose.uy_x, world_pose.uy_y, world_pose.uy_z);
    std::tie(uzx, uzy, uzz) =
        to_patient_vector(world_pose.uz_x, world_pose.uz_y, world_pose.uz_z);

    // Reoriented CT: GPU (x,y,z) stores patient (y, z_centered, x) with beam +GPU-Z.
    // Lateral: GPU x = patient y (matches reorient_ct_grid_tps_90).
    // Depth: map patient ±X into +GPU-Z so the central ray always enters z=0.
    // For a centered CT, low/high patient-X edges are ±|ct_axis_min|.
    const double ct_axis_max_mm = -ct_axis_min_mm;
    SpotSourcePose pose{};
    if (uzx >= 0.0) {
        // Beam along +patient X: entrance at patient X = ct_axis_min.
        pose.origin_x_mm = oy;
        pose.origin_y_mm = oz;
        pose.origin_z_mm = ox - ct_axis_min_mm;
        pose.ux_x = uxy;
        pose.ux_y = uxz;
        pose.ux_z = uxx;
        pose.uy_x = uyy;
        pose.uy_y = uyz;
        pose.uy_z = uyx;
        pose.uz_x = uzy;
        pose.uz_y = uzz;
        pose.uz_z = uzx;
    } else {
        // Beam along -patient X: entrance at patient X = ct_axis_max.
        // The map patient→GPU (y, z, ct_max-x) has det=-1. Negate uy so the
        // transported beam frame stays right-handed (ux·(uy×uz)=+1), matching
        // TOPAS BiGaussian axes and the +patient-X branch. Origins/uz unchanged.
        pose.origin_x_mm = oy;
        pose.origin_y_mm = oz;
        pose.origin_z_mm = ct_axis_max_mm - ox;
        pose.ux_x = uxy;
        pose.ux_y = uxz;
        pose.ux_z = -uxx;
        pose.uy_x = -uyy;
        pose.uy_y = -uyz;
        pose.uy_z = uyx;
        pose.uz_x = uzy;
        pose.uz_y = uzz;
        pose.uz_z = -uzx;
    }
    return pose;
}
}
