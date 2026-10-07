#include "carbon/transport.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace carbon {

double TransportResult::physical_relative_energy_balance_error() const noexcept {
    if (initial_energy_MeV == 0.0) {
        return 0.0;
    }
    return std::abs(initial_energy_MeV + 
                    electron_variance_reduction_weight_gain_MeV - total_deposited_energy_MeV -
                    escaped_energy_MeV - beamline_removed_energy_MeV -
                    untracked_nuclear_energy_MeV -
                    primary_other_terminal_kinetic_MeV - material_electron_untracked_MeV) /
           initial_energy_MeV;
}

double choose_step_mm(double energy_MeV,
                      double stopping_power_MeV_per_mm,
                      double maximum_step_mm,
                      double maximum_relative_energy_loss) {
    if (energy_MeV <= 0.0 || stopping_power_MeV_per_mm <= 0.0 || maximum_step_mm <= 0.0 ||
        maximum_relative_energy_loss <= 0.0) {
        throw std::invalid_argument("choose_step_mm received a nonpositive physical input");
    }
    const auto energy_limited_step =
        maximum_relative_energy_loss * energy_MeV / stopping_power_MeV_per_mm;
    return std::min(maximum_step_mm, energy_limited_step);
}

std::vector<double> compute_idd_from_3d_voxel_dose(
    const std::vector<double>& voxel_dose_MeV,
    std::size_t nx, std::size_t ny, std::size_t nz) {
    if (voxel_dose_MeV.size() < nx * ny * nz) {
        throw std::invalid_argument("compute_idd_from_3d_voxel_dose: voxel_dose size mismatch");
    }
    std::vector<double> idd(nz, 0.0);
    const std::size_t slice_size = nx * ny;
    for (std::size_t z = 0; z < nz; ++z) {
        double sum = 0.0;
        const std::size_t z_offset = z * slice_size;
        for (std::size_t xy = 0; xy < slice_size; ++xy) {
            sum += voxel_dose_MeV[z_offset + xy];
        }
        idd[z] = sum;
    }
    return idd;
}

BraggPeakMetrics compute_bragg_peak_metrics(
    const std::vector<double>& idd_energy_MeV,
    double bin_width_z_mm,
    double z_min_mm) {
    BraggPeakMetrics metrics{};
    if (idd_energy_MeV.empty() || bin_width_z_mm <= 0.0) {
        return metrics;
    }
    std::size_t max_bin = 0;
    double max_val = idd_energy_MeV[0];
    for (std::size_t i = 1; i < idd_energy_MeV.size(); ++i) {
        if (idd_energy_MeV[i] > max_val) {
            max_val = idd_energy_MeV[i];
            max_bin = i;
        }
    }
    metrics.peak_depth_mm = z_min_mm + (static_cast<double>(max_bin) + 0.5) * bin_width_z_mm;
    metrics.peak_dose_MeV = max_val;

    if (max_val <= 0.0) {
        return metrics;
    }

    const double target_80 = 0.8 * max_val;
    const double target_50 = 0.5 * max_val;

    // Distal falloff search (first downward crossing after peak)
    for (std::size_t i = max_bin; i + 1 < idd_energy_MeV.size(); ++i) {
        const double v0 = idd_energy_MeV[i];
        const double v1 = idd_energy_MeV[i + 1];
        const double z0 = z_min_mm + (static_cast<double>(i) + 0.5) * bin_width_z_mm;
        const double z1 = z_min_mm + (static_cast<double>(i + 1) + 0.5) * bin_width_z_mm;

        if (!metrics.found_r80 && v0 >= target_80 && v1 <= target_80 && v0 != v1) {
            const double frac = (v0 - target_80) / (v0 - v1);
            metrics.r80_distal_mm = z0 + frac * (z1 - z0);
            metrics.found_r80 = true;
        }
        if (!metrics.found_r50 && v0 >= target_50 && v1 <= target_50 && v0 != v1) {
            const double frac = (v0 - target_50) / (v0 - v1);
            metrics.r50_distal_mm = z0 + frac * (z1 - z0);
            metrics.found_r50 = true;
        }
    }
    return metrics;
}

}  // namespace carbon
