#pragma once

#include "carbon/transport_config.hpp"

#include <cstddef>
#include <filesystem>
#include <limits>
#include <vector>

namespace carbon {

struct TpsSpot {
    int spot_id{0};
    double energy_MeVu{0.0};
    double x_mm{0.0};
    double y_mm{0.0};
    double mu_weight{1.0};
    // NaN means inherit the corresponding TransportConfig source value.
    double energy_spread_percent{std::numeric_limits<double>::quiet_NaN()};
    double sigma_x_mm{std::numeric_limits<double>::quiet_NaN()};
    double sigma_y_mm{std::numeric_limits<double>::quiet_NaN()};
    double sigma_x_prime{std::numeric_limits<double>::quiet_NaN()};
    double sigma_y_prime{std::numeric_limits<double>::quiet_NaN()};
    double correlation_x{std::numeric_limits<double>::quiet_NaN()};
    double correlation_y{std::numeric_limits<double>::quiet_NaN()};
};

struct TpsSourcePose {
    double origin_x_mm{0.0}, origin_y_mm{0.0}, origin_z_mm{0.0};
    double ux_x{1.0}, ux_y{0.0}, ux_z{0.0};
    double uy_x{0.0}, uy_y{1.0}, uy_z{0.0};
    double uz_x{0.0}, uz_y{0.0}, uz_z{-1.0};
};

struct TpsSourcePlan {
    std::vector<TpsSpot> spots{};
    double total_mu{0.0};

    [[nodiscard]] static TpsSourcePlan from_config(const TransportConfig& config);
    [[nodiscard]] static TpsSourcePlan from_csv(const std::filesystem::path& path);
    [[nodiscard]] TpsSourcePose pose_for_spot(const TransportConfig& config,
                                              const TpsSpot& spot) const;
    [[nodiscard]] std::vector<std::size_t> allocate_histories(
        std::size_t total_histories) const;
    [[nodiscard]] std::vector<PrimarySpotBatchEntry> make_primary_batch(
        const TransportConfig& config) const;
    [[nodiscard]] std::size_t active_spot_count() const noexcept;
};

}  // namespace carbon
