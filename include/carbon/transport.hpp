#pragma once

#include "carbon/stopping_power.hpp"
#include "carbon/transport_config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace carbon {

struct TransportResult {
    std::vector<double> deposited_energy_MeV;
    double initial_energy_MeV{0.0};
    double total_deposited_energy_MeV{0.0};
    double escaped_energy_MeV{0.0};
    std::uint64_t total_steps{0};
    double elapsed_seconds{0.0};
    std::string backend;

    [[nodiscard]] double relative_energy_balance_error() const noexcept;
};

[[nodiscard]] double choose_step_mm(double energy_MeV,
                                    double stopping_power_MeV_per_mm,
                                    double maximum_step_mm,
                                    double maximum_relative_energy_loss);

TransportResult transport_serial(const TransportConfig& config,
                                 const StoppingPowerTable& stopping_power);

#ifdef CARBON_HAS_SYCL
TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const std::string& device_name);
std::string describe_sycl_device(const std::string& device_name);
#endif

}  // namespace carbon
