#include "carbon/transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace carbon {

double TransportResult::relative_energy_balance_error() const noexcept {
    if (initial_energy_MeV == 0.0) {
        return 0.0;
    }
    return std::abs(initial_energy_MeV - total_deposited_energy_MeV - escaped_energy_MeV) /
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

TransportResult transport_serial(const TransportConfig& config,
                                 const StoppingPowerTable& stopping_power) {
    config.validate();
    const auto start = std::chrono::steady_clock::now();
    TransportResult result;
    result.backend = "serial";
    result.deposited_energy_MeV.assign(config.number_of_bins(), 0.0);

    // CSDA is deterministic. Simulate one trajectory and scale it by the number of
    // primaries; stochastic modules will switch this to true per-history transport.
    std::vector<double> one_history(config.number_of_bins(), 0.0);
    auto energy_MeV = config.initial_total_energy_MeV();
    auto position_mm = 0.0;
    std::uint64_t steps = 0;

    while (energy_MeV > config.energy_cutoff_MeV && position_mm < config.phantom_length_mm) {
        const auto energy_MeVu = energy_MeV / static_cast<double>(config.mass_number);
        const auto stopping_power_MeV_per_mm = stopping_power.interpolate(energy_MeVu);
        auto step_mm = choose_step_mm(energy_MeV,
                                      stopping_power_MeV_per_mm,
                                      config.maximum_step_mm,
                                      config.maximum_relative_energy_loss);

        const auto bin = std::min(static_cast<std::size_t>(position_mm / config.depth_bin_width_mm),
                                  config.number_of_bins() - 1);
        const auto next_bin_boundary_mm =
            static_cast<double>(bin + 1) * config.depth_bin_width_mm;
        step_mm = std::min(step_mm, next_bin_boundary_mm - position_mm);
        step_mm = std::min(step_mm, config.phantom_length_mm - position_mm);
        if (step_mm <= 0.0) {
            throw std::runtime_error("Transport stalled at a depth-bin boundary");
        }

        const auto deposited_MeV = std::min(stopping_power_MeV_per_mm * step_mm, energy_MeV);
        one_history[bin] += deposited_MeV;
        energy_MeV -= deposited_MeV;
        position_mm += step_mm;
        ++steps;
    }

    if (energy_MeV > 0.0 && position_mm < config.phantom_length_mm) {
        const auto bin = std::min(static_cast<std::size_t>(position_mm / config.depth_bin_width_mm),
                                  config.number_of_bins() - 1);
        one_history[bin] += energy_MeV;
        energy_MeV = 0.0;
    }

    const auto history_scale = static_cast<double>(config.number_of_histories);
    std::transform(one_history.begin(), one_history.end(), result.deposited_energy_MeV.begin(),
                   [history_scale](double value) { return value * history_scale; });
    result.initial_energy_MeV = config.initial_total_energy_MeV() * history_scale;
    result.total_deposited_energy_MeV =
        std::accumulate(result.deposited_energy_MeV.begin(), result.deposited_energy_MeV.end(), 0.0);
    result.escaped_energy_MeV = energy_MeV * history_scale;
    result.total_steps = steps * config.number_of_histories;
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

}  // namespace carbon
