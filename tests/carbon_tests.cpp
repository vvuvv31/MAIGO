#include "carbon/cross_section.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

const carbon::CrossSectionTable& zero_cross_section() {
    static const carbon::CrossSectionTable table({0.01, 400.0}, {0.0, 0.0});
    return table;
}

void test_units() {
    carbon::TransportConfig config;
    config.initial_energy_MeVu = 200.0;
    config.mass_number = 12;
    require_near(config.initial_total_energy_MeV(), 2400.0, 1.0e-12,
                 "MeV/u to total kinetic energy conversion failed");
    require(config.number_of_bins() == 800, "Depth-bin count failed");
}

void test_interpolation() {
    const carbon::StoppingPowerTable table({1.0, 2.0, 3.0}, {12.0, 8.0, 6.0});
    require_near(table.interpolate(1.5), 10.0, 1.0e-12, "Linear interpolation failed");
    require_near(table.interpolate(0.1), 12.0, 1.0e-12, "Low-energy clamp failed");
    require_near(table.interpolate(9.0), 6.0, 1.0e-12, "High-energy clamp failed");

    const carbon::CrossSectionTable cross_section({1.0, 2.0, 3.0}, {0.01, 0.02, 0.04});
    require_near(cross_section.interpolate(2.5), 0.03, 1.0e-12,
                 "Cross-section interpolation failed");
    require_near(cross_section.interpolate(0.1), 0.01, 1.0e-12,
                 "Cross-section low-energy clamp failed");
}

void test_step_selection() {
    require_near(carbon::choose_step_mm(100.0, 10.0, 0.5, 0.01), 0.1, 1.0e-12,
                 "Energy-limited step failed");
    require_near(carbon::choose_step_mm(1000.0, 1.0, 0.5, 0.01), 0.5, 1.0e-12,
                 "Maximum-step limit failed");
}

void test_philox_rng() {
    const auto block = carbon::rng::philox4x32_10({0U, 0U, 0U, 0U}, 0U, 0U);
    require(block[0] == 0x6627E8D5U && block[1] == 0xE169C58DU &&
                block[2] == 0xBC57AC4CU && block[3] == 0x9B00DBD8U,
            "Philox4x32-10 reference vector failed");

    constexpr std::size_t samples = 100'000;
    double sum = 0.0;
    double squared_sum = 0.0;
    for (std::size_t index = 0; index < samples; ++index) {
        const auto value = static_cast<double>(carbon::rng::uniform01(1234, index, 0, 0));
        require(value > 0.0 && value < 1.0, "Uniform RNG left the open unit interval");
        sum += value;
        squared_sum += value * value;
    }
    const auto mean = sum / static_cast<double>(samples);
    const auto variance = squared_sum / static_cast<double>(samples) - mean * mean;
    require_near(mean, 0.5, 0.003, "Uniform RNG mean failed");
    require_near(variance, 1.0 / 12.0, 0.001, "Uniform RNG variance failed");
}

void test_bohr_straggling() {
    const auto sigma_half_mm = carbon::bohr_straggling_sigma_MeV(200.0, 0.5, 1.0);
    const auto sigma_two_mm = carbon::bohr_straggling_sigma_MeV(200.0, 2.0, 1.0);
    require(sigma_half_mm > 0.0, "Bohr straggling sigma must be positive");
    require_near(sigma_two_mm / sigma_half_mm, 2.0, 1.0e-12,
                 "Bohr sigma must scale with sqrt(step length)");
    require_near(carbon::clamp_sampled_energy_loss(1.0, 2.0, -2.0, 10.0), 0.0,
                 1.0e-12, "Negative sampled loss clamp failed");
    require_near(carbon::clamp_sampled_energy_loss(9.0, 2.0, 2.0, 10.0), 10.0,
                 1.0e-12, "Available-energy clamp failed");
}

void test_energy_conservation() {
    carbon::TransportConfig config;
    config.number_of_histories = 7;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 1000.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    const carbon::StoppingPowerTable table({0.01, 20.0}, {2.0, 2.0});
    const auto result = carbon::transport_serial(config, table, zero_cross_section());
    require(result.relative_energy_balance_error() < 1.0e-12,
            "Stopped-particle energy balance failed");
    require_near(result.escaped_energy_MeV, 0.0, 1.0e-12, "Unexpected escape energy");
}

void test_escape_energy_conservation() {
    carbon::TransportConfig config;
    config.number_of_histories = 3;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 1.0;
    config.depth_bin_width_mm = 0.5;
    config.maximum_step_mm = 0.25;
    const carbon::StoppingPowerTable table({0.01, 20.0}, {1.0, 1.0});
    const auto result = carbon::transport_serial(config, table, zero_cross_section());
    require(result.escaped_energy_MeV > 0.0, "Expected nonzero escape energy");
    require(result.relative_energy_balance_error() < 1.0e-12,
            "Escaping-particle energy balance failed");
}

void test_straggling_reproducibility() {
    carbon::TransportConfig config;
    config.number_of_histories = 32;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_energy_straggling = true;
    config.random_seed = 987654321;
    const carbon::StoppingPowerTable table({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto first = carbon::transport_serial(config, table, zero_cross_section());
    const auto second = carbon::transport_serial(config, table, zero_cross_section());
    require(first.deposited_energy_MeV == second.deposited_energy_MeV,
            "Straggling run was not exactly reproducible");
    require(first.relative_energy_balance_error() < 1.0e-12,
            "Straggling energy balance failed");
}

void test_primary_attenuation_energy_accounting() {
    carbon::TransportConfig config;
    config.number_of_histories = 128;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_primary_attenuation = true;
    const carbon::StoppingPowerTable table({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const carbon::CrossSectionTable high_cross_section({0.01, 20.01}, {100.0, 100.0});
    const auto result = carbon::transport_serial(config, table, high_cross_section);
    require(result.nuclear_interactions == config.number_of_histories,
            "High-cross-section attenuation did not terminate every primary");
    require(result.untracked_nuclear_energy_MeV > 0.0,
            "Nuclear interaction energy was not accounted separately");
    require(result.relative_energy_balance_error() < 1.0e-12,
            "Primary attenuation energy balance failed");

    const auto no_attenuation = carbon::transport_serial(config, table, zero_cross_section());
    require(no_attenuation.nuclear_interactions == 0,
            "Zero cross section produced a nuclear interaction");
    require(no_attenuation.relative_energy_balance_error() < 1.0e-12,
            "Zero-cross-section energy balance failed");
}

#ifdef CARBON_HAS_SYCL
void test_serial_sycl_cpu_match() {
    carbon::TransportConfig config;
    config.number_of_histories = 64;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    const carbon::StoppingPowerTable table({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto serial = carbon::transport_serial(config, table, zero_cross_section());
    const auto sycl_cpu =
        carbon::transport_sycl(config, table, zero_cross_section(), "cpu");
    require(sycl_cpu.relative_energy_balance_error() < 1.0e-4,
            "SYCL CPU energy balance failed");

    double absolute_difference = 0.0;
    for (std::size_t bin = 0; bin < serial.deposited_energy_MeV.size(); ++bin) {
        absolute_difference +=
            std::abs(serial.deposited_energy_MeV[bin] - sycl_cpu.deposited_energy_MeV[bin]);
    }
    require(absolute_difference / serial.total_deposited_energy_MeV < 1.0e-4,
            "Serial/SYCL CPU dose tally mismatch");

    config.enable_energy_straggling = true;
    config.random_seed = 42;
    const auto serial_straggling =
        carbon::transport_serial(config, table, zero_cross_section());
    const auto sycl_straggling =
        carbon::transport_sycl(config, table, zero_cross_section(), "cpu");
    absolute_difference = 0.0;
    for (std::size_t bin = 0; bin < serial_straggling.deposited_energy_MeV.size(); ++bin) {
        absolute_difference += std::abs(serial_straggling.deposited_energy_MeV[bin] -
                                        sycl_straggling.deposited_energy_MeV[bin]);
    }
    const auto relative_tally_difference =
        absolute_difference / serial_straggling.total_deposited_energy_MeV;
    require(relative_tally_difference < 5.0e-3,
            "Serial/SYCL CPU straggling tally mismatch: relative L1=" +
                std::to_string(relative_tally_difference));

    config.enable_primary_attenuation = true;
    const carbon::CrossSectionTable attenuation_cross_section(
        {0.01, 20.01}, {0.01, 0.01});
    const auto serial_attenuation =
        carbon::transport_serial(config, table, attenuation_cross_section);
    const auto sycl_attenuation =
        carbon::transport_sycl(config, table, attenuation_cross_section, "cpu");
    require(serial_attenuation.nuclear_interactions == sycl_attenuation.nuclear_interactions,
            "Serial/SYCL CPU nuclear interaction count mismatch");
    require(sycl_attenuation.relative_energy_balance_error() < 1.0e-4,
            "SYCL attenuation energy balance failed");
}
#endif

}  // namespace

int main() {
    try {
        test_units();
        test_interpolation();
        test_step_selection();
        test_philox_rng();
        test_bohr_straggling();
        test_energy_conservation();
        test_escape_energy_conservation();
        test_straggling_reproducibility();
        test_primary_attenuation_energy_accounting();
#ifdef CARBON_HAS_SYCL
        test_serial_sycl_cpu_match();
#endif
        std::cout << "All carbon_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_tests: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
