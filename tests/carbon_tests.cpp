#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/particle.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
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

template <typename Operation>
void require_throws(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void require_voxel_idd_closure(const carbon::TransportConfig& config,
                               const carbon::TransportResult& result,
                               double tolerance_MeV_per_primary) {
    require(result.voxel_deposited_energy_MeV.size() == config.number_of_voxels(),
            "Voxel tally has the wrong size");
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto histories = static_cast<double>(config.number_of_histories);
    for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
        const auto begin = result.voxel_deposited_energy_MeV.begin() +
                           static_cast<std::ptrdiff_t>(z * plane_size);
        const auto reconstructed =
            std::accumulate(begin, begin + static_cast<std::ptrdiff_t>(plane_size), 0.0);
        require_near(reconstructed / histories,
                     result.deposited_energy_MeV[z] / histories,
                     tolerance_MeV_per_primary,
                     "Voxel x/y sum does not close to IDD at z bin " +
                         std::to_string(z));
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
    require(config.number_of_voxels() == 2'880'000, "Voxel count failed");
    config.enable_secondary_transport = true;
    require_throws([&config]() { config.validate(); },
                   "Secondary transport without generation was accepted");
    config.enable_secondary_transport = false;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 0;
    require_throws([&config]() { config.validate(); },
                   "Enabled voxel scorer accepted a zero x-bin count");
}

void test_serial_voxel_idd_closure() {
    carbon::TransportConfig config;
    config.number_of_histories = 7;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 5;
    config.voxel_bins_y = 7;
    const carbon::StoppingPowerTable table({0.01, 20.0}, {2.0, 2.0});
    const auto result = carbon::transport_serial(config, table, zero_cross_section());
    require_voxel_idd_closure(config, result, 1.0e-12);
}

void test_charged_dose_categories() {
    require(carbon::charged_dose_category(6, 12) == 0, "Carbon category failed");
    require(carbon::charged_dose_category(5, 11) == 1, "Boron category failed");
    require(carbon::charged_dose_category(4, 9) == 2, "Beryllium category failed");
    require(carbon::charged_dose_category(3, 7) == 3, "Lithium category failed");
    require(carbon::charged_dose_category(2, 4) == 4, "Helium category failed");
    require(carbon::charged_dose_category(1, 1) == 5, "Proton category failed");
    require(carbon::charged_dose_category(1, 2) == 6, "Deuteron category failed");
    require(carbon::charged_dose_category(7, 14) == 6, "Other charged category failed");
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

void test_fragment_stopping_power_scale() {
    require_near(carbon::stopping_power_scale_from_carbon(6, 200.0), 1.0, 1.0e-12,
                 "C-12 stopping-power scale failed");
    const auto proton_scale = carbon::stopping_power_scale_from_carbon(1, 200.0);
    const auto helium_scale = carbon::stopping_power_scale_from_carbon(2, 200.0);
    const auto boron_scale = carbon::stopping_power_scale_from_carbon(5, 200.0);
    require(proton_scale > 0.0 && proton_scale < helium_scale && helium_scale < boron_scale &&
                boron_scale < 1.0,
            "Fragment stopping-power charge ordering failed");
    require(carbon::stopping_power_scale_from_carbon(1, 1.0) > proton_scale,
            "Low-energy effective-charge scaling failed");
    require_throws([]() { (void)carbon::stopping_power_scale_from_carbon(0, 10.0); },
                   "Invalid fragment atomic number was accepted");
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

void test_reaction_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory / "validation/results/topas_200MeVu_reaction_packages_development.bin";
    const auto table = carbon::ReactionPackageTable::from_binary(package_path);
    require(table.energy_bins().size() == 201, "Reaction package energy-bin count failed");
    require(table.reactions().size() == 37'657, "Reaction package reaction count failed");
    require(table.secondaries().size() == 330'659,
            "Reaction package secondary count failed");
    require_near(table.minimum_energy_MeV_per_u(), 0.0, 1.0e-7,
                 "Reaction package minimum energy failed");
    require_near(table.energy_bin_width_MeV_per_u(), 1.0, 1.0e-7,
                 "Reaction package energy-bin width failed");
    require(table.energy_bin_index(-1.0F) == 0,
            "Reaction package low-energy clamp failed");
    require(table.energy_bin_index(200.0F) == 200,
            "Reaction package exact energy-bin lookup failed");
    require(table.energy_bin_index(500.0F) == 200,
            "Reaction package high-energy clamp failed");
    require(table.energy_bin_index(std::numeric_limits<float>::quiet_NaN()) == 0,
            "Reaction package non-finite energy handling failed");

    std::uint64_t reactions_from_bins = 0;
    for (const auto& bin : table.energy_bins()) {
        require(bin.reaction_count > 0, "Reaction package contains an empty energy bin");
        reactions_from_bins += bin.reaction_count;
    }
    require(reactions_from_bins == table.reactions().size(),
            "Reaction package energy-bin closure failed");

    std::uint64_t secondaries_from_reactions = 0;
    std::size_t empty_reactions = 0;
    for (const auto& reaction : table.reactions()) {
        secondaries_from_reactions += reaction.secondary_count;
        empty_reactions += reaction.secondary_count == 0 ? 1U : 0U;
    }
    require(secondaries_from_reactions == table.secondaries().size(),
            "Reaction package secondary closure failed");
    require(empty_reactions == 2, "Reaction package zero-secondary count failed");

    std::size_t protons = 0;
    std::size_t neutrons = 0;
    std::size_t gammas = 0;
    std::size_t alphas = 0;
    for (const auto& secondary : table.secondaries()) {
        protons += secondary.pdg_id == 2212 ? 1U : 0U;
        neutrons += secondary.pdg_id == 2112 ? 1U : 0U;
        gammas += secondary.pdg_id == 22 ? 1U : 0U;
        alphas += secondary.atomic_number == 2 && secondary.mass_number == 4 ? 1U : 0U;
    }
    require(protons == 93'437 && neutrons == 80'666 && gammas == 38'733 &&
                alphas == 56'094,
            "Reaction package particle composition failed");

    const auto invalid_path =
        std::filesystem::temp_directory_path() / "carbon_invalid_reaction_package.bin";
    {
        std::ofstream invalid(invalid_path, std::ios::binary | std::ios::trunc);
        invalid << "not a reaction package";
    }
    require_throws(
        [&invalid_path]() { (void)carbon::ReactionPackageTable::from_binary(invalid_path); },
        "Invalid reaction package was accepted");
    std::filesystem::remove(invalid_path);
}

void test_cascade_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory / "validation/results/topas_200MeVu_cascade_smoke.bin";
    const auto table = carbon::CascadePackageTable::from_binary(package_path);
    require(table.projectiles().size() == 11, "Cascade projectile count failed");
    require(table.cross_sections().size() == 252, "Cascade cross-section count failed");
    require(table.interactions().size() == 58, "Cascade interaction count failed");
    require(table.products().size() == 399, "Cascade product count failed");
    const auto* alpha = table.find_projectile(2, 4);
    require(alpha != nullptr && alpha->interaction_count == 9,
            "Cascade alpha lookup failed");
    require(table.find_projectile(5, 12) == nullptr,
            "Cascade missing-projectile lookup failed");
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
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 5;
    config.voxel_bins_y = 7;
    const carbon::StoppingPowerTable table({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto serial = carbon::transport_serial(config, table, zero_cross_section());
    const auto sycl_cpu =
        carbon::transport_sycl(config, table, zero_cross_section(), "cpu");
    require_voxel_idd_closure(config, serial, 1.0e-12);
    require_voxel_idd_closure(config, sycl_cpu, 1.0e-9);
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

void test_sycl_secondary_queue_generation() {
    const auto package_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                              "validation/results/"
                              "topas_200MeVu_reaction_packages_development.bin";
    const auto reaction_packages = carbon::ReactionPackageTable::from_binary(package_path);
    carbon::TransportConfig config;
    config.number_of_histories = 64;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_primary_attenuation = true;
    config.enable_secondary_generation = true;
    config.secondary_queue_capacity = 10'000;
    config.random_seed = 31415926;
    const carbon::StoppingPowerTable stopping_power(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const carbon::CrossSectionTable forced_reaction(
        {0.01, 20.01}, {100.0, 100.0});

    const auto first = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    const auto second = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    require(first.nuclear_interactions == config.number_of_histories,
            "Secondary-generation test did not force every reaction");
    require(first.sampled_reaction_packages == first.nuclear_interactions,
            "Not every nuclear interaction sampled a reaction package");
    require(first.generated_direct_secondaries > first.queued_secondaries,
            "Direct-secondary classification did not retain untransported particles");
    require(first.queued_secondaries > 0 && first.queued_secondary_energy_MeV > 0.0,
            "Secondary queue remained empty");
    require(first.secondary_queue_overflow == 0 &&
                first.secondary_queue_overflow_energy_MeV == 0.0,
            "Unexpected secondary queue overflow");
    require(first.untransported_neutral_energy_MeV > 0.0,
            "Untransported neutral energy was not recorded");
    require(first.untransported_unsupported_charged_energy_MeV == 0.0,
            "A charged ion was not accepted by the generic secondary queue");
    require_near(
        first.generated_direct_secondary_energy_MeV,
        first.queued_secondary_energy_MeV + first.untransported_neutral_energy_MeV,
        1.0e-5, "Direct-secondary energy category closure failed");
    require(first.relative_energy_balance_error() < 1.0e-4,
            "Secondary generation changed primary energy accounting");
    require(first.generated_direct_secondaries == second.generated_direct_secondaries &&
                first.queued_secondaries == second.queued_secondaries &&
                first.queued_secondary_energy_MeV == second.queued_secondary_energy_MeV &&
                first.untransported_neutral_energy_MeV ==
                    second.untransported_neutral_energy_MeV,
            "Secondary generation was not deterministic");

    config.secondary_queue_capacity = 1;
    const auto overflow = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    require(overflow.secondary_queue_overflow > 0 &&
                overflow.secondary_queue_overflow_energy_MeV > 0.0,
            "A one-particle queue did not report whole-package overflow");
    require(overflow.queued_secondaries + overflow.secondary_queue_overflow <=
                overflow.generated_direct_secondaries,
            "Secondary queue accounting exceeded direct-secondary production");

    config.secondary_queue_capacity = 10'000;
    config.enable_secondary_transport = true;
    const auto transported = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    require(transported.transported_secondaries == transported.queued_secondaries &&
                transported.transported_secondaries > 0,
            "Not every queued charged secondary was transported");
    require_near(
        transported.secondary_deposited_energy_MeV +
            transported.secondary_escaped_energy_MeV,
        transported.queued_secondary_energy_MeV,
        1.0e-2, "Secondary transport energy closure failed");
    require(transported.secondary_deposited_energy_MeV > 0.0 &&
                transported.secondary_escaped_energy_MeV > 0.0 &&
                transported.secondary_transport_steps > 0,
            "Secondary transport did not record deposition, escape, and steps");
    require(transported.relative_energy_balance_error() < 1.0e-4,
            "Secondary transport total energy balance failed");
    const std::vector<const std::vector<double>*> species{
        &transported.secondary_carbon_deposited_energy_MeV,
        &transported.boron_deposited_energy_MeV,
        &transported.beryllium_deposited_energy_MeV,
        &transported.lithium_deposited_energy_MeV,
        &transported.helium_deposited_energy_MeV,
        &transported.proton_deposited_energy_MeV,
        &transported.other_charged_deposited_energy_MeV,
    };
    require(transported.primary_c12_deposited_energy_MeV.size() ==
                config.number_of_bins(),
            "Primary species tally has the wrong size");
    double species_deposited_energy = 0.0;
    for (const auto* tally : species) {
        require(tally->size() == config.number_of_bins(),
                "Fragment species tally has the wrong size");
        species_deposited_energy =
            std::accumulate(tally->begin(), tally->end(), species_deposited_energy);
    }
    require_near(species_deposited_energy, transported.secondary_deposited_energy_MeV,
                 1.0e-2, "Fragment species tally energy closure failed");
    for (std::size_t bin = 0; bin < config.number_of_bins(); ++bin) {
        auto reconstructed = transported.primary_c12_deposited_energy_MeV[bin];
        for (const auto* tally : species) {
            reconstructed += (*tally)[bin];
        }
        require_near(reconstructed, transported.deposited_energy_MeV[bin], 1.0e-9,
                     "Total dose bin does not close over species");
    }

    const auto cascade_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                              "validation/results/topas_200MeVu_cascade_smoke.bin";
    const auto cascade_packages = carbon::CascadePackageTable::from_binary(cascade_path);
    config.enable_fragment_cascade = true;
    config.maximum_cascade_generations = 1;
    config.secondary_queue_capacity = 100'000;
    const auto cascaded = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages,
        &cascade_packages);
    require(cascaded.cascade_interactions > 0 &&
                cascaded.generated_cascade_products > 0 &&
                cascaded.queued_cascade_secondaries > 0,
            "Fragment cascade did not generate a second interaction generation");
    require(cascaded.cascade_queue_overflow == 0,
            "Unexpected fragment cascade queue overflow");
    require(cascaded.transported_secondaries ==
                cascaded.queued_secondaries + cascaded.queued_cascade_secondaries,
            "Cascade queue generation count did not close");
    require(cascaded.relative_energy_balance_error() < 1.0e-4,
            "Fragment cascade total energy balance failed");
}
#endif

}  // namespace

int main() {
    try {
        test_units();
        test_serial_voxel_idd_closure();
        test_charged_dose_categories();
        test_interpolation();
        test_fragment_stopping_power_scale();
        test_step_selection();
        test_philox_rng();
        test_bohr_straggling();
        test_energy_conservation();
        test_escape_energy_conservation();
        test_straggling_reproducibility();
        test_primary_attenuation_energy_accounting();
        test_reaction_package_loading();
        test_cascade_package_loading();
#ifdef CARBON_HAS_SYCL
        test_serial_sycl_cpu_match();
        test_sycl_secondary_queue_generation();
#endif
        std::cout << "All carbon_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_tests: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
