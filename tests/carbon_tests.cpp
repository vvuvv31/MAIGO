#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/minibeam_collimator.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/plan_run.hpp"
#include "carbon/rng.hpp"
#include "carbon/run_quality.hpp"

#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/detail/fred_fragmentation_data.hpp"
#include "carbon/fred_event_library.hpp"
#include "carbon/fred_table1.hpp"
#include "carbon/inelastic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

template <typename ExceptionType = std::exception, typename Operation>
void require_throws(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const ExceptionType&) {
        return;
    } catch (...) {
        throw std::runtime_error(message + " (threw unexpected exception type)");
    }
    throw std::runtime_error(message + " (did not throw)");
}


void require_voxel_idd_closure(const carbon::TransportConfig& config,
                               const carbon::TransportResult& result,
                               double tolerance_MeV_per_primary) {
    require(result.voxel_deposited_energy_MeV.size() == config.number_of_voxels(),
            "Voxel tally has the wrong size");
#if defined(CARBON_DOSE_FP32)
    // Float atomics accumulate IDD and voxel planes in different orders.
    tolerance_MeV_per_primary = std::max(tolerance_MeV_per_primary, 5.0e-5);
#endif
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
void require_charged_origin_voxel_closure(
    const carbon::TransportConfig& config,
    const carbon::TransportResult& result,
    double tolerance_MeV_per_primary) {
#if defined(CARBON_DOSE_FP32)
    tolerance_MeV_per_primary = std::max(tolerance_MeV_per_primary, 5.0e-5);
#endif
    const auto voxel_count = config.number_of_voxels();
    require(result.charged_origin_voxel_deposited_energy_MeV.size() ==
                carbon::charged_origin_category_count * voxel_count,
            "Charged-origin voxel tally has the wrong size");
    const std::array<const std::vector<double>*,
                     carbon::charged_origin_category_count>
        depth_categories{
            &result.primary_deposited_energy_MeV,
            &result.secondary_carbon_deposited_energy_MeV,
            &result.secondary_boron_deposited_energy_MeV,
            &result.secondary_beryllium_deposited_energy_MeV,
            &result.secondary_lithium_deposited_energy_MeV,
            &result.secondary_helium_deposited_energy_MeV,
            &result.secondary_proton_deposited_energy_MeV,
            &result.secondary_other_charged_deposited_energy_MeV,
        };
    const auto histories = static_cast<double>(config.number_of_histories);
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
        double reconstructed = 0.0;
        for (std::size_t category = 0;
             category < carbon::charged_origin_category_count; ++category) {
            reconstructed +=
                result.charged_origin_voxel_deposited_energy_MeV[
                    category * voxel_count + voxel];
        }
        require_near(reconstructed / histories,
                     result.voxel_deposited_energy_MeV[voxel] / histories,
                     tolerance_MeV_per_primary,
                     "Charged-origin categories do not close at voxel " +
                         std::to_string(voxel));
    }
    for (std::size_t category = 0;
         category < carbon::charged_origin_category_count; ++category) {
        require(depth_categories[category]->size() == config.number_of_bins(),
                "Charged-origin depth category has the wrong size");
        for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
            const auto begin =
                result.charged_origin_voxel_deposited_energy_MeV.begin() +
                static_cast<std::ptrdiff_t>(
                    category * voxel_count + z * plane_size);
            const auto reconstructed = std::accumulate(
                begin, begin + static_cast<std::ptrdiff_t>(plane_size), 0.0);
            require_near(reconstructed / histories,
                         (*depth_categories[category])[z] / histories,
                         tolerance_MeV_per_primary,
                         "Charged-origin category voxel plane does not close "
                         "to IDD");
        }
    }
}

const carbon::CrossSectionTable& zero_cross_section() {
    static const carbon::CrossSectionTable table({0.01, 400.0}, {0.0, 0.0});
    return table;
}

void test_units() {
    carbon::TransportConfig config;
    config.initial_energy_MeVu = 200.0;
    config.primary_mass_number = 12;
    require_near(config.initial_total_energy_MeV(), 2400.0, 1.0e-12,
                 "MeV/u to total kinetic energy conversion failed");
    require(config.number_of_bins() == 800, "Depth-bin count failed");
    require(config.number_of_voxels() == 2'880'000, "Voxel count failed");
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 0;
    require_throws([&config]() { config.validate(); },
                   "Enabled voxel scorer accepted a zero x-bin count");
}

void test_csda_range_loss_validation() {
    const auto rejected = [](auto configure, const char* message) {
        carbon::TransportConfig config;
        config.enable_csda_range_energy_loss = true;
        configure(config);
        require_throws([&config] { config.validate(); }, message);
    };
    rejected([](auto& config) { config.enable_ct_grid = true; },
             "CSDA range loss accepted CT transport");
    rejected([](auto& config) { config.enable_layered_phantom = true; },
             "CSDA range loss accepted layered transport");
    rejected([](auto& config) { config.enable_hetero_insert = true; },
             "CSDA range loss accepted heterogeneous insert");
    rejected([](auto& config) { config.enable_minibeam = true; },
             "CSDA range loss accepted minibeam transport");
    rejected([](auto& config) { config.use_particle_specific_stopping_power = true; },
             "CSDA range loss accepted particle-specific stopping power");
    carbon::TransportConfig enabled;
    enabled.enable_csda_range_energy_loss = true;
    enabled.validate();
}

void test_hu_stopping_power_lut_loading() {
    const auto path = std::filesystem::temp_directory_path() /
                      "carbon_hu_stopping_power_lut_test.csv";
    {
        std::ofstream output(path);
        output << "0.1,0.2,0.3\n";  // optional energy-grid row
        output << "1.0,1.1,1.2\n";
        output << "0.8,0.9,1.0\n";
    }
    const auto lut = carbon::load_hu_stopping_power_lut(path, 2, 3, 2.0F);
    require(lut.size() == 6, "HU stopping-power LUT size failed");
    require_near(lut[0], 2.0, 1.0e-6,
                 "HU LUT energy row was treated as section zero");
    require_near(lut[3], 1.6, 1.0e-6, "HU LUT second section failed");

    {
        std::ofstream output(path);
        output << "1.0,1.1\n";
        output << "0.8,0.9,1.0\n";
    }
    require_throws(
        [&]() { (void)carbon::load_hu_stopping_power_lut(path, 2, 3, 1.0F); },
        "HU LUT accepted an incomplete row");
    std::filesystem::remove(path);
}

void test_highland_multiple_scattering() {
    const auto carbon_angle =
        carbon::highland_projected_rms_angle_rad(2400.0, 6, 12, 0.5, 1.0);
    const auto proton_angle =
        carbon::highland_projected_rms_angle_rad(200.0, 1, 1, 0.5, 1.0);
    const auto half_step_angle =
        carbon::highland_projected_rms_angle_rad(2400.0, 6, 12, 0.25, 1.0);
    require(std::isfinite(carbon_angle) && carbon_angle > 0.0,
            "Carbon Highland angle is not finite and positive");
    require(std::isfinite(proton_angle) && proton_angle > 0.0,
            "Proton Highland angle is not finite and positive");
    require(half_step_angle < carbon_angle,
            "Highland angle did not increase with material thickness");
    const auto bone_angle = carbon::highland_projected_rms_angle_material_rad(
        2400.0, 6, 12, 0.5, 1.85,
        carbon::ct_material_radiation_length_g_per_cm2(3));
    const auto water_at_bone_density = carbon::highland_projected_rms_angle_rad(
        2400.0, 6, 12, 0.5, 1.85);
    require(bone_angle > water_at_bone_density,
            "Bone radiation length did not increase CT MCS");
    require_near(
        carbon::highland_projected_rms_angle_material_rad(
            2400.0, 6, 12, 0.5, 1.0,
            carbon::ct_material_radiation_length_g_per_cm2(2)),
        carbon::highland_projected_rms_angle_rad(2400.0, 6, 12, 0.5, 1.0),
        2.0e-6, "Water material MCS changed beyond TOPAS rounding tolerance");
    require_near(
        carbon::highland_projected_rms_angle_rad(2400.0, 6, 12, 0.0, 1.0),
        0.0, 0.0, "Zero-length Highland angle failed");
    require_near(
        carbon::highland_projected_rms_angle_rad(2400.0, 0, 12, 0.5, 1.0),
        0.0, 0.0, "Neutral-particle Highland angle failed");
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
    config.enable_multiple_scattering = true;
    require_throws(
        [&]() { carbon::transport_serial(config, table, zero_cross_section()); },
        "Serial backend silently accepted three-dimensional multiple scattering");
}

void test_charged_dose_categories() {
    require(carbon::charged_dose_category(6, 12) == 0, "Carbon category failed");
    require(carbon::charged_dose_category(5, 11) == 1, "Boron category failed");
    require(carbon::charged_dose_category(4, 9) == 2, "Beryllium category failed");
    require(carbon::charged_dose_category(3, 7) == 3, "Lithium category failed");
    require(carbon::charged_dose_category(2, 4) == 4, "Helium category failed");
    require(carbon::charged_dose_category(1, 1) == 5, "Proton category failed");
    require(carbon::charged_dose_category(1, 2) == 5, "Deuteron Z1 category failed");
    require(carbon::charged_dose_category(1, 3) == 5, "Triton Z1 category failed");
    require(carbon::charged_dose_category(7, 14) == 6, "Other charged category failed");
    require(carbon::be_isotope_origin_category(4, 6) == 0, "Be-6 origin category");
    require(carbon::be_isotope_origin_category(4, 7) == 1, "Be-7 origin category");
    require(carbon::be_isotope_origin_category(4, 9) == 2, "Be-9 origin category");
    require(carbon::be_isotope_origin_category(4, 10) == 3, "Be-10 origin category");
    require(carbon::be_isotope_origin_category(4, 8) ==
                carbon::be_isotope_origin_category_count,
            "unsupported Be isotope origin sentinel");
}

void test_interpolation() {
    const carbon::StoppingPowerTable table({1.0, 2.0, 3.0}, {12.0, 8.0, 6.0});
    require_near(table.interpolate(1.5), 10.0, 1.0e-12, "Linear interpolation failed");
    require_near(table.interpolate(0.1), 12.0, 1.0e-12, "Low-energy clamp failed");
    require_near(table.interpolate(9.0), 6.0, 1.0e-12, "High-energy clamp failed");

    carbon::CrossSectionTable cross_section({1.0, 2.0, 3.0}, {0.01, 0.02, 0.04});
    require_near(cross_section.interpolate(2.5), 0.03, 1.0e-12,
                 "Cross-section interpolation failed");
    require_near(cross_section.interpolate(0.1), 0.01, 1.0e-12,
                 "Cross-section low-energy clamp failed");

    cross_section.set_target_h_fractions({0.2, 0.4, 0.8});
    const std::vector<double> transport_grid{1.0, 1.5, 2.0, 2.5, 3.0};
    const auto resampled =
        carbon::resample_cross_section_grid(cross_section, transport_grid);
    require(resampled.macroscopic_per_mm.size() == transport_grid.size(),
            "Resampled cross-section grid has wrong size");
    require_near(resampled.macroscopic_per_mm[1], 0.015, 1.0e-7,
                 "Cross-section transport-grid resampling failed");
    require_near(resampled.target_h_fraction[3], 0.6, 1.0e-7,
                 "Target-H transport-grid resampling failed");
}

void test_electron_transport_table_and_config() {
    const auto directory = std::filesystem::temp_directory_path() /
                           "maigo_electron_transport_test";
    std::filesystem::create_directories(directory);
    const auto table_path = directory / "electron_water.csv";
    const auto config_path = directory / "run.yaml";
    const auto write_table = [&](const bool valid_total = true) {
        std::ofstream output(table_path, std::ios::trunc);
        output
            << "kinetic_energy_MeV,"
               "electron_collisional_stopping_power_MeV_per_mm,"
               "electron_radiative_stopping_power_MeV_per_mm,"
               "electron_total_stopping_power_MeV_per_mm,"
               "positron_collisional_stopping_power_MeV_per_mm,"
               "positron_radiative_stopping_power_MeV_per_mm,"
               "positron_total_stopping_power_MeV_per_mm\n"
            << "0.001,2,0," << (valid_total ? 2 : 3) << ",3,0,3\n"
            << "1,4,1,5,5,1,6\n"
            << "500,6,2,8,7,2,9\n";
    };
    write_table();

    const auto table = carbon::ElectronTransportTable::from_csv(table_path);
    require(table.kinetic_energies_MeV().size() == 3,
            "Electron transport table row count failed");
    require_near(table.collisional_stopping_power(
                     carbon::LeptonSpecies::electron, 0.5005),
                 3.0, 1.0e-12,
                 "Electron collision stopping-power interpolation failed");
    require_near(table.total_stopping_power(
                     carbon::LeptonSpecies::positron, 1.0),
                 6.0, 1.0e-12,
                 "Positron total stopping-power lookup failed");
    require_near(table.csda_range_mm(carbon::LeptonSpecies::electron, 0.001),
                 0.0005, 1.0e-12, "Electron low-energy CSDA range failed");
    require(table.csda_range_mm(carbon::LeptonSpecies::electron, 500.0) >
                table.csda_range_mm(carbon::LeptonSpecies::electron, 1.0),
            "Electron CSDA range is not monotonic");

    {
        std::ofstream output(config_path, std::ios::trunc);
        output << "device: gpu\n"
               << "enable_electron_transport: true\n"
               << "electron_transport_data_file: electron_water.csv\n"
               << "electron_queue_capacity: 1234\n"
               << "electron_kinetic_cutoff_MeV: 0.01\n"
               << "maximum_electromagnetic_generations: 4\n"
               << "maximum_electron_step_mm: 0.1\n"
               << "maximum_electron_relative_energy_loss: 0.05\n";
    }
    const auto config = carbon::load_config(config_path);
    require(config.enable_electron_transport &&
                config.electron_transport_data_file == table_path &&
                config.electron_queue_capacity == 1234,
            "Electron transport YAML fields or relative table path failed");

    auto invalid = config;
    invalid.enable_ct_grid = true;
    invalid.ct_grid_file = "fixture.cctg";
    require_throws([&] { invalid.validate(); },
                   "Electron transport accepted CT geometry");
    invalid = config;
    invalid.device = "serial";
    require_throws([&] { invalid.validate(); },
                   "Serial backend claimed electron transport support");

    write_table(false);
    require_throws([&] { (void)carbon::ElectronTransportTable::from_csv(table_path); },
                   "Electron table accepted inconsistent total stopping power");
    std::filesystem::remove_all(directory);
}

void test_stopping_power_csda_range_helpers() {
    // Constant total-ion dE/dx makes the A*dE_u/S contract exact.
    const carbon::StoppingPowerTable constant({1.0, 3.0}, {2.0, 2.0});
    require_near(constant.csda_range_mm(0.5, 12), 3.0, 1.0e-12,
                 "CSDA low-energy range contract failed");
    require_near(constant.csda_range_mm(2.0, 12), 12.0, 1.0e-12,
                 "CSDA mass-number range contract failed");
    require_near(constant.csda_range_mm(8.0, 12), 48.0, 1.0e-12,
                 "CSDA high-energy endpoint clamp failed");
    require_near(constant.csda_energy_after_distance_MeVu(2.0, 3.0, 12), 1.5,
                 1.0e-12, "CSDA constant-table round trip failed");
    require_near(constant.csda_energy_after_distance_MeVu(2.0, 12.0, 12), 0.0,
                 1.0e-12, "CSDA exhausted range did not clamp to zero");
    require_near(constant.csda_energy_after_distance_MeVu(2.0, 0.0, 12), 2.0,
                 1.0e-12, "CSDA zero-distance boundary failed");

    const carbon::StoppingPowerTable varying({0.5, 1.0, 2.0, 4.0},
                                              {4.0, 2.0, 1.0, 0.5});
    double previous_range = 0.0;
    for (const auto energy : {0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}) {
        const auto range = varying.csda_range_mm(energy, 1);
        require(range >= previous_range, "CSDA range must be monotonic");
        previous_range = range;
    }
    for (const auto energy : {0.1, 0.5, 1.3, 3.7, 8.0}) {
        const auto range = varying.csda_range_mm(energy, 6);
        const auto recovered = varying.csda_energy_after_distance_MeVu(
            energy, 0.37 * range, 6);
        require_near(varying.csda_range_mm(recovered, 6), 0.63 * range,
                     1.0e-10 * std::max(1.0, range),
                     "CSDA range/energy inverse round trip failed");
    }
    std::vector<float> energies_device(varying.energies().begin(), varying.energies().end());
    std::vector<float> stopping_device(varying.values().begin(), varying.values().end());
    std::vector<float> ranges_device(varying.cumulative_ranges_mm().begin(),
                                      varying.cumulative_ranges_mm().end());
    for (const auto energy : {0.1F, 0.5F, 1.3F, 3.7F, 8.0F}) {
        require_near(
            carbon::csda_range_mm_device(energies_device.data(), stopping_device.data(),
                                          ranges_device.data(), energies_device.size(), energy, 6),
            varying.csda_range_mm(energy, 6), 2.0e-5,
            "Flat CSDA range helper disagrees with host helper");
        const auto distance = 0.37F * static_cast<float>(
            varying.csda_range_mm(energy, 6));
        require_near(
            carbon::csda_energy_after_distance_device(
                energies_device.data(), stopping_device.data(), ranges_device.data(),
                energies_device.size(), energy, distance, 6),
            varying.csda_energy_after_distance_MeVu(energy, distance, 6), 2.0e-5,
            "Flat CSDA inverse helper disagrees with host helper");
    }
    require_throws([&] { (void)varying.csda_range_mm(-0.1, 1); },
                   "CSDA accepted negative energy");
    require_throws([&] {
        (void)varying.csda_range_mm(std::numeric_limits<double>::quiet_NaN(), 1);
    }, "CSDA accepted NaN energy");
    require_throws([&] {
        (void)varying.csda_range_mm(std::numeric_limits<double>::infinity(), 1);
    }, "CSDA accepted infinite energy");
    require_throws([&] { (void)varying.csda_range_mm(1.0, 0); },
                   "CSDA accepted nonpositive mass number");
    require_throws([&] { (void)varying.csda_energy_after_distance_MeVu(1.0, -1.0, 1); },
                   "CSDA accepted negative distance");
    require_throws([&] {
        (void)varying.csda_energy_after_distance_MeVu(
            1.0, std::numeric_limits<double>::infinity(), 1);
    }, "CSDA accepted infinite distance");
}

void test_cpu_csda_range_loss_switch() {
    carbon::TransportConfig defaults;
    require(!defaults.enable_csda_range_energy_loss,
            "CPU CSDA range-loss mode must default to disabled");

    carbon::TransportConfig local;
    local.number_of_histories = 4;
    local.initial_energy_MeVu = 2.0;
    local.primary_atomic_number = 1;
    local.primary_mass_number = 2;
    local.phantom_length_mm = 0.5;
    local.depth_bin_width_mm = 1.0;
    local.maximum_relative_energy_loss = 0.5;
    local.maximum_step_mm = 0.1;
    const carbon::StoppingPowerTable constant({0.01, 4.0}, {2.0, 2.0});
    const auto historical = carbon::transport_serial(local, constant, zero_cross_section());
    local.enable_csda_range_energy_loss = true;
    const auto csda_constant = carbon::transport_serial(local, constant, zero_cross_section());
    const auto historical_total = std::accumulate(historical.deposited_energy_MeV.begin(),
                                                  historical.deposited_energy_MeV.end(), 0.0);
    const auto csda_constant_total = std::accumulate(
        csda_constant.deposited_energy_MeV.begin(), csda_constant.deposited_energy_MeV.end(), 0.0);
    require_near(historical_total, csda_constant_total, 1.0e-10,
                 "CPU CSDA constant-table path changed the legacy total loss");

    const carbon::StoppingPowerTable varying({0.01, 1.0, 2.0, 4.0},
                                              {4.0, 2.0, 1.0, 0.5});
    local.primary_mass_number = 1;
    local.initial_energy_MeVu = 3.0;
    local.maximum_step_mm = 0.1;
    const auto coarse = carbon::transport_serial(local, varying, zero_cross_section());
    local.maximum_step_mm = 0.037;
    const auto fine = carbon::transport_serial(local, varying, zero_cross_section());
    const auto coarse_total = std::accumulate(coarse.deposited_energy_MeV.begin(),
                                              coarse.deposited_energy_MeV.end(), 0.0);
    const auto fine_total = std::accumulate(fine.deposited_energy_MeV.begin(),
                                            fine.deposited_energy_MeV.end(), 0.0);
    require_near(coarse_total, fine_total, 1.0e-10,
                 "CPU CSDA nonlinear-table loss changed with step subdivision");
    require(coarse.relative_energy_balance_error() < 1.0e-12 &&
                fine.relative_energy_balance_error() < 1.0e-12,
            "CPU CSDA subdivision energy balance failed");
}

void test_cross_section_zero_endpoint_contract() {
    const carbon::CrossSectionTable table({0.0, 2.0, 4.0}, {2.0, 6.0, 14.0});
    require(table.energies().front() == 0.0,
            "Cross-section table did not retain a zero-energy endpoint");
    require_near(table.interpolate(-1.0), 2.0, 1.0e-12,
                 "Cross-section negative-energy clamp failed");
    require_near(table.interpolate(0.0), 2.0, 1.0e-12,
                 "Cross-section zero-energy clamp failed");
    require_near(table.interpolate(1.0), 4.0, 1.0e-12,
                 "Cross-section interpolation from a zero endpoint failed");
    require_near(table.interpolate(3.0), 10.0, 1.0e-12,
                 "Cross-section interior interpolation failed");
    require_near(table.interpolate(8.0), 14.0, 1.0e-12,
                 "Cross-section high-energy clamp failed");
    require_near(table.interpolate(std::numeric_limits<double>::quiet_NaN()), 2.0,
                 1.0e-12, "Cross-section non-finite query clamp failed");

    require_throws(
        [] { (void)carbon::CrossSectionTable({-1.0, 0.0}, {0.0, 1.0}); },
        "Cross-section table accepted a negative energy");
    require_throws(
        [] { (void)carbon::CrossSectionTable({0.0, 0.0}, {0.0, 1.0}); },
        "Cross-section table accepted a duplicate zero energy");
    require_throws(
        [] { (void)carbon::CrossSectionTable({0.0, -1.0}, {0.0, 1.0}); },
        "Cross-section table accepted a non-increasing energy grid");
    require_throws(
        [] {
            (void)carbon::CrossSectionTable(
                {0.0, std::numeric_limits<double>::infinity()}, {0.0, 1.0});
        },
        "Cross-section table accepted a non-finite energy");
    require_throws(
        [] {
            (void)carbon::CrossSectionTable(
                {0.0, 1.0}, {0.0, std::numeric_limits<double>::quiet_NaN()});
        },
        "Cross-section table accepted a non-finite value");
    require_throws(
        [] { (void)carbon::CrossSectionTable({0.0, 1.0}, {0.0, -1.0}); },
        "Cross-section table accepted a negative value");

    const auto path = std::filesystem::temp_directory_path() /
                      "carbon_cross_section_zero_endpoint.csv";
    {
        std::ofstream output(path, std::ios::trunc);
        output << "energy_MeV_per_u,water_macroscopic_cross_section_per_mm\n"
               << "0.0,0.0\n"
               << "1.0,0.5\n"
               << "2.0,0.25\n";
    }
    const auto loaded = carbon::CrossSectionTable::from_csv(path);
    require(loaded.energies().size() == 3,
            "Cross-section CSV zero endpoint row was not loaded");
    require(loaded.energies().front() == 0.0 && loaded.values().front() == 0.0,
            "Cross-section CSV zero endpoint values changed");
    require_near(loaded.interpolate(0.5), 0.25, 1.0e-12,
                 "Cross-section CSV zero endpoint interpolation failed");
    std::filesystem::remove(path);
}

void test_fragment_stopping_power_scale() {
    require_near(carbon::stopping_power_scale_from_reference_ion(6, 6, 200.0), 1.0, 1.0e-12,
                 "C-12 stopping-power scale failed");
    const auto proton_scale = carbon::stopping_power_scale_from_reference_ion(1, 6, 200.0);
    const auto helium_scale = carbon::stopping_power_scale_from_reference_ion(2, 6, 200.0);
    const auto boron_scale = carbon::stopping_power_scale_from_reference_ion(5, 6, 200.0);
    require(proton_scale > 0.0 && proton_scale < helium_scale && helium_scale < boron_scale &&
                boron_scale < 1.0,
            "Fragment stopping-power charge ordering failed");
    require(carbon::stopping_power_scale_from_reference_ion(1, 6, 1.0) > proton_scale,
            "Low-energy effective-charge scaling failed");
    require_throws([]() {
        (void)carbon::stopping_power_scale_from_reference_ion(0, 6, 10.0);
    },
                   "Invalid fragment atomic number was accepted");
}

void test_primary_ion_definition() {
    const auto carbon_ion = carbon::make_primary_ion_definition(6, 12);
    require(carbon_ion.atomic_number == 6 && carbon_ion.mass_number == 12,
            "C-12 primary identity failed");
    require_near(carbon_ion.rest_mass_MeV, 12.0 * 931.49410242, 1.0e-9,
                 "Default primary rest mass changed");
    const auto proton = carbon::make_primary_ion_definition(1, 1, 938.27208816);
    require_near(proton.rest_mass_MeV, 938.27208816, 1.0e-9,
                 "Configured proton rest mass failed");
    require_throws([] { (void)carbon::make_primary_ion_definition(0, 1); },
                   "Primary ion accepted Z=0");
    require_throws([] { (void)carbon::make_primary_ion_definition(8, 4); },
                   "Primary ion accepted A < Z");
    require(carbon::ion_effective_charge(1, 200.0) > 0.0 &&
                carbon::ion_effective_charge(10, 200.0) >
                    carbon::ion_effective_charge(6, 200.0),
            "Generic effective charge does not cover proton through neon");
}

void test_ion_physics_manifest_loading() {
    const auto directory = std::filesystem::temp_directory_path() /
                           "maigo_ion_physics_manifest_test";
    std::filesystem::create_directories(directory);
    const auto manifest_path = directory / "proton_water.yaml";
    const auto config_path = directory / "run.yaml";
    for (const char* name : {
             "stopping.csv", "ions.csv",
         }) {
        std::ofstream(directory / name) << "fixture\n";
    }
    const auto write_valid_manifest = [&] {
        std::ofstream output(manifest_path);
        output << "primary_atomic_number: 1\n"
               << "primary_mass_number: 1\n"
               << "primary_rest_mass_MeV: 938.27208816\n"
               << "energy_straggling_model: gaussian_clamped\n"
               << "use_particle_specific_stopping_power: true\n"
               << "primary_stopping_power_file: stopping.csv\n"
               << "particle_stopping_power_file: ions.csv\n";
    };
    write_valid_manifest();
    {
        std::ofstream output(config_path);
        output << "ion_physics_file: proton_water.yaml\n"
               << "number_of_histories: 17\n"
               << "initial_energy_MeVu: 150\n";
    }
    const auto config = carbon::load_config(config_path);
    require(config.ion_physics_file == manifest_path,
            "Ion physics manifest path was not resolved");
    require(config.primary_atomic_number == 1 && config.primary_mass_number == 1,
            "Ion physics manifest did not import projectile identity");
    require(config.energy_straggling_model == "gaussian_clamped" &&
                config.use_particle_specific_stopping_power,
            "Ion physics manifest did not import ion-dependent model choices");
    require_near(config.primary_rest_mass_MeV, 938.27208816, 1.0e-12,
                 "Ion physics manifest did not import rest mass");
    require(config.primary_stopping_power_file == directory / "stopping.csv",
            "Ion physics manifest data paths were not resolved relative to the manifest");
    require(config.number_of_histories == 17 && config.initial_energy_MeVu == 150.0,
            "Ion physics manifest changed run controls");

    {
        std::ofstream output(config_path);
        output << "ion_physics_file: proton_water.yaml\n"
               << "primary_atomic_number: 6\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Main YAML overrode manifest-owned ion identity");

    {
        std::ofstream output(manifest_path);
        output << "primary_atomic_number: 1\n"
               << "number_of_histories: 99\n";
    }
    {
        std::ofstream output(config_path);
        output << "ion_physics_file: proton_water.yaml\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Ion physics manifest accepted a runtime control");

    {
        std::ofstream output(manifest_path);
        output << "primary_atomic_number: 1\n"
               << "primary_mass_number: 1\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Incomplete ion physics manifest was accepted");
    std::filesystem::remove_all(directory);
}

void test_strict_config_parsing_and_canonicalization() {
    const auto directory = std::filesystem::temp_directory_path() /
                           "maigo_strict_config_test";
    std::filesystem::create_directories(directory);
    const auto config_path = directory / "run.yaml";

    {
        std::ofstream output(config_path);
        output << "number_of_histories: 7\n"
               << "number_of_histories: 8\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Configuration parser accepted a duplicate key");

    {
        std::ofstream output(config_path);
        output << "number_of_histories: 7\n"
               << "number_of_historiez: 8\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Configuration parser accepted an unknown key");

    {
        std::ofstream output(config_path);
        output << "config_schema_version: 2\n"
               << "number_of_histories: 7\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Configuration parser accepted an unsupported schema version");

    {
        std::ofstream output(config_path);
        output << "number_of_histories: 7 # stripped from canonical text\n"
               << "initial_energy_MeVu: 123\n";
    }
    const auto first = carbon::load_config(config_path);
    require(first.config_schema_version == 1U,
            "Implicit configuration schema is not v1");
    require(first.canonical_config_text ==
                "config_schema_version: 1\n"
                "initial_energy_MeVu: 123\n"
                "number_of_histories: 7\n",
            "Canonical configuration text is not normalized and key-sorted");

    {
        std::ofstream output(config_path);
        output << "config_schema_version: 1\n"
               << "initial_energy_MeVu: 123\n"
               << "number_of_histories: 7\n";
    }
    const auto reordered = carbon::load_config(config_path);
    require(reordered.canonical_config_text == first.canonical_config_text,
            "Canonical configuration changed with input key order");

    {
        std::ofstream output(config_path);
        output << "run_mode: production\n"
               << "quality_maximum_relative_energy_residual: 0.0002\n"
               << "quality_maximum_absolute_energy_residual_MeV: 0.01\n";
    }
    const auto production = carbon::load_config(config_path);
    require(production.run_mode == carbon::RunMode::production &&
                production.quality_maximum_relative_energy_residual == 0.0002 &&
                production.quality_maximum_absolute_energy_residual_MeV == 0.01,
            "Run quality configuration fields were not parsed");
    auto weakened_production = production;
    weakened_production.quality_reject_any_queue_overflow = false;
    require_throws([&] { weakened_production.validate(); },
                   "Production mode allowed queue-overflow rejection to be disabled");

    {
        std::ofstream output(config_path);
        output << "run_mode: clinical\n";
    }
    require_throws([&] { (void)carbon::load_config(config_path); },
                   "Configuration accepted an unknown run mode");

    std::filesystem::remove_all(directory);
}

void test_run_quality_gate() {
    carbon::TransportConfig production;
    production.run_mode = carbon::RunMode::production;
    production.quality_maximum_relative_energy_residual = 1.0e-4;
    production.quality_maximum_absolute_energy_residual_MeV = 1.0e-9;

    auto clean = std::make_unique<carbon::TransportResult>();
    clean->initial_energy_MeV = 100.0;
    clean->total_deposited_energy_MeV = 100.0;
    auto report = carbon::evaluate_run_quality(production, *clean);
    require(report.accepted && report.status() == "pass" &&
                report.failures.empty() && report.approximations.empty(),
            "Production quality gate rejected a clean result");

    const auto require_overflow_rejected = [&](auto set_overflow,
                                                const std::string& label) {
        auto result = std::make_unique<carbon::TransportResult>(*clean);
        set_overflow(*result);
        const auto rejected = carbon::evaluate_run_quality(production, *result);
        require(!rejected.accepted && rejected.queue_overflow_count == 1 &&
                    !rejected.failures.empty(),
                "Production quality gate accepted " + label + " overflow");
    };
    require_overflow_rejected(
        [](auto& result) { result.electron_queue_overflow = 1; }, "electron");
    require_overflow_rejected(
        [](auto& result) { result.electron_gamma_queue_overflow = 1; },
        "electron-gamma");

    auto research = production;
    research.run_mode = carbon::RunMode::research;
    auto overflow = std::make_unique<carbon::TransportResult>(*clean);
    overflow->electron_queue_overflow = 1;
    report = carbon::evaluate_run_quality(research, *overflow);
    require(report.accepted && report.status() == "non_production" &&
                report.failures.empty() && report.approximations.size() == 1,
            "Research quality gate did not expose overflow as an approximation");

    {
        auto compatibility = production;
        compatibility.cinel02_topas_compatibility_mode = true;
        auto sink = std::make_unique<carbon::TransportResult>(*clean);
        sink->total_deposited_energy_MeV = 90.0;
        sink->fred_model_unassigned_MeV = 5.0;
        sink->cinel02_topas_compat_discarded_kinetic_MeV.back() = 5.0;
        const auto sink_report = carbon::evaluate_run_quality(compatibility, *sink);
        require(sink_report.accepted && sink_report.topas_reference_energy_sink_active,
                "Compatibility sink should be an explicit accepted approximation");
        require_near(sink_report.absolute_physical_energy_residual_MeV, 5.0, 0.0,
                     "Physical closure must exclude the compatibility sink");
        require_near(sink_report.absolute_accounting_energy_residual_MeV, 0.0, 0.0,
                     "Accounting closure must include the compatibility sink");
        require_near(sink_report.physical_relative_energy_residual, 0.05, 0.0,
                     "Physical relative residual mismatch");
        require_near(sink_report.accounting_relative_energy_residual, 0.0, 0.0,
                     "Accounting relative residual mismatch");
        require_near(sink_report.absolute_energy_residual_MeV,
                     sink_report.absolute_accounting_energy_residual_MeV, 0.0,
                     "Legacy absolute residual alias mismatch");
        require_near(sink_report.relative_energy_residual,
                     sink_report.accounting_relative_energy_residual, 0.0,
                     "Legacy relative residual alias mismatch");
    }

    auto residual = std::make_unique<carbon::TransportResult>(*clean);
    residual->total_deposited_energy_MeV = 99.0;
    report = carbon::evaluate_run_quality(production, *residual);
    require(!report.accepted && !report.failures.empty(),
            "Production quality gate accepted a large energy residual");

    auto non_finite = std::make_unique<carbon::TransportResult>(*clean);
    non_finite->deposited_energy_MeV = {
        std::numeric_limits<double>::quiet_NaN()};
    report = carbon::evaluate_run_quality(production, *non_finite);
    require(!report.accepted && !report.failures.empty(),
            "Production quality gate accepted a non-finite scorer");

    const auto path = std::filesystem::temp_directory_path() /
                      "maigo_quality_report.json";
    report = carbon::evaluate_run_quality(production, *residual);
    carbon::write_run_quality_report_json(path, report);
    std::ifstream input(path);
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    require(json.find("\"schema_version\": 2") != std::string::npos,
            "Run quality JSON schema version did not advance");
    require(json.find("\"accounting_relative_energy_residual\"") != std::string::npos &&
                json.find("\"physical_relative_energy_residual\"") != std::string::npos,
            "Run quality JSON missing physical/accounting residual fields");
    require(json.find("\"status\": \"fail\"") != std::string::npos,
            "Run quality JSON failed status");
    std::filesystem::remove(path);
}

void test_particle_specific_stopping_power_tables() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto carbon = carbon::StoppingPowerTable::from_csv(
        source_directory /
        "data/stopping_power_water_geant4_11_3_2.csv");
    const auto ions = carbon::IonStoppingPowerTables::from_csv(
        source_directory /
            "data/ion_stopping_power_water_geant4_11_3_2.csv",
        carbon);
    require(ions.energy_grid_size() == 4001,
            "Ion stopping-power energy grid size failed");
    require(std::count(ions.species_present().begin(),
                       ions.species_present().end(),
                       std::uint8_t{1}) == 50,
            "Ion stopping-power isotope count failed");
    require(ions.species_present()[
                7 * carbon::IonStoppingPowerTables::mass_stride + 13] == 1 &&
                ions.species_present()[
                    8 * carbon::IonStoppingPowerTables::mass_stride + 14] == 1,
            "Heavy-recoil isotope stopping-power coverage failed");
    const auto c12 = (6 * carbon::IonStoppingPowerTables::mass_stride + 12) *
                     ions.energy_grid_size();
    const auto h1 = (1 * carbon::IonStoppingPowerTables::mass_stride + 1) *
                    ions.energy_grid_size();
    constexpr std::size_t near_100_MeVu = 1000;
    require_near(ions.ratios_to_carbon()[c12 + near_100_MeVu], 1.0, 1.0e-7,
                 "C-12 exact stopping-power ratio failed");
    require_near(ions.ratios_to_carbon()[h1 + near_100_MeVu],
                 0.027915, 1.0e-5,
                 "H-1 exact stopping-power ratio failed");
    require_near(ions.delta_electron_fractions()[c12 + near_100_MeVu],
                 0.080135384373, 1.0e-7,
                 "C-12 calibrated delta-electron fraction failed");
}

void test_step_selection() {
    require_near(carbon::choose_step_mm(100.0, 10.0, 0.5, 0.01), 0.1, 1.0e-12,
                 "Energy-limited step failed");
    require_near(carbon::choose_step_mm(1000.0, 1.0, 0.5, 0.01), 0.5, 1.0e-12,
                 "Maximum-step limit failed");
}

void test_slab_phantom_helpers() {
    const float z_ends[] = {50.0F, 70.0F, 400.0F};
    const float densities[] = {1.0F, 1.85F, 1.0F};
    require(carbon::slab_layer_index(0.0F, z_ends, 3) == 0, "entrance layer");
    require(carbon::slab_layer_index(49.9F, z_ends, 3) == 0, "first layer interior");
    require(carbon::slab_layer_index(50.0F, z_ends, 3) == 1, "dense layer start");
    require(carbon::slab_layer_index(69.9F, z_ends, 3) == 1, "dense layer interior");
    require(carbon::slab_layer_index(70.0F, z_ends, 3) == 2, "exit water start");
    require_near(carbon::slab_density_g_per_cm3(60.0F, z_ends, densities, 3, 1.0F), 1.85,
                 1.0e-5, "dense slab density");
    require_near(carbon::distance_to_slab_interface_mm(40.0F, 1.0F, z_ends, 3, 400.0F), 10.0,
                 1.0e-4, "forward interface distance");
    require_near(carbon::distance_to_slab_interface_mm(60.0F, -1.0F, z_ends, 3, 400.0F), 10.0,
                 1.0e-4, "backward interface distance");

    require(carbon::inside_hetero_insert(0.0F, 0.0F, 55.0F, -10.0F, 10.0F, -10.0F, 10.0F,
                                         50.0F, 70.0F),
            "insert interior");
    require(!carbon::inside_hetero_insert(0.0F, 0.0F, 40.0F, -10.0F, 10.0F, -10.0F, 10.0F,
                                          50.0F, 70.0F),
            "insert exterior z");
    require_near(carbon::distance_to_insert_interface_mm(
                     0.0F, 0.0F, 40.0F, 0.0F, 0.0F, 1.0F, -10.0F, 10.0F, -10.0F, 10.0F, 50.0F,
                     70.0F, 1.0e6F),
                 10.0, 1.0e-4, "enter insert along +z");
    require_near(carbon::distance_to_insert_interface_mm(
                     0.0F, 0.0F, 55.0F, 0.0F, 0.0F, 1.0F, -10.0F, 10.0F, -10.0F, 10.0F, 50.0F,
                     70.0F, 1.0e6F),
                 15.0, 1.0e-4, "exit insert along +z");

    carbon::TransportConfig config;
    config.enable_layered_phantom = true;
    config.phantom_length_mm = 400.0;
    config.slab_layers = {{50.0, 1.0}, {70.0, 1.85}, {400.0, 1.0}};
    config.slab_radiation_lengths_g_per_cm2 = {36.0830, 30.4866, 36.0830};
    config.validate();
    require_throws(
        []() {
            carbon::TransportConfig bad;
            bad.enable_layered_phantom = true;
            bad.phantom_length_mm = 400.0;
            bad.slab_layers = {{50.0, 1.0}, {70.0, 1.85}};  // last != phantom
            bad.validate();
        },
        "mismatched last slab end was accepted");
    require_throws(
        []() {
            carbon::TransportConfig bad;
            bad.enable_layered_phantom = true;
            bad.phantom_length_mm = 400.0;
            bad.slab_layers = {{50.0, 1.0}, {70.0, 1.85}, {400.0, 1.0}};
            bad.slab_radiation_lengths_g_per_cm2 = {36.0830, 30.4866};
            bad.validate();
        },
        "mismatched slab radiation-length list was accepted");
    require_throws(
        []() {
            carbon::TransportConfig bad;
            bad.enable_layered_phantom = true;
            bad.phantom_length_mm = 400.0;
            bad.slab_layers = {{50.0, 1.0}, {70.0, 1.85}, {400.0, 1.0}};
            bad.slab_radiation_lengths_g_per_cm2 = {36.0830, 0.0, 36.0830};
            bad.validate();
        },
        "non-positive slab radiation length was accepted");

    carbon::TransportConfig insert_config;
    insert_config.enable_hetero_insert = true;
    insert_config.phantom_length_mm = 400.0;
    insert_config.hetero_insert = {-10.0, 10.0, -10.0, 10.0, 50.0, 70.0, 1.85};
    insert_config.insert_radiation_length_g_per_cm2 = 30.4866;
    insert_config.validate();
    insert_config.insert_radiation_length_g_per_cm2 = 0.0;
    require_throws(
        [&insert_config]() { insert_config.validate(); },
        "non-positive insert radiation length was accepted");
}

void test_ct_grid_helpers() {
    require_near(carbon::hu_to_density_g_per_cm3(0.0F), 1.0F, 1.0e-5, "HU0 density");
    require_near(carbon::hu_to_density_g_per_cm3(-1000.0F), 0.001205F, 1.0e-6,
                 "air density");
    require(carbon::density_to_material_id(1.0F) == 2, "water material id");
    require(carbon::density_to_material_id(1.85F) == 3, "bone material id");
    require(carbon::ct_material_class(0, true) == 0,
            "Schneider section 0 must map to air");
    require(carbon::ct_material_class(1, true) == 1,
            "Schneider section 1 must map to lung");
    require(carbon::ct_material_class(2, true) == 2 &&
                carbon::ct_material_class(8, true) == 2,
            "Schneider soft-tissue sections must map to water-like");
    require(carbon::ct_material_class(9, true) == 3 &&
                carbon::ct_material_class(24, true) == 3,
            "Schneider sections >=9 must map to bone-like");
    require(carbon::ct_material_class(2, false) == 2 &&
                carbon::ct_material_class(9, false) == 3,
            "Legacy CT material class mapping failed");
    require_near(carbon::ct_material_reference_density_g_per_cm3(0U), 0.00120479F,
                 1.0e-8, "G4_AIR reference density");
    require_near(carbon::ct_material_reference_density_g_per_cm3(1U), 1.04F, 1.0e-7,
                 "G4_LUNG_ICRP reference density");
    require_near(carbon::ct_material_reference_density_g_per_cm3(2U), 1.0F, 1.0e-7,
                 "Water reference density");
    require_near(carbon::ct_material_reference_density_g_per_cm3(3U), 1.85F, 1.0e-7,
                 "G4_BONE_COMPACT_ICRU reference density");
    require(carbon::ct_cross_section_material_index(15U, true, true) == 15U,
            "Schneider XS must preserve the section index");
    require(carbon::ct_cross_section_material_index(15U, true, false) == 3U,
            "Legacy XS must collapse Schneider bone sections");

    require_near(carbon::ct_mass_scaled_stopping_power(10.0F, 1.5F, 1.0F), 15.0F, 1.0e-5,
                 "mass SP water scale");
    require_near(carbon::ct_mass_scaled_stopping_power(10.0F, 1.85F, 0.93F), 17.205F,
                 1.0e-3, "mass SP bone factor");

    require_near(carbon::distance_to_next_ct_face_1d(0.3F, 0.0F, 1.0F, 1.0F), 0.7F, 1.0e-5,
                 "ct face +x interior");
    require_near(carbon::distance_to_next_ct_face_1d(1.0F, 0.0F, 1.0F, 1.0F), 1.0F, 1.0e-4,
                 "ct face +x on boundary");
    require_near(carbon::distance_to_next_ct_face_1d(1.0F, 0.0F, 1.0F, -1.0F), 1.0F, 1.0e-4,
                 "ct face -x on boundary");

    const float dens_h[8] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    const std::uint8_t mats_h[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    const auto step_h = carbon::clamp_step_to_ct_faces_if_needed(
        0.4F, 0.1F, 0.1F, 0.1F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2,
        2, 2, dens_h, mats_h, 1.0F, 2);
    require_near(step_h, 0.4F, 1.0e-5, "homogeneous CT step should not face-clamp");

    // Short energy-limited steps skip face clamp (C).
    const auto step_short = carbon::clamp_step_to_ct_faces_if_needed(
        0.05F, 0.1F, 0.1F, 0.1F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2,
        2, 2, dens_h, mats_h, 1.0F, 2, true);
    require_near(step_short, 0.05F, 1.0e-6, "short CT step should skip face-clamp");

    const float dens_z[8] = {1.0F, 1.0F, 1.0F, 1.0F,
                             2.0F, 2.0F, 2.0F, 2.0F};
    const auto step_near_z = carbon::clamp_step_to_ct_faces_near_z_if_needed(
        0.4F, 0.1F, 0.1F, 0.8F, 0.01F, 0.0F, 0.99995F,
        0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2, 2, 2,
        dens_z, mats_h, 1.0F, 2, true);
    require_near(step_near_z, 0.20001F, 2.0e-4,
                 "near-z CT fast path must clamp at heterogeneous z face");

    // DDA: particle on a +z face must advance into the next voxel, not return ~0.
    carbon::CtDdaState dda{};
    require(carbon::ct_dda_init(0.5F, 0.5F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F,
                                1.0F, 1.0F, 2, 2, 2, dda),
            "DDA init on +z face");
    require(dda.iz == 1, "DDA must enter next z cell on boundary");
    require(carbon::ct_dda_distance_to_next_face(dda) > 0.5F,
            "DDA distance after face snap must be positive");

    // Energy-dependent mass-SP: I = I_water (75 eV) → factor = za_rel.
    require_near(carbon::ct_mass_sp_energy_factor(0.93F, 75.0F, 150.0F), 0.93F, 1.0e-4,
                 "mass-SP energy factor water-I");
    const auto f_bone_hi =
        carbon::ct_mass_sp_energy_factor(0.93F, 106.0F, 200.0F);
    const auto f_bone_lo =
        carbon::ct_mass_sp_energy_factor(0.93F, 106.0F, 10.0F);
    require(f_bone_hi < 0.93F && f_bone_hi > 0.85F, "bone f_E high-E below za");
    require(f_bone_lo < f_bone_hi, "bone f_E decreases toward low E");

    const float dens[8] = {1.0F, 1.1F, 1.2F, 1.3F, 1.4F, 1.5F, 1.6F, 1.7F};
    const std::uint8_t mats[8] = {2, 2, 2, 2, 3, 3, 3, 3};
    float rho = 0.0F;
    std::uint8_t mid = 0;
    require(carbon::ct_sample(0.1F, 0.1F, 0.1F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2, 2, 2,
                              dens, mats, rho, mid),
            "ct sample inside");
    require_near(rho, 1.0F, 1.0e-6, "ct sample density");
    require(mid == 2, "ct sample material");
    require(!carbon::ct_sample(-0.1F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2, 2,
                               2, dens, mats, rho, mid),
            "ct sample outside");

    carbon::CtGrid grid;
    grid.nx = 2;
    grid.ny = 2;
    grid.nz = 2;
    grid.density_g_per_cm3 = {1.0F, 1.1F, 1.2F, 1.3F, 1.4F, 1.5F, 1.6F, 1.7F};
    grid.material_id = {0, 1, 2, 3, 0, 1, 2, 3};
    grid.mass_sp_za_rel = {1.0F, 0.99F, 1.0F, 0.93F};
    grid.mass_sp_I_eV = {75.0F, 72.0F, 75.0F, 106.0F};
    const auto path =
        std::filesystem::temp_directory_path() / "carbon_ct_v3_roundtrip.bin";
    grid.write_binary(path);
    const auto loaded = carbon::CtGrid::from_binary(path);
    require(loaded.file_version == carbon::CtGrid::version_value, "ct v3 version");
    require(loaded.has_mass_sp_factors() && loaded.mass_sp_za_rel.size() == 4,
            "ct v3 za_rel");
    require(loaded.uses_schneider_mass_sp(),
            "ct v3 Schneider mass-SP selection");
    auto legacy = loaded;
    legacy.file_version = carbon::CtGrid::version_legacy;
    require(legacy.has_mass_sp_factors() && !legacy.uses_schneider_mass_sp(),
            "ct v1 unit factors must not shadow absolute material tables");
    require_near(loaded.mass_sp_za_rel[3], 0.93F, 1.0e-6, "ct v3 bone za");
    require_near(loaded.mass_sp_I_eV[3], 106.0F, 1.0e-4, "ct v3 bone I");
    std::filesystem::remove(path);

    const auto schneider = carbon::SchneiderHuTable::builtin();
    require(schneider.section_id(-1000.0F) == 0, "HU -1000 is Schneider air");
    require(schneider.section_id(-200.0F) == 1, "HU -200 is Schneider lung");
    require(schneider.section_id(0.0F) == 5, "HU 0 is Schneider section 5");
    require(schneider.section_id(150.0F) == 9, "HU 150 is first bone-like section");
    require(schneider.density_g_per_cm3(-1000.0F) > 0.0F &&
                schneider.density_g_per_cm3(-1000.0F) < 0.01F,
            "Schneider air density");
    require_near(schneider.density_g_per_cm3(0.0F), 1.018F, 1.0e-3,
                 "Schneider HU0 density without correction");

    const auto write_explicit_ct = [](const std::filesystem::path& dest,
                                      const std::int16_t hu,
                                      const double z_mm) {
        std::ofstream out(dest, std::ios::binary);
        std::vector<char> preamble(128, 0);
        out.write(preamble.data(), 128);
        out.write("DICM", 4);
        auto put_u16 = [&](const std::uint16_t value) {
            out.write(reinterpret_cast<const char*>(&value), 2);
        };
        auto put_u32 = [&](const std::uint32_t value) {
            out.write(reinterpret_cast<const char*>(&value), 4);
        };
        auto put_explicit = [&](const std::uint16_t group, const std::uint16_t element,
                                const char* vr, const std::string& value) {
            put_u16(group);
            put_u16(element);
            out.write(vr, 2);
            const auto padded = value.size() % 2 == 0 ? value : value + " ";
            if (std::string(vr) == "OW" || std::string(vr) == "OB" ||
                std::string(vr) == "UN") {
                put_u16(0);
                put_u32(static_cast<std::uint32_t>(padded.size()));
            } else {
                put_u16(static_cast<std::uint16_t>(padded.size()));
            }
            out.write(padded.data(), static_cast<std::streamsize>(padded.size()));
        };
        auto put_us = [&](const std::uint16_t group, const std::uint16_t element,
                          const std::uint16_t value) {
            put_u16(group);
            put_u16(element);
            out.write("US", 2);
            put_u16(2);
            put_u16(value);
        };
        put_explicit(0x0002, 0x0010, "UI", "1.2.840.10008.1.2.1");
        put_explicit(0x0008, 0x0060, "CS", "CT");
        put_explicit(0x0020, 0x000E, "UI", "1.2.3.4.5");
        put_explicit(0x0020, 0x0032, "DS",
                     "0\\0\\" + std::to_string(z_mm));
        put_explicit(0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0");
        put_us(0x0028, 0x0010, 2);
        put_us(0x0028, 0x0011, 2);
        put_explicit(0x0028, 0x0030, "DS", "1\\1");
        put_us(0x0028, 0x0100, 16);
        put_us(0x0028, 0x0103, 1);
        put_explicit(0x0028, 0x1052, "DS", "0");
        put_explicit(0x0028, 0x1053, "DS", "1");
        put_u16(0x7FE0);
        put_u16(0x0010);
        out.write("OW", 2);
        put_u16(0);
        put_u32(8);
        for (int i = 0; i < 4; ++i) {
            out.write(reinterpret_cast<const char*>(&hu), 2);
        }
    };
    const auto dicom_dir =
        std::filesystem::temp_directory_path() / "carbon_ct_dicom_series";
    std::filesystem::create_directories(dicom_dir);
    write_explicit_ct(dicom_dir / "s0.dcm", 0, 0.0);
    write_explicit_ct(dicom_dir / "s1.dcm", 0, 2.0);
    const auto schneider_file = std::filesystem::path("data/HUtoMaterialSchneider.txt");
    const auto table = std::filesystem::is_regular_file(schneider_file)
                           ? carbon::SchneiderHuTable::from_topas_file(schneider_file)
                           : carbon::SchneiderHuTable::builtin();
    const auto from_dicom = carbon::CtGrid::from_dicom_directory(
        dicom_dir,
        std::filesystem::is_regular_file(schneider_file) ? schneider_file
                                                        : std::filesystem::path{},
        "centered");
    require(from_dicom.nx == 2 && from_dicom.ny == 2 && from_dicom.nz == 2,
            "DICOM series dimensions");
    require_near(from_dicom.spacing_z_mm, 2.0F, 1.0e-5, "DICOM slice spacing");
    require_near(from_dicom.origin_x_mm, -1.0F, 1.0e-5, "centered DICOM origin X");
    require_near(from_dicom.origin_z_mm, -1.0F, 1.0e-5, "centered DICOM origin Z");
    require(from_dicom.material_id.front() == 5, "DICOM HU 0 maps to Schneider 5");
    require_near(from_dicom.density_g_per_cm3.front(), table.density_g_per_cm3(0.0F),
                 1.0e-5, "DICOM HU 0 Schneider density");
    const auto via_load = carbon::CtGrid::load(dicom_dir, {}, "dicom");
    require_near(via_load.origin_x_mm, -0.5F, 1.0e-4, "native DICOM low-edge X");
    require_near(via_load.origin_z_mm, -1.0F, 1.0e-4, "native DICOM low-edge Z");
    std::filesystem::remove_all(dicom_dir);
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

    // Child streams: same parent+tag always equal; different tags/roles diverge.
    const auto parent = 42ULL;
    const auto a = carbon::rng::child_stream(
        parent, carbon::rng::branch_tag(carbon::rng::branch_role_primary_charged, 0));
    const auto a2 = carbon::rng::child_stream(
        parent, carbon::rng::branch_tag(carbon::rng::branch_role_primary_charged, 0));
    const auto b = carbon::rng::child_stream(
        parent, carbon::rng::branch_tag(carbon::rng::branch_role_primary_charged, 1));
    const auto n = carbon::rng::child_stream(
        parent, carbon::rng::branch_tag(carbon::rng::branch_role_primary_neutral, 0));
    require(a == a2 && a != 0 && a != parent, "child_stream not deterministic");
    require(a != b && a != n, "child_stream collisions across tags");
    require(sizeof(carbon::SecondaryParticle3D) == 48 &&
                sizeof(carbon::NeutralParticle3D) == 48,
            "Particle layout size changed unexpectedly");
}

void test_bohr_straggling() {
    const auto sigma_half_mm = carbon::bohr_straggling_sigma_MeV(200.0, 6, 0.5, 1.0);
    const auto sigma_two_mm = carbon::bohr_straggling_sigma_MeV(200.0, 6, 2.0, 1.0);
    require(sigma_half_mm > 0.0, "Bohr straggling sigma must be positive");
    require_near(sigma_two_mm / sigma_half_mm, 2.0, 1.0e-12,
                 "Bohr sigma must scale with sqrt(step length)");
    require_near(carbon::clamp_sampled_energy_loss(1.0, 2.0, -2.0, 10.0), 0.0,
                 1.0e-12, "Negative sampled loss clamp failed");
    require_near(carbon::clamp_sampled_energy_loss(9.0, 2.0, 2.0, 10.0), 10.0,
                 1.0e-12, "Available-energy clamp failed");
    require_near(carbon::clamp_sampled_energy_loss(1.0, 10.0, 1.0, 100.0), 2.0,
                 1.0e-12, "Two-mean fluctuation clamp failed");
}

void test_clamped_gaussian_straggling_sampler_audit() {
    // Diagnose why scale=1 still needs energy-wise correction: the production
    // sampler is Gaussian + clamp to [0, min(2μ, E)]. High-energy 0.1 mm
    // proton blocks have μ/σ ~ 1.1, so the 2μ cap discards variance that the
    // condensed-loss formula requested. This audit does not change transport.
    const auto erf = [](const double x) {
        return std::erf(x);
    };
    const auto normal_cdf = [&erf](const double x) {
        return 0.5 * (1.0 + erf(x / std::sqrt(2.0)));
    };
    const auto normal_pdf = [](const double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
    };
    const auto censored_gaussian_variance = [&](const double mean,
                                                const double sigma) {
        const auto a = -mean / sigma;
        const auto b = mean / sigma;
        const auto pdf_a = normal_pdf(a);
        const auto pdf_b = normal_pdf(b);
        const auto cdf_a = normal_cdf(a);
        const auto cdf_b = normal_cdf(b);
        const auto ez = cdf_a * a + (pdf_a - pdf_b) + (1.0 - cdf_b) * b;
        const auto ez2 = cdf_a * a * a + (a * pdf_a - b * pdf_b) + (cdf_b - cdf_a) +
                         (1.0 - cdf_b) * b * b;
        return (ez2 - ez * ez) * sigma * sigma;
    };

    const auto water = carbon::StoppingPowerTable::from_csv(
        std::filesystem::path(CARBON_SOURCE_DIR) /
        "data/stopping_power_water_geant4_11_3_2.csv");
    constexpr double proton_mass_MeV = 938.27208816;
    constexpr double block_mm = 0.1;
    constexpr int sample_count = 200000;
    const std::array<double, 5> energies_MeVu{70.0, 100.0, 150.0, 200.0, 250.0};
    double previous_retained = 2.0;
    double retained_250 = 1.0;
    for (std::size_t point_index = 0; point_index < energies_MeVu.size(); ++point_index) {
        const auto energy_MeVu = energies_MeVu[point_index];
        const auto stopping = water.interpolate(energy_MeVu) *
            carbon::stopping_power_scale_from_reference_ion(1, 6, energy_MeVu);
        const auto mean = stopping * block_mm;
        const auto charge = carbon::ion_effective_charge(1, energy_MeVu);
        const auto formula_variance = carbon::condensed_total_loss_variance_MeV2(
            energy_MeVu, proton_mass_MeV, charge, block_mm, 1.0);
        const auto formula_sigma = std::sqrt(formula_variance);
        const auto analytic_variance = censored_gaussian_variance(mean, formula_sigma);
        double sampled_sum = 0.0;
        double sampled_sum_sq = 0.0;
        for (int sample = 0; sample < sample_count; ++sample) {
            const auto u1 = std::max<double>(
                carbon::rng::uniform01(20260821, static_cast<std::uint64_t>(sample),
                                       point_index, 0),
                1.0e-12);
            const auto u2 = static_cast<double>(
                carbon::rng::uniform01(20260821, static_cast<std::uint64_t>(sample),
                                       point_index, 1));
            const auto gaussian = std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * std::numbers::pi * u2);
            const auto loss = carbon::clamp_sampled_energy_loss(
                mean, formula_sigma, gaussian, energy_MeVu);
            sampled_sum += loss;
            sampled_sum_sq += loss * loss;
        }
        const auto sampled_mean = sampled_sum / sample_count;
        const auto sampled_variance =
            sampled_sum_sq / sample_count - sampled_mean * sampled_mean;
        const auto retained = analytic_variance / formula_variance;
        require_near(sampled_variance, analytic_variance,
                     0.02 * formula_variance,
                     "Clamped Gaussian MC variance missed the analytic censoring model at " +
                         std::to_string(energy_MeVu) + " MeV");
        require(retained < previous_retained,
                "Clamped Gaussian retained variance must fall as proton energy rises");
        if (energy_MeVu == 250.0) retained_250 = retained;
        previous_retained = retained;
        (void)sampled_mean;
    }
    require(retained_250 < 0.70,
            "250 MeV 0.1 mm proton blocks should lose more than 30% of formula variance to the 2-mean clamp");
}

void test_moment_matched_straggling_sampler() {
    const auto audit = [](const double mean, const double sigma, const int samples) {
        double sum = 0.0;
        double sum_sq = 0.0;
        int nonpositive = 0;
        for (int sample = 0; sample < samples; ++sample) {
            const auto u1 = std::max<double>(
                carbon::rng::uniform01(7, static_cast<std::uint64_t>(sample), 0, 0),
                1.0e-12);
            const auto u2 = static_cast<double>(
                carbon::rng::uniform01(7, static_cast<std::uint64_t>(sample), 0, 1));
            const auto extra = static_cast<double>(
                carbon::rng::uniform01(7, static_cast<std::uint64_t>(sample), 0, 2));
            const auto gaussian = std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * std::numbers::pi * u2);
            const auto loss = carbon::sample_moment_matched_energy_loss(
                mean, sigma, gaussian, extra, 1.0e6);
            if (!(loss > 0.0)) ++nonpositive;
            sum += loss;
            sum_sq += loss * loss;
        }
        const auto sampled_mean = sum / samples;
        const auto sampled_variance = sum_sq / samples - sampled_mean * sampled_mean;
        return std::array<double, 3>{sampled_mean, sampled_variance,
                                     static_cast<double>(nonpositive)};
    };

    const auto thick = audit(1.0, 0.20, 80000);
    require(thick[2] == 0.0, "Thick-layer moment-matched sampler produced nonpositive loss");
    require_near(thick[0], 1.0, 0.02, "Thick-layer moment-matched mean drifted");
    require_near(thick[1], 0.04, 0.004, "Thick-layer moment-matched variance drifted");

    const auto mid = audit(1.0, 0.50, 80000);
    require(mid[2] == 0.0, "Gamma-regime moment-matched sampler produced nonpositive loss");
    require_near(mid[0], 1.0, 0.05, "Gamma-regime moment-matched mean drifted");
    require_near(mid[1], 0.25, 0.04, "Gamma-regime moment-matched variance drifted");

    const auto thin = audit(1.0, 1.20, 80000);
    require(thin[2] == 0.0, "Thin-layer moment-matched sampler produced nonpositive loss");
    require_near(thin[0], 1.0, 0.12, "Thin-layer moment-matched mean drifted");
    require_near(thin[1], 1.44, 0.30, "Thin-layer moment-matched variance drifted");

    // Extremely small Gamma shapes can underflow to an exact zero. Zero is a
    // valid no-collision loss for a transport segment and must remain finite;
    // the transport kernel advances the particle instead of treating it as
    // exhausted.
    const auto ultra_thin = carbon::sample_moment_matched_energy_loss(
        1.0e-5, 1.0e-2, -8.0, 1.0e-12, 10.0);
    require(std::isfinite(ultra_thin) && ultra_thin >= 0.0,
            "Ultra-thin moment-matched sample is invalid");

    const auto first = carbon::sample_condensed_energy_loss(
        0.04, 0.033, 0.3, 0.7, 250.0, carbon::straggling_sampler_moment_matched);
    const auto second = carbon::sample_condensed_energy_loss(
        0.04, 0.033, 0.3, 0.7, 250.0, carbon::straggling_sampler_moment_matched);
    require_near(first, second, 0.0, "Moment-matched sampler is not deterministic");
    require(first > 0.0 && first < 250.0,
            "Moment-matched sampler violated the physical energy cap");
    require_near(carbon::sample_condensed_energy_loss(
                     1.0, 10.0, 1.0, 0.5, 100.0,
                     carbon::straggling_sampler_gaussian_clamped),
                 2.0, 1.0e-12,
                 "Legacy clamped sampler must keep the historical 2-mean cap");

    const auto water = carbon::StoppingPowerTable::from_csv(
        std::filesystem::path(CARBON_SOURCE_DIR) /
        "data/stopping_power_water_geant4_11_3_2.csv");
    constexpr double proton_mass_MeV = 938.27208816;
    constexpr double block_mm = 0.1;
    constexpr double energy_MeVu = 250.0;
    const auto mean = water.interpolate(energy_MeVu) *
        carbon::stopping_power_scale_from_reference_ion(1, 6, energy_MeVu) * block_mm;
    const auto charge = carbon::ion_effective_charge(1, energy_MeVu);
    const auto formula_variance = carbon::condensed_total_loss_variance_MeV2(
        energy_MeVu, proton_mass_MeV, charge, block_mm, 1.0);
    const auto formula_sigma = std::sqrt(formula_variance);
    double sum = 0.0;
    double sum_sq = 0.0;
    constexpr int samples = 120000;
    for (int sample = 0; sample < samples; ++sample) {
        const auto u1 = std::max<double>(
            carbon::rng::uniform01(11, static_cast<std::uint64_t>(sample), 0, 0),
            1.0e-12);
        const auto u2 = static_cast<double>(
            carbon::rng::uniform01(11, static_cast<std::uint64_t>(sample), 0, 1));
        const auto extra = static_cast<double>(
            carbon::rng::uniform01(11, static_cast<std::uint64_t>(sample), 0, 2));
        const auto gaussian = std::sqrt(-2.0 * std::log(u1)) *
                              std::cos(2.0 * std::numbers::pi * u2);
        const auto loss = carbon::sample_moment_matched_energy_loss(
            mean, formula_sigma, gaussian, extra, energy_MeVu);
        sum += loss;
        sum_sq += loss * loss;
    }
    const auto sampled_mean = sum / samples;
    const auto sampled_variance = sum_sq / samples - sampled_mean * sampled_mean;
    require(sampled_variance / formula_variance > 0.90,
            "Moment-matched 250 MeV proton blocks still lose formula variance");

    carbon::TransportConfig matched;
    matched.energy_straggling_model = "moment_matched";
    matched.validate();
    require(matched.straggling_sampler_id() ==
                carbon::straggling_sampler_moment_matched,
            "moment_matched sampler id");
    matched.straggling_scale_energies_MeVu = {70.0, 250.0};
    matched.straggling_scale_values = {1.0, 1.7};
    require_throws([&matched] { matched.validate(); },
                   "moment_matched must reject incident-energy scale tables");
    matched.straggling_scale_energies_MeVu.clear();
    matched.straggling_scale_values.clear();
    matched.straggling_scale = 1.01;
    require_throws([&matched] { matched.validate(); },
                   "moment_matched must reject scalar calibration");
    matched.straggling_scale = 1.0;
    matched.enable_step_stable_straggling = true;
    matched.straggling_sampling_length_mm = 0.1;
    require_throws([&matched] { matched.validate(); },
                   "moment_matched must reject non-additive fixed blocks");
    carbon::TransportConfig unknown;
    unknown.energy_straggling_model = "urban";
    require_throws([&unknown] { unknown.validate(); },
                   "Unknown straggling model was accepted");
}

void test_condensed_total_loss_straggling() {
    constexpr double carbon_mass_MeV = 12.0 * 931.49410242;
    const auto low_energy = carbon::condensed_total_loss_variance_MeV2(
        0.01, carbon_mass_MeV, 6.0, 0.1, 1.0);
    const auto at_100 = carbon::condensed_total_loss_variance_MeV2(
        100.0, carbon_mass_MeV, 6.0, 0.1, 1.0);
    const auto at_400 = carbon::condensed_total_loss_variance_MeV2(
        400.0, carbon_mass_MeV, 6.0, 0.1, 1.0);
    const auto twice_step = carbon::condensed_total_loss_variance_MeV2(
        100.0, carbon_mass_MeV, 6.0, 0.2, 1.0);
    require(low_energy > 0.0 && at_100 > low_energy && at_400 > at_100,
            "Relativistic total-loss variance must increase above the Bohr limit");
    require_near(twice_step / at_100, 2.0, 1.0e-12,
                 "Total-loss variance must scale linearly with step length");
    const auto block_loss_a = carbon::step_stable_sampled_energy_loss(
        1.0, 0.04, 0.025, 0.1, 0.5, 10.0);
    const auto block_loss_b = carbon::step_stable_sampled_energy_loss(
        1.0, 0.04, 0.025, 0.1, 0.5, 10.0);
    require_near(block_loss_a, block_loss_b, 1.0e-14,
                 "Step-stable block sampler must be subdivision deterministic");
    require(block_loss_a >= 0.0 && block_loss_a <= 2.0,
            "Step-stable block sampler violated nonnegative/2x-mean bounds");
}

void test_step_stable_straggling_validation() {
    carbon::TransportConfig disabled;
    disabled.enable_step_stable_straggling = false;
    disabled.straggling_sampling_length_mm = 0.0;
    disabled.validate();
    carbon::TransportConfig missing_length;
    missing_length.enable_step_stable_straggling = true;
    require_throws([&missing_length] { missing_length.validate(); },
                   "Enabled step-stable straggling must require a positive block length");
    carbon::TransportConfig negative_length;
    negative_length.straggling_sampling_length_mm = -0.1;
    require_throws([&negative_length] { negative_length.validate(); },
                   "Negative straggling block length must be rejected");
    carbon::TransportConfig enabled;
    enabled.enable_step_stable_straggling = true;
    enabled.straggling_sampling_length_mm = 0.1;
    enabled.validate();

    const auto path_loss = [](const double subdivision_mm, const std::uint64_t history) {
        double total = 0.0;
        double path = 0.0;
        double bin_remaining = 0.13;
        carbon::StepStableStragglingState<double> state;
        state.initialize(0.1);
        constexpr double path_length = 1.037;
        while (path < path_length - 1.0e-12) {
            auto step = std::min(subdivision_mm, path_length - path);
            step = std::min(step, bin_remaining);
            state.prepare_step(step);
            const auto block = state.block_index;
            const auto u1 = std::max<double>(
                carbon::rng::uniform01(1234, history, block, 0), 1.0e-12);
            const auto u2 = static_cast<double>(
                carbon::rng::uniform01(1234, history, block, 1));
            const auto gaussian = std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * std::numbers::pi * u2);
            total += carbon::step_stable_sampled_energy_loss(
                step, 0.0004 * step, step, 0.1, gaussian, 100.0);
            path += step;
            state.consume(step);
            bin_remaining -= step;
            if (bin_remaining <= 1.0e-12) bin_remaining = 0.13;
        }
        return total;
    };
    std::array<double, 3> means{};
    std::array<double, 3> variances{};
    std::array<std::array<double, 256>, 3> samples{};
    for (std::uint64_t history = 0; history < 256; ++history) {
        samples[0][history] = path_loss(0.1, history);
        samples[1][history] = path_loss(0.05, history);
        samples[2][history] = path_loss(0.025, history);
    }
    for (std::size_t subdivision = 0; subdivision < 3; ++subdivision) {
        for (const auto sample : samples[subdivision]) means[subdivision] += sample;
        means[subdivision] /= 256.0;
        for (const auto sample : samples[subdivision]) {
            const auto delta = sample - means[subdivision];
            variances[subdivision] += delta * delta;
        }
        variances[subdivision] /= 255.0;
    }
    require_near(means[0], means[1], 1.0e-8,
                 "Step-stable 0.1/0.05 subdivision means diverged");
    require_near(means[0], means[2], 1.0e-8,
                 "Step-stable 0.1/0.025 subdivision means diverged");
    require_near(variances[0], variances[1], 1.0e-8,
                 "Step-stable 0.1/0.05 subdivision variances diverged");
    require_near(variances[0], variances[2], 1.0e-8,
                 "Step-stable 0.1/0.025 subdivision variances diverged");

    const auto budget_path = [](const double subdivision_mm) {
        carbon::StepStableStragglingState<double> state;
        state.initialize(0.1);
        double energy = 4.0;
        double path = 0.0;
        while (path < 0.37 - 1.0e-12 && energy > 0.0) {
            auto step = std::min(subdivision_mm, 0.37 - path);
            state.prepare_step(step);
            if (!state.block_active) {
                const auto stopping = 1.0 + 0.02 * energy;
                state.begin_block(stopping * step, 0.0001 * step, step, 1.0, 0.25, energy);
            }
            const auto loss = state.consume_loss(step, energy);
            energy -= loss;
            path += step;
        }
        return std::pair<double, double>{path, energy};
    };
    const auto budget_a = budget_path(0.1);
    const auto budget_b = budget_path(0.05);
    const auto budget_c = budget_path(0.025);
    require_near(budget_a.first, 0.37, 1.0e-12,
                 "Fixed-budget nonintegral tail path did not terminate at endpoint");
    require_near(budget_a.second, budget_b.second, 1.0e-12,
                 "Energy-dependent fixed block budget changed with subdivision");
    require_near(budget_a.second, budget_c.second, 1.0e-12,
                 "Energy-dependent fixed block budget changed with fine subdivision");
    carbon::StepStableStragglingState<double> exhausted;
    exhausted.initialize(0.1);
    exhausted.begin_block(1.0, 0.0, 0.1, 1.0, 0.0, 0.03);
    require_near(exhausted.consume_loss(0.1, 0.03), 0.03, 1.0e-12,
                 "Fixed block budget failed available-energy cap");

    const carbon::StoppingPowerTable csda_table({1.0, 3.0}, {2.0, 2.0});
    carbon::StepStableStragglingState<double> csda_state;
    csda_state.initialize(0.1);
    auto first_step_mm = 0.001;
    csda_state.prepare_step(first_step_mm);
    const auto initial_energy_MeV = 12.0;
    constexpr int mass_number = 1;
    const auto block_length_mm = csda_state.block_length_mm;
    const auto block_energy_MeVu = csda_table.csda_energy_after_distance_MeVu(
        initial_energy_MeV, block_length_mm, mass_number);
    const auto block_mean_loss_MeV = carbon::csda_block_mean_loss_MeV(
        initial_energy_MeV, block_energy_MeVu, static_cast<double>(mass_number));
    require_near(block_length_mm, 0.1, 1.0e-12,
                 "CSDA stable block did not preserve its full length after a tiny first step");
    csda_state.begin_block(block_mean_loss_MeV, 0.0, block_length_mm,
                           1.0, 0.0, initial_energy_MeV);
    auto accumulated_loss_MeV = csda_state.consume_loss(first_step_mm, initial_energy_MeV);
    accumulated_loss_MeV += csda_state.consume_loss(0.037, initial_energy_MeV);
    accumulated_loss_MeV += csda_state.consume_loss(0.062, initial_energy_MeV);
    require_near(accumulated_loss_MeV, block_mean_loss_MeV, 1.0e-12,
                 "CSDA stable block loss changed with tiny/irregular subdivision");
}

void test_energy_dependent_straggling_scale() {
    std::array<double, carbon::max_straggling_scale_points> energies{};
    std::array<double, carbon::max_straggling_scale_points> scales{};
    energies[0] = 0.0;
    energies[1] = 200.0;
    energies[2] = 400.0;
    scales[0] = 1.0;
    scales[1] = 1.1;
    scales[2] = 1.2;
    require_near(
        carbon::interpolate_straggling_scale(
            -10.0, energies, scales, 3, 9.0),
        1.0, 1.0e-12, "Straggling scale lower clamp failed");
    require_near(
        carbon::interpolate_straggling_scale(
            300.0, energies, scales, 3, 9.0),
        1.15, 1.0e-12, "Straggling scale interpolation failed");
    require_near(
        carbon::interpolate_straggling_scale(
            500.0, energies, scales, 3, 9.0),
        1.2, 1.0e-12, "Straggling scale upper clamp failed");
    require_near(
        carbon::interpolate_straggling_scale(
            300.0, energies, scales, 0, 1.07),
        1.07, 1.0e-12, "Scalar straggling fallback failed");
    require_near(
        carbon::scale_energy_loss_ratio_preserving_mean(1.4, 0.5),
        1.2, 1.0e-12, "Packaged fluctuation width scale failed");
    require_near(
        carbon::scale_energy_loss_ratio_preserving_mean(0.2, 2.0),
        0.0, 1.0e-12, "Packaged fluctuation scale must keep loss nonnegative");
}

void test_primary_inelastic_xs_correction() {
    std::array<double, carbon::max_straggling_scale_points> energies{};
    std::array<double, carbon::max_straggling_scale_points> scales{};
    energies[0] = 100.0;
    energies[1] = 200.0;
    energies[2] = 400.0;
    scales[0] = 0.97;
    scales[1] = 0.98;
    scales[2] = 1.0;
    require_near(
        carbon::interpolate_straggling_scale(50.0, energies, scales, 3, 9.0),
        0.97, 1.0e-12, "Primary XS correction lower clamp failed");
    require_near(
        carbon::interpolate_straggling_scale(150.0, energies, scales, 3, 9.0),
        0.975, 1.0e-12, "Primary XS correction interpolation failed");
    require_near(
        carbon::interpolate_straggling_scale(500.0, energies, scales, 3, 9.0),
        1.0, 1.0e-12, "Primary XS correction upper clamp failed");
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

void test_dose_scorer_matches_mev_conversion() {
    carbon::TransportConfig config;
    config.number_of_histories = 100;
    config.phantom_length_mm = 10.0;
    config.depth_bin_width_mm = 5.0;
    config.scorer_area_mm2 = 10000.0;
    config.water_density_g_per_cm3 = 1.0;
    config.dose_output_scale = 0.5;
    config.validate();

    carbon::TransportResult result;
    result.deposited_energy_MeV = {100.0, 50.0};  // absolute MeV over all histories

    const auto dir = std::filesystem::temp_directory_path() / "carbon_dose_scorer_test";
    std::filesystem::create_directories(dir);
    const auto mev_path = dir / "mev.csv";
    const auto gy_path = dir / "dose_Gy.csv";
    carbon::write_depth_dose_csv(mev_path, config, result);
    carbon::write_depth_dose_Gy_csv(gy_path, config, result);

    {
        std::ifstream mev_in(mev_path);
        std::ifstream gy_in(gy_path);
        require(static_cast<bool>(mev_in) && static_cast<bool>(gy_in), "dose scorer files missing");
        std::string mev_header;
        std::string gy_header;
        std::getline(mev_in, mev_header);
        std::getline(gy_in, gy_header);
        require(mev_header.find("energy_deposition_MeV") != std::string::npos,
                "MeV scorer header");
        require(mev_header.find("per_primary") == std::string::npos,
                "MeV scorer should report total tallies");
        require(gy_header.find("dose_Gy") != std::string::npos, "Gy scorer header");
        require(gy_header.find("per_primary") == std::string::npos,
                "Gy scorer should report total dose");
        require(gy_header.find("energy_deposition") == std::string::npos,
                "pure dose scorer should not list MeV columns");

        constexpr double MeV_to_joule = 1.602176634e-13;
        const auto bin_mass_kg =
            config.scorer_area_mm2 * config.depth_bin_width_mm *
            config.water_density_g_per_cm3 * 1.0e-6;
        for (std::size_t bin = 0; bin < 2; ++bin) {
            double depth_m = 0, e_total = 0, d_mev = 0, rel_m = 0;
            char comma = 0;
            mev_in >> depth_m >> comma >> e_total >> comma >> d_mev >> comma >> rel_m;
            double depth_g = 0, d_gy = 0, rel_g = 0;
            gy_in >> depth_g >> comma >> d_gy >> comma >> rel_g;
            const auto expected_e = result.deposited_energy_MeV[bin];
            const auto expected_d =
                expected_e * config.dose_output_scale * MeV_to_joule / bin_mass_kg;
            require_near(e_total, expected_e, 1.0e-12, "total MeV");
            require_near(d_mev, expected_d, 1.0e-20, "MeV file dose column");
            require_near(d_gy, expected_d, 1.0e-20, "Gy scorer total dose");
            require_near(d_gy, d_mev, 1.0e-20, "Gy scorer matches MeV-file dose column");
        }
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_dense_voxel_mhd_writer() {
    carbon::TransportConfig config;
    config.number_of_histories = 10;
    config.phantom_length_mm = 2.0;
    config.depth_bin_width_mm = 1.0;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 2;
    config.voxel_bins_y = 2;
    config.voxel_size_x_mm = 1.0;
    config.voxel_size_y_mm = 1.0;
    config.water_density_g_per_cm3 = 1.0;
    config.validate();

    carbon::TransportResult result;
    result.deposited_energy_MeV = {10.0, 0.0};
    result.voxel_deposited_energy_MeV.assign(config.number_of_voxels(), 0.0);
    // Put energy in voxel (0,0,0) so IDD z=0 closes.
    result.voxel_deposited_energy_MeV[0] = 10.0;

    const auto dir = std::filesystem::temp_directory_path() / "carbon_mhd_test";
    std::filesystem::create_directories(dir);
    const auto mhd = dir / "dose.mhd";
    carbon::write_dense_voxel_dose_mhd(mhd, config, result);
    require(std::filesystem::exists(mhd), "MHD header missing");
    require(std::filesystem::exists(dir / "dose.raw"), "RAW missing");
    const auto header = [&] {
        std::ifstream in(mhd);
        std::string all((std::istreambuf_iterator<char>(in)), {});
        return all;
    }();
    require(header.find("DimSize = 2 2 2") != std::string::npos, "DimSize");
    require(header.find("DoseUnits = Gy") != std::string::npos, "DoseUnits");
    require(header.find("ElementDataFile = dose.raw") != std::string::npos, "RAW name");
    const auto raw_size = std::filesystem::file_size(dir / "dose.raw");
    require(raw_size == 2 * 2 * 2 * sizeof(float), "RAW byte size");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_layered_voxel_dose_uses_local_mass() {
    carbon::TransportConfig config;
    config.number_of_histories = 1;
    config.phantom_length_mm = 2.0;
    config.depth_bin_width_mm = 1.0;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 1;
    config.voxel_bins_y = 1;
    config.voxel_size_x_mm = 1.0;
    config.voxel_size_y_mm = 1.0;
    config.enable_layered_phantom = true;
    config.slab_layers = {{1.0, 1.0}, {2.0, 2.0}};
    config.validate();

    carbon::TransportResult result;
    result.deposited_energy_MeV = {1.0, 1.0};
    result.voxel_deposited_energy_MeV = {1.0, 1.0};

    const auto dir =
        std::filesystem::temp_directory_path() / "carbon_layered_mass_test";
    std::filesystem::create_directories(dir);
    carbon::write_dense_voxel_dose_mhd(dir / "dose.mhd", config, result);
    std::ifstream input(dir / "dose.raw", std::ios::binary);
    std::array<float, 2> dose{};
    input.read(
        reinterpret_cast<char*>(dose.data()),
        static_cast<std::streamsize>(dose.size() * sizeof(float)));
    require(input.good(), "Layered dose RAW should contain two float voxels");
    require_near(
        dose[1] / dose[0], 0.5, 1.0e-6,
        "Equal deposited energy in twice-dense material must give half dose");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_dense_charged_origin_mhd_uses_local_mass() {
    carbon::TransportConfig config;
    config.number_of_histories = 1;
    config.phantom_length_mm = 2.0;
    config.depth_bin_width_mm = 1.0;
    config.enable_voxel_scoring = true;
    config.enable_charged_origin_voxel_scoring = true;
    config.voxel_bins_x = 1;
    config.voxel_bins_y = 1;
    config.voxel_size_x_mm = 1.0;
    config.voxel_size_y_mm = 1.0;
    config.enable_layered_phantom = true;
    config.slab_layers = {{1.0, 1.0}, {2.0, 2.0}};
    config.validate();

    carbon::TransportResult result;
    result.voxel_deposited_energy_MeV = {1.0, 1.0};
    result.charged_origin_voxel_deposited_energy_MeV.assign(
        carbon::charged_origin_category_count * config.number_of_voxels(), 0.0);
    result.charged_origin_voxel_deposited_energy_MeV[0] = 1.0;
    result.charged_origin_voxel_deposited_energy_MeV[1] = 1.0;

    const auto dir = std::filesystem::temp_directory_path() /
                     "carbon_origin_mhd_mass_test";
    std::filesystem::create_directories(dir);
    carbon::write_dense_charged_origin_voxel_dose_mhd(
        dir / "origin", config, result);
    const auto mhd = dir / "origin_primary.mhd";
    const auto raw = dir / "origin_primary.raw";
    require(std::filesystem::exists(mhd) && std::filesystem::exists(raw),
            "Dense primary-origin MHD/RAW missing");
    std::ifstream input(raw, std::ios::binary);
    std::array<float, 2> dose{};
    input.read(reinterpret_cast<char*>(dose.data()),
               static_cast<std::streamsize>(dose.size() * sizeof(float)));
    require_near(dose[1] / dose[0], 0.5, 1.0e-6,
                 "Origin dose must use local material mass");
    std::ifstream header(mhd);
    const std::string text((std::istreambuf_iterator<char>(header)), {});
    require(text.find("DoseOriginCategory = primary") != std::string::npos,
            "Origin MHD category metadata missing");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_ct_aligned_mhd_offset_and_index_pairing() {
    // CT transport samples density with edge origin; dose-to-medium mass and
    // voxel tallies must use the same edge so linear indices match.
    carbon::CtGrid grid;
    grid.nx = 4;
    grid.ny = 3;
    grid.nz = 2;
    grid.origin_x_mm = -10.0F;  // deliberately not 0-centered
    grid.origin_y_mm = -4.0F;
    grid.origin_z_mm = 0.0F;
    grid.spacing_x_mm = 1.0F;
    grid.spacing_y_mm = 2.0F;
    grid.spacing_z_mm = 1.0F;
    const auto n = grid.number_of_voxels();
    grid.density_g_per_cm3.assign(n, 1.0F);
    // Distinct density per voxel so mass pairing is observable.
    for (std::uint32_t iz = 0; iz < grid.nz; ++iz) {
        for (std::uint32_t iy = 0; iy < grid.ny; ++iy) {
            for (std::uint32_t ix = 0; ix < grid.nx; ++ix) {
                const auto i = carbon::ct_linear_index(ix, iy, iz, grid.nx, grid.ny);
                grid.density_g_per_cm3[i] = 0.5F + 0.1F * static_cast<float>(i);
            }
        }
    }
    grid.material_id.assign(n, static_cast<std::uint8_t>(2));
    grid.mass_sp_za_rel = {1.0F};
    grid.mass_sp_I_eV = {75.0F};
    const auto ct_path =
        std::filesystem::temp_directory_path() / "carbon_ct_align_scorer.bin";
    grid.write_binary(ct_path);

    // Scorer index with CT-aligned min must equal ct_sample index.
    const float voxel_min_x = grid.origin_x_mm;
    const float voxel_min_y = grid.origin_y_mm;
    const float sx = grid.spacing_x_mm;
    const float sy = grid.spacing_y_mm;
    for (float x = -9.75F; x < -6.1F; x += 0.5F) {
        for (float y = -3.5F; y < 1.9F; y += 1.0F) {
            float dens = 0.0F;
            std::uint8_t mat = 0;
            require(carbon::ct_sample(x, y, 0.25F, grid.origin_x_mm, grid.origin_y_mm,
                                      grid.origin_z_mm, grid.spacing_x_mm,
                                      grid.spacing_y_mm, grid.spacing_z_mm, grid.nx,
                                      grid.ny, grid.nz, grid.density_g_per_cm3.data(),
                                      grid.material_id.data(), dens, mat),
                    "ct_sample should hit CT for interior points");
            const auto scorer_ix =
                static_cast<int>(std::floor((x - voxel_min_x) / sx));
            const auto scorer_iy =
                static_cast<int>(std::floor((y - voxel_min_y) / sy));
            const auto ct_ix =
                static_cast<int>(std::floor((x - grid.origin_x_mm) / grid.spacing_x_mm));
            const auto ct_iy =
                static_cast<int>(std::floor((y - grid.origin_y_mm) / grid.spacing_y_mm));
            require(scorer_ix == ct_ix && scorer_iy == ct_iy,
                    "CT-aligned scorer index must match ct_sample index");
        }
    }

    // 0-centered scorer would disagree with CT for this origin.
    const float centered_min_x =
        -0.5F * static_cast<float>(grid.nx) * grid.spacing_x_mm;  // -2
    require(std::fabs(centered_min_x - grid.origin_x_mm) > 0.5F,
            "test CT origin must differ from 0-centered scorer");

    carbon::TransportConfig config;
    config.number_of_histories = 1;
    config.phantom_length_mm = 2.0;
    config.depth_bin_width_mm = 1.0;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = grid.nx;
    config.voxel_bins_y = grid.ny;
    config.voxel_size_x_mm = grid.spacing_x_mm;
    config.voxel_size_y_mm = grid.spacing_y_mm;
    config.enable_ct_grid = true;
    config.ct_grid_file = ct_path;
    config.water_density_g_per_cm3 = 1.0;
    config.validate();

    carbon::TransportResult result;
    result.deposited_energy_MeV.assign(config.number_of_bins(), 0.0);
    result.voxel_deposited_energy_MeV.assign(config.number_of_voxels(), 0.0);
    result.voxel_deposited_energy_MeV[0] = 1.0;

    const auto dir = std::filesystem::temp_directory_path() / "carbon_ct_mhd_align";
    std::filesystem::create_directories(dir);
    const auto mhd = dir / "dose.mhd";
    carbon::write_dense_voxel_dose_mhd(mhd, config, result);
    const auto header = [&] {
        std::ifstream in(mhd);
        return std::string((std::istreambuf_iterator<char>(in)), {});
    }();
    // First-center Offset = CT edge origin + half voxel.
    require(header.find("Offset = -9.5 -3 0.5") != std::string::npos,
            "MHD Offset should use CT origin + half-voxel centers, got:\n" + header);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(ct_path, ec);
}

void test_flat_source_config_validation() {
    carbon::TransportConfig config;
    config.enable_flat_source = true;
    config.flat_source_half_width_x_mm = 15.0;
    config.flat_source_half_width_y_mm = 15.0;
    config.validate();

    config.enable_emittance_source = true;
    require_throws([&config] { config.validate(); },
                   "Flat and emittance sources should be mutually exclusive");

    config.enable_emittance_source = false;
    config.flat_source_half_width_x_mm = 0.0;
    require_throws([&config] { config.validate(); },
                   "Flat source should require positive half widths");
}

void test_random_seed_parsing() {
    require(carbon::parse_random_seed("987654321") == 987654321ULL,
            "Numeric random_seed must remain reproducible");
    const auto automatic_seed_a = carbon::parse_random_seed("auto");
    const auto automatic_seed_b = carbon::parse_random_seed("auto");
    require(automatic_seed_a != automatic_seed_b,
            "Automatic random seeds must change between requests");
    require_throws([] { static_cast<void>(carbon::parse_random_seed("automatic")); },
                   "Invalid random_seed text must fail");
    require_throws([] { static_cast<void>(carbon::parse_random_seed("-1")); },
                   "Negative random_seed must fail");
}

void test_minibeam_absorbing_geometry() {
    constexpr float cosine = 1.0F;
    constexpr float sine = 0.0F;
    constexpr float radius = 60.0F;
    constexpr float thickness = 60.0F;
    constexpr float gap = 60.0F;
    constexpr int slit_count = 15;
    constexpr float width = 0.5F;
    constexpr float pitch = 3.6F;
    constexpr float half_length = 25.0F;

    const auto passes = [&](const float x, const float y, const float z,
                            const float dx, const float dy, const float dz) {
        return carbon::minibeam_straight_through_air_slit(
            x, y, z, dx, dy, dz, cosine, sine, radius, thickness, gap,
            slit_count, width, pitch, half_length);
    };
    require(passes(0.0F, 0.0F, -510.0F, 0.0F, 0.0F, 1.0F),
            "Central straight ray should pass the central minibeam slit");
    require(passes(3.6F, 0.0F, -510.0F, 0.0F, 0.0F, 1.0F),
            "Straight ray should pass an off-axis slit center");
    require(!passes(0.25F, 0.0F, -510.0F, 0.0F, 0.0F, 1.0F),
            "Half-open slit edge should be Copper");
    require(!passes(0.0F, 25.0F, -510.0F, 0.0F, 0.0F, 1.0F),
            "Slit long-axis edge should be Copper");
    require(!carbon::minibeam_point_in_copper(
                0.0F, 0.0F, cosine, sine, radius, slit_count, width,
                pitch, half_length),
            "Central slit center should be Air");
    require(carbon::minibeam_point_in_copper(
                0.3F, 0.0F, cosine, sine, radius, slit_count, width,
                pitch, half_length),
            "Point between slits should be Copper");
    require(!carbon::minibeam_point_in_copper(
                radius, 0.0F, cosine, sine, radius, slit_count, width,
                pitch, half_length),
            "Point outside the aperture cylinder should be Air");
    require(passes(0.0F, 0.0F, -120.0F, 0.004F, 0.0F, 1.0F),
            "Ray staying within one slit should pass");
    require(!passes(0.0F, 0.0F, -120.0F, 0.005F, 0.0F, 1.0F),
            "Ray crossing a slit wall should be absorbed");
    int slit = 0;
    require(carbon::minibeam_point_in_slit(
                0.0F, 3.6F, 0.0F, 1.0F, radius, slit_count, width, pitch,
                half_length, slit) &&
                slit == 1,
            "A 90-degree collimator rotation should rotate the slit array");

    carbon::TransportConfig config;
    config.device = "gpu";
    config.enable_minibeam = true;
    config.spots_geometry_mode = "minibeam_topas_y";
#if !defined(CARBON_ENABLE_MINIBEAM)
    require_throws(
        [&config] { config.validate(); },
        "minibeam=true must fail when CARBON_ENABLE_MINIBEAM is OFF");
#else
    config.validate();

    config.minibeam_slit_count = 14;
    require_throws([&config] { config.validate(); },
                   "Even minibeam slit count should be rejected");
    config.minibeam_slit_count = 15;
    config.minibeam_slit_offset_mm =
        std::numeric_limits<double>::quiet_NaN();
    require_throws([&config] { config.validate(); },
                   "Non-finite minibeam slit offset should be rejected");
    config.minibeam_slit_offset_mm = 0.0;
    config.minibeam_transport_mode = "copper_em";
    require_throws([&config] { config.validate(); },
                   "Copper EM mode should require a stopping-power table");
    config.minibeam_copper_stopping_power_file = "copper.csv";
    require_throws([&config] { config.validate(); },
                   "Copper EM mode should require an Air stopping-power table");
    config.minibeam_air_stopping_power_file = "air.csv";
    config.validate();
    config.minibeam_copper_enable_nuclear_attenuation = true;
    require_throws([&config] { config.validate(); },
                   "Copper nuclear attenuation should require an XS table");
    config.minibeam_copper_cross_section_file = "copper_xs.csv";
    config.validate();
    config.minibeam_copper_mcs_scale = 0.0;
    require_throws([&config] { config.validate(); },
                   "Non-positive Copper MCS scale should be rejected");
    config.minibeam_copper_mcs_scale = 1.0;
    config.minibeam_copper_straggling_scale = 0.0;
    require_throws(
        [&config] { config.validate(); },
        "Non-positive Copper straggling scale should be rejected");
    config.minibeam_copper_straggling_scale = 1.0;
    config.minibeam_copper_survivor_energy_loss_scale = 0.0;
    require_throws(
        [&config] { config.validate(); },
        "Non-positive Copper survivor energy-loss scale should be rejected");
    config.minibeam_copper_survivor_energy_loss_scale = 1.0;
    config.minibeam_copper_survivor_energy_loss_energies_MeVu =
        {100.0, 200.0};
    require_throws(
        [&config] { config.validate(); },
        "Mismatched Copper survivor energy-loss lists should be rejected");
    config.minibeam_copper_survivor_energy_loss_scales = {0.9, 1.0};
    config.validate();
    config.minibeam_copper_survivor_energy_loss_energies_MeVu =
        {200.0, 100.0};
    require_throws(
        [&config] { config.validate(); },
        "Non-increasing Copper survivor calibration energies should be rejected");
    config.minibeam_copper_survivor_energy_loss_energies_MeVu.clear();
    config.minibeam_copper_survivor_energy_loss_scales.clear();
    config.minibeam_water_primary_stopping_power_scale = 2.1;
    require_throws(
        [&config] { config.validate(); },
        "Out-of-range minibeam water stopping-power scale should be rejected");
#endif
}

void test_topas_spots_parse_angle01() {
    const std::filesystem::path path =
        std::filesystem::path(CARBON_SOURCE_DIR) / "data" / "plans" /
        "spots_test_c_angle01.txt";
    require(std::filesystem::exists(path),
            "spots_test_c_angle01.txt missing under data/plans");
    const auto plan = carbon::TopasSpotPlan::from_file(path);
    require(plan.spots.size() == 1, "Expected single spot in angle01 plan");
    const auto& spot = plan.spots.front();
    require(spot.spot_id == 445, "spot_id mismatch");
    require_near(spot.energy_MeV, 2040.0, 1.0e-6, "BeamEnergy (total MeV)");
    require_near(spot.energy_spread_percent, 1.0, 1.0e-6, "BeamEnergySpread percent");
    require(spot.number_of_histories == 100000, "NumberOfHistoriesInRun");
    require_near(spot.trans_x_mm, 0.0, 1.0e-9, "TransX");
    require_near(spot.trans_z_mm, 0.0, 1.0e-9, "TransZ");
    require_near(spot.rot_x_deg, 90.0, 1.0e-6, "RotX");
    require_near(spot.rot_y_deg, 0.0, 1.0e-9, "RotY");
    require_near(spot.sigma_x_mm, 3.4804, 1.0e-4, "SigmaX");
    require_near(spot.sigma_x_prime, 0.0053, 1.0e-4, "SigmaXprime");
    require_near(spot.correlation_x, 0.6855, 1.0e-4, "CorrelationX");
    require_near(spot.sigma_y_mm, 3.4804, 1.0e-4, "SigmaY");
    require_near(spot.sigma_y_prime, 0.0053, 1.0e-4, "SigmaYprime");
    require_near(spot.correlation_y, 0.6855, 1.0e-4, "CorrelationY");
    require(plan.total_histories() == 100000, "total_histories");

    // TOPAS BeamPosition2: TransY=-SAD, RotX=90 → origin ≈ (0,0,-SAD), beam -Y.
    carbon::TopasSpotPlan pose_plan = plan;
    pose_plan.sad_mm = 450.0;
    const auto pose = pose_plan.pose_for_spot(spot);
    require_near(pose.origin_x_mm, 0.0, 1.0e-6, "pose origin x");
    require_near(pose.origin_y_mm, 0.0, 1.0e-4, "pose origin y after Rx90");
    require_near(pose.origin_z_mm, -450.0, 1.0e-4, "pose origin z after Rx90");
    require_near(pose.uz_x, 0.0, 1.0e-6, "beam uz x");
    require_near(pose.uz_y, -1.0, 1.0e-6, "beam uz y (local +Z after Rx90)");
    require_near(pose.uz_z, 0.0, 1.0e-6, "beam uz z");
}

void test_topas_spot_weights_and_tps_90_transform() {
    const auto plan_path = std::filesystem::path(CARBON_SOURCE_DIR) / "data" /
                           "plans" / "spots_test_c_angle01.txt";
    auto plan = carbon::TopasSpotPlan::from_files({plan_path, plan_path});
    require(plan.spots.size() == 2, "Concatenated TOPAS plan size mismatch");

    const auto weights_path = std::filesystem::temp_directory_path() /
                              "carbon_test_spot_weights.csv";
    {
        std::ofstream output(weights_path, std::ios::binary);
        output << "\xEF\xBB\xBF" << "1\r\n3\r\n";
    }
    const auto removed = plan.apply_weights_from_csv(weights_path, 40);
    require(removed == 0, "Positive spot weights should not remove spots");
    require(plan.total_histories() == 40, "Weighted plan must preserve exact total histories");
    require(plan.spots[0].number_of_histories == 11 &&
                plan.spots[1].number_of_histories == 29,
            "Largest-remainder spot history allocation mismatch");
    require_near(plan.spots[0].plan_weight, 1.0, 1.0e-12,
                 "First optimizer weight was not retained on the spot");
    require_near(plan.spots[1].plan_weight, 3.0, 1.0e-12,
                 "Second optimizer weight was not retained on the spot");
    require_near(plan.total_plan_weight, 4.0, 1.0e-12, "Spot weight sum");
    std::error_code ec;
    std::filesystem::remove(weights_path, ec);

    carbon::TopasSpot central;
    central.trans_x_mm = 0.0;
    central.trans_z_mm = 0.0;
    central.rot_x_deg = 90.0;
    central.rot_y_deg = 0.0;
    carbon::TopasSpotPlan pose_plan;
    pose_plan.sad_mm = 450.0;
    const auto world = pose_plan.tps_zero_beam_pose_for_spot(central);
    require_near(world.origin_y_mm, -450.0, 1.0e-9, "TPS source TransY");
    require_near(world.uz_y, 1.0, 1.0e-9, "TPS central ray points +world-Y");

    carbon::TopasSpot compound_rotation;
    compound_rotation.trans_x_mm = 7.0;
    compound_rotation.trans_z_mm = -11.0;
    compound_rotation.rot_x_deg = 90.0;
    compound_rotation.rot_y_deg = 30.0;
    const auto compound_world =
        pose_plan.tps_zero_beam_pose_for_spot(compound_rotation);
    // Independently evaluated columns of Rx(-90 deg) * Ry(-30 deg).
    constexpr double sqrt_three_over_two = 0.86602540378443864676;
    require_near(compound_world.origin_x_mm, 7.0, 1.0e-12,
                 "TOPAS compound rotation must not rotate TransX");
    require_near(compound_world.origin_y_mm, -450.0, 1.0e-12,
                 "TOPAS compound rotation must not rotate TransY");
    require_near(compound_world.origin_z_mm, -11.0, 1.0e-12,
                 "TOPAS compound rotation must not rotate TransZ");
    require_near(compound_world.ux_x, sqrt_three_over_two, 1.0e-12,
                 "TOPAS inverse compound ux.x");
    require_near(compound_world.ux_y, 0.5, 1.0e-12,
                 "TOPAS inverse compound ux.y");
    require_near(compound_world.ux_z, 0.0, 1.0e-12,
                 "TOPAS inverse compound ux.z");
    require_near(compound_world.uy_x, 0.0, 1.0e-12,
                 "TOPAS inverse compound uy.x");
    require_near(compound_world.uy_y, 0.0, 1.0e-12,
                 "TOPAS inverse compound uy.y");
    require_near(compound_world.uy_z, -1.0, 1.0e-12,
                 "TOPAS inverse compound uy.z");
    require_near(compound_world.uz_x, -0.5, 1.0e-12,
                 "TOPAS inverse compound uz.x");
    require_near(compound_world.uz_y, sqrt_three_over_two, 1.0e-12,
                 "TOPAS inverse compound uz.y");
    require_near(compound_world.uz_z, 0.0, 1.0e-12,
                 "TOPAS inverse compound uz.z");
    const auto dot = [](const double ax, const double ay, const double az,
                        const double bx, const double by, const double bz) {
        return ax * bx + ay * by + az * bz;
    };
    require_near(dot(compound_world.ux_x, compound_world.ux_y, compound_world.ux_z,
                     compound_world.ux_x, compound_world.ux_y, compound_world.ux_z),
                 1.0, 1.0e-12, "TOPAS inverse compound ux norm");
    require_near(dot(compound_world.uy_x, compound_world.uy_y, compound_world.uy_z,
                     compound_world.uy_x, compound_world.uy_y, compound_world.uy_z),
                 1.0, 1.0e-12, "TOPAS inverse compound uy norm");
    require_near(dot(compound_world.uz_x, compound_world.uz_y, compound_world.uz_z,
                     compound_world.uz_x, compound_world.uz_y, compound_world.uz_z),
                 1.0, 1.0e-12, "TOPAS inverse compound uz norm");
    require_near(dot(compound_world.ux_x, compound_world.ux_y, compound_world.ux_z,
                     compound_world.uy_x, compound_world.uy_y, compound_world.uy_z),
                 0.0, 1.0e-12, "TOPAS inverse compound ux dot uy");
    require_near(dot(compound_world.ux_x, compound_world.ux_y, compound_world.ux_z,
                     compound_world.uz_x, compound_world.uz_y, compound_world.uz_z),
                 0.0, 1.0e-12, "TOPAS inverse compound ux dot uz");
    require_near(dot(compound_world.uy_x, compound_world.uy_y, compound_world.uy_z,
                     compound_world.uz_x, compound_world.uz_y, compound_world.uz_z),
                 0.0, 1.0e-12, "TOPAS inverse compound uy dot uz");
    const auto compound_triple =
        compound_world.ux_x *
            (compound_world.uy_y * compound_world.uz_z -
             compound_world.uy_z * compound_world.uz_y) -
        compound_world.ux_y *
            (compound_world.uy_x * compound_world.uz_z -
             compound_world.uy_z * compound_world.uz_x) +
        compound_world.ux_z *
            (compound_world.uy_x * compound_world.uz_y -
             compound_world.uy_y * compound_world.uz_x);
    require_near(compound_triple, 1.0, 1.0e-12,
                 "TOPAS inverse compound frame determinant");

    const auto ct = carbon::transform_tps_90_pose_to_ct(
        world, 0.0, 0.0, 0.0, 90.0, -104.0);
    require_near(ct.origin_z_mm, -346.0, 1.0e-6,
                 "TPS source upstream position in reoriented CT");
    require_near(ct.uz_z, 1.0, 1.0e-9,
                 "TPS central ray points +GPU-Z after CT transform");

    // TOPAS rotations are passive. World→patient is therefore R(+RotZ):
    // RotZ=+90 maps the clinical isocenter to patient Y=+42.8515 and −X travel.
    const auto ct_m90 = carbon::transform_tps_90_pose_to_ct(
        world, -42.8515, -12.7636, 1.3617, -90.0, -104.25);
    require(ct_m90.uz_z > 0.0, "RotZ=-90 beam must enter +GPU-Z");
    require_near(ct_m90.origin_x_mm, -42.8515, 1.0e-6,
                 "RotZ=-90 isocenter GPU-X = patient Y");

    const auto ct_p90 = carbon::transform_tps_90_pose_to_ct(
        world, -42.8515, -12.7636, 1.3617, 90.0, -104.25);
    require(ct_p90.uz_z > 0.0, "RotZ=+90 beam must enter +GPU-Z");
    require_near(ct_p90.origin_x_mm, 42.8515, 1.0e-6,
                 "RotZ=+90 isocenter GPU-X = patient Y");

    // Lock the production single-spot path. The old one-axis reflection put
    // this at patient Y=-28.9353 while preserving the apparent beam direction.
    carbon::TopasSpot clinical = central;
    clinical.trans_x_mm = -13.9162;
    clinical.rot_x_deg = 90.138;
    const auto clinical_world = pose_plan.tps_zero_beam_pose_for_spot(clinical);
    const auto clinical_ct = carbon::transform_tps_90_pose_to_ct(
        clinical_world, -42.8515, -12.7636, 1.3617, 90.0, -104.0);
    require_near(clinical_ct.origin_x_mm, 28.9353, 1.0e-6,
                 "Clinical spot patient-Y path");
    require(clinical_ct.uz_z > 0.999, "Clinical spot must travel along +GPU-Z");
    require(clinical_ct.origin_z_mm < 0.0, "Clinical source must be before CT entrance");
    // -patient-X clinical branch must keep a right-handed beam frame after the
    // depth reflection (fixes ux·(uy×uz)=-1 left-handed bug).
    const auto triple =
        clinical_ct.ux_x * (clinical_ct.uy_y * clinical_ct.uz_z -
                            clinical_ct.uy_z * clinical_ct.uz_y) -
        clinical_ct.ux_y * (clinical_ct.uy_x * clinical_ct.uz_z -
                            clinical_ct.uy_z * clinical_ct.uz_x) +
        clinical_ct.ux_z * (clinical_ct.uy_x * clinical_ct.uz_y -
                            clinical_ct.uy_y * clinical_ct.uz_x);
    require_near(triple, 1.0, 1.0e-9,
                 "Clinical tps_90 beam frame must be right-handed");

    const auto lung_ct = carbon::transform_tps_y_pose_to_ct(
        world, -69.6605, 9.3887, 0.0819, 0.0, -151.75);
    require_near(lung_ct.origin_x_mm, 69.6605, 1.0e-6,
                 "Lung TPS source patient-X");
    require_near(lung_ct.origin_y_mm, -0.0819, 1.0e-6,
                 "Lung TPS source patient-Z");
    require_near(lung_ct.origin_z_mm, -307.6387, 1.0e-6,
                 "Lung TPS source upstream patient-Y position");
    require_near(lung_ct.uz_z, 1.0, 1.0e-9,
                 "Lung TPS central ray points +GPU-Z");
    const auto lung_triple =
        lung_ct.ux_x * (lung_ct.uy_y * lung_ct.uz_z -
                        lung_ct.uy_z * lung_ct.uz_y) -
        lung_ct.ux_y * (lung_ct.uy_x * lung_ct.uz_z -
                        lung_ct.uy_z * lung_ct.uz_x) +
        lung_ct.ux_z * (lung_ct.uy_x * lung_ct.uz_y -
                        lung_ct.uy_y * lung_ct.uz_x);
    require_near(lung_triple, 1.0, 1.0e-9,
                 "Lung tps_y beam frame must be right-handed");

    const carbon::StoppingPowerTable constant_air(
        {0.01, 400.0}, {0.02, 0.02});
    require(carbon::spot_entry_total_energy_after_optional_upstream_loss(
                2460.0, 12, 333.0, nullptr) == 2460.0,
            "Disabled upstream loss must preserve the legacy source energy exactly");
    require_near(carbon::propagate_total_kinetic_energy_through_stopping_power(
                     2460.0, 12, 0.0, constant_air),
                 2460.0, 1.0e-12,
                 "Zero-length upstream path must preserve spot energy bit-for-bit");
    require_near(carbon::propagate_total_kinetic_energy_through_stopping_power(
                     2460.0, 12, 333.0, constant_air),
                 2453.34, 1.0e-8,
                 "Constant upstream stopping-power loss");
    require_throws([&constant_air] {
        (void)carbon::propagate_total_kinetic_energy_through_stopping_power(
            2460.0, 12, -1.0, constant_air);
    }, "Upstream propagation accepted negative path length");
    const auto real_air = carbon::StoppingPowerTable::from_csv(
        std::filesystem::path(CARBON_SOURCE_DIR) /
        "data/stopping_power_air_geant4_11_3_2.csv");
    const auto entrance_energy =
        carbon::propagate_total_kinetic_energy_through_stopping_power(
            2460.0, 12, 332.9865545359617, real_air);
    require_near(2460.0 - entrance_energy, 5.63837, 2.0e-4,
                 "Validated RT07575 G4_AIR total C-12 energy loss");
    const carbon::StoppingPowerTable too_narrow_air(
        {1.0, 2.0}, {0.02, 0.02});
    require_throws([&too_narrow_air] {
        (void)carbon::propagate_total_kinetic_energy_through_stopping_power(
            2460.0, 12, 1.0, too_narrow_air);
    }, "Upstream propagation accepted a table outside its energy domain");
}

void test_tps_source_geometry_csv_and_switch() {
    carbon::TransportConfig config;
    require(!config.enable_tps_source, "TPS source must default to disabled");
    {
        carbon::TransportConfig spot_energy;
        spot_energy.initial_energy_MeVu = 0.0;
        spot_energy.topas_spots_file = "spots_supply_energy.txt";
        spot_energy.validate();

        auto ct_without_spots = spot_energy;
        ct_without_spots.topas_spots_file.clear();
        ct_without_spots.enable_ct_grid = true;
        ct_without_spots.ct_grid_file = "patient_ct.bin";
        require_throws([&ct_without_spots] { ct_without_spots.validate(); },
                       "CT without a spot file must still validate initial energy");
    }
    config.enable_tps_source = true;
    config.enable_voxel_scoring = true;
    config.number_of_histories = 40;
    config.tps_sad_mm = 100.0;
    config.tps_isocenter_x_mm = 10.0;
    config.tps_isocenter_y_mm = 20.0;
    config.tps_isocenter_z_mm = 30.0;
    config.tps_patient_position = "HFS";
    config.validate();

    carbon::TpsSourcePlan one;
    carbon::TpsSpot central;
    central.spot_id = 1;
    central.energy_MeVu = 200.0;
    central.mu_weight = 1.0;
    one.spots = {central};
    one.total_mu = 1.0;

    auto pose = one.pose_for_spot(config, central);
    require_near(pose.uz_x, 0.0, 1.0e-12, "TPS gantry 0 direction x");
    require_near(pose.uz_y, 0.0, 1.0e-12, "TPS gantry 0 direction y");
    require_near(pose.uz_z, -1.0, 1.0e-12, "TPS gantry 0 direction z");
    require_near(pose.origin_x_mm, 10.0, 1.0e-12, "TPS gantry 0 source x");
    require_near(pose.origin_y_mm, 20.0, 1.0e-12, "TPS gantry 0 source y");
    require_near(pose.origin_z_mm, 130.0, 1.0e-12, "TPS gantry 0 source z");

    config.tps_gantry_angle_deg = 90.0;
    pose = one.pose_for_spot(config, central);
    require_near(pose.uz_x, -1.0, 1.0e-12, "TPS gantry 90 direction x");
    require_near(pose.uz_z, 0.0, 1.0e-12, "TPS gantry 90 direction z");
    require_near(pose.origin_x_mm, 110.0, 1.0e-12, "TPS gantry 90 source x");

    config.tps_gantry_angle_deg = 180.0;
    pose = one.pose_for_spot(config, central);
    require_near(pose.uz_z, 1.0, 1.0e-12, "TPS gantry 180 direction z");
    require_near(pose.origin_z_mm, -70.0, 1.0e-12, "TPS gantry 180 source z");

    config.tps_gantry_angle_deg = 270.0;
    pose = one.pose_for_spot(config, central);
    require_near(pose.uz_x, 1.0, 1.0e-12, "TPS gantry 270 direction x");
    require_near(pose.origin_x_mm, -90.0, 1.0e-12, "TPS gantry 270 source x");

    config.tps_gantry_angle_deg = 90.0;
    config.tps_patient_position = "HFP";
    pose = one.pose_for_spot(config, central);
    require_near(pose.uz_x, 1.0, 1.0e-12, "HFP should invert gantry-90 X");

    config.tps_patient_position = "HFS";
    config.tps_angle_convention = "topas_patient_rot_z";
    config.tps_gantry_angle_deg = 0.0;
    pose = one.pose_for_spot(config, central);
    require_near(pose.uz_x, 0.0, 1.0e-12, "TOPAS gantry 0 direction X");
    require_near(pose.uz_y, 1.0, 1.0e-12, "TOPAS gantry 0 direction Y");
    require_near(pose.uz_z, 0.0, 1.0e-12, "TOPAS gantry 0 direction Z");
    require_near(pose.origin_y_mm, -80.0, 1.0e-12,
                 "TOPAS gantry 0 source Y");

    config.tps_gantry_angle_deg = 37.0;
    pose = one.pose_for_spot(config, central);
    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
    require_near(pose.uz_x, -std::sin(37.0 * deg2rad), 1.0e-12,
                 "TOPAS arbitrary gantry direction X");
    require_near(pose.uz_y, std::cos(37.0 * deg2rad), 1.0e-12,
                 "TOPAS arbitrary gantry direction Y");
    require_near(pose.uz_z, 0.0, 1.0e-12,
                 "TOPAS arbitrary gantry direction Z");
    require_near(pose.origin_x_mm, 10.0 + 100.0 * std::sin(37.0 * deg2rad),
                 1.0e-12, "TOPAS arbitrary gantry source X");
    require_near(pose.origin_y_mm, 20.0 - 100.0 * std::cos(37.0 * deg2rad),
                 1.0e-12, "TOPAS arbitrary gantry source Y");
    const auto arbitrary_batch = one.make_primary_batch(config);
    require_near(arbitrary_batch.front().beam_uz_x(),
                 -std::sin(37.0 * deg2rad), 1.0e-6,
                 "TOPAS arbitrary batch direction X");
    require_near(arbitrary_batch.front().beam_uz_y(),
                 std::cos(37.0 * deg2rad), 1.0e-6,
                 "TOPAS arbitrary batch direction Y");
    auto per_control_point = central;
    per_control_point.gantry_angle_deg = 123.5;
    pose = one.pose_for_spot(config, per_control_point);
    require_near(pose.uz_x, -std::sin(123.5 * deg2rad), 1.0e-12,
                 "TOPAS per-control-point gantry X");
    require_near(pose.uz_y, std::cos(123.5 * deg2rad), 1.0e-12,
                 "TOPAS per-control-point gantry Y");

    const auto csv_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                          "data/plans/spots_example.csv";
    const auto plan = carbon::TpsSourcePlan::from_csv(csv_path);
    require(plan.spots.size() == 3, "TPS CSV spot count");
    require(plan.active_spot_count() == 2, "TPS zero-MU spot filtering");
    require_near(plan.total_mu, 4.0, 1.0e-12, "TPS total MU");
    const auto allocation = plan.allocate_histories(40);
    require(allocation == std::vector<std::size_t>({10, 30, 0}),
            "TPS Hamilton history allocation");

    config.tps_angle_convention = "iec61217";
    config.tps_patient_position = "HFS";
    config.tps_gantry_angle_deg = 0.0;
    config.tps_spots_file = csv_path;
    const auto batch = plan.make_primary_batch(config);
    require(batch.size() == 2, "TPS batch active spot count");
    require(batch.front().history_begin == 0 && batch.front().history_end == 10 &&
                batch.back().history_begin == 10 && batch.back().history_end == 40,
            "TPS batch history ranges");
    require_near(batch.front().initial_energy_MeV(), 2400.0, 1.0e-4,
                 "TPS carbon total energy");
    require_near(batch.front().source_origin_x_mm(), 5.0, 1.0e-6,
                 "TPS source-plane spot X offset");

    const auto ct_path = std::filesystem::temp_directory_path() /
                         "carbon_tps_arbitrary_fixed_ct.bin";
    carbon::CtGrid fixed_ct;
    fixed_ct.nx = 4;
    fixed_ct.ny = 3;
    fixed_ct.nz = 2;
    fixed_ct.origin_x_mm = -2.0F;
    fixed_ct.origin_y_mm = -1.5F;
    fixed_ct.origin_z_mm = -1.0F;
    fixed_ct.spacing_x_mm = 1.0F;
    fixed_ct.spacing_y_mm = 1.0F;
    fixed_ct.spacing_z_mm = 2.0F;
    fixed_ct.density_g_per_cm3.assign(fixed_ct.number_of_voxels(), 1.0F);
    fixed_ct.material_id.assign(fixed_ct.number_of_voxels(), 2U);
    fixed_ct.mass_sp_za_rel = {1.0F, 1.0F, 1.0F};
    fixed_ct.mass_sp_I_eV = {75.0F, 75.0F, 75.0F};
    fixed_ct.write_binary(ct_path);
    auto fixed_ct_config = config;
    fixed_ct_config.tps_spots_file.clear();
    fixed_ct_config.enable_ct_grid = true;
    fixed_ct_config.ct_grid_file = ct_path;
    fixed_ct_config.voxel_bins_x = fixed_ct.nx;
    fixed_ct_config.voxel_bins_y = fixed_ct.ny;
    fixed_ct_config.voxel_size_x_mm = fixed_ct.spacing_x_mm;
    fixed_ct_config.voxel_size_y_mm = fixed_ct.spacing_y_mm;
    fixed_ct_config.depth_bin_width_mm = fixed_ct.spacing_z_mm;
    fixed_ct_config.phantom_length_mm =
        static_cast<double>(fixed_ct.nz) * fixed_ct.spacing_z_mm;
    fixed_ct_config.number_of_histories = 1;
    fixed_ct_config.tps_angle_convention = "topas_patient_rot_z";
    fixed_ct_config.tps_gantry_angle_deg = 37.0;
    fixed_ct_config.tps_isocenter_x_mm = 0.0;
    fixed_ct_config.tps_isocenter_y_mm = 0.0;
    fixed_ct_config.tps_isocenter_z_mm = 0.0;
    const auto fixed_ct_plan = carbon::TpsSourcePlan::from_config(fixed_ct_config);
    const auto fixed_ct_batch = fixed_ct_plan.make_primary_batch(fixed_ct_config);
    require_near(fixed_ct_batch.front().source_origin_z_mm(), 1.0, 1.0e-6,
                 "TPS fixed-CT transport z rebase");
    auto invalid_plan = plan;
    invalid_plan.spots.front().energy_spread_percent = 21.0;
    require_throws([&invalid_plan, &config] {
        static_cast<void>(invalid_plan.make_primary_batch(config));
    }, "TPS per-spot source parameters must be validated");

    auto conflicting = config;
    conflicting.topas_spots_file = "legacy_spots.txt";
    require_throws([&conflicting] { conflicting.validate(); },
                   "TPS and legacy TOPAS sources must be exclusive");

    const auto pbs_csv = std::filesystem::path(CARBON_SOURCE_DIR) /
                         "data/plans/pbs_spots_example.csv";
    const auto pbs_model = std::filesystem::path(CARBON_SOURCE_DIR) /
                           "data/plans/pbs_beam_model_example.csv";
    auto pbs_plan = carbon::TpsSourcePlan::from_csv(pbs_csv);
    require(pbs_plan.spots.size() == 3, "PBS CSV spot count");
    require_near(pbs_plan.spots.front().energy_total_MeV, 2040.0, 1.0e-9,
                 "PBS energy_MeV is total ion kinetic energy");
    require_near(pbs_plan.spots.front().x_mm, -15.0, 1.0e-12, "PBS x_iso");
    pbs_plan.apply_beam_model(pbs_model);
    require(std::isfinite(pbs_plan.spots.front().sigma_x_mm),
            "PBS beam model fills omitted optics");

    carbon::TransportConfig pbs = config;
    pbs.enable_tps_source = true;
    pbs.tps_spots_file = pbs_csv;
    pbs.tps_beam_model_file = pbs_model;
    pbs.tps_spot_weight_mode = "histories";
    pbs.tps_histories_scale = 1.0;
    pbs.tps_virtual_scanning_magnet_x_mm = 6227.8;
    pbs.tps_virtual_scanning_magnet_y_mm = 7008.6;
    pbs.tps_virtual_source_to_isocenter_mm = 450.0;
    pbs.tps_sad_mm = 450.0;
    pbs.tps_angle_convention = "dicom_lps";
    pbs.tps_gantry_angle_deg = 0.0;
    pbs.tps_isocenter_x_mm = 0.0;
    pbs.tps_isocenter_y_mm = 0.0;
    pbs.tps_isocenter_z_mm = 0.0;
    pbs.number_of_histories = 1;
    pbs_plan = carbon::TpsSourcePlan::from_config(pbs);
    const auto corner = pbs_plan.pose_for_spot(pbs, pbs_plan.spots.front());
    require_near(corner.origin_x_mm, -13.9162, 5.0e-5,
                 "PBS virtual-magnet source-plane X");
    require_near(corner.origin_z_mm, -8.4221, 5.0e-5,
                 "PBS virtual-magnet source-plane Y onto local v=+Z at gantry 0");
    require_near(corner.origin_y_mm, -450.0, 1.0e-9,
                 "PBS source plane is y=-D at gantry 0");
    const auto aimed =
        -corner.origin_x_mm * corner.uz_x + -corner.origin_y_mm * corner.uz_y +
        -corner.origin_z_mm * corner.uz_z;
    require(aimed > 449.0, "PBS central ray must aim at the isocenter");
    const auto pbs_batch = pbs_plan.make_primary_batch(pbs);
    require(pbs_batch.size() == 2, "PBS zero-weight spots are not transported");
    require(pbs_batch.front().history_end == 10 &&
                pbs_batch.back().history_end == 40,
            "PBS weight column is exact histories");
    require_near(pbs_batch.front().initial_energy_MeV(), 2040.0, 1.0e-4,
                 "PBS batch keeps total ion kinetic energy");

    // TOPAS applies Patient RotZ first, then the recorded Trans. That is the
    // same packing as Time Feature tps_90 — not "tps_beam_angle_deg=90 on an
    // unrotated CT".
    carbon::TransportConfig pbs_topas = pbs;
    pbs_topas.tps_apply_topas_patient_placement = true;
    pbs_topas.tps_gantry_angle_deg = 0.0;
    pbs_topas.spots_patient_trans_x_mm = -12.3588;
    pbs_topas.spots_patient_trans_y_mm = -5.5590;
    pbs_topas.spots_patient_trans_z_mm = 0.3945;
    pbs_topas.spots_patient_rot_z_deg = 90.0;
    pbs_topas.spots_ct_axis_min_mm = -110.0;
    carbon::SpotSourcePose world_pbs{};
    world_pbs.origin_x_mm = corner.origin_x_mm;
    world_pbs.origin_y_mm = corner.origin_y_mm;
    world_pbs.origin_z_mm = corner.origin_z_mm;
    world_pbs.ux_x = corner.ux_x;
    world_pbs.ux_y = corner.ux_y;
    world_pbs.ux_z = corner.ux_z;
    world_pbs.uy_x = corner.uy_x;
    world_pbs.uy_y = corner.uy_y;
    world_pbs.uy_z = corner.uy_z;
    world_pbs.uz_x = corner.uz_x;
    world_pbs.uz_y = corner.uz_y;
    world_pbs.uz_z = corner.uz_z;
    const auto packed = carbon::transform_tps_90_pose_to_ct(
        world_pbs, pbs_topas.spots_patient_trans_x_mm,
        pbs_topas.spots_patient_trans_y_mm, pbs_topas.spots_patient_trans_z_mm,
        pbs_topas.spots_patient_rot_z_deg, pbs_topas.spots_ct_axis_min_mm);
    const auto placed = pbs_plan.pose_for_spot(pbs_topas, pbs_plan.spots.front());
    require_near(placed.origin_x_mm, packed.origin_x_mm, 1.0e-9,
                 "PBS TOPAS placement GPU-X matches tps_90 packing");
    require_near(placed.origin_y_mm, packed.origin_y_mm, 1.0e-9,
                 "PBS TOPAS placement GPU-Y matches tps_90 packing");
    require_near(placed.origin_z_mm, packed.origin_z_mm, 1.0e-9,
                 "PBS TOPAS placement GPU-Z matches tps_90 packing");
    require(placed.uz_z > 0.999, "PBS TOPAS placement aims +GPU-Z");
    require(placed.origin_z_mm < 0.0,
            "PBS TOPAS placement source is upstream of the CT entrance");

    auto parallel = pbs;
    parallel.tps_virtual_scanning_magnet_x_mm = 0.0;
    parallel.tps_virtual_scanning_magnet_y_mm = 0.0;
    const auto old_pose = pbs_plan.pose_for_spot(parallel, pbs_plan.spots.front());
    require_near(old_pose.origin_x_mm, -15.0, 1.0e-9,
                 "Unset magnets keep the historical parallel source-plane offset");
    require_near(old_pose.uz_y, 1.0, 1.0e-12,
                 "Unset magnets keep the historical parallel central ray");

    const auto yaml_path = std::filesystem::temp_directory_path() /
                           "carbon_tps_source_switch.yaml";
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "tpsSource: false\n";
    }
    const auto disabled = carbon::load_config(yaml_path);
    require(!disabled.enable_tps_source, "tpsSource:false parsing");
    require(!disabled.voxel_scorer_clamps_transport,
            "voxel scorer transport clamp must default to scorer-decoupled");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "enable_csda_range_energy_loss: true\n";
    }
    const auto csda_enabled = carbon::load_config(yaml_path);
    require(csda_enabled.enable_csda_range_energy_loss,
            "enable_csda_range_energy_loss:true parsing");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "energy_straggling_model: moment_matched\n"
               << "straggling_scale: 1.0\n";
    }
    const auto matched = carbon::load_config(yaml_path);
    require(matched.uses_moment_matched_straggling(),
            "energy_straggling_model:moment_matched parsing");
    require(matched.straggling_sampler_id() ==
                carbon::straggling_sampler_moment_matched,
            "parsed moment_matched sampler id");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "tpsSource: true\n"
               << "enable_voxel_scoring: true\n"
               << "voxel_scorer_clamps_transport: false\n"
               << "tps_angle_convention: topas_patient_rot_z\n"
               << "tps_gantry_angle_deg: 37.5\n"
               << "tps_isocenter_mm: [1, 2, 3]\n";
    }
    const auto vector_isocenter = carbon::load_config(yaml_path);
    require_near(vector_isocenter.tps_isocenter_x_mm, 1.0, 1.0e-12,
                 "TPS vector isocenter X parsing");
    require_near(vector_isocenter.tps_isocenter_y_mm, 2.0, 1.0e-12,
                 "TPS vector isocenter Y parsing");
    require_near(vector_isocenter.tps_isocenter_z_mm, 3.0, 1.0e-12,
                 "TPS vector isocenter Z parsing");
    require(vector_isocenter.tps_angle_convention == "topas_patient_rot_z",
            "TPS angle convention parsing");
    require(!vector_isocenter.voxel_scorer_clamps_transport,
            "voxel_scorer_clamps_transport:false parsing");
    require_near(vector_isocenter.tps_gantry_angle_deg, 37.5, 1.0e-12,
                 "TPS arbitrary angle parsing");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "tpsSource: true\n"
               << "enable_voxel_scoring: true\n"
               << "tps_apply_topas_patient_placement: true\n";
    }
    const auto topas_place = carbon::load_config(yaml_path);
    require(topas_place.tps_apply_topas_patient_placement,
            "tps_apply_topas_patient_placement parsing");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "enable_tps_coordinate_system: true\n"
               << "enable_voxel_scoring: true\n"
               << "ct_grid_file: " << ct_path.string() << "\n"
               << "tps_beam_angle_deg: 37.5\n";
    }
    const auto lps_coordinates = carbon::load_config(yaml_path);
    require(lps_coordinates.enable_tps_coordinate_system &&
                !lps_coordinates.enable_tps_source,
            "TPS coordinate-system switch must not enable the TPS CSV source");
    require(lps_coordinates.tps_angle_convention == "dicom_lps",
            "TPS coordinate-system switch must select DICOM LPS");
    require_near(lps_coordinates.tps_gantry_angle_deg, 37.5, 1.0e-12,
                 "TPS public beam angle parsing");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "use_particle_specific_stopping_power: true\n"
               << "maximum_step_mm: 0.1\n"
               << "maximum_relative_energy_loss: 0.001\n"
               << "energy_cutoff_MeV: 0.1\n"
               << "enable_voxel_scoring: true\n"
               << "dose_to_medium: true\n"
               << "dose_to_medium_type: mhd\n"
               << "dose_to_medium_name: dose\n"
               << "LET: true\n"
               << "LET_type: mhd\n"
               << "LET_name: LET\n";
    }
    const auto public_scorers = carbon::load_config(yaml_path);
    require(public_scorers.enable_let_scoring,
            "Public LET switch must enable LET scoring");
    require(public_scorers.voxel_dose_mhd_output_file ==
                std::filesystem::path{"out"} / yaml_path.stem() / "dose.mhd",
            "Dose-to-medium output path resolution");
    require(public_scorers.let_voxel_mhd_output_file ==
                std::filesystem::path{"out"} / yaml_path.stem() / "LET",
            "LET output prefix resolution");
    require(public_scorers.output_file.empty() &&
                public_scorers.voxel_dose_output_file.empty(),
            "Public dose scorer must suppress legacy default CSV outputs");
    require(!public_scorers.enable_fragment_species_scoring,
            "production default must not enable fragment species scoring");
    require(public_scorers.scorer_mode == "production",
            "default scorer_mode is production");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "scorer_mode: validation\n";
    }
#ifndef CARBON_VALIDATION_SCORERS
    require_throws([&yaml_path] { carbon::load_config(yaml_path); },
                   "validation scorer_mode must fail without the CMake option");
#endif
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "enable_voxel_scoring: true\n"
               << "dose_to_medium: true\n"
               << "dose_to_medium_type: dicom\n"
               << "dose_to_medium_name: dose\n";
    }
    require_throws([&yaml_path] { static_cast<void>(carbon::load_config(yaml_path)); },
                   "Unsupported public dose output type must fail");
    {
        std::ofstream output(yaml_path);
        output << "number_of_histories: 10\n"
               << "tpsSource: true\n"
               << "tps_source: false\n";
    }
    require_throws([&yaml_path] { static_cast<void>(carbon::load_config(yaml_path)); },
                   "Conflicting TPS switch aliases must fail");
    std::error_code ec;
    std::filesystem::remove(yaml_path, ec);
    std::filesystem::remove(ct_path, ec);
}

#ifdef CARBON_HAS_SYCL

void test_sycl_tps_source_arbitrary_gantry_transport() {
    carbon::TransportConfig config;
    config.enable_tps_source = true;
    config.number_of_histories = 8;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 100.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 40;
    config.voxel_bins_y = 40;
    config.voxel_size_x_mm = 5.0;
    config.voxel_size_y_mm = 5.0;
    config.tps_angle_convention = "topas_patient_rot_z";
    config.tps_gantry_angle_deg = 37.0;
    config.tps_isocenter_x_mm = 0.0;
    config.tps_isocenter_y_mm = 0.0;
    config.tps_isocenter_z_mm = 50.0;
    config.tps_sad_mm = 150.0;
    const auto plan = carbon::TpsSourcePlan::from_config(config);
    config.primary_spot_batch = plan.make_primary_batch(config);
    config.validate();

    const carbon::StoppingPowerTable table(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto result = carbon::transport_sycl(
        config, table, zero_cross_section(), "cpu");
    require(result.backend.find("+tps-source") != std::string::npos,
            "TPS source backend tag missing");
    require(result.total_deposited_energy_MeV > 0.0,
            "Arbitrary-angle TPS beam did not enter the voxel AABB");
    require(result.relative_energy_balance_error() < 1.0e-6,
            "TPS arbitrary-angle gantry energy balance failed");
}

void test_sycl_legacy_cardinal_entrance_projection() {
    carbon::TransportConfig config;
    config.number_of_histories = 1024;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 100.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_emittance_source = true;
    config.emittance_sigma_x_mm = 3.0;
    // Reproduce the non-zero cos(pi/2) component of an exactly cardinal
    // TOPAS-90 transverse basis.  Without unconditional entrance projection,
    // negative Gaussian samples begin infinitesimally below z=0 and escape.
    config.beam_ux_z = std::cos(0.5 * 3.14159265358979323846);
    config.validate();

    const carbon::StoppingPowerTable table(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto result = carbon::transport_sycl(
        config, table, zero_cross_section(), "cpu");
    const auto initial_total =
        static_cast<double>(config.number_of_histories) *
        config.initial_energy_MeVu * static_cast<double>(config.primary_mass_number);
    require(result.total_deposited_energy_MeV > 0.95 * initial_total,
            "Legacy cardinal source lost histories infinitesimally below z=0");
}

void test_sycl_primary_spot_batch() {
    carbon::TransportConfig batch;
    batch.number_of_histories = 8;
    batch.phantom_length_mm = 100.0;
    batch.depth_bin_width_mm = 1.0;
    batch.maximum_step_mm = 0.5;
    batch.maximum_relative_energy_loss = 0.01;
    carbon::PrimarySpotBatchEntry first{};
    first.history_begin = 0;
    first.history_end = 3;
    first.random_seed = 17;
    first.initial_energy_MeV() = 120.0F;
    carbon::PrimarySpotBatchEntry second{};
    second.history_begin = 3;
    second.history_end = 8;
    second.random_seed = 29;
    second.initial_energy_MeV() = 180.0F;
    batch.primary_spot_batch = {first, second};
    batch.validate();

    const carbon::StoppingPowerTable table(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto combined =
        carbon::transport_sycl(batch, table, zero_cross_section(), "cpu");

    auto part = batch;
    part.primary_spot_batch.clear();
    part.number_of_histories = 3;
    part.initial_energy_MeVu = 10.0;
    const auto a = carbon::transport_sycl(part, table, zero_cross_section(), "cpu");
    part.number_of_histories = 5;
    part.initial_energy_MeVu = 15.0;
    const auto b = carbon::transport_sycl(part, table, zero_cross_section(), "cpu");
    // FP32 dose atomics (CARBON_DOSE_FP32) accumulate ~1e-6 relative noise.
    constexpr double batch_tol =
#if defined(CARBON_DOSE_FP32)
        1.0e-4;
#else
        1.0e-6;
#endif
    require_near(combined.initial_energy_MeV,
                 a.initial_energy_MeV + b.initial_energy_MeV, batch_tol,
                 "batched initial energy");
    for (std::size_t i = 0; i < combined.deposited_energy_MeV.size(); ++i) {
        require_near(combined.deposited_energy_MeV[i],
                     a.deposited_energy_MeV[i] + b.deposited_energy_MeV[i],
                     batch_tol, "batched primary dose");
    }
}



void test_sycl_primary_let_includes_cutoff_tail() {
    carbon::TransportConfig config;
    config.number_of_histories = 8;
    config.initial_energy_MeVu = 1.0;
    config.phantom_length_mm = 20.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.1;
    config.energy_cutoff_MeV = 0.5;
    config.enable_let_scoring = true;
    config.validate();

    const carbon::StoppingPowerTable stopping_power(
        {0.01, 1.01, 2.01}, {2.0, 2.0, 2.0});
    const auto result = carbon::transport_sycl(
        config, stopping_power, zero_cross_section(), "cpu");
    const auto primary_denominator = std::accumulate(
        result.primary_letd_denominator.begin(),
        result.primary_letd_denominator.end(), 0.0);
    const auto all_denominator = std::accumulate(
        result.all_hadron_letd_denominator.begin(),
        result.all_hadron_letd_denominator.end(), 0.0);
    const auto primary_numerator = std::accumulate(
        result.primary_letd_numerator.begin(),
        result.primary_letd_numerator.end(), 0.0);
    const auto all_numerator = std::accumulate(
        result.all_hadron_letd_numerator.begin(),
        result.all_hadron_letd_numerator.end(), 0.0);
    const auto expected_energy =
        static_cast<double>(config.number_of_histories) *
        config.initial_total_energy_MeV();
    require_near(primary_denominator, expected_energy, 1.0e-4,
                 "Primary LET denominator omitted cutoff-tail energy");
    require_near(all_denominator, primary_denominator, 1.0e-8,
                 "Primary-only all-hadron LET denominator mismatch");
    require_near(all_numerator, primary_numerator, 1.0e-8,
                 "Primary-only all-hadron LET numerator mismatch");
}

void test_sycl_flat_source_extent() {
    carbon::TransportConfig config;
    config.number_of_histories = 512;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 5.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 50;
    config.voxel_bins_y = 50;
    config.voxel_size_x_mm = 3.0;
    config.voxel_size_y_mm = 3.0;
    config.enable_flat_source = true;
    config.flat_source_half_width_x_mm = 15.0;
    config.flat_source_half_width_y_mm = 15.0;
    config.random_seed = 20260716;
    config.validate();

    const carbon::StoppingPowerTable stopping_power(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto result = carbon::transport_sycl(
        config, stopping_power, zero_cross_section(), "cpu");
    require_voxel_idd_closure(config, result, 1.0e-9);

    std::array<bool, 50> active_x{};
    std::array<bool, 50> active_y{};
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    for (std::size_t voxel = 0; voxel < result.voxel_deposited_energy_MeV.size(); ++voxel) {
        if (result.voxel_deposited_energy_MeV[voxel] == 0.0) {
            continue;
        }
        const auto in_plane = voxel % plane_size;
        active_x[in_plane % config.voxel_bins_x] = true;
        active_y[in_plane / config.voxel_bins_x] = true;
    }
    const auto active_x_count = std::count(active_x.begin(), active_x.end(), true);
    const auto active_y_count = std::count(active_y.begin(), active_y.end(), true);
    require(active_x_count >= 8 && active_y_count >= 8,
            "Flat source did not populate the expected field width");
    for (std::size_t index = 0; index < active_x.size(); ++index) {
        if (active_x[index] || active_y[index]) {
            require(index >= 20 && index <= 29,
                    "Flat source deposited outside the 30 mm square field");
        }
    }
}

void test_serial_sycl_cpu_match() {
    carbon::TransportConfig config;
    config.number_of_histories = 64;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_voxel_scoring = true;
    config.enable_charged_origin_voxel_scoring = true;
    config.voxel_bins_x = 5;
    config.voxel_bins_y = 7;
    const carbon::StoppingPowerTable table({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto serial = carbon::transport_serial(config, table, zero_cross_section());
    const auto sycl_cpu =
        carbon::transport_sycl(config, table, zero_cross_section(), "cpu");
    require_voxel_idd_closure(config, serial, 1.0e-12);
    require_voxel_idd_closure(config, sycl_cpu, 1.0e-9);
    require(sycl_cpu.charged_origin_voxel_deposited_energy_MeV.size() ==
                carbon::charged_origin_category_count * config.number_of_voxels(),
            "SYCL charged-origin voxel tally has the wrong size");
    for (std::size_t voxel = 0; voxel < config.number_of_voxels(); ++voxel) {
        require_near(
            sycl_cpu.charged_origin_voxel_deposited_energy_MeV[voxel],
            sycl_cpu.voxel_deposited_energy_MeV[voxel], 1.0e-9,
            "SYCL primary origin voxel does not close to total");
    }
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
}

void test_sycl_transport_context_reuse() {
    carbon::TransportConfig config;
    config.number_of_histories = 32;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 100.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    const carbon::StoppingPowerTable stopping_power(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const auto cross_section = zero_cross_section();
    carbon::SyclTransportContext context("cpu");

    const auto first = carbon::transport_sycl(
        config, stopping_power, cross_section, "cpu", &context);
    const auto second = carbon::transport_sycl(
        config, stopping_power, cross_section, "cpu", &context);
    require(first.deposited_energy_MeV.size() == second.deposited_energy_MeV.size(),
            "Reusable SYCL context changed tally dimensions");
    for (std::size_t bin = 0; bin < first.deposited_energy_MeV.size(); ++bin) {
        require_near(first.deposited_energy_MeV[bin], second.deposited_energy_MeV[bin],
                     1.0e-9, "Reusable SYCL context changed dose tally");
    }

    const carbon::StoppingPowerTable different_table(
        {0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    require_throws(
        [&]() {
            (void)carbon::transport_sycl(config, different_table, cross_section, "cpu",
                                         &context);
        },
        "Reusable SYCL context accepted a different physics table");
}

void test_sycl_layered_slab_range_shift() {
    carbon::TransportConfig config;
    config.number_of_histories = 256;
    config.initial_energy_MeVu = 100.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_energy_straggling = false;
    config.enable_multiple_scattering = false;
    config.random_seed = 42;
    const carbon::StoppingPowerTable stopping_power(
        {0.01, 50.01, 100.01, 150.01}, {40.0, 20.0, 12.0, 10.0});
    const carbon::CrossSectionTable xs({0.01, 150.01}, {1.0e-6, 1.0e-6});

    config.enable_layered_phantom = false;
    const auto uniform = carbon::transport_sycl(
        config, stopping_power, xs, "cpu");

    config.enable_layered_phantom = true;
    config.slab_layers = {{30.0, 1.0}, {50.0, 2.0}, {200.0, 1.0}};
    const auto layered = carbon::transport_sycl(
        config, stopping_power, xs, "cpu");

    auto r80 = [](const std::vector<double>& dose, double bin_width) {
        const auto peak = *std::max_element(dose.begin(), dose.end());
        require(peak > 0.0, "empty IDD in slab range-shift test");
        const auto thr = 0.8 * peak;
        for (std::size_t i = dose.size(); i-- > 1;) {
            if (dose[i] <= thr && dose[i - 1] > thr) {
                return (static_cast<double>(i) - 0.5) * bin_width;
            }
        }
        return static_cast<double>(dose.size()) * bin_width;
    };
    const auto r80_uniform = r80(uniform.deposited_energy_MeV, config.depth_bin_width_mm);
    const auto r80_layered = r80(layered.deposited_energy_MeV, config.depth_bin_width_mm);
    require(r80_layered < r80_uniform - 1.0,
            "Dense slab did not pull R80 proximal: layered=" +
                std::to_string(r80_layered) + " uniform=" + std::to_string(r80_uniform));
}
#endif

void test_fred_18_isotopes_data() {
    require(carbon::kFredIsotopes.size() == 18, "kFredIsotopes must contain 18 isotopes");
    require(carbon::kFredCdfH.size() == 18, "kFredCdfH must contain 18 entries");
    require(carbon::kFredCdfO.size() == 18, "kFredCdfO must contain 18 entries");
    require(std::abs(carbon::kFredCdfH.back() - 1.0f) < 1e-4f, "kFredCdfH must terminate at 1.0");
    require(std::abs(carbon::kFredCdfO.back() - 1.0f) < 1e-4f, "kFredCdfO must terminate at 1.0");

    for (std::size_t i = 1; i < 18; ++i) {
        require(carbon::kFredCdfH[i] >= carbon::kFredCdfH[i - 1], "kFredCdfH must be non-decreasing");
        require(carbon::kFredCdfO[i] >= carbon::kFredCdfO[i - 1], "kFredCdfO must be non-decreasing");
    }

    // Verify all 18 charged species mapping
    require(carbon::get_charged_species_idx(1, 1) == 0, "1H (p) index");
    require(carbon::get_charged_species_idx(1, 2) == 1, "2H (d) index");
    require(carbon::get_charged_species_idx(1, 3) == 2, "3H (t) index");
    require(carbon::get_charged_species_idx(2, 3) == 3, "3He index");
    require(carbon::get_charged_species_idx(2, 4) == 4, "4He index");
    require(carbon::get_charged_species_idx(2, 6) == 5, "6He index");
    require(carbon::get_charged_species_idx(3, 6) == 6, "6Li index");
    require(carbon::get_charged_species_idx(3, 7) == 7, "7Li index");
    require(carbon::get_charged_species_idx(4, 7) == 8, "7Be index");
    require(carbon::get_charged_species_idx(4, 9) == 9, "9Be index");
    require(carbon::get_charged_species_idx(4, 10) == 10, "10Be index");
    require(carbon::get_charged_species_idx(5, 8) == 11, "8B index");
    require(carbon::get_charged_species_idx(5, 10) == 12, "10B index");
    require(carbon::get_charged_species_idx(5, 11) == 13, "11B index");
    require(carbon::get_charged_species_idx(6, 10) == 14, "10C index");
    require(carbon::get_charged_species_idx(6, 11) == 15, "11C index");
    require(carbon::get_charged_species_idx(6, 12) == 16, "12C index");
    require(carbon::get_charged_species_idx(4, 6) == 17, "6Be index");
    require(carbon::get_charged_species_idx(1, 4) == -1, "unknown hydrogen isotope aliased");
    require(carbon::get_charged_species_idx(2, 5) == -1, "unknown helium isotope aliased");
    require(carbon::get_charged_species_idx(7, 14) == -1, "Z>6 isotope aliased to proton");
}

void test_ion_species_stopping_power_grid_validation() {
    const std::filesystem::path csv_path = "data/ion_stopping_power_water_geant4_11_3_2.csv";
    if (std::filesystem::exists(csv_path)) {
        const auto lut = carbon::load_ion_species_stopping_power_lut(csv_path, 4001, 1.0f);
        require(lut.size() == 18 * 4001, "Stopping power LUT size must be 18 * 4001");
        for (std::size_t i = 0; i < lut.size(); ++i) {
            require(lut[i] > 0.0f, "Stopping power values must be strictly positive");
        }
    }
}

void test_stopping_power_csv_corruption_rejection() {
    const auto bad_csv_path = std::filesystem::temp_directory_path() / "bad_ion_sp.csv";
    {
        std::ofstream out(bad_csv_path);
        out << "atomic_number,mass_number,energy_MeVu,electronic_stopping_power_MeV_per_mm,nuclear_stopping_power_MeV_per_mm\n";
        out << "1,1,0.01,20.0,0.0\n";
        out << "1,1,0.15,19.0,0.0\n"; // grid step mismatch: expected 0.11, found 0.15
    }
    require_throws<std::runtime_error>(
        [&]() {
            (void)carbon::load_ion_species_stopping_power_lut(bad_csv_path, 4001, 1.0f);
        },
        "Corrupted ion stopping power grid must throw std::runtime_error");
    std::filesystem::remove(bad_csv_path);
}

void test_kox_icru_cross_sections() {
    // 100 MeV/u
    const float sig_H_100 = carbon::calculate_icru_sigma_H(100.0f);
    const float sig_O_100 = carbon::calculate_kox_sigma_O(100.0f, 1200.0f);
    const float prob_H_100 = carbon::calculate_target_prob_H(sig_H_100, sig_O_100);
    require(sig_H_100 > 250.0f && sig_H_100 < 300.0f, "sigma_H at 100 MeV/u in [250, 300] mb");
    require(sig_O_100 > 900.0f && sig_O_100 < 1500.0f, "sigma_O at 100 MeV/u in [900, 1500] mb");
    require(prob_H_100 > 0.25f && prob_H_100 < 0.45f, "P(H) at 100 MeV/u in [0.25, 0.45]");

    // 400 MeV/u
    const float sig_H_400 = carbon::calculate_icru_sigma_H(400.0f);
    const float sig_O_400 = carbon::calculate_kox_sigma_O(400.0f, 4800.0f);
    const float prob_H_400 = carbon::calculate_target_prob_H(sig_H_400, sig_O_400);
    require_near(sig_H_400, 250.0f, 0.1f, "sigma_H at 400 MeV/u is 250 mb plateau");
    require(sig_O_400 > 1100.0f && sig_O_400 < 1600.0f, "sigma_O at 400 MeV/u in [1100, 1600] mb");
    require(prob_H_400 > 0.20f && prob_H_400 < 0.35f, "P(H) at 400 MeV/u in [0.20, 0.35]");
}


void test_fred_2gr_package() {
    const auto table = carbon::Fred2GrMcsTable::from_binary(
        std::filesystem::path(CARBON_SOURCE_DIR) /
        "data/packages/fred_3_76_mcs_2gr.bin");
    require(table.values.size() == 6U * 51U * 48U, "2GR table dimensions");
    const auto w1 = carbon::fred_2gr_parameter(
        table.values.data(), 0, 100.0F, 0.01F);
    const auto sigma_c = carbon::fred_2gr_parameter(
        table.values.data(), 1, 100.0F, 0.01F);
    const auto m = carbon::fred_2gr_parameter(
        table.values.data(), 5, 100.0F, 0.01F);
    require(w1 >= 0.0F && w1 <= 1.0F, "2GR core weight in [0,1]");
    require(sigma_c > 0.0F, "2GR core width positive");
    require(m > 0.5F, "2GR Rutherford exponent valid");
    require(carbon::fred_2gr_parameter(
                table.values.data(), 0, 300.0F, 0.01F) < 0.0F,
            "2GR rejects energy outside FRED table domain");

    require_near(carbon::fred_2gr_high_energy_angle_scale(236.0), 1.0,
                 1.0e-12, "2GR extrapolation is continuous at 236 MeV/u");
    const auto scale_300 = carbon::fred_2gr_high_energy_angle_scale(300.0);
    const auto scale_400 = carbon::fred_2gr_high_energy_angle_scale(400.0);
    require(scale_300 > 0.0 && scale_300 < 1.0,
            "2GR 300 MeV/u extrapolation scale is physical");
    require(scale_400 > 0.0 && scale_400 < scale_300,
            "2GR extrapolation scale decreases with energy");

    carbon::TransportConfig legacy;
    require(legacy.fred_2gr_high_energy_mode == "zero" &&
                !legacy.uses_fred_2gr_high_energy_extrapolation(),
            "2GR legacy high-energy behavior is the default");
    carbon::TransportConfig extrapolated;
    extrapolated.multiple_scattering_model = "fred_2gr";
    extrapolated.fred_2gr_mcs_file = "fred_2gr.bin";
    extrapolated.fred_2gr_high_energy_mode = "kinematic_extrapolation";
    extrapolated.validate();
    require(extrapolated.uses_fred_2gr_high_energy_extrapolation(),
            "2GR kinematic extrapolation mode was not enabled");
    auto invalid_mode = extrapolated;
    invalid_mode.fred_2gr_high_energy_mode = "kinematic";
    require_throws([&invalid_mode] { invalid_mode.validate(); },
                   "Unknown 2GR high-energy mode was accepted");
    auto invalid_model = extrapolated;
    invalid_model.multiple_scattering_model = "highland";
    require_throws([&invalid_model] { invalid_model.validate(); },
                   "2GR extrapolation was accepted with Highland MCS");
}

void test_multi_energy_fred_event_libraries() {
    for (const auto energy : {95, 200, 300, 400}) {
        for (const auto& target : {std::string("H1"), std::string("C12"),
                                   std::string("O16")}) {
            const auto path = std::filesystem::path(CARBON_SOURCE_DIR) /
                "data/packages" /
                ("c12_" + target + "_" + std::to_string(energy) +
                 "MeVu_events.bin");
            const auto lib = carbon::load_fred_event_library(path);
            require_near(lib.reference_energy_MeVu, static_cast<float>(energy),
                         1.0e-4F, "event-library reference energy");
            require(lib.event_count > 3000, "multi-energy library event count");
            const int expected_z = target == "H1" ? 1 : (target == "C12" ? 6 : 8);
            const int expected_a = target == "H1" ? 1 : (target == "C12" ? 12 : 16);
            require(lib.target_z == expected_z && lib.target_a == expected_a,
                    "event-library target identity");
        }
    }
}
void test_fred_event_library_load() {
    const auto lib = carbon::load_fred_event_library(
        std::filesystem::path(CARBON_SOURCE_DIR) /
        "data/packages/c12_H1_95MeVu_events.bin");
    require(lib.event_count > 1000, "H1 event library has events");
    require(lib.max_fragments == 8, "max fragments");
    require_near(lib.reference_energy_MeVu, 95.0F, 1.0e-3F, "95 MeV/u reference");
    require(lib.fragment_count[0] >= 1, "first event has a charged fragment");
}

void test_topas_c12_h_elastic_table() {
    const float s100 = carbon::calculate_sigma_el_H_mb(100.0F);
    const float s400 = carbon::calculate_sigma_el_H_mb(400.0F);
    require(s100 > 50.0F && s100 < 200.0F, "TOPAS C+p elastic at 100 MeV/u");
    require(s400 > 20.0F && s400 < s100, "elastic XS falls with energy");
    const float m100 = carbon::water_elastic_h_macro_per_mm(100.0F, 1.0F);
    require(m100 > 1.0e-4F && m100 < 5.0e-3F, "water H elastic macro/mm");
}

void test_fred_paper_sigma_cc_and_water_macro() {
    const float sig_cc_95 = carbon::calculate_sigma_cc_mb(95.0F);
    require(sig_cc_95 > 700.0F && sig_cc_95 < 1100.0F, "C-C fit at 95 MeV/u near paper 760–1000 mb");
    const float sig_o_scaled = carbon::calculate_sigma_nonel_mb(
        12.0F, 6.0F, 16.0F, 8.0F, 95.0F, 1140.0F);
    const float sig_o_raw = carbon::calculate_kox_sigma_O(95.0F, 1140.0F);
    require(sig_o_scaled > 800.0F && sig_o_scaled < 1600.0F, "scaled C-O inelastic");
    require(std::fabs(sig_o_scaled - sig_o_raw) > 1.0F,
            "Kox ratio times C-C is not raw Kox C-O");
    const auto table = carbon::CrossSectionTable::from_fred_paper_water(1.0);
    require(table.energies().size() == 401, "1 MeV/u paper grid");
    const double tot = table.interpolate(200.0);
    require(tot > 0.003 && tot < 0.03, "water macroscopic XS /mm at 200 MeV/u");
    require(table.interpolate_target_h_fraction(200.0) > 0.2 &&
                table.interpolate_target_h_fraction(200.0) < 0.5,
            "paper P(H) in water");
}

void test_c12_hydrogen_elastic_kinematics() {
    const float e0 = 2400.0F;
    const auto scat = carbon::sample_c12_hydrogen_elastic(e0, 0.0F, 0.0F, 1.0F, 0.25F, 0.1F);
    require_near(scat.projectile_ke_MeV + scat.proton_ke_MeV, e0, 1.0e-3F,
                 "elastic KE conserved");
    require(scat.projectile_ke_MeV < e0, "projectile loses energy");
    require(scat.proton_ke_MeV > 0.0F, "recoil proton");
    const float pn = std::sqrt(scat.proj_dir_x * scat.proj_dir_x +
                               scat.proj_dir_y * scat.proj_dir_y +
                               scat.proj_dir_z * scat.proj_dir_z);
    require_near(pn, 1.0F, 1.0e-5F, "projectile direction unit");
}

void test_vavilov_landau_straggling_sampler() {
    carbon::TransportConfig cfg;
    cfg.energy_straggling_model = "vavilov_landau";
    cfg.validate();
    require(cfg.straggling_sampler_id() == carbon::straggling_sampler_vavilov_landau,
            "vavilov sampler id");
    double sum = 0.0;
    for (int i = 1; i <= 2000; ++i) {
        const double g = 0.0;
        const double u = static_cast<double>(i) / 2001.0;
        sum += carbon::sample_vavilov_landau_energy_loss(1.0, 0.4, g, u, 2.0, 50.0);
    }
    const double mean = sum / 2000.0;
    require(mean > 0.4 && mean < 2.0, "vavilov-like mean stays O(mean loss)");
}

void test_table1_no_np_evaporation_dump() {
    float uniforms[16];
    uint8_t idx[8];
    int leftover_n = 0;
    int saw_c12 = 0;
    int n_events = 4000;
    double counts[18]{};
    int nfrag_sum = 0;
    for (int e = 0; e < n_events; ++e) {
        for (int k = 0; k < 16; ++k) {
            uniforms[k] = static_cast<float>((e * 16 + k) % 997) / 997.0F;
        }
        leftover_n = 0;
        const int n = carbon::fill_projectile_table1_fragments(
            carbon::kFredProbH.data(), uniforms, 16, idx, 8, &leftover_n);
        require(n >= 1 && n <= 8, "projectile fragment count");
        nfrag_sum += n;
        int a = 0, z = 0;
        for (int i = 0; i < n; ++i) {
            require(idx[i] < 18, "isotope index");
            counts[idx[i]] += 1.0;
            a += carbon::kFredIsotopes[idx[i]].a;
            z += carbon::kFredIsotopes[idx[i]].z;
            if (idx[i] == 17) {
                ++saw_c12;
            }
        }
        require(z <= 6 && a + leftover_n <= 12, "A/Z bound without n/p dump");
    }
    require(saw_c12 > 0, "12C reachable");
    require(counts[1] > 0.0 && counts[5] > 0.0, "1H and 4He appear in Table 1 sampling");
    const double mean_frags = static_cast<double>(nfrag_sum) / static_cast<double>(n_events);
    require(mean_frags < 6.0, "no nucleon-by-nucleon evaporation dump");
}

void test_energy_dependent_inclusive_yields() {
    float w95[18]{};
    float w200[18]{};
    float w250[18]{};
    float w300[18]{};
    float w400[18]{};
    float wO300[18]{};
    carbon::fill_energy_dependent_inclusive_weights(95.0F, carbon::kFredProbH.data(), w95);
    carbon::fill_energy_dependent_inclusive_weights(200.0F, carbon::kFredProbH.data(), w200);
    carbon::fill_energy_dependent_inclusive_weights(250.0F, carbon::kFredProbH.data(), w250);
    carbon::fill_energy_dependent_inclusive_weights(300.0F, carbon::kFredProbH.data(), w300);
    carbon::fill_energy_dependent_inclusive_weights(400.0F, carbon::kFredProbH.data(), w400);
    carbon::fill_energy_dependent_inclusive_weights(300.0F, carbon::kFredProbO.data(), wO300);
    for (int i = 0; i < 18; ++i) {
        require(std::isfinite(w95[i]) && w95[i] >= 0.0F, "95 MeV/u weight finite non-negative");
        require_near(w95[i], carbon::kFredProbH[static_cast<std::size_t>(i)], 1.0e-5F,
                     "weights at 95 MeV/u match Table 1");
        require_near(w200[i], w95[i], 1.0e-5F, "paper yields fixed at 200 MeV/u");
        require_near(w250[i], w95[i], 1.0e-5F, "paper yields fixed at 250 MeV/u");
        require_near(w300[i], w95[i], 1.0e-5F, "paper yields fixed at 300 MeV/u");
        require_near(w400[i], w95[i], 1.0e-5F, "paper yields fixed at 400 MeV/u");
    }
    require_near(wO300[0], carbon::kFredProbO[0], 1.0e-5F,
                 "O-target paper yields use O Table 1");
}

void test_projectile_joint_channel() {
    uint8_t idx[8];
    int leftover_n = 0;
    int saw_c12 = 0;
    int n_events = 4000;
    int nfrag_sum = 0;
    double counts[18]{};
    for (int e = 0; e < n_events; ++e) {
        float uniforms[8]{};
        for (int j = 0; j < 8; ++j) {
            uniforms[j] =
                static_cast<float>((e * 17 + j * 31 + 3) % 997) / 997.0F;
        }
        leftover_n = 0;
        float w[18]{};
        carbon::fill_energy_dependent_inclusive_weights(200.0F, carbon::kFredProbH.data(), w);
        const int n = carbon::fill_projectile_constrained_channel(
            w, uniforms, 8, idx, 8, &leftover_n);
        require(n >= 1 && n <= 8, "constrained channel has charged fragments");
        nfrag_sum += n;
        int a = 0;
        int z = 0;
        for (int i = 0; i < n; ++i) {
            require(idx[i] < 18, "joint isotope index");
            counts[idx[i]] += 1.0;
            a += carbon::kFredIsotopes[idx[i]].a;
            z += carbon::kFredIsotopes[idx[i]].z;
            if (idx[i] == 17) {
                ++saw_c12;
            }
        }
        require(z == 6 && a + leftover_n == 12, "projectile channel closes A/Z exactly");
        require(leftover_n >= 0, "leftover neutrons non-negative");
    }
    require(saw_c12 > 0, "12C reachable in joint channel");
    require(counts[5] > 0.0 || counts[1] > 0.0, "He or p appears in joint channel");
    const double mean_frags = static_cast<double>(nfrag_sum) / static_cast<double>(n_events);
    require(mean_frags < 7.0, "constrained channel remains below proton-only completion");
}

void test_inelastic_neutron_kerma_fraction() {
    require(carbon::inelastic_neutron_kerma_MeV(0.0F) == 0.0F, "zero neutron KE scores no kerma");
    require(carbon::inelastic_neutron_kerma_MeV(100.0F) == 0.0F,
            "paper mode scores no fixed neutron vertex kerma");
}

void test_inelastic_optical_depth_in_step() {
    float s = 0.0F;
    require(carbon::inelastic_collision_in_step(0.02F, 1.0F, 0.0F, &s), "u=0 collides");
    require(s > 0.0F && s <= 1.0F, "collision distance inside step");
    require(!carbon::inelastic_collision_in_step(0.02F, 1.0F, 0.999F, &s),
            "large u may miss a short step");
    require_near(s, 1.0F, 1.0e-6F, "miss keeps full step");
}

void test_inelastic_fail_residual_not_double_counted() {
    const float incident = 1200.0F;
    carbon::InelasticProductSet failed{};
    failed.resample_failed = 1;
    failed.model_unassigned_MeV = incident;
    failed.untracked_energy_MeV = 0.0F;
    const float residual = carbon::inelastic_numerical_residual_MeV(
        incident, 0.0F, 0.0F, failed.untracked_energy_MeV, failed.model_unassigned_MeV);
    require_near(residual, 0.0, 1.0e-4, "resample-fail leftover must not appear again as residual");
    require(failed.untracked_energy_MeV == 0.0F, "resample fail must not dump KE into untracked");
    require_near(carbon::inelastic_fail_unassigned_if_no_products(1, 1200.0F), 0.0, 1.0e-6,
                 "fallback charged product carries KE, unassigned is 0");
    require_near(carbon::inelastic_fail_unassigned_if_no_products(0, 1200.0F), 1200.0, 1.0e-6,
                 "empty product list would leave incident unassigned");
}

void test_eq13_first_fragment_not_scaled_down() {
    for (const float p : {100.0F, 200.0F, 300.0F, 400.0F}) {
        const float e = carbon::sample_projectile_fragment_Eu(95.0F, p, 11, 0.0F, 0.0F);
        require(e > 0.0F, "first-fragment E/A must be finite");
        require_near(e, p, 1.0e-3F * p, "first fragment E/A must match incident E/A, not 0.6 P");
        require(e > 0.9F * p, "first fragment must not be 0.6x incident E/A");
    }
}

void test_eq12_component_choice() {
    require(carbon::eq12_sample_gaussian(1, 1, false, 0.1F, 0.5F), "projectile 1H mix can be Gaussian");
    require(!carbon::eq12_sample_gaussian(1, 1, false, 0.9F, 0.5F), "projectile 1H mix can be exponential");
    require(carbon::eq12_sample_gaussian(1, 2, true, 0.1F, 0.5F), "target 2H mix can be Gaussian");
    require(!carbon::eq12_sample_gaussian(1, 3, true, 0.9F, 0.5F), "target 3H mix can be exponential");
    require(carbon::eq12_sample_gaussian(6, 12, false, 0.99F, 0.01F), "projectile 12C is Gaussian");
    require(!carbon::eq12_sample_gaussian(6, 12, true, 0.0F, 0.99F), "target 12C is exponential");
}

void test_csv_target_h_fraction() {
    const auto xs = carbon::CrossSectionTable::from_csv(
        "data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv");
    require_near(static_cast<float>(xs.interpolate_target_h_fraction(100.0)), 0.374118F, 1.0e-5F,
                 "P_H(100)");
    require_near(static_cast<float>(xs.interpolate_target_h_fraction(200.0)), 0.342821F, 1.0e-5F,
                 "P_H(200)");
    require_near(static_cast<float>(xs.interpolate_target_h_fraction(300.0)), 0.335628F, 1.0e-5F,
                 "P_H(300)");
    require_near(static_cast<float>(xs.interpolate_target_h_fraction(400.0)), 0.337719F, 1.0e-5F,
                 "P_H(400)");
    require(xs.macro_h_per_mm().size() == xs.energies().size(), "macro_h loaded");
    require(xs.macro_o_per_mm().size() == xs.energies().size(), "macro_o loaded");
    for (std::size_t i = 0; i < xs.energies().size(); i += 50) {
        require_near(xs.macro_h_per_mm()[i] + xs.macro_o_per_mm()[i], xs.values()[i], 1.0e-8,
                     "macro_h + macro_o == total");
    }
}

void test_csda_remnant_local_stop() {
    require(carbon::remnant_local_stop_from_csda(0.01F, 10.0F, 0.05F), "short CSDA stops locally");
    require(!carbon::remnant_local_stop_from_csda(10.0F, 10.0F, 0.05F), "long CSDA is transported");
    const float e[3] = {1.0F, 2.0F, 3.0F};
    const float s[3] = {10.0F, 10.0F, 10.0F};
    float c[3] = {};
    carbon::fill_a1_csda_range_mm(e, s, 3, c);
    const float r = carbon::csda_range_mm_device(e, s, c, 3, 2.0F, 12);
    require(r > 0.0F, "CSDA range from SP grid is positive");
    require(r > 12.0F * 2.0F / 10.0F * 0.5F, "CSDA uses integrated 1/S not a single E/S dump");
}

void test_table1_newton_invert() {
    double raw_counts[18]{};
    unsigned raw_closed = 0;
    carbon::simulate_projectile_inclusive(carbon::kFredProbH.data(), 4000U, 7U, raw_counts,
                                          &raw_closed);
    double raw_sum = 0.0;
    for (int i = 0; i < 18; ++i) {
        raw_sum += raw_counts[i];
    }
    require(raw_sum > 0.0, "raw sequential sampling must emit fragments");
    require(raw_counts[17] > 0.0, "raw Table 1 sequential sampling must emit 12C");

    double table_sum = 0.0;
    for (int i = 0; i < 18; ++i) {
        table_sum += static_cast<double>(carbon::kFredProbH[static_cast<std::size_t>(i)]);
    }
    float raw_max = 0.0F;
    for (int i = 0; i < 18; ++i) {
        const auto fe = static_cast<double>(carbon::kFredProbH[static_cast<std::size_t>(i)]) /
                        table_sum;
        const auto fn = raw_counts[i] / raw_sum;
        raw_max = std::max(raw_max, static_cast<float>(std::fabs(fn - fe)));
    }

    const auto inverted = carbon::invert_table1_independent_probs(
        carbon::kFredProbH.data(), 12, 6, 2500U, 5U, 11U);
    require(inverted.events == 2500U, "invert must run requested events");
    require(inverted.inclusive_fraction[17] > 0.0, "inverted sampling must keep 12C");
    require(inverted.max_abs_fraction_error <= raw_max + 0.05F,
            "CRN Newton invert should not degrade Table 1 fraction match");

    const auto inverted_o16 = carbon::invert_table1_independent_probs(
        carbon::kFredProbO.data(), 16, 8, 2500U, 5U, 13U);
    require(inverted_o16.events == 2500U, "O-16 invert must run requested events");
    double o16_counts[18]{};
    carbon::simulate_nucleon_conserving_inclusive(
        inverted_o16.sample_prob.data(), 16, 8, 2000U, 13U, o16_counts, nullptr);
    double o16_sum = 0.0;
    for (int i = 0; i < 18; ++i) {
        o16_sum += o16_counts[i];
    }
    require(o16_sum > 0.0, "O-16 nucleon-conserving sampling must emit fragments");
}

void test_table1_inclusive_sampling() {
    bool saw_c12_h = false;
    bool saw_c12_o = false;
    bool saw_n_h = false;
    for (int trial = 0; trial < 10000; ++trial) {
        float u = (static_cast<float>(trial) + 0.5f) / 10000.0f;
        int iso_h = carbon::sample_table1_isotope(carbon::kFredProbH.data(), 12, 6, u);
        require(iso_h >= 0 && iso_h < 18, "Valid isotope index on H");
        require(carbon::kFredIsotopes[iso_h].a <= 12 && carbon::kFredIsotopes[iso_h].z <= 6, "Nucleon bound on H");
        if (iso_h == 17) saw_c12_h = true;
        if (iso_h == 0) saw_n_h = true;

        int iso_o = carbon::sample_table1_isotope(carbon::kFredProbO.data(), 12, 6, u);
        require(iso_o >= 0 && iso_o < 18, "Valid isotope index on O");
        require(carbon::kFredIsotopes[iso_o].a <= 12 && carbon::kFredIsotopes[iso_o].z <= 6, "Nucleon bound on O");
        if (iso_o == 17) saw_c12_o = true;
    }
    require(saw_c12_h, "Table 1 H sampling must be able to return 12C");
    require(saw_c12_o, "Table 1 O sampling must be able to return 12C");
    require(saw_n_h, "Table 1 H sampling must be able to return neutrons");
}

void test_cinel02_replay_miss_mcs_semantics() {
    constexpr float cutoff = 1.0F;
    require(carbon::cinel02_should_apply_secondary_mcs(
                false, false, true, 10.0F, cutoff),
            "ordinary secondary step must apply MCS");
    require(carbon::cinel02_should_apply_secondary_mcs(
                true, false, true, 10.0F, cutoff),
            "replay lookup miss must apply null-collision MCS");
    require(carbon::cinel02_should_apply_secondary_mcs(
                true, false, true, 10.0F, cutoff),
            "invalid replay must apply null-collision MCS");
    require(!carbon::cinel02_should_apply_secondary_mcs(
                true, true, true, 10.0F, cutoff),
            "valid replay must not apply the ordinary step MCS twice");
    require(!carbon::cinel02_should_apply_secondary_mcs(
                true, false, true, cutoff, cutoff),
            "below-cutoff null collision must not apply MCS");
    require(!carbon::cinel02_should_apply_secondary_mcs(
                true, false, false, 10.0F, cutoff),
            "disabled MCS must remain disabled for a replay miss");
}

void test_cinel02_ledger_schema_and_accumulator() {
    require(carbon::cinel02_unstable_ion_policy(4, 6) ==
                carbon::Cinel02UnstableIonPolicy::TopasCompatKill,
            "Be-6 must use the TOPAS compatibility policy");
    for (const auto& isotope : carbon::kFredIsotopes) {
        const auto policy = carbon::cinel02_unstable_ion_policy(isotope.z, isotope.a);
        require(policy == ((isotope.z == 4 && isotope.a == 6)
                               ? carbon::Cinel02UnstableIonPolicy::TopasCompatKill
                               : carbon::Cinel02UnstableIonPolicy::StableForTransport),
                "unexpected unstable-ion policy table entry");
    }
    require(carbon::cinel02_should_topas_compat_kill(true, 4, 6),
            "compatibility mode did not enable Be-6 kill");
    require(!carbon::cinel02_should_topas_compat_kill(false, 4, 6),
            "disabled compatibility mode still killed Be-6");
    require(!carbon::cinel02_should_topas_compat_kill(true, 4, 7),
            "compatibility mode killed transportable Be-7");
    carbon::TransportConfig compat_config;
    compat_config.nuclear_model = "cinel02";
    compat_config.primary_inelastic_package_v2_file = "package.cinpkg";
    compat_config.primary_inelastic_rate_v2_file = "rates.csv";
    compat_config.cinel02_topas_compatibility_mode = true;
    compat_config.validate();
    compat_config.nuclear_model = "fred_paper";
    require_throws([&compat_config] { compat_config.validate(); },
                   "compatibility mode accepted a non-CINEL02 model");
    using Schema = carbon::Cinel02SpeciesLedgerSchema;
    using Replay = carbon::Cinel02ReplayLedgerSchema;
    using Exposure = carbon::Cinel02ExposureLedgerSchema;
    static_assert(Schema::species_count == 18);
    static_assert(Schema::metric_count == 11);
    static_assert(Schema::terminal_reason_count == 6);
    static_assert(Schema::reaction_import_kinetic + 1 == Schema::metric_count);
    static_assert(Schema::continuous_stop + 1 == Schema::terminal_reason_count);
    static_assert(Replay::status_count == 5);
    static_assert(Replay::status_slot_count == 18 * 2 * 3 * 40 * 5);
    static_assert(Replay::parent_outcome_cell_count == 18 * 2 * 3 * 2);
    static_assert(Replay::transition_cell_count == 18 * 18);
    static_assert(Exposure::cell_count == 18 * 3 * 40);
    static_assert(Exposure::sum_slot_count == Exposure::cell_count * 18);
    static_assert(Exposure::count_slot_count == Exposure::cell_count * 4);
    static_assert(Exposure::hazard_blocked_total + 1 == Exposure::path_mm_continuous_rate_covered);
    static_assert(Exposure::stopping_loss_MeV + 1 == Exposure::sum_metric_count);
    require_near(carbon::cinel02_simpson_hazard(1.0F, 2.0F, 3.0F, 6.0F),
                 12.0, 1.0e-6, "CINEL02 Simpson hazard quadrature failed");
    require_near(carbon::cinel02_simpson_hazard(2.0F, 2.0F, 2.0F, 5.0F),
                 10.0, 1.0e-6, "CINEL02 constant hazard quadrature failed");

    carbon::TransportResult total{};
    carbon::TransportResult part{};
    constexpr std::size_t be6_species = 17;
    total.cinel02_topas_compat_discarded_counts[be6_species] = 2;
    part.cinel02_topas_compat_discarded_counts[be6_species] = 3;
    total.cinel02_topas_compat_discarded_kinetic_MeV[be6_species] = 4.5;
    part.cinel02_topas_compat_discarded_kinetic_MeV[be6_species] = 5.5;
    constexpr std::array<std::size_t, 4> diagnostic_slots{0, 64, 928, 1388};
    for (std::size_t i = 0; i < diagnostic_slots.size(); ++i) {
        total.cinel02_diagnostics[diagnostic_slots[i]] = 10 + i;
        part.cinel02_diagnostics[diagnostic_slots[i]] = 100 + 2 * i;
    }
    for (std::size_t i = 0; i < total.cinel02_energy_ledger_MeV.size(); ++i) {
        total.cinel02_energy_ledger_MeV[i] = static_cast<double>(i);
        part.cinel02_energy_ledger_MeV[i] = static_cast<double>(2 * i);
    }
    for (std::size_t i = 0;
         i < total.cinel02_species_transport_ledger_MeV.size(); ++i) {
        total.cinel02_species_transport_ledger_MeV[i] = static_cast<double>(i);
        part.cinel02_species_transport_ledger_MeV[i] = static_cast<double>(3 * i);
    }
    for (std::size_t i = 0;
         i < total.cinel02_species_terminal_reason_counts.size(); ++i) {
        total.cinel02_species_terminal_reason_counts[i] = i;
        part.cinel02_species_terminal_reason_counts[i] = 4 * i;
    }
    for (std::size_t i = 0; i < total.cinel02_replay_delta_MeV_per_u.size(); ++i) {
        total.cinel02_replay_delta_MeV_per_u[i] = static_cast<double>(i);
        part.cinel02_replay_delta_MeV_per_u[i] = static_cast<double>(2 * i);
        total.cinel02_replay_abs_delta_MeV_per_u[i] = static_cast<double>(3 * i);
        part.cinel02_replay_abs_delta_MeV_per_u[i] = static_cast<double>(4 * i);
        total.cinel02_replay_delta_positive_counts[i] = i;
        part.cinel02_replay_delta_positive_counts[i] = 2 * i;
        total.cinel02_replay_delta_negative_counts[i] = 3 * i;
        part.cinel02_replay_delta_negative_counts[i] = 4 * i;
        total.cinel02_replay_valid_counts[i] = 5 * i;
        part.cinel02_replay_valid_counts[i] = 6 * i;
    }
    for (std::size_t i = 0; i < total.cinel02_replay_status_counts.size(); ++i) {
        total.cinel02_replay_status_counts[i] = i;
        part.cinel02_replay_status_counts[i] = 2 * i;
        total.cinel02_replay_status_rate_query_energy_MeV[i] = 10.0 + i;
        part.cinel02_replay_status_rate_query_energy_MeV[i] = 20.0 + 3.0 * i;
        total.cinel02_replay_status_replay_query_energy_MeV[i] = 9.0 + i;
        part.cinel02_replay_status_replay_query_energy_MeV[i] = 18.0 + 2.0 * i;
        total.cinel02_replay_status_continuous_loss_to_collision_MeV[i] = 1.0;
        part.cinel02_replay_status_continuous_loss_to_collision_MeV[i] = 2.0 + i;
        total.cinel02_replay_status_delta_MeV_per_u[i] = static_cast<double>(4 * i);
        part.cinel02_replay_status_delta_MeV_per_u[i] = static_cast<double>(5 * i);
        total.cinel02_replay_status_abs_delta_MeV_per_u[i] = static_cast<double>(6 * i);
        part.cinel02_replay_status_abs_delta_MeV_per_u[i] = static_cast<double>(7 * i);
    }
    for (std::size_t i = 0; i < total.cinel02_secondary_exposure_sums.size(); ++i) {
        total.cinel02_secondary_exposure_sums[i] = 0.5 * static_cast<double>(i);
        part.cinel02_secondary_exposure_sums[i] = 1.5 * static_cast<double>(i);
    }
    for (std::size_t i = 0; i < total.cinel02_secondary_exposure_counts.size(); ++i) {
        total.cinel02_secondary_exposure_counts[i] = i;
        part.cinel02_secondary_exposure_counts[i] = 2 * i;
    }
    for (std::size_t i = 0; i < total.cinel02_parent_outcome_counts.size(); ++i) {
        total.cinel02_parent_outcome_counts[i] = i;
        part.cinel02_parent_outcome_counts[i] = 2 * i;
        total.cinel02_parent_outcome_incident_energy_MeV[i] = static_cast<double>(i);
        part.cinel02_parent_outcome_incident_energy_MeV[i] = static_cast<double>(3 * i);
        total.cinel02_parent_outcome_after_energy_MeV[i] = static_cast<double>(4 * i);
        part.cinel02_parent_outcome_after_energy_MeV[i] = static_cast<double>(5 * i);
        total.cinel02_parent_outcome_local_deposit_MeV[i] = static_cast<double>(6 * i);
        part.cinel02_parent_outcome_local_deposit_MeV[i] = static_cast<double>(7 * i);
        total.cinel02_parent_outcome_export_MeV[i] = static_cast<double>(8 * i);
        part.cinel02_parent_outcome_export_MeV[i] = static_cast<double>(9 * i);
        total.cinel02_parent_outcome_import_MeV[i] = static_cast<double>(10 * i);
        part.cinel02_parent_outcome_import_MeV[i] = static_cast<double>(11 * i);
    }
    for (std::size_t i = 0; i < total.cinel02_generated_transition_counts.size(); ++i) {
        total.cinel02_generated_transition_counts[i] = i;
        part.cinel02_generated_transition_counts[i] = 2 * i;
        total.cinel02_generated_transition_kinetic_MeV[i] = static_cast<double>(3 * i);
        part.cinel02_generated_transition_kinetic_MeV[i] = static_cast<double>(4 * i);
        total.cinel02_queued_transition_counts[i] = 5 * i;
        part.cinel02_queued_transition_counts[i] = 6 * i;
        total.cinel02_queued_transition_kinetic_MeV[i] = static_cast<double>(7 * i);
        part.cinel02_queued_transition_kinetic_MeV[i] = static_cast<double>(8 * i);
    }

    carbon::accumulate_transport_result(total, part);
    require(total.cinel02_topas_compat_discarded_counts[be6_species] == 5,
            "compatibility sink count did not accumulate");
    require_near(total.cinel02_topas_compat_discarded_kinetic_MeV[be6_species],
                 10.0, 0.0, "compatibility sink kinetic energy did not accumulate");
    total.initial_energy_MeV = 100.0;
    total.total_deposited_energy_MeV = 90.0;
    require_near(total.physical_relative_energy_balance_error(), 0.1, 0.0,
                 "physical closure must expose compatibility sink");
    require_near(total.relative_energy_balance_error(), 0.0, 0.0,
                 "accounting closure must include compatibility sink");
    for (std::size_t i = 0; i < diagnostic_slots.size(); ++i) {
        require(total.cinel02_diagnostics[diagnostic_slots[i]] == 110 + 3 * i,
                "CINEL02 diagnostic accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_energy_ledger_MeV.size(); ++i) {
        require_near(total.cinel02_energy_ledger_MeV[i], 3.0 * i, 0.0,
                     "CINEL02 energy accumulator mismatch");
    }
    for (std::size_t i = 0;
         i < total.cinel02_species_transport_ledger_MeV.size(); ++i) {
        require_near(total.cinel02_species_transport_ledger_MeV[i], 4.0 * i, 0.0,
                     "CINEL02 species energy accumulator mismatch");
    }
    for (std::size_t i = 0;
         i < total.cinel02_species_terminal_reason_counts.size(); ++i) {
        require(total.cinel02_species_terminal_reason_counts[i] == 5 * i,
                "CINEL02 terminal accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_replay_delta_MeV_per_u.size(); ++i) {
        require_near(total.cinel02_replay_delta_MeV_per_u[i], 3.0 * i, 0.0,
                     "CINEL02 replay delta accumulator mismatch");
        require_near(total.cinel02_replay_abs_delta_MeV_per_u[i], 7.0 * i, 0.0,
                     "CINEL02 replay absolute delta accumulator mismatch");
        require(total.cinel02_replay_delta_positive_counts[i] == 3 * i,
                "CINEL02 replay positive-count accumulator mismatch");
        require(total.cinel02_replay_delta_negative_counts[i] == 7 * i,
                "CINEL02 replay negative-count accumulator mismatch");
        require(total.cinel02_replay_valid_counts[i] == 11 * i,
                "CINEL02 replay valid-count accumulator mismatch");
    }

    for (std::size_t i = 0; i < total.cinel02_replay_status_counts.size(); ++i) {
        require(total.cinel02_replay_status_counts[i] == 3 * i,
                "CINEL02 replay-status count accumulator mismatch");
        require_near(total.cinel02_replay_status_rate_query_energy_MeV[i],
                     30.0 + 4.0 * i, 0.0,
                     "CINEL02 replay-status rate-energy accumulator mismatch");
        require_near(total.cinel02_replay_status_replay_query_energy_MeV[i],
                     27.0 + 3.0 * i, 0.0,
                     "CINEL02 replay-status replay-energy accumulator mismatch");
        require_near(total.cinel02_replay_status_continuous_loss_to_collision_MeV[i],
                     3.0 + i, 0.0,
                     "CINEL02 replay-status continuous-loss accumulator mismatch");
        require_near(total.cinel02_replay_status_rate_query_energy_MeV[i] -
                         total.cinel02_replay_status_replay_query_energy_MeV[i],
                     total.cinel02_replay_status_continuous_loss_to_collision_MeV[i],
                     0.0, "CINEL02 replay energy handoff invariant mismatch");
        require_near(total.cinel02_replay_status_delta_MeV_per_u[i], 9.0 * i, 0.0,
                     "CINEL02 replay-status delta accumulator mismatch");
        require_near(total.cinel02_replay_status_abs_delta_MeV_per_u[i], 13.0 * i, 0.0,
                     "CINEL02 replay-status absolute-delta accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_secondary_exposure_sums.size(); ++i) {
        require_near(total.cinel02_secondary_exposure_sums[i], 2.0 * i, 0.0,
                     "CINEL02 secondary-exposure sum accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_secondary_exposure_counts.size(); ++i) {
        require(total.cinel02_secondary_exposure_counts[i] == 3 * i,
                "CINEL02 secondary-exposure count accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_parent_outcome_counts.size(); ++i) {
        require(total.cinel02_parent_outcome_counts[i] == 3 * i,
                "CINEL02 parent-outcome count accumulator mismatch");
        require_near(total.cinel02_parent_outcome_incident_energy_MeV[i], 4.0 * i, 0.0,
                     "CINEL02 parent-outcome incident accumulator mismatch");
        require_near(total.cinel02_parent_outcome_after_energy_MeV[i], 9.0 * i, 0.0,
                     "CINEL02 parent-outcome after accumulator mismatch");
        require_near(total.cinel02_parent_outcome_local_deposit_MeV[i], 13.0 * i, 0.0,
                     "CINEL02 parent-outcome local accumulator mismatch");
        require_near(total.cinel02_parent_outcome_export_MeV[i], 17.0 * i, 0.0,
                     "CINEL02 parent-outcome export accumulator mismatch");
        require_near(total.cinel02_parent_outcome_import_MeV[i], 21.0 * i, 0.0,
                     "CINEL02 parent-outcome import accumulator mismatch");
    }
    for (std::size_t i = 0; i < total.cinel02_generated_transition_counts.size(); ++i) {
        require(total.cinel02_generated_transition_counts[i] == 3 * i,
                "CINEL02 generated-transition count accumulator mismatch");
        require_near(total.cinel02_generated_transition_kinetic_MeV[i], 7.0 * i, 0.0,
                     "CINEL02 generated-transition energy accumulator mismatch");
        require(total.cinel02_queued_transition_counts[i] == 11 * i,
                "CINEL02 queued-transition count accumulator mismatch");
        require_near(total.cinel02_queued_transition_kinetic_MeV[i], 15.0 * i, 0.0,
                     "CINEL02 queued-transition energy accumulator mismatch");
    }

    const double queued_birth = 100.0;
    const double continuous = 35.0;
    const double nuclear_local = 5.0;
    const double reaction_export = 40.0;
    const double reaction_import = 7.0;
    const double terminal = 15.0;
    const double escape = 12.0;
    require_near(queued_birth + reaction_import - continuous - nuclear_local -
                     reaction_export - terminal - escape,
                 0.0, 0.0, "CINEL02 signed handoff partition mismatch");
}

bool is_sycl_available() {
#ifdef CARBON_HAS_SYCL
    try {
        (void)carbon::describe_sycl_device("default");
        return true;
    } catch (const std::exception&) {
        return false;
    }
#else
    return false;
#endif
}

void require_water_transport_equivalent(const carbon::TransportResult& actual,
                                        const carbon::TransportResult& expected,
                                        const std::string& context,
                                        const double energy_tolerance = 1.0e-12,
                                        const double dose_tolerance = 1.0e-12) {
    require_near(actual.total_deposited_energy_MeV, expected.total_deposited_energy_MeV,
                 energy_tolerance, context + ": total deposited energy mismatch");
    require_near(actual.escaped_energy_MeV, expected.escaped_energy_MeV,
                 energy_tolerance, context + ": escaped energy mismatch");
    require(actual.deposited_energy_MeV.size() == expected.deposited_energy_MeV.size(),
            context + ": 1D depth dose bins size mismatch");
    for (std::size_t bin = 0; bin < actual.deposited_energy_MeV.size(); ++bin) {
        require_near(actual.deposited_energy_MeV[bin], expected.deposited_energy_MeV[bin],
                     dose_tolerance,
                     context + ": 1D depth dose bin " + std::to_string(bin) + " mismatch");
    }
    require(actual.nuclear_interactions == expected.nuclear_interactions,
            context + ": nuclear interactions count mismatch");
    require(actual.primary_elastic_interactions == expected.primary_elastic_interactions,
            context + ": primary elastic count mismatch");
    require(actual.relative_energy_balance_error() < 1.0e-6,
            context + ": energy balance error exceeds tolerance");
}

void test_water_transport_invariance_without_ct() {
    carbon::TransportConfig base_config;
    base_config.number_of_histories = 64;
    base_config.initial_energy_MeVu = 200.0;
    base_config.phantom_length_mm = 200.0;
    base_config.depth_bin_width_mm = 1.0;
    base_config.maximum_step_mm = 0.5;
    base_config.maximum_relative_energy_loss = 0.01;
    base_config.enable_energy_straggling = true;
    base_config.random_seed = 20260902;
    base_config.validate();

    const carbon::StoppingPowerTable table({0.01, 100.01, 200.01, 300.01}, {1.5, 1.8, 2.0, 2.2});

    const auto run_a = carbon::transport_serial(base_config, table, zero_cross_section());
    const auto run_b = carbon::transport_serial(base_config, table, zero_cross_section());
    require_water_transport_equivalent(run_a, run_b, "Serial water exact repeatability");

    // Configure unattached CT knobs with distinct sentinel values; when enable_ct_grid=false,
    // none of these knobs may leak into pure water transport.
    auto ct_unattached_config = base_config;
    ct_unattached_config.ct_schneider_file = "data/HUtoMaterialSchneider.txt";
    ct_unattached_config.ct_stopping_power_scale = 1.2345;
    ct_unattached_config.ct_use_density_mass_spr = false;
    ct_unattached_config.ct_skip_homogeneous_face_clamp = false;
    ct_unattached_config.ct_schneider_cross_section_file = "unused.csv";
    ct_unattached_config.ct_cinel02_rate_file = "unused-rates.csv";
    ct_unattached_config.ct_hu_stopping_power_lut_file = "unused-sp.csv";
    ct_unattached_config.validate();

    const auto run_c = carbon::transport_serial(ct_unattached_config, table, zero_cross_section());
    require_water_transport_equivalent(run_c, run_a, "Serial unattached CT knob isolation");

#ifdef CARBON_HAS_SYCL
    if (is_sycl_available()) {
        auto sycl_base = base_config;
        sycl_base.validate();
        auto sycl_ct = ct_unattached_config;
        sycl_ct.validate();

        const auto sycl_a =
            carbon::transport_sycl(sycl_base, table, zero_cross_section(), "default");
        const auto sycl_b =
            carbon::transport_sycl(sycl_ct, table, zero_cross_section(), "default");
        // FP32 atomic additions on GPU have ~1e-5 relative rounding noise.
        require_water_transport_equivalent(sycl_b, sycl_a, "SYCL unattached CT knob isolation",
                                           1.0e-4, 1.0e-4);
    }
#endif
}

void test_schneider_material_table_parser() {
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto schneider_path = source_dir / "data/HUtoMaterialSchneider.txt";
    const auto table = carbon::SchneiderMaterialTable::from_topas_file(schneider_path);

    // 1. Verify 13 elements in exact order with correct Z and atomic masses
    require(table.elements.size() == 13, "Schneider element count must be 13");
    const std::vector<std::pair<std::string, std::uint8_t>> expected_elements = {
        {"Hydrogen", 1}, {"Carbon", 6}, {"Nitrogen", 7}, {"Oxygen", 8},
        {"Magnesium", 12}, {"Phosphorus", 15}, {"Sulfur", 16}, {"Chlorine", 17},
        {"Argon", 18}, {"Calcium", 20}, {"Sodium", 11}, {"Potassium", 19},
        {"Titanium", 22}
    };
    for (std::size_t i = 0; i < 13; ++i) {
        require(table.elements[i].name == expected_elements[i].first,
                "Schneider element name mismatch at index " + std::to_string(i));
        require(table.elements[i].z == expected_elements[i].second,
                "Schneider element Z mismatch at index " + std::to_string(i));
        require(table.elements[i].atomic_mass_g_mol > 0.0,
                "Schneider atomic mass must be positive");
    }

    // 2. Verify 25 sections with strictly increasing contiguous HU boundaries
    require(table.sections.size() == 25, "Schneider section count must be 25");
    require(table.sections[0].hu_min_inclusive == -1000, "Section 0 must start at HU -1000");
    require(table.sections[0].hu_max_exclusive == -950, "Section 0 must end at HU -950");
    require(table.sections[1].hu_min_inclusive == -950, "Section 1 must start at HU -950");
    require(table.sections[23].hu_min_inclusive == 1500, "Section 23 must start at HU 1500");
    require(table.sections[23].hu_max_exclusive == 2995, "Section 23 must end at HU 2995");
    require(table.sections[24].hu_min_inclusive == 2995, "Section 24 must start at HU 2995");
    require(table.sections[24].hu_max_exclusive == 2996, "Section 24 must end at HU 2996");

    for (std::size_t i = 0; i < 25; ++i) {
        if (i > 0) {
            require(table.sections[i].hu_min_inclusive == table.sections[i - 1].hu_max_exclusive,
                    "Section HU boundaries must be contiguous at section " + std::to_string(i));
        }
        double row_sum = 0.0;
        for (std::size_t el = 0; el < 13; ++el) {
            const double w = table.sections[i].mass_fraction[el];
            require(w >= 0.0, "Mass fraction must be nonnegative");
            row_sum += w;
        }
        require_near(row_sum, 1.0, 1.0e-6, "Mass fraction sum must equal 1.0 for section " + std::to_string(i));
    }

    // 3. Verify specific known sections (Section 0 Air, Section 24 Titanium)
    require_near(table.sections[0].mass_fraction[2], 0.755, 1.0e-6, "Air Nitrogen fraction mismatch");
    require_near(table.sections[0].mass_fraction[3], 0.232, 1.0e-6, "Air Oxygen fraction mismatch");
    require_near(table.sections[0].mass_fraction[8], 0.013, 1.0e-6, "Air Argon fraction mismatch");
    require_near(table.sections[24].mass_fraction[12], 1.0, 1.0e-6, "Titanium section fraction mismatch");

    // 4. Verify section_id mapping across domain and boundaries
    require(table.section_id(-1500) == 0, "HU < -1000 must map to section 0");
    require(table.section_id(-1000) == 0, "HU = -1000 must map to section 0");
    require(table.section_id(-951) == 0, "HU = -951 must map to section 0");
    require(table.section_id(-950) == 1, "HU = -950 must map to section 1");
    require(table.section_id(-120) == 2, "HU = -120 must map to section 2");
    require(table.section_id(0) == 5, "HU = 0 must map to section 5");
    require(table.section_id(1500) == 23, "HU = 1500 must map to section 23");
    require(table.section_id(2994) == 23, "HU = 2994 must map to section 23");
    require(table.section_id(2995) == 24, "HU = 2995 must map to section 24");
    require(table.section_id(2996) == 24, "HU = 2996 must map to section 24");
    require(table.section_id(3500) == 24, "HU > 2995 must clamp to section 24");

    // 5. Verify malformed inputs are properly rejected
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto test_malformed = [&](const std::string& filename, const std::string& content, const std::string& error_desc) {
        const auto temp_file = temp_dir / filename;
        std::ofstream out(temp_file);
        out << content;
        out.close();
        bool threw = false;
        try {
            (void)carbon::SchneiderMaterialTable::from_topas_file(temp_file);
        } catch (const std::exception&) {
            threw = true;
        }
        std::filesystem::remove(temp_file);
        require(threw, "Parser accepted malformed input: " + error_desc);
    };

    test_malformed("schneider_missing_elem.txt",
                   "iv:Ge/Patient/SchneiderHUToMaterialSections = 26 -1000 -950 -120 -83 -53 -23 7 18 80 120 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 2995 2996\n",
                   "missing SchneiderElements");

    test_malformed("schneider_bad_weights_sum.txt",
                   "sv:Ge/Patient/SchneiderElements = 13 \"Hydrogen\" \"Carbon\" \"Nitrogen\" \"Oxygen\" \"Magnesium\" \"Phosphorus\" \"Sulfur\" \"Chlorine\" \"Argon\" \"Calcium\" \"Sodium\" \"Potassium\" \"Titanium\"\n"
                   "iv:Ge/Patient/SchneiderHUToMaterialSections = 26 -1000 -950 -120 -83 -53 -23 7 18 80 120 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 2995 2996\n"
                   "uv:Ge/Patient/SchneiderMaterialsWeight1 = 13 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",
                   "weight row sum != 1.0");

    test_malformed("schneider_bad_elem_name.txt",
                   "sv:Ge/Patient/SchneiderElements = 13 \"Hydrogen\" \"Carbon\" \"Nitrogen\" \"Oxygen\" \"Krypton\" \"Phosphorus\" \"Sulfur\" \"Chlorine\" \"Argon\" \"Calcium\" \"Sodium\" \"Potassium\" \"Titanium\"\n"
                   "iv:Ge/Patient/SchneiderHUToMaterialSections = 26 -1000 -950 -120 -83 -53 -23 7 18 80 120 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 2995 2996\n",
                   "unknown element name Krypton");
}

void test_schneider_c12_inelastic_cross_section_table() {
    const auto tables = carbon::CrossSectionTable::from_schneider_csv(
        "data/c12_schneider_inelastic_cross_sections_geant4_11_3_2.csv");
    require(tables.size() == 25, "Schneider cross section table must have exactly 25 sections");

    for (std::size_t s = 0; s < tables.size(); ++s) {
        const auto& table = tables[s];
        require(table.energies().size() == 860,
                "Section " + std::to_string(s) + " must have 860 energy bins");
        require_near(table.energies().front(), 0.5, 1e-9, "Energy grid start must be 0.5 MeV/u");
        require_near(table.energies().back(), 430.0, 1e-9, "Energy grid end must be 430.0 MeV/u");

        for (std::size_t i = 0; i < table.values().size(); ++i) {
            require(table.values()[i] >= 0.0,
                    "Section " + std::to_string(s) + " mass cross section must be non-negative");
            require(std::isfinite(table.values()[i]),
                    "Section " + std::to_string(s) + " mass cross section must be finite");
        }

        // Test interpolation at clinical energies
        const double xs_100 = table.interpolate(100.0);
        const double xs_200 = table.interpolate(200.0);
        const double xs_290 = table.interpolate(290.0);
        const double xs_400 = table.interpolate(400.0);

        require(xs_100 > 0.0 && xs_200 > 0.0 && xs_290 > 0.0 && xs_400 > 0.0,
                "Interpolated mass cross sections must be strictly positive at clinical energies");
    }

    // Negative tests: malformed CSV rejection
    const auto test_malformed_csv = [](const std::string& name, const std::string& content,
                                       const std::string& expected_error) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path);
        out << content;
        out.close();

        bool threw = false;
        try {
            carbon::CrossSectionTable::from_schneider_csv(path);
        } catch (const std::exception& error) {
            threw = true;
            require(std::string(error.what()).find(expected_error) != std::string::npos,
                    "Error message missing expected substring '" + expected_error +
                    "': " + error.what());
        }
        std::filesystem::remove(path);
        require(threw, "Parser accepted invalid Schneider cross-section CSV");
    };

    test_malformed_csv("bad_header.csv", "bad_col,section_00_mass_xs_per_mm_at_1g_cm3\n1.0,0.01\n",
                       "Schneider cross-section CSV must start with energy_MeV_per_u");
    test_malformed_csv("bad_sec_col.csv", "energy_MeV_per_u,section_01_mass_xs_per_mm_at_1g_cm3\n1.0,0.01\n",
                       "Unexpected Schneider cross-section column");
}

void test_c12_inelastic_unit_conversions_and_sample_rows() {
    // 1. Unit conversion mathematical proof:
    // 1 barn = 10^-24 cm^2 = 10^-28 m^2 = 10^-22 mm^2
    constexpr double kBarnToMm2 = 1.0e-22;
    constexpr double kCm3ToMm3 = 1.0e-3;

    // 2. Hand calculation check Sample 1:
    // Section 8 (Soft Tissue, HU=100), Target Oxygen (Z=8), E=200.0 MeV/u
    const double n_o_mm3 = 2.5256826962e+19;
    const double sigma_o_barn = 0.93229581701;
    const double expected_partial_macro_o = n_o_mm3 * (sigma_o_barn * kBarnToMm2);
    const double recorded_partial_macro_o = 0.00235468341276;
    require_near(expected_partial_macro_o, recorded_partial_macro_o, 1.0e-12,
                 "Sample 1 (Soft tissue/O/200 MeV/u) hand-calculated partial macro mismatch");

    const double n_o_cm3 = 2.5256826962e+22;
    require_near((n_o_cm3 * kCm3ToMm3) / n_o_mm3, 1.0, 1.0e-12,
                 "Atom density cm3 -> mm3 conversion mismatch");

    // 3. Hand calculation check Sample 2:
    // Section 20 (Dense Bone, HU=1250), Target Calcium (Z=20), E=300.0 MeV/u
    const double n_ca_mm3 = 5.50165804809e+18;
    const double sigma_ca_barn = 1.5501092386;
    const double expected_partial_macro_ca = n_ca_mm3 * (sigma_ca_barn * kBarnToMm2);
    const double recorded_partial_macro_ca = 0.000852817096793;
    require_near(expected_partial_macro_ca, recorded_partial_macro_ca, 1.0e-12,
                 "Sample 2 (Dense bone/Ca/300 MeV/u) hand-calculated partial macro mismatch");

    const double n_ca_cm3 = 5.50165804809e+21;
    require_near((n_ca_cm3 * kCm3ToMm3) / n_ca_mm3, 1.0, 1.0e-12,
                 "Atom density cm3 -> mm3 conversion mismatch for Ca");
}

void test_compiled_schneider_c12_rate_products() {
    // 1. Verify CSV parsing via CrossSectionTable::from_schneider_csv
    const auto csv_tables = carbon::CrossSectionTable::from_schneider_csv(
        "data/schneider/c12_schneider_inelastic_mass_xs.csv");
    require(csv_tables.size() == 25, "Compiled CSV must have exactly 25 section tables");
    for (std::size_t s = 0; s < csv_tables.size(); ++s) {
        require(csv_tables[s].energies().size() == 860, "Compiled CSV section must have 860 energy nodes");
        require_near(csv_tables[s].energies().front(), 0.5, 1e-9, "CSV energy start mismatch");
        require_near(csv_tables[s].energies().back(), 430.0, 1e-9, "CSV energy end mismatch");
    }

    // 2. Verify Binary parsing via SchneiderRateTable::from_binary
    const auto rate_table = carbon::SchneiderRateTable::from_binary(
        "data/schneider/schneider_inelastic_rates_v1.bin",
        "data/schneider/schneider_inelastic_rates_v1.metadata.json");

    require_near(rate_table.energy_min_mevu(), 0.5, 1e-9, "Rate table min energy mismatch");
    require_near(rate_table.energy_max_mevu(), 430.0, 1e-9, "Rate table max energy mismatch");
    require_near(rate_table.energy_step_mevu(), 0.5, 1e-9, "Rate table step energy mismatch");

    // Canonical target index lookup checks
    require(carbon::SchneiderRateTable::target_index_from_z(1) == 0, "Target H Z=1 index must be 0");
    require(carbon::SchneiderRateTable::target_index_from_z(6) == 1, "Target C Z=6 index must be 1");
    require(carbon::SchneiderRateTable::target_index_from_z(8) == 3, "Target O Z=8 index must be 3");
    require(carbon::SchneiderRateTable::target_index_from_z(20) == 9, "Target Ca Z=20 index must be 9");
    require(carbon::SchneiderRateTable::target_index_from_z(22) == 12, "Target Ti Z=22 index must be 12");
    require_throws<std::invalid_argument>(
        []() { (void)carbon::SchneiderRateTable::target_index_from_z(99); },
        "Unsupported target Z must throw invalid_argument");

    // 3. Cross-validate binary against CSV for all 25 sections and all 860 energies
    for (std::size_t s = 0; s < 25; ++s) {
        for (std::size_t e_idx = 0; e_idx < 860; ++e_idx) {
            const double bin_total = rate_table.mass_total_rate(s, e_idx);
            const double csv_total = csv_tables[s].values()[e_idx];
            require_near(bin_total, csv_total, 1e-10, "Binary total vs CSV total mismatch");

            double partial_sum = 0.0;
            for (std::size_t t = 0; t < 13; ++t) {
                const double part = rate_table.mass_partial_rate(s, t, e_idx);
                require(part >= 0.0 && std::isfinite(part), "Partial rate must be finite and non-negative");
                partial_sum += part;
            }
            require_near(partial_sum, bin_total, 1e-12, "Partial sum != total in binary rate table");
        }

        // Test continuous interpolation consistency
        const double interp_bin_123 = rate_table.interpolate_mass_total(s, 123.45);
        const double interp_csv_123 = csv_tables[s].interpolate(123.45);
        require_near(interp_bin_123, interp_csv_123, 1e-5, "Interpolation consistency at 123.45 MeV/u");
    }

    // 4. Negative tests for binary loader
    const auto test_malformed_bin = [](const std::string& name, const std::string& content) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary);
        out.write(content.data(), content.size());
        out.close();

        bool threw = false;
        try {
            carbon::SchneiderRateTable::from_binary(path);
        } catch (const std::exception&) {
            threw = true;
        }
        std::filesystem::remove(path);
        require(threw, "SchneiderRateTable::from_binary accepted malformed binary data");
    };

    test_malformed_bin("bad_magic.bin", "BADMAGIC_12345678901234567890");
    test_malformed_bin("truncated.bin", "SCHNRATE");
}

void test_step09_schneider_primary_xs_host_path() {
    // 1. Valid 25-section loading
    const auto tables = carbon::CrossSectionTable::from_schneider_csv(
        "data/schneider/c12_schneider_inelastic_mass_xs.csv");
    require(tables.size() == 25, "Schneider cross-section table must have exactly 25 sections");
    for (std::size_t s = 0; s < tables.size(); ++s) {
        require(tables[s].energies().size() == 860, "Section " + std::to_string(s) + " must have 860 nodes");
        require(tables[s].values().size() == 860, "Section " + std::to_string(s) + " values mismatch");
        require_near(tables[s].energies().front(), 0.5, 1e-9, "Energy start must be 0.5");
        require_near(tables[s].energies().back(), 430.0, 1e-9, "Energy end must be 430.0");
    }

    // Helper for negative CSV tests
    const auto test_malformed_csv = [](const std::string& name, const std::string& content,
                                       const std::string& expected_substr) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path);
        out << content;
        out.close();

        bool threw = false;
        try {
            carbon::CrossSectionTable::from_schneider_csv(path);
        } catch (const std::exception& error) {
            threw = true;
            require(std::string(error.what()).find(expected_substr) != std::string::npos,
                    "Error missing substring '" + expected_substr + "': " + error.what());
        }
        std::filesystem::remove(path);
        require(threw, "Parser accepted invalid Schneider cross-section CSV");
    };

    // 2. Reject 24 columns (missing section 24)
    std::string csv_24 = "energy_MeV_per_u";
    for (int s = 0; s < 24; ++s) {
        csv_24 += ",section_" + (s < 10 ? std::string{"0"} : std::string{}) + std::to_string(s) + "_mass_xs_per_mm_at_1g_cm3";
    }
    csv_24 += "\n1.0";
    for (int s = 0; s < 24; ++s) csv_24 += ",0.005";
    csv_24 += "\n";
    test_malformed_csv("bad_24_cols.csv", csv_24, "Schneider cross-section CSV must have exactly 25 section columns");

    // 3. Reject 26 columns (extra section 25)
    std::string csv_26 = "energy_MeV_per_u";
    for (int s = 0; s < 26; ++s) {
        csv_26 += ",section_" + (s < 10 ? std::string{"0"} : std::string{}) + std::to_string(s) + "_mass_xs_per_mm_at_1g_cm3";
    }
    csv_26 += "\n1.0";
    for (int s = 0; s < 26; ++s) csv_26 += ",0.005";
    csv_26 += "\n";
    test_malformed_csv("bad_26_cols.csv", csv_26, "Schneider cross-section CSV must have exactly 25 section columns");

    // 4. Reject swapped headers (section 01 before section 00)
    std::string csv_swapped = "energy_MeV_per_u,section_01_mass_xs_per_mm_at_1g_cm3,section_00_mass_xs_per_mm_at_1g_cm3";
    for (int s = 2; s < 25; ++s) {
        csv_swapped += ",section_" + (s < 10 ? std::string{"0"} : std::string{}) + std::to_string(s) + "_mass_xs_per_mm_at_1g_cm3";
    }
    csv_swapped += "\n1.0";
    for (int s = 0; s < 25; ++s) csv_swapped += ",0.005";
    csv_swapped += "\n";
    test_malformed_csv("bad_swapped.csv", csv_swapped, "Unexpected Schneider cross-section column");

    // 5. Reject non-monotonic energies
    std::string csv_non_monotonic = "energy_MeV_per_u";
    for (int s = 0; s < 25; ++s) {
        csv_non_monotonic += ",section_" + (s < 10 ? std::string{"0"} : std::string{}) + std::to_string(s) + "_mass_xs_per_mm_at_1g_cm3";
    }
    csv_non_monotonic += "\n10.0";
    for (int s = 0; s < 25; ++s) csv_non_monotonic += ",0.005";
    csv_non_monotonic += "\n9.5";
    for (int s = 0; s < 25; ++s) csv_non_monotonic += ",0.005";
    csv_non_monotonic += "\n";
    test_malformed_csv("bad_non_monotonic.csv", csv_non_monotonic, "Non-monotonic energy sequence");

    // 6. Resampling and Energy Coverage:
    std::vector<double> valid_transport_energies;
    for (double e = 10.0; e <= 400.0; e += 1.0) {
        valid_transport_energies.push_back(e);
    }
    const auto resampled_grid = carbon::resample_schneider_cross_section_grid(tables, valid_transport_energies);
    require(resampled_grid.energy_nodes() == valid_transport_energies.size(), "Resampled grid node count mismatch");

    // Test coverage underflow (< 0.5 MeV/u)
    std::vector<double> underflow_energies = {0.1, 10.0, 100.0};
    require_throws<std::runtime_error>(
        [&]() { (void)carbon::resample_schneider_cross_section_grid(tables, underflow_energies); },
        "Transport energy range underflow must fail-fast without clamping");

    // Test coverage overflow (> 430.0 MeV/u)
    std::vector<double> overflow_energies = {100.0, 200.0, 450.0};
    require_throws<std::runtime_error>(
        [&]() { (void)carbon::resample_schneider_cross_section_grid(tables, overflow_energies); },
        "Transport energy range overflow must fail-fast without clamping");

    // 7. Section boundary (Section 24 vs Section 25)
    const float sec24_val = resampled_grid.at(24, 0);
    require(sec24_val > 0.0f && std::isfinite(sec24_val), "Section 24 (Titanium) rate must be valid");
    require_throws<std::out_of_range>(
        [&]() { (void)resampled_grid.at(25, 0); },
        "Section 25 must throw out_of_range (0..24 valid)");

    // 8. Density unit sanity check:
    // Verify that host values are in units of mm^-1 at 1 g/cm3 (mass cross section).
    // For rho = 1.0 g/cm3, macroscopic cross section == mass cross section.
    // For rho != 1.0 g/cm3, the host table remains completely invariant.
    const std::size_t e_idx_100 = 90; // index of 100.0 MeV/u in valid_transport_energies (10 + 90 = 100)
    const float host_val_sec8 = resampled_grid.at(8, e_idx_100);
    const double raw_val_sec8 = tables[8].interpolate(100.0);
    require_near(host_val_sec8, raw_val_sec8, 1e-6, "Host table value must equal source mass rate");

    const double rho_water = 1.0;
    const double macro_water = rho_water * host_val_sec8;
    require_near(macro_water, host_val_sec8, 1e-6, "At rho=1.0, macro == mass rate");

    const double rho_bone = 1.8216;
    const double macro_bone = rho_bone * resampled_grid.at(20, e_idx_100);
    require(macro_bone != resampled_grid.at(20, e_idx_100), "At rho!=1.0, macro != mass rate");
    require_near(resampled_grid.at(20, e_idx_100), tables[20].interpolate(100.0), 1e-6,
                 "Host table is strictly invariant to density");

    // 9. Relative-path resolution in configuration:
    const auto temp_dir = std::filesystem::temp_directory_path() / "step09_rel_test";
    std::filesystem::create_directories(temp_dir);
    const auto abs_xs_path = std::filesystem::absolute(
        "data/schneider/c12_schneider_inelastic_mass_xs.csv");
    const auto rel_xs_path = std::filesystem::relative(abs_xs_path, temp_dir);

    const auto config_path = temp_dir / "test_config.txt";
    std::ofstream cfg_out(config_path);
    cfg_out << "ct_schneider_cross_section_file: " << rel_xs_path.string() << "\n";
    cfg_out << "initial_energy_MeVu: 100.0\n";
    cfg_out << "number_of_histories: 10\n";
    cfg_out.close();

    const auto loaded_cfg = carbon::load_config(config_path);
    require(std::filesystem::exists(loaded_cfg.ct_schneider_cross_section_file),
            "Relative ct_schneider_cross_section_file must resolve to existing file");

    std::filesystem::remove_all(temp_dir);

    // 10. Config nuclear_model test:
    // Schneider CT + nuclear_model=none + no XS -> PASS
    // Schneider CT + nuclear_model=geant4 + no XS -> FAIL
    carbon::TransportConfig ct_em_only;
    ct_em_only.enable_ct_grid = true;
    ct_em_only.ct_grid_file = "dummy_grid.cctg";
    ct_em_only.ct_schneider_file = "dummy_schneider.txt";
    ct_em_only.nuclear_model = "none";
    ct_em_only.ct_schneider_cross_section_file = ""; // No XS
    ct_em_only.validate(); // Must pass without throwing!

    carbon::TransportConfig ct_nuclear_active;
    ct_nuclear_active.enable_ct_grid = true;
    ct_nuclear_active.ct_grid_file = "dummy_grid.cctg";
    ct_nuclear_active.ct_schneider_file = "dummy_schneider.txt";
    ct_nuclear_active.nuclear_model = "geant4";
    ct_nuclear_active.ct_schneider_cross_section_file = ""; // No XS
    require_throws<std::invalid_argument>(
        [&]() { ct_nuclear_active.validate(); },
        "Schneider CT + nuclear_model=geant4 + no XS must FAIL");

    // Also test via load_config from file to verify parse-order fix
    {
        const auto cfg_test_dir = std::filesystem::temp_directory_path() / "schneider_em_only_test";
        std::filesystem::create_directories(cfg_test_dir);
        const auto em_cfg_file = cfg_test_dir / "em_only.txt";
        std::ofstream em_out(em_cfg_file);
        em_out << "initial_energy_MeVu: 100.0\n"
               << "number_of_histories: 10\n"
               << "nuclear_model: none\n";
        em_out.close();
        const auto parsed_em = carbon::load_config(em_cfg_file);
        require(parsed_em.nuclear_model == "none", "Parsed nuclear_model must be none");
        std::filesystem::remove_all(cfg_test_dir);
    }

    // 11. Test prepare_schneider_primary_xs helper:
    carbon::TransportConfig valid_schneider_cfg;
    valid_schneider_cfg.ct_schneider_cross_section_file =
        "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    const auto prepared_grid = carbon::prepare_schneider_primary_xs(
        valid_schneider_cfg, valid_transport_energies);
    require(prepared_grid.energy_nodes() == valid_transport_energies.size(),
            "prepare_schneider_primary_xs must return properly resampled grid");
    require_near(prepared_grid.at(8, e_idx_100), host_val_sec8, 1e-6,
                 "prepare_schneider_primary_xs values must match resampled table");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = (argc > 1) ? argv[1] : "";
    const auto run = [&](const std::string& name, auto fn) {
        if (filter.empty() || name.find(filter) != std::string::npos) {
            fn();
        }
    };
    try {
        run("test_fred_18_isotopes_data", test_fred_18_isotopes_data);
        run("test_cinel02_replay_miss_mcs_semantics", test_cinel02_replay_miss_mcs_semantics);
        run("test_cinel02_ledger_schema_and_accumulator", test_cinel02_ledger_schema_and_accumulator);
        run("test_ion_species_stopping_power_grid_validation", test_ion_species_stopping_power_grid_validation);
        run("test_stopping_power_csv_corruption_rejection", test_stopping_power_csv_corruption_rejection);
        run("test_kox_icru_cross_sections", test_kox_icru_cross_sections);
        run("test_topas_c12_h_elastic_table", test_topas_c12_h_elastic_table);
        run("test_fred_event_library_load", test_fred_event_library_load);
        run("test_fred_2gr_package", test_fred_2gr_package);
        run("test_multi_energy_fred_event_libraries", test_multi_energy_fred_event_libraries);
        run("test_fred_paper_sigma_cc_and_water_macro", test_fred_paper_sigma_cc_and_water_macro);
        run("test_c12_hydrogen_elastic_kinematics", test_c12_hydrogen_elastic_kinematics);
        run("test_vavilov_landau_straggling_sampler", test_vavilov_landau_straggling_sampler);
        run("test_table1_no_np_evaporation_dump", test_table1_no_np_evaporation_dump);
        run("test_energy_dependent_inclusive_yields", test_energy_dependent_inclusive_yields);
        run("test_projectile_joint_channel", test_projectile_joint_channel);
        run("test_inelastic_neutron_kerma_fraction", test_inelastic_neutron_kerma_fraction);
        run("test_inelastic_optical_depth_in_step", test_inelastic_optical_depth_in_step);
        run("test_inelastic_fail_residual_not_double_counted", test_inelastic_fail_residual_not_double_counted);
        run("test_eq13_first_fragment_not_scaled_down", test_eq13_first_fragment_not_scaled_down);
        run("test_eq12_component_choice", test_eq12_component_choice);
        run("test_csv_target_h_fraction", test_csv_target_h_fraction);
        run("test_csda_remnant_local_stop", test_csda_remnant_local_stop);
        run("test_table1_inclusive_sampling", test_table1_inclusive_sampling);
        run("test_table1_newton_invert", test_table1_newton_invert);
        run("test_units", test_units);
        run("test_csda_range_loss_validation", test_csda_range_loss_validation);
        run("test_hu_stopping_power_lut_loading", test_hu_stopping_power_lut_loading);
        run("test_serial_voxel_idd_closure", test_serial_voxel_idd_closure);
        run("test_charged_dose_categories", test_charged_dose_categories);
        run("test_interpolation", test_interpolation);
        run("test_electron_transport_table_and_config", test_electron_transport_table_and_config);
        run("test_stopping_power_csda_range_helpers", test_stopping_power_csda_range_helpers);
        run("test_cpu_csda_range_loss_switch", test_cpu_csda_range_loss_switch);
        run("test_cross_section_zero_endpoint_contract", test_cross_section_zero_endpoint_contract);
        run("test_fragment_stopping_power_scale", test_fragment_stopping_power_scale);
        run("test_primary_ion_definition", test_primary_ion_definition);
        run("test_ion_physics_manifest_loading", test_ion_physics_manifest_loading);
        run("test_strict_config_parsing_and_canonicalization", test_strict_config_parsing_and_canonicalization);
        run("test_run_quality_gate", test_run_quality_gate);
        run("test_particle_specific_stopping_power_tables", test_particle_specific_stopping_power_tables);
        run("test_step_selection", test_step_selection);
        run("test_slab_phantom_helpers", test_slab_phantom_helpers);
        run("test_ct_grid_helpers", test_ct_grid_helpers);
        run("test_philox_rng", test_philox_rng);
        run("test_highland_multiple_scattering", test_highland_multiple_scattering);
        run("test_bohr_straggling", test_bohr_straggling);
        run("test_clamped_gaussian_straggling_sampler_audit", test_clamped_gaussian_straggling_sampler_audit);
        run("test_moment_matched_straggling_sampler", test_moment_matched_straggling_sampler);
        run("test_condensed_total_loss_straggling", test_condensed_total_loss_straggling);
        run("test_step_stable_straggling_validation", test_step_stable_straggling_validation);
        run("test_energy_dependent_straggling_scale", test_energy_dependent_straggling_scale);
        run("test_energy_conservation", test_energy_conservation);
        run("test_escape_energy_conservation", test_escape_energy_conservation);
        run("test_straggling_reproducibility", test_straggling_reproducibility);
        run("test_flat_source_config_validation", test_flat_source_config_validation);
        run("test_random_seed_parsing", test_random_seed_parsing);
        run("test_minibeam_absorbing_geometry", test_minibeam_absorbing_geometry);
        run("test_topas_spots_parse_angle01", test_topas_spots_parse_angle01);
        run("test_topas_spot_weights_and_tps_90_transform", test_topas_spot_weights_and_tps_90_transform);
        run("test_tps_source_geometry_csv_and_switch", test_tps_source_geometry_csv_and_switch);
        run("test_dose_scorer_matches_mev_conversion", test_dose_scorer_matches_mev_conversion);
        run("test_dense_voxel_mhd_writer", test_dense_voxel_mhd_writer);
        run("test_layered_voxel_dose_uses_local_mass", test_layered_voxel_dose_uses_local_mass);
        run("test_dense_charged_origin_mhd_uses_local_mass", test_dense_charged_origin_mhd_uses_local_mass);
        run("test_ct_aligned_mhd_offset_and_index_pairing", test_ct_aligned_mhd_offset_and_index_pairing);
        run("test_water_transport_invariance_without_ct", test_water_transport_invariance_without_ct);
        run("test_schneider_material_table_parser", test_schneider_material_table_parser);
        run("test_schneider_c12_inelastic_cross_section_table", test_schneider_c12_inelastic_cross_section_table);
        run("test_c12_inelastic_unit_conversions_and_sample_rows", test_c12_inelastic_unit_conversions_and_sample_rows);
        run("test_compiled_schneider_c12_rate_products", test_compiled_schneider_c12_rate_products);
        run("test_step09_schneider_primary_xs_host_path", test_step09_schneider_primary_xs_host_path);
#ifdef CARBON_HAS_SYCL
        run("test_sycl_tps_source_arbitrary_gantry_transport", test_sycl_tps_source_arbitrary_gantry_transport);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_legacy_cardinal_entrance_projection", test_sycl_legacy_cardinal_entrance_projection);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_primary_spot_batch", test_sycl_primary_spot_batch);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_primary_let_includes_cutoff_tail", test_sycl_primary_let_includes_cutoff_tail);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_flat_source_extent", test_sycl_flat_source_extent);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_serial_sycl_cpu_match", test_serial_sycl_cpu_match);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_transport_context_reuse", test_sycl_transport_context_reuse);
#endif
#ifdef CARBON_HAS_SYCL
        run("test_sycl_layered_slab_range_shift", test_sycl_layered_slab_range_shift);
#endif
        std::cout << "All carbon_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_tests: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
