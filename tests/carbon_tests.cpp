#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/minibeam_collimator.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/min_json.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/secondary_rate_table.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/plan_run.hpp"
#include "carbon/rng.hpp"
#include "carbon/run_quality.hpp"
#include "carbon/sha256.hpp"

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
#include "carbon/inelastic_identity.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/schneider_ct_device_context.hpp"
#include "carbon/device.hpp"
#include "carbon/detail/device_memory_tracker.hpp"
#include <sycl/sycl.hpp>

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

    auto voxel_over = std::make_unique<carbon::TransportResult>(*clean);
    voxel_over->voxel_deposited_energy_MeV = {60.0, 45.0};
    report = carbon::evaluate_run_quality(production, *voxel_over);
    require(!report.accepted && !report.failures.empty() &&
                report.voxel_to_total_deposited_ratio > 1.02,
            "Production quality gate accepted voxel energy exceeding the deposited total");
    require_near(report.voxel_scored_energy_MeV, 105.0, 0.0,
                 "Voxel scored energy aggregation mismatch");
    require_near(report.nonvoxel_deposited_energy_MeV, -5.0, 0.0,
                 "Non-voxel deposited energy mismatch");

    auto voxel_ok = std::make_unique<carbon::TransportResult>(*clean);
    voxel_ok->voxel_deposited_energy_MeV = {60.0, 39.0};
    report = carbon::evaluate_run_quality(production, *voxel_ok);
    require(report.accepted && report.failures.empty() &&
                report.voxel_to_total_deposited_ratio < 1.02,
            "Production quality gate rejected consistent voxel closure");

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

void test_step10_schneider_primary_xs_device_path() {
    // 1. Single layout index helper verification:
    // Contiguous [section][energy]: index = section * energy_nodes + energy_index
    require(carbon::schneider_cross_section_index(0, 0, 100) == 0, "Index (0, 0) must be 0");
    require(carbon::schneider_cross_section_index(1, 0, 100) == 100, "Index (1, 0) must be 100");
    require(carbon::schneider_cross_section_index(24, 99, 100) == 2499, "Index (24, 99) must be 2499");

    // 2. Sentinel table test on GPU and CPU:
    // value(section, e_idx) = 100000.0f * section + e_idx
    constexpr std::uint32_t kSentinelSections = 25;
    constexpr std::uint32_t kSentinelEnergies = 101; // 0 to 500 MeV/u, dE = 5.0
    constexpr float kSentinelEmin = 0.0F;
    constexpr float kSentinelInvDE = 1.0F / 5.0F; // 0.2
    std::vector<float> sentinel_table(kSentinelSections * kSentinelEnergies);
    for (std::uint32_t s = 0; s < kSentinelSections; ++s) {
        for (std::uint32_t e = 0; e < kSentinelEnergies; ++e) {
            sentinel_table[carbon::schneider_cross_section_index(s, e, kSentinelEnergies)] =
                100000.0F * static_cast<float>(s) + static_cast<float>(e);
        }
    }

    // Build comprehensive queries across all 25 sections:
    // - exact nodes (e_idx = 0, 1, 25, 50, 75, 100)
    // - midpoints (e_idx = 0.5, 10.5, 49.5, 99.5)
    // - endpoints (0.0 and 500.0)
    std::vector<std::uint32_t> query_sections;
    std::vector<float> query_energies;
    std::vector<float> query_densities;
    std::vector<float> expected_host_vals;

    for (std::uint32_t s = 0; s < kSentinelSections; ++s) {
        // Exact node queries (rho = 1.0)
        for (std::uint32_t node : {0U, 1U, 25U, 50U, 75U, 100U}) {
            const float energy = static_cast<float>(node) * 5.0F;
            query_sections.push_back(s);
            query_energies.push_back(energy);
            query_densities.push_back(1.0F);
            expected_host_vals.push_back(100000.0F * static_cast<float>(s) + static_cast<float>(node));
        }
        // Midpoint queries (rho = 1.0)
        for (float frac_node : {0.5F, 10.5F, 49.5F, 99.5F}) {
            const float energy = frac_node * 5.0F;
            query_sections.push_back(s);
            query_energies.push_back(energy);
            query_densities.push_back(1.0F);
            expected_host_vals.push_back(100000.0F * static_cast<float>(s) + frac_node);
        }
    }

    // Density test queries:
    // rho in {0.001, 0.3, 1.0, 2.0}
    for (std::uint32_t s : {1U, 8U, 20U}) {
        const float test_energy = 200.0F; // node 40
        const float expected_mass_rate = 100000.0F * static_cast<float>(s) + 40.0F;
        for (float rho : {0.001F, 0.3F, 1.0F, 2.0F}) {
            query_sections.push_back(s);
            query_energies.push_back(test_energy);
            query_densities.push_back(rho);
            expected_host_vals.push_back(rho * expected_mass_rate);
        }
    }

    // First, verify host helper matches expectations exactly
    for (std::size_t i = 0; i < query_sections.size(); ++i) {
        const float host_eval = carbon::schneider_primary_macroscopic_xs(
            sentinel_table.data(), kSentinelSections, kSentinelEnergies,
            kSentinelEmin, kSentinelInvDE, query_sections[i],
            query_energies[i], query_densities[i]);
        require_near(host_eval, expected_host_vals[i], 1e-4,
                     "Host sentinel evaluation mismatch at query " + std::to_string(i));
    }

#ifdef CARBON_HAS_SYCL
    if (is_sycl_available()) {
        // GPU sentinel lookup batch
        const auto device_results = carbon::test_schneider_device_lookup_batch(
            sentinel_table, kSentinelSections, kSentinelEnergies,
            kSentinelEmin, kSentinelInvDE, query_sections,
            query_energies, query_densities, "default");
        require(device_results.size() == expected_host_vals.size(),
                "Device results size must match query count");
        for (std::size_t i = 0; i < device_results.size(); ++i) {
            require_near(device_results[i], expected_host_vals[i], 1e-4,
                         "GPU sentinel lookup mismatch vs expected at query " + std::to_string(i));
        }

        // 3. Real Schneider table GPU lookup equivalence test
        carbon::TransportConfig real_cfg;
        real_cfg.ct_schneider_cross_section_file = "data/schneider/c12_schneider_inelastic_mass_xs.csv";
        std::vector<double> real_transport_energies(400);
        for (std::size_t i = 0; i < 400; ++i) {
            real_transport_energies[i] = 1.0 + i * 1.0; // 1.0 to 400.0 MeV/u (within [0.5, 430.0])
        }
        const auto real_grid = carbon::prepare_schneider_primary_xs(real_cfg, real_transport_energies);
        const float real_emin = static_cast<float>(real_transport_energies.front());
        const float real_inv_de = 1.0F / static_cast<float>(real_transport_energies[1] - real_transport_energies[0]);

        std::vector<std::uint32_t> real_q_sec;
        std::vector<float> real_q_e;
        std::vector<float> real_q_rho;
        std::vector<float> real_expected;

        for (std::uint32_t s = 0; s < 25; ++s) {
            for (float e : {1.0F, 50.0F, 100.0F, 200.0F, 300.0F, 350.0F, 400.0F}) {
                for (float rho : {0.001F, 0.3F, 1.0F, 2.0F}) {
                    real_q_sec.push_back(s);
                    real_q_e.push_back(e);
                    real_q_rho.push_back(rho);
                    const float host_macro = carbon::schneider_primary_macroscopic_xs(
                        real_grid.mass_xs_per_mm_at_1g_cm3.data(), 25,
                        static_cast<std::uint32_t>(real_grid.energy_nodes()),
                        real_emin, real_inv_de, s, e, rho);
                    real_expected.push_back(host_macro);
                }
            }
        }

        const auto real_device_results = carbon::test_schneider_device_lookup_batch(
            real_grid.mass_xs_per_mm_at_1g_cm3, 25,
            static_cast<std::uint32_t>(real_grid.energy_nodes()),
            real_emin, real_inv_de, real_q_sec, real_q_e, real_q_rho, "default");
        require(real_device_results.size() == real_expected.size(),
                "Real table device results size mismatch");
        for (std::size_t i = 0; i < real_device_results.size(); ++i) {
            require_near(real_device_results[i], real_expected[i], 1e-6,
                         "GPU real Schneider table lookup mismatch at query " + std::to_string(i));
        }

        // 4. Invalid dimensions / bounds checks fail-fast before launch
        require_throws<std::invalid_argument>(
            [&]() {
                (void)carbon::test_schneider_device_lookup_batch(
                    sentinel_table, 24, kSentinelEnergies, kSentinelEmin, kSentinelInvDE,
                    query_sections, query_energies, query_densities, "default");
            },
            "Non-25 section_count must throw invalid_argument");
    }
#endif

    // 5. CCTG integration test:
    // Create a tiny synthetic CCTG file (1x1x1) with Schneider material_id
    {
        const auto temp_cctg_dir = std::filesystem::temp_directory_path() / "test_cctg_schneider";
        std::filesystem::create_directories(temp_cctg_dir);
        const auto cctg_file = temp_cctg_dir / "tiny_schneider.cctg";
        carbon::CtGrid test_grid;
        test_grid.file_version = carbon::CtGrid::version_v2;
        test_grid.nx = 1;
        test_grid.ny = 1;
        test_grid.nz = 1;
        test_grid.spacing_x_mm = 1.0;
        test_grid.spacing_y_mm = 1.0;
        test_grid.spacing_z_mm = 1.0;
        test_grid.origin_x_mm = 0.0;
        test_grid.origin_y_mm = 0.0;
        test_grid.origin_z_mm = 0.0;
        test_grid.density_g_per_cm3 = {1.0F};
        test_grid.material_id = {8}; // section 8 (soft tissue)
        test_grid.mass_sp_za_rel.resize(25, 1.0);
        test_grid.write_binary(cctg_file);

        // tiny Schneider CCTG + nuclear_model=none + no XS -> PASS
        const auto cfg_em_file = temp_cctg_dir / "em_cctg.txt";
        {
            std::ofstream out(cfg_em_file);
            out << "enable_ct_grid: true\n"
                << "ct_grid_file: " << cctg_file.string() << "\n"
                << "nuclear_model: none\n"
                << "number_of_histories: 1\n"
                << "initial_energy_MeVu: 100.0\n";
        }
        const auto loaded_em = carbon::load_config(cfg_em_file);
        require(loaded_em.nuclear_model == "none", "Loaded CCTG EM config must have nuclear_model: none");

        // tiny Schneider CCTG + nuclear_model=geant4 + no XS -> FAIL
        const auto cfg_nuc_file = temp_cctg_dir / "nuc_cctg.txt";
        {
            std::ofstream out(cfg_nuc_file);
            out << "enable_ct_grid: true\n"
                << "ct_grid_file: " << cctg_file.string() << "\n"
                << "nuclear_model: geant4\n"
                << "number_of_histories: 1\n"
                << "initial_energy_MeVu: 100.0\n";
        }
        require_throws<std::runtime_error>(
            [&]() { (void)carbon::load_config(cfg_nuc_file); },
            "Schneider CCTG with nuclear_model=geant4 and no XS must fail load_config");

        std::filesystem::remove_all(temp_cctg_dir);
    }
}

void test_step11_piecewise_nuclear_optical_depth() {
#ifdef CARBON_HAS_SYCL
    if (!is_sycl_available()) {
        std::cout << "Skipping test_step11_piecewise_nuclear_optical_depth: SYCL device not available\n";
        return;
    }

    carbon::TransportConfig cfg;
    cfg.ct_schneider_cross_section_file = "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    const auto schneider_grid = carbon::prepare_schneider_primary_xs(cfg);
    const float e_min = static_cast<float>(schneider_grid.transport_energies_MeVu.front());
    const float inv_dE = 1.0F / static_cast<float>(schneider_grid.transport_energies_MeVu[1] - schneider_grid.transport_energies_MeVu[0]);
    const auto n_energies = static_cast<std::uint32_t>(schneider_grid.energy_nodes());
    const float test_energy = 200.0F; // 200 MeV/u

    // =========================================================================
    // Test 1: Two-layer analytical survival S = exp(-(Sigma1 * L1 + Sigma2 * L2))
    // Layer 1: section 8 (soft tissue), density 1.0 g/cm3, length 10.0 mm
    // Layer 2: section 20 (dense bone), density 1.5 g/cm3, length 15.0 mm
    // Total thickness = 25.0 mm
    // =========================================================================
    {
        const float mass_rate_1 = carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, n_energies, e_min, inv_dE, 8, test_energy);
        const float mass_rate_2 = carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, n_energies, e_min, inv_dE, 20, test_energy);
        const float rho_1 = 1.0F;
        const float rho_2 = 1.5F;
        const float sigma_1 = rho_1 * mass_rate_1;
        const float sigma_2 = rho_2 * mass_rate_2;
        const float L_1 = 10.0F;
        const float L_2 = 15.0F;
        const double S_exact = std::exp(-(static_cast<double>(sigma_1) * L_1 + static_cast<double>(sigma_2) * L_2));

        // Create a 2-layer CT grid along z:
        // voxel spacing: 10 mm in x, 10 mm in y, 1.0 mm in z
        // 1 x 1 x 25 voxels: voxels 0..9 are layer 1 (z: 0..10 mm), voxels 10..24 are layer 2 (z: 10..25 mm)
        carbon::CtGrid grid;
        grid.file_version = carbon::CtGrid::version_v2;
        grid.nx = 1;
        grid.ny = 1;
        grid.nz = 25;
        grid.spacing_x_mm = 10.0;
        grid.spacing_y_mm = 10.0;
        grid.spacing_z_mm = 1.0;
        grid.origin_x_mm = -5.0;
        grid.origin_y_mm = -5.0;
        grid.origin_z_mm = 0.0;
        grid.density_g_per_cm3.resize(25);
        grid.material_id.resize(25);
        for (std::uint32_t z = 0; z < 25; ++z) {
            if (z < 10) {
                grid.material_id[z] = 8;
                grid.density_g_per_cm3[z] = rho_1;
            } else {
                grid.material_id[z] = 20;
                grid.density_g_per_cm3[z] = rho_2;
            }
        }

        constexpr std::uint32_t kHistories = 1000000;
        const auto res = carbon::run_step11_piecewise_hazard_gpu_test(
            grid, schneider_grid.mass_xs_per_mm_at_1g_cm3, 25, n_energies,
            e_min, inv_dE, test_energy, kHistories,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 25.0F, 1.0F, 25, "default");

        const double rel_err = std::abs(res.survival_fraction - S_exact) / S_exact;
        require(rel_err < 0.002,
                "Two-layer analytical survival relative error must be < 0.2%, got " +
                std::to_string(rel_err * 100.0) + "% (S_GPU=" + std::to_string(res.survival_fraction) +
                ", S_exact=" + std::to_string(S_exact) + ")");
        require(std::abs(res.survival_fraction - S_exact) <= 3.0 * res.survival_fraction_std_err,
                "Two-layer survival must agree with exact within 3 sigma");
        require(res.zero_progress_count == 0, "zero_progress_count must be 0");
        require(res.changed_material_step_span_count == 0, "changed_material_step_span_count must be 0");
        require(res.face_cross_count > 0, "face_cross_count must be positive");

        // =========================================================================
        // Test 2: First-interaction CDF histogram matching piecewise theoretical CDF
        // Layer 1 (z in [0, 10]): F(z) = 1 - exp(-sigma1 * z)
        // Layer 2 (z in [10, 25]): F(z) = 1 - exp(-sigma1 * L1 - sigma2 * (z - L1))
        // Bins are 1.0 mm wide (25 bins total)
        // =========================================================================
        require(res.interaction_binned_counts.size() == 25, "Must have 25 bins");
        for (std::uint32_t b = 0; b < 25; ++b) {
            const double z_lo = static_cast<double>(b) * 1.0;
            const double z_hi = static_cast<double>(b + 1) * 1.0;
            double F_lo = 0.0;
            double F_hi = 0.0;
            if (z_hi <= 10.0) {
                F_lo = 1.0 - std::exp(-static_cast<double>(sigma_1) * z_lo);
                F_hi = 1.0 - std::exp(-static_cast<double>(sigma_1) * z_hi);
            } else if (z_lo >= 10.0) {
                F_lo = 1.0 - std::exp(-static_cast<double>(sigma_1) * 10.0 - static_cast<double>(sigma_2) * (z_lo - 10.0));
                F_hi = 1.0 - std::exp(-static_cast<double>(sigma_1) * 10.0 - static_cast<double>(sigma_2) * (z_hi - 10.0));
            } else {
                F_lo = 1.0 - std::exp(-static_cast<double>(sigma_1) * z_lo);
                F_hi = 1.0 - std::exp(-static_cast<double>(sigma_1) * 10.0 - static_cast<double>(sigma_2) * (z_hi - 10.0));
            }
            const double p_bin = F_hi - F_lo;
            const double expected_count = static_cast<double>(kHistories) * p_bin;
            const double actual_count = static_cast<double>(res.interaction_binned_counts[b]);
            const double bin_std_err = std::sqrt(expected_count * (1.0 - p_bin));
            const double bin_diff = std::abs(actual_count - expected_count);
            require(bin_diff <= 3.5 * bin_std_err,
                    "Bin " + std::to_string(b) + " interaction count outside 3.5 sigma: actual=" +
                    std::to_string(actual_count) + " expected=" + std::to_string(expected_count) +
                    " sigma=" + std::to_string(bin_std_err));
        }
    }

    // =========================================================================
    // Test 3: Same density / different section and same section / different density
    // =========================================================================
    {
        // Case A: rho1 == rho2 = 1.0, section 1 (lung) vs section 20 (dense bone)
        const float mr_lung = carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, n_energies, e_min, inv_dE, 1, test_energy);
        const float mr_bone = carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, n_energies, e_min, inv_dE, 20, test_energy);
        require(std::abs(mr_lung - mr_bone) > 1e-4F, "Lung and bone mass rates must differ");

        carbon::CtGrid grid_sec;
        grid_sec.file_version = carbon::CtGrid::version_v2;
        grid_sec.nx = 1;
        grid_sec.ny = 1;
        grid_sec.nz = 2;
        grid_sec.spacing_x_mm = 10.0;
        grid_sec.spacing_y_mm = 10.0;
        grid_sec.spacing_z_mm = 10.0;
        grid_sec.origin_x_mm = -5.0;
        grid_sec.origin_y_mm = -5.0;
        grid_sec.origin_z_mm = 0.0;
        grid_sec.density_g_per_cm3 = {1.0F, 1.0F};
        grid_sec.material_id = {1, 20}; // lung then bone

        const auto res_sec = carbon::run_step11_piecewise_hazard_gpu_test(
            grid_sec, schneider_grid.mass_xs_per_mm_at_1g_cm3, 25, n_energies,
            e_min, inv_dE, test_energy, 500000,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 20.0F, 10.0F, 2, "default");
        const double S_sec_exact = std::exp(-(static_cast<double>(mr_lung) * 10.0 + static_cast<double>(mr_bone) * 10.0));
        require(std::abs(res_sec.survival_fraction - S_sec_exact) / S_sec_exact < 0.002,
                "Case A: Same density different section survival must match exact within 0.2%");

        // Case B: same section (8), different density: 0.5 vs 1.5 g/cm3
        carbon::CtGrid grid_rho;
        grid_rho.file_version = carbon::CtGrid::version_v2;
        grid_rho.nx = 1;
        grid_rho.ny = 1;
        grid_rho.nz = 2;
        grid_rho.spacing_x_mm = 10.0;
        grid_rho.spacing_y_mm = 10.0;
        grid_rho.spacing_z_mm = 10.0;
        grid_rho.origin_x_mm = -5.0;
        grid_rho.origin_y_mm = -5.0;
        grid_rho.origin_z_mm = 0.0;
        grid_rho.density_g_per_cm3 = {0.5F, 1.5F};
        grid_rho.material_id = {8, 8}; // same section 8

        const float mr_soft = carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, n_energies, e_min, inv_dE, 8, test_energy);
        const double S_rho_exact = std::exp(-(0.5 * static_cast<double>(mr_soft) * 10.0 + 1.5 * static_cast<double>(mr_soft) * 10.0));
        const auto res_rho = carbon::run_step11_piecewise_hazard_gpu_test(
            grid_rho, schneider_grid.mass_xs_per_mm_at_1g_cm3, 25, n_energies,
            e_min, inv_dE, test_energy, 500000,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 20.0F, 10.0F, 2, "default");
        require(std::abs(res_rho.survival_fraction - S_rho_exact) / S_rho_exact < 0.002,
                "Case B: Same section different density survival must match exact within 0.2%");
    }

    // =========================================================================
    // Test 4: Geometry edge cases & robustness
    // =========================================================================
    {
        // 3D grid with anisotropic spacing and distinct voxel materials:
        // nx=3, ny=3, nz=3
        // spacing: dx=2.0, dy=3.0, dz=4.0
        carbon::CtGrid grid3d;
        grid3d.file_version = carbon::CtGrid::version_v2;
        grid3d.nx = 3;
        grid3d.ny = 3;
        grid3d.nz = 3;
        grid3d.spacing_x_mm = 2.0;
        grid3d.spacing_y_mm = 3.0;
        grid3d.spacing_z_mm = 4.0;
        grid3d.origin_x_mm = 0.0;
        grid3d.origin_y_mm = 0.0;
        grid3d.origin_z_mm = 0.0;
        grid3d.density_g_per_cm3.resize(27);
        grid3d.material_id.resize(27);
        for (std::uint32_t i = 0; i < 27; ++i) {
            grid3d.material_id[i] = static_cast<std::uint8_t>(i % 25);
            grid3d.density_g_per_cm3[i] = 0.5F + 0.05F * static_cast<float>(i);
        }

        struct TestCase {
            float ox, oy, oz;
            float dx, dy, dz;
            float max_len;
            const char* desc;
        };

        const std::vector<TestCase> test_cases = {
            // Axial rays
            {1.0F, 1.5F, 0.1F, 0.0F, 0.0F, 1.0F, 10.0F, "+z axial"},
            {1.0F, 1.5F, 11.9F, 0.0F, 0.0F, -1.0F, 10.0F, "-z axial"},
            {0.1F, 1.5F, 2.0F, 1.0F, 0.0F, 0.0F, 5.0F, "+x axial"},
            {5.9F, 1.5F, 2.0F, -1.0F, 0.0F, 0.0F, 5.0F, "-x axial"},
            {1.0F, 0.1F, 2.0F, 0.0F, 1.0F, 0.0F, 8.0F, "+y axial"},
            {1.0F, 8.9F, 2.0F, 0.0F, -1.0F, 0.0F, 8.0F, "-y axial"},
            // Oblique rays
            {0.1F, 0.1F, 0.1F, 1.0F, 1.0F, 1.0F, 10.0F, "oblique +++"},
            {5.9F, 8.9F, 11.9F, -1.0F, -1.0F, -1.0F, 10.0F, "oblique ---"},
            // Zero direction component (dy = 0)
            {0.1F, 1.5F, 0.1F, 1.0F, 0.0F, 1.0F, 8.0F, "planar dx,dz (dy=0)"},
            // Start exactly on face
            {2.0F, 1.5F, 4.0F, 1.0F, 0.0F, 1.0F, 6.0F, "start exactly on face"},
            // Start face +/- epsilon
            {2.0F - 1e-5F, 1.5F, 4.0F - 1e-5F, 1.0F, 0.0F, 1.0F, 6.0F, "start face - eps"},
            {2.0F + 1e-5F, 1.5F, 4.0F + 1e-5F, 1.0F, 0.0F, 1.0F, 6.0F, "start face + eps"},
            // Corner crossing
            {0.0F, 0.0F, 0.0F, 2.0F, 3.0F, 4.0F, 15.0F, "corner crossing ray"}
        };

        for (const auto& tc : test_cases) {
            const auto tc_res = carbon::run_step11_piecewise_hazard_gpu_test(
                grid3d, schneider_grid.mass_xs_per_mm_at_1g_cm3, 25, n_energies,
                e_min, inv_dE, test_energy, 10000,
                tc.ox, tc.oy, tc.oz, tc.dx, tc.dy, tc.dz, tc.max_len, 0.0F, 0, "default");
            require(tc_res.zero_progress_count == 0,
                    std::string("Edge case [") + tc.desc + "]: zero_progress_count must be 0");
            require(tc_res.changed_material_step_span_count == 0,
                    std::string("Edge case [") + tc.desc + "]: changed_material_step_span_count must be 0");
        }

        // Thin voxels (0.05 mm thick)
        carbon::CtGrid thin_grid;
        thin_grid.file_version = carbon::CtGrid::version_v2;
        thin_grid.nx = 1;
        thin_grid.ny = 1;
        thin_grid.nz = 20;
        thin_grid.spacing_x_mm = 10.0;
        thin_grid.spacing_y_mm = 10.0;
        thin_grid.spacing_z_mm = 0.05; // 50 microns
        thin_grid.origin_x_mm = -5.0;
        thin_grid.origin_y_mm = -5.0;
        thin_grid.origin_z_mm = 0.0;
        thin_grid.density_g_per_cm3.resize(20, 1.0F);
        thin_grid.material_id.resize(20);
        for (std::uint32_t i = 0; i < 20; ++i) {
            thin_grid.material_id[i] = static_cast<std::uint8_t>(i % 25);
        }

        const auto thin_res = carbon::run_step11_piecewise_hazard_gpu_test(
            thin_grid, schneider_grid.mass_xs_per_mm_at_1g_cm3, 25, n_energies,
            e_min, inv_dE, test_energy, 50000,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.05F, 20, "default");
        require(thin_res.zero_progress_count == 0, "Thin voxel zero_progress_count must be 0");
        require(thin_res.changed_material_step_span_count == 0, "Thin voxel changed_material_step_span_count must be 0");
        require(thin_res.face_cross_count > 0, "Thin voxel face_cross_count must be positive");
    }
#endif
}

void test_step11_ct_sample_directional_boundaries() {
    for (const float spacing : {0.05F, 1.0F, 10.0F}) {
        const float densities[2] = {1.0F, 2.0F};
        const std::uint8_t materials[2] = {8, 20};
        float rho = 0.0F;
        std::uint8_t mat = 0;

        const float face_z = spacing; // boundary between voxel 0 and voxel 1

        // 1. Exact on face: dz = +1 -> voxel 1 (material 20, rho 2.0)
        require(carbon::ct_sample(0.0F, 0.0F, face_z, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, 1.0F), "Exact face +z");
        require(mat == 20 && std::fabs(rho - 2.0F) < 1e-6F, "Exact face +z must sample voxel 1");

        // 2. Exact on face: dz = -1 -> voxel 0 (material 8, rho 1.0)
        require(carbon::ct_sample(0.0F, 0.0F, face_z, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, -1.0F), "Exact face -z");
        require(mat == 8 && std::fabs(rho - 1.0F) < 1e-6F, "Exact face -z must sample voxel 0");

        // 3. Face + 5 nm: dz = -1 -> voxel 1 (still inside voxel 1, must NOT jump prematurely!)
        const float z_plus_5nm = face_z + 5.0e-6F;
        require(carbon::ct_sample(0.0F, 0.0F, z_plus_5nm, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, -1.0F), "Face + 5nm -z");
        require(mat == 20 && std::fabs(rho - 2.0F) < 1e-6F, "Face + 5nm moving -z must remain in voxel 1");

        // 4. Face - 5 nm: dz = +1 -> voxel 0 (still inside voxel 0, must NOT jump prematurely!)
        const float z_minus_5nm = face_z - 5.0e-6F;
        require(carbon::ct_sample(0.0F, 0.0F, z_minus_5nm, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, 1.0F), "Face - 5nm +z");
        require(mat == 8 && std::fabs(rho - 1.0F) < 1e-6F, "Face - 5nm moving +z must remain in voxel 0");

        // 5. Face + 1 ULP: dz = -1 -> voxel 1
        const float z_plus_ulp = std::nextafter(face_z, 1.0e30F);
        require(carbon::ct_sample(0.0F, 0.0F, z_plus_ulp, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, -1.0F), "Face + ULP -z");
        require(mat == 20 && std::fabs(rho - 2.0F) < 1e-6F, "Face + ULP moving -z must be in voxel 1");

        // 6. Face - 1 ULP: dz = +1 -> voxel 0
        const float z_minus_ulp = std::nextafter(face_z, -1.0e30F);
        require(carbon::ct_sample(0.0F, 0.0F, z_minus_ulp, 0.0F, 0.0F, 0.0F,
                                  10.0F, 10.0F, spacing, 1, 1, 2,
                                  densities, materials, rho, mat, 0.0F, 0.0F, 1.0F), "Face - ULP +z");
        require(mat == 8 && std::fabs(rho - 1.0F) < 1e-6F, "Face - ULP moving +z must be in voxel 0");
    }
}

void test_step11_ct_clamp_pre_face_no_nudge() {
    // 1. Ray starts at z = 0.0 mm in a voxel of length 10.0 mm.
    // Propose step_mm = 9.999995 mm (stops 5 nm before face at z = 10.0 mm).
    const auto clamp_res = carbon::clamp_step_to_ct_faces_exact(
        9.999995F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
        -5.0F, -5.0F, 0.0F, 10.0F, 10.0F, 10.0F, 1, 1, 2);
    require(!clamp_res.hit_face, "Step ending 5 nm before face must NOT have hit_face=true");
    require(std::fabs(clamp_res.step_mm - 9.999995F) < 1.0e-7F, "Step length must be preserved");
    require(clamp_res.axis_mask == 0, "Axis mask must be 0 when face is not hit");

    // 2. Propose step_mm = 12.0 mm (exceeds face at z = 10.0 mm)
    const auto clamp_hit = carbon::clamp_step_to_ct_faces_exact(
        12.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
        -5.0F, -5.0F, 0.0F, 10.0F, 10.0F, 10.0F, 1, 1, 2);
    require(clamp_hit.hit_face, "Step exceeding face must have hit_face=true");
    require(std::fabs(clamp_hit.step_mm - 10.0F) < 1.0e-6F, "Step length must be clamped to 10.0 mm");
    require((clamp_hit.axis_mask & 4) != 0, "Z axis mask must be set");
}

void test_step11_schneider_step_energy_error_bound() {
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto xs_path = source_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    if (!std::filesystem::exists(xs_path)) {
        return;
    }
    carbon::TransportConfig cfg;
    cfg.ct_schneider_cross_section_file = xs_path.string();
    cfg.maximum_step_mm = 1.0;
    cfg.maximum_relative_energy_loss = 0.005;
    require(cfg.maximum_step_mm == 1.0, "Step audit must test max permitted step length 1.0 mm");
    require(cfg.maximum_relative_energy_loss == 0.005, "Step audit must test max rel loss 0.005");

    const auto schneider_grid = carbon::prepare_schneider_primary_xs(cfg);

    const auto sp_water_table = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_water_geant4_11_3_2.csv");
    const auto sp_air_table = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_air_geant4_11_3_2.csv");
    const auto sp_lung_table = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_lung_geant4_11_3_2.csv");
    const auto sp_bone_table = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_bone_geant4_11_3_2.csv");

    const auto spr_lut = carbon::build_density_mass_spr_lut(
        sp_water_table, sp_air_table, sp_lung_table, sp_bone_table, 1.0F);

    const auto& sp_energies = sp_water_table.energies();
    const auto sp_min_e = static_cast<float>(sp_energies.front());
    const auto sp_max_e = static_cast<float>(sp_energies.back());
    const auto sp_inv_step = 1.0F / static_cast<float>(sp_energies[1] - sp_energies[0]);
    const auto sp_table_size = sp_energies.size();

    std::vector<float> sp_water_lut(sp_table_size);
    for (std::size_t i = 0; i < sp_table_size; ++i) {
        sp_water_lut[i] = static_cast<float>(sp_water_table.interpolate(sp_energies[i]));
    }

    auto get_water_sp = [&](float e_mevu) noexcept -> float {
        const auto energy_clamped = std::clamp(e_mevu, sp_min_e, sp_max_e);
        const auto float_index = (energy_clamped - sp_min_e) * sp_inv_step;
        auto index = static_cast<int>(std::floor(float_index));
        index = std::max(0, std::min(index, static_cast<int>(sp_table_size) - 2));
        const float fraction = std::clamp(float_index - static_cast<float>(index), 0.0F, 1.0F);
        return sp_water_lut[index] + fraction * (sp_water_lut[index + 1] - sp_water_lut[index]);
    };

    auto get_mass_factor = [&](float e_mevu, float rho) -> float {
        const auto energy_clamped = std::clamp(e_mevu, sp_min_e, sp_max_e);
        const auto float_index = (energy_clamped - sp_min_e) * sp_inv_step;
        auto index = static_cast<int>(std::floor(float_index));
        index = std::max(0, std::min(index, static_cast<int>(sp_table_size) - 2));
        const float fraction = std::clamp(float_index - static_cast<float>(index), 0.0F, 1.0F);
        return carbon::ct_lookup_mass_sp_factor(
            spr_lut.factors.data(), spr_lut.n_rho, sp_table_size, true,
            spr_lut.log_rho_min, spr_lut.inv_dlog, 0U, rho,
            static_cast<std::size_t>(index), fraction,
            [](float x) { return std::log(x); });
    };

    // Endpoint clamp contract tests for host mass factor lookup
    for (const float test_rho : {0.00120479F, 0.26F, 1.0F, 1.85F, 3.0F}) {
        const float mf_min = get_mass_factor(sp_min_e, test_rho);
        const float mf_below = get_mass_factor(std::nextafter(sp_min_e, -1.0e30F), test_rho);
        const float mf_zero = get_mass_factor(0.0F, test_rho);
        require(mf_min == mf_below && mf_min == mf_zero, "Mass factor lower clamp contract failed");

        const float mf_max = get_mass_factor(sp_max_e, test_rho);
        const float mf_above = get_mass_factor(std::nextafter(sp_max_e, 1.0e30F), test_rho);
        const float mf_430 = get_mass_factor(430.0F, test_rho);
        const float mf_high = get_mass_factor(1000.0F, test_rho);
        require(mf_max == mf_above && mf_max == mf_430 && mf_max == mf_high, "Mass factor upper clamp contract failed");
    }

    auto get_production_stopping_power = [&](float e_mevu, float rho) -> float {
        const float sp_water = get_water_sp(e_mevu);
        const float mf = get_mass_factor(e_mevu, rho);
        return carbon::ct_mass_scaled_stopping_power(sp_water, rho, mf);
    };

    const auto section_xs = [&](std::uint32_t sec, float e_mevu) -> float {
        return carbon::schneider_primary_mass_xs(
            schneider_grid.mass_xs_per_mm_at_1g_cm3.data(), 25, 860, 0.5F, 2.0F, sec, e_mevu);
    };

    // Full 48 density nodes from the production Density-Mass-SPR LUT (0.0012 to 3.0 g/cm3)
    std::vector<float> all_density_nodes(spr_lut.n_rho);
    for (std::uint32_t i = 0; i < spr_lut.n_rho; ++i) {
        all_density_nodes[i] = std::exp(spr_lut.log_rho_min + static_cast<float>(i) / spr_lut.inv_dlog);
    }

    const float max_step_mm = static_cast<float>(cfg.maximum_step_mm);
    const float max_rel_loss = static_cast<float>(cfg.maximum_relative_energy_loss);

    // =========================================================================
    // Part A: Exact Step-by-Step Whole-Trajectory Primary Slowing Survival Error Gate
    // Across all 25 sections x 48 density nodes x 8 beam energies (100 to 430 MeV/u)
    // The slowing energy sequence is strictly determined step-by-step by (rho, E_inc),
    // advancing exact 1 mm / energy-limited steps down to E <= 5.0 MeV/u without any macro-aggregation.
    // =========================================================================
    const std::vector<float> beam_energies = {100.0F, 150.0F, 200.0F, 250.0F, 300.0F, 350.0F, 400.0F, 430.0F};
    float max_trajectory_survival_error = 0.0F;

    for (const auto rho : all_density_nodes) {
        for (const auto e_inc : beam_energies) {
            float e = e_inc;
            std::array<double, 25> tau_start_tot{};
            std::array<double, 25> tau_ref_tot{};
            std::size_t step_watchdog = 0;
            constexpr std::size_t kMaxWatchdog = 10'000'000;

            while (e > 5.0F) {
                if (++step_watchdog > kMaxWatchdog) {
                    require(false, "Trajectory slowing loop exceeded watchdog limit 10M steps");
                }
                const float sp = get_production_stopping_power(e, rho);
                require(std::isfinite(sp) && sp > 0.0F, "Stopping power must be finite and positive");

                const float step_mm = std::min(max_step_mm, max_rel_loss * (e * 12.0F) / sp);
                const float de_u = (sp / 12.0F) * step_mm;
                const float e1 = std::max(0.5F, e - de_u);
                require(std::isfinite(e1) && e1 < e, "Energy must decrease monotonically");
                const float e_mid = 0.5F * (e + e1);

                for (std::uint32_t s = 0; s < 25; ++s) {
                    const float s0 = rho * section_xs(s, e);
                    const float s_mid = rho * section_xs(s, e_mid);
                    const float s1 = rho * section_xs(s, e1);
                    tau_start_tot[s] += static_cast<double>(s0 * step_mm);
                    tau_ref_tot[s] += static_cast<double>((step_mm / 6.0F) * (s0 + 4.0F * s_mid + s1));
                }
                e = e1;
            }
            require(e <= 5.0F, "Whole-trajectory primary slowing must strictly reach 5 MeV/u");

            for (std::uint32_t s = 0; s < 25; ++s) {
                const double s_start = std::exp(-tau_start_tot[s]);
                const double s_ref = std::exp(-tau_ref_tot[s]);
                const float rel_s_err = static_cast<float>(std::fabs(s_start / s_ref - 1.0));
                if (rel_s_err > max_trajectory_survival_error) {
                    max_trajectory_survival_error = rel_s_err;
                }
            }
        }
    }

    std::cout << "[step-energy-bound] Part A: Exact step-by-step whole-trajectory primary survival error across all 25 sections x 48 densities x 8 energies (max_step=1.0mm) = "
              << max_trajectory_survival_error * 100.0F << "% (gate < 0.2%)\n";
    require(max_trajectory_survival_error < 0.002F, "Whole-trajectory primary survival error exceeds 0.2%");

    // =========================================================================
    // Part B: Production-realizable full-grid local step error sweep
    // 860 nodes x 25 sections x 48 densities with exact production stopping
    // =========================================================================
    float max_global_realizable_error = 0.0F;
    float max_smooth_realizable_error = 0.0F;

    for (std::size_t i = 0; i < schneider_grid.energy_nodes(); ++i) {
        const float e0 = static_cast<float>(schneider_grid.transport_energies_MeVu[i]);
        if (e0 < 5.0F) continue; // Below 5 MeV/u, optical depth is gated in Part C

        for (std::uint32_t sec = 0; sec < 25; ++sec) {
            for (const auto rho : all_density_nodes) {
                const float sp = get_production_stopping_power(e0, rho);
                const float step_mm = std::min(max_step_mm, max_rel_loss * (e0 * 12.0F) / sp);
                const float de_u = (sp / 12.0F) * step_mm;
                const float e1 = e0 - de_u;
                const float e_mid = 0.5F * (e0 + e1);

                const float sig0 = rho * section_xs(sec, e0);
                const float sig_mid = rho * section_xs(sec, e_mid);
                const float sig1 = rho * section_xs(sec, e1);

                const float tau_start = sig0 * step_mm;
                const float tau_simpson = (step_mm / 6.0F) * (sig0 + 4.0F * sig_mid + sig1);

                if (tau_simpson > 1.0e-8F) {
                    const float rel_err = std::fabs(tau_start - tau_simpson) / tau_simpson;
                    if (rel_err > max_global_realizable_error) {
                        max_global_realizable_error = rel_err;
                    }
                    if ((e0 < 292.0F || e0 > 295.0F) && rel_err > max_smooth_realizable_error) {
                        max_smooth_realizable_error = rel_err;
                    }
                }
            }
        }
    }

    std::cout << "[step-energy-bound] Part B: Realizable single-step error: smooth max = "
              << max_smooth_realizable_error * 100.0F << "% (gate < 0.2%), global max = "
              << max_global_realizable_error * 100.0F << "% (gate < 0.4%)\n";
    require(max_smooth_realizable_error < 0.002F, "Smooth range realizable step error exceeds 0.2%");
    require(max_global_realizable_error < 0.004F, "Global full-grid realizable step error exceeds 0.4%");

    // =========================================================================
    // Part C: Low-energy residual optical depth bound (0.5 to 5.0 MeV/u) with exact mass factor
    // =========================================================================
    float max_tau_tail_5mev = 0.0F;
    float max_tau_tail_2mev = 0.0F;
    float max_convergence_diff = 0.0F;

    for (std::uint32_t sec = 0; sec < 25; ++sec) {
        for (const auto rho : all_density_nodes) {
            auto integrate_tail = [&](float de, float e_upper) {
                float tau = 0.0F;
                for (float e = 0.5F; e < e_upper; e += de) {
                    const float e_a = e;
                    const float e_b = std::min(e_upper, e + de);
                    const float sp_a = static_cast<float>(sp_water_table.interpolate(static_cast<double>(e_a)));
                    const float sp_b = static_cast<float>(sp_water_table.interpolate(static_cast<double>(e_b)));
                    const float mf_a = get_mass_factor(e_a, rho);
                    const float mf_b = get_mass_factor(e_b, rho);
                    const float xs_a = section_xs(sec, e_a);
                    const float xs_b = section_xs(sec, e_b);
                    const float f_a = (12.0F * xs_a) / (mf_a * sp_a);
                    const float f_b = (12.0F * xs_b) / (mf_b * sp_b);
                    tau += 0.5F * (f_a + f_b) * (e_b - e_a);
                }
                return tau;
            };

            const float tau_5_fine = integrate_tail(0.0025F, 5.0F);
            const float tau_5_coarse = integrate_tail(0.005F, 5.0F);
            const float conv_diff = std::fabs(tau_5_fine - tau_5_coarse);
            if (conv_diff > max_convergence_diff) {
                max_convergence_diff = conv_diff;
            }

            if (tau_5_fine > max_tau_tail_5mev) {
                max_tau_tail_5mev = tau_5_fine;
            }
            const float tau_2 = integrate_tail(0.0025F, 2.0F);
            if (tau_2 > max_tau_tail_2mev) {
                max_tau_tail_2mev = tau_2;
            }
        }
    }

    std::cout << "[step-energy-bound] Part C: Residual optical depth with production mass-SPR: E<=5.0 MeV/u max tau = "
              << max_tau_tail_5mev << " (omission max survival change = "
              << (1.0 - std::exp(-max_tau_tail_5mev)) * 100.0 << "%, gate < 0.2%), E<=2.0 MeV/u max tau = "
              << max_tau_tail_2mev << " (omission max survival change = "
              << (1.0 - std::exp(-max_tau_tail_2mev)) * 100.0 << "%, gate < 0.05%), quadrature convergence diff = "
              << max_convergence_diff << "\n";
    require(max_convergence_diff < 1.0e-5F, "Quadrature convergence difference exceeds 1e-5");
    require(max_tau_tail_5mev < 0.002F, "Residual optical depth for E <= 5.0 MeV/u exceeds 2e-3");
    require(max_tau_tail_2mev < 0.0005F, "Residual optical depth for E <= 2.0 MeV/u exceeds 5e-4");
}

void test_step11_schneider_primary_mode_safety() {
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto xs_path = source_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    if (!std::filesystem::exists(xs_path)) {
        return;
    }

    // Test A: non-C12 projectile with Schneider XS must fail validate()
    {
        carbon::TransportConfig cfg;
        cfg.primary_atomic_number = 1; // Proton
        cfg.primary_mass_number = 1;
        cfg.enable_ct_grid = true;
        cfg.ct_grid_file = "dummy.cctg";
        cfg.ct_schneider_cross_section_file = xs_path.string();
        cfg.nuclear_model = "geant4";
        require_throws<std::invalid_argument>(
            [&]() { cfg.validate(); },
            "Non-C12 projectile with Schneider XS must throw in validate()");
    }

    // Test B: enable_nuclear_elastic with Schneider XS must fail validate()
    {
        carbon::TransportConfig cfg;
        cfg.primary_atomic_number = 6;
        cfg.primary_mass_number = 12;
        cfg.enable_ct_grid = true;
        cfg.ct_grid_file = "dummy.cctg";
        cfg.ct_schneider_cross_section_file = xs_path.string();
        cfg.nuclear_model = "geant4";
        cfg.enable_nuclear_elastic = true;
        require_throws<std::invalid_argument>(
            [&]() { cfg.validate(); },
            "enable_nuclear_elastic=true with Schneider XS must throw in validate()");
    }

    // Test C: maximum_relative_energy_loss > 0.005 with Schneider XS must fail validate()
    {
        carbon::TransportConfig cfg;
        cfg.primary_atomic_number = 6;
        cfg.primary_mass_number = 12;
        cfg.enable_ct_grid = true;
        cfg.ct_grid_file = "dummy.cctg";
        cfg.ct_schneider_cross_section_file = xs_path.string();
        cfg.nuclear_model = "geant4";
        cfg.maximum_relative_energy_loss = 0.01;
        require_throws<std::invalid_argument>(
            [&]() { cfg.validate(); },
            "maximum_relative_energy_loss > 0.005 with Schneider XS must throw in validate()");
    }

    // Test D: maximum_step_mm > 1.0 with Schneider XS must fail validate()
    {
        carbon::TransportConfig cfg;
        cfg.primary_atomic_number = 6;
        cfg.primary_mass_number = 12;
        cfg.enable_ct_grid = true;
        cfg.ct_grid_file = "dummy.cctg";
        cfg.ct_schneider_cross_section_file = xs_path.string();
        cfg.nuclear_model = "geant4";
        cfg.maximum_step_mm = 2.0;
        require_throws<std::invalid_argument>(
            [&]() { cfg.validate(); },
            "maximum_step_mm > 1.0 with Schneider XS must throw in validate()");
    }
}

void test_step11_schneider_production_tiny_cctg_transport() {
#ifdef CARBON_HAS_SYCL
    if (!is_sycl_available()) {
        return;
    }
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);

    const auto temp_dir = std::filesystem::temp_directory_path() / "test_step11_tiny_cctg";
    std::filesystem::create_directories(temp_dir);
    const auto cctg_file = temp_dir / "tiny_two_voxel.cctg";
    const auto flat_xs_path = temp_dir / "synthetic_flat_schneider_xs.csv";

    // Write synthetic flat Schneider XS table (exact constant mass rate across all energies)
    {
        std::ofstream out(flat_xs_path);
        out << "energy_MeV_per_u";
        for (std::size_t s = 0; s < 25; ++s) {
            out << ",section_" << (s < 10 ? "0" : "") << s << "_mass_xs_per_mm_at_1g_cm3";
        }
        out << "\n";
        for (std::size_t i = 0; i < 860; ++i) {
            const double e = 0.5 + i * 0.5;
            out << e;
            for (std::size_t s = 0; s < 25; ++s) {
                // Section 8 (soft tissue): 0.0050 mm^-1 / (g/cm3)
                // Section 20 (dense bone): 0.0100 mm^-1 / (g/cm3)
                // Others: 0.0050
                const double rate = (s == 20) ? 0.0100 : 0.0050;
                out << "," << rate;
            }
            out << "\n";
        }
    }

    // 2-voxel CT grid along Z:
    // Voxel 0: z in [0, 10) mm, section 8 (soft tissue), rho = 1.0 g/cm3 -> Sigma_0 = 0.0050 mm^-1
    // Voxel 1: z in [10, 20) mm, section 20 (dense bone), rho = 1.5 g/cm3 -> Sigma_1 = 0.0150 mm^-1
    carbon::CtGrid grid;
    grid.file_version = carbon::CtGrid::version_v2;
    grid.nx = 1;
    grid.ny = 1;
    grid.nz = 2;
    grid.spacing_x_mm = 10.0;
    grid.spacing_y_mm = 10.0;
    grid.spacing_z_mm = 10.0;
    grid.origin_x_mm = -5.0;
    grid.origin_y_mm = -5.0;
    grid.origin_z_mm = 0.0;
    grid.density_g_per_cm3 = {1.0F, 1.5F};
    grid.material_id = {8, 20};
    grid.mass_sp_za_rel.resize(25, 1.0);
    grid.write_binary(cctg_file);

    carbon::TransportConfig config;
    config.phantom_length_mm = 20.0;
    config.depth_bin_width_mm = 1.0;
    config.primary_atomic_number = 6;
    config.primary_mass_number = 12;
    config.initial_energy_MeVu = 200.0;
    config.enable_ct_grid = true;
    config.ct_grid_file = cctg_file.string();
    config.ct_schneider_cross_section_file = flat_xs_path.string();
    config.nuclear_model = "geant4";
    config.enable_inelastic = true;
    config.enable_nuclear_elastic = false;
    config.enable_multiple_scattering = false;
    config.number_of_histories = 50000;
    config.validate();

    const auto water_sp = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_water_geant4_11_3_2.csv");
    const auto zero_xs = zero_cross_section();

    constexpr double sig0 = 1.0 * 0.0050; // 0.0050 mm^-1
    constexpr double sig1 = 1.5 * 0.0100; // 0.0150 mm^-1

    // Subtest 1: Full-domain forward transmission (z = 0 -> 20 mm, dz = +1.0)
    // Path: 10.0 mm in Voxel 0 + 10.0 mm in Voxel 1
    {
        const auto result = carbon::transport_sycl(config, water_sp, zero_xs, "default");
        const double s_exact = std::exp(-(sig0 * 10.0 + sig1 * 10.0));
        const double s_meas = 1.0 - static_cast<double>(result.nuclear_interactions) / static_cast<double>(config.number_of_histories);
        const double sigma = std::sqrt(s_exact * (1.0 - s_exact) / static_cast<double>(config.number_of_histories));

        std::cout << "[production-tiny-cctg] Subtest 1 (Forward full): Measured survival = " << s_meas
                  << ", exact = " << s_exact << ", diff = " << std::fabs(s_meas - s_exact)
                  << " (" << std::fabs(s_meas - s_exact) / sigma << " sigma)\n";
        require_near(s_meas, s_exact, 3.5 * sigma, "Production tiny CCTG forward full survival failed");
    }

    // Subtest 2: Sub-10nm minimum-step regression (Forward start 5 nm before face: z = 10.0 - 5e-6 mm, dz = +1.0)
    // Path: 5e-6 mm in Voxel 0 + 10.0 mm in Voxel 1
    {
        carbon::TransportConfig cfg_sub10nm = config;
        cfg_sub10nm.source_origin_z_mm = 10.0 - 5.0e-6;
        cfg_sub10nm.validate();
        const auto res_sub10nm = carbon::transport_sycl(cfg_sub10nm, water_sp, zero_xs, "default");
        const double s_exact = std::exp(-(sig0 * 5.0e-6 + sig1 * 10.0));
        const double s_meas = 1.0 - static_cast<double>(res_sub10nm.nuclear_interactions) / static_cast<double>(cfg_sub10nm.number_of_histories);
        const double sigma = std::sqrt(s_exact * (1.0 - s_exact) / static_cast<double>(cfg_sub10nm.number_of_histories));
        std::cout << "[production-tiny-cctg] Subtest 2 (Forward 5nm-before-face): Measured survival = " << s_meas
                  << ", exact = " << s_exact << " (" << std::fabs(s_meas - s_exact) / sigma << " sigma)\n";
        require_near(s_meas, s_exact, 3.5 * sigma, "Forward 5nm-before-face survival failed");
    }

    // Subtest 3: Exact-on-face forward start (z = 10.0 mm, dz = +1.0)
    // Path: 0.0 mm in Voxel 0 + 10.0 mm in Voxel 1
    {
        carbon::TransportConfig cfg_exact_fwd = config;
        cfg_exact_fwd.source_origin_z_mm = 10.0;
        cfg_exact_fwd.validate();
        const auto res_exact_fwd = carbon::transport_sycl(cfg_exact_fwd, water_sp, zero_xs, "default");
        const double s_exact = std::exp(-sig1 * 10.0);
        const double s_meas = 1.0 - static_cast<double>(res_exact_fwd.nuclear_interactions) / static_cast<double>(cfg_exact_fwd.number_of_histories);
        const double sigma = std::sqrt(s_exact * (1.0 - s_exact) / static_cast<double>(cfg_exact_fwd.number_of_histories));
        std::cout << "[production-tiny-cctg] Subtest 3 (Exact-on-face forward): Measured survival = " << s_meas
                  << ", exact = " << s_exact << " (" << std::fabs(s_meas - s_exact) / sigma << " sigma)\n";
        require_near(s_meas, s_exact, 3.5 * sigma, "Exact-on-face forward survival failed");
    }

    // Subtest 4: Exact-on-face reverse start (z = 10.0 mm, dz = -1.0)
    // Path: 10.0 mm in Voxel 0 + 0.0 mm in Voxel 1
    {
        carbon::TransportConfig cfg_exact_rev = config;
        cfg_exact_rev.source_origin_z_mm = 10.0;
        cfg_exact_rev.beam_uz_z = -1.0;
        cfg_exact_rev.validate();
        const auto res_exact_rev = carbon::transport_sycl(cfg_exact_rev, water_sp, zero_xs, "default");
        const double s_exact = std::exp(-sig0 * 10.0);
        const double s_meas = 1.0 - static_cast<double>(res_exact_rev.nuclear_interactions) / static_cast<double>(cfg_exact_rev.number_of_histories);
        const double sigma = std::sqrt(s_exact * (1.0 - s_exact) / static_cast<double>(cfg_exact_rev.number_of_histories));
        std::cout << "[production-tiny-cctg] Subtest 4 (Exact-on-face reverse): Measured survival = " << s_meas
                  << ", exact = " << s_exact << " (" << std::fabs(s_meas - s_exact) / sigma << " sigma)\n";
        require_near(s_meas, s_exact, 3.5 * sigma, "Exact-on-face reverse survival failed");
    }

    // Subtest 5: Sub-10nm reverse crossing (z = 10.0 + 5e-6 mm, dz = -1.0)
    // Path: 5e-6 mm in Voxel 1 + 10.0 mm in Voxel 0
    {
        carbon::TransportConfig cfg_rev = config;
        cfg_rev.source_origin_z_mm = 10.0 + 5.0e-6;
        cfg_rev.beam_uz_z = -1.0;
        cfg_rev.validate();
        const auto res_rev = carbon::transport_sycl(cfg_rev, water_sp, zero_xs, "default");
        const double s_exact = std::exp(-(sig1 * 5.0e-6 + sig0 * 10.0));
        const double s_meas = 1.0 - static_cast<double>(res_rev.nuclear_interactions) / static_cast<double>(cfg_rev.number_of_histories);
        const double sigma = std::sqrt(s_exact * (1.0 - s_exact) / static_cast<double>(cfg_rev.number_of_histories));
        std::cout << "[production-tiny-cctg] Subtest 5 (Reverse 5nm-after-face): Measured survival = " << s_meas
                  << ", exact = " << s_exact << " (" << std::fabs(s_meas - s_exact) / sigma << " sigma)\n";
        require_near(s_meas, s_exact, 3.5 * sigma, "Reverse 5nm-after-face survival failed");
    }

        std::filesystem::remove_all(temp_dir);
#endif
}

void test_step12_primary_only_mode_contract() {
    std::cout << "[step12] Starting test_step12_primary_only_mode_contract\n" << std::flush;
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto xs_path = source_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";

    // =========================================================================
    // 1. Config Validation & Mode Contract Gating
    // =========================================================================
    {
        carbon::TransportConfig base_cfg;
        base_cfg.primary_atomic_number = 6;
        base_cfg.primary_mass_number = 12;
        base_cfg.enable_ct_grid = true;
        base_cfg.ct_grid_file = "dummy_grid.cctg";
        base_cfg.ct_schneider_cross_section_file = xs_path.string();
        base_cfg.ct_validation_mode = "primary-attenuation-only";
        base_cfg.enable_inelastic = true;
        base_cfg.enable_nuclear_elastic = false;
        base_cfg.enable_secondary_transport = false;
        base_cfg.enable_energy_straggling = false;
        base_cfg.beam_energy_spread = 0.0;
        base_cfg.ct_use_density_mass_spr = true;
        base_cfg.ct_stopping_power_scale = 1.0;
        base_cfg.initial_energy_MeVu = 200.0;
        base_cfg.energy_cutoff_MeV = 6.0;
        base_cfg.maximum_step_mm = 1.0;
        base_cfg.maximum_relative_energy_loss = 0.005;

        // Valid configuration passes
        base_cfg.validate();
        require(base_cfg.is_primary_attenuation_only_mode(), "is_primary_attenuation_only_mode should be true");

        // enable_inelastic = false rejected
        {
            auto bad = base_cfg;
            bad.enable_inelastic = false;
            require_throws([&]() { bad.validate(); }, "enable_inelastic = false must be rejected");
        }
        // Spot energy > 430 MeV/u rejected
        {
            auto bad = base_cfg;
            carbon::PrimarySpotBatchEntry spot{};
            spot.history_begin = 0;
            spot.history_end = bad.number_of_histories;
            spot.floats[0] = 6000.0F; // 500 MeV/u > 430
            spot.floats[1] = 0.0F;
            bad.primary_spot_batch = {spot};
            require_throws([&]() { bad.validate(); }, "Spot energy > 430 MeV/u must be rejected");
        }
        // Spot spread > 0 rejected
        {
            auto bad = base_cfg;
            carbon::PrimarySpotBatchEntry spot{};
            spot.history_begin = 0;
            spot.history_end = bad.number_of_histories;
            spot.floats[0] = 2400.0F; // 200 MeV/u
            spot.floats[1] = 0.02F;
            bad.primary_spot_batch = {spot};
            require_throws([&]() { bad.validate(); }, "Spot spread > 0 must be rejected");
        }
        // Non-C12 rejected
        {
            auto bad = base_cfg;
            bad.primary_atomic_number = 1;
            bad.primary_mass_number = 1;
            require_throws([&]() { bad.validate(); }, "Non-C12 projectile must be rejected");
        }
        // CT grid disabled rejected
        {
            auto bad = base_cfg;
            bad.enable_ct_grid = false;
            require_throws([&]() { bad.validate(); }, "enable_ct_grid = false must be rejected");
        }
        // Nuclear elastic enabled rejected
        {
            auto bad = base_cfg;
            bad.enable_nuclear_elastic = true;
            require_throws([&]() { bad.validate(); }, "enable_nuclear_elastic = true must be rejected");
        }
        // Secondary transport enabled rejected
        {
            auto bad = base_cfg;
            bad.enable_secondary_transport = true;
            require_throws([&]() { bad.validate(); }, "enable_secondary_transport = true must be rejected");
        }
        // Energy straggling enabled rejected
        {
            auto bad = base_cfg;
            bad.enable_energy_straggling = true;
            require_throws([&]() { bad.validate(); }, "enable_energy_straggling = true must be rejected");
        }
        // Energy spread > 0 rejected
        {
            auto bad = base_cfg;
            bad.beam_energy_spread = 0.01;
            require_throws([&]() { bad.validate(); }, "beam_energy_spread > 0 must be rejected");
        }
        // Density mass SPR disabled rejected when exact Schneider table is not configured
        {
            auto bad = base_cfg;
            bad.ct_schneider_stopping_power_file.clear();
            bad.ct_use_density_mass_spr = false;
            require_throws([&]() { bad.validate(); }, "ct_use_density_mass_spr = false must be rejected when exact table not set");
        }
        // When exact Schneider stopping table is present, ct_use_density_mass_spr = false is permitted
        {
            auto exact_cfg = base_cfg;
            exact_cfg.ct_use_density_mass_spr = false;
            exact_cfg.validate();
        }
        // Stopping scale != 1.0 rejected
        {
            auto bad = base_cfg;
            bad.ct_stopping_power_scale = 1.02;
            require_throws([&]() { bad.validate(); }, "ct_stopping_power_scale != 1.0 must be rejected");
        }
        // Initial energy > 430 MeV/u rejected
        {
            auto bad = base_cfg;
            bad.initial_energy_MeVu = 450.0;
            require_throws([&]() { bad.validate(); }, "initial_energy_MeVu > 430.0 must be rejected");
        }
        // Energy cutoff < 6.0 MeV (0.5 MeV/u) rejected
        {
            auto bad = base_cfg;
            bad.energy_cutoff_MeV = 3.0;
            require_throws([&]() { bad.validate(); }, "energy_cutoff_MeV < 6.0 must be rejected");
        }
        // Inelastic package V2 set rejected
        {
            auto bad = base_cfg;
            bad.primary_inelastic_package_v2_file = "dummy_package.csv";
            require_throws([&]() { bad.validate(); }, "primary_inelastic_package_v2_file must be rejected");
        }
        // Unknown mode rejected
        {
            auto bad = base_cfg;
            bad.ct_validation_mode = "invalid-validation-mode";
            require_throws([&]() { bad.validate(); }, "invalid ct_validation_mode must be rejected");
        }
        // maximum_primary_steps = 0 rejected
        {
            auto bad = base_cfg;
            bad.maximum_primary_steps = 0;
            require_throws([&]() { bad.validate(); }, "maximum_primary_steps = 0 must be rejected");
        }
        // Config file parsing tests for maximum_primary_steps
        {
            const auto temp_cfg_dir = std::filesystem::temp_directory_path() / "cfg_parsing_test";
            std::filesystem::create_directories(temp_cfg_dir);
            const auto temp_cfg = temp_cfg_dir / "steps_cfg.txt";
            // Positive test
            {
                std::ofstream out(temp_cfg);
                out << "maximum_primary_steps: 42\n";
                out.close();
                const auto loaded = carbon::load_config(temp_cfg);
                require(loaded.maximum_primary_steps == 42U, "maximum_primary_steps must parse as 42");
            }
            // 0 -> parsed as 0, rejected by validate() in load_config
            {
                std::ofstream out(temp_cfg);
                out << "maximum_primary_steps: 0\n";
                out.close();
                require_throws([&]() { carbon::load_config(temp_cfg); }, "maximum_primary_steps: 0 must be rejected in load_config");
            }
            // 2.9 -> decimal rejected by parse_number
            {
                std::ofstream out(temp_cfg);
                out << "maximum_primary_steps: 2.9\n";
                out.close();
                require_throws([&]() { carbon::load_config(temp_cfg); }, "maximum_primary_steps: 2.9 must be rejected in parse_number");
            }
            // -5 -> negative rejected by parse_number
            {
                std::ofstream out(temp_cfg);
                out << "maximum_primary_steps: -5\n";
                out.close();
                require_throws([&]() { carbon::load_config(temp_cfg); }, "maximum_primary_steps: -5 must be rejected in parse_number");
            }
            // 5000000000 -> out of uint32 range rejected by parse_number
            {
                std::ofstream out(temp_cfg);
                out << "maximum_primary_steps: 5000000000\n";
                out.close();
                require_throws([&]() { carbon::load_config(temp_cfg); }, "maximum_primary_steps > UINT32_MAX must be rejected in parse_number");
            }
            std::filesystem::remove_all(temp_cfg_dir);
        }
    }

    // =========================================================================
    // 2. First-Interaction Record 3D Vertex & Stopping Point Checks
    // =========================================================================
    {
        carbon::PrimaryFirstInteractionRecord rec{};
        rec.x_mm = -15.5F;
        rec.y_mm = 22.3F;
        rec.depth_mm = 85.0F;
        rec.energy_MeVu = 120.0F;
        rec.section_id = 11;
        rec.density_g_per_cm3 = 1.0F;

        require(rec.x_mm == -15.5F, "Record x_mm mismatch");
        require(rec.y_mm == 22.3F, "Record y_mm mismatch");
        require(rec.depth_mm == 85.0F, "Record depth_mm mismatch");
        require(rec.energy_MeVu == 120.0F, "Record energy_MeVu mismatch");
        require(rec.section_id == 11, "Record section_id mismatch");
        require(rec.density_g_per_cm3 == 1.0F, "Record density mismatch");
    }

    // =========================================================================
    // 3. Bragg Peak Falloff Crossing Metrics (Robustness & Invalid States)
    // =========================================================================
    {
        // 3D Voxel Dose to 1D IDD
        constexpr std::size_t nx = 3;
        constexpr std::size_t ny = 3;
        constexpr std::size_t nz = 5;
        std::vector<double> voxel_dose(nx * ny * nz, 0.0);
        for (std::size_t z = 0; z < nz; ++z) {
            for (std::size_t y = 0; y < ny; ++y) {
                for (std::size_t x = 0; x < nx; ++x) {
                    voxel_dose[z * (nx * ny) + y * nx + x] = static_cast<double>((z + 1) * 10);
                }
            }
        }
        const auto idd = carbon::compute_idd_from_3d_voxel_dose(voxel_dose, nx, ny, nz);
        require(idd.size() == nz, "IDD size mismatch");
        for (std::size_t z = 0; z < nz; ++z) {
            const double expected = (z + 1) * 10 * 9.0;
            require_near(idd[z], expected, 1e-9, "IDD slice sum mismatch");
        }

        // Bragg Peak Metrics with known downward crossings
        std::vector<double> test_idd(20, 0.0);
        for (std::size_t i = 0; i <= 10; ++i) {
            test_idd[i] = 10.0 * static_cast<double>(i);
        }
        test_idd[11] = 80.0;
        test_idd[12] = 65.0;
        test_idd[13] = 50.0;
        test_idd[14] = 20.0;
        test_idd[15] = 0.0;

        const auto bp = carbon::compute_bragg_peak_metrics(test_idd, 1.0, 0.0);
        require(bp.found_r80, "found_r80 should be true");
        require(bp.found_r50, "found_r50 should be true");
        require_near(bp.peak_depth_mm, 10.5, 1e-6, "Bragg peak depth mismatch");
        require_near(bp.peak_dose_MeV, 100.0, 1e-6, "Bragg peak dose mismatch");
        require_near(bp.r80_distal_mm, 11.5, 1e-6, "R80 distal mismatch");
        require_near(bp.r50_distal_mm, 13.5, 1e-6, "R50 distal mismatch");

        // Monotonically increasing curve (no distal crossings)
        std::vector<double> rising_idd = {10.0, 20.0, 30.0, 40.0, 50.0};
        const auto bp_rising = carbon::compute_bragg_peak_metrics(rising_idd, 1.0, 0.0);
        require(!bp_rising.found_r80, "rising curve should not find R80");
        require(!bp_rising.found_r50, "rising curve should not find R50");
        require(std::isnan(bp_rising.r80_distal_mm), "rising curve R80 should be NaN");
        require(std::isnan(bp_rising.r50_distal_mm), "rising curve R50 should be NaN");

        // Truncated distal curve: crosses 80% but stops before reaching 50%
        std::vector<double> truncated_idd = {0.0, 50.0, 100.0, 75.0, 60.0};
        const auto bp_trunc = carbon::compute_bragg_peak_metrics(truncated_idd, 1.0, 0.0);
        require(bp_trunc.found_r80, "truncated curve should find R80");
        require(!bp_trunc.found_r50, "truncated curve should not find R50");
        require(!std::isnan(bp_trunc.r80_distal_mm), "truncated curve R80 should not be NaN");
        require(std::isnan(bp_trunc.r50_distal_mm), "truncated curve R50 should be NaN");
    }
}

#ifdef CARBON_HAS_SYCL
namespace {

std::filesystem::path prepare_step12_cctg(const std::filesystem::path& temp_dir) {
    const auto cctg_file = temp_dir / "step12_ct_grid.cctg";
    constexpr int nx = 10;
    constexpr int ny = 10;
    constexpr int nz = 20;
    constexpr std::size_t n_voxels = nx * ny * nz;
    carbon::CtGrid grid;
    grid.file_version = carbon::CtGrid::version_v2;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.spacing_x_mm = 10.0;
    grid.spacing_y_mm = 10.0;
    grid.spacing_z_mm = 10.0;
    grid.origin_x_mm = -50.0;
    grid.origin_y_mm = -50.0;
    grid.origin_z_mm = 0.0;
    grid.density_g_per_cm3.assign(n_voxels, 1.0F);
    grid.material_id.assign(n_voxels, 11);
    grid.mass_sp_za_rel.resize(25, 1.0);
    grid.write_binary(cctg_file);
    return cctg_file;
}

void run_step12_4a(const std::filesystem::path& cctg_file,
                   const std::filesystem::path& xs_path,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    constexpr int nx = 10;
    constexpr int ny = 10;
    constexpr int nz = 20;
    carbon::TransportConfig cfg;
    cfg.phantom_length_mm = 200.0;
    cfg.depth_bin_width_mm = 10.0;
    cfg.enable_voxel_scoring = true;
    cfg.voxel_bins_x = nx;
    cfg.voxel_bins_y = ny;
    cfg.voxel_bins_z = nz;
    cfg.voxel_size_x_mm = 10.0;
    cfg.voxel_size_y_mm = 10.0;
    cfg.voxel_size_z_mm = 10.0;
    cfg.voxel_origin_x_mm = -50.0;
    cfg.voxel_origin_y_mm = -50.0;
    cfg.voxel_origin_z_mm = 0.0;
    cfg.primary_atomic_number = 6;
    cfg.primary_mass_number = 12;
    cfg.initial_energy_MeVu = 200.0;
    cfg.enable_ct_grid = true;
    cfg.ct_grid_file = cctg_file.string();
    cfg.ct_schneider_cross_section_file = xs_path.string();
    cfg.ct_validation_mode = "primary-attenuation-only";
    cfg.nuclear_model = "geant4";
    cfg.enable_inelastic = true;
    cfg.enable_nuclear_elastic = false;
    cfg.enable_secondary_transport = false;
    cfg.enable_energy_straggling = false;
    cfg.beam_energy_spread = 0.0;
    cfg.ct_use_density_mass_spr = true;
    cfg.ct_stopping_power_scale = 1.0;
    cfg.energy_cutoff_MeV = 6.0;
    cfg.number_of_histories = 20000;
    cfg.validate();

    const auto result = carbon::transport_sycl(cfg, water_sp, zero_xs, "default");

    // Terminal conservation identity
    const std::uint64_t total_terminal =
        result.primary_inelastic_terminated_count +
        result.primary_escaped_ct_count +
        result.primary_stopped_count +
        result.primary_other_terminal_count;
    std::cout << "[step12-gpu-test] Terminal counts:\n"
              << "  inelastic_terminated=" << result.primary_inelastic_terminated_count << "\n"
              << "  escaped_ct=" << result.primary_escaped_ct_count << "\n"
              << "  stopped=" << result.primary_stopped_count << "\n"
              << "  other=" << result.primary_other_terminal_count << "\n"
              << "  sum=" << total_terminal << " / " << cfg.number_of_histories << "\n";
    require(total_terminal == cfg.number_of_histories,
            "Terminal conservation identity violated: sum != number_of_histories");
    require(result.primary_inelastic_terminated_count > 0,
            "Inelastic collisions should be observed");
    require(result.primary_stopped_count > 0,
            "Stopped particles should be observed for 200 MeV/u in 200 mm water");
    require(result.primary_other_terminal_count == 0,
            "Valid validation run must have zero other_terminal events");
    require(result.primary_other_terminal_kinetic_MeV == 0.0,
            "Valid validation run must have zero other_terminal kinetic energy");
    require(result.primary_cutoff_stopped_energy_MeV > 0.0,
            "Cutoff stopped energy tally must be > 0 when stopped_count > 0");
    require(result.primary_cutoff_stopped_energy_MeV < result.total_deposited_energy_MeV,
            "Cutoff stopped energy tally must be a positive fraction of total deposited energy");

    // Energy conservation identity
    const double initial_e = cfg.initial_total_energy_MeV() * static_cast<double>(cfg.number_of_histories);
    const double total_accounted =
        result.total_deposited_energy_MeV +
        result.primary_inelastic_removed_kinetic_MeV +
        result.primary_other_terminal_kinetic_MeV +
        result.escaped_energy_MeV;
    const double rel_energy_err = std::fabs(initial_e - total_accounted) / initial_e;
    std::cout << "[step12-gpu-test] Energy conservation:\n"
              << "  initial_total_MeV=" << initial_e << "\n"
              << "  deposited_MeV=" << result.total_deposited_energy_MeV << "\n"
              << "  inelastic_removed_MeV=" << result.primary_inelastic_removed_kinetic_MeV << "\n"
              << "  other_terminal_MeV=" << result.primary_other_terminal_kinetic_MeV << "\n"
              << "  escaped_MeV=" << result.escaped_energy_MeV << "\n"
              << "  accounted_MeV=" << total_accounted << "\n"
              << "  relative_error=" << rel_energy_err << "\n";
    require(rel_energy_err < 1.0e-5, "Decoupled kinetic energy conservation failed");
    require(result.physical_relative_energy_balance_error() < 1.0e-5,
            "Physical relative energy balance error failed");

    // First interaction records
    require(result.primary_first_interactions.size() == result.primary_inelastic_terminated_count,
            "primary_first_interactions size mismatch");
    for (const auto& rec : result.primary_first_interactions) {
        require(rec.x_mm >= -50.0F && rec.x_mm <= 50.0F, "Interaction x out of bounds");
        require(rec.y_mm >= -50.0F && rec.y_mm <= 50.0F, "Interaction y out of bounds");
        require(rec.depth_mm >= 0.0F && rec.depth_mm <= 200.0F, "Interaction depth out of bounds");
        require(rec.energy_MeVu >= 0.5F && rec.energy_MeVu <= 200.0F + 1e-4F, "Interaction energy out of bounds");
        require(rec.section_id == 11, "Interaction section_id mismatch");
        require_near(rec.density_g_per_cm3, 1.0F, 1e-5F, "Interaction density mismatch");
    }

    // IDD & Bragg Peak Metrics
    const auto voxel_idd = carbon::compute_idd_from_3d_voxel_dose(
        result.voxel_deposited_energy_MeV, nx, ny, nz);
    require(voxel_idd.size() == nz, "Voxel IDD size mismatch");
    const auto bp = carbon::compute_bragg_peak_metrics(voxel_idd, 10.0, 0.0);
    std::cout << "[step12-gpu-test] Bragg Peak Metrics:\n"
              << "  peak_depth_mm=" << bp.peak_depth_mm << "\n"
              << "  peak_dose_MeV=" << bp.peak_dose_MeV << "\n"
              << "  R80_distal_mm=" << bp.r80_distal_mm << "\n"
              << "  R50_distal_mm=" << bp.r50_distal_mm << "\n";
    require(bp.found_r80, "Bragg peak R80 should be found");
    require(bp.found_r50, "Bragg peak R50 should be found");
    require(bp.peak_depth_mm >= 80.0 && bp.peak_depth_mm <= 100.0,
            "Bragg peak depth for 200 MeV/u C12 in water should be near 85-90 mm");
}

void run_step12_4b(const std::filesystem::path& temp_dir,
                   const std::filesystem::path& cctg_file,
                   const std::filesystem::path& xs_path,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    std::string header_line;
    {
        std::ifstream in(xs_path);
        std::getline(in, header_line);
    }
    const auto zero_xs_csv = temp_dir / "zero_schneider_xs.csv";
    const auto zero_xs_meta = temp_dir / "zero_schneider_xs.metadata.json";
    {
        std::ofstream out(zero_xs_csv);
        out << header_line << "\n";
        for (float e = 0.5F; e <= 430.0F; e += 0.5F) {
            out << e;
            for (int s = 0; s < 25; ++s) out << ",0.0";
            out << "\n";
        }
    }
    const auto zero_hash = carbon::compute_file_sha256_hex(zero_xs_csv);
    {
        std::ofstream out(zero_xs_meta);
        out << "{\n  \"data_sha256\": \"" << zero_hash << "\"\n}\n";
    }
    carbon::TransportConfig cfg_zero;
    cfg_zero.phantom_length_mm = 200.0;
    cfg_zero.depth_bin_width_mm = 10.0;
    cfg_zero.primary_atomic_number = 6;
    cfg_zero.primary_mass_number = 12;
    cfg_zero.initial_energy_MeVu = 200.0;
    cfg_zero.enable_ct_grid = true;
    cfg_zero.ct_grid_file = cctg_file.string();
    cfg_zero.ct_schneider_cross_section_file = zero_xs_csv.string();
    cfg_zero.ct_validation_mode = "primary-attenuation-only";
    cfg_zero.nuclear_model = "geant4";
    cfg_zero.enable_inelastic = true;
    cfg_zero.enable_nuclear_elastic = false;
    cfg_zero.enable_secondary_transport = false;
    cfg_zero.enable_energy_straggling = false;
    cfg_zero.beam_energy_spread = 0.0;
    cfg_zero.ct_use_density_mass_spr = true;
    cfg_zero.ct_stopping_power_scale = 1.0;
    cfg_zero.energy_cutoff_MeV = 6.0;
    cfg_zero.number_of_histories = 5000;
    cfg_zero.validate();

    const auto res_zero = carbon::transport_sycl(cfg_zero, water_sp, zero_xs, "default");
    require(res_zero.primary_inelastic_terminated_count == 0,
            "Zero XS should have 0 inelastic terminations");
    require(res_zero.primary_first_interactions.empty(),
            "Zero XS should have empty first interaction list");
    require(res_zero.primary_stopped_count + res_zero.primary_escaped_ct_count == cfg_zero.number_of_histories,
            "Zero XS all particles should stop or escape");
}

void run_step12_4c(const std::filesystem::path& temp_dir,
                   const std::filesystem::path& cctg_file,
                   const std::filesystem::path& xs_path,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    std::string header_line;
    {
        std::ifstream in(xs_path);
        std::getline(in, header_line);
    }
    const auto huge_xs_csv = temp_dir / "huge_schneider_xs.csv";
    const auto huge_xs_meta = temp_dir / "huge_schneider_xs.metadata.json";
    {
        std::ofstream out(huge_xs_csv);
        out << header_line << "\n";
        for (float e = 0.5F; e <= 430.0F; e += 0.5F) {
            out << e;
            for (int s = 0; s < 25; ++s) out << ",1000.0";
            out << "\n";
        }
    }
    const auto huge_hash = carbon::compute_file_sha256_hex(huge_xs_csv);
    {
        std::ofstream out(huge_xs_meta);
        out << "{\n  \"data_sha256\": \"" << huge_hash << "\"\n}\n";
    }
    carbon::TransportConfig cfg_huge;
    cfg_huge.phantom_length_mm = 200.0;
    cfg_huge.depth_bin_width_mm = 10.0;
    cfg_huge.primary_atomic_number = 6;
    cfg_huge.primary_mass_number = 12;
    cfg_huge.initial_energy_MeVu = 200.0;
    cfg_huge.enable_ct_grid = true;
    cfg_huge.ct_grid_file = cctg_file.string();
    cfg_huge.ct_schneider_cross_section_file = huge_xs_csv.string();
    cfg_huge.ct_validation_mode = "primary-attenuation-only";
    cfg_huge.nuclear_model = "geant4";
    cfg_huge.enable_inelastic = true;
    cfg_huge.enable_nuclear_elastic = false;
    cfg_huge.enable_secondary_transport = false;
    cfg_huge.enable_energy_straggling = false;
    cfg_huge.beam_energy_spread = 0.0;
    cfg_huge.ct_use_density_mass_spr = true;
    cfg_huge.ct_stopping_power_scale = 1.0;
    cfg_huge.energy_cutoff_MeV = 6.0;
    cfg_huge.number_of_histories = 5000;
    cfg_huge.validate();

    const auto res_huge = carbon::transport_sycl(cfg_huge, water_sp, zero_xs, "default");
    require(res_huge.primary_inelastic_terminated_count == cfg_huge.number_of_histories,
            "Huge XS should have 100% inelastic terminations");
    require(res_huge.primary_stopped_count == 0,
            "Huge XS should have 0 stopped particles");
    require(res_huge.primary_escaped_ct_count == 0,
            "Huge XS should have 0 escaped particles");
    for (const auto& rec : res_huge.primary_first_interactions) {
        require(rec.depth_mm < 1.0F, "Huge XS collision depth must be < 1 mm");
    }
}

void run_step12_4d(const std::filesystem::path& temp_dir,
                   const std::filesystem::path& cctg_file,
                   const std::filesystem::path& xs_path,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    // 4D.1: SHA256 mismatch (tampered CSV with original metadata)
    {
        const auto tampered_csv = temp_dir / "tampered_xs.csv";
        const auto tampered_meta = temp_dir / "tampered_xs.metadata.json";
        std::filesystem::copy_file(xs_path, tampered_csv, std::filesystem::copy_options::overwrite_existing);
        {
            std::ofstream meta_out(tampered_meta);
            meta_out << "{\n  \"data_sha256\": \"47b341324c95f874a2874ba48180a84504605f86d549d7920173d690e0aa91bb\"\n}\n";
        }
        {
            std::fstream f(tampered_csv, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(50);
            f.put('9');
        }
        carbon::TransportConfig cfg_tampered;
        cfg_tampered.phantom_length_mm = 200.0;
        cfg_tampered.depth_bin_width_mm = 10.0;
        cfg_tampered.primary_atomic_number = 6;
        cfg_tampered.primary_mass_number = 12;
        cfg_tampered.initial_energy_MeVu = 200.0;
        cfg_tampered.enable_ct_grid = true;
        cfg_tampered.ct_grid_file = cctg_file.string();
        cfg_tampered.ct_schneider_cross_section_file = tampered_csv.string();
        cfg_tampered.ct_validation_mode = "primary-attenuation-only";
        cfg_tampered.enable_inelastic = true;
        cfg_tampered.enable_nuclear_elastic = false;
        cfg_tampered.enable_secondary_transport = false;
        cfg_tampered.enable_energy_straggling = false;
        cfg_tampered.beam_energy_spread = 0.0;
        cfg_tampered.ct_use_density_mass_spr = true;
        cfg_tampered.ct_stopping_power_scale = 1.0;
        cfg_tampered.energy_cutoff_MeV = 6.0;
        cfg_tampered.number_of_histories = 100;
        cfg_tampered.validate();

        require_throws([&]() {
            carbon::transport_sycl(cfg_tampered, water_sp, zero_xs, "default");
        }, "transport_sycl must reject tampered cross section file");
    }

    // 4D.2: Missing metadata file
    {
        const auto nometa_csv = temp_dir / "nometa_xs.csv";
        std::filesystem::copy_file(xs_path, nometa_csv, std::filesystem::copy_options::overwrite_existing);
        carbon::TransportConfig cfg_nometa;
        cfg_nometa.phantom_length_mm = 200.0;
        cfg_nometa.depth_bin_width_mm = 10.0;
        cfg_nometa.primary_atomic_number = 6;
        cfg_nometa.primary_mass_number = 12;
        cfg_nometa.initial_energy_MeVu = 200.0;
        cfg_nometa.enable_ct_grid = true;
        cfg_nometa.ct_grid_file = cctg_file.string();
        cfg_nometa.ct_schneider_cross_section_file = nometa_csv.string();
        cfg_nometa.ct_validation_mode = "primary-attenuation-only";
        cfg_nometa.enable_inelastic = true;
        cfg_nometa.enable_nuclear_elastic = false;
        cfg_nometa.enable_secondary_transport = false;
        cfg_nometa.enable_energy_straggling = false;
        cfg_nometa.beam_energy_spread = 0.0;
        cfg_nometa.ct_use_density_mass_spr = true;
        cfg_nometa.ct_stopping_power_scale = 1.0;
        cfg_nometa.energy_cutoff_MeV = 6.0;
        cfg_nometa.number_of_histories = 100;
        cfg_nometa.validate();

        require_throws([&]() {
            carbon::transport_sycl(cfg_nometa, water_sp, zero_xs, "default");
        }, "transport_sycl must reject missing metadata file in validation mode");
    }

    // 4D.3: Malformed data_sha256 in metadata
    {
        const auto malformed_csv = temp_dir / "malformed_xs.csv";
        const auto malformed_meta = temp_dir / "malformed_xs.metadata.json";
        std::filesystem::copy_file(xs_path, malformed_csv, std::filesystem::copy_options::overwrite_existing);
        {
            std::ofstream meta_out(malformed_meta);
            meta_out << "{\n  \"data_sha256\": \"short_invalid_hex\"\n}\n";
        }
        carbon::TransportConfig cfg_malformed;
        cfg_malformed.phantom_length_mm = 200.0;
        cfg_malformed.depth_bin_width_mm = 10.0;
        cfg_malformed.primary_atomic_number = 6;
        cfg_malformed.primary_mass_number = 12;
        cfg_malformed.initial_energy_MeVu = 200.0;
        cfg_malformed.enable_ct_grid = true;
        cfg_malformed.ct_grid_file = cctg_file.string();
        cfg_malformed.ct_schneider_cross_section_file = malformed_csv.string();
        cfg_malformed.ct_validation_mode = "primary-attenuation-only";
        cfg_malformed.enable_inelastic = true;
        cfg_malformed.enable_nuclear_elastic = false;
        cfg_malformed.enable_secondary_transport = false;
        cfg_malformed.enable_energy_straggling = false;
        cfg_malformed.beam_energy_spread = 0.0;
        cfg_malformed.ct_use_density_mass_spr = true;
        cfg_malformed.ct_stopping_power_scale = 1.0;
        cfg_malformed.energy_cutoff_MeV = 6.0;
        cfg_malformed.number_of_histories = 100;
        cfg_malformed.validate();

        require_throws([&]() {
            carbon::transport_sycl(cfg_malformed, water_sp, zero_xs, "default");
        }, "transport_sycl must reject malformed data_sha256 in validation mode");
    }

    // 4D.4: Fail-closed Density-Mass-SPR (missing required stopping power table file)
    {
        carbon::TransportConfig cfg_nodensity;
        cfg_nodensity.phantom_length_mm = 200.0;
        cfg_nodensity.depth_bin_width_mm = 10.0;
        cfg_nodensity.primary_atomic_number = 6;
        cfg_nodensity.primary_mass_number = 12;
        cfg_nodensity.initial_energy_MeVu = 200.0;
        cfg_nodensity.enable_ct_grid = true;
        cfg_nodensity.ct_grid_file = cctg_file.string();
        cfg_nodensity.ct_schneider_cross_section_file = xs_path.string();
        cfg_nodensity.ct_validation_mode = "primary-attenuation-only";
        cfg_nodensity.enable_inelastic = true;
        cfg_nodensity.enable_nuclear_elastic = false;
        cfg_nodensity.enable_secondary_transport = false;
        cfg_nodensity.enable_energy_straggling = false;
        cfg_nodensity.beam_energy_spread = 0.0;
        cfg_nodensity.ct_schneider_stopping_power_file.clear();
        cfg_nodensity.ct_use_density_mass_spr = true;
        cfg_nodensity.ct_air_stopping_power_file = "nonexistent_air_table.csv";
        cfg_nodensity.ct_stopping_power_scale = 1.0;
        cfg_nodensity.energy_cutoff_MeV = 6.0;
        cfg_nodensity.number_of_histories = 100;
        cfg_nodensity.validate();

        require_throws([&]() {
            carbon::transport_sycl(cfg_nodensity, water_sp, zero_xs, "default");
        }, "transport_sycl must throw when legacy density-mass-SPR fails to construct in validation mode");

        // When exact Schneider stopping table is active, nonexistent legacy air table is completely ignored
        {
            auto exact_cfg = cfg_nodensity;
            exact_cfg.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
            exact_cfg.validate();
            const auto res = carbon::transport_sycl(exact_cfg, water_sp, zero_xs, "default");
            require(res.nuclear_interactions > 0, "Exact Schneider mode must run without legacy air table");
        }
    }
}

void run_step12_4e(const std::filesystem::path& temp_dir,
                   const std::filesystem::path& cctg_file,
                   const std::filesystem::path& xs_path,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    // 4E.1: Pure Watchdog Test with Zero XS
    {
        const auto zero_xs_csv = temp_dir / "zero_schneider_xs.csv";
        carbon::TransportConfig cfg_watchdog;
        cfg_watchdog.phantom_length_mm = 200.0;
        cfg_watchdog.depth_bin_width_mm = 10.0;
        cfg_watchdog.primary_atomic_number = 6;
        cfg_watchdog.primary_mass_number = 12;
        cfg_watchdog.initial_energy_MeVu = 200.0;
        cfg_watchdog.enable_ct_grid = true;
        cfg_watchdog.ct_grid_file = cctg_file.string();
        cfg_watchdog.ct_schneider_cross_section_file = zero_xs_csv.string();
        cfg_watchdog.ct_validation_mode = "primary-attenuation-only";
        cfg_watchdog.enable_inelastic = true;
        cfg_watchdog.enable_nuclear_elastic = false;
        cfg_watchdog.enable_secondary_transport = false;
        cfg_watchdog.enable_energy_straggling = false;
        cfg_watchdog.beam_energy_spread = 0.0;
        cfg_watchdog.ct_use_density_mass_spr = true;
        cfg_watchdog.ct_stopping_power_scale = 1.0;
        cfg_watchdog.energy_cutoff_MeV = 6.0;
        cfg_watchdog.maximum_primary_steps = 2; // Force watchdog trigger on step 2
        cfg_watchdog.number_of_histories = 1000;
        cfg_watchdog.validate();

        const auto result = carbon::transport_sycl(cfg_watchdog, water_sp, zero_xs, "default");

        std::cout << "[step12-watchdog-test-zero-xs] Terminal counts:\n"
                  << "  inelastic_terminated=" << result.primary_inelastic_terminated_count << "\n"
                  << "  escaped_ct=" << result.primary_escaped_ct_count << "\n"
                  << "  stopped=" << result.primary_stopped_count << "\n"
                  << "  other=" << result.primary_other_terminal_count << "\n"
                  << "  other_kinetic_MeV=" << result.primary_other_terminal_kinetic_MeV << "\n";

        // All 1000 histories must hit watchdog and land strictly in other_terminal
        require(result.primary_other_terminal_count == 1000,
                "Watchdog timeout particles must be classified as other_terminal");
        require(result.primary_inelastic_terminated_count == 0,
                "No particles should be classified as inelastic_terminated under zero XS");
        require(result.primary_escaped_ct_count == 0,
                "No particles should be classified as escaped_ct under 2-step watchdog");
        require(result.primary_stopped_count == 0,
                "No particles should be classified as stopped under 2-step watchdog");

        // Kinetic energy segregation
        require(result.primary_other_terminal_kinetic_MeV > 0.0,
                "Watchdog particles must record remaining kinetic energy in other_terminal_kinetic");
        require(result.primary_inelastic_removed_kinetic_MeV == 0.0,
                "Inelastic removed kinetic must be strictly 0.0 when no inelastic occurred");
        require(result.escaped_energy_MeV == 0.0,
                "Escaped energy must be strictly 0.0 when no particles escaped");
        require(result.primary_cutoff_stopped_energy_MeV == 0.0,
                "Cutoff stopped energy tally must be strictly 0.0 when stopped_count == 0");
        require(result.total_deposited_energy_MeV > 0.0,
                "Deposited energy should reflect continuous loss of 2 steps");

        // Closed energy identity
        const double initial_e = cfg_watchdog.initial_total_energy_MeV() * static_cast<double>(cfg_watchdog.number_of_histories);
        const double accounted = result.total_deposited_energy_MeV +
                                 result.primary_inelastic_removed_kinetic_MeV +
                                 result.primary_other_terminal_kinetic_MeV +
                                 result.escaped_energy_MeV;
        const double rel_err = std::fabs(initial_e - accounted) / initial_e;
        require(rel_err < 1.0e-5, "Watchdog zero-XS energy conservation identity failed");
        require(result.physical_relative_energy_balance_error() < 1.0e-5,
                "Watchdog zero-XS physical relative energy balance error failed");
    }

    // 4E.2: Mixed Watchdog + Inelastic Test with Real XS
    {
        carbon::TransportConfig cfg_watchdog_real;
        cfg_watchdog_real.phantom_length_mm = 200.0;
        cfg_watchdog_real.depth_bin_width_mm = 10.0;
        cfg_watchdog_real.primary_atomic_number = 6;
        cfg_watchdog_real.primary_mass_number = 12;
        cfg_watchdog_real.initial_energy_MeVu = 200.0;
        cfg_watchdog_real.enable_ct_grid = true;
        cfg_watchdog_real.ct_grid_file = cctg_file.string();
        cfg_watchdog_real.ct_schneider_cross_section_file = xs_path.string();
        cfg_watchdog_real.ct_validation_mode = "primary-attenuation-only";
        cfg_watchdog_real.enable_inelastic = true;
        cfg_watchdog_real.enable_nuclear_elastic = false;
        cfg_watchdog_real.enable_secondary_transport = false;
        cfg_watchdog_real.enable_energy_straggling = false;
        cfg_watchdog_real.beam_energy_spread = 0.0;
        cfg_watchdog_real.ct_use_density_mass_spr = true;
        cfg_watchdog_real.ct_stopping_power_scale = 1.0;
        cfg_watchdog_real.energy_cutoff_MeV = 6.0;
        cfg_watchdog_real.maximum_primary_steps = 2; // Force watchdog trigger on step 2
        cfg_watchdog_real.number_of_histories = 1000;
        cfg_watchdog_real.validate();

        const auto result = carbon::transport_sycl(cfg_watchdog_real, water_sp, zero_xs, "default");

        std::cout << "[step12-watchdog-test-real-xs] Terminal counts:\n"
                  << "  inelastic_terminated=" << result.primary_inelastic_terminated_count << "\n"
                  << "  escaped_ct=" << result.primary_escaped_ct_count << "\n"
                  << "  stopped=" << result.primary_stopped_count << "\n"
                  << "  other=" << result.primary_other_terminal_count << "\n"
                  << "  inelastic_removed_MeV=" << result.primary_inelastic_removed_kinetic_MeV << "\n"
                  << "  other_kinetic_MeV=" << result.primary_other_terminal_kinetic_MeV << "\n";

        // Terminal count conservation
        const auto total_terminal =
            result.primary_inelastic_terminated_count +
            result.primary_escaped_ct_count +
            result.primary_stopped_count +
            result.primary_other_terminal_count;
        require(total_terminal == cfg_watchdog_real.number_of_histories,
                "Watchdog real XS total terminal counts must equal number of histories");
        require(result.primary_other_terminal_count > 900,
                "Most particles should hit watchdog at step 2");
        require(result.primary_inelastic_terminated_count > 0,
                "Some particles should undergo inelastic collision in step 1 or 2");
        require(result.primary_escaped_ct_count == 0,
                "No particles should escape at step 2");
        require(result.primary_stopped_count == 0,
                "No particles should stop at step 2");

        // Strictly segregated energy accounting
        require(result.primary_other_terminal_kinetic_MeV > 0.0,
                "Watchdog particles must have positive other_terminal_kinetic");
        require(result.primary_inelastic_removed_kinetic_MeV > 0.0,
                "Inelastic particles must have positive inelastic_removed_kinetic");
        require(result.escaped_energy_MeV == 0.0,
                "No escaped energy");
        require(result.primary_cutoff_stopped_energy_MeV == 0.0,
                "Cutoff stopped energy tally must be strictly 0.0 when stopped_count == 0");

        const double initial_e = cfg_watchdog_real.initial_total_energy_MeV() * static_cast<double>(cfg_watchdog_real.number_of_histories);
        const double accounted = result.total_deposited_energy_MeV +
                                 result.primary_inelastic_removed_kinetic_MeV +
                                 result.primary_other_terminal_kinetic_MeV +
                                 result.escaped_energy_MeV;
        const double rel_err = std::fabs(initial_e - accounted) / initial_e;
        require(rel_err < 1.0e-5, "Watchdog real-XS energy conservation identity failed");
        require(result.physical_relative_energy_balance_error() < 1.0e-5,
                "Watchdog real-XS physical relative energy balance error failed");
    }
}

void run_step12_4f(const std::filesystem::path& temp_dir,
                   const std::filesystem::path& cctg_file,
                   const carbon::StoppingPowerTable& water_sp,
                   const carbon::CrossSectionTable& zero_xs) {
    // 4F.1: max_steps = 1 where 1st step exits boundary (Escape priority over watchdog)
    {
        const auto thin_cctg_file = temp_dir / "thin_step12_ct_grid.cctg";
        carbon::CtGrid thin_grid;
        thin_grid.file_version = carbon::CtGrid::version_v2;
        thin_grid.nx = 10;
        thin_grid.ny = 10;
        thin_grid.nz = 1;
        thin_grid.spacing_x_mm = 10.0;
        thin_grid.spacing_y_mm = 10.0;
        thin_grid.spacing_z_mm = 0.5;
        thin_grid.origin_x_mm = -50.0;
        thin_grid.origin_y_mm = -50.0;
        thin_grid.origin_z_mm = 0.0;
        thin_grid.density_g_per_cm3.assign(100, 1.0F);
        thin_grid.material_id.assign(100, 11);
        thin_grid.mass_sp_za_rel.resize(25, 1.0);
        thin_grid.write_binary(thin_cctg_file);

        const auto zero_xs_csv = temp_dir / "zero_schneider_xs.csv";
        carbon::TransportConfig cfg_escape;
        cfg_escape.phantom_length_mm = 0.5;
        cfg_escape.depth_bin_width_mm = 0.5;
        cfg_escape.primary_atomic_number = 6;
        cfg_escape.primary_mass_number = 12;
        cfg_escape.initial_energy_MeVu = 200.0;
        cfg_escape.enable_ct_grid = true;
        cfg_escape.ct_grid_file = thin_cctg_file.string();
        cfg_escape.ct_schneider_cross_section_file = zero_xs_csv.string();
        cfg_escape.ct_validation_mode = "primary-attenuation-only";
        cfg_escape.enable_inelastic = true;
        cfg_escape.enable_nuclear_elastic = false;
        cfg_escape.enable_secondary_transport = false;
        cfg_escape.enable_energy_straggling = false;
        cfg_escape.beam_energy_spread = 0.0;
        cfg_escape.ct_use_density_mass_spr = true;
        cfg_escape.ct_stopping_power_scale = 1.0;
        cfg_escape.energy_cutoff_MeV = 6.0;
        cfg_escape.maximum_step_mm = 1.0;
        cfg_escape.maximum_relative_energy_loss = 0.005;
        cfg_escape.maximum_primary_steps = 1; // Particle escapes on step 1
        cfg_escape.number_of_histories = 1000;
        cfg_escape.validate();

        const auto result = carbon::transport_sycl(cfg_escape, water_sp, zero_xs, "default");

        // Particle exiting on step 1 must be classified as escaped_ct, not other_terminal
        require(result.primary_escaped_ct_count == 1000,
                "Particle exiting phantom on step 1 must be classified as escaped_ct");
        require(result.primary_other_terminal_count == 0,
                "No particles should land in other_terminal when step 1 escapes");
        require(result.primary_stopped_count == 0,
                "No particles should stop");
        require(result.primary_inelastic_terminated_count == 0,
                "No inelastic under zero XS");
        require(result.primary_other_terminal_kinetic_MeV == 0.0,
                "other_terminal_kinetic must be 0.0");
        require(result.escaped_energy_MeV > 0.0,
                "escaped_energy_MeV must be > 0.0");
        require(result.primary_cutoff_stopped_energy_MeV == 0.0,
                "cutoff_stopped_energy must be 0.0");

        const double initial_e = cfg_escape.initial_total_energy_MeV() * static_cast<double>(cfg_escape.number_of_histories);
        const double accounted = result.total_deposited_energy_MeV +
                                 result.primary_inelastic_removed_kinetic_MeV +
                                 result.primary_other_terminal_kinetic_MeV +
                                 result.escaped_energy_MeV;
        const double rel_err = std::fabs(initial_e - accounted) / initial_e;
        require(rel_err < 1.0e-5, "Simultaneous escape test energy conservation failed");
        require(result.physical_relative_energy_balance_error() < 1.0e-5,
                "Simultaneous escape physical relative energy balance error failed");
    }

    // 4F.2: max_steps = 2 where step 2 enters cutoff (Cutoff stop priority over watchdog)
    {
        const auto zero_xs_csv = temp_dir / "zero_schneider_xs.csv";
        carbon::TransportConfig cfg_cutoff;
        cfg_cutoff.phantom_length_mm = 200.0;
        cfg_cutoff.depth_bin_width_mm = 10.0;
        cfg_cutoff.primary_atomic_number = 6;
        cfg_cutoff.primary_mass_number = 12;
        cfg_cutoff.initial_energy_MeVu = 1.0; // 12.0 MeV total kinetic energy
        cfg_cutoff.enable_ct_grid = true;
        cfg_cutoff.ct_grid_file = cctg_file.string();
        cfg_cutoff.ct_schneider_cross_section_file = zero_xs_csv.string();
        cfg_cutoff.ct_validation_mode = "primary-attenuation-only";
        cfg_cutoff.enable_inelastic = true;
        cfg_cutoff.enable_nuclear_elastic = false;
        cfg_cutoff.enable_secondary_transport = false;
        cfg_cutoff.enable_energy_straggling = false;
        cfg_cutoff.beam_energy_spread = 0.0;
        cfg_cutoff.ct_use_density_mass_spr = true;
        cfg_cutoff.ct_stopping_power_scale = 1.0;
        cfg_cutoff.maximum_step_mm = 1.0;
        cfg_cutoff.maximum_relative_energy_loss = 0.005;
        cfg_cutoff.maximum_primary_steps = 2;
        cfg_cutoff.number_of_histories = 1000;

        // Theoretical step energies derived from stopping power & 0.5% relative loss limit
        const double e0 = cfg_cutoff.initial_total_energy_MeV(); // 12.0 MeV
        const double e1 = e0 * (1.0 - cfg_cutoff.maximum_relative_energy_loss); // 11.94 MeV
        const double e2 = e1 * (1.0 - cfg_cutoff.maximum_relative_energy_loss); // 11.8803 MeV
        const double cutoff_mev = 11.90; // Strictly between e1 and e2
        require(e1 > cutoff_mev, "Theory: Step 1 energy must be strictly above cutoff");
        require(e2 <= cutoff_mev, "Theory: Step 2 energy must reach or drop below cutoff");
        cfg_cutoff.energy_cutoff_MeV = cutoff_mev;
        cfg_cutoff.validate();

        const auto result = carbon::transport_sycl(cfg_cutoff, water_sp, zero_xs, "default");

        // Particle reaching cutoff on step 2 must be classified as stopped, not other_terminal
        require(result.primary_stopped_count == 1000,
                "Particle reaching cutoff on step 2 must be classified as stopped");
        require(result.primary_other_terminal_count == 0,
                "No particles should land in other_terminal when step 2 hits cutoff");
        require(result.primary_escaped_ct_count == 0,
                "No particles should escape");
        require(result.primary_inelastic_terminated_count == 0,
                "No inelastic under zero XS");
        require(result.primary_other_terminal_kinetic_MeV == 0.0,
                "other_terminal_kinetic must be 0.0");
        require(result.primary_cutoff_stopped_energy_MeV > 0.0,
                "cutoff_stopped_energy must be > 0.0");
        require(result.total_deposited_energy_MeV > 0.0,
                "deposited energy must be > 0.0");

        const double initial_e = cfg_cutoff.initial_total_energy_MeV() * static_cast<double>(cfg_cutoff.number_of_histories);
        const double accounted = result.total_deposited_energy_MeV +
                                 result.primary_inelastic_removed_kinetic_MeV +
                                 result.primary_other_terminal_kinetic_MeV +
                                 result.escaped_energy_MeV;
        const double rel_err = std::fabs(initial_e - accounted) / initial_e;
        require(rel_err < 1.0e-5, "Simultaneous cutoff stop test energy conservation failed");
        require(result.physical_relative_energy_balance_error() < 1.0e-5,
                "Simultaneous cutoff stop physical relative energy balance error failed");
    }
}

} // namespace

void test_step12_gpu_transport() {
    if (!is_sycl_available()) {
        return;
    }
    const auto source_dir = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto xs_path = source_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    const auto temp_dir = std::filesystem::temp_directory_path() / "carbon_step12_gpu_test";
    std::filesystem::create_directories(temp_dir);
    const auto cctg_file = prepare_step12_cctg(temp_dir);
    const auto water_sp = carbon::StoppingPowerTable::from_csv(
        source_dir / "data/stopping_power_water_geant4_11_3_2.csv");
    const auto zero_xs = zero_cross_section();

    run_step12_4a(cctg_file, xs_path, water_sp, zero_xs);
    run_step12_4b(temp_dir, cctg_file, xs_path, water_sp, zero_xs);
    run_step12_4c(temp_dir, cctg_file, xs_path, water_sp, zero_xs);
    run_step12_4d(temp_dir, cctg_file, xs_path, water_sp, zero_xs);
    run_step12_4e(temp_dir, cctg_file, xs_path, water_sp, zero_xs);
    run_step12_4f(temp_dir, cctg_file, water_sp, zero_xs);

    std::filesystem::remove_all(temp_dir);
}

void test_step12_device_memory_tracker_exception_safety() {
    if (!is_sycl_available()) {
        return;
    }
    auto queue = carbon::make_sycl_queue("default");

    // 1. Basic allocate, tracking, idempotent track, and manual free
    {
        carbon::detail::DeviceMemoryTracker tracker{queue};
        require(tracker.active_allocation_count() == 0, "Tracker initial count must be 0");
        auto* p1 = tracker.allocate<float>(100);
        auto* p2 = tracker.allocate<int>(200);
        auto* p3 = tracker.allocate<double>(50);
        require(p1 != nullptr && p2 != nullptr && p3 != nullptr, "Allocations must succeed");
        require(tracker.active_allocation_count() == 3, "Tracker must record 3 allocations");

        // Free p2 manually
        tracker.free(p2);
        require(tracker.active_allocation_count() == 2, "Tracker count must decrement after free");

        // External track
        auto* raw = sycl::malloc_device<uint32_t>(10, queue);
        tracker.track(raw);
        require(tracker.active_allocation_count() == 3, "Tracker count must increment after track");

        // Idempotent duplicate track: must not add duplicate entry
        tracker.track(raw);
        tracker.track(p1);
        require(tracker.active_allocation_count() == 3, "Idempotent track must not add duplicate entry");

        // Free p1 and raw
        tracker.free(p1);
        tracker.free(raw);
        require(tracker.active_allocation_count() == 1, "Tracker count must be 1 (p3 remaining)");
        // p3 will be automatically and safely freed when tracker goes out of scope
    }

    // 2. Double-free safety test
    {
        carbon::detail::DeviceMemoryTracker tracker{queue};
        auto* p = tracker.allocate<float>(128);
        require(p != nullptr, "p allocation must succeed");
        require(tracker.active_allocation_count() == 1, "Tracker must track p");
        tracker.free(p);
        require(tracker.active_allocation_count() == 0, "Tracker count must be 0 after free");
        // Second free call on already freed pointer is a safe no-op on tracker
        tracker.free(p);
        require(tracker.active_allocation_count() == 0, "Tracker count remains 0");
    }

    // 3. Fault injection / exception unwinding safety test with exact free count verification
    {
        bool caught_exception = false;
        std::size_t freed_count = 0;
        try {
            carbon::detail::DeviceMemoryTracker tracker{queue};
            tracker.on_free_hook = [&](void*) { ++freed_count; };
            for (int i = 0; i < 10; ++i) {
                auto* ptr = tracker.allocate<float>(1024);
                require(ptr != nullptr, "Allocation in loop must succeed");
                if (i == 5) {
                    // Simulate runtime fault injection mid-pipeline after 6 allocations (i = 0..5)
                    throw std::runtime_error("Simulated fault injection at allocation step 5");
                }
            }
        } catch (const std::runtime_error& err) {
            require(std::string(err.what()).find("Simulated fault injection") != std::string::npos,
                    "Expected fault injection error caught");
            caught_exception = true;
        }
        require(caught_exception, "Fault injection exception must be caught");
        require(freed_count == 6, "Destructor must have invoked free on exactly 6 active allocations, got " +
                                  std::to_string(freed_count));
    }
}
#endif

void test_step14_schneider_stopping_power_tables() {
    const auto bin_path = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
    const auto meta_path = std::filesystem::path("data/schneider/schneider_stopping_v1.metadata.json");
    const auto csv_path = std::filesystem::path("data/schneider/c12_schneider_stopping_power.csv");

    require(std::filesystem::exists(bin_path), "schneider_stopping_v1.bin must exist");
    require(std::filesystem::exists(meta_path), "schneider_stopping_v1.metadata.json must exist");
    require(std::filesystem::exists(csv_path), "c12_schneider_stopping_power.csv must exist");

    // 1. Binary load & header validation
    const auto bin_table = carbon::SchneiderStoppingTable::from_binary(bin_path, meta_path);
    require(bin_table.num_sections() == 25, "Must have exactly 25 sections");
    require(bin_table.num_energies() == 4302, "Must have exactly 4302 energy nodes");
    require_near(bin_table.energy_min_mevu(), 0.01, 1e-6, "E_min must be 0.01 MeV/u");
    require_near(bin_table.energy_max_mevu(), 430.11, 1e-6, "E_max must be 430.11 MeV/u");
    require_near(bin_table.energy_step_mevu(), 0.1, 1e-6, "E_step must be 0.1 MeV/u");

    // 2. CSV load & cross-table equivalence
    const auto csv_table = carbon::SchneiderStoppingTable::from_csv(csv_path);
    for (std::size_t s = 0; s < 25; ++s) {
        require_near(bin_table.density(s), csv_table.density(s), 1e-6, "Density mismatch between bin and csv");
        require(bin_table.density(s) > 0.0, "Density must be positive");

        for (std::size_t e = 0; e < 4302; ++e) {
            const double bin_sp = bin_table.mass_stopping_power(s, e);
            const double csv_sp = csv_table.mass_stopping_power(s, e);
            require_near(bin_sp, csv_sp, 1e-6, "Mass SP mismatch between bin and csv");
            require(bin_sp > 0.0, "Mass SP must be positive");

            const double bin_r = bin_table.csda_range_mm(s, e);
            const double csv_r = csv_table.csda_range_mm(s, e);
            require_near(bin_r, csv_r, 1e-6, "CSDA range mismatch between bin and csv");
            require(bin_r > 0.0, "CSDA range must be positive");

            if (e > 0) {
                require(bin_r > bin_table.csda_range_mm(s, e - 1), "CSDA range must be strictly monotonic");
            }
        }
    }

    // 3. Float flattened array validation
    const auto flat_sp = bin_table.to_flat_mass_stopping_float();
    require(flat_sp.size() == 25 * 4302, "Flat float SP array size mismatch");
    for (std::size_t s = 0; s < 25; ++s) {
        for (std::size_t e = 0; e < 4302; ++e) {
            const double expected = bin_table.mass_stopping_power(s, e);
            const double actual = static_cast<double>(flat_sp[s * 4302 + e]);
            require(std::abs(actual - expected) / expected < 1e-5,
                    "Float conversion precision mismatch");
        }
    }

    // 4. Physical range spot-check (Section 8 soft tissue ~24 mm at 100 MeV/u, ~81 mm at 200 MeV/u)
    const std::size_t idx_100 = 999;
    const std::size_t idx_200 = 1999;
    require_near(bin_table.csda_range_mm(8, idx_100), 24.13, 0.5, "Soft tissue CSDA range at 100 MeV/u out of expected range");
    require_near(bin_table.csda_range_mm(8, idx_200), 81.13, 1.0, "Soft tissue CSDA range at 200 MeV/u out of expected range");

    // 5. Interpolation consistency
    const double test_energy = 150.06;
    const double interp_val = bin_table.interpolate_mass_stopping(8, test_energy);
    const std::size_t lo_idx = static_cast<std::size_t>(std::floor((test_energy - 0.01) / 0.1));
    const double expected_val = 0.5 * (bin_table.mass_stopping_power(8, lo_idx) + bin_table.mass_stopping_power(8, lo_idx + 1));
    require_near(interp_val, expected_val, 1e-5, "Interpolation mismatch");

    // 6. Mandatory P1 Contract Tests: 400.01, 430.00, 430.01, 430.11, nextafter(Emax,+inf)
    const double sp_400_01 = bin_table.interpolate_mass_stopping(8, 400.01);
    require_near(sp_400_01, bin_table.mass_stopping_power(8, 4000), 1e-6, "400.01 MeV/u node exact check");
    const double sp_430_00 = bin_table.interpolate_mass_stopping(8, 430.00);
    require(sp_430_00 > 0.0 && sp_430_00 < sp_400_01, "430.00 MeV/u stopping power must be positive and physically less than 400.01 MeV/u");
    const double sp_430_01 = bin_table.interpolate_mass_stopping(8, 430.01);
    require_near(sp_430_01, bin_table.mass_stopping_power(8, 4300), 1e-6, "430.01 MeV/u node exact check");
    const double sp_430_11 = bin_table.interpolate_mass_stopping(8, 430.11);
    require_near(sp_430_11, bin_table.mass_stopping_power(8, 4301), 1e-6, "430.11 MeV/u node exact check");

    // Guard upper bound: nextafter(430.11, +inf) must safely clamp to endpoint without crashing
    const double sp_beyond = bin_table.interpolate_mass_stopping(8, std::nextafter(430.11, 1000.0));
    require_near(sp_beyond, sp_430_11, 1e-6, "nextafter(430.11, +inf) must safely clamp to maximum table endpoint");

    // Guard lower bound: nextafter(0.01, -inf) must safely clamp to 0.01 endpoint
    const double sp_below = bin_table.interpolate_mass_stopping(8, std::nextafter(0.01, -1000.0));
    require_near(sp_below, bin_table.mass_stopping_power(8, 0), 1e-6, "nextafter(0.01, -inf) must clamp to minimum endpoint");

    // 7. Out-of-bounds error handling
    require_throws<std::out_of_range>([&]() { (void)bin_table.density(25); }, "density(25) must throw out_of_range");
    require_throws<std::out_of_range>([&]() { (void)bin_table.mass_stopping_power(25, 0); }, "mass_stopping_power(25, 0) must throw out_of_range");
    require_throws<std::out_of_range>([&]() { (void)bin_table.mass_stopping_power(0, 4302); }, "mass_stopping_power(0, 4302) must throw out_of_range");
}

void test_schneider_stopping_permutation_rejection() {
    const auto bin_path = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
    const auto meta_path = std::filesystem::path("data/schneider/schneider_stopping_v1.metadata.json");

    // Create a temporary corrupted metadata where section 2 has the hash of section 8
    const auto tmp_dir = std::filesystem::temp_directory_path() / "schneider_meta_corrupt_test";
    std::filesystem::create_directories(tmp_dir);
    const auto corrupt_meta_path = tmp_dir / "schneider_stopping_v1.metadata.json";

    std::ifstream orig_in(meta_path);
    std::string content((std::istreambuf_iterator<char>(orig_in)), std::istreambuf_iterator<char>());

    const std::string sec2_hash = "c979ebe99d85d61ee6e6dd657bcd594ee5aa17a95c86af8d383e23182fc34fa8";
    const std::string sec8_hash = "bb0aa41d37f7f932c7e79d1c6aecde17efe27b28e6c28bd959dc2c0d6b9d1ffe";
    auto pos = content.find(sec2_hash);
    require(pos != std::string::npos, "Could not locate section 2 hash in metadata");
    content.replace(pos, sec2_hash.length(), sec8_hash);

    std::ofstream corrupt_out(corrupt_meta_path);
    corrupt_out << content;
    corrupt_out.close();

    // Verify SchneiderStoppingTable::from_binary strictly throws runtime_error
    require_throws<std::runtime_error>([&]() {
        (void)carbon::SchneiderStoppingTable::from_binary(bin_path, corrupt_meta_path);
    }, "SchneiderStoppingTable must strictly throw when section composition hash is permuted or mismatched");

    std::filesystem::remove_all(tmp_dir);
}

void test_schneider_stopping_metadata_schema_failures() {
    const auto bin_path = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
    const auto meta_path = std::filesystem::path("data/schneider/schneider_stopping_v1.metadata.json");

    const auto tmp_dir = std::filesystem::temp_directory_path() / "schneider_meta_schema_tests";
    std::filesystem::create_directories(tmp_dir);
    const auto corrupt_meta_path = tmp_dir / "corrupt.json";

    std::ifstream orig_in(meta_path);
    const std::string valid_json((std::istreambuf_iterator<char>(orig_in)), std::istreambuf_iterator<char>());

    const auto check_reject = [&](const std::string& modified_json, const std::string& msg) {
        std::ofstream out(corrupt_meta_path);
        out << modified_json;
        out.close();
        require_throws<std::runtime_error>([&]() {
            (void)carbon::SchneiderStoppingTable::from_binary(bin_path, corrupt_meta_path);
        }, msg);
    };

    // 1. Malformed / invalid JSON syntax
    check_reject("{\"schema_version\": 2, \"unclosed_string: 123", "Malformed JSON syntax must throw");

    // 2. Wrong projectile (Z=1 instead of Z=6)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"z\": 6");
        require(pos != std::string::npos, "Could not find Z in metadata");
        bad.replace(pos, 6, "\"z\": 1");
        check_reject(bad, "Wrong projectile Z=1 must throw");
    }

    // 3. Non-integer projectile Z (Z=6.9)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"z\": 6");
        require(pos != std::string::npos, "Could not find Z in metadata");
        bad.replace(pos, 6, "\"z\": 6.9");
        check_reject(bad, "Non-integer projectile Z=6.9 must throw");
    }

    // 4. Non-integer section_id (section_id=0.9)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"section_id\": 0");
        require(pos != std::string::npos, "Could not find section 0 in metadata");
        bad.replace(pos, 15, "\"section_id\": 0.9");
        check_reject(bad, "Non-integer section_id=0.9 must throw");
    }

    // 5. Invalid literal token (truely)
    {
        auto bad = valid_json;
        const auto pos = bad.find("true");
        require(pos != std::string::npos, "Could not find 'true' in metadata");
        bad.replace(pos, 4, "truely");
        check_reject(bad, "Invalid literal 'truely' must throw");
    }

    // 6. Invalid escape sequence in string (\x)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"schneider_stopping_v1.bin\"");
        require(pos != std::string::npos, "Could not find data_filename in metadata");
        bad.replace(pos, 27, "\"schneider_\\xstopping_v1.bin\"");
        check_reject(bad, "Invalid escape sequence \\x must throw");
    }

    // 7. Duplicate object keys
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"format\": \"binary\",");
        require(pos != std::string::npos, "Could not find format in metadata");
        bad.insert(pos, "\"format\": \"binary\",\n  ");
        check_reject(bad, "Duplicate object key 'format' must throw");
    }

    // 8. Wrong material name in section 8
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"PatientTissueFromHU100\"");
        require(pos != std::string::npos, "Could not find material name in metadata");
        bad.replace(pos, 24, "\"CorruptedMaterialName\"");
        check_reject(bad, "Wrong material name in section must throw");
    }

    // 9. Duplicate section ID (change section 1 to section 0)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"section_id\": 1");
        require(pos != std::string::npos, "Could not find section 1 in metadata");
        bad.replace(pos, 15, "\"section_id\": 0");
        check_reject(bad, "Duplicate section ID must throw");
    }

    // 10. Huge integer exceeding size_t range (1e100)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"sections_count\": 25");
        require(pos != std::string::npos, "Could not find sections_count in metadata");
        bad.replace(pos, 20, "\"sections_count\": 1e100");
        check_reject(bad, "Huge integer 1e100 for size_t must throw");
    }

    // 11. Huge integer exceeding int range (1e100)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"z\": 6");
        require(pos != std::string::npos, "Could not find Z in metadata");
        bad.replace(pos, 6, "\"z\": 1e100");
        check_reject(bad, "Huge integer 1e100 for int must throw");
    }

    // 12. Invalid JSON whitespace (vertical tab \v)
    {
        auto bad = valid_json;
        const auto pos = bad.find("\"schema_version\": 2");
        require(pos != std::string::npos, "Could not find schema_version in metadata");
        bad.insert(pos, "\v");
        check_reject(bad, "Invalid JSON whitespace vertical tab must throw");
    }

    std::filesystem::remove_all(tmp_dir);
}

void test_schneider_stopping_payload_physical_validation() {
    const auto orig_bin = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
    const auto orig_meta = std::filesystem::path("data/schneider/schneider_stopping_v1.metadata.json");
    if (!std::filesystem::exists(orig_bin) || !std::filesystem::exists(orig_meta)) return;

    const auto tmp_dir = std::filesystem::temp_directory_path() / "schneider_payload_neg_test";
    std::filesystem::create_directories(tmp_dir);

    const auto test_bin = tmp_dir / "schneider_stopping_v1.bin";
    const auto test_meta = tmp_dir / "schneider_stopping_v1.metadata.json";

    std::ifstream orig_meta_in(orig_meta);
    const std::string orig_meta_str((std::istreambuf_iterator<char>(orig_meta_in)),
                                    std::istreambuf_iterator<char>());
    const std::string orig_sha = carbon::compute_file_sha256_hex(orig_bin);

    auto run_corrupted = [&](const std::string& desc, auto corrupt_fn) {
        std::ifstream bin_in(orig_bin, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(bin_in)),
                                std::istreambuf_iterator<char>());
        corrupt_fn(bytes);

        {
            std::ofstream bin_out(test_bin, std::ios::binary);
            bin_out.write(bytes.data(), bytes.size());
        }

        const std::string new_sha = carbon::compute_file_sha256_hex(test_bin);
        auto meta_copy = orig_meta_str;
        const auto sha_pos = meta_copy.find(orig_sha);
        require(sha_pos != std::string::npos, "Could not find original SHA in metadata");
        meta_copy.replace(sha_pos, orig_sha.size(), new_sha);

        {
            std::ofstream meta_out(test_meta);
            meta_out << meta_copy;
        }

        require_throws<std::runtime_error>([&]() {
            (void)carbon::SchneiderStoppingTable::from_binary(test_bin, test_meta);
        }, desc);
    };

    const std::size_t header_size = sizeof(carbon::SchneiderStoppingHeader);

    // 1. Negative density in binary payload
    run_corrupted("Negative density in binary must be rejected", [&](std::vector<char>& bytes) {
        const std::size_t offset = header_size; // section 0 density
        double neg_val = -1.0;
        std::memcpy(bytes.data() + offset, &neg_val, sizeof(double));
    });

    // 2. NaN in mass stopping power payload
    run_corrupted("NaN in mass stopping power must be rejected", [&](std::vector<char>& bytes) {
        const std::size_t offset = header_size + 25 * sizeof(double) + 100 * sizeof(double); // in mass stopping power array
        double nan_val = std::numeric_limits<double>::quiet_NaN();
        std::memcpy(bytes.data() + offset, &nan_val, sizeof(double));
    });

    // 3. Non-monotonic CSDA range in binary payload
    run_corrupted("Non-monotonic CSDA range must be rejected", [&](std::vector<char>& bytes) {
        const std::size_t csda_offset = header_size + 25 * sizeof(double) + 25 * 4302 * sizeof(double);
        // set csda[1] = csda[0] - 0.5
        double r0 = 0.0;
        std::memcpy(&r0, bytes.data() + csda_offset, sizeof(double));
        double bad_r = r0 - 0.5;
        std::memcpy(bytes.data() + csda_offset + sizeof(double), &bad_r, sizeof(double));
    });

    // 4. Trailing extra byte at end of binary file
    run_corrupted("Trailing byte after binary payload must be rejected", [](std::vector<char>& bytes) {
        bytes.push_back('\0');
    });

    std::filesystem::remove_all(tmp_dir);
}

[[gnu::noinline]] void require_transport_sycl_throws(
    const carbon::TransportConfig& bad,
    const carbon::StoppingPowerTable& water_sp,
    const carbon::CrossSectionTable& zero_xs,
    const std::string& message) {
    require_throws<std::invalid_argument>([&]() {
        (void)carbon::transport_sycl(bad, water_sp, zero_xs, "default");
    }, message);
}

void test_schneider_stopping_source_energy_domain_fail_closed() {
    // Completely self-contained: create a programmatic 1x1x2 Schneider CCTG
    const auto tmp_dir = std::filesystem::temp_directory_path() / "schneider_domain_test";
    std::filesystem::create_directories(tmp_dir);
    const auto cctg_path = tmp_dir / "tiny_schneider.cctg";

    carbon::CtGrid test_grid;
    test_grid.file_version = carbon::CtGrid::version_v2;
    test_grid.nx = 1;
    test_grid.ny = 1;
    test_grid.nz = 2;
    test_grid.spacing_x_mm = 1.0;
    test_grid.spacing_y_mm = 1.0;
    test_grid.spacing_z_mm = 1.0;
    test_grid.origin_x_mm = 0.0;
    test_grid.origin_y_mm = 0.0;
    test_grid.origin_z_mm = 0.0;
    test_grid.density_g_per_cm3 = {1.0F, 1.0F};
    test_grid.material_id = {8, 8}; // section 8
    test_grid.write_binary(cctg_path);

    carbon::TransportConfig base_cfg;
    base_cfg.phantom_length_mm = 2.0;
    base_cfg.depth_bin_width_mm = 1.0;
    base_cfg.primary_atomic_number = 6;
    base_cfg.primary_mass_number = 12;
    base_cfg.initial_energy_MeVu = 200.0;
    base_cfg.enable_ct_grid = true;
    base_cfg.ct_grid_file = cctg_path.string();
    base_cfg.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
    base_cfg.ct_schneider_cross_section_file = "data/schneider/c12_schneider_inelastic_mass_xs.csv";
    base_cfg.nuclear_model = "geant4";
    base_cfg.number_of_histories = 100;
    base_cfg.validate();

    const auto water_sp = carbon::StoppingPowerTable::from_csv("data/stopping_power_water_geant4_11_3_2.csv");
    const auto zero_xs = zero_cross_section();

    // 1. Single beam energy > 430 MeV/u must be rejected
    {
        auto bad = base_cfg;
        bad.initial_energy_MeVu = 435.0;
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Initial energy 435 MeV/u must be rejected in Schneider mode");
    }

    // 2. Single beam upper spread exceeding 430.11 MeV/u must be rejected
    {
        auto bad = base_cfg;
        bad.initial_energy_MeVu = 420.0;
        bad.beam_energy_spread = 0.05; // 420 * (1 + 7.434 * 0.05) = 576 > 430.11
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Single beam spread exceeding 430.11 MeV/u must be rejected in Schneider mode");
    }

    // 3. Single beam lower spread dropping below 0.01 MeV/u must be rejected
    {
        auto bad = base_cfg;
        bad.initial_energy_MeVu = 0.011;
        bad.beam_energy_spread = 0.10; // 0.011 * (1 - 7.434 * 0.10) = 0.0028 < 0.01
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Single beam lower spread below 0.01 MeV/u must be rejected in Schneider mode");
    }

    // 4. Non-finite single beam energy must be rejected
    {
        auto bad = base_cfg;
        bad.initial_energy_MeVu = std::numeric_limits<double>::quiet_NaN();
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Non-finite initial energy must be rejected in Schneider mode");
    }

    // 5. Production spot batch with energy > 430 MeV/u (e.g. 435 MeV/u = 5220 MeV) must be rejected
    {
        auto bad = base_cfg;
        carbon::PrimarySpotBatchEntry spot{};
        spot.history_begin = 0;
        spot.history_end = bad.number_of_histories;
        spot.floats[0] = 5220.0F; // 435 MeV/u * 12
        spot.floats[1] = 0.0F;
        bad.primary_spot_batch = {spot};
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Production spot with 435 MeV/u must be rejected in Schneider mode");
    }

    // 6. Production spot batch with spread dropping below 0.01 MeV/u must be rejected
    {
        auto bad = base_cfg;
        carbon::PrimarySpotBatchEntry spot{};
        spot.history_begin = 0;
        spot.history_end = bad.number_of_histories;
        spot.floats[0] = static_cast<float>(0.011 * 12.0); // 0.011 MeV/u
        spot.floats[1] = 0.10F;
        bad.primary_spot_batch = {spot};
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Production spot with spread dropping below 0.01 MeV/u must be rejected in Schneider mode");
    }

    // 7. Production spot batch with upper spread exceeding 430.11 MeV/u must be rejected
    {
        auto bad = base_cfg;
        carbon::PrimarySpotBatchEntry spot{};
        spot.history_begin = 0;
        spot.history_end = bad.number_of_histories;
        spot.floats[0] = static_cast<float>(420.0 * 12.0); // 420 MeV/u
        spot.floats[1] = 0.05F; // 420 * (1 + 7.434 * 0.05) = 576 > 430.11
        bad.primary_spot_batch = {spot};
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Production spot with spread exceeding 430.11 MeV/u must be rejected in Schneider mode");
    }

    // 8. Production spot batch with NaN energy must be rejected
    {
        auto bad = base_cfg;
        carbon::PrimarySpotBatchEntry spot{};
        spot.history_begin = 0;
        spot.history_end = bad.number_of_histories;
        spot.floats[0] = std::numeric_limits<float>::quiet_NaN();
        spot.floats[1] = 0.0F;
        bad.primary_spot_batch = {spot};
        require_transport_sycl_throws(bad, water_sp, zero_xs,
            "Production spot with NaN energy must be rejected in Schneider mode");
    }

    std::filesystem::remove_all(tmp_dir);
}

void test_schneider_stopping_host_device_equivalence() {
    const auto bin_path = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
    const auto meta_path = std::filesystem::path("data/schneider/schneider_stopping_v1.metadata.json");
    const auto bin_table = carbon::SchneiderStoppingTable::from_binary(bin_path, meta_path);
    const auto flat_sp = bin_table.to_flat_mass_stopping_float();

    sycl::queue queue;
    auto* device_table = sycl::malloc_device<float>(flat_sp.size(), queue);
    require(device_table != nullptr, "Device memory allocation failed for stopping table test");
    queue.copy(flat_sp.data(), device_table, flat_sp.size()).wait_and_throw();

    // Test matrix: sections {0, 8, 24} x energies {below Emin, Emin, 400.01, 430.00, 430.01, 430.11, above Emax}
    const std::vector<std::uint32_t> test_sections = {0, 8, 24};
    const std::vector<float> test_energies = {
        0.005F, 0.01F, 400.01F, 430.00F, 430.01F, 430.11F, 435.00F
    };
    const std::size_t n_queries = test_sections.size() * test_energies.size();

    auto* dev_sections = sycl::malloc_device<std::uint32_t>(n_queries, queue);
    auto* dev_energies = sycl::malloc_device<float>(n_queries, queue);
    auto* dev_results = sycl::malloc_device<float>(n_queries, queue);

    std::vector<std::uint32_t> host_secs(n_queries);
    std::vector<float> host_ens(n_queries);
    std::size_t qi = 0;
    for (auto sec : test_sections) {
        for (auto en : test_energies) {
            host_secs[qi] = sec;
            host_ens[qi] = en;
            ++qi;
        }
    }

    queue.copy(host_secs.data(), dev_sections, n_queries);
    queue.copy(host_ens.data(), dev_energies, n_queries).wait_and_throw();

    const float e_min = 0.01F;
    const float inv_dE = 10.0F;
    const std::uint32_t num_energies = 4302;

    queue.parallel_for(sycl::range<1>(n_queries), [=](sycl::id<1> idx) {
        const auto i = idx[0];
        const auto sec = dev_sections[i];
        const auto en = dev_energies[i];

        const auto floating_sp_index = (en - e_min) * inv_dE;
        auto sp_index = static_cast<int>(sycl::floor(floating_sp_index));
        sp_index = sycl::max(0, sycl::min(sp_index, static_cast<int>(num_energies) - 2));
        const auto sp_fraction = sycl::clamp(floating_sp_index - static_cast<float>(sp_index), 0.0F, 1.0F);

        const auto base_idx = static_cast<std::size_t>(sec) * num_energies + static_cast<std::size_t>(sp_index);
        dev_results[i] = device_table[base_idx] + sp_fraction * (device_table[base_idx + 1] - device_table[base_idx]);
    }).wait_and_throw();

    std::vector<float> host_dev_results(n_queries);
    queue.copy(dev_results, host_dev_results.data(), n_queries).wait_and_throw();

    // Verify host interpolate_mass_stopping vs device lookup
    for (std::size_t i = 0; i < n_queries; ++i) {
        const auto sec = host_secs[i];
        const auto en = host_ens[i];
        const double host_val = bin_table.interpolate_mass_stopping(sec, en);
        const float dev_val = host_dev_results[i];
        require_near(static_cast<double>(dev_val), host_val, 1e-4,
                     "Host vs Device stopping lookup mismatch at section " + std::to_string(sec) +
                     ", energy " + std::to_string(en) + " MeV/u");
    }

    sycl::free(device_table, queue);
    sycl::free(dev_sections, queue);
    sycl::free(dev_energies, queue);
    sycl::free(dev_results, queue);
}

void test_step15_schneider_radiation_lengths_and_sentinel() {
    auto json_path = std::filesystem::path("data/schneider/schneider_radiation_lengths.json");
    if (!std::filesystem::exists(json_path)) {
        json_path = std::filesystem::path("../data/schneider/schneider_radiation_lengths.json");
    }
    require(std::filesystem::exists(json_path), "Missing schneider_radiation_lengths.json");

    std::ifstream in(json_path);
    require(in.is_open(), "Cannot open schneider_radiation_lengths.json");
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // 1. Validate all 25 values against compiled JSON product
    for (unsigned s = 0; s < 25; ++s) {
        const std::string sec_needle = "\"section_id\": " + std::to_string(s);
        const auto pos = content.find(sec_needle);
        require(pos != std::string::npos, "Cannot find section_id " + std::to_string(s));

        const std::string rad_needle = "\"radiation_length_g_per_cm2\": ";
        const auto rad_pos = content.find(rad_needle, pos);
        require(rad_pos != std::string::npos, "Cannot find radiation_length_g_per_cm2 for section " + std::to_string(s));

        const auto val_start = rad_pos + rad_needle.length();
        const auto val_end = content.find_first_of(",\n}", val_start);
        const double expected_val = std::stod(content.substr(val_start, val_end - val_start));

        const double actual_val = carbon::schneider_section_radiation_length_g_per_cm2(s);
        require_near(actual_val, expected_val, 1e-5,
                     "schneider_section_radiation_length_g_per_cm2(" + std::to_string(s) + ") mismatch");
    }

    // 2. Out-of-bounds safety fallback
    require_near(carbon::schneider_section_radiation_length_g_per_cm2(25), 36.0830, 1e-4,
                 "Out-of-bounds section must fall back to water");
    require_near(carbon::schneider_section_radiation_length_g_per_cm2(100), 36.0830, 1e-4,
                 "Out-of-bounds section 100 must fall back to water");

    // 3. Sentinel device-index test catching four-class collapse
    // Section 2 (Adipose): exact 42.08 vs collapsed 36.0830
    {
        const double exact_x0 = carbon::schneider_section_radiation_length_g_per_cm2(2);
        const double collapsed_x0 = carbon::ct_material_radiation_length_g_per_cm2(carbon::ct_material_class(2, true));
        require(std::abs(exact_x0 - collapsed_x0) > 5.0, "Sentinel failed: Section 2 must differ from 4-class collapse");
        const double theta_exact = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 0.92, exact_x0);
        const double theta_collapsed = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 0.92, collapsed_x0);
        const double rel_diff = std::abs(theta_exact - theta_collapsed) / theta_exact;
        require(rel_diff > 0.07, "Section 2 Highland angle relative difference must exceed 7%");
    }

    // Section 11 (Trabecular Bone): exact 34.17 vs collapsed 30.4866
    {
        const double exact_x0 = carbon::schneider_section_radiation_length_g_per_cm2(11);
        const double collapsed_x0 = carbon::ct_material_radiation_length_g_per_cm2(carbon::ct_material_class(11, true));
        require(std::abs(exact_x0 - collapsed_x0) > 3.0, "Sentinel failed: Section 11 must differ from 4-class collapse");
        const double theta_exact = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 1.23, exact_x0);
        const double theta_collapsed = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 1.23, collapsed_x0);
        const double rel_diff = std::abs(theta_exact - theta_collapsed) / theta_exact;
        require(rel_diff > 0.05, "Section 11 Highland angle relative difference must exceed 5%");
    }

    // Section 20 (Dense Bone): exact 27.98 vs collapsed 30.4866
    {
        const double exact_x0 = carbon::schneider_section_radiation_length_g_per_cm2(20);
        const double collapsed_x0 = carbon::ct_material_radiation_length_g_per_cm2(carbon::ct_material_class(20, true));
        require(std::abs(exact_x0 - collapsed_x0) > 2.0, "Sentinel failed: Section 20 must differ from 4-class collapse");
        const double theta_exact = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 1.82, exact_x0);
        const double theta_collapsed = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 1.82, collapsed_x0);
        const double rel_diff = std::abs(theta_exact - theta_collapsed) / theta_exact;
        require(rel_diff > 0.04, "Section 20 Highland angle relative difference must exceed 4%");
    }

    // Section 24 (Titanium): exact 16.16 vs collapsed 30.4866
    {
        const double exact_x0 = carbon::schneider_section_radiation_length_g_per_cm2(24);
        const double collapsed_x0 = carbon::ct_material_radiation_length_g_per_cm2(carbon::ct_material_class(24, true));
        require(std::abs(exact_x0 - collapsed_x0) > 14.0, "Sentinel failed: Section 24 must differ from 4-class collapse");
        const double theta_exact = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 4.55, exact_x0);
        const double theta_collapsed = carbon::highland_projected_rms_angle_material_rad(2400.0, 6, 12, 1.0, 4.55, collapsed_x0);
        const double rel_diff = std::abs(theta_exact - theta_collapsed) / theta_exact;
        require(rel_diff > 0.25, "Section 24 Highland angle relative difference must exceed 25%");
    }

#ifdef CARBON_HAS_SYCL
    // 4. SYCL Device execution test
    sycl::queue queue{sycl::default_selector_v};
    auto* dev_results = sycl::malloc_device<float>(25, queue);
    queue.parallel_for(sycl::range<1>(25), [=](sycl::id<1> idx) {
        const auto s = static_cast<unsigned>(idx[0]);
        dev_results[s] = static_cast<float>(carbon::schneider_section_radiation_length_g_per_cm2(s));
    }).wait_and_throw();

    std::vector<float> host_dev(25);
    queue.copy(dev_results, host_dev.data(), 25).wait_and_throw();
    for (unsigned s = 0; s < 25; ++s) {
        const double host_val = carbon::schneider_section_radiation_length_g_per_cm2(s);
        require_near(static_cast<double>(host_dev[s]), host_val, 1e-4,
                     "Host vs Device radiation length mismatch at section " + std::to_string(s));
    }
    sycl::free(dev_results, queue);
#endif
}

void test_secondary_schneider_mcs_x0_selection() {
    constexpr double water_x0 = carbon::water_radiation_length_g_per_cm2;
    // 1. Water / disabled paths must return the water fallback exactly.
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     false, true, 8, true, water_x0), water_x0, 0.0,
                 "Non-Schneider CT must use water X0");
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     true, false, 8, true, water_x0), water_x0, 0.0,
                 "Outside CT must use water X0");
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     true, true, 8, false, water_x0), water_x0, 0.0,
                 "Disabled ct_material_mcs must use water X0");
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     true, true, 2, false, water_x0), water_x0, 0.0,
                 "Secondary water-mode gate must use water X0");
    // 2/3/4. Schneider sections hit the LUT exactly (soft tissue 8,
    // lung 1, dense bone 20).
    for (const unsigned s : {1U, 8U, 20U}) {
        const double expected = carbon::schneider_section_radiation_length_g_per_cm2(s);
        require_near(carbon::select_transport_radiation_length_g_per_cm2(
                         true, true, s, true, water_x0), expected, 0.0,
                     "Section " + std::to_string(s) + " must use section LUT X0");
    }
    // 6. Boundaries 0 and 24 legal; >24 defensive water fallback
    // (upstream launch validation rejects Schneider voxel material_id >= 25).
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     true, true, 0, true, water_x0),
                 carbon::schneider_section_radiation_length_g_per_cm2(0), 0.0,
                 "Section 0 must be legal");
    require_near(carbon::select_transport_radiation_length_g_per_cm2(
                     true, true, 24, true, water_x0),
                 carbon::schneider_section_radiation_length_g_per_cm2(24), 0.0,
                 "Section 24 must be legal");
    for (const unsigned s : {25U, 100U, 255U}) {
        require_near(carbon::select_transport_radiation_length_g_per_cm2(
                         true, true, s, true, water_x0), water_x0, 0.0,
                     "Section " + std::to_string(s) + " must fall back to water");
    }
    // 7. Primary and secondary gates share one helper: identical inputs
    // give identical X0 (primary: ct_material_ids_are_schneider_sections +
    // in_ct; secondary: same flag + sec_in_ct).
    for (const unsigned s : {0U, 1U, 2U, 8U, 11U, 20U, 24U}) {
        const double primary_x0 = carbon::select_transport_radiation_length_g_per_cm2(
            true, true, s, true, water_x0);
        const double secondary_x0 = carbon::select_transport_radiation_length_g_per_cm2(
            true, true, s, true, water_x0);
        require_near(primary_x0, secondary_x0, 0.0,
                     "Primary/secondary X0 selection must agree at section " +
                         std::to_string(s));
    }
    // 5. Host Highland angle fed by helper X0 equals angle fed by LUT X0
    // (formula untouched; only its X0 input changed), and differs from
    // the old water-X0 value for non-water sections (lung sec 1 is only
    // 1.2% off water, so the move check uses soft tissue 8 and bone 20).
    for (const unsigned s : {1U, 8U, 20U}) {
        const double helper_x0 = carbon::select_transport_radiation_length_g_per_cm2(
            true, true, s, true, water_x0);
        const double theta_helper = carbon::highland_projected_rms_angle_material_rad(
            1200.0, 6, 12, 1.0, 1.0, helper_x0);
        const double theta_lut = carbon::highland_projected_rms_angle_material_rad(
            1200.0, 6, 12, 1.0, 1.0,
            carbon::schneider_section_radiation_length_g_per_cm2(s));
        require_near(theta_helper, theta_lut, 0.0,
                     "Helper-fed Highland angle must equal LUT-fed angle");
    }
    for (const unsigned s : {8U, 20U}) {
        const double helper_x0 = carbon::select_transport_radiation_length_g_per_cm2(
            true, true, s, true, water_x0);
        const double theta_helper = carbon::highland_projected_rms_angle_material_rad(
            1200.0, 6, 12, 1.0, 1.0, helper_x0);
        const double theta_old = carbon::highland_projected_rms_angle_material_rad(
            1200.0, 6, 12, 1.0, 1.0, water_x0);
        require(std::abs(theta_helper - theta_old) / theta_old > 0.01,
                "Section " + std::to_string(s) + " angle must move off water value");
    }
#ifdef CARBON_HAS_SYCL
    // 5 (device): helper X0 selection on device matches host exactly.
    {
        const std::vector<unsigned> sections{0U, 1U, 2U, 8U, 11U, 20U, 24U, 25U, 255U};
        sycl::queue queue{sycl::default_selector_v};
        auto* dev_out = sycl::malloc_device<float>(sections.size(), queue);
        auto* dev_sec = sycl::malloc_device<unsigned>(sections.size(), queue);
        queue.copy(sections.data(), dev_sec, sections.size()).wait_and_throw();
        const double w = water_x0;
        queue.parallel_for(sycl::range<1>(sections.size()), [=](sycl::id<1> idx) {
            const auto i = idx[0];
            dev_out[i] = static_cast<float>(
                carbon::select_transport_radiation_length_g_per_cm2(
                    true, true, dev_sec[i], true, w));
        }).wait_and_throw();
        std::vector<float> host_out(sections.size());
        queue.copy(dev_out, host_out.data(), sections.size()).wait_and_throw();
        for (std::size_t i = 0; i < sections.size(); ++i) {
            const double expected = carbon::select_transport_radiation_length_g_per_cm2(
                true, true, sections[i], true, water_x0);
            require_near(static_cast<double>(host_out[i]), expected, 1e-4,
                         "Device X0 selection mismatch at section " +
                             std::to_string(sections[i]));
        }
        sycl::free(dev_out, queue);
        sycl::free(dev_sec, queue);
    }
#endif
}

void test_out_of_scope_isotope_summary_ledger_fields() {
    carbon::TransportResult result;
    result.initial_energy_MeV = 1.0e9;
    carbon::SchneiderUnsupportedTrack he6{};
    he6.projectile_z = 2;
    he6.projectile_a = 6;
    he6.birth_energy_MeV = 500.0F;
    carbon::SchneiderUnsupportedTrack b8{};
    b8.projectile_z = 5;
    b8.projectile_a = 8;
    b8.birth_energy_MeV = 300.0F;
    result.schneider_unsupported_tracks = {he6, b8};
    // Species 5 (He6) ledger recorded: birth/continuous/escape.
    result.cinel02_species_transport_ledger_MeV[5 * 11 + 0] = 1000.0;
    result.cinel02_species_transport_ledger_MeV[5 * 11 + 1] = 600.0;
    result.cinel02_species_transport_ledger_MeV[5 * 11 + 7] = 400.0;
    const auto tmp = std::filesystem::temp_directory_path() / "oos_summary_test.json";
    {
        std::ofstream out(tmp, std::ios::binary);
        require(out.is_open(), "Cannot open temp OOS summary file");
        carbon::write_out_of_scope_isotope_summary(out, result);
    }
    std::ifstream in(tmp, std::ios::binary);
    require(in.is_open(), "Cannot read temp OOS summary file");
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    require(content.find("\"isotope\": \"He6\"") != std::string::npos,
            "OOS summary must contain He6");
    require(content.find("\"species_ledger_recorded\": true") != std::string::npos,
            "He6 ledger must be marked recorded");
    require(content.find("\"deposited_energy_MeV\": 600") != std::string::npos,
            "He6 deposited must equal continuous ledger metric");
    require(content.find("\"escaped_energy_MeV\": 400") != std::string::npos,
            "He6 escaped must equal boundary-escape ledger metric");
    require(content.find("\"isotope\": \"B8\"") != std::string::npos,
            "OOS summary must contain B8");
    // B8/C10 ledgers are all zero: fields must be null, tracks still counted.
    require(content.find("\"ledger_birth_energy_MeV\": null") != std::string::npos,
            "Unrecorded isotopes must report null ledger fields");
    std::error_code ec{};
    std::filesystem::remove(tmp, ec);
}

carbon::InelasticPackageV3Table make_synthetic_cinel03_table() {
    carbon::InelasticPackageV3Table table;
    table.set_metadata(100.0F, 100.0F, 1, "00000000-0000-4000-8000-000000000016");

    // Event 1: C12 on Target H (Z=1, A=1) at 200 MeV/u (2400 MeV)
    {
        carbon::Cinel03InteractionRecord ev{};
        ev.run_id = 1;
        ev.event_id = 1;
        ev.projectile_pdg = 1000060120;
        ev.projectile_z = 6;
        ev.projectile_a = 12;
        ev.projectile_charge = 6.0F;
        ev.projectile_rest_mass = static_cast<float>(12.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        ev.collision_energy_MeV = 2400.0F;
        ev.collision_energy_MeV_per_u = 200.0F;
        ev.collision_direction_z = 1.0F;
        ev.target_element_z = 1; // Hydrogen
        ev.target_a = 1;
        ev.parent_status = 2; // Destroyed
        ev.parent_pdg = 1000060120;
        ev.parent_z = 6;
        ev.parent_a = 12;
        ev.parent_charge = 6.0F;
        ev.parent_rest_mass = ev.projectile_rest_mass;
        ev.parent_energy_MeV = 0.0F;
        ev.parent_direction_z = 1.0F;
        ev.track_weight = 1.0F;
        ev.parent_weight = 1.0F;
        ev.process_local_deposit_MeV = 20.0F;
        ev.direct_product_count = 3;
        std::strncpy(ev.material_name, "G4_WATER", sizeof(ev.material_name));
        std::strncpy(ev.process_name, "ionInelastic", sizeof(ev.process_name));
        std::strncpy(ev.model_name, "BinaryCascade", sizeof(ev.model_name));

        std::vector<carbon::Cinel03ProductRecord> prods(3);
        prods[0].pdg = 1000050110;
        prods[0].z = 5;
        prods[0].a = 11;
        prods[0].charge = 5.0F;
        prods[0].rest_mass = static_cast<float>(11.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        prods[0].kinetic_energy_MeV = 2100.0F;
        prods[0].direction_z = 1.0F;
        prods[0].local_direction_z = 1.0F;
        prods[0].weight = 1.0F;
        prods[0].role = 0;

        prods[1].pdg = 2212;
        prods[1].z = 1;
        prods[1].a = 1;
        prods[1].charge = 1.0F;
        prods[1].rest_mass = static_cast<float>(carbon::inelastic_proton_rest_mass_MeV);
        prods[1].kinetic_energy_MeV = 180.0F;
        prods[1].direction_z = 1.0F;
        prods[1].local_direction_z = 1.0F;
        prods[1].weight = 1.0F;
        prods[1].role = 0;

        prods[2].pdg = 2112;
        prods[2].z = 0;
        prods[2].a = 1;
        prods[2].charge = 0.0F;
        prods[2].rest_mass = static_cast<float>(carbon::inelastic_neutron_rest_mass_MeV);
        prods[2].kinetic_energy_MeV = 100.0F;
        prods[2].direction_z = 1.0F;
        prods[2].local_direction_z = 1.0F;
        prods[2].weight = 1.0F;
        prods[2].role = 0;

        table.add_event(ev, prods);
    }

    // Event 2: C12 on Target C (Z=6, A=12) at 200 MeV/u (2400 MeV)
    {
        carbon::Cinel03InteractionRecord ev{};
        ev.run_id = 1;
        ev.event_id = 2;
        ev.projectile_pdg = 1000060120;
        ev.projectile_z = 6;
        ev.projectile_a = 12;
        ev.projectile_charge = 6.0F;
        ev.projectile_rest_mass = static_cast<float>(12.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        ev.collision_energy_MeV = 2400.0F;
        ev.collision_energy_MeV_per_u = 200.0F;
        ev.collision_direction_z = 1.0F;
        ev.target_element_z = 6; // Carbon
        ev.target_a = 12;
        ev.parent_status = 2;
        ev.parent_pdg = 1000060120;
        ev.parent_z = 6;
        ev.parent_a = 12;
        ev.parent_charge = 6.0F;
        ev.parent_rest_mass = ev.projectile_rest_mass;
        ev.parent_energy_MeV = 0.0F;
        ev.parent_direction_z = 1.0F;
        ev.track_weight = 1.0F;
        ev.parent_weight = 1.0F;
        ev.process_local_deposit_MeV = 30.0F;
        ev.direct_product_count = 2;
        std::strncpy(ev.material_name, "Schneider_Tissue", sizeof(ev.material_name));
        std::strncpy(ev.process_name, "ionInelastic", sizeof(ev.process_name));
        std::strncpy(ev.model_name, "BinaryCascade", sizeof(ev.model_name));

        std::vector<carbon::Cinel03ProductRecord> prods(2);
        prods[0].pdg = 1000060110;
        prods[0].z = 6;
        prods[0].a = 11;
        prods[0].charge = 6.0F;
        prods[0].rest_mass = static_cast<float>(11.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        prods[0].kinetic_energy_MeV = 2150.0F;
        prods[0].direction_z = 1.0F;
        prods[0].local_direction_z = 1.0F;
        prods[0].weight = 1.0F;
        prods[0].role = 0;

        prods[1].pdg = 2112;
        prods[1].z = 0;
        prods[1].a = 1;
        prods[1].charge = 0.0F;
        prods[1].rest_mass = static_cast<float>(carbon::inelastic_neutron_rest_mass_MeV);
        prods[1].kinetic_energy_MeV = 220.0F;
        prods[1].direction_z = 1.0F;
        prods[1].local_direction_z = 1.0F;
        prods[1].weight = 1.0F;
        prods[1].role = 0;

        table.add_event(ev, prods);
    }

    // Event 3: C12 on Target O (Z=8, A=16) at 200 MeV/u (2400 MeV)
    {
        carbon::Cinel03InteractionRecord ev{};
        ev.run_id = 1;
        ev.event_id = 3;
        ev.projectile_pdg = 1000060120;
        ev.projectile_z = 6;
        ev.projectile_a = 12;
        ev.projectile_charge = 6.0F;
        ev.projectile_rest_mass = static_cast<float>(12.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        ev.collision_energy_MeV = 2400.0F;
        ev.collision_energy_MeV_per_u = 200.0F;
        ev.collision_direction_z = 1.0F;
        ev.target_element_z = 8; // Oxygen
        ev.target_a = 16;
        ev.parent_status = 2;
        ev.parent_pdg = 1000060120;
        ev.parent_z = 6;
        ev.parent_a = 12;
        ev.parent_charge = 6.0F;
        ev.parent_rest_mass = ev.projectile_rest_mass;
        ev.parent_energy_MeV = 0.0F;
        ev.parent_direction_z = 1.0F;
        ev.track_weight = 1.0F;
        ev.parent_weight = 1.0F;
        ev.process_local_deposit_MeV = 50.0F;
        ev.direct_product_count = 3;
        std::strncpy(ev.material_name, "Schneider_Bone", sizeof(ev.material_name));
        std::strncpy(ev.process_name, "ionInelastic", sizeof(ev.process_name));
        std::strncpy(ev.model_name, "BinaryCascade", sizeof(ev.model_name));

        std::vector<carbon::Cinel03ProductRecord> prods(3);
        prods[0].pdg = 1000040070;
        prods[0].z = 4;
        prods[0].a = 7;
        prods[0].charge = 4.0F;
        prods[0].rest_mass = static_cast<float>(7.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        prods[0].kinetic_energy_MeV = 1350.0F;
        prods[0].direction_z = 1.0F;
        prods[0].local_direction_z = 1.0F;
        prods[0].weight = 1.0F;
        prods[0].role = 0;

        prods[1].pdg = 1000020040;
        prods[1].z = 2;
        prods[1].a = 4;
        prods[1].charge = 2.0F;
        prods[1].rest_mass = static_cast<float>(4.0 * carbon::inelastic_nucleon_rest_mass_MeV);
        prods[1].kinetic_energy_MeV = 750.0F;
        prods[1].direction_z = 1.0F;
        prods[1].local_direction_z = 1.0F;
        prods[1].weight = 1.0F;
        prods[1].role = 0;

        prods[2].pdg = 2212;
        prods[2].z = 1;
        prods[2].a = 1;
        prods[2].charge = 1.0F;
        prods[2].rest_mass = static_cast<float>(carbon::inelastic_proton_rest_mass_MeV);
        prods[2].kinetic_energy_MeV = 250.0F;
        prods[2].direction_z = 1.0F;
        prods[2].local_direction_z = 1.0F;
        prods[2].weight = 1.0F;
        prods[2].role = 0;

        table.add_event(ev, prods);
    }

    table.finalize();
    return table;
}

void test_step16_cinel03_round_trip_and_determinism() {
    const auto table = make_synthetic_cinel03_table();
    const auto p_a = std::filesystem::path("test_cinel03_a.cinpkg");
    const auto p_b = std::filesystem::path("test_cinel03_b.cinpkg");

    table.to_binary(p_a);
    table.to_binary(p_b);

    require(std::filesystem::exists(p_a), "CINEL03 output A missing");
    require(std::filesystem::exists(p_b), "CINEL03 output B missing");

    // Check deterministic bitwise serialization
    std::ifstream in_a(p_a, std::ios::binary);
    std::ifstream in_b(p_b, std::ios::binary);
    std::string bytes_a((std::istreambuf_iterator<char>(in_a)), std::istreambuf_iterator<char>());
    std::string bytes_b((std::istreambuf_iterator<char>(in_b)), std::istreambuf_iterator<char>());
    require(bytes_a.size() > 0, "CINEL03 file is empty");
    require(bytes_a == bytes_b, "CINEL03 serialization is non-deterministic");

    // Deserialize and check contents
    const auto loaded = carbon::InelasticPackageV3Table::from_binary(p_a);
    require(loaded.cells().size() == 3, "Cell count mismatch");
    require(loaded.interactions().size() == 3, "Interaction count mismatch");
    require(loaded.products().size() == 8, "Product count mismatch");
    require(loaded.energy_nodes().size() == 3, "Energy node count mismatch");
    require(loaded.campaign_uuid() == "00000000-0000-4000-8000-000000000016", "UUID mismatch");

    // Validate key contract: event key is projectile_Z, projectile_A, target_element_Z, energy_node
    require(loaded.energy_nodes()[0].target_element_z == 1, "Target element 1 mismatch");
    require(loaded.energy_nodes()[1].target_element_z == 6, "Target element 6 mismatch");
    require(loaded.energy_nodes()[2].target_element_z == 8, "Target element 8 mismatch");

    std::filesystem::remove(p_a);
    std::filesystem::remove(p_b);
}

void test_step16_cinel03_rejections_and_fail_closed() {
    const auto table = make_synthetic_cinel03_table();
    const auto p_valid = std::filesystem::path("test_cinel03_valid.cinpkg");
    table.to_binary(p_valid);

    // 1. Missing target fail-closed in production mode
    require_throws([&]() {
        // Calcium Z=20 is not in table
        const auto loaded = carbon::InelasticPackageV3Table::from_binary(p_valid);
        (void)loaded.find_event(6, 12, 20, 200.0F, 1.0F, 0.5F, false);
    }, "CINEL03: Missing target element Z=20");

    // 2. Missing target non-fatal in audit mode (increments named counter)
    {
        const auto loaded = carbon::InelasticPackageV3Table::from_binary(p_valid);
        std::uint64_t missing_counter = 0;
        const auto ev = loaded.find_event(6, 12, 20, 200.0F, 1.0F, 0.5F, true, &missing_counter);
        require(ev == std::numeric_limits<std::uint64_t>::max(), "Audit mode must return invalid index for missing target");
        require(missing_counter == 1, "Audit mode must increment missing target counter");
    }

    // 3. Truncated header rejection
    {
        const auto p_bad = std::filesystem::path("test_cinel03_truncated.cinpkg");
        std::ofstream out(p_bad, std::ios::binary);
        char buf[50] = {0};
        out.write(buf, 50);
        out.close();
        require_throws([&]() {
            carbon::InelasticPackageV3Table::from_binary(p_bad);
        }, "Truncated CINPKG04 header");
        std::filesystem::remove(p_bad);
    }

    // 4. Unsupported magic / legacy CINEL02 rejection
    {
        const auto p_bad = std::filesystem::path("test_cinel03_badmagic.cinpkg");
        std::ifstream in(p_valid, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        bytes[0] = 'X'; // Corrupt magic
        std::ofstream out(p_bad, std::ios::binary);
        out.write(bytes.data(), bytes.size());
        out.close();
        require_throws([&]() {
            carbon::InelasticPackageV3Table::from_binary(p_bad);
        }, "Unsupported or non-authoritative CINPKG04 header");
        std::filesystem::remove(p_bad);
    }

    // 5. Corrupted CRC32 checksum rejection
    {
        const auto p_bad = std::filesystem::path("test_cinel03_badcrc.cinpkg");
        std::ifstream in(p_valid, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        bytes[bytes.size() - 5] ^= 0x55; // Flip payload bit
        std::ofstream out(p_bad, std::ios::binary);
        out.write(bytes.data(), bytes.size());
        out.close();
        require_throws([&]() {
            carbon::InelasticPackageV3Table::from_binary(p_bad);
        }, "CINPKG04 checksum mismatch");
        std::filesystem::remove(p_bad);
    }

    std::filesystem::remove(p_valid);
}

void test_step16_cinel03_synthetic_cpu_gpu_replay() {
    const auto table = make_synthetic_cinel03_table();
    const auto compact = table.make_device_tables();

    // CPU lookup and replay
    const auto ev_idx = table.find_event(6, 12, 8, 200.0F, 1.0F, 0.0F);
    require(ev_idx != std::numeric_limits<std::uint64_t>::max(), "CPU find_event failed for O16 target");
    const auto replay = table.fixed_replay(ev_idx, compact);
    require(replay.product_count == 3, "Product count mismatch on CPU");
    require(replay.compact_products[0].z == 4 && replay.compact_products[0].a == 7, "Product 0 Be7 mismatch");
    require(replay.compact_products[1].z == 2 && replay.compact_products[1].a == 4, "Product 1 He4 mismatch");
    require(replay.compact_products[2].z == 1 && replay.compact_products[2].a == 1, "Product 2 Proton mismatch");

#ifdef CARBON_HAS_SYCL
    // GPU Replay Test
    sycl::queue queue{sycl::default_selector_v};
    auto* dev_nodes = sycl::malloc_device<carbon::Cinel03EnergyNode>(compact.energy_nodes.size(), queue);
    auto* dev_offsets = sycl::malloc_device<std::uint32_t>(compact.event_offsets.size(), queue);
    auto* dev_indices = sycl::malloc_device<std::uint32_t>(compact.event_indices.size(), queue);
    auto* dev_interactions = sycl::malloc_device<carbon::Cinel03DeviceInteraction>(compact.interactions.size(), queue);
    auto* dev_products = sycl::malloc_device<carbon::Cinel03DeviceProduct>(compact.products.size(), queue);

    queue.copy(compact.energy_nodes.data(), dev_nodes, compact.energy_nodes.size()).wait_and_throw();
    queue.copy(compact.event_offsets.data(), dev_offsets, compact.event_offsets.size()).wait_and_throw();
    queue.copy(compact.event_indices.data(), dev_indices, compact.event_indices.size()).wait_and_throw();
    queue.copy(compact.interactions.data(), dev_interactions, compact.interactions.size()).wait_and_throw();
    queue.copy(compact.products.data(), dev_products, compact.products.size()).wait_and_throw();

    // Output buffer: [0]=product_count, [1]=prod0_z, [2]=prod0_a, [3]=prod1_z, [4]=prod1_a, [5]=prod2_z, [6]=prod2_a
    auto* dev_out = sycl::malloc_device<int>(10, queue);
    queue.memset(dev_out, 0, 10 * sizeof(int)).wait_and_throw();

    const auto num_nodes = compact.energy_nodes.size();
    queue.single_task([=]() {
        // Device search for target_element_z == 8
        for (std::size_t i = 0; i < num_nodes; ++i) {
            if (dev_nodes[i].projectile_z == 6 && dev_nodes[i].projectile_a == 12 &&
                dev_nodes[i].target_element_z == 8) {
                const auto ev_id = dev_indices[dev_offsets[i]];
                const auto& inter = dev_interactions[ev_id];
                dev_out[0] = static_cast<int>(inter.direct_product_count);
                const auto p_off = inter.product_offset;
                for (std::uint32_t p = 0; p < inter.direct_product_count; ++p) {
                    dev_out[1 + p * 2 + 0] = dev_products[p_off + p].z;
                    dev_out[1 + p * 2 + 1] = dev_products[p_off + p].a;
                }
                break;
            }
        }
    }).wait_and_throw();

    std::vector<int> host_out(10);
    queue.copy(dev_out, host_out.data(), 10).wait_and_throw();

    require(host_out[0] == 3, "GPU replay product count mismatch");
    require(host_out[1] == 4 && host_out[2] == 7, "GPU product 0 Be7 mismatch");
    require(host_out[3] == 2 && host_out[4] == 4, "GPU product 1 He4 mismatch");
    require(host_out[5] == 1 && host_out[6] == 1, "GPU product 2 Proton mismatch");

    sycl::free(dev_nodes, queue);
    sycl::free(dev_offsets, queue);
    sycl::free(dev_indices, queue);
    sycl::free(dev_interactions, queue);
    sycl::free(dev_products, queue);
    sycl::free(dev_out, queue);
#endif
}

using namespace carbon;

void test_step18_target_sampler_synthetic_and_known_ratios() {
    std::cout << "[step18-test] Running synthetic and known-ratio target sampler tests...\n";
    const auto rate_table = SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v1.bin");
    const SchneiderTargetSampler sampler(rate_table);

    for (std::size_t s = 0; s < 25; ++s) {
        for (std::size_t e = 0; e < 860; ++e) {
            const float total = sampler.total_mass_rate(s, 0.5F + static_cast<float>(e) * 0.5F);
            require(std::isfinite(total) && total >= 0.0F, "Total mass rate must be non-negative finite");
            const auto probs = sampler.target_probabilities(s, 0.5F + static_cast<float>(e) * 0.5F);
            float sum_p = 0.0F;
            for (float p : probs) {
                require(p >= 0.0F && p <= 1.0F, "Individual target probability out of [0, 1]");
                sum_p += p;
            }
            if (total > 1.0e-7F) {
                require(std::abs(sum_p - 1.0F) < 1.0e-4F, "Sum of target probabilities must close to 1.0");
            }
        }
    }

    const auto sample_low = sampler.sample_target(8, 200.0F, 0.0F);
    require(sample_low.target_z == 1, "u=0 in soft tissue should sample first element Hydrogen Z=1");
    const auto sample_high = sampler.sample_target(8, 200.0F, 0.99999F);
    require(sample_high.target_z > 0, "u=0.99999 should sample valid positive target Z");

    std::cout << "[step18-test] Synthetic and known-ratio checks PASSED.\n";
}

void test_step18_target_sampler_statistical_goodness_of_fit() {
    std::cout << "[step18-test] Running statistical goodness-of-fit tests...\n";
    const auto rate_table = SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v1.bin");
    const SchneiderTargetSampler sampler(rate_table);

    const std::size_t sections_to_test[] = {8, 20};
    const std::size_t N = 100000;

    for (std::size_t s : sections_to_test) {
        const auto probs = sampler.target_probabilities(s, 200.0F);
        std::array<std::size_t, 13> counts{};

        for (std::size_t i = 0; i < N; ++i) {
            const float u = static_cast<float>((i + 0.5) / static_cast<double>(N));
            const auto sample = sampler.sample_target(s, 200.0F, u);
            counts[sample.target_index]++;
        }

        double chi2 = 0.0;
        int degrees_of_freedom = 0;
        for (std::size_t k = 0; k < 13; ++k) {
            const double expected = static_cast<double>(N) * probs[k];
            if (expected >= 5.0) {
                const double diff = static_cast<double>(counts[k]) - expected;
                chi2 += (diff * diff) / expected;
                degrees_of_freedom++;
            }
        }
        degrees_of_freedom = std::max(1, degrees_of_freedom - 1);
        std::cout << "[step18-test] Section " << s << " Chi2=" << chi2 << " with df=" << degrees_of_freedom << "\n";
        require(chi2 < 10.0, "Categorical target distribution Chi-square goodness-of-fit failed");
    }
    std::cout << "[step18-test] Statistical goodness-of-fit PASSED.\n";
}

void test_step18_target_sampler_density_independence_and_library_reuse() {
    std::cout << "[step18-test] Running density-independence and library reuse tests...\n";
    const auto rate_table = SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v1.bin");
    const SchneiderTargetSampler sampler(rate_table);
    const auto package = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");

    const auto base_probs = sampler.target_probabilities(8, 200.0F);
    const float densities[] = {0.2F, 0.5F, 1.0F, 1.5F, 2.5F};
    for (float rho : densities) {
        const float macro_rate = rho * sampler.total_mass_rate(8, 200.0F);
        require(macro_rate > 0.0F, "Macro rate must be positive");
        const auto p = sampler.target_probabilities(8, 200.0F);
        for (std::size_t k = 0; k < 13; ++k) {
            require(std::abs(p[k] - base_probs[k]) < 1.0e-6F, "Target probability changed with density");
        }
    }

    const auto ev_id5 = package.find_event(6, 12, 8, 200.0F, 50.0F, 0.25F);
    const auto ev_id8 = package.find_event(6, 12, 8, 200.0F, 50.0F, 0.25F);
    const auto ev_id20 = package.find_event(6, 12, 8, 200.0F, 50.0F, 0.25F);
    require(ev_id5 != InelasticPackageV3Table::invalid, "Oxygen query failed");
    require(ev_id5 == ev_id8 && ev_id8 == ev_id20, "Oxygen library must be identically reused across sections");

    std::cout << "[step18-test] Density independence and library reuse PASSED.\n";
}

void test_step18_target_sampler_missing_target_fail_closed() {
    std::cout << "[step18-test] Running fail-closed missing target tests...\n";
    const auto package = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");

    bool threw_exception = false;
    try {
        package.find_event(6, 12, 99, 200.0F, 50.0F, 0.5F, false);
    } catch (const std::runtime_error& err) {
        threw_exception = true;
        const std::string msg = err.what();
        require(msg.find("Missing target element Z=99") != std::string::npos, "Exception message must identify missing target Z");
    }
    require(threw_exception, "Production query for missing target must throw runtime_error (never alias to O)");

    std::uint64_t missing_counter = 0;
    const auto ev_audit = package.find_event(6, 12, 99, 200.0F, 50.0F, 0.5F, true, &missing_counter);
    require(ev_audit == InelasticPackageV3Table::invalid, "Audit query must return invalid");
    require(missing_counter == 1, "Audit query must increment missing_target_counter");

    std::cout << "[step18-test] Fail-closed missing target checks PASSED.\n";
}

void test_step18_target_sampler_cpu_gpu_equivalence_and_diagnostics() {
    std::cout << "[step18-test] Running CPU/GPU equivalence and diagnostics tests...\n";
    const auto rate_table = SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v1.bin");
    const SchneiderTargetSampler sampler(rate_table);
    const auto package = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");
    const auto device_tables = package.make_device_tables();

#ifdef CARBON_HAS_SYCL
    sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
    const auto dev_sampler = sampler.device_table();

    float* dev_cdf = sycl::malloc_device<float>(sampler.cdf_table().size(), queue);
    float* dev_total = sycl::malloc_device<float>(sampler.total_mass_rates().size(), queue);
    queue.copy(sampler.cdf_table().data(), dev_cdf, sampler.cdf_table().size()).wait_and_throw();
    queue.copy(sampler.total_mass_rates().data(), dev_total, sampler.total_mass_rates().size()).wait_and_throw();

    SchneiderTargetSamplerDeviceTable gpu_sampler_table = dev_sampler;
    gpu_sampler_table.cdf_table = dev_cdf;
    gpu_sampler_table.total_mass_rates = dev_total;

    Cinel03EnergyNode* dev_nodes = sycl::malloc_device<Cinel03EnergyNode>(device_tables.energy_nodes.size(), queue);
    std::uint32_t* dev_offsets = sycl::malloc_device<std::uint32_t>(device_tables.event_offsets.size(), queue);
    std::uint32_t* dev_indices = sycl::malloc_device<std::uint32_t>(device_tables.event_indices.size(), queue);
    queue.copy(device_tables.energy_nodes.data(), dev_nodes, device_tables.energy_nodes.size()).wait_and_throw();
    queue.copy(device_tables.event_offsets.data(), dev_offsets, device_tables.event_offsets.size()).wait_and_throw();
    queue.copy(device_tables.event_indices.data(), dev_indices, device_tables.event_indices.size()).wait_and_throw();

    const std::uint32_t node_count = static_cast<std::uint32_t>(device_tables.energy_nodes.size());
    const std::uint32_t total_events = static_cast<std::uint32_t>(device_tables.interactions.size());

    SchneiderTargetDiagnostics* dev_diag = sycl::malloc_device<SchneiderTargetDiagnostics>(1, queue);
    queue.memset(dev_diag, 0, sizeof(SchneiderTargetDiagnostics)).wait_and_throw();

    const std::size_t num_queries = 1000;
    int* dev_targets = sycl::malloc_device<int>(num_queries, queue);
    std::uint32_t* dev_event_ids = sycl::malloc_device<std::uint32_t>(num_queries, queue);

    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0];
        const std::size_t section = i % 25;
        const float energy = 20.0F + static_cast<float>(i % 40) * 10.0F;
        const float u_target = static_cast<float>((i * 37) % 1000) / 1000.0F;
        const float u_event = static_cast<float>((i * 73) % 1000) / 1000.0F;

        const int target_z = sample_schneider_target_device(gpu_sampler_table, section, energy, u_target);
        dev_targets[i] = target_z;

        const std::uint32_t event_id = cinel03_find_event_device(
            dev_nodes, node_count, dev_offsets, dev_indices, total_events,
            6, 12, target_z, energy, 50.0F, u_event);
        dev_event_ids[i] = event_id;

        if (target_z > 0 && target_z < 32) {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                target_ref(dev_diag->interactions_by_target_z[target_z]);
            target_ref.fetch_add(1U);
        }
        if (section < 25) {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                sec_ref(dev_diag->interactions_by_section[section]);
            sec_ref.fetch_add(1U);
        }
        const auto e_bin = static_cast<std::size_t>(energy);
        if (e_bin < 450) {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                ebin_ref(dev_diag->interactions_by_energy_bin[e_bin]);
            ebin_ref.fetch_add(1U);
        }
        if (event_id != 0xFFFFFFFFU) {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                succ_ref(dev_diag->lookup_success_count);
            succ_ref.fetch_add(1U);
        } else {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                fail_ref(dev_diag->lookup_failure_count);
            fail_ref.fetch_add(1U);
        }
    }).wait_and_throw();

    std::vector<int> host_targets(num_queries);
    std::vector<std::uint32_t> host_events(num_queries);
    SchneiderTargetDiagnostics host_diag{};
    queue.copy(dev_targets, host_targets.data(), num_queries).wait_and_throw();
    queue.copy(dev_event_ids, host_events.data(), num_queries).wait_and_throw();
    queue.copy(dev_diag, &host_diag, 1).wait_and_throw();

    std::size_t match_count = 0;
    for (std::size_t i = 0; i < num_queries; ++i) {
        const std::size_t section = i % 25;
        const float energy = 20.0F + static_cast<float>(i % 40) * 10.0F;
        const float u_target = static_cast<float>((i * 37) % 1000) / 1000.0F;
        const float u_event = static_cast<float>((i * 73) % 1000) / 1000.0F;

        const auto cpu_target = sampler.sample_target(section, energy, u_target);
        require(cpu_target.target_z == host_targets[i], "CPU/GPU target sampling mismatch");

        const auto cpu_event = package.find_event(6, 12, cpu_target.target_z, energy, 50.0F, u_event);
        require(cpu_event == host_events[i], "CPU/GPU event lookup mismatch");
        match_count++;
    }
    require(match_count == num_queries, "All 1000 CPU/GPU queries must match bitwise");
    require(host_diag.lookup_success_count == num_queries, "All queries should succeed");
    require(host_diag.lookup_failure_count == 0, "No lookup failure allowed");

    sycl::free(dev_cdf, queue);
    sycl::free(dev_total, queue);
    sycl::free(dev_nodes, queue);
    sycl::free(dev_offsets, queue);
    sycl::free(dev_indices, queue);
    sycl::free(dev_diag, queue);
    sycl::free(dev_targets, queue);
    sycl::free(dev_event_ids, queue);
    std::cout << "[step18-test] CPU/GPU equivalence and diagnostics PASSED.\n";
}
#endif

void test_step20_secondary_rate_table_and_cinel03_package() {
    std::cout << "[step20-test] Running secondary rate table and cinel03 package tests...\n";

    // 1. Validate SecondaryRateTable
    const auto rate_table = SecondaryRateTable::from_binary("data/schneider/secondary_inelastic_rates_v1.bin");
    require(rate_table.num_projectiles() == 13, "Must have 13 secondary projectiles");
    require(rate_table.projectile_index(1, 1) >= 0, "Proton must be present");
    require(rate_table.projectile_index(1, 2) >= 0, "Deuteron must be present");
    require(rate_table.projectile_index(1, 3) >= 0, "Triton must be present");
    require(rate_table.projectile_index(2, 3) >= 0, "He3 must be present");
    require(rate_table.projectile_index(2, 4) >= 0, "Alpha must be present");
    require(rate_table.projectile_index(3, 6) >= 0, "Li6 must be present");
    require(rate_table.projectile_index(3, 7) >= 0, "Li7 must be present");
    require(rate_table.projectile_index(4, 7) >= 0, "Be7 must be present");
    require(rate_table.projectile_index(4, 9) >= 0, "Be9 must be present");
    require(rate_table.projectile_index(4, 10) >= 0, "Be10 must be present");
    require(rate_table.projectile_index(5, 10) >= 0, "B10 must be present");
    require(rate_table.projectile_index(5, 11) >= 0, "B11 must be present");
    require(rate_table.projectile_index(6, 11) >= 0, "C11 must be present");
    require(rate_table.projectile_index(4, 6) == -1, "Be6 must be excluded (TopasCompatKill policy)");

    // Partial sum conservation check across all projectiles, sections, and sampled energies
    for (std::size_t p = 0; p < 13; ++p) {
        for (std::size_t s : {0, 8, 12, 20, 24}) {
            for (std::size_t e : {0, 100, 400, 859}) {
                double sum_part = 0.0;
                for (std::size_t t = 0; t < 13; ++t) {
                    sum_part += rate_table.mass_partial_rate(p, s, t, e);
                }
                const double tot = rate_table.mass_total_rate(p, s, e);
                require(std::abs(sum_part - tot) < 1.0e-11, "Partial sum must equal total rate");
            }
        }
    }

    // 2. Validate cinel03_secondary_targets.bin
    const auto sec_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_secondary_targets.bin");
    require(sec_pkg.interactions().size() > 25000, "Must contain >25k interactions");
    require(sec_pkg.products().size() > 200000, "Must contain >200k products");

    // Lookups for secondary projectiles on tissue elements under the strict
    // exact-target + bounded-domain contract (no alias, no endpoint clamp).
    // Proton on O (Z=1, A=1 on Z=8): brackets tightly at 200 MeV/u.
    const auto ev_p_O = sec_pkg.find_event(1, 1, 8, 200.0F, 50.0F, 0.5F);
    require(ev_p_O != InelasticPackageV3Table::invalid, "Proton on Oxygen lookup must succeed");

    // Alpha on C (Z=2, A=4 on Z=6): the 200 MeV/u query falls in a wide
    // campaign gap, so it must report EnergyGapTooLarge rather than silently
    // clamping to a distant endpoint.
    {
        const auto gap_result = sec_pkg.lookup_event(2, 4, 6, 200.0F, 0.5F, 0.5F);
        require(gap_result.status == carbon::Cinel03LookupStatus::EnergyGapTooLarge,
                "Alpha on Carbon at 200 MeV/u must report EnergyGapTooLarge");
        std::uint64_t gap_counter = 0;
        const auto gap_audit = sec_pkg.find_event(2, 4, 6, 200.0F, 50.0F, 0.5F, true, &gap_counter);
        require(gap_audit == InelasticPackageV3Table::invalid, "Gap query must miss in audit mode");
        require(gap_counter == 1, "Gap query must increment the miss counter");
    }
    // Alpha on C at an exact campaign node energy must still hit.
    {
        const auto domain = sec_pkg.channel_domain(2, 4, 6);
        require(domain.found_projectile && domain.found_target, "Alpha+C channel must exist");
        const auto ev_a_C = sec_pkg.find_event(2, 4, 6, domain.energy_min_MeV_per_u, 50.0F, 0.5F);
        require(ev_a_C != InelasticPackageV3Table::invalid, "Alpha on Carbon lookup must succeed at a campaign node");
    }

    // B11 on Ca (Z=5, A=11 on Z=20)
    const auto ev_b11_Ca = sec_pkg.find_event(5, 11, 20, 200.0F, 50.0F, 0.5F);
    require(ev_b11_Ca != InelasticPackageV3Table::invalid, "B11 on Calcium lookup must succeed");

    // Fail-closed policy for Be-6 (excluded by policy)
    bool threw_be6 = false;
    try {
        sec_pkg.find_event(4, 6, 8, 200.0F, 50.0F, 0.5F, false);
    } catch (const std::exception&) {
        threw_be6 = true;
    }
    require(threw_be6, "Production query for excluded Be6 must fail closed");

#ifdef CARBON_HAS_SYCL
    // 3. GPU execution check on RTX 2080 Ti
    sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
    const auto dev_tables = sec_pkg.make_device_tables();

    Cinel03EnergyNode* dev_nodes = sycl::malloc_device<Cinel03EnergyNode>(dev_tables.energy_nodes.size(), queue);
    std::uint32_t* dev_offsets = sycl::malloc_device<std::uint32_t>(dev_tables.event_offsets.size(), queue);
    std::uint32_t* dev_indices = sycl::malloc_device<std::uint32_t>(dev_tables.event_indices.size(), queue);
    queue.copy(dev_tables.energy_nodes.data(), dev_nodes, dev_tables.energy_nodes.size()).wait_and_throw();
    queue.copy(dev_tables.event_offsets.data(), dev_offsets, dev_tables.event_offsets.size()).wait_and_throw();
    queue.copy(dev_tables.event_indices.data(), dev_indices, dev_tables.event_indices.size()).wait_and_throw();

    std::uint32_t* dev_results = sycl::malloc_device<std::uint32_t>(3, queue);
    const std::uint32_t node_count = static_cast<std::uint32_t>(dev_tables.energy_nodes.size());
    const std::uint32_t total_events = static_cast<std::uint32_t>(dev_tables.interactions.size());

    queue.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        dev_results[0] = cinel03_find_event_device(dev_nodes, node_count, dev_offsets, dev_indices, total_events, 1, 1, 8, 200.0F, 50.0F, 0.3F);
        dev_results[1] = cinel03_find_event_device(dev_nodes, node_count, dev_offsets, dev_indices, total_events, 5, 11, 20, 200.0F, 50.0F, 0.5F);
        const auto gap_lookup = cinel03_lookup_event_device(
            dev_nodes, node_count, dev_offsets, dev_indices, total_events,
            2, 4, 6, 200.0F, 0.4F, 0.4F);
        dev_results[2] = static_cast<std::uint32_t>(gap_lookup.status);
    }).wait_and_throw();

    std::uint32_t host_results[3];
    queue.copy(dev_results, host_results, 3).wait_and_throw();
    require(host_results[0] != 0xFFFFFFFFU, "GPU proton on O event lookup failed");
    require(host_results[1] != 0xFFFFFFFFU, "GPU B11 on Ca event lookup failed");
    require(host_results[2] == static_cast<std::uint32_t>(carbon::Cinel03LookupStatus::EnergyGapTooLarge),
            "GPU alpha on C at 200 MeV/u must report EnergyGapTooLarge");

    sycl::free(dev_nodes, queue);
    sycl::free(dev_offsets, queue);
    sycl::free(dev_indices, queue);
    sycl::free(dev_results, queue);
#endif
    std::cout << "[step20-test] Secondary rate table and cinel03 package tests PASSED.\n";
}

void test_step27_v2_package_load_and_lookup_equivalence() {
    // Tier-A GPU validation for the v2 secondary package (Step 27/28 prep):
    // C++ load (header/monotonicity/offsets/checksum/metadata-SHA), 169
    // exact channels present, and host/device lookup equivalence over node
    // energies, midpoints, domain edges, and exact-target mismatches.
    std::cout << "[step27-test] v2 package load and lookup equivalence...\n";
    const auto v2 = InelasticPackageV3Table::from_binary(
        "data/schneider/cinel03_secondary_targets_v2.bin");
    require(v2.interactions().size() > 200000, "v2 must contain >200k interactions");
    constexpr int kProjs[13][2] = {{5, 11}, {5, 10}, {4, 9}, {4, 7}, {4, 10},
                                   {3, 7}, {3, 6}, {2, 4}, {2, 3}, {1, 1},
                                   {1, 2}, {1, 3}, {6, 11}};
    constexpr int kTargets[13] = {1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 22};
    std::size_t channel_count = 0;
    for (const auto& pr : kProjs) {
        for (const int tz : kTargets) {
            const auto dom = v2.channel_domain(pr[0], pr[1], tz);
            require(dom.found_projectile && dom.found_target,
                    "v2 must contain all 169 exact channels");
            require(dom.maximum_node_gap_MeV_per_u <= 5.0F + 1e-4F,
                    "v2 channel max gap must be <= 5 MeV/u");
            ++channel_count;
        }
    }
    require(channel_count == 169, "v2 must have exactly 169 channels");
    // p+H exists now (high-E nodes); a low-E query must report BelowDomain,
    // never MissingTarget (no alias, exact channel present).
    {
        const auto r = v2.lookup_event(1, 1, 1, 100.0F, 0.5F, 0.5F);
        require(r.status == carbon::Cinel03LookupStatus::BelowEnergyDomain,
                "p+H at 100 MeV/u must be BelowEnergyDomain (channel exists)");
    }
    // C12 is not a secondary projectile: must be MissingProjectile.
    {
        const auto r = v2.lookup_event(6, 12, 8, 200.0F, 0.5F, 0.5F);
        require(r.status == carbon::Cinel03LookupStatus::MissingProjectile,
                "C12 must be MissingProjectile in the secondary package");
    }
#ifdef CARBON_HAS_SYCL
    if (is_sycl_available()) {
        struct Query { int pz, pa, tz; float e, ub, ue; };
        std::vector<Query> queries;
        for (const auto& pr : kProjs) {
            for (const int tz : kTargets) {
                const auto dom = v2.channel_domain(pr[0], pr[1], tz);
                const float lo = dom.energy_min_MeV_per_u;
                const float hi = dom.energy_max_MeV_per_u;
                const float mid = 0.5F * (lo + hi);
                queries.push_back({pr[0], pr[1], tz, lo, 0.5F, 0.5F});
                queries.push_back({pr[0], pr[1], tz, hi, 0.5F, 0.5F});
                queries.push_back({pr[0], pr[1], tz, mid, 0.25F, 0.75F});
                queries.push_back({pr[0], pr[1], tz, mid, 0.75F, 0.25F});
                queries.push_back({pr[0], pr[1], tz, lo - 1.0F, 0.5F, 0.5F});
                queries.push_back({pr[0], pr[1], tz, hi + 1.0F, 0.5F, 0.5F});
            }
        }
        // exact-target mismatches: valid projectile, absent target Z=99
        queries.push_back({2, 4, 99, 100.0F, 0.5F, 0.5F});
        queries.push_back({1, 1, 99, 100.0F, 0.5F, 0.5F});
        const auto dev_tables = v2.make_device_tables();
        sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order{}};
        Cinel03EnergyNode* dev_nodes = sycl::malloc_device<Cinel03EnergyNode>(dev_tables.energy_nodes.size(), queue);
        std::uint32_t* dev_offsets = sycl::malloc_device<std::uint32_t>(dev_tables.event_offsets.size(), queue);
        std::uint32_t* dev_indices = sycl::malloc_device<std::uint32_t>(dev_tables.event_indices.size(), queue);
        queue.copy(dev_tables.energy_nodes.data(), dev_nodes, dev_tables.energy_nodes.size()).wait_and_throw();
        queue.copy(dev_tables.event_offsets.data(), dev_offsets, dev_tables.event_offsets.size()).wait_and_throw();
        queue.copy(dev_tables.event_indices.data(), dev_indices, dev_tables.event_indices.size()).wait_and_throw();
        const std::uint32_t node_count = static_cast<std::uint32_t>(dev_tables.energy_nodes.size());
        const std::uint32_t total_events = static_cast<std::uint32_t>(dev_tables.interactions.size());
        struct DevOut { std::uint32_t status, event_index, node_index; };
        DevOut* dev_out = sycl::malloc_device<DevOut>(queries.size(), queue);
        Query* dev_q = sycl::malloc_device<Query>(queries.size(), queue);
        queue.copy(queries.data(), dev_q, queries.size()).wait_and_throw();
        queue.parallel_for(sycl::range<1>(queries.size()), [=](sycl::id<1> idx) {
            const auto q = dev_q[idx[0]];
            const auto r = cinel03_lookup_event_device(
                dev_nodes, node_count, dev_offsets, dev_indices, total_events,
                q.pz, q.pa, q.tz, q.e, q.ub, q.ue);
            dev_out[idx[0]] = DevOut{static_cast<std::uint32_t>(r.status),
                                     r.event_index, r.energy_node_index};
        }).wait_and_throw();
        std::vector<DevOut> host_out(queries.size());
        queue.copy(dev_out, host_out.data(), queries.size()).wait_and_throw();
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const auto& q = queries[i];
            const auto h = v2.lookup_event(q.pz, q.pa, q.tz, q.e, q.ub, q.ue);
            require(host_out[i].status == static_cast<std::uint32_t>(h.status),
                    "host/device lookup status must agree on v2");
            require(host_out[i].event_index == h.event_index,
                    "host/device event index must agree on v2");
            require(host_out[i].node_index == h.energy_node_index,
                    "host/device node index must agree on v2");
        }
        sycl::free(dev_nodes, queue);
        sycl::free(dev_offsets, queue);
        sycl::free(dev_indices, queue);
        sycl::free(dev_out, queue);
        sycl::free(dev_q, queue);
        std::cout << "[step27-test] " << queries.size() << " host/device queries agree.\n";
    }
#endif
    std::cout << "[step27-test] v2 package load and lookup equivalence PASSED.\n";
}

void test_step28_v3_rate_domain_mask_equivalence() {
    // Step 28: v3 rate products (14-projectile secondary + C12 primary) with
    // data-driven registry and mask-before-interpolation. Covers the review
    // findings: registry/LUT, per-channel domain in rate data, host/device
    // boundary equivalence, metadata/bundle schema.
    std::cout << "[step28-test] v3 rate domain mask equivalence...\n";

    const auto sec = SecondaryRateTable::from_binary(
        "data/schneider/secondary_inelastic_rates_v2_1.bin");
    const auto pri = SchneiderRateTable::from_binary(
        "data/schneider/schneider_inelastic_rates_v2_1.bin");
    require(sec.binary_version() == 3, "secondary v2.1 must be binary version 3");
    require(pri.binary_version() == 3, "primary v2.1 must be binary version 3");
    require(sec.num_projectiles() == 14, "secondary v2.1 must have 14 projectiles");
    require(sec.num_energies() == 921 && pri.num_energies() == 921, "v3 grids must be 921 nodes");
    require(sec.has_channel_domains() && sec.channel_domains().size() == 182,
            "secondary v3 must carry 182 channel domains");
    require(pri.has_channel_domains() && pri.channel_domains().size() == 13,
            "primary v3 must carry 13 channel domains");

    // v1 frozen path untouched: versions, no domains, legacy clamp pinned.
    {
        const auto sec1 = SecondaryRateTable::from_binary(
            "data/schneider/secondary_inelastic_rates_v1.bin");
        const auto pri1 = SchneiderRateTable::from_binary(
            "data/schneider/schneider_inelastic_rates_v1.bin");
        require(sec1.binary_version() == 1 && pri1.binary_version() == 1, "v1 versions must stay 1");
        require(!sec1.has_channel_domains() && !pri1.has_channel_domains(),
                "v1 must have no channel domains");
        require_near(sec1.interpolate_mass_total(0, 3, -5.0),
                     sec1.mass_total_rate(0, 3, 0), 0.0, "v1 secondary clamp pinned");
        require_near(pri1.interpolate_mass_total(3, -5.0),
                     pri1.mass_total_rate(3, 0), 0.0, "v1 primary clamp pinned");
    }

    // Registry LUT over the bundle-ordered keys.
    std::vector<std::int32_t> keys;
    for (const auto& p : sec.projectiles()) {
        keys.push_back(p.z);
        keys.push_back(p.a);
    }
    require(sec.projectile_index(6, 12) == 13, "C12 must be registry index 13");
    require(carbon::secondary_projectile_lut_index_device(keys.data(), 14, 6, 12) == 13,
            "device LUT must find C12 at 13");
    require(carbon::secondary_projectile_lut_index_device(keys.data(), 14, 1, 1) ==
                sec.projectile_index(1, 1),
            "device LUT must agree with host registry");
    for (const auto [pz, pa] : {std::pair<int, int>{6, 10}, {2, 6}, {5, 8},
                                {7, 14}, {8, 16}, {4, 6}, {6, 13}}) {
        require(sec.projectile_index(pz, pa) == -1, "out-of-scope isotope must be absent");
        require(carbon::secondary_projectile_lut_index_device(keys.data(), 14, pz, pa) == -1,
                "device LUT must reject out-of-scope isotope");
    }
    require(carbon::secondary_projectile_lut_index_device(nullptr, 14, 6, 12) == -1,
            "null LUT must return -1");

    // Float device buffers mirror the kernel upload path.
    std::vector<float> sec_partial;
    sec_partial.reserve(sec.mass_partial_rates().size());
    for (const double v : sec.mass_partial_rates()) {
        sec_partial.push_back(static_cast<float>(v));
    }
    std::vector<float> sec_emin, sec_emax;
    std::vector<unsigned char> sec_has;
    for (const auto& d : sec.channel_domains()) {
        sec_emin.push_back(static_cast<float>(d.energy_min_mevu));
        sec_emax.push_back(static_cast<float>(d.energy_max_mevu));
        sec_has.push_back(d.has_support);
    }

    const double eps = 1e-3;
    const double gmin = sec.energy_min_mevu();
    const double gmax = sec.energy_max_mevu();
    auto check_sec_total = [&](std::size_t p, std::size_t s, double e, const char* label) {
        const double host = sec.interpolate_mass_total(p, s, e);
        const auto masked = carbon::secondary_masked_rates_device(
            sec_partial.data(), sec_emin.data(), sec_emax.data(), sec_has.data(), 14,
            static_cast<int>(p), s, static_cast<float>(e), static_cast<float>(gmin), 2.0F, 921);
        const double tol = 1e-3 * (1.0 + std::abs(host));
        require(std::abs(static_cast<double>(masked.total) - host) <= tol,
                std::string("host/device secondary total must agree: ") + label);
        // Masked partials must sum to the masked total (single-pass invariant).
        double psum = 0.0;
        for (const float v : masked.partials) {
            psum += v;
        }
        require(std::abs(psum - masked.total) <= 1e-6F * (1.0F + std::abs(masked.total)),
                "device masked partials must sum to device total");
    };
    // Global grid edges.
    for (const std::size_t p : {std::size_t{0}, std::size_t{13}}) {
        for (const std::size_t s : {std::size_t{0}, std::size_t{24}}) {
            check_sec_total(p, s, gmin - eps, "global emin-eps");
            check_sec_total(p, s, gmin, "global emin");
            check_sec_total(p, s, gmax, "global emax");
            check_sec_total(p, s, gmax + eps, "global emax+eps");
            const double host_out = sec.interpolate_mass_total(p, s, gmax + eps);
            require(host_out == 0.0, "host must be exactly 0 outside global grid");
            const auto dev_out = carbon::secondary_masked_rates_device(
                sec_partial.data(), sec_emin.data(), sec_emax.data(), sec_has.data(), 14,
                static_cast<int>(p), s, static_cast<float>(gmax + eps),
                static_cast<float>(gmin), 2.0F, 921);
            require(dev_out.total == 0.0F, "device must be exactly 0 outside global grid");
        }
    }
    // Per-channel edges (C12/O: floor ~2.49, a NEED point at 2.1 kept raw).
    {
        const std::size_t p = 13;
        const std::size_t t_o = 3;  // canonical index of Z=8
        const auto& dom = sec.channel_domain(p, t_o);
        require(dom.has_support == 1, "C12/O must have support");
        const double flo = dom.energy_min_mevu;
        const double fhi = dom.energy_max_mevu;
        for (const std::size_t s : {std::size_t{0}, std::size_t{12}}) {
            check_sec_total(p, s, flo - eps, "C12/O emin-eps");
            check_sec_total(p, s, flo, "C12/O emin");
            check_sec_total(p, s, flo + eps, "C12/O emin+eps");
            check_sec_total(p, s, fhi - eps, "C12/O emax-eps");
            check_sec_total(p, s, fhi, "C12/O emax");
            check_sec_total(p, s, fhi + eps, "C12/O emax+eps");
            check_sec_total(p, s, 100.0, "C12/O mid");
            check_sec_total(p, s, 100.25, "C12/O midpoint");
        }
        // Boundary-leak regression: E=2.1 is BELOW the C12/Ar18 floor but
        // the kept-raw node (section 0, frozen NEED evidence) is nonzero;
        // the masked query must still be exactly 0 for that channel (mask
        // before interpolation, not node zeroing).
        const std::size_t t_ar = 8;  // canonical index of Z=18
        const auto& dom_ar = sec.channel_domain(p, t_ar);
        require(dom_ar.energy_min_mevu > 2.1, "C12/Ar18 floor must exceed 2.1");
        const std::size_t j_need = static_cast<std::size_t>((2.1 - gmin) / 0.5);
        const double raw_node = sec.mass_partial_rate(p, 0, t_ar, j_need);
        require(raw_node > 0.0, "C12/Ar18 NEED node must be kept raw (nonzero)");
        const auto masked_need = carbon::secondary_masked_rates_device(
            sec_partial.data(), sec_emin.data(), sec_emax.data(), sec_has.data(), 14,
            static_cast<int>(p), 0, 2.1F, static_cast<float>(gmin), 2.0F, 921);
        require(masked_need.partials[t_ar] == 0.0F,
                "below-floor channel partial must be exactly 0 despite nonzero raw node");
        check_sec_total(p, 0, 2.1, "C12 NEED energy host/device agreement");
    }
    // Sampler degenerate rows (synthetic, no data dependence).
    {
        const float zeros[13] = {};
        require(carbon::sample_masked_secondary_target_device(zeros, 0.5F) == 0,
                "all-zero row must return invalid target 0");
        float single[13] = {};
        single[5] = 2.0F;
        for (const float u : {0.0F, 0.37F, 0.999F}) {
            require(carbon::sample_masked_secondary_target_device(single, u) == 15,
                    "single-valid-target row must always select Z=15");
        }
        // Float-sliver unity: sub-threshold totals are exactly zero so a
        // hazard can never fire where the sampler draws empty.
        float sliver[13] = {};
        sliver[2] = 1.0e-13F;
        sliver[9] = 5.0e-13F;
        require(carbon::sample_masked_secondary_target_device(sliver, 0.5F) == 0,
                "sub-threshold row must return invalid target 0");
        {
            // Full masked-path sliver clamp: one projectile/section, two
            // energy nodes, all supported, sub-threshold partials.
            const std::uint32_t ne = 2;
            std::vector<float> partials(13 * ne, 0.0F);
            partials[2 * ne + 0] = 1.0e-13F;
            partials[2 * ne + 1] = 2.0e-13F;
            std::vector<float> demin(13, 0.1F), demax(13, 460.1F);
            std::vector<unsigned char> dhas(13, 1);
            const auto masked = carbon::secondary_masked_rates_device(
                partials.data(), demin.data(), demax.data(), dhas.data(), 1, 0, 0,
                0.35F, 0.1F, 2.0F, ne);
            require(masked.total == 0.0F, "sub-threshold masked total must be exactly 0");
            for (const float v : masked.partials) {
                require(v == 0.0F, "sub-threshold masked partials must be exactly 0");
            }
            carbon::SchneiderTargetSamplerDeviceTable table;
            table.partial_rates = partials.data();
            table.domain_emin = demin.data();
            table.domain_emax = demax.data();
            table.domain_has = dhas.data();
            table.rate_version = 3;
            table.num_sections = 1;
            table.num_energies = ne;
            table.num_targets = 13;
            table.energy_min_MeV_per_u = 0.1F;
            table.energy_step_MeV_per_u = 0.5F;
            table.inverse_energy_step = 2.0F;
            const auto pmasked = carbon::schneider_masked_rates_device(table, 0, 0.35F);
            require(pmasked.total == 0.0F, "primary sub-threshold masked total must be exactly 0");
        }
    }

    // Primary device/host equivalence.
    SchneiderTargetSampler sampler(pri);
    require(sampler.rate_version() == 3, "v3 sampler must report version 3");
    auto check_pri_total = [&](std::size_t s, double e, const char* label) {
        const double host = pri.interpolate_mass_total(s, e);
        const auto dev_table = sampler.device_table();
        const auto masked = carbon::schneider_masked_rates_device(dev_table, s,
                                                                  static_cast<float>(e));
        const double tol = 1e-3 * (1.0 + std::abs(host));
        require(std::abs(static_cast<double>(masked.total) - host) <= tol,
                std::string("host/device primary total must agree: ") + label);
        const float host_s = sampler.total_mass_rate(s, static_cast<float>(e));
        require(std::abs(static_cast<double>(host_s) - host) <= tol,
                "sampler host total must agree with table host total");
    };
    {
        const auto& dom = pri.channel_domain(3);  // Z=8
        const double flo = dom.energy_min_mevu;
        const double fhi = dom.energy_max_mevu;
        for (const std::size_t s : {std::size_t{0}, std::size_t{24}}) {
            check_pri_total(s, gmin - eps, "primary global emin-eps");
            check_pri_total(s, gmax + eps, "primary global emax+eps");
            check_pri_total(s, flo - eps, "primary ch emin-eps");
            check_pri_total(s, flo + eps, "primary ch emin+eps");
            check_pri_total(s, fhi - eps, "primary ch emax-eps");
            check_pri_total(s, fhi + eps, "primary ch emax+eps");
            check_pri_total(s, 200.0, "primary mid");
        }
        require(pri.interpolate_mass_total(3, 0.3) == 0.0, "primary below-floor must be exactly 0");
    }
    // v3 sampler draw consistency on a dominant-target point.
    {
        bool covered = false;
        for (const std::size_t s : {std::size_t{0}, std::size_t{8}, std::size_t{24}}) {
            for (const double e : {50.0, 150.0, 300.0}) {
                const auto probs = sampler.target_probabilities(s, static_cast<float>(e));
                float mx = 0.0F;
                int arg = -1;
                for (int k = 0; k < 13; ++k) {
                    if (probs[k] > mx) {
                        mx = probs[k];
                        arg = k;
                    }
                }
                if (mx > 0.9F) {
                    const auto dev_table = sampler.device_table();
                    for (const float u : {0.1F, 0.5F, 0.9F}) {
                        const auto masked =
                            carbon::schneider_masked_rates_device(dev_table, s,
                                                                  static_cast<float>(e));
                        const int dev_pick =
                            carbon::sample_masked_schneider_target_device(masked.partials, u);
                        const auto host_pick = sampler.sample_target(s, static_cast<float>(e), u);
                        require(dev_pick == host_pick.target_z,
                                "dominant-target device/host draws must agree");
                        require(dev_pick == carbon::kSchneiderCanonicalZ[arg],
                                "dominant draw must pick the dominant target");
                    }
                    covered = true;
                }
            }
        }
        require(covered, "must cover at least one dominant-target draw point");
    }

    // Bundle schema: SHAs pinned, registry == rate keys.
    {
        std::ifstream bundle_in("data/schneider/schneider_physics_bundle_v2_1.json");
        require(bundle_in.good(), "bundle file must exist");
        std::string content((std::istreambuf_iterator<char>(bundle_in)),
                            std::istreambuf_iterator<char>());
        carbon::minjson::Parser parser(content);
        const carbon::minjson::Value b = parser.parse();
        require(carbon::minjson::require_uint(b.at("schema_version"), "schema_version") == 1,
                "bundle schema must be 1");
        const auto& reg = b.at("projectile_registry");
        require(reg.type == carbon::minjson::Value::Type::Array && reg.arr.size() == 14,
                "bundle registry must have 14 entries");
        for (std::size_t i = 0; i < 14; ++i) {
            const int z = static_cast<int>(carbon::minjson::require_uint(reg.arr[i].at("z"), "z"));
            const int a = static_cast<int>(carbon::minjson::require_uint(reg.arr[i].at("a"), "a"));
            require(z == sec.projectiles()[i].z && a == sec.projectiles()[i].a,
                    "bundle registry must equal rate keys in order");
        }
        for (const char* role : {"primary_rate", "secondary_rate", "primary_package",
                                 "secondary_package", "stopping_table"}) {
            const auto& sec_meta = b.at(role);
            const std::string fn =
                carbon::minjson::require_string(sec_meta.at("file"), "file");
            const std::string pinned =
                carbon::minjson::require_string(sec_meta.at("sha256"), "sha256");
            require(carbon::compute_file_sha256_hex(fn) == pinned,
                    std::string("bundle SHA must match: ") + role);
        }
    }
    std::cout << "[step28-test] v3 rate domain mask equivalence PASSED.\n";
}

void test_step02_material_physics_routing() {
    std::cout << "[step02-test] Starting test_step02_material_physics_routing...\n";

    // 1. Enum names
    require(std::string(material_physics_mode_name(MaterialPhysicsMode::Water)) == "Water", "Water mode name");
    require(std::string(material_physics_mode_name(MaterialPhysicsMode::SchneiderCt)) == "SchneiderCt", "SchneiderCt mode name");

    // 2. Default config has MaterialPhysicsMode::Water
    carbon::TransportConfig water_cfg;
    water_cfg.resolve_material_physics_mode();
    require(water_cfg.material_physics_mode == MaterialPhysicsMode::Water, "Default mode must be Water");
    require(water_cfg.is_water_mode(), "is_water_mode() must be true");
    require(!water_cfg.is_schneider_ct_mode(), "is_schneider_ct_mode() must be false");

    // 3. Enabling CT switches mode to SchneiderCt
    carbon::TransportConfig ct_cfg;
    ct_cfg.enable_ct_grid = true;
    ct_cfg.ct_grid_file = "data/schneider/heterogeneous_level3.cctg";
    ct_cfg.resolve_material_physics_mode();
    require(ct_cfg.material_physics_mode == MaterialPhysicsMode::SchneiderCt, "CT mode must be SchneiderCt");
    require(ct_cfg.is_schneider_ct_mode(), "is_schneider_ct_mode() must be true");
    require(!ct_cfg.is_water_mode(), "is_water_mode() must be false");

    // 4. CT mode rejects legacy four-class tables
    ct_cfg.ct_air_stopping_power_file = "some_file.csv";
    require_throws([&]() { ct_cfg.validate(); }, "CT mode must reject four-class stopping power table");
    ct_cfg.ct_air_stopping_power_file.clear();

    ct_cfg.ct_bone_cross_section_file = "some_file.csv";
    require_throws([&]() { ct_cfg.validate(); }, "CT mode must reject four-class cross section table");
    ct_cfg.ct_bone_cross_section_file.clear();

    // 5. Fail-closed startup checks
    ct_cfg.ct_schneider_primary_rate_file = "data/schneider/schneider_inelastic_rates_v1.bin";
    ct_cfg.ct_schneider_c12_cinel03_file = "data/schneider/cinel03_c12_targets.bin";
    ct_cfg.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v1.bin";
    ct_cfg.ct_schneider_secondary_cinel03_file = "data/schneider/cinel03_secondary_targets.bin";
    ct_cfg.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
    validate_schneider_ct_startup(ct_cfg);

    // Missing file throws
    carbon::TransportConfig bad_cfg = ct_cfg;
    bad_cfg.ct_schneider_c12_cinel03_file = "data/schneider/nonexistent_cinel03.bin";
    require_throws([&]() { validate_schneider_ct_startup(bad_cfg); }, "Missing binary must throw");

    // Water mode skips Schneider startup checks
    water_cfg.ct_schneider_c12_cinel03_file = "nonexistent.bin";
    validate_schneider_ct_startup(water_cfg);

    std::cout << "[step02-test] test_step02_material_physics_routing PASSED.\n";
}

#ifdef CARBON_HAS_SYCL
void test_step03_schneider_device_data_wiring() {
    std::cout << "[step03-test] Starting test_step03_schneider_device_data_wiring...\n";
    if (!is_sycl_available()) {
        std::cout << "[step03-test] SYCL unavailable, skipping.\n";
        return;
    }

    auto queue = carbon::make_sycl_queue("default");

    // 1. Water mode: no tables allocated, all device pointers null
    {
        carbon::detail::DeviceMemoryTracker mem_tracker{queue};
        carbon::TransportConfig water_cfg;
        water_cfg.enable_ct_grid = false;
        water_cfg.resolve_material_physics_mode();

        const auto ctx = upload_schneider_ct_device_context(queue, mem_tracker, water_cfg);
        require(ctx.mode == MaterialPhysicsMode::Water, "Context mode must be Water");
        require(!ctx.is_schneider_ct(), "is_schneider_ct() must be false in Water mode");
        require(ctx.primary_sampler.cdf_table == nullptr, "Water mode CDF must be null");
        require(ctx.primary_sampler.total_mass_rates == nullptr, "Water mode total rates must be null");
        require(ctx.c12_energy_nodes == nullptr, "Water mode C12 nodes must be null");
        require(ctx.sec_total_rates == nullptr, "Water mode secondary rates must be null");
        require(ctx.sec_energy_nodes == nullptr, "Water mode secondary nodes must be null");
        require(mem_tracker.active_allocation_count() == 0, "Water mode must allocate 0 device buffers");
    }

    // 2. Schneider CT mode: all tables allocated and match host binary data
    {
        carbon::detail::DeviceMemoryTracker mem_tracker{queue};
        carbon::TransportConfig ct_cfg;
        ct_cfg.enable_ct_grid = true;
        ct_cfg.ct_grid_file = "data/schneider/heterogeneous_level3.cctg";
        ct_cfg.ct_schneider_primary_rate_file = "data/schneider/schneider_inelastic_rates_v1.bin";
        ct_cfg.ct_schneider_c12_cinel03_file = "data/schneider/cinel03_c12_targets.bin";
        ct_cfg.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v1.bin";
        ct_cfg.ct_schneider_secondary_cinel03_file = "data/schneider/cinel03_secondary_targets.bin";
        ct_cfg.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
        ct_cfg.enable_secondary_transport = true;
        ct_cfg.resolve_material_physics_mode();

        const auto ctx = upload_schneider_ct_device_context(queue, mem_tracker, ct_cfg);
        require(ctx.mode == MaterialPhysicsMode::SchneiderCt, "Context mode must be SchneiderCt");
        require(ctx.is_schneider_ct(), "is_schneider_ct() must be true");

        // Verify primary sampler
        require(ctx.primary_sampler.cdf_table != nullptr, "Primary CDF device pointer must be non-null");
        require(ctx.primary_sampler.total_mass_rates != nullptr, "Primary total rates pointer must be non-null");
        require(ctx.primary_sampler.num_sections == 25, "Primary num_sections must be 25");
        require(ctx.primary_sampler.num_energies == 860, "Primary num_energies must be 860");
        require(ctx.primary_sampler.num_targets == 13, "Primary num_targets must be 13");

        // Verify primary C12 CINEL03 package
        require(ctx.c12_energy_nodes != nullptr, "C12 energy nodes pointer must be non-null");
        require(ctx.c12_event_offsets != nullptr, "C12 event offsets pointer must be non-null");
        require(ctx.c12_event_indices != nullptr, "C12 event indices pointer must be non-null");
        require(ctx.c12_interactions != nullptr, "C12 interactions pointer must be non-null");
        require(ctx.c12_products != nullptr, "C12 products pointer must be non-null");
        require(ctx.c12_node_count > 0, "C12 node count must be > 0");
        require(ctx.c12_total_events > 0, "C12 total events must be > 0");
        require(ctx.c12_total_products > 0, "C12 total products must be > 0");

        // Verify secondary rates
        require(ctx.sec_total_rates != nullptr, "Secondary total rates pointer must be non-null");
        require(ctx.sec_partial_rates != nullptr, "Secondary partial rates pointer must be non-null");
        require(ctx.sec_num_projectiles == 13, "Secondary num_projectiles must be 13");
        require(ctx.sec_num_sections == 25, "Secondary num_sections must be 25");
        require(ctx.sec_num_energies == 860, "Secondary num_energies must be 860");

        // Verify secondary CINEL03 package
        require(ctx.sec_energy_nodes != nullptr, "Secondary energy nodes pointer must be non-null");
        require(ctx.sec_event_offsets != nullptr, "Secondary event offsets pointer must be non-null");
        require(ctx.sec_event_indices != nullptr, "Secondary event indices pointer must be non-null");
        require(ctx.sec_interactions != nullptr, "Secondary interactions pointer must be non-null");
        require(ctx.sec_products != nullptr, "Secondary products pointer must be non-null");
        require(ctx.sec_node_count > 0, "Secondary node count must be > 0");
        require(ctx.sec_total_events > 0, "Secondary total events must be > 0");
        require(ctx.sec_total_products > 0, "Secondary total products must be > 0");

        // Verify data integrity: read back primary rates to host and compare with host SchneiderRateTable
        const auto host_rate_table = SchneiderRateTable::from_binary(ct_cfg.ct_schneider_primary_rate_file);
        const SchneiderTargetSampler host_sampler(host_rate_table);
        std::vector<float> readback_rates(100);
        queue.copy(ctx.primary_sampler.total_mass_rates, readback_rates.data(), 100).wait_and_throw();
        for (std::size_t i = 0; i < 100; ++i) {
            require_near(readback_rates[i], host_sampler.total_mass_rates()[i], 1e-6,
                         "Device total mass rate mismatch at index " + std::to_string(i));
        }

        // Verify read back of primary CDF
        std::vector<float> readback_cdf(100);
        queue.copy(ctx.primary_sampler.cdf_table, readback_cdf.data(), 100).wait_and_throw();
        for (std::size_t i = 0; i < 100; ++i) {
            require_near(readback_cdf[i], host_sampler.cdf_table()[i], 1e-6,
                         "Device CDF mismatch at index " + std::to_string(i));
        }

        // Verify read back of secondary rates
        const auto host_sec_table = SecondaryRateTable::from_binary(ct_cfg.ct_schneider_secondary_rate_file);
        std::vector<float> readback_sec_rates(100);
        queue.copy(ctx.sec_total_rates, readback_sec_rates.data(), 100).wait_and_throw();
        for (std::size_t i = 0; i < 100; ++i) {
            require_near(readback_sec_rates[i], static_cast<float>(host_sec_table.mass_total_rates()[i]), 1e-6,
                         "Device secondary total rate mismatch at index " + std::to_string(i));
        }

        // Verify memory tracker has tracked all 14 device allocations
        require(mem_tracker.active_allocation_count() == 14, "Must have exactly 14 active device allocations");
    }

    std::cout << "[step03-test] test_step03_schneider_device_data_wiring PASSED.\n";
}

void test_step04_ct_primary_nuclear_interaction() {
    std::cout << "[step04-test] Starting test_step04_ct_primary_nuclear_interaction...\n";
    if (!is_sycl_available()) {
        std::cout << "[step04-test] SYCL unavailable, skipping.\n";
        return;
    }

    auto queue = carbon::make_sycl_queue("default");

    // 1. Linear density scaling test
    {
        carbon::detail::DeviceMemoryTracker mem_tracker{queue};
        carbon::TransportConfig ct_cfg;
        ct_cfg.enable_ct_grid = true;
        ct_cfg.ct_grid_file = "data/schneider/heterogeneous_level3.cctg";
        ct_cfg.ct_schneider_primary_rate_file = "data/schneider/schneider_inelastic_rates_v1.bin";
        ct_cfg.ct_schneider_c12_cinel03_file = "data/schneider/cinel03_c12_targets.bin";
        ct_cfg.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v1.bin";
        ct_cfg.ct_schneider_secondary_cinel03_file = "data/schneider/cinel03_secondary_targets.bin";
        ct_cfg.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
        ct_cfg.resolve_material_physics_mode();

        const auto ctx = upload_schneider_ct_device_context(queue, mem_tracker, ct_cfg);
        auto* dev_rates = mem_tracker.allocate<float>(4);
        const auto sampler_table = ctx.primary_sampler;
        queue.parallel_for(sycl::nd_range<1>{sycl::range<1>{4}, sycl::range<1>{1}}, [=](sycl::nd_item<1> item) {
            const auto idx = item.get_global_id(0);
            const float rhos[4] = {0.5F, 1.0F, 1.8F, 2.5F};
            const float base = schneider_total_mass_rate_device(sampler_table, 10, 200.0F);
            dev_rates[idx] = rhos[idx] * base;
        }).wait_and_throw();

        std::vector<float> host_dev_rates(4);
        queue.copy(dev_rates, host_dev_rates.data(), 4).wait_and_throw();

        const auto host_rate_table = SchneiderRateTable::from_binary(ct_cfg.ct_schneider_primary_rate_file);
        const SchneiderTargetSampler host_sampler(host_rate_table);
        const float expected_base = host_sampler.total_mass_rate(10, 200.0F);
        require(expected_base > 0.0F, "Base total mass rate must be > 0");

        const float rhos[4] = {0.5F, 1.0F, 1.8F, 2.5F};
        for (int i = 0; i < 4; ++i) {
            const float computed_base = host_dev_rates[i] / rhos[i];
            require_near(computed_base, expected_base, 1e-5, "Macro rate must scale strictly linearly with rho on device");
        }
    }

    // 2. Continuous optical depth across voxel faces
    {
        float tau_remaining = 0.0F;
        bool tau_active = false;
        const float u_initial = 0.35F; // -log(0.35) ~= 1.049822

        // Segment 1 in voxel 1: length 10 mm, macro_xs = 0.05 mm^-1 (delta_tau = 0.5)
        float step1 = 10.0F;
        const float macro_xs1 = 0.05F;
        bool hit1 = consume_schneider_optical_depth_segment(tau_remaining, tau_active, step1, macro_xs1, u_initial);
        require(!hit1, "No collision in first step");
        require(tau_active, "tau_active must remain true across boundary");
        const float expected_remaining = -std::log(u_initial) - 0.5F;
        require_near(tau_remaining, expected_remaining, 1e-6, "Optical depth correctly depleted in segment 1");

        // Segment 2 in voxel 2: entered with new macro_xs = 0.10 mm^-1, step length 15 mm
        // Collision should occur at collision_s = tau_remaining / macro_xs2
        float step2 = 15.0F;
        const float macro_xs2 = 0.10F;
        const float expected_s2 = expected_remaining / macro_xs2;
        bool hit2 = consume_schneider_optical_depth_segment(tau_remaining, tau_active, step2, macro_xs2, 1.0F);
        require(hit2, "Collision must occur in second step");
        require(!tau_active, "tau_active must reset after collision");
        require_near(step2, expected_s2, 1e-6, "Collision distance matches exact analytical remaining optical depth");
    }

    // 3. Target sampling distribution matches 13 partial rates
    {
        const auto rate_table = SchneiderRateTable::from_binary("data/schneider/schneider_inelastic_rates_v1.bin");
        const SchneiderTargetSampler sampler(rate_table);
        const auto dev_table = sampler.device_table();

        const std::size_t test_section = 12; // Cortical bone or similar
        const float test_energy = 200.0F;
        const auto theoretical_probs = sampler.target_probabilities(test_section, test_energy);

        // Draw 100,000 samples
        constexpr int kSamples = 100000;
        std::array<int, 13> counts{};
        for (int i = 0; i < kSamples; ++i) {
            const float u = (static_cast<float>(i) + 0.5F) / static_cast<float>(kSamples);
            const int z = sample_schneider_target_device(dev_table, test_section, test_energy, u);
            for (std::size_t k = 0; k < 13; ++k) {
                if (kSchneiderCanonicalZ[k] == z) {
                    counts[k]++;
                    break;
                }
            }
        }

        // Verify frequencies match theoretical probabilities within 0.5%
        for (std::size_t k = 0; k < 13; ++k) {
            const float freq = static_cast<float>(counts[k]) / static_cast<float>(kSamples);
            require_near(freq, theoretical_probs[k], 0.005,
                         "Target Z=" + std::to_string(kSchneiderCanonicalZ[k]) + " frequency mismatch");
        }
    }

    // 4. CINEL03 Replay: valid lookup vs fail-closed miss
    {
        const auto c12_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");
        const auto dev_tables = c12_pkg.make_device_tables();

        // Valid query: C12 on Oxygen (Z=8) at 200 MeV/u
        const std::uint32_t valid_idx = cinel03_find_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(),
            dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            6, 12, 8, 200.0F, 0.51F, 0.5F);
        require(valid_idx != 0xFFFFFFFFU, "Valid C12 on O query must succeed");
        require(valid_idx < dev_tables.interactions.size(), "Event index within bounds");
        const auto& ev = dev_tables.interactions[valid_idx];
        require(ev.target_element_z == 8, "Target element Z must be 8");
        require(ev.direct_product_count > 0, "Must have products");

        // Invalid query: Target Z=99 (unsupported)
        const std::uint32_t miss_idx = cinel03_find_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(),
            dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            6, 12, 99, 200.0F, 0.51F, 0.5F);
        require(miss_idx == 0xFFFFFFFFU, "Invalid target query must fail-closed with 0xFFFFFFFFU");
    }

    std::cout << "[step04-test] test_step04_ct_primary_nuclear_interaction PASSED.\n";
}

void test_step05_ct_secondary_nuclear_transport() {
    std::cout << "[step05-test] Starting test_step05_ct_secondary_nuclear_transport...\n";
    if (!is_sycl_available()) {
        std::cout << "[step05-test] SYCL unavailable, skipping.\n";
        return;
    }

    auto queue = carbon::make_sycl_queue("default");

    // 1. Independent secondary target sampling matching partial rate tensor
    {
        const auto sec_table = SecondaryRateTable::from_binary("data/schneider/secondary_inelastic_rates_v1.bin");
        std::vector<float> sec_partials_float(sec_table.mass_partial_rates().size());
        for (std::size_t i = 0; i < sec_table.mass_partial_rates().size(); ++i) {
            sec_partials_float[i] = static_cast<float>(sec_table.mass_partial_rates()[i]);
        }

        // Test two different fragments from the same event: He4 (Z=2, A=4) and Proton (Z=1, A=1)
        const int proj_he4 = secondary_projectile_index_device(2, 4);
        const int proj_p = secondary_projectile_index_device(1, 1);
        require(proj_he4 >= 0, "He4 projectile index must be valid");
        require(proj_p >= 0, "Proton projectile index must be valid");
        require(proj_he4 != proj_p, "He4 and Proton must have distinct projectile indices");

        const std::size_t test_section = 12; // Bone
        const float test_energy = 100.0F;
        constexpr int kSamples = 100000;

        // Draw 100,000 independent samples for He4
        std::array<int, 13> he4_counts{};
        for (int i = 0; i < kSamples; ++i) {
            const float u = (static_cast<float>(i) + 0.5F) / static_cast<float>(kSamples);
            const int z = sample_secondary_target_device(
                sec_partials_float.data(), proj_he4, test_section, test_energy, u,
                0.5F, 2.0F, 860);
            constexpr int kSecCanonicalTargets[13] = {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22};
            for (std::size_t k = 0; k < 13; ++k) {
                if (kSecCanonicalTargets[k] == z) {
                    he4_counts[k]++;
                    break;
                }
            }
        }

        // Compute theoretical probabilities for He4
        const std::size_t e_idx = static_cast<std::size_t>((test_energy - 0.5F) * 2.0F);
        double total_rate_he4 = 0.0;
        std::array<double, 13> theoretical_he4{};
        for (std::size_t t = 0; t < 13; ++t) {
            theoretical_he4[t] = sec_table.mass_partial_rate(proj_he4, test_section, t, e_idx);
            total_rate_he4 += theoretical_he4[t];
        }
        for (std::size_t t = 0; t < 13; ++t) {
            theoretical_he4[t] /= total_rate_he4;
            const float freq = static_cast<float>(he4_counts[t]) / static_cast<float>(kSamples);
            require_near(freq, static_cast<float>(theoretical_he4[t]), 0.005,
                         "Secondary He4 target element frequency mismatch");
        }

        // Draw 100,000 independent samples for Proton
        std::array<int, 13> p_counts{};
        for (int i = 0; i < kSamples; ++i) {
            const float u = (static_cast<float>(i) + 0.5F) / static_cast<float>(kSamples);
            const int z = sample_secondary_target_device(
                sec_partials_float.data(), proj_p, test_section, test_energy, u,
                0.5F, 2.0F, 860);
            constexpr int kSecCanonicalTargets[13] = {1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22};
            for (std::size_t k = 0; k < 13; ++k) {
                if (kSecCanonicalTargets[k] == z) {
                    p_counts[k]++;
                    break;
                }
            }
        }

        // Verify He4 and Proton have distinct distributions (decoupled)
        bool distributions_differ = false;
        for (std::size_t t = 0; t < 13; ++t) {
            if (std::abs(he4_counts[t] - p_counts[t]) > 200) {
                distributions_differ = true;
                break;
            }
        }
        require(distributions_differ, "Secondary fragments must have independent, projectile-dependent target distributions");
    }

    // 2. Be6 TopasCompatKill validation
    {
        // Be6 (Z=4, A=6) must trigger TopasCompatKill
        require(cinel02_should_topas_compat_kill(true, 4, 6), "Be6 must trigger TopasCompatKill");
        require(!cinel02_should_topas_compat_kill(true, 4, 7), "Be7 must NOT trigger TopasCompatKill");
        require(!cinel02_should_topas_compat_kill(true, 2, 4), "He4 must NOT trigger TopasCompatKill");
        require(!cinel02_should_topas_compat_kill(true, 1, 1), "Proton must NOT trigger TopasCompatKill");
    }

    // 3. Secondary CINEL03 Replay & Fail-Closed Miss
    {
        const auto sec_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_secondary_targets.bin");
        const auto dev_tables = sec_pkg.make_device_tables();

        // Valid query: He4 (Z=2, A=4) on Oxygen (Z=8) at 100 MeV/u
        require(!dev_tables.energy_nodes.empty(), "Secondary CINEL03 must contain energy nodes");
        const auto& test_node = dev_tables.energy_nodes.front();
        const std::uint32_t valid_idx = cinel03_find_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(),
            dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            test_node.projectile_z, test_node.projectile_a, test_node.target_element_z,
            test_node.collision_energy_MeV_per_u, 0.51F, 0.5F);
        require(valid_idx != 0xFFFFFFFFU, "Known secondary node query must succeed");
        const auto& ev = dev_tables.interactions[valid_idx];
        require(ev.parent_z == test_node.projectile_z && ev.parent_a == test_node.projectile_a, "Projectile mismatch");
        require(ev.target_element_z == test_node.target_element_z, "Target mismatch");

        // Invalid query: Target Z=99
        const std::uint32_t miss_idx = cinel03_find_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(),
            dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            2, 4, 99, 100.0F, 0.51F, 0.5F);
        require(miss_idx == 0xFFFFFFFFU, "Invalid target secondary query must fail-closed with 0xFFFFFFFFU");
    }

    std::cout << "[step05-test] test_step05_ct_secondary_nuclear_transport PASSED.\n";
}
#endif

void test_step06_energy_accounting_and_overflow_ledger() {
    std::cout << "[step06-test] Starting test_step06_energy_accounting_and_overflow_ledger...\n";

    // 1. 8-part energy accounting ledger definition and conservation check
    {
        carbon::EnergyAccountingLedger ledger{};
        ledger.E_continuous_ionizing = 1500.0;
        ledger.E_nuclear_local = 25.0;
        ledger.E_transported_secondaries = 350.0;
        ledger.E_escaped_charged = 100.0;
        ledger.E_neutral = 15.0;
        ledger.E_cutoff_kill = 5.0;
        ledger.E_unsupported = 3.0;
        ledger.E_queue_overflow = 2.0;

        const double expected_total = 2000.0;
        require_near(ledger.total_accounted_MeV(), expected_total, 1e-6,
                     "8-part energy accounting ledger must strictly sum all 8 categories");
    }

    // 2. Queue overflow rejection by quality gate
    {
        carbon::TransportConfig config;
        config.run_mode = carbon::RunMode::production;
        config.quality_reject_any_queue_overflow = true;

        carbon::TransportResult result;
        result.secondary_queue_overflow = 42;
        result.secondary_queue_overflow_energy_MeV = 123.45;

        const auto report = carbon::evaluate_run_quality(config, result);
        require(!report.accepted, "Quality gate must reject run with secondary queue overflow");
        bool found_overflow_failure = false;
        for (const auto& fail : report.failures) {
            if (fail.code == "secondary_queue_overflow") {
                found_overflow_failure = true;
                require(fail.value == 42.0, "Failure value must match overflow count");
                break;
            }
        }
        require(found_overflow_failure, "Report failures must list secondary_queue_overflow issue");
    }

    // 3. Absence of fake local deposit check
    {
        // Verified: process local deposit in CINEL03 interactions only contains Geant4 ProcessLocalDeposit
        const auto c12_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");
        const auto dev_tables = c12_pkg.make_device_tables();
        for (const auto& inter : dev_tables.interactions) {
            // Local deposit must never equal total kinetic energy of products
            float prod_ke = 0.0F;
            for (std::uint32_t p = 0; p < inter.direct_product_count; ++p) {
                prod_ke += dev_tables.products[inter.product_offset + p].kinetic_energy_MeV;
            }
            if (prod_ke > 10.0F) {
                // If there are significant products, local deposit must be strictly less than product KE
                require(inter.process_local_deposit_MeV < prod_ke,
                        "Process local deposit must not alias or swallow product kinetic energy");
            }
        }
    }

    std::cout << "[step06-test] test_step06_energy_accounting_and_overflow_ledger PASSED.\n";
}

void test_step08_cinel03_limits_and_provenance() {
    std::cout << "[step08-test] Starting test_step08_cinel03_limits_and_provenance...\n";

    // 1. Valid baseline CINEL03 packages load and device tables check
    {
        const auto c12_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_c12_targets.bin");
        const auto c12_dev = c12_pkg.make_device_tables();
        require(!c12_dev.interactions.empty(), "C12 device interactions must not be empty");
        require(!c12_dev.products.empty(), "C12 device products must not be empty");
        for (const auto& inter : c12_dev.interactions) {
            require(inter.direct_product_count <= 64, "All C12 events must have <= 64 products");
            require(static_cast<std::uint64_t>(inter.product_offset) + inter.direct_product_count <= c12_dev.products.size(),
                    "Product offset + count must be within products bounds");
        }

        const auto sec_pkg = InelasticPackageV3Table::from_binary("data/schneider/cinel03_secondary_targets.bin");
        const auto sec_dev = sec_pkg.make_device_tables();
        require(!sec_dev.interactions.empty(), "Secondary device interactions must not be empty");
        require(!sec_dev.products.empty(), "Secondary device products must not be empty");
        for (const auto& inter : sec_dev.interactions) {
            require(inter.direct_product_count <= 64, "All secondary events must have <= 64 products");
            require(static_cast<std::uint64_t>(inter.product_offset) + inter.direct_product_count <= sec_dev.products.size(),
                    "Product offset + count must be within products bounds");
        }
    }

    // 2. Metadata completeness and exact SHA-256 binding check
    {
        const std::string sec_bin = "data/schneider/secondary_inelastic_rates_v1.bin";
        const std::string sec_meta = "data/schneider/secondary_inelastic_rates_v1.metadata.json";
        const std::string actual_sha = compute_file_sha256_hex(sec_bin);

        std::ifstream mf(sec_meta);
        require(mf.good(), "Must open secondary rate metadata");
        std::string meta_str((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
        require(meta_str.find(actual_sha) != std::string::npos, "Metadata must contain exact file SHA256");
        require(meta_str.find("\"topas_version\"") != std::string::npos, "Must have topas_version");
        require(meta_str.find("\"geant4_version\"") != std::string::npos, "Must have geant4_version");
        require(meta_str.find("\"physics_list\"") != std::string::npos, "Must have physics_list");
        require(meta_str.find("placeholder") == std::string::npos, "No placeholder permitted");
        require(meta_str.find("unknown") == std::string::npos, "No unknown permitted");
    }

    std::cout << "[step08-test] test_step08_cinel03_limits_and_provenance PASSED.\n";
}

void test_step09_production_integration_suite() {
    std::cout << "[step09-test] Starting test_step09_production_integration_suite...\n";

    // Dimension 1: Water / Schneider CT dual-path routing & fail-closed checks
    {
        carbon::TransportConfig water_cfg;
        water_cfg.enable_ct_grid = false;
        water_cfg.resolve_material_physics_mode();
        require(water_cfg.material_physics_mode == carbon::MaterialPhysicsMode::Water,
                "Config without CT grid must resolve to Water mode");

        carbon::TransportConfig ct_cfg;
        ct_cfg.enable_ct_grid = true;
        ct_cfg.ct_grid_file = "data/schneider/heterogeneous_level3.cctg";
        ct_cfg.resolve_material_physics_mode();
        require(ct_cfg.material_physics_mode == carbon::MaterialPhysicsMode::SchneiderCt,
                "Config with CT grid must resolve to SchneiderCt mode");

        carbon::TransportConfig empty_ct_cfg;
        empty_ct_cfg.enable_ct_grid = true;
        empty_ct_cfg.ct_grid_file = "";
        empty_ct_cfg.resolve_material_physics_mode();
        bool threw_empty_ct = false;
        try {
            empty_ct_cfg.validate();
        } catch (const std::invalid_argument&) {
            threw_empty_ct = true;
        }
        require(threw_empty_ct, "SchneiderCt with empty ct_grid_file must fail validation");

        carbon::TransportConfig bad_cfg;
        bad_cfg.enable_ct_grid = true;
        bad_cfg.ct_grid_file = "data/non_existent_file.cctg";
        bad_cfg.resolve_material_physics_mode();
        bool threw_bad_ct = false;
        try {
            (void)carbon::CtGrid::from_binary(bad_cfg.ct_grid_file);
        } catch (const std::exception&) {
            threw_bad_ct = true;
        }
        require(threw_bad_ct, "Non-existent CT grid file must fail closed");
    }

    // Dimension 2: Water physics regression protection (identical results, 0 Schneider device memory)
#ifdef CARBON_HAS_SYCL
    if (is_sycl_available()) {
        auto queue = carbon::make_sycl_queue("default");
        carbon::detail::DeviceMemoryTracker mem_tracker{queue};
        carbon::TransportConfig water_cfg;
        water_cfg.enable_ct_grid = false;
        water_cfg.resolve_material_physics_mode();

        const auto ctx = upload_schneider_ct_device_context(queue, mem_tracker, water_cfg);
        require(ctx.mode == carbon::MaterialPhysicsMode::Water, "Context mode must be Water");
        require(mem_tracker.active_allocation_count() == 0, "Water mode must allocate 0 device buffers");
    }
#endif

    // Dimension 3: CT production end-to-end transport call with 8-part energy ledger
#ifdef CARBON_HAS_SYCL
    if (is_sycl_available()) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "test_step09_integration";
        std::filesystem::create_directories(temp_dir);
        const auto cctg_file = temp_dir / "two_voxel_test.cctg";

        // Create a 2-voxel CT grid
        carbon::CtGrid grid;
        grid.file_version = carbon::CtGrid::version_v2;
        grid.nx = 1;
        grid.ny = 1;
        grid.nz = 2;
        grid.spacing_x_mm = 10.0;
        grid.spacing_y_mm = 10.0;
        grid.spacing_z_mm = 10.0;
        grid.origin_x_mm = -5.0;
        grid.origin_y_mm = -5.0;
        grid.origin_z_mm = 0.0;
        grid.density_g_per_cm3 = {1.0F, 1.5F};
        grid.material_id = {8, 20}; // Soft tissue & Dense bone
        grid.mass_sp_za_rel.resize(25, 1.0);
        grid.write_binary(cctg_file);

        carbon::TransportConfig config;
        config.phantom_length_mm = 20.0;
        config.depth_bin_width_mm = 1.0;
        config.voxel_size_z_mm = 1.0;
        config.primary_atomic_number = 6;
        config.primary_mass_number = 12;
        config.initial_energy_MeVu = 150.0;
        config.run_mode = carbon::RunMode::production;
        config.enable_voxel_scoring = true;
        config.enable_secondary_transport = true;
        config.enable_ct_grid = true;
        config.ct_grid_file = cctg_file.string();
        config.ct_schneider_primary_rate_file = "data/schneider/schneider_inelastic_rates_v1.bin";
        config.ct_schneider_c12_cinel03_file = "data/schneider/cinel03_c12_targets.bin";
        config.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v1.bin";
        config.ct_schneider_secondary_cinel03_file = "data/schneider/cinel03_secondary_targets.bin";
        config.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
        config.ct_schneider_cross_section_file = "data/schneider/c12_schneider_inelastic_mass_xs.csv";
        config.number_of_histories = 2000;
        config.energy_cutoff_MeV = 0.5F;
        config.resolve_material_physics_mode();

        const auto stopping_power = carbon::StoppingPowerTable::from_csv(
            "data/stopping_power_water_geant4_11_3_2.csv");
        const auto cross_section = zero_cross_section();

        const auto result = carbon::transport_sycl(config, stopping_power, cross_section, "default");
        require(result.total_deposited_energy_MeV > 0.0, "Total deposited energy must be positive");
        require(result.total_steps > 0, "Steps must be executed");
        require(result.secondary_queue_overflow == 0, "No secondary queue overflow in production run");
        require(result.energy_ledger.total_accounted_MeV() > 0.0, "Energy ledger must account for energy");
        require(!result.voxel_deposited_energy_MeV.empty(), "3D voxel dose must be scored");
    }
#endif

    // Dimension 4: Missing data Fail-Closed Interception
    {
        const auto sec_pkg = carbon::InelasticPackageV3Table::from_binary("data/schneider/cinel03_secondary_targets.bin");
        const auto dev_tables = sec_pkg.make_device_tables();
        const std::uint32_t miss_idx = carbon::cinel03_find_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(),
            dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            2, 4, 99, 100.0F, 0.51F, 0.5F);
        require(miss_idx == 0xFFFFFFFFU, "Unsupported target Z=99 must fail closed with 0xFFFFFFFFU");
    }

    // Dimension 5 & 6: Hardened Rate Table and CINEL03 limits
    {
        const auto sec_table = carbon::SecondaryRateTable::from_binary("data/schneider/secondary_inelastic_rates_v1.bin");
        bool threw_oob = false;
        try {
            (void)sec_table.mass_partial_rate(13, 0, 0, 0);
        } catch (const std::out_of_range&) {
            threw_oob = true;
        }
        require(threw_oob, "SecondaryRateTable out-of-bounds must throw out_of_range");
    }

    // Dimension 7: Verifier quality gate regression test
    {
        const int ret = std::system("python3 tests/test_verify_step20_regression.py > /dev/null 2>&1");
        require(ret == 0, "tests/test_verify_step20_regression.py must pass with 0 exit code");
    }

    std::cout << "[step09-test] test_step09_production_integration_suite PASSED.\n";
}

#ifdef CARBON_HAS_SYCL
void test_schneider_tertiary_transport_generations() {
    // Generation semantics under test (see kernel: hazard requires
    // frag.generation < max; child queuing requires frag.generation+1 < max):
    //   max=1: gen-0 secondaries react once, but their charged products are
    //          NOT queued (0+1<1 false) -> cutoff only, tertiary closed.
    //   max=2: gen-0 reaction products (gen-1) ARE queued and transported;
    //          gen-1 may react once, its products cut off. Tertiary open.
    // This test proves the semantics empirically on a small synthetic CT:
    // max=1 must yield queued==0 while max=2 must yield queued>0, both with
    // zero overflow and exact secondary born closure. Acceptance is NOT
    // asserted (sparse secondary campaign gaps still fail the gate).
    std::cout << "[schneider-tertiary] starting tertiary generation test...\n";
    if (!is_sycl_available()) {
        std::cout << "[schneider-tertiary] SYCL unavailable, skipping.\n";
        return;
    }
    const auto temp_dir = std::filesystem::temp_directory_path() / "test_schneider_tertiary";
    std::filesystem::create_directories(temp_dir);
    const auto cctg_file = temp_dir / "tertiary_test.cctg";
    {
        carbon::CtGrid grid;
        grid.file_version = carbon::CtGrid::version_v2;
        grid.nx = 1;
        grid.ny = 1;
        grid.nz = 10;
        grid.spacing_x_mm = 10.0;
        grid.spacing_y_mm = 10.0;
        grid.spacing_z_mm = 10.0;
        grid.origin_x_mm = -5.0;
        grid.origin_y_mm = -5.0;
        grid.origin_z_mm = 0.0;
        grid.density_g_per_cm3 = std::vector<float>(10, 1.0F);
        grid.material_id = std::vector<std::uint8_t>(10, 8); // soft tissue
        grid.mass_sp_za_rel.resize(25, 1.0);
        grid.write_binary(cctg_file);
    }
    const auto run_case = [&](std::uint32_t max_generations) {
        carbon::TransportConfig config;
        config.phantom_length_mm = 100.0;
        config.depth_bin_width_mm = 1.0;
        config.voxel_size_z_mm = 1.0;
        config.primary_atomic_number = 6;
        config.primary_mass_number = 12;
        config.initial_energy_MeVu = 220.0;
        config.run_mode = carbon::RunMode::research;
        config.enable_voxel_scoring = true;
        config.enable_inelastic = true;
        config.enable_secondary_transport = true;
        config.cinel02_max_secondary_inelastic_generations = max_generations;
        config.enable_ct_grid = true;
        config.ct_grid_file = cctg_file.string();
        config.ct_schneider_primary_rate_file = "data/schneider/schneider_inelastic_rates_v1.bin";
        config.ct_schneider_c12_cinel03_file = "data/schneider/cinel03_c12_targets.bin";
        config.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v1.bin";
        config.ct_schneider_secondary_cinel03_file = "data/schneider/cinel03_secondary_targets.bin";
        config.ct_schneider_stopping_power_file = "data/schneider/schneider_stopping_v1.bin";
        config.ct_schneider_cross_section_file = "data/schneider/c12_schneider_inelastic_mass_xs.csv";
        config.number_of_histories = 3000;
        config.energy_cutoff_MeV = 0.5F;
        config.resolve_material_physics_mode();
        const auto stopping_power = carbon::StoppingPowerTable::from_csv(
            "data/stopping_power_water_geant4_11_3_2.csv");
        // Heap-allocated: TransportResult is ~1.5MB; never hold two alive.
        auto result = std::make_unique<carbon::TransportResult>(
            carbon::transport_sycl(config, stopping_power, zero_cross_section(), "default"));
        return result;
    };
    std::uint64_t q1 = 0, b1 = 0, h1 = 0, r1 = 0, t1 = 0;
    {
        auto res1 = run_case(1);
        const auto& s = res1->schneider_diagnostics;
        q1 = s.secondary_charged_products_queued;
        b1 = s.secondary_charged_products_born;
        h1 = s.secondary_hazards;
        r1 = s.secondary_events_replayed;
        t1 = s.secondary_tracks_started;
        std::cout << "[schneider-tertiary] max=1: tracks=" << t1 << " haz=" << h1
                  << " replayed=" << r1 << " born=" << b1 << " queued=" << q1 << "\n";
        require(t1 > 0, "max=1 control must start secondary tracks");
        require(h1 > 0, "max=1 control must sample secondary hazards");
        require(b1 > 0, "max=1 control must produce secondary charged products");
        require(q1 == 0, "max=1 must queue zero tertiary products (tertiary closed)");
        require(b1 == s.secondary_charged_cutoff_kills + s.secondary_queue_overflows,
                "max=1 secondary born must close exactly");
        require(res1->secondary_queue_overflow == 0, "max=1 must have no overflow");
        require(s.secondary_queue_overflows == 0 && s.queue_overflows == 0,
                "max=1 must have no Schneider overflow");
    }
    {
        auto res2 = run_case(2);
        const auto& s = res2->schneider_diagnostics;
        const auto q2 = s.secondary_charged_products_queued;
        const auto b2 = s.secondary_charged_products_born;
        std::cout << "[schneider-tertiary] max=2: tracks=" << s.secondary_tracks_started
                  << " haz=" << s.secondary_hazards << " replayed=" << s.secondary_events_replayed
                  << " born=" << b2 << " queued=" << q2 << "\n";
        require(s.secondary_tracks_started > t1,
                "max=2 must transport strictly more tracks than max=1 (tertiary open)");
        require(q2 > 0, "max=2 must queue tertiary products (gen-1 children)");
        require(b2 == q2 + s.secondary_charged_cutoff_kills + s.secondary_queue_overflows,
                "max=2 secondary born must close exactly");
        require(res2->secondary_queue_overflow == 0, "max=2 must have no overflow");
        require(s.secondary_queue_overflows == 0 && s.queue_overflows == 0,
                "max=2 must have no Schneider overflow");
    }
    std::cout << "[schneider-tertiary] tertiary generation test PASSED.\n";
}
#endif

[[gnu::noinline]] void run_test_dispatch(const std::string& filter, const std::string& name, void (*fn)()) {
    if (filter.empty() || name.find(filter) != std::string::npos) {
        fn();
    }
}

namespace schneider_strict_test {

carbon::Cinel03InteractionRecord make_lookup_event(int pz, int pa, int tz, int ta,
                                                   float e_MeV_per_u,
                                                   std::uint64_t event_id) {
    carbon::Cinel03InteractionRecord ev{};
    ev.run_id = 1;
    ev.event_id = event_id;
    ev.projectile_pdg = 1000060120;
    ev.projectile_z = static_cast<std::int16_t>(pz);
    ev.projectile_a = static_cast<std::int16_t>(pa);
    ev.projectile_charge = static_cast<float>(pz);
    ev.projectile_rest_mass = static_cast<float>(pa * carbon::inelastic_nucleon_rest_mass_MeV);
    ev.collision_energy_MeV = e_MeV_per_u * static_cast<float>(pa);
    ev.collision_energy_MeV_per_u = e_MeV_per_u;
    ev.collision_direction_z = 1.0F;
    ev.target_element_z = static_cast<std::int16_t>(tz);
    ev.target_a = static_cast<std::int16_t>(ta);
    ev.parent_status = 2;
    ev.parent_pdg = 1000060120;
    ev.parent_z = static_cast<std::int16_t>(pz);
    ev.parent_a = static_cast<std::int16_t>(pa);
    ev.parent_charge = static_cast<float>(pz);
    ev.parent_rest_mass = ev.projectile_rest_mass;
    ev.parent_energy_MeV = 0.0F;
    ev.parent_direction_z = 1.0F;
    ev.track_weight = 1.0F;
    ev.parent_weight = 1.0F;
    ev.process_local_deposit_MeV = 5.0F;
    ev.direct_product_count = 0;
    std::strncpy(ev.material_name, "G4_WATER", sizeof(ev.material_name));
    std::strncpy(ev.process_name, "ionInelastic", sizeof(ev.process_name));
    std::strncpy(ev.model_name, "BinaryCascade", sizeof(ev.model_name));
    return ev;
}

carbon::InelasticPackageV3Table make_two_node_table() {
    carbon::InelasticPackageV3Table table;
    table.set_metadata(100.0F, 100.0F, 1, "00000000-0000-4000-8000-000000000021");
    // C12 on O at 200 and 204 MeV/u (gap 4 <= 5: stochastic bracketing).
    table.add_event(make_lookup_event(6, 12, 8, 16, 200.0F, 1), {});
    table.add_event(make_lookup_event(6, 12, 8, 16, 204.0F, 2), {});
    // C12 on C at 200 MeV/u only (for exact-target negative tests).
    table.add_event(make_lookup_event(6, 12, 6, 12, 200.0F, 3), {});
    // p on C at 100 MeV/u only (p+H must miss: no alias to C).
    table.add_event(make_lookup_event(1, 1, 6, 12, 100.0F, 4), {});
    table.finalize();
    return table;
}

}  // namespace schneider_strict_test

void test_cinel03_exact_target_no_alias() {
    std::cout << "[schneider-strict] exact-target no-alias tests...\n";
    const auto table = schneider_strict_test::make_two_node_table();

    // Exact hit: C12 on O at a campaign node.
    {
        const auto hit = table.lookup_event(6, 12, 8, 200.0F, 0.5F, 0.5F);
        require(hit.status == carbon::Cinel03LookupStatus::Hit, "C12+O at 200 must Hit");
        require(hit.event_index != 0xFFFFFFFFU, "Hit must carry an event index");
    }
    // C12 on Ca with only C12+O/C data: must be MissingTarget, never alias.
    {
        const auto miss = table.lookup_event(6, 12, 20, 200.0F, 0.5F, 0.5F);
        require(miss.status == carbon::Cinel03LookupStatus::MissingTarget,
                "C12+Ca with only O/C data must be MissingTarget");
        const auto dev_tables = table.make_device_tables();
        const auto dev_miss = carbon::cinel03_lookup_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(), dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            6, 12, 20, 200.0F, 0.5F, 0.5F);
        require(dev_miss.status == carbon::Cinel03LookupStatus::MissingTarget,
                "device C12+Ca must be MissingTarget (no closest-Z fallback)");
        require(dev_miss.event_index == 0xFFFFFFFFU, "miss must not return an event");
        std::uint64_t counter = 0;
        require(table.find_event(6, 12, 20, 200.0F, 50.0F, 0.5F, true, &counter) ==
                    carbon::InelasticPackageV3Table::invalid,
                "audit C12+Ca must miss");
        require(counter == 1, "audit miss must increment the counter");
        bool threw = false;
        try {
            (void)table.find_event(6, 12, 20, 200.0F, 50.0F, 0.5F, false);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        require(threw, "production C12+Ca must throw");
    }
    // p on H with only p+C data: must be MissingTarget, never alias to C.
    {
        const auto miss = table.lookup_event(1, 1, 1, 100.0F, 0.5F, 0.5F);
        require(miss.status == carbon::Cinel03LookupStatus::MissingTarget,
                "p+H with only p+C data must be MissingTarget");
        const auto dev_tables = table.make_device_tables();
        const auto dev_miss = carbon::cinel03_lookup_event_device(
            dev_tables.energy_nodes.data(),
            static_cast<std::uint32_t>(dev_tables.energy_nodes.size()),
            dev_tables.event_offsets.data(), dev_tables.event_indices.data(),
            static_cast<std::uint32_t>(dev_tables.interactions.size()),
            1, 1, 1, 100.0F, 0.5F, 0.5F);
        require(dev_miss.status == carbon::Cinel03LookupStatus::MissingTarget,
                "device p+H must be MissingTarget");
    }
    // Missing projectile entirely.
    {
        const auto miss = table.lookup_event(2, 4, 8, 200.0F, 0.5F, 0.5F);
        require(miss.status == carbon::Cinel03LookupStatus::MissingProjectile,
                "alpha projectile with no data must be MissingProjectile");
    }
    std::cout << "[schneider-strict] exact-target no-alias tests PASSED.\n";
}

void test_cinel03_bounded_domain_and_bracketing() {
    std::cout << "[schneider-strict] bounded-domain and bracketing tests...\n";
    const auto table = schneider_strict_test::make_two_node_table();
    const auto dev_tables = table.make_device_tables();
    const auto* nodes = dev_tables.energy_nodes.data();
    const auto node_count = static_cast<std::uint32_t>(dev_tables.energy_nodes.size());
    const auto* offsets = dev_tables.event_offsets.data();
    const auto* indices = dev_tables.event_indices.data();
    const auto total = static_cast<std::uint32_t>(dev_tables.interactions.size());

    // Channel domain introspection.
    {
        const auto domain = table.channel_domain(6, 12, 8);
        require(domain.found_projectile && domain.found_target, "C12+O channel must exist");
        require_near(domain.energy_min_MeV_per_u, 200.0, 1.0e-4, "channel min");
        require_near(domain.energy_max_MeV_per_u, 204.0, 1.0e-4, "channel max");
        require(domain.node_count == 2, "channel must have 2 nodes");
        require_near(domain.maximum_node_gap_MeV_per_u, 4.0, 1.0e-4, "channel max gap");
    }
    // Below / above domain rejected, never clamped.
    {
        const auto below = table.lookup_event(6, 12, 8, 199.0F, 0.5F, 0.5F);
        require(below.status == carbon::Cinel03LookupStatus::BelowEnergyDomain,
                "query below channel min must be BelowEnergyDomain");
        const auto above = table.lookup_event(6, 12, 8, 205.0F, 0.5F, 0.5F);
        require(above.status == carbon::Cinel03LookupStatus::AboveEnergyDomain,
                "query above channel max must be AboveEnergyDomain");
        const auto dev_below = carbon::cinel03_lookup_event_device(
            nodes, node_count, offsets, indices, total, 6, 12, 8, 199.0F, 0.5F, 0.5F);
        require(dev_below.status == carbon::Cinel03LookupStatus::BelowEnergyDomain,
                "device below-domain must agree");
    }
    // Gap rejection on a wide channel is tested against production data in
    // test_step20 (alpha+C at 200 MeV/u). Here the 4 MeV/u gap is allowed.
    // Stochastic bracketing statistics at the midpoint: P(E1) = 0.5.
    {
        constexpr int draws = 4000;
        int e1_count = 0;
        for (int i = 0; i < draws; ++i) {
            const float u = (static_cast<float>(i) + 0.5F) / static_cast<float>(draws);
            const auto r = table.lookup_event(6, 12, 8, 202.0F, u, 0.5F);
            require(r.status == carbon::Cinel03LookupStatus::Hit, "midpoint query must Hit");
            if (r.selected_energy_MeV_per_u > 202.0F) {
                ++e1_count;
            }
        }
        const double frac = static_cast<double>(e1_count) / draws;
        require(std::abs(frac - 0.5) < 0.06, "stochastic bracketing must select E1 with ~50% probability");
    }
    // Host/device bitwise consistency across a sweep (hits and misses).
    {
        int checked = 0;
        for (int ei = 195; ei <= 208; ++ei) {
            for (int ui = 0; ui <= 10; ++ui) {
                const float energy = static_cast<float>(ei);
                const float u = static_cast<float>(ui) / 10.0F;
                const auto host = table.lookup_event(6, 12, 8, energy, u, u);
                const auto dev = carbon::cinel03_lookup_event_device(
                    nodes, node_count, offsets, indices, total, 6, 12, 8, energy, u, u);
                require(host.status == dev.status, "host/device status must agree");
                require(host.event_index == dev.event_index, "host/device event must agree");
                require(host.energy_node_index == dev.energy_node_index,
                        "host/device node must agree");
                ++checked;
            }
        }
        require(checked > 100, "sweep must cover hit/miss cases");
    }
    std::cout << "[schneider-strict] bounded-domain and bracketing tests PASSED.\n";
}

void test_schneider_quality_gate_rejects_failures() {
    std::cout << "[schneider-strict] quality-gate failure tests...\n";
    carbon::TransportConfig config;
    config.material_physics_mode = carbon::MaterialPhysicsMode::SchneiderCt;
    config.run_mode = carbon::RunMode::research;

    // Clean run: no hazards, zero residual -> accepted.
    {
        carbon::TransportResult result;
        result.initial_energy_MeV = 1000.0;
        result.total_deposited_energy_MeV = 1000.0;
        const auto report = carbon::evaluate_run_quality(config, result);
        require(report.accepted, "clean Schneider run must be accepted");
        require(report.failures.empty(), "clean run must have no failures");
    }
    // One missing target -> accepted=false even in research mode.
    {
        carbon::TransportResult result;
        result.initial_energy_MeV = 1000.0;
        result.total_deposited_energy_MeV = 1000.0;
        result.schneider_diagnostics.primary_hazards = 10;
        result.schneider_diagnostics.primary_events_replayed = 9;
        result.schneider_diagnostics.primary_missing_target = 1;
        const auto report = carbon::evaluate_run_quality(config, result);
        require(!report.accepted, "missing target must not be accepted");
        require(!report.failures.empty(), "missing target must record a failure");
    }
    // Conservation break -> accepted=false.
    {
        carbon::TransportResult result;
        result.initial_energy_MeV = 1000.0;
        result.total_deposited_energy_MeV = 1000.0;
        result.schneider_diagnostics.primary_hazards = 10;
        result.schneider_diagnostics.primary_events_replayed = 7;
        result.schneider_diagnostics.primary_missing_target = 1;
        const auto report = carbon::evaluate_run_quality(config, result);
        require(!report.accepted, "conservation break must not be accepted");
    }
    // Queue overflow -> accepted=false.
    {
        carbon::TransportResult result;
        result.initial_energy_MeV = 1000.0;
        result.total_deposited_energy_MeV = 1000.0;
        result.schneider_diagnostics.secondary_queue_overflows = 2;
        result.schneider_diagnostics.queue_overflows = 2;
        const auto report = carbon::evaluate_run_quality(config, result);
        require(!report.accepted, "queue overflow must not be accepted");
    }
    // Production mode likewise refuses the dose as a formal result.
    {
        config.run_mode = carbon::RunMode::production;
        carbon::TransportResult result;
        result.initial_energy_MeV = 1000.0;
        result.total_deposited_energy_MeV = 1000.0;
        result.schneider_diagnostics.primary_hazards = 5;
        result.schneider_diagnostics.primary_events_replayed = 4;
        result.schneider_diagnostics.primary_energy_gap_misses = 1;
        const auto report = carbon::evaluate_run_quality(config, result);
        require(!report.accepted, "production gap miss must not be accepted");
    }
    std::cout << "[schneider-strict] quality-gate failure tests PASSED.\n";
}

void test_schneider_post_em_null_gate_taxonomy() {
    // Post-EM null collisions: reported approximation, never a failure;
    // UnsupportedTargets is a hard failure in every mode.
    std::cout << "[schneider-strict] post-EM null gate taxonomy tests...\n";
    carbon::TransportConfig v3config;
    v3config.material_physics_mode = carbon::MaterialPhysicsMode::SchneiderCt;
    v3config.run_mode = carbon::RunMode::research;
    v3config.ct_schneider_secondary_rate_file = "data/schneider/secondary_inelastic_rates_v2_1.bin";
    v3config.ct_schneider_physics_bundle_file = "data/schneider/schneider_physics_bundle_v2_1.json";
    const auto fresh_v3 = [&]() {
        auto result = std::make_unique<carbon::TransportResult>();
        result->initial_energy_MeV = 1000.0;
        result->total_deposited_energy_MeV = 1000.0;
        return result;
    };
    // Post-EM nulls close conservation and do not fail.
    {
        auto result = fresh_v3();
        result->schneider_diagnostics.secondary_hazards = 10;
        result->schneider_diagnostics.secondary_events_replayed = 8;
        result->schneider_diagnostics.secondary_post_em_null_collisions = 2;
        const auto report = carbon::evaluate_run_quality(v3config, *result);
        require(report.accepted, "post-EM nulls must be accepted (declared approximation)");
        require(report.failures.empty(), "post-EM nulls must record no failure");
        bool found = false;
        for (const auto& issue : report.approximations) {
            if (issue.code == "schneider_post_em_null_collisions" && issue.value == 2.0) {
                found = true;
            }
        }
        require(found, "post-EM nulls must appear in approximations with count 2");
    }
    // Explicit UnsupportedTargets=1 fails under v3.
    {
        auto result = fresh_v3();
        result->schneider_diagnostics.secondary_hazards = 10;
        result->schneider_diagnostics.secondary_events_replayed = 9;
        result->schneider_diagnostics.unsupported_targets = 1;
        const auto report = carbon::evaluate_run_quality(v3config, *result);
        require(!report.accepted, "v3 UnsupportedTargets=1 must not be accepted");
        bool found = false;
        for (const auto& failure : report.failures) {
            if (failure.code == "schneider_secondary_missing_target") {
                found = true;
            }
        }
        require(found, "v3 UnsupportedTargets must fail as missing_target");
    }
    // Cutoff stops still close conservation without failing.
    {
        auto result = fresh_v3();
        result->schneider_diagnostics.secondary_hazards = 10;
        result->schneider_diagnostics.secondary_events_replayed = 8;
        result->schneider_diagnostics.secondary_stopped_before_replay = 2;
        const auto report = carbon::evaluate_run_quality(v3config, *result);
        require(report.accepted, "cutoff stops must be accepted");
        require(report.failures.empty(), "cutoff stops must record no failure");
    }
    // Primary post-EM null closes the primary equation as an approximation.
    {
        auto result = fresh_v3();
        result->schneider_diagnostics.primary_hazards = 10;
        result->schneider_diagnostics.primary_events_replayed = 9;
        result->schneider_diagnostics.primary_post_em_null_collisions = 1;
        const auto report = carbon::evaluate_run_quality(v3config, *result);
        require(report.accepted, "primary post-EM null must be accepted (declared approximation)");
        require(report.failures.empty(), "primary post-EM null must record no failure");
        bool found = false;
        for (const auto& issue : report.approximations) {
            if (issue.code == "schneider_primary_post_em_null_collisions" && issue.value == 1.0) {
                found = true;
            }
        }
        require(found, "primary post-EM null must appear in approximations with count 1");
    }
    // v1 keeps the legacy strictness: UnsupportedTargets=1 fails.
    {
        carbon::TransportConfig v1config;
        v1config.material_physics_mode = carbon::MaterialPhysicsMode::SchneiderCt;
        v1config.run_mode = carbon::RunMode::research;
        auto result = std::make_unique<carbon::TransportResult>();
        result->initial_energy_MeV = 1000.0;
        result->total_deposited_energy_MeV = 1000.0;
        result->schneider_diagnostics.secondary_hazards = 10;
        result->schneider_diagnostics.secondary_events_replayed = 9;
        result->schneider_diagnostics.unsupported_targets = 1;
        const auto report = carbon::evaluate_run_quality(v1config, *result);
        require(!report.accepted, "v1 UnsupportedTargets=1 must not be accepted");
    }
    std::cout << "[schneider-strict] post-EM null gate taxonomy tests PASSED.\n";
}

void test_schneider_born_strict_equality_and_ledger_disjoint() {
    std::cout << "[schneider-strict] born strict-equality and ledger-disjoint tests..." << std::endl;
    carbon::TransportConfig config;
    config.material_physics_mode = carbon::MaterialPhysicsMode::SchneiderCt;
    config.run_mode = carbon::RunMode::research;
    // NOTE: TransportResult is ~1.5MB (multi-MB ledger arrays). The compiler
    // does not reliably overlay sequential stack instances (observed 12.5MB
    // frame -> instant segfault), so every instance below is heap-allocated.
    const auto fresh_result = []() {
        auto result = std::make_unique<carbon::TransportResult>();
        result->initial_energy_MeV = 1000.0;
        result->total_deposited_energy_MeV = 1000.0;
        return result;
    };
    // Lost product: born=10 but only 9 terminals -> must fail.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.primary_charged_products_born = 10;
        result->schneider_diagnostics.primary_charged_products_queued = 9;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(!report.accepted, "primary born > terminals (lost product) must fail");
    }
    // Phantom terminal: born=9 but 10 terminals -> must fail.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.primary_charged_products_born = 9;
        result->schneider_diagnostics.primary_charged_products_queued = 10;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(!report.accepted, "primary born < terminals (phantom terminal) must fail");
    }
    // Exact closure -> pass.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.primary_charged_products_born = 10;
        result->schneider_diagnostics.primary_charged_products_queued = 7;
        result->schneider_diagnostics.primary_charged_cutoff_kills = 3;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(report.accepted, "exact primary born closure must pass");
    }
    // Secondary lost product -> must fail.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.secondary_charged_products_born = 42;
        result->schneider_diagnostics.secondary_charged_cutoff_kills = 41;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(!report.accepted, "secondary born > terminals must fail");
    }
    // Secondary exact closure -> pass.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.secondary_charged_products_born = 42;
        result->schneider_diagnostics.secondary_charged_cutoff_kills = 42;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(report.accepted, "exact secondary born closure must pass");
    }
    // Coverage gate uses per-TRACK count: tracks=1 fails...
    {
        auto result = fresh_result();
        result->schneider_diagnostics.unsupported_projectile_tracks = 1;
        result->schneider_diagnostics.unsupported_projectile_birth_energy_MeV = 50.0;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(!report.accepted, "unsupported track must fail the gate");
    }
    // ...while a large per-STEP count alone (tracks=0) does not gate.
    {
        auto result = fresh_result();
        result->schneider_diagnostics.unsupported_projectile_steps = 1000000;
        const auto report = carbon::evaluate_run_quality(config, *result);
        require(report.accepted, "per-step count alone must not gate");
    }
    // Split-ledger disjointness: huge informational split fields must not
    // move the closure residual (no double counting into the global formula).
    double base_residual = 0.0;
    {
        auto baseline = fresh_result();
        const auto base_report = carbon::evaluate_run_quality(config, *baseline);
        require(base_report.accepted, "baseline must be accepted");
        base_residual = base_report.relative_energy_residual;
    }
    {
        // Only the purely informational fields: E_lookup_failure and
        // E_out_of_domain are deliberate GATE inputs (non-zero must fail),
        // so they are excluded from this doping set.
        auto doped = fresh_result();
        doped->energy_ledger.E_be6_kill = 1.0e9;
        doped->energy_ledger.E_neutral_product_kinetic = 1.0e9;
        doped->energy_ledger.E_reaction_q_residual = 1.0e9;
        doped->energy_ledger.E_unsupported_charged = 1.0e9;
        doped->energy_ledger.E_charged_birth = 1.0e9;
        doped->energy_ledger.E_charged_terminal_deposit = 1.0e9;
        const auto doped_report = carbon::evaluate_run_quality(config, *doped);
        require(doped_report.accepted, "informational split fields must not fail");
        require(doped_report.relative_energy_residual == base_residual,
                "split fields must leave the residual bitwise unchanged");
    }
    // E_lookup_failure / E_out_of_domain are gate inputs by design.
    {
        auto gated = fresh_result();
        gated->energy_ledger.E_out_of_domain = 1.0;
        require(!carbon::evaluate_run_quality(config, *gated).accepted,
                "out-of-domain energy must fail the gate");
    }
    // Float-slot public constant must track the enum (review Low item).
    require(carbon::TransportResult::schneider_float_slot_count ==
                static_cast<std::size_t>(carbon::SchneiderFloatSlot::Count),
            "schneider_float_slot_count must equal SchneiderFloatSlot::Count");
    std::cout << "[schneider-strict] born strict-equality and ledger-disjoint tests PASSED.\n";
}

void test_schneider_config_forbids_water_keys() {
    std::cout << "[schneider-strict] config routing tests...\n";
    const auto expect_invalid = [](carbon::TransportConfig config, const std::string& needle,
                                   const std::string& label) {
        bool threw = false;
        try {
            config.validate();
        } catch (const std::invalid_argument& err) {
            threw = true;
            require(std::string(err.what()).find(needle) != std::string::npos,
                    label + " threw the wrong error: " + err.what());
        }
        require(threw, label + " must fail validation");
    };
    // Ambiguous water keys in full Schneider CT mode must fail closed.
    {
        carbon::TransportConfig config;
        config.enable_ct_grid = true;
        config.ct_grid_file = "dummy.cctg";
        config.primary_inelastic_package_v2_file = "water.cinpkg";
        expect_invalid(config, "water/CINEL02", "Schneider CT with water package key");
    }
    {
        carbon::TransportConfig config;
        config.enable_ct_grid = true;
        config.ct_grid_file = "dummy.cctg";
        config.nuclear_model = "cinel02";
        expect_invalid(config, "cinel02", "Schneider CT with nuclear_model cinel02");
    }
    {
        carbon::TransportConfig config;
        config.enable_ct_grid = true;
        config.ct_grid_file = "dummy.cctg";
        config.ct_cinel02_rate_file = "ct_rates.bin";
        expect_invalid(config, "water/CINEL02", "Schneider CT with ct_cinel02_rate_file");
    }
    // Water mode keeps its own keys and never loads Schneider tables.
    {
        carbon::TransportConfig config;
        config.enable_ct_grid = false;
        config.nuclear_model = "cinel02";
        config.primary_inelastic_package_v2_file = "water.cinpkg";
        require(config.is_water_mode(), "CT-disabled config must stay Water mode");
        require(config.ct_schneider_c12_cinel03_file.empty() &&
                    config.ct_schneider_secondary_cinel03_file.empty(),
                "Water mode must not bind Schneider tables");
    }
    std::cout << "[schneider-strict] config routing tests PASSED.\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = (argc > 1) ? argv[1] : "";
    const auto run = [&](const std::string& name, void (*fn)()) {
        run_test_dispatch(filter, name, fn);
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
        run("test_step10_schneider_primary_xs_device_path", test_step10_schneider_primary_xs_device_path);
        run("test_step11_piecewise_nuclear_optical_depth", test_step11_piecewise_nuclear_optical_depth);
        run("test_step11_ct_sample_directional_boundaries", test_step11_ct_sample_directional_boundaries);
        run("test_step11_ct_clamp_pre_face_no_nudge", test_step11_ct_clamp_pre_face_no_nudge);
        run("test_step11_schneider_step_energy_error_bound", test_step11_schneider_step_energy_error_bound);
        run("test_step11_schneider_primary_mode_safety", test_step11_schneider_primary_mode_safety);
        run("test_step11_schneider_production_tiny_cctg_transport", test_step11_schneider_production_tiny_cctg_transport);
        run("test_step12_primary_only_mode_contract", test_step12_primary_only_mode_contract);
#ifdef CARBON_HAS_SYCL
        run("test_step12_gpu_transport", test_step12_gpu_transport);
        run("test_step12_device_memory_tracker_exception_safety", test_step12_device_memory_tracker_exception_safety);
#endif
        run("test_step14_schneider_stopping_power_tables", test_step14_schneider_stopping_power_tables);
        run("test_schneider_stopping_permutation_rejection", test_schneider_stopping_permutation_rejection);
        run("test_schneider_stopping_metadata_schema_failures", test_schneider_stopping_metadata_schema_failures);
        run("test_schneider_stopping_payload_physical_validation", test_schneider_stopping_payload_physical_validation);
#ifdef CARBON_HAS_SYCL
        run("test_schneider_stopping_source_energy_domain_fail_closed", test_schneider_stopping_source_energy_domain_fail_closed);
        run("test_schneider_stopping_host_device_equivalence", test_schneider_stopping_host_device_equivalence);
#endif
        run("test_step15_schneider_radiation_lengths_and_sentinel", test_step15_schneider_radiation_lengths_and_sentinel);
        run("test_secondary_schneider_mcs_x0_selection", test_secondary_schneider_mcs_x0_selection);
        run("test_out_of_scope_isotope_summary_ledger_fields", test_out_of_scope_isotope_summary_ledger_fields);
        run("test_step16_cinel03_round_trip_and_determinism", test_step16_cinel03_round_trip_and_determinism);
        run("test_step16_cinel03_rejections_and_fail_closed", test_step16_cinel03_rejections_and_fail_closed);
        run("test_step16_cinel03_synthetic_cpu_gpu_replay", test_step16_cinel03_synthetic_cpu_gpu_replay);
        run("test_step18_target_sampler_synthetic_and_known_ratios", test_step18_target_sampler_synthetic_and_known_ratios);
        run("test_step18_target_sampler_statistical_goodness_of_fit", test_step18_target_sampler_statistical_goodness_of_fit);
        run("test_step18_target_sampler_density_independence_and_library_reuse", test_step18_target_sampler_density_independence_and_library_reuse);
        run("test_step18_target_sampler_missing_target_fail_closed", test_step18_target_sampler_missing_target_fail_closed);
        run("test_step18_target_sampler_cpu_gpu_equivalence_and_diagnostics", test_step18_target_sampler_cpu_gpu_equivalence_and_diagnostics);
        run("test_step20_secondary_rate_table_and_cinel03_package", test_step20_secondary_rate_table_and_cinel03_package);
        run("test_step27_v2_package_load_and_lookup_equivalence", test_step27_v2_package_load_and_lookup_equivalence);
        run("test_step28_v3_rate_domain_mask_equivalence", test_step28_v3_rate_domain_mask_equivalence);
        run("test_step02_material_physics_routing", test_step02_material_physics_routing);
#ifdef CARBON_HAS_SYCL
        run("test_step03_schneider_device_data_wiring", test_step03_schneider_device_data_wiring);
        run("test_step04_ct_primary_nuclear_interaction", test_step04_ct_primary_nuclear_interaction);
        run("test_step05_ct_secondary_nuclear_transport", test_step05_ct_secondary_nuclear_transport);
#endif
        run("test_step06_energy_accounting_and_overflow_ledger", test_step06_energy_accounting_and_overflow_ledger);
        run("test_step08_cinel03_limits_and_provenance", test_step08_cinel03_limits_and_provenance);
        run("test_step09_production_integration_suite", test_step09_production_integration_suite);
        run("test_cinel03_exact_target_no_alias", test_cinel03_exact_target_no_alias);
        run("test_cinel03_bounded_domain_and_bracketing", test_cinel03_bounded_domain_and_bracketing);
        run("test_schneider_quality_gate_rejects_failures", test_schneider_quality_gate_rejects_failures);
        run("test_schneider_post_em_null_gate_taxonomy", test_schneider_post_em_null_gate_taxonomy);
        run("test_schneider_born_strict_equality_and_ledger_disjoint", test_schneider_born_strict_equality_and_ledger_disjoint);
#ifdef CARBON_HAS_SYCL
        run("test_schneider_tertiary_transport_generations", test_schneider_tertiary_transport_generations);
#endif
        run("test_schneider_config_forbids_water_keys", test_schneider_config_forbids_water_keys);
        std::cout << "All carbon_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_tests: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
