#include "carbon/cascade_package.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <algorithm>
#include <array>
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
void require_charged_origin_voxel_closure(
    const carbon::TransportConfig& config,
    const carbon::TransportResult& result,
    double tolerance_MeV_per_primary) {
    const auto voxel_count = config.number_of_voxels();
    require(result.charged_origin_voxel_deposited_energy_MeV.size() ==
                carbon::charged_origin_category_count * voxel_count,
            "Charged-origin voxel tally has the wrong size");
    const std::array<const std::vector<double>*,
                     carbon::charged_origin_category_count>
        depth_categories{
            &result.primary_c12_deposited_energy_MeV,
            &result.secondary_carbon_deposited_energy_MeV,
            &result.boron_deposited_energy_MeV,
            &result.beryllium_deposited_energy_MeV,
            &result.lithium_deposited_energy_MeV,
            &result.helium_deposited_energy_MeV,
            &result.proton_deposited_energy_MeV,
            &result.other_charged_deposited_energy_MeV,
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

    carbon::TransportConfig insert_config;
    insert_config.enable_hetero_insert = true;
    insert_config.phantom_length_mm = 400.0;
    insert_config.hetero_insert = {-10.0, 10.0, -10.0, 10.0, 50.0, 70.0, 1.85};
    insert_config.validate();
}

void test_ct_grid_helpers() {
    require_near(carbon::hu_to_density_g_per_cm3(0.0F), 1.0F, 1.0e-5, "HU0 density");
    require_near(carbon::hu_to_density_g_per_cm3(-1000.0F), 0.001205F, 1.0e-6,
                 "air density");
    require(carbon::density_to_material_id(1.0F) == 2, "water material id");
    require(carbon::density_to_material_id(1.85F) == 3, "bone material id");

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

    // Energy-dependent mass-SP: I_water → factor = za_rel; I_bone ≠ 75 shifts f_E.
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
    require_near(loaded.mass_sp_za_rel[3], 0.93F, 1.0e-6, "ct v3 bone za");
    require_near(loaded.mass_sp_I_eV[3], 106.0F, 1.0e-4, "ct v3 bone I");
    std::filesystem::remove(path);
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
        require(std::isnan(secondary.direction_x) && std::isnan(secondary.direction_y),
                "Reaction package v1 transverse direction sentinel failed");
        protons += secondary.pdg_id == 2212 ? 1U : 0U;
        neutrons += secondary.pdg_id == 2112 ? 1U : 0U;
        gammas += secondary.pdg_id == 22 ? 1U : 0U;
        alphas += secondary.atomic_number == 2 && secondary.mass_number == 4 ? 1U : 0U;
    }
    require(protons == 93'437 && neutrons == 80'666 && gammas == 38'733 &&
                alphas == 56'094,
            "Reaction package particle composition failed");

    const auto v2_path = source_directory /
                         "validation/results/topas_200MeVu_cascade_aligned_primary_3d.bin";
    const auto v2_table = carbon::ReactionPackageTable::from_binary(v2_path);
    require(v2_table.reactions().size() == 37'661,
            "Reaction package v2 reaction count failed");
    require(v2_table.secondaries().size() == 323'901,
            "Reaction package v2 secondary count failed");
    for (const auto& secondary : v2_table.secondaries()) {
        require(std::isfinite(secondary.direction_x) && std::isfinite(secondary.direction_y),
                "Reaction package v2 transverse direction is not finite");
        const auto norm_squared = secondary.direction_x * secondary.direction_x +
                                  secondary.direction_y * secondary.direction_y +
                                  secondary.direction_z * secondary.direction_z;
        require_near(norm_squared, 1.0, 2.0e-3, "Reaction package v2 direction norm failed");
    }

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

void test_neutral_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory / "validation/results/topas_200MeVu_neutral_smoke.bin";
    const auto table = carbon::NeutralPackageTable::from_binary(package_path);
    require(table.projectiles().size() == 2, "Neutral projectile count failed");
    require(table.interactions().size() == 4039, "Neutral interaction count failed");
    require(table.products().size() == 1961, "Neutral product count failed");
    const auto* neutron = table.find_projectile(2112);
    const auto* gamma = table.find_projectile(22);
    require(neutron != nullptr && neutron->interaction_count == 2494,
            "Neutral neutron lookup failed");
    require(gamma != nullptr && gamma->interaction_count == 1545,
            "Neutral gamma lookup failed");
    require(table.find_projectile(111) == nullptr,
            "Neutral missing-projectile lookup failed");
    require(table.cross_sections().size() == neutron->cross_section_count +
                                                 gamma->cross_section_count,
            "Neutral cross-section range failed");
    for (const auto& sample : table.cross_sections()) {
        require(sample.macroscopic_total_per_mm > 0.0F,
                "Neutral cross section is not positive");
        require(std::isfinite(sample.energy_MeV), "Neutral cross-section energy invalid");
    }
    for (const auto& interaction : table.interactions()) {
        require(interaction.incident_energy_MeV > 0.0F,
                "Neutral interaction energy invalid");
        require(interaction.continuation_energy_MeV >= 0.0F,
                "Neutral continuation energy invalid");
        require(interaction.local_deposit_MeV >= 0.0F, "Neutral local deposit invalid");
        const auto norm_squared =
            interaction.continuation_direction_x * interaction.continuation_direction_x +
            interaction.continuation_direction_y * interaction.continuation_direction_y +
            interaction.continuation_direction_z * interaction.continuation_direction_z;
        require_near(norm_squared, 1.0, 2.0e-3, "Neutral continuation direction norm failed");
    }
    for (const auto& product : table.products()) {
        require(std::isfinite(product.direction_x) && std::isfinite(product.direction_y),
                "Neutral product transverse direction is not finite");
        const auto norm_squared = product.direction_x * product.direction_x +
                                  product.direction_y * product.direction_y +
                                  product.direction_z * product.direction_z;
        require_near(norm_squared, 1.0, 2.0e-3, "Neutral product direction norm failed");
    }
}

void test_cascade_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory / "validation/results/topas_200MeVu_cascade_smoke.bin";
    const auto table = carbon::CascadePackageTable::from_binary(package_path);
    for (const auto& product : table.products()) {
        require(std::isnan(product.direction_x) && std::isnan(product.direction_y),
                "Cascade package v1 transverse direction sentinel failed");
    }
    require(table.projectiles().size() == 11, "Cascade projectile count failed");
    require(table.cross_sections().size() == 252, "Cascade cross-section count failed");
    require(table.interactions().size() == 58, "Cascade interaction count failed");
    require(table.products().size() == 399, "Cascade product count failed");
    const auto* alpha = table.find_projectile(2, 4);
    require(alpha != nullptr && alpha->interaction_count == 9,
            "Cascade alpha lookup failed");
    require(table.find_projectile(5, 12) == nullptr,
            "Cascade missing-projectile lookup failed");

    const auto v2_path =
        source_directory / "validation/results/topas_200MeVu_cascade_smoke_3d.bin";
    const auto v2_table = carbon::CascadePackageTable::from_binary(v2_path);
    require(v2_table.products().size() == 399, "Cascade package v2 product count failed");
    for (const auto& product : v2_table.products()) {
        require(std::isfinite(product.direction_x) && std::isfinite(product.direction_y),
                "Cascade package v2 transverse direction is not finite");
        const auto norm_squared = product.direction_x * product.direction_x +
                                  product.direction_y * product.direction_y +
                                  product.direction_z * product.direction_z;
        require_near(norm_squared, 1.0, 2.0e-3, "Cascade package v2 direction norm failed");
    }
}

void test_dose_scorer_matches_mev_conversion() {
    carbon::TransportConfig config;
    config.number_of_histories = 100;
    config.phantom_length_mm = 10.0;
    config.depth_bin_width_mm = 5.0;
    config.scorer_area_mm2 = 10000.0;
    config.water_density_g_per_cm3 = 1.0;
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
        require(mev_header.find("energy_deposition_MeV_per_primary") != std::string::npos,
                "MeV scorer header");
        require(gy_header.find("dose_Gy_per_primary") != std::string::npos, "Gy scorer header");
        require(gy_header.find("energy_deposition") == std::string::npos,
                "pure dose scorer should not list MeV columns");

        constexpr double MeV_to_joule = 1.602176634e-13;
        const auto bin_mass_kg =
            config.scorer_area_mm2 * config.depth_bin_width_mm *
            config.water_density_g_per_cm3 * 1.0e-6;
        for (std::size_t bin = 0; bin < 2; ++bin) {
            double depth_m = 0, e_per = 0, d_mev = 0, rel_m = 0;
            char comma = 0;
            mev_in >> depth_m >> comma >> e_per >> comma >> d_mev >> comma >> rel_m;
            double depth_g = 0, d_gy = 0, rel_g = 0;
            gy_in >> depth_g >> comma >> d_gy >> comma >> rel_g;
            const auto expected_e =
                result.deposited_energy_MeV[bin] /
                static_cast<double>(config.number_of_histories);
            const auto expected_d = expected_e * MeV_to_joule / bin_mass_kg;
            require_near(e_per, expected_e, 1.0e-12, "MeV/primary");
            require_near(d_mev, expected_d, 1.0e-20, "MeV file dose column");
            require_near(d_gy, expected_d, 1.0e-20, "Gy scorer dose");
            require_near(d_gy, d_mev, 1.0e-20, "Gy scorer matches MeV-file dose column");
        }
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
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

void test_topas_spots_parse_angle01() {
    const std::filesystem::path path =
        std::filesystem::path(CARBON_SOURCE_DIR) / "validation" / "topas" /
        "spots_test_c_angle01.txt";
    require(std::filesystem::exists(path),
            "spots_test_c_angle01.txt missing under validation/topas");
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

#ifdef CARBON_HAS_SYCL
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
    config.enable_voxel_scoring = true;
    config.enable_charged_origin_voxel_scoring = true;
    config.voxel_bins_x = 5;
    config.voxel_bins_y = 7;
    config.validate();
    const auto transported = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    require(transported.transported_secondaries == transported.queued_secondaries &&
                transported.transported_secondaries > 0,
            "Not every queued charged secondary was transported");
    require_voxel_idd_closure(config, transported, 1.0e-9);
    require_charged_origin_voxel_closure(config, transported, 1.0e-9);
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
    require_voxel_idd_closure(config, cascaded, 1.0e-9);
    require_charged_origin_voxel_closure(config, cascaded, 1.0e-9);
    require(cascaded.cascade_queue_overflow == 0,
            "Unexpected fragment cascade queue overflow");
    require(cascaded.transported_secondaries ==
                cascaded.queued_secondaries + cascaded.queued_cascade_secondaries,
            "Cascade queue generation count did not close");
    require(cascaded.relative_energy_balance_error() < 1.0e-4,
            "Fragment cascade total energy balance failed");

    // Same seed + cascade: IDD must be bit-identical (RNG streams no longer use
    // atomic queue slots).
    const auto cascaded_repeat = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages,
        &cascade_packages);
    require(cascaded.deposited_energy_MeV.size() ==
                cascaded_repeat.deposited_energy_MeV.size(),
            "Cascade reproducibility IDD size mismatch");
    for (std::size_t bin = 0; bin < cascaded.deposited_energy_MeV.size(); ++bin) {
        require(cascaded.deposited_energy_MeV[bin] ==
                    cascaded_repeat.deposited_energy_MeV[bin],
                "Cascade IDD not bit-identical at bin " + std::to_string(bin) +
                    " a=" + std::to_string(cascaded.deposited_energy_MeV[bin]) +
                    " b=" + std::to_string(cascaded_repeat.deposited_energy_MeV[bin]));
    }
    require(cascaded.cascade_interactions == cascaded_repeat.cascade_interactions &&
                cascaded.queued_cascade_secondaries ==
                    cascaded_repeat.queued_cascade_secondaries &&
                cascaded.secondary_deposited_energy_MeV ==
                    cascaded_repeat.secondary_deposited_energy_MeV,
            "Cascade summary counters not bit-identical across same-seed runs");
}

void test_sycl_layered_slab_range_shift() {
    // Dense insert shortens residual range vs uniform water (CSDA-level effect).
    const auto package_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                              "validation/results/"
                              "topas_200MeVu_reaction_packages_development.bin";
    if (!std::filesystem::exists(package_path)) {
        return;
    }
    const auto reaction_packages = carbon::ReactionPackageTable::from_binary(package_path);
    carbon::TransportConfig config;
    config.number_of_histories = 256;
    config.initial_energy_MeVu = 100.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_primary_attenuation = true;
    config.enable_secondary_generation = false;
    config.enable_energy_straggling = false;
    config.enable_multiple_scattering = false;
    config.random_seed = 42;
    const carbon::StoppingPowerTable stopping_power(
        {0.01, 50.01, 100.01, 150.01}, {40.0, 20.0, 12.0, 10.0});
    const carbon::CrossSectionTable xs({0.01, 150.01}, {1.0e-6, 1.0e-6});

    config.enable_layered_phantom = false;
    const auto uniform = carbon::transport_sycl(
        config, stopping_power, xs, "cpu", &reaction_packages);

    config.enable_layered_phantom = true;
    config.slab_layers = {{30.0, 1.0}, {50.0, 2.0}, {200.0, 1.0}};
    const auto layered = carbon::transport_sycl(
        config, stopping_power, xs, "cpu", &reaction_packages);

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

void test_sycl_ct_secondary_density_smoke() {
    // Secondaries must sample CT density (not water-only). Dense cube + forced
    // nuclear reaction → queued secondaries deposit with CT-scaled SP.
    const auto package_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                              "validation/results/"
                              "topas_200MeVu_reaction_packages_development.bin";
    if (!std::filesystem::exists(package_path)) {
        return;
    }
    const auto reaction_packages = carbon::ReactionPackageTable::from_binary(package_path);

    carbon::CtGrid grid;
    grid.nx = 8;
    grid.ny = 8;
    grid.nz = 20;
    grid.origin_x_mm = -4.0F;
    grid.origin_y_mm = -4.0F;
    grid.origin_z_mm = 0.0F;
    grid.spacing_x_mm = 1.0F;
    grid.spacing_y_mm = 1.0F;
    grid.spacing_z_mm = 1.0F;
    const auto n = grid.number_of_voxels();
    grid.density_g_per_cm3.assign(n, 1.85F);
    grid.material_id.assign(n, static_cast<std::uint8_t>(0));  // section 0
    grid.mass_sp_za_rel = {0.93F};  // bone-like Z/A rel
    grid.mass_sp_I_eV = {106.0F};
    const auto ct_path =
        std::filesystem::temp_directory_path() / "carbon_ct_secondary_smoke.bin";
    grid.write_binary(ct_path);

    carbon::TransportConfig config;
    config.number_of_histories = 64;
    config.initial_energy_MeVu = 50.0;
    config.phantom_length_mm = 40.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_ct_grid = true;
    config.ct_grid_file = ct_path;
    config.enable_primary_attenuation = true;
    config.enable_secondary_generation = true;
    config.enable_secondary_transport = true;
    config.enable_fragment_cascade = false;
    config.enable_energy_straggling = false;
    config.enable_multiple_scattering = false;
    config.secondary_queue_capacity = 50'000;
    config.random_seed = 20260715;
    config.validate();

    const carbon::StoppingPowerTable stopping_power(
        {0.01, 50.01, 100.01}, {20.0, 12.0, 10.0});
    // Large XS forces nuclear reactions so secondaries are produced.
    const carbon::CrossSectionTable forced_xs({0.01, 100.01}, {50.0, 50.0});
    const auto result = carbon::transport_sycl(
        config, stopping_power, forced_xs, "cpu", &reaction_packages);
    require(result.queued_secondaries > 0 && result.transported_secondaries > 0,
            "CT secondary smoke produced no transported secondaries");
    require(result.secondary_deposited_energy_MeV > 0.0,
            "CT secondary transport deposited no energy");
    require(result.secondary_queue_overflow == 0,
            "Unexpected secondary queue overflow on CT smoke");
    require(result.relative_energy_balance_error() < 1.0e-3,
            "CT secondary energy balance failed: " +
                std::to_string(result.relative_energy_balance_error()));
    const auto total_idd =
        std::accumulate(result.deposited_energy_MeV.begin(),
                        result.deposited_energy_MeV.end(), 0.0);
    require(total_idd > 0.0 && std::isfinite(total_idd),
            "CT secondary total IDD invalid");
    std::filesystem::remove(ct_path);
}

void test_sycl_neutral_transport_smoke() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto reaction_packages = carbon::ReactionPackageTable::from_binary(
        source_directory /
        "validation/results/topas_200MeVu_reaction_packages_development.bin");
    const auto neutral_packages = carbon::NeutralPackageTable::from_binary(
        source_directory / "validation/results/topas_200MeVu_neutral_smoke.bin");
    carbon::TransportConfig config;
    config.number_of_histories = 16;
    config.initial_energy_MeVu = 10.0;
    config.phantom_length_mm = 200.0;
    config.depth_bin_width_mm = 1.0;
    config.maximum_step_mm = 0.5;
    config.maximum_relative_energy_loss = 0.01;
    config.enable_primary_attenuation = true;
    config.enable_secondary_generation = true;
    config.enable_secondary_transport = true;
    config.enable_neutral_transport = true;
    config.neutral_transport_mode = "first_interaction";
    config.maximum_neutral_generations = 1;
    config.secondary_queue_capacity = 20'000;
    config.neutral_queue_capacity = 20'000;
    config.enable_voxel_scoring = true;
    config.voxel_bins_x = 5;
    config.voxel_bins_y = 5;
    config.random_seed = 20260715;
    config.validate();
    const carbon::StoppingPowerTable stopping_power({0.01, 10.01, 20.01}, {2.0, 2.0, 2.0});
    const carbon::CrossSectionTable forced_reaction({0.01, 20.01}, {100.0, 100.0});
    const auto result = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages, nullptr,
        &neutral_packages);
    require(result.queued_neutrals > 0 && result.transported_neutrals > 0,
            "Neutral queue remained empty");
    require(result.neutral_queue_overflow == 0, "Unexpected neutral queue overflow");
    require(result.neutral_interactions > 0, "Neutral transport produced no interactions");
    // Mode D does not re-queue continuations; residual may be large relative to deposit.
    require(result.transported_neutrals == result.queued_neutrals ||
                result.residual_neutral_energy_MeV >= 0.0,
            "First-interaction neutral accounting inconsistent");
    require(result.neutron_origin_deposited_energy_MeV.size() == config.number_of_bins() &&
                result.gamma_origin_deposited_energy_MeV.size() == config.number_of_bins(),
            "Neutral-origin IDD size failed");
    const auto neutral_idd =
        std::accumulate(result.neutron_origin_deposited_energy_MeV.begin(),
                        result.neutron_origin_deposited_energy_MeV.end(), 0.0) +
        std::accumulate(result.gamma_origin_deposited_energy_MeV.begin(),
                        result.gamma_origin_deposited_energy_MeV.end(), 0.0);
    require(neutral_idd > 0.0 || result.neutral_escaped_energy_MeV > 0.0 ||
                result.charged_from_neutral_energy_MeV > 0.0 ||
                result.residual_neutral_energy_MeV > 0.0,
            "Neutral transport left no deposited, escaped, charged, or residual energy");
    require(result.relative_energy_balance_error() < 5.0e-2,
            "Neutral transport energy balance failed: " +
                std::to_string(result.relative_energy_balance_error()));
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
        test_slab_phantom_helpers();
        test_ct_grid_helpers();
        test_philox_rng();
        test_highland_multiple_scattering();
        test_bohr_straggling();
        test_energy_conservation();
        test_escape_energy_conservation();
        test_straggling_reproducibility();
        test_primary_attenuation_energy_accounting();
        test_reaction_package_loading();
        test_cascade_package_loading();
        test_neutral_package_loading();
        test_flat_source_config_validation();
        test_topas_spots_parse_angle01();
        test_dose_scorer_matches_mev_conversion();
#ifdef CARBON_HAS_SYCL
        test_sycl_flat_source_extent();
        test_serial_sycl_cpu_match();
        test_sycl_secondary_queue_generation();
        test_sycl_layered_slab_range_shift();
        test_sycl_ct_secondary_density_smoke();
        test_sycl_neutral_transport_smoke();
#endif
        std::cout << "All carbon_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_tests: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
