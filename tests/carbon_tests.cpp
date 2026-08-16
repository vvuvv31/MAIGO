#include "carbon/cascade_package.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/minibeam_collimator.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/rng.hpp"

#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

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
    require_near(carbon::clamp_sampled_energy_loss(1.0, 10.0, 1.0, 100.0), 2.0,
                 1.0e-12, "Two-mean fluctuation clamp failed");
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
        source_directory /
        "data/packages/topas_400MeVu_water_100k_primary_3d.bin";
    const auto table = carbon::ReactionPackageTable::from_binary(package_path);
    require(table.energy_bins().size() == 101, "Reaction package energy-bin count failed");
    require(table.reactions().size() == 73'871, "Reaction package reaction count failed");
    require(table.secondaries().size() == 760'939,
            "Reaction package secondary count failed");
    require_near(table.minimum_energy_MeV_per_u(), 0.0, 1.0e-7,
                 "Reaction package minimum energy failed");
    require_near(table.energy_bin_width_MeV_per_u(), 4.0, 1.0e-7,
                 "Reaction package energy-bin width failed");
    require(table.energy_bin_index(-1.0F) == 0,
            "Reaction package low-energy clamp failed");
    require(table.energy_bin_index(200.0F) == 50,
            "Reaction package exact energy-bin lookup failed");
    require(table.energy_bin_index(500.0F) == 100,
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
        require(std::isfinite(reaction.local_deposit_MeV) &&
                    reaction.local_deposit_MeV >= 0.0F,
                "Reaction package local deposit is invalid");
        secondaries_from_reactions += reaction.secondary_count;
        empty_reactions += reaction.secondary_count == 0 ? 1U : 0U;
    }
    require(secondaries_from_reactions == table.secondaries().size(),
            "Reaction package secondary closure failed");
    require(empty_reactions < table.reactions().size(),
            "Reaction package unexpectedly contains only empty reactions");

    std::size_t protons = 0;
    std::size_t neutrons = 0;
    std::size_t gammas = 0;
    std::size_t alphas = 0;
    for (const auto& secondary : table.secondaries()) {
        require(std::isfinite(secondary.direction_x) && std::isfinite(secondary.direction_y),
                "Reaction package transverse direction is not finite");
        protons += secondary.pdg_id == 2212 ? 1U : 0U;
        neutrons += secondary.pdg_id == 2112 ? 1U : 0U;
        gammas += secondary.pdg_id == 22 ? 1U : 0U;
        alphas += secondary.atomic_number == 2 && secondary.mass_number == 4 ? 1U : 0U;
    }
    require(protons > 0 && neutrons > 0 && gammas > 0 && alphas > 0,
            "Reaction package is missing a major secondary species");

    const auto legacy_path =
        std::filesystem::temp_directory_path() / "carbon_legacy_reaction_package.bin";
    {
        std::array<char, 64> header{};
        std::ifstream source(package_path, std::ios::binary);
        source.read(header.data(), static_cast<std::streamsize>(header.size()));
        const std::uint32_t legacy_record_size = 16;
        std::memcpy(header.data() + 20, &legacy_record_size, sizeof(legacy_record_size));
        std::ofstream legacy(legacy_path, std::ios::binary | std::ios::trunc);
        legacy.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    require_throws(
        [&legacy_path]() { (void)carbon::ReactionPackageTable::from_binary(legacy_path); },
        "Legacy reaction package layout was accepted");
    std::filesystem::remove(legacy_path);

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

    const std::array<const char*, 4> material_packages{
        "topas_400MeVu_soft_tissue_inclxx_100k_primary_3d.bin",
        "topas_400MeVu_schneider_section07_inclxx_100k_primary_3d.bin",
        "topas_400MeVu_lung_inclxx_100k_primary_3d.bin",
        "topas_400MeVu_bone_inclxx_100k_primary_3d.bin",
    };
    for (const auto* filename : material_packages) {
        const auto material_table = carbon::ReactionPackageTable::from_binary(
            source_directory / "data/packages" / filename);
        require(!material_table.reactions().empty() &&
                    !material_table.secondaries().empty(),
                std::string("Current reaction package failed to load: ") + filename);
    }
}

void test_neutral_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory / "data/packages/topas_200MeVu_neutral_development.bin";
    if (!std::filesystem::exists(package_path) ||
        std::filesystem::file_size(package_path) < 1024) {
        std::cout << "SKIP neutral package loading: no Geant4 11.3.2 neutral fixture\n";
        return;
    }
    const auto table = carbon::NeutralPackageTable::from_binary(package_path);
    require(table.projectiles().size() == 2, "Neutral projectile count failed");
    require(table.interactions().size() > 100'000, "Neutral interaction table is too small");
    require(!table.products().empty(), "Neutral product table is empty");
    const auto* neutron = table.find_projectile(2112);
    const auto* gamma = table.find_projectile(22);
    require(neutron != nullptr && neutron->interaction_count > 0,
            "Neutral neutron lookup failed");
    require(gamma != nullptr && gamma->interaction_count > 0,
            "Neutral gamma lookup failed");
    require(neutron->interaction_count + gamma->interaction_count ==
                table.interactions().size(),
            "Neutral projectile interaction ranges do not close");
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

void test_neutral_cross_section_loading() {
    const auto path = std::filesystem::temp_directory_path() /
                      "carbon_neutral_cross_sections.csv";
    {
        std::ofstream output(path, std::ios::trunc);
        output << "pdg_id,energy_MeV,macroscopic_total_per_mm,"
                  "mean_free_path_mm\n"
               << "22,0.1,0.4,2.5\n"
               << "22,1.0,0.05,20\n"
               << "2112,0.1,0.03,33.333333\n"
               << "2112,1.0,0.02,50\n";
    }
    const auto table = carbon::NeutralCrossSectionTables::from_csv(path);
    require(table.energy_grid_size() == 2,
            "Neutral cross-section grid size failed");
    require(table.values().size() == 4,
            "Neutral cross-section value count failed");
    require_near(table.values()[0], 0.4, 1.0e-7,
                 "Neutral gamma cross section failed");
    require_near(table.values()[3], 0.02, 1.0e-7,
                 "Neutral neutron cross section failed");
    require_near(
        table.log_energy_step(), std::log(10.0), 1.0e-6,
        "Neutral cross-section logarithmic grid failed");
    std::filesystem::remove(path);
}

void test_cascade_package_loading() {
    const auto source_directory = std::filesystem::path(CARBON_SOURCE_DIR);
    const auto package_path =
        source_directory /
        "data/packages/topas_400MeVu_water_100k_cascade_3d.bin";
    const auto table = carbon::CascadePackageTable::from_binary(package_path);
    for (const auto& product : table.products()) {
        require(std::isfinite(product.direction_x) && std::isfinite(product.direction_y),
                "Cascade package transverse direction is not finite");
    }
    require(table.projectiles().size() == 31, "Cascade projectile count failed");
    require(table.cross_sections().size() == 24'936, "Cascade cross-section count failed");
    require(table.interactions().size() == 289'712, "Cascade interaction count failed");
    require(table.products().size() == 2'065'922, "Cascade product count failed");
    const auto* alpha = table.find_projectile(2, 4);
    require(alpha != nullptr && alpha->interaction_count > 0,
            "Cascade alpha lookup failed");
    require(table.find_projectile(99, 999) == nullptr,
            "Cascade missing-projectile lookup failed");
    for (const auto& interaction : table.interactions()) {
        require(std::isfinite(interaction.local_deposit_MeV) &&
                    interaction.local_deposit_MeV >= 0.0F,
                "Cascade package local deposit is invalid");
    }

    const auto legacy_path =
        std::filesystem::temp_directory_path() / "carbon_legacy_cascade_package.bin";
    {
        std::array<char, 72> header{};
        std::ifstream source(package_path, std::ios::binary);
        source.read(header.data(), static_cast<std::streamsize>(header.size()));
        const std::uint32_t legacy_interaction_size = 12;
        const std::uint32_t legacy_product_size = 16;
        std::memcpy(header.data() + 24, &legacy_interaction_size,
                    sizeof(legacy_interaction_size));
        std::memcpy(header.data() + 28, &legacy_product_size,
                    sizeof(legacy_product_size));
        std::ofstream legacy(legacy_path, std::ios::binary | std::ios::trunc);
        legacy.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    require_throws(
        [&legacy_path]() { (void)carbon::CascadePackageTable::from_binary(legacy_path); },
        "Legacy cascade package layout was accepted");
    std::filesystem::remove(legacy_path);

    const std::array<const char*, 4> material_packages{
        "topas_400MeVu_soft_tissue_inclxx_100k_cascade_3d.bin",
        "topas_400MeVu_schneider_section07_inclxx_100k_cascade_3d.bin",
        "topas_400MeVu_lung_inclxx_100k_cascade_3d.bin",
        "topas_400MeVu_bone_inclxx_100k_cascade_3d.bin",
    };
    for (const auto* filename : material_packages) {
        const auto material_table = carbon::CascadePackageTable::from_binary(
            source_directory / "data/packages" / filename);
        require(!material_table.interactions().empty() &&
                    !material_table.products().empty(),
                std::string("Current cascade package failed to load: ") + filename);
    }
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
    config.enable_secondary_generation = true;
    config.enable_secondary_transport = true;
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
    const auto mhd = dir / "origin_primary_c12.mhd";
    const auto raw = dir / "origin_primary_c12.raw";
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
    require(text.find("DoseOriginCategory = primary_c12") != std::string::npos,
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
    config.minibeam_copper_enable_reaction_products = true;
    require_throws([&config] { config.validate(); },
                   "Copper reaction products should require a package");
    config.minibeam_copper_reaction_package_file = "copper_reactions.bin";
    config.minibeam_copper_ion_stopping_power_file = "copper_ions.csv";
    config.minibeam_copper_ion_cross_section_file = "copper_ion_xs.csv";
    config.enable_secondary_generation = true;
    config.enable_secondary_transport = true;
    config.enable_primary_attenuation = true;
    config.reaction_package_file = "water_reactions.bin";
    config.validate();
    config.enable_neutral_transport = true;
    require_throws([&config] { config.validate(); },
                   "Copper neutral transport should require a Copper XS table");
    config.minibeam_copper_neutral_cross_section_file =
        "copper_neutral_xs.csv";
    require_throws([&config] { config.validate(); },
                   "Copper neutral transport should require a Copper package");
    config.minibeam_copper_neutral_package_file =
        "copper_neutral_packages.bin";
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

void test_secondary_optimization_config_validation() {
    carbon::TransportConfig config;
    config.enable_secondary_generation = true;
    config.enable_secondary_transport = true;
    config.enable_secondary_energy_sorting = true;
    config.secondary_local_deposit_cutoff_MeV = 1.0;
    config.enable_fragment_species_scoring = false;
    config.output_file.clear();
    config.dose_output_file.clear();
    config.fragment_species_output_file.clear();
    config.fragment_species_dose_output_file.clear();
    config.validate();

    auto bad_sort = config;
    bad_sort.enable_secondary_transport = false;
    require_throws([&bad_sort] { bad_sort.validate(); },
                   "Secondary energy sorting should require secondary transport");

    auto bad_cutoff = config;
    bad_cutoff.secondary_local_deposit_cutoff_MeV = -1.0;
    require_throws([&bad_cutoff] { bad_cutoff.validate(); },
                   "Secondary local-deposit cutoff should be non-negative");

    auto bad_output = config;
    bad_output.output_file = "total_idd_would_be_incomplete.csv";
    require_throws([&bad_output] { bad_output.validate(); },
                   "Disabling fragment scoring should reject total IDD output");

    carbon::TransportConfig numeric;
    numeric.enable_primary_attenuation = true;
    numeric.enable_secondary_generation = true;
    numeric.enable_secondary_transport = true;
    numeric.enable_fragment_cascade = true;
    numeric.maximum_cascade_generations = 2;
    numeric.validate();
    require_near(numeric.effective_secondary_local_deposit_cutoff_MeV(),
                 numeric.energy_cutoff_MeV, 1.0e-12,
                 "Unset secondary cutoff must inherit the primary energy cutoff");

    auto numeric_ct = numeric;
    numeric_ct.enable_ct_grid = true;
    numeric_ct.ct_grid_file = "synthetic-fast-profile-grid.bin";
    numeric_ct.validate();

    auto dose_let = numeric;
    dose_let.maximum_step_mm = 0.25;
    dose_let.maximum_relative_energy_loss = 0.0025;
    dose_let.energy_cutoff_MeV = 0.1;
    dose_let.secondary_local_deposit_cutoff_MeV = 0.5;
    dose_let.secondary_condensed_step_mm = 0.5;
    dose_let.enable_let_scoring = true;
    dose_let.use_particle_specific_stopping_power = true;
    dose_let.validate();
    require_near(dose_let.effective_secondary_local_deposit_cutoff_MeV(),
                 0.5, 1.0e-12,
                 "Configured 0.5 MeV secondary cutoff must be used as written");

    auto inherit_cutoff = dose_let;
    inherit_cutoff.secondary_local_deposit_cutoff_MeV = 0.0;
    require_near(
        inherit_cutoff.effective_secondary_local_deposit_cutoff_MeV(),
        inherit_cutoff.energy_cutoff_MeV, 1.0e-12,
        "Zero secondary cutoff must inherit the primary energy cutoff");

    auto coarse_let = dose_let;
    coarse_let.maximum_step_mm = 0.5;
    coarse_let.validate();

    auto let_off = dose_let;
    let_off.enable_let_scoring = false;
    let_off.validate();

    const auto config_root =
        std::filesystem::path(CARBON_SOURCE_DIR) / "config";
    const auto best_config = carbon::load_config(
        config_root / "beam_ct_fullplan_rt07575_let_soft_tissue.yaml");
    require(best_config.enable_let_scoring,
            "RT07575 production config must score LET");
    require(best_config.spots_enable_upstream_air_energy_loss &&
                best_config.spots_upstream_air_stopping_power_file ==
                    "data/stopping_power_air_geant4_11_3_2.csv" &&
                best_config.ct_grid_file ==
                    "benchmark/ct/grids/patient_ct_tps_90_xneg_edge_corrected.bin" &&
                best_config.spots_ct_axis_min_mm == -104.25,
            "RT07575 production upstream-air/corrected-origin contract");
    require(best_config.nuclear_residual_heat_mfp_mm == 0.5,
            "RT07575 nuclear residual heat MFP production contract");
    require(best_config.nuclear_residual_heat_scale == 0.9,
            "RT07575 nuclear residual heat scale production contract");
    auto bad_residual_scale = best_config;
    bad_residual_scale.nuclear_residual_heat_scale = -0.1;
    require_throws([&bad_residual_scale] { bad_residual_scale.validate(); },
                   "nuclear_residual_heat_scale must reject negatives");
    bad_residual_scale.nuclear_residual_heat_scale = 2.5;
    // light-ion forward mix bounds (default 0)
    require(best_config.reaction_light_ion_forward_mix == 0.0,
            "production reaction_light_ion_forward_mix default 0");
    auto bad_forward_mix = best_config;
    bad_forward_mix.reaction_light_ion_forward_mix = -0.1;
    require_throws([&bad_forward_mix] { bad_forward_mix.validate(); },
                   "reaction_light_ion_forward_mix must reject negatives");
    bad_forward_mix.reaction_light_ion_forward_mix = 1.5;
    require_throws([&bad_forward_mix] { bad_forward_mix.validate(); },
                   "reaction_light_ion_forward_mix must reject values > 1");
    require_throws([&bad_residual_scale] { bad_residual_scale.validate(); },
                   "nuclear_residual_heat_scale must reject values > 2");

    carbon::TransportConfig upstream_air;
    require(upstream_air.reaction_package_file ==
                "data/packages/topas_400MeVu_water_100k_primary_3d.bin" &&
                upstream_air.cascade_package_file ==
                    "data/packages/topas_400MeVu_water_100k_cascade_3d.bin" &&
                upstream_air.neutral_package_file ==
                    "data/packages/topas_200MeVu_neutral_development.bin",
            "Default physics package paths must use the production datasets");
    require(!upstream_air.spots_enable_upstream_air_energy_loss,
            "Upstream air energy loss must default to disabled");
    upstream_air.spots_enable_upstream_air_energy_loss = true;
    upstream_air.spots_geometry_mode = "tps_90";
    require_throws([&upstream_air] { upstream_air.validate(); },
                   "Enabled upstream air loss accepted an empty stopping table path");
    upstream_air.spots_upstream_air_stopping_power_file = "air.csv";
    upstream_air.spots_geometry_mode = "topas";
    require_throws([&upstream_air] { upstream_air.validate(); },
                   "Upstream air loss accepted a non-TPS spot geometry");
    upstream_air.spots_geometry_mode = "tps_gantry_y";
    upstream_air.validate();
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
    config.tps_particle_type = "carbon";
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
    require(disabled.voxel_scorer_clamps_transport,
            "voxel scorer transport clamp must remain compatibility default");
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
               << "enable_primary_attenuation: true\n"
               << "enable_secondary_generation: true\n"
               << "enable_secondary_transport: true\n"
               << "enable_fragment_cascade: true\n"
               << "maximum_cascade_generations: 2\n"
               << "maximum_step_mm: 0.1\n"
               << "maximum_relative_energy_loss: 0.001\n"
               << "energy_cutoff_MeV: 0.1\n"
               << "secondary_local_deposit_cutoff_MeV: 0.1\n"
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
        config.initial_energy_MeVu * static_cast<double>(config.mass_number);
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
        result.primary_c12_letd_denominator.begin(),
        result.primary_c12_letd_denominator.end(), 0.0);
    const auto all_denominator = std::accumulate(
        result.all_hadron_letd_denominator.begin(),
        result.all_hadron_letd_denominator.end(), 0.0);
    const auto primary_numerator = std::accumulate(
        result.primary_c12_letd_numerator.begin(),
        result.primary_c12_letd_numerator.end(), 0.0);
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
        config, stopping_power, cross_section, "cpu", nullptr, nullptr, nullptr, &context);
    const auto second = carbon::transport_sycl(
        config, stopping_power, cross_section, "cpu", nullptr, nullptr, nullptr, &context);
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
                                         nullptr, nullptr, nullptr, &context);
        },
        "Reusable SYCL context accepted a different physics table");
}

void test_sycl_secondary_queue_generation() {
    const auto package_path = std::filesystem::path(CARBON_SOURCE_DIR) /
                              "data/packages/"
                              "topas_400MeVu_water_100k_primary_3d.bin";
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
                              "data/packages/"
                              "topas_400MeVu_water_100k_cascade_3d.bin";
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

    // Same seed + cascade: IDD must match (RNG streams no longer use atomic queue
    // slots). FP32 dose atomics can reassociate concurrent residual deposits, so
    // allow a tiny absolute band when CARBON_DOSE_FP32 is on.
    const auto cascaded_repeat = carbon::transport_sycl(
        config, stopping_power, forced_reaction, "cpu", &reaction_packages,
        &cascade_packages);
    require(cascaded.deposited_energy_MeV.size() ==
                cascaded_repeat.deposited_energy_MeV.size(),
            "Cascade reproducibility IDD size mismatch");
#if defined(CARBON_DOSE_FP32)
    constexpr double cascade_repro_tol = 1.0e-3;
#else
    constexpr double cascade_repro_tol = 0.0;
#endif
    for (std::size_t bin = 0; bin < cascaded.deposited_energy_MeV.size(); ++bin) {
        require_near(cascaded.deposited_energy_MeV[bin],
                     cascaded_repeat.deposited_energy_MeV[bin], cascade_repro_tol,
                     "Cascade IDD not reproducible at bin " + std::to_string(bin) +
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
                              "data/packages/"
                              "topas_400MeVu_water_100k_primary_3d.bin";
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
                              "data/packages/"
                              "topas_400MeVu_water_100k_primary_3d.bin";
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
    const auto neutral_path =
        source_directory / "data/packages/topas_200MeVu_neutral_development.bin";
    if (!std::filesystem::exists(neutral_path) ||
        std::filesystem::file_size(neutral_path) < 1024) {
        std::cout << "SKIP neutral SYCL smoke: no Geant4 11.3.2 neutral fixture\n";
        return;
    }
    const auto reaction_packages = carbon::ReactionPackageTable::from_binary(
        source_directory /
        "data/packages/topas_400MeVu_water_100k_primary_3d.bin");
    const auto neutral_packages = carbon::NeutralPackageTable::from_binary(neutral_path);
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
    const auto total_idd =
        std::accumulate(result.deposited_energy_MeV.begin(),
                        result.deposited_energy_MeV.end(), 0.0);
    const auto total_voxel =
        std::accumulate(result.voxel_deposited_energy_MeV.begin(),
                        result.voxel_deposited_energy_MeV.end(), 0.0);
#if defined(CARBON_DOSE_FP32)
    constexpr double neutral_voxel_rel = 1.0e-5;
#else
    constexpr double neutral_voxel_rel = 1.0e-8;
#endif
    require(std::abs(total_idd - total_voxel) <=
                neutral_voxel_rel * std::max(1.0, std::abs(total_idd)),
            "Neutral-origin dose missing from aggregate voxel scorer");
    require(result.relative_energy_balance_error() < 5.0e-2,
            "Neutral transport energy balance failed: " +
                std::to_string(result.relative_energy_balance_error()));

    // The neutral-off kerma proxy must also reach the aggregate voxel scorer;
    // historically it was written only to the 1D "other" fragment channel.
    auto kerma_config = config;
    kerma_config.enable_neutral_transport = false;
    kerma_config.neutral_local_kerma_fraction = 0.298;
    kerma_config.neutral_kerma_mean_free_path_mm = 110.0;
    kerma_config.validate();
    const auto kerma_result = carbon::transport_sycl(
        kerma_config, stopping_power, forced_reaction, "cpu", &reaction_packages);
    const auto kerma_idd =
        std::accumulate(kerma_result.deposited_energy_MeV.begin(),
                        kerma_result.deposited_energy_MeV.end(), 0.0);
    const auto kerma_voxel =
        std::accumulate(kerma_result.voxel_deposited_energy_MeV.begin(),
                        kerma_result.voxel_deposited_energy_MeV.end(), 0.0);
    require(std::abs(kerma_idd - kerma_voxel) <=
                neutral_voxel_rel * std::max(1.0, std::abs(kerma_idd)),
            "Neutral local kerma missing from aggregate voxel scorer");
    require(kerma_result.relative_energy_balance_error() < 1.0e-3,
            "Neutral local kerma counted twice in the energy balance: " +
                std::to_string(kerma_result.relative_energy_balance_error()));
}
#endif

}  // namespace

int main() {
    try {
        test_units();
        test_hu_stopping_power_lut_loading();
        test_serial_voxel_idd_closure();
        test_charged_dose_categories();
        test_interpolation();
        test_fragment_stopping_power_scale();
        test_particle_specific_stopping_power_tables();
        test_step_selection();
        test_slab_phantom_helpers();
        test_ct_grid_helpers();
        test_philox_rng();
        test_highland_multiple_scattering();
        test_bohr_straggling();
        test_condensed_total_loss_straggling();
        test_energy_dependent_straggling_scale();
        test_energy_conservation();
        test_escape_energy_conservation();
        test_straggling_reproducibility();
        test_primary_attenuation_energy_accounting();
        test_reaction_package_loading();
        test_cascade_package_loading();
        test_neutral_package_loading();
        test_neutral_cross_section_loading();
        test_flat_source_config_validation();
        test_random_seed_parsing();
        test_minibeam_absorbing_geometry();
        test_secondary_optimization_config_validation();
        test_topas_spots_parse_angle01();
        test_topas_spot_weights_and_tps_90_transform();
        test_tps_source_geometry_csv_and_switch();
        test_dose_scorer_matches_mev_conversion();
        test_dense_voxel_mhd_writer();
        test_layered_voxel_dose_uses_local_mass();
        test_dense_charged_origin_mhd_uses_local_mass();
        test_ct_aligned_mhd_offset_and_index_pairing();
#ifdef CARBON_HAS_SYCL
        test_sycl_tps_source_arbitrary_gantry_transport();
        test_sycl_legacy_cardinal_entrance_projection();
        test_sycl_primary_spot_batch();
        test_sycl_primary_let_includes_cutoff_tail();
        test_sycl_flat_source_extent();
        test_serial_sycl_cpu_match();
        test_sycl_transport_context_reuse();
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
