#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--config FILE] [--device DEVICE] [--histories N]"
                 " [--physics-profile accurate|best|medium|fast]"
                 " [--spots FILE] [--straggling-scale X] [--output FILE]"
                 " [--dose-output FILE] [--scorer-let|--no-scorer-let]"
                 " [--let-output FILE] [--plan-only] [--sequential-spots]\n"
                 "  --device DEVICE      serial | cpu | gpu | default |\n"
                 "                       cuda|nvidia | level_zero|intel|arc | opencl\n"
                 "                       (gpu respects ONEAPI_DEVICE_SELECTOR;\n"
                 "                        cuda/level_zero pin the SYCL backend)\n"
                 "  --spots FILE         TOPAS-format spots_*.txt; repeat to concatenate files\n"
                 "  --spot-weights FILE  One optimization weight per concatenated spot\n"
                 "  --histories N        With weights: total plan histories; otherwise per spot\n"
                 "  --random-seed N      Override the configured reproducible RNG seed\n"
                 "  --physics-profile P  accurate (legacy), best, medium, or fast\n"
                 "  --ct-grid FILE       Override the configured CCTG patient grid\n"
                 "  --ct-stopping-power-scale X  Override the CT mass stopping-power scale\n"
                 "  --secondary-queue-capacity N  Override charged secondary queue capacity\n"
                 "  --neutral-queue-capacity N  Override neutral queue capacity\n"
                 "  --plan-only          Parse/allocate/transform plan without transport\n"
                 "  --sequential-spots   Validation A/B: disable batched SYCL plan launch\n"
                 "  --dij-dose-threshold-gy X  Validation only: run every active spot\n"
                 "                       independently and threshold its voxel dose before\n"
                 "                       multiplying by the optimizer weight\n"
                 "  --dij-histories-per-spot N  Histories used for every active spot in\n"
                 "                       --dij-dose-threshold-gy mode\n"
                 "  --output FILE        MeV energy-deposition scorer CSV\n"
                 "  --dose-output FILE   Dose scorer CSV (total Gy); empty disables\n"
                 "  --scorer-let         Enable primary-C12 and all-hadron LET_d scoring\n"
                 "  --no-scorer-let      Disable LET_d scoring (overrides YAML scorerLET)\n"
                 "  --let-output FILE    LET_d CSV including raw numerator/denominator\n"
                 "  --voxel-dose-mhd FILE  Override dense voxel dose MHD output\n";
}

void add_vector_in_place(std::vector<double>& total, const std::vector<double>& part) {
    if (part.empty()) {
        return;
    }
    if (total.empty()) {
        total = part;
        return;
    }
    if (total.size() != part.size()) {
        throw std::runtime_error("TransportResult vector size mismatch while accumulating spots");
    }
    for (std::size_t i = 0; i < total.size(); ++i) {
        total[i] += part[i];
    }
}

void add_vector_in_place(std::vector<std::uint64_t>& total,
                         const std::vector<std::uint64_t>& part) {
    if (part.empty()) {
        return;
    }
    if (total.empty()) {
        total = part;
        return;
    }
    if (total.size() != part.size()) {
        throw std::runtime_error("TransportResult vector size mismatch while accumulating spots");
    }
    for (std::size_t i = 0; i < total.size(); ++i) {
        total[i] += part[i];
    }
}

std::vector<double> voxel_dose_Gy_per_MeV(const carbon::TransportConfig& config) {
    constexpr double MeV_to_joule = 1.602176634e-13;
    const auto count = config.number_of_voxels();
    const auto volume_mm3 = config.voxel_size_x_mm * config.voxel_size_y_mm *
                            config.depth_bin_width_mm;
    std::vector<double> density_g_per_cm3(count, config.water_density_g_per_cm3);
    if (config.enable_ct_grid) {
        const auto grid = carbon::CtGrid::from_binary(config.ct_grid_file);
        if (grid.nx != config.voxel_bins_x || grid.ny != config.voxel_bins_y ||
            grid.nz != config.number_of_bins()) {
            throw std::invalid_argument(
                "Dij threshold mode requires the voxel scorer to match the CT grid");
        }
        density_g_per_cm3.assign(grid.density_g_per_cm3.begin(),
                                 grid.density_g_per_cm3.end());
    }
    std::vector<double> factors(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto density = std::max(1.0e-6, density_g_per_cm3[i]);
        const auto mass_kg = volume_mm3 * density * 1.0e-6;
        factors[i] = config.dose_output_scale * MeV_to_joule / mass_kg;
    }
    return factors;
}

struct DijThresholdStats {
    std::uint64_t kept_voxels{0};
    std::uint64_t dropped_voxels{0};
    double kept_weighted_Gy_voxel{0.0};
    double dropped_weighted_Gy_voxel{0.0};
};

DijThresholdStats apply_weighted_dij_dose_threshold(
    carbon::TransportResult& result,
    const std::vector<double>& dose_Gy_per_MeV,
    const double threshold_Gy,
    const double spot_weight) {
    if (result.voxel_deposited_energy_MeV.size() != dose_Gy_per_MeV.size()) {
        throw std::runtime_error("Dij threshold voxel tally size mismatch");
    }
    DijThresholdStats stats;
    for (std::size_t i = 0; i < dose_Gy_per_MeV.size(); ++i) {
        auto& energy = result.voxel_deposited_energy_MeV[i];
        const auto dose_Gy = energy * dose_Gy_per_MeV[i];
        if (dose_Gy != 0.0 && dose_Gy < threshold_Gy) {
            ++stats.dropped_voxels;
            stats.dropped_weighted_Gy_voxel += dose_Gy * spot_weight;
            energy = 0.0;
        } else if (dose_Gy != 0.0) {
            ++stats.kept_voxels;
            stats.kept_weighted_Gy_voxel += dose_Gy * spot_weight;
            energy *= spot_weight;
        }
    }
    return stats;
}

void accumulate_transport_result(carbon::TransportResult& total,
                                 const carbon::TransportResult& part) {
    add_vector_in_place(total.deposited_energy_MeV, part.deposited_energy_MeV);
    add_vector_in_place(total.voxel_deposited_energy_MeV, part.voxel_deposited_energy_MeV);
    add_vector_in_place(total.charged_origin_voxel_deposited_energy_MeV,
                        part.charged_origin_voxel_deposited_energy_MeV);
    add_vector_in_place(total.neutral_origin_voxel_deposited_energy_MeV,
                        part.neutral_origin_voxel_deposited_energy_MeV);
    add_vector_in_place(total.primary_c12_deposited_energy_MeV,
                        part.primary_c12_deposited_energy_MeV);
    add_vector_in_place(total.secondary_carbon_deposited_energy_MeV,
                        part.secondary_carbon_deposited_energy_MeV);
    add_vector_in_place(total.boron_deposited_energy_MeV, part.boron_deposited_energy_MeV);
    add_vector_in_place(total.beryllium_deposited_energy_MeV,
                        part.beryllium_deposited_energy_MeV);
    add_vector_in_place(total.lithium_deposited_energy_MeV, part.lithium_deposited_energy_MeV);
    add_vector_in_place(total.helium_deposited_energy_MeV, part.helium_deposited_energy_MeV);
    add_vector_in_place(total.proton_deposited_energy_MeV, part.proton_deposited_energy_MeV);
    add_vector_in_place(total.other_charged_deposited_energy_MeV,
                        part.other_charged_deposited_energy_MeV);
    add_vector_in_place(total.primary_c12_letd_numerator,
                        part.primary_c12_letd_numerator);
    add_vector_in_place(total.primary_c12_letd_denominator,
                        part.primary_c12_letd_denominator);
    add_vector_in_place(total.all_hadron_letd_numerator,
                        part.all_hadron_letd_numerator);
    add_vector_in_place(total.all_hadron_letd_denominator,
                        part.all_hadron_letd_denominator);
    add_vector_in_place(total.charged_origin_letd_numerator,
                        part.charged_origin_letd_numerator);
    add_vector_in_place(total.charged_origin_letd_denominator,
                        part.charged_origin_letd_denominator);
    add_vector_in_place(total.light_isotope_letd_numerator,
                        part.light_isotope_letd_numerator);
    add_vector_in_place(total.light_isotope_letd_denominator,
                        part.light_isotope_letd_denominator);
    add_vector_in_place(total.birth_counts_by_generation,
                        part.birth_counts_by_generation);
    add_vector_in_place(total.birth_ke_sum_MeV_by_generation,
                        part.birth_ke_sum_MeV_by_generation);
    add_vector_in_place(total.birth_mevu_hist, part.birth_mevu_hist);
    add_vector_in_place(total.birth_depth_hist, part.birth_depth_hist);
    add_vector_in_place(total.birth_cos_hist, part.birth_cos_hist);
    add_vector_in_place(total.birth_parent_mevu_hist, part.birth_parent_mevu_hist);
    add_vector_in_place(total.birth_parent_z_hist, part.birth_parent_z_hist);
    add_vector_in_place(total.birth_parent_product_mevu_hist,
                        part.birth_parent_product_mevu_hist);
    add_vector_in_place(total.primary_c12_voxel_letd_numerator,
                        part.primary_c12_voxel_letd_numerator);
    add_vector_in_place(total.primary_c12_voxel_letd_denominator,
                        part.primary_c12_voxel_letd_denominator);
    add_vector_in_place(total.all_hadron_voxel_letd_numerator,
                        part.all_hadron_voxel_letd_numerator);
    add_vector_in_place(total.all_hadron_voxel_letd_denominator,
                        part.all_hadron_voxel_letd_denominator);
    add_vector_in_place(total.neutron_origin_deposited_energy_MeV,
                        part.neutron_origin_deposited_energy_MeV);
    add_vector_in_place(total.gamma_origin_deposited_energy_MeV,
                        part.gamma_origin_deposited_energy_MeV);

    total.initial_energy_MeV += part.initial_energy_MeV;
    total.total_deposited_energy_MeV += part.total_deposited_energy_MeV;
    total.escaped_energy_MeV += part.escaped_energy_MeV;
    total.beamline_removed_energy_MeV +=
        part.beamline_removed_energy_MeV;
    total.untracked_nuclear_energy_MeV += part.untracked_nuclear_energy_MeV;
    total.nuclear_interactions += part.nuclear_interactions;
    total.sampled_reaction_packages += part.sampled_reaction_packages;
    total.generated_direct_secondaries += part.generated_direct_secondaries;
    total.queued_secondaries += part.queued_secondaries;
    total.secondary_queue_overflow += part.secondary_queue_overflow;
    total.generated_direct_secondary_energy_MeV += part.generated_direct_secondary_energy_MeV;
    total.queued_secondary_energy_MeV += part.queued_secondary_energy_MeV;
    total.secondary_queue_overflow_energy_MeV += part.secondary_queue_overflow_energy_MeV;
    total.untransported_neutral_energy_MeV += part.untransported_neutral_energy_MeV;
    total.untransported_unsupported_charged_energy_MeV +=
        part.untransported_unsupported_charged_energy_MeV;
    total.nuclear_energy_not_in_direct_secondaries_MeV +=
        part.nuclear_energy_not_in_direct_secondaries_MeV;
    total.transported_secondaries += part.transported_secondaries;
    total.secondary_transport_steps += part.secondary_transport_steps;
    total.secondary_deposited_energy_MeV += part.secondary_deposited_energy_MeV;
    total.secondary_escaped_energy_MeV += part.secondary_escaped_energy_MeV;
    total.cascade_interactions += part.cascade_interactions;
    total.generated_cascade_products += part.generated_cascade_products;
    total.queued_cascade_secondaries += part.queued_cascade_secondaries;
    total.cascade_queue_overflow += part.cascade_queue_overflow;
    total.queued_cascade_energy_MeV += part.queued_cascade_energy_MeV;
    total.cascade_nuclear_energy_MeV += part.cascade_nuclear_energy_MeV;
    total.queued_neutrals += part.queued_neutrals;
    total.neutral_queue_overflow += part.neutral_queue_overflow;
    total.transported_neutrals += part.transported_neutrals;
    total.neutral_interactions += part.neutral_interactions;
    total.neutral_transport_steps += part.neutral_transport_steps;
    total.queued_neutral_energy_MeV += part.queued_neutral_energy_MeV;
    total.neutral_queue_overflow_energy_MeV += part.neutral_queue_overflow_energy_MeV;
    total.neutral_deposited_energy_MeV += part.neutral_deposited_energy_MeV;
    total.neutral_escaped_energy_MeV += part.neutral_escaped_energy_MeV;
    total.residual_neutral_energy_MeV += part.residual_neutral_energy_MeV;
    total.charged_from_neutral_energy_MeV += part.charged_from_neutral_energy_MeV;
    total.minibeam.enabled = total.minibeam.enabled || part.minibeam.enabled;
    total.minibeam.incident_histories += part.minibeam.incident_histories;
    total.minibeam.direct_air_slit_histories +=
        part.minibeam.direct_air_slit_histories;
    total.minibeam.copper_touched_histories +=
        part.minibeam.copper_touched_histories;
    total.minibeam.copper_nuclear_interactions +=
        part.minibeam.copper_nuclear_interactions;
    total.minibeam.copper_generated_direct_secondaries +=
        part.minibeam.copper_generated_direct_secondaries;
    total.minibeam.copper_charged_survivors +=
        part.minibeam.copper_charged_survivors;
    total.minibeam.copper_neutral_survivors +=
        part.minibeam.copper_neutral_survivors;
    total.minibeam.beamline_removed_energy_MeV +=
        part.minibeam.beamline_removed_energy_MeV;
    total.minibeam.copper_charged_survivor_energy_MeV +=
        part.minibeam.copper_charged_survivor_energy_MeV;
    total.minibeam.copper_neutral_survivor_energy_MeV +=
        part.minibeam.copper_neutral_survivor_energy_MeV;
    for (std::size_t category = 0;
         category <
         total.minibeam.copper_charged_survivors_by_species.size();
         ++category) {
        total.minibeam.copper_charged_survivors_by_species[category] +=
            part.minibeam.copper_charged_survivors_by_species[category];
        total.minibeam
            .copper_charged_survivor_energy_by_species_MeV[category] +=
            part.minibeam
                .copper_charged_survivor_energy_by_species_MeV[category];
    }
    total.minibeam.water_entrance_primary_c12 +=
        part.minibeam.water_entrance_primary_c12;
    total.minibeam.energy_sum_MeV += part.minibeam.energy_sum_MeV;
    total.minibeam.energy_squared_sum_MeV2 +=
        part.minibeam.energy_squared_sum_MeV2;
    total.minibeam.x_sum_mm += part.minibeam.x_sum_mm;
    total.minibeam.x_squared_sum_mm2 += part.minibeam.x_squared_sum_mm2;
    total.minibeam.y_sum_mm += part.minibeam.y_sum_mm;
    total.minibeam.y_squared_sum_mm2 += part.minibeam.y_squared_sum_mm2;
    total.minibeam.direction_x_sum += part.minibeam.direction_x_sum;
    total.minibeam.direction_x_squared_sum +=
        part.minibeam.direction_x_squared_sum;
    total.minibeam.direction_y_sum += part.minibeam.direction_y_sum;
    total.minibeam.direction_y_squared_sum +=
        part.minibeam.direction_y_squared_sum;
    total.total_steps += part.total_steps;
    total.elapsed_seconds += part.elapsed_seconds;
    total.primary_kernel_seconds += part.primary_kernel_seconds;
    total.secondary_kernel_seconds += part.secondary_kernel_seconds;
    total.neutral_kernel_seconds += part.neutral_kernel_seconds;
    total.charged_after_neutral_kernel_seconds +=
        part.charged_after_neutral_kernel_seconds;
    if (total.backend.empty()) {
        total.backend = part.backend;
    }
}

void apply_spot_to_config(carbon::TransportConfig& config,
                          const carbon::TopasSpotPlan& plan,
                          const carbon::TopasSpot& spot,
                          std::size_t spot_index,
                          std::uint64_t base_seed) {
    if (config.mass_number <= 0) {
        throw std::invalid_argument("mass_number must be positive for spots plans");
    }
    config.number_of_histories = spot.number_of_histories;
    config.initial_energy_MeVu = spot.energy_MeV / static_cast<double>(config.mass_number);
    // TOPAS BeamEnergySpread is percent (1.0 => 1%); GPU uses relative RMS.
    config.beam_energy_spread = spot.energy_spread_percent / 100.0;

    // A flat field is shared by every energy layer. Otherwise apply the spot's
    // TOPAS BiGaussian emittance values (zero values retain a pencil source).
    config.enable_emittance_source = !config.enable_flat_source;
    if (config.enable_emittance_source) {
        config.emittance_sigma_x_mm = spot.sigma_x_mm;
        config.emittance_sigma_y_mm = spot.sigma_y_mm;
        config.emittance_sigma_x_prime = spot.sigma_x_prime;
        config.emittance_sigma_y_prime = spot.sigma_y_prime;
        config.emittance_correlation_x = spot.correlation_x;
        config.emittance_correlation_y = spot.correlation_y;
    }

    if (config.spots_geometry_mode == "beam_plus_z") {
        // Water-IDD convenience: beam along +z from z=0; lateral offsets only.
        config.source_origin_x_mm = spot.trans_x_mm;
        config.source_origin_y_mm = spot.trans_z_mm;
        config.source_origin_z_mm = 0.0;
        config.beam_ux_x = 1.0;
        config.beam_ux_y = 0.0;
        config.beam_ux_z = 0.0;
        config.beam_uy_x = 0.0;
        config.beam_uy_y = 1.0;
        config.beam_uy_z = 0.0;
        config.beam_uz_x = 0.0;
        config.beam_uz_y = 0.0;
        config.beam_uz_z = 1.0;
    } else if (config.spots_geometry_mode == "tps_90" ||
               config.spots_geometry_mode == "tps_gantry_y") {
        // tps_zero_beam_pose_for_spot uses inverted RotX/RotY relative to the
        // raw TOPAS component rotation so the world beam matches Geant4's
        // volume orientation. The patient transform determines whether the beam
        // travels +patient-X (normal packing) or −patient-X (xneg packing).
        const auto world_pose = plan.tps_zero_beam_pose_for_spot(spot);
        auto pose =
            config.spots_geometry_mode == "tps_90"
                ? carbon::transform_tps_90_pose_to_ct(
                      world_pose, config.spots_patient_trans_x_mm,
                      config.spots_patient_trans_y_mm,
                      config.spots_patient_trans_z_mm,
                      config.spots_patient_rot_z_deg,
                      config.spots_ct_axis_min_mm)
                : carbon::transform_tps_y_pose_to_ct(
                      world_pose, config.spots_patient_trans_x_mm,
                      config.spots_patient_trans_y_mm,
                      config.spots_patient_trans_z_mm,
                      config.spots_patient_rot_z_deg,
                      config.spots_ct_axis_min_mm);
        if (pose.uz_z <= 1.0e-6) {
            throw std::runtime_error(
                "TPS spot does not point into the reoriented CT (+z)");
        }
        const auto distance_to_entrance_mm = -pose.origin_z_mm / pose.uz_z;
        if (distance_to_entrance_mm < 0.0) {
            throw std::runtime_error(
                "TPS source is downstream of the CT entrance plane");
        }
        pose.origin_x_mm += distance_to_entrance_mm * pose.uz_x;
        pose.origin_y_mm += distance_to_entrance_mm * pose.uz_y;
        pose.origin_z_mm = 0.0;

        // Propagate the source-plane emittance covariance through the air gap
        // to the CT entrance: x_entry = x + L*x' (and likewise for y).
        const auto propagate = [distance_to_entrance_mm](
                                   double& sigma, const double sigma_prime,
                                   double& correlation) {
            if (sigma <= 0.0 || sigma_prime <= 0.0) {
                return;
            }
            const auto covariance = correlation * sigma * sigma_prime;
            const auto variance = sigma * sigma +
                                  2.0 * distance_to_entrance_mm * covariance +
                                  distance_to_entrance_mm * distance_to_entrance_mm *
                                      sigma_prime * sigma_prime;
            const auto propagated_sigma = std::sqrt(std::max(0.0, variance));
            const auto propagated_covariance =
                covariance + distance_to_entrance_mm * sigma_prime * sigma_prime;
            sigma = propagated_sigma;
            correlation = propagated_sigma > 0.0
                              ? std::clamp(propagated_covariance /
                                               (propagated_sigma * sigma_prime),
                                           -1.0, 1.0)
                              : 0.0;
        };
        propagate(config.emittance_sigma_x_mm, config.emittance_sigma_x_prime,
                  config.emittance_correlation_x);
        propagate(config.emittance_sigma_y_mm, config.emittance_sigma_y_prime,
                  config.emittance_correlation_y);

        config.source_origin_x_mm = pose.origin_x_mm;
        config.source_origin_y_mm = pose.origin_y_mm;
        config.source_origin_z_mm = pose.origin_z_mm;
        config.beam_ux_x = pose.ux_x;
        config.beam_ux_y = pose.ux_y;
        config.beam_ux_z = pose.ux_z;
        config.beam_uy_x = pose.uy_x;
        config.beam_uy_y = pose.uy_y;
        config.beam_uy_z = pose.uy_z;
        config.beam_uz_x = pose.uz_x;
        config.beam_uz_y = pose.uz_y;
        config.beam_uz_z = pose.uz_z;
    } else if (config.spots_geometry_mode == "minibeam_topas_y") {
        // TOPAS reference beam travels world +Y. Preserve the source-plane
        // emittance until the upstream collimator, then map to the canonical
        // GPU frame: (x,y,z)_gpu = (X,Z,Y-Y_water_entry)_topas.
        const auto world = plan.tps_zero_beam_pose_for_spot(spot);
        config.source_origin_x_mm = world.origin_x_mm;
        config.source_origin_y_mm = world.origin_z_mm;
        config.source_origin_z_mm =
            world.origin_y_mm - config.minibeam_water_entrance_world_y_mm;
        config.beam_ux_x = world.ux_x;
        config.beam_ux_y = world.ux_z;
        config.beam_ux_z = world.ux_y;
        config.beam_uy_x = world.uy_x;
        config.beam_uy_y = world.uy_z;
        config.beam_uy_z = world.uy_y;
        config.beam_uz_x = world.uz_x;
        config.beam_uz_y = world.uz_z;
        config.beam_uz_z = world.uz_y;
    } else {
        const auto pose = plan.pose_for_spot(spot);
        config.source_origin_x_mm = pose.origin_x_mm;
        config.source_origin_y_mm = pose.origin_y_mm;
        config.source_origin_z_mm = pose.origin_z_mm;
        config.beam_ux_x = pose.ux_x;
        config.beam_ux_y = pose.ux_y;
        config.beam_ux_z = pose.ux_z;
        config.beam_uy_x = pose.uy_x;
        config.beam_uy_y = pose.uy_y;
        config.beam_uy_z = pose.uy_z;
        config.beam_uz_x = pose.uz_x;
        config.beam_uz_y = pose.uz_y;
        config.beam_uz_z = pose.uz_z;
    }

    // Independent RNG stream per spot (still deterministic given base seed).
    config.random_seed = base_seed + static_cast<std::uint64_t>(spot_index) * 1'000'003ULL;
}

carbon::PrimarySpotBatchEntry make_spot_batch_entry(
    const carbon::TransportConfig& spot_config,
    const std::uint64_t history_begin) {
    carbon::PrimarySpotBatchEntry entry{};
    entry.history_begin = history_begin;
    entry.history_end = history_begin + spot_config.number_of_histories;
    entry.random_seed = spot_config.random_seed;
    entry.initial_energy_MeV() =
        static_cast<float>(spot_config.initial_total_energy_MeV());
    entry.beam_energy_spread() = static_cast<float>(spot_config.beam_energy_spread);
    entry.emittance_sigma_x_mm() = static_cast<float>(spot_config.emittance_sigma_x_mm);
    entry.emittance_sigma_y_mm() = static_cast<float>(spot_config.emittance_sigma_y_mm);
    entry.emittance_sigma_x_prime() =
        static_cast<float>(spot_config.emittance_sigma_x_prime);
    entry.emittance_sigma_y_prime() =
        static_cast<float>(spot_config.emittance_sigma_y_prime);
    entry.emittance_correlation_x() =
        static_cast<float>(spot_config.emittance_correlation_x);
    entry.emittance_correlation_y() =
        static_cast<float>(spot_config.emittance_correlation_y);
    entry.source_origin_x_mm() = static_cast<float>(spot_config.source_origin_x_mm);
    entry.source_origin_y_mm() = static_cast<float>(spot_config.source_origin_y_mm);
    entry.source_origin_z_mm() = static_cast<float>(spot_config.source_origin_z_mm);
    entry.beam_ux_x() = static_cast<float>(spot_config.beam_ux_x);
    entry.beam_ux_y() = static_cast<float>(spot_config.beam_ux_y);
    entry.beam_ux_z() = static_cast<float>(spot_config.beam_ux_z);
    entry.beam_uy_x() = static_cast<float>(spot_config.beam_uy_x);
    entry.beam_uy_y() = static_cast<float>(spot_config.beam_uy_y);
    entry.beam_uy_z() = static_cast<float>(spot_config.beam_uy_z);
    entry.beam_uz_x() = static_cast<float>(spot_config.beam_uz_x);
    entry.beam_uz_y() = static_cast<float>(spot_config.beam_uz_y);
    entry.beam_uz_z() = static_cast<float>(spot_config.beam_uz_z);
    return entry;
}

carbon::TransportResult run_transport(
    const carbon::TransportConfig& config,
    const carbon::StoppingPowerTable& stopping_power,
    const carbon::CrossSectionTable& cross_section,
    const std::optional<carbon::ReactionPackageTable>& reaction_packages,
    const std::optional<carbon::CascadePackageTable>& cascade_packages,
    const std::optional<carbon::NeutralPackageTable>& neutral_packages,
    carbon::SyclTransportContext* sycl_context = nullptr) {
    if (config.device == "serial") {
        if (config.enable_let_scoring) {
            throw std::invalid_argument(
                "scorerLET is currently implemented only by the SYCL backend");
        }
        if (config.enable_secondary_generation) {
            throw std::invalid_argument(
                "Secondary generation is currently implemented only by the SYCL backend");
        }
        return carbon::transport_serial(config, stopping_power, cross_section);
    }
#ifdef CARBON_HAS_SYCL
    // serial is handled above. SYCL names are resolved in make_sycl_queue.
    if (config.device != "cpu" && config.device != "gpu" && config.device != "default" &&
        config.device != "cuda" && config.device != "nvidia" &&
        config.device != "level_zero" && config.device != "intel" &&
        config.device != "arc" && config.device != "opencl") {
        throw std::invalid_argument(
            "SYCL device must be cpu, gpu, default, cuda, nvidia, level_zero, intel, arc, "
            "or opencl");
    }
    return carbon::transport_sycl(config, stopping_power, cross_section, config.device,
                                  reaction_packages ? &*reaction_packages : nullptr,
                                  cascade_packages ? &*cascade_packages : nullptr,
                                  neutral_packages ? &*neutral_packages : nullptr,
                                  sycl_context);
#else
    (void)sycl_context;
    throw std::runtime_error(
        "This binary was built without SYCL. Reconfigure with CARBON_ENABLE_SYCL=ON and icpx.");
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
    // Line-buffer stdout so progress is visible when piped (tee/logs) and during
    // long CUDA kernels that would otherwise freeze WSL with no feedback.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::filesystem::path config_path{"config/beam_200MeVu.yaml"};
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config" && index + 1 < argc) {
                config_path = argv[++index];
            } else if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
        }

        auto config = carbon::load_config(config_path);
        bool histories_cli_override = false;
        bool plan_only = false;
        bool sequential_spots = false;
        bool spots_cli_override = false;
        double dij_dose_threshold_Gy = 0.0;
        std::size_t dij_histories_per_spot = 0;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config") {
                ++index;
            } else if (argument == "--device" && index + 1 < argc) {
                config.device = argv[++index];
            } else if (argument == "--histories" && index + 1 < argc) {
                config.number_of_histories = std::stoull(argv[++index]);
                histories_cli_override = true;
            } else if (argument == "--random-seed" && index + 1 < argc) {
                config.random_seed = std::stoull(argv[++index]);
            } else if (argument == "--physics-profile" && index + 1 < argc) {
                config.physics_profile = argv[++index];
            } else if (argument == "--ct-grid" && index + 1 < argc) {
                config.ct_grid_file = argv[++index];
            } else if (argument == "--ct-stopping-power-scale" && index + 1 < argc) {
                config.ct_stopping_power_scale = std::stod(argv[++index]);
            } else if (argument == "--secondary-queue-capacity" &&
                       index + 1 < argc) {
                config.secondary_queue_capacity = std::stoull(argv[++index]);
            } else if (argument == "--neutral-queue-capacity" &&
                       index + 1 < argc) {
                config.neutral_queue_capacity = std::stoull(argv[++index]);
            } else if (argument == "--spots" && index + 1 < argc) {
                if (!spots_cli_override) {
                    config.topas_spots_file.clear();
                    config.topas_spots_files.clear();
                    spots_cli_override = true;
                }
                config.topas_spots_files.emplace_back(argv[++index]);
            } else if (argument == "--spot-weights" && index + 1 < argc) {
                config.spot_weights_file = argv[++index];
            } else if (argument == "--straggling-scale" && index + 1 < argc) {
                config.straggling_scale = std::stod(argv[++index]);
            } else if (argument == "--output" && index + 1 < argc) {
                config.output_file = argv[++index];
            } else if (argument == "--dose-output" && index + 1 < argc) {
                const std::string path = argv[++index];
                config.dose_output_file =
                    path.empty() ? std::filesystem::path{} : std::filesystem::path{path};
            } else if (argument == "--scorer-let") {
                config.enable_let_scoring = true;
            } else if (argument == "--no-scorer-let") {
                config.enable_let_scoring = false;
            } else if (argument == "--let-output" && index + 1 < argc) {
                config.let_output_file = argv[++index];
            } else if (argument == "--voxel-dose-mhd" && index + 1 < argc) {
                config.voxel_dose_mhd_output_file = argv[++index];
            } else if (argument == "--plan-only") {
                plan_only = true;
            } else if (argument == "--sequential-spots") {
                sequential_spots = true;
            } else if (argument == "--dij-dose-threshold-gy" &&
                       index + 1 < argc) {
                dij_dose_threshold_Gy = std::stod(argv[++index]);
            } else if (argument == "--dij-histories-per-spot" &&
                       index + 1 < argc) {
                dij_histories_per_spot = std::stoull(argv[++index]);
            } else if (argument != "--help" && argument != "-h") {
                throw std::invalid_argument("Unknown or incomplete argument: " + argument);
            }
        }
        config.validate();
        if ((dij_dose_threshold_Gy > 0.0) != (dij_histories_per_spot > 0)) {
            throw std::invalid_argument(
                "--dij-dose-threshold-gy and --dij-histories-per-spot must be used together");
        }
        if (dij_dose_threshold_Gy > 0.0 && !std::isfinite(dij_dose_threshold_Gy)) {
            throw std::invalid_argument("Dij dose threshold must be finite and positive");
        }

        const auto stopping_power = carbon::StoppingPowerTable::from_csv(config.stopping_power_file);
        const auto cross_section =
            carbon::CrossSectionTable::from_csv(config.nuclear_cross_section_file);
        std::optional<carbon::ReactionPackageTable> reaction_packages;
        std::optional<carbon::CascadePackageTable> cascade_packages;
        std::optional<carbon::NeutralPackageTable> neutral_packages;
        if (config.enable_secondary_generation) {
            reaction_packages =
                carbon::ReactionPackageTable::from_binary(config.reaction_package_file);
            std::cout << "Reaction packages: " << reaction_packages->reactions().size()
                      << "; direct secondaries: " << reaction_packages->secondaries().size()
                      << '\n';
        }
        if (config.enable_fragment_cascade) {
            cascade_packages =
                carbon::CascadePackageTable::from_binary(config.cascade_package_file);
            std::cout << "Cascade projectiles: " << cascade_packages->projectiles().size()
                      << "; interactions: " << cascade_packages->interactions().size()
                      << "; products: " << cascade_packages->products().size() << '\n';
        }
        if (config.enable_neutral_transport) {
            neutral_packages =
                carbon::NeutralPackageTable::from_binary(config.neutral_package_file);
            std::cout << "Neutral projectiles: " << neutral_packages->projectiles().size()
                      << "; interactions: " << neutral_packages->interactions().size()
                      << "; products: " << neutral_packages->products().size() << '\n';
        }

#ifdef CARBON_HAS_SYCL
        if (config.device != "serial") {
            std::cout << "SYCL device: " << carbon::describe_sycl_device(config.device) << '\n';
        }
#endif

        std::vector<std::filesystem::path> spots_files = config.topas_spots_files;
        if (spots_files.empty() && !config.topas_spots_file.empty()) {
            spots_files.push_back(config.topas_spots_file);
        }

        carbon::SyclTransportContext* sycl_context = nullptr;
#ifdef CARBON_HAS_SYCL
        std::unique_ptr<carbon::SyclTransportContext> sycl_context_storage;
        if (config.device != "serial" &&
            (!spots_files.empty() || config.enable_tps_source)) {
            sycl_context_storage =
                std::make_unique<carbon::SyclTransportContext>(config.device);
            sycl_context = sycl_context_storage.get();
        }
#endif

        carbon::TransportResult result;
        const auto base_seed = config.random_seed;

        if (config.enable_tps_source) {
            if (config.device == "serial") {
                throw std::invalid_argument(
                    "tpsSource=true requires a SYCL device (cpu/gpu/cuda/etc.)");
            }
            if (sequential_spots) {
                throw std::invalid_argument(
                    "tpsSource=true uses the GPU primary batch and does not support "
                    "--sequential-spots");
            }
            const auto plan = carbon::TpsSourcePlan::from_config(config);
            const auto batch = plan.make_primary_batch(config);
            std::cout << "TPS source: "
                      << (config.tps_spots_file.empty()
                              ? std::string{"single YAML spot"}
                              : config.tps_spots_file.string())
                      << '\n'
                      << "  spots: " << batch.size() << "/" << plan.spots.size()
                      << " active; total histories: " << config.number_of_histories
                      << "; total MU: " << plan.total_mu << '\n'
                      << "  gantry/couch/collimator: "
                      << config.tps_gantry_angle_deg << "/"
                      << config.tps_couch_angle_deg << "/"
                      << config.tps_collimator_angle_deg << " deg; SAD: "
                      << config.tps_sad_mm << " mm; patient: "
                      << config.tps_patient_position << "; convention: "
                      << config.tps_angle_convention << '\n';
            if (plan_only) {
                double min_x = std::numeric_limits<double>::infinity();
                double max_x = -min_x;
                double min_y = min_x;
                double max_y = -min_x;
                double min_z = min_x;
                double max_z = -min_x;
                for (const auto& spot : plan.spots) {
                    if (spot.mu_weight <= 0.0) {
                        continue;
                    }
                    const auto pose = plan.pose_for_spot(config, spot);
                    min_x = std::min(min_x, pose.origin_x_mm);
                    max_x = std::max(max_x, pose.origin_x_mm);
                    min_y = std::min(min_y, pose.origin_y_mm);
                    max_y = std::max(max_y, pose.origin_y_mm);
                    min_z = std::min(min_z, pose.origin_z_mm);
                    max_z = std::max(max_z, pose.origin_z_mm);
                }
                const auto& direction = batch.front();
                std::cout << "  plan-only validation passed; source bounds x=["
                          << min_x << ", " << max_x << "] y=[" << min_y << ", "
                          << max_y << "] z=[" << min_z << ", " << max_z << "] mm\n"
                          << "  central direction: (" << direction.beam_uz_x() << ", "
                          << direction.beam_uz_y() << ", "
                          << direction.beam_uz_z() << ")\n";
                return EXIT_SUCCESS;
            }
            auto batch_config = config;
            batch_config.primary_spot_batch = batch;
            batch_config.enable_emittance_source = std::any_of(
                batch.begin(), batch.end(), [](const auto& entry) {
                    return entry.emittance_sigma_x_mm() > 0.0F ||
                           entry.emittance_sigma_y_mm() > 0.0F ||
                           entry.emittance_sigma_x_prime() > 0.0F ||
                           entry.emittance_sigma_y_prime() > 0.0F;
                });
            batch_config.validate();
            std::cout << "  batched SYCL launch: " << batch.size() << " TPS spots, "
                      << config.number_of_histories << " histories\n";
            result = run_transport(batch_config, stopping_power, cross_section,
                                   reaction_packages, cascade_packages, neutral_packages,
                                   sycl_context);
            if (!plan.spots.empty()) {
                config.initial_energy_MeVu = plan.spots.front().energy_MeVu;
            }
        } else if (!spots_files.empty()) {
            auto plan = carbon::TopasSpotPlan::from_files(spots_files);
            plan.sad_mm = config.spots_sad_mm;
            std::size_t removed_zero_weight_spots = 0;
            if (!config.spot_weights_file.empty()) {
                removed_zero_weight_spots = plan.apply_weights_from_csv(
                    config.spot_weights_file, config.number_of_histories);
            } else if (histories_cli_override) {
                for (auto& spot : plan.spots) {
                    spot.number_of_histories = config.number_of_histories;
                }
            }
            const auto threshold_dij_mode = dij_dose_threshold_Gy > 0.0;
            if (threshold_dij_mode) {
                if (config.spot_weights_file.empty()) {
                    throw std::invalid_argument(
                        "Dij dose threshold mode requires --spot-weights/YAML spot_weights_file");
                }
                sequential_spots = true;
                for (auto& spot : plan.spots) {
                    spot.number_of_histories = dij_histories_per_spot;
                }
                // These auxiliary scorers are not weighted/thresholded in this
                // diagnostic. Avoid silently writing internally inconsistent
                // files; the dense voxel dose and its reconstructed depth dose
                // below are the supported outputs.
                config.fragment_species_output_file.clear();
                config.fragment_species_dose_output_file.clear();
                config.fragment_species_let_output_file.clear();
                config.light_isotope_let_output_file.clear();
                config.fragment_birth_spectrum_output_file.clear();
                config.voxel_dose_output_file.clear();
                config.voxel_dose_Gy_output_file.clear();
            }
            const auto total_histories = plan.total_histories();
            std::cout << "TOPAS spots plan files:";
            for (const auto& path : spots_files) {
                std::cout << ' ' << path.string();
            }
            std::cout << '\n'
                      << "  spots: " << plan.spots.size()
                      << "  total histories: " << total_histories
                      << "  geometry: " << config.spots_geometry_mode
                      << "  SAD: " << plan.sad_mm << " mm\n"
                      << "  (history ranges preserve file order; no TimeFeature timeline)\n";
            if (!config.spot_weights_file.empty()) {
                std::cout << "  optimization weights: "
                          << config.spot_weights_file.string()
                          << " sum=" << plan.total_plan_weight
                          << " zero-weight spots removed=" << removed_zero_weight_spots
                          << '\n';
            }
            if (plan_only) {
                std::size_t min_histories = plan.spots.front().number_of_histories;
                std::size_t max_histories = min_histories;
                double min_entry_x = std::numeric_limits<double>::infinity();
                double max_entry_x = -min_entry_x;
                double min_entry_y = min_entry_x;
                double max_entry_y = -min_entry_x;
                double min_direction_z = min_entry_x;
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    const auto& spot = plan.spots[i];
                    min_histories = std::min(min_histories, spot.number_of_histories);
                    max_histories = std::max(max_histories, spot.number_of_histories);
                    auto transformed = config;
                    apply_spot_to_config(transformed, plan, spot, i, base_seed);
                    transformed.validate();
                    min_entry_x = std::min(min_entry_x, transformed.source_origin_x_mm);
                    max_entry_x = std::max(max_entry_x, transformed.source_origin_x_mm);
                    min_entry_y = std::min(min_entry_y, transformed.source_origin_y_mm);
                    max_entry_y = std::max(max_entry_y, transformed.source_origin_y_mm);
                    min_direction_z = std::min(min_direction_z, transformed.beam_uz_z);
                }
                std::cout << "  plan-only validation passed; histories/active spot min="
                          << min_histories << " max=" << max_histories << '\n'
                          << "  CT entrance means: x=[" << min_entry_x << ", "
                          << max_entry_x << "] mm y=[" << min_entry_y << ", "
                          << max_entry_y << "] mm; min beam dz=" << min_direction_z
                          << '\n';
                return EXIT_SUCCESS;
            }

            if (config.device != "serial" && !sequential_spots) {
                auto batch_config = config;
                batch_config.primary_spot_batch.clear();
                batch_config.primary_spot_batch.reserve(plan.spots.size());
                std::uint64_t history_begin = 0;
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    auto spot_config = config;
                    apply_spot_to_config(spot_config, plan, plan.spots[i], i, base_seed);
                    spot_config.validate();
                    auto entry = make_spot_batch_entry(spot_config, history_begin);
                    history_begin = entry.history_end;
                    batch_config.primary_spot_batch.push_back(entry);
                }
                batch_config.number_of_histories = total_histories;
                batch_config.enable_emittance_source = !batch_config.enable_flat_source;
                batch_config.validate();
                std::cout << "  batched SYCL launch: "
                          << batch_config.primary_spot_batch.size() << " spots, "
                          << total_histories << " histories\n";
                result = run_transport(batch_config, stopping_power, cross_section,
                                       reaction_packages, cascade_packages, neutral_packages,
                                       sycl_context);
            } else {
                const auto dose_Gy_per_MeV = threshold_dij_mode
                    ? voxel_dose_Gy_per_MeV(config)
                    : std::vector<double>{};
                DijThresholdStats threshold_stats;
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    const auto& spot = plan.spots[i];
                    auto spot_config = config;
                    apply_spot_to_config(spot_config, plan, spot, i, base_seed);
                    spot_config.validate();
                    auto spot_result = run_transport(
                        spot_config, stopping_power, cross_section, reaction_packages,
                        cascade_packages, neutral_packages, sycl_context);
                    if (threshold_dij_mode) {
                        const auto part = apply_weighted_dij_dose_threshold(
                            spot_result, dose_Gy_per_MeV, dij_dose_threshold_Gy,
                            spot.plan_weight);
                        threshold_stats.kept_voxels += part.kept_voxels;
                        threshold_stats.dropped_voxels += part.dropped_voxels;
                        threshold_stats.kept_weighted_Gy_voxel +=
                            part.kept_weighted_Gy_voxel;
                        threshold_stats.dropped_weighted_Gy_voxel +=
                            part.dropped_weighted_Gy_voxel;
                    }
                    if (i == 0) {
                        result = std::move(spot_result);
                    } else {
                        accumulate_transport_result(result, spot_result);
                    }
                    if (threshold_dij_mode &&
                        ((i + 1) % 25 == 0 || i + 1 == plan.spots.size())) {
                        std::cout << "  Dij threshold spots: " << (i + 1) << '/'
                                  << plan.spots.size() << "; kept/dropped voxels: "
                                  << threshold_stats.kept_voxels << '/'
                                  << threshold_stats.dropped_voxels << '\n';
                    }
                }
                if (threshold_dij_mode) {
                    result.deposited_energy_MeV.assign(config.number_of_bins(), 0.0);
                    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
                    for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
                        const auto begin = result.voxel_deposited_energy_MeV.begin() +
                                           static_cast<std::ptrdiff_t>(z * plane_size);
                        result.deposited_energy_MeV[z] = std::accumulate(
                            begin, begin + static_cast<std::ptrdiff_t>(plane_size), 0.0);
                    }
                    std::cout << std::setprecision(12)
                              << "  Dij per-spot dose threshold: "
                              << dij_dose_threshold_Gy << " Gy at "
                              << dij_histories_per_spot << " histories/spot\n"
                              << "  weighted kept/dropped integral: "
                              << threshold_stats.kept_weighted_Gy_voxel << " / "
                              << threshold_stats.dropped_weighted_Gy_voxel
                              << " Gy-voxel\n";
                }
            }
            // Output writers divide by number_of_histories → absolute MeV / total primaries.
            config.number_of_histories = total_histories;
            if (!plan.spots.empty()) {
                config.initial_energy_MeVu =
                    plan.spots.front().energy_MeV / static_cast<double>(config.mass_number);
                config.beam_energy_spread =
                    plan.spots.front().energy_spread_percent / 100.0;
            }
        } else {
            result = run_transport(config, stopping_power, cross_section, reaction_packages,
                                   cascade_packages, neutral_packages);
        }

        // MeV energy-deposition scorers (empty path disables that file).
        if (!config.output_file.empty()) {
            carbon::write_depth_dose_csv(config.output_file, config, result);
        }
        if (config.enable_secondary_transport &&
            config.enable_fragment_species_scoring &&
            !config.fragment_species_output_file.empty()) {
            carbon::write_fragment_species_csv(
                config.fragment_species_output_file, config, result);
        }
        if (config.enable_let_scoring && !config.let_output_file.empty()) {
            carbon::write_letd_csv(config.let_output_file, config, result);
        }
        if (config.enable_let_scoring &&
            !config.fragment_species_let_output_file.empty()) {
            carbon::write_fragment_species_letd_csv(
                config.fragment_species_let_output_file, config, result);
        }
        if (config.enable_let_scoring &&
            !config.light_isotope_let_output_file.empty()) {
            carbon::write_light_isotope_letd_csv(
                config.light_isotope_let_output_file, config, result);
        }
        if (!config.fragment_birth_spectrum_output_file.empty()) {
            carbon::write_fragment_birth_spectrum_csv(
                config.fragment_birth_spectrum_output_file, config, result);
        }
        if (config.enable_let_scoring && config.enable_voxel_scoring &&
            !config.let_voxel_mhd_output_file.empty()) {
            carbon::write_dense_voxel_letd_mhd(
                config.let_voxel_mhd_output_file, config, result);
        }
        if (config.enable_voxel_scoring && !config.voxel_dose_output_file.empty()) {
            carbon::write_sparse_voxel_dose_csv(config.voxel_dose_output_file, config, result);
        }
        if (config.enable_charged_origin_voxel_scoring &&
            !config.charged_origin_voxel_output_file.empty()) {
            carbon::write_sparse_charged_origin_voxel_dose_csv(
                config.charged_origin_voxel_output_file, config, result);
        }
        // Dose scorers (total Gy over all histories). Independent outputs; empty path skips.
        if (!config.dose_output_file.empty()) {
            carbon::write_depth_dose_Gy_csv(config.dose_output_file, config, result);
        }
        if (config.enable_secondary_transport &&
            config.enable_fragment_species_scoring &&
            !config.fragment_species_dose_output_file.empty()) {
            carbon::write_fragment_species_dose_Gy_csv(
                config.fragment_species_dose_output_file, config, result);
        }
        if (config.enable_voxel_scoring && !config.voxel_dose_Gy_output_file.empty()) {
            carbon::write_sparse_voxel_dose_Gy_csv(
                config.voxel_dose_Gy_output_file, config, result);
        }
        if (config.enable_voxel_scoring && !config.voxel_dose_mhd_output_file.empty()) {
            carbon::write_dense_voxel_dose_mhd(config.voxel_dose_mhd_output_file, config,
                                              result);
        }
        if (config.enable_charged_origin_voxel_scoring &&
            !config.charged_origin_voxel_dose_Gy_output_file.empty()) {
            carbon::write_sparse_charged_origin_voxel_dose_Gy_csv(
                config.charged_origin_voxel_dose_Gy_output_file, config, result);
        }
        const auto histories_per_second =
            result.elapsed_seconds > 0.0 ? static_cast<double>(config.number_of_histories) /
                                               result.elapsed_seconds
                                         : 0.0;
        std::cout << std::setprecision(8)
                  << "Backend: " << result.backend << '\n'
                  << "Histories: " << config.number_of_histories << '\n'
                  << "Initial energy: " << config.initial_energy_MeVu << " MeV/u = "
                  << config.initial_total_energy_MeV() << " MeV per C-12\n"
                  << "Steps: " << result.total_steps << '\n'
                  << "Elapsed: " << result.elapsed_seconds << " s\n"
                  << "Throughput: " << histories_per_second << " histories/s\n"
                  << "Kernel time: primary=" << result.primary_kernel_seconds
                  << " s secondary=" << result.secondary_kernel_seconds
                  << " s neutral=" << result.neutral_kernel_seconds
                  << " s charged-after-neutral="
                  << result.charged_after_neutral_kernel_seconds << " s\n"
                  << "Energy balance error: " << result.relative_energy_balance_error() << '\n'
                  << "Nuclear interactions: " << result.nuclear_interactions << '\n';
        if (result.minibeam.enabled) {
            const auto count = static_cast<double>(
                result.minibeam.water_entrance_primary_c12);
            const auto mean_and_std = [count](const double sum,
                                              const double squared_sum) {
                if (count <= 0.0) {
                    return std::pair{0.0, 0.0};
                }
                const auto mean = sum / count;
                return std::pair{
                    mean,
                    std::sqrt(std::max(
                        0.0, squared_sum / count - mean * mean))};
            };
            const auto [energy_mean, energy_std] = mean_and_std(
                result.minibeam.energy_sum_MeV,
                result.minibeam.energy_squared_sum_MeV2);
            const auto [x_mean, x_std] = mean_and_std(
                result.minibeam.x_sum_mm,
                result.minibeam.x_squared_sum_mm2);
            const auto [y_mean, y_std] = mean_and_std(
                result.minibeam.y_sum_mm,
                result.minibeam.y_squared_sum_mm2);
            const auto [dx_mean, dx_std] = mean_and_std(
                result.minibeam.direction_x_sum,
                result.minibeam.direction_x_squared_sum);
            const auto [dy_mean, dy_std] = mean_and_std(
                result.minibeam.direction_y_sum,
                result.minibeam.direction_y_squared_sum);
            std::cout
                << "Minibeam incident/direct/Copper-touched/Copper-nuclear/"
                   "water-primary: "
                << result.minibeam.incident_histories << '/'
                << result.minibeam.direct_air_slit_histories << '/'
                << result.minibeam.copper_touched_histories << '/'
                << result.minibeam.copper_nuclear_interactions << '/'
                << result.minibeam.water_entrance_primary_c12 << '\n'
                << "Minibeam beamline removed energy: "
                << result.minibeam.beamline_removed_energy_MeV << " MeV\n"
                << "Minibeam Copper products generated/charged-survivor/"
                   "neutral-survivor: "
                << result.minibeam.copper_generated_direct_secondaries << '/'
                << result.minibeam.copper_charged_survivors << '/'
                << result.minibeam.copper_neutral_survivors << '\n'
                << "Minibeam Copper survivor charged/neutral energy: "
                << result.minibeam.copper_charged_survivor_energy_MeV << '/'
                << result.minibeam.copper_neutral_survivor_energy_MeV
                << " MeV\n"
                << "Minibeam water entrance primary energy mean/std: "
                << energy_mean << '/' << energy_std << " MeV\n"
                << "Minibeam water entrance x mean/std: "
                << x_mean << '/' << x_std << " mm\n"
                << "Minibeam water entrance y mean/std: "
                << y_mean << '/' << y_std << " mm\n"
                << "Minibeam water entrance dir-x mean/std: "
                << dx_mean << '/' << dx_std << '\n'
                << "Minibeam water entrance dir-y mean/std: "
                << dy_mean << '/' << dy_std << '\n';
            std::cout << "Minibeam water entrance primary C-12 by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .water_entrance_primary_c12_by_slit[slit];
            }
            std::cout << '\n';
            std::cout
                << "Minibeam collimator entrance primary C-12 by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .collimator_entrance_primary_c12_by_slit[slit];
            }
            std::cout << '\n';
            std::cout << "Minibeam direct-air primary C-12 by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .direct_air_primary_c12_by_slit[slit];
            }
            std::cout << '\n';
            const auto direct_primary_count =
                result.minibeam.direct_air_slit_histories;
            const auto copper_touched_primary_count =
                result.minibeam.water_entrance_primary_c12 -
                direct_primary_count;
            std::cout
                << "Minibeam water entrance direct/touched primary "
                   "energy mean: "
                << (direct_primary_count > 0
                        ? result.minibeam.direct_air_primary_energy_MeV /
                              static_cast<double>(direct_primary_count)
                        : 0.0)
                << '/'
                << (copper_touched_primary_count > 0
                        ? result.minibeam
                                  .copper_touched_primary_energy_MeV /
                              static_cast<double>(
                                  copper_touched_primary_count)
                        : 0.0)
                << " MeV\n";
            std::cout
                << "Minibeam Copper-touched primary energy histogram "
                   "(200 MeV bins):";
            for (std::size_t energy_bin = 0;
                 energy_bin <
                 carbon::MinibeamDiagnostics::touched_energy_bin_count;
                 ++energy_bin) {
                std::cout
                    << (energy_bin == 0 ? ' ' : '/')
                    << result.minibeam
                           .copper_touched_primary_energy_histogram[
                               energy_bin];
            }
            std::cout << '\n';
            constexpr std::array<const char*, 9> species_labels{
                "C", "B", "Be", "Li", "He", "p", "d", "t",
                "other"};
            std::cout << "Minibeam Copper charged survivors by species:";
            for (std::size_t category = 0;
                 category < species_labels.size(); ++category) {
                std::cout
                    << ' ' << species_labels[category] << '='
                    << result.minibeam
                           .copper_charged_survivors_by_species[category]
                    << '/'
                    << result.minibeam
                           .copper_charged_survivor_energy_by_species_MeV[
                               category]
                    << "MeV";
            }
            std::cout << '\n';
            const auto print_fragment_histogram =
                [](const char* label, const auto& histogram) {
                    std::cout << "Minibeam water-entrance " << label
                              << " energy histogram (50 MeV bins):";
                    for (const auto count : histogram) {
                        std::cout << ' ' << count;
                    }
                    std::cout << '\n';
                };
            print_fragment_histogram(
                "d",
                result.minibeam.copper_deuteron_energy_histogram);
            print_fragment_histogram(
                "t", result.minibeam.copper_triton_energy_histogram);
            print_fragment_histogram(
                "He", result.minibeam.copper_helium_energy_histogram);
        }
        if (result.profile.enabled) {
            std::cout << result.profile.summary();
        }
        if (config.enable_secondary_generation) {
            std::cout << "Sampled reaction packages: " << result.sampled_reaction_packages << '\n'
                      << "Generated direct secondaries: "
                      << result.generated_direct_secondaries << '\n'
                      << "Generated direct-secondary energy: "
                      << result.generated_direct_secondary_energy_MeV << " MeV\n"
                      << "Queued charged secondaries: " << result.queued_secondaries << '\n'
                      << "Secondary queue overflow: " << result.secondary_queue_overflow << '\n'
                      << "Queued secondary energy: " << result.queued_secondary_energy_MeV
                      << " MeV\n"
                      << "Secondary queue overflow energy: "
                      << result.secondary_queue_overflow_energy_MeV << " MeV\n"
                      << "Untransported neutral energy: "
                      << result.untransported_neutral_energy_MeV << " MeV\n"
                      << "Untransported unsupported charged energy: "
                      << result.untransported_unsupported_charged_energy_MeV << " MeV\n"
                      << "Nuclear energy not in sampled direct secondaries: "
                      << result.nuclear_energy_not_in_direct_secondaries_MeV << " MeV\n";
            if (config.enable_secondary_transport) {
                std::cout << "Transported charged secondaries: "
                          << result.transported_secondaries << '\n'
                          << "Secondary transport steps: "
                          << result.secondary_transport_steps << '\n'
                          << "Secondary deposited energy: "
                          << result.secondary_deposited_energy_MeV << " MeV\n"
                          << "Secondary escaped energy: "
                          << result.secondary_escaped_energy_MeV << " MeV\n"
                          << "Cascade interactions: " << result.cascade_interactions << '\n'
                          << "Generated cascade products: "
                          << result.generated_cascade_products << '\n'
                          << "Queued cascade secondaries: "
                          << result.queued_cascade_secondaries << '\n'
                          << "Cascade queue overflow: " << result.cascade_queue_overflow
                          << '\n';
                if (config.enable_fragment_species_scoring &&
                    !config.fragment_species_output_file.empty()) {
                    std::cout << "Fragment species output: "
                              << config.fragment_species_output_file.string() << '\n';
                }
            }
            if (config.enable_neutral_transport) {
                std::cout << "Neutral mode: " << config.neutral_transport_mode << '\n'
                          << "Queued neutrals: " << result.queued_neutrals << '\n'
                          << "Neutral queue overflow: " << result.neutral_queue_overflow
                          << '\n'
                          << "Transported neutrals: " << result.transported_neutrals << '\n'
                          << "Neutral interactions: " << result.neutral_interactions << '\n'
                          << "Neutral deposited energy: "
                          << result.neutral_deposited_energy_MeV << " MeV\n"
                          << "Neutral escaped energy: " << result.neutral_escaped_energy_MeV
                          << " MeV\n"
                          << "Residual neutral energy: "
                          << result.residual_neutral_energy_MeV << " MeV\n"
                          << "Charged-from-neutral energy: "
                          << result.charged_from_neutral_energy_MeV << " MeV\n";
            }
        }
        std::cout << "Untracked nuclear energy: " << result.untracked_nuclear_energy_MeV
                  << " MeV\n"
                  << "MeV scorer output: " << config.output_file.string() << '\n';
        if (!config.dose_output_file.empty()) {
            std::cout << "Dose scorer output (Gy): " << config.dose_output_file.string()
                      << '\n';
        }
        if (config.dose_output_scale != 1.0) {
            std::cout << "Dose output scale (independent calibration): "
                      << config.dose_output_scale << '\n';
        }
        if (config.enable_secondary_transport &&
            config.enable_fragment_species_scoring &&
            !config.fragment_species_dose_output_file.empty()) {
            std::cout << "Fragment-species dose (Gy): "
                      << config.fragment_species_dose_output_file.string() << '\n';
        }
        if (config.enable_let_scoring && !config.let_output_file.empty()) {
            std::cout << "LET_d scorer output: " << config.let_output_file.string()
                      << '\n';
        }
        if (config.enable_let_scoring &&
            !config.fragment_species_let_output_file.empty()) {
            std::cout << "Fragment-species LET_d output: "
                      << config.fragment_species_let_output_file.string() << '\n';
        }
        if (config.enable_let_scoring &&
            !config.light_isotope_let_output_file.empty()) {
            std::cout << "Light-isotope LET_d output: "
                      << config.light_isotope_let_output_file.string() << '\n';
        }
        if (!config.fragment_birth_spectrum_output_file.empty()) {
            std::cout << "Fragment birth spectrum prefix: "
                      << config.fragment_birth_spectrum_output_file.string()
                      << '\n';
        }
        if (config.enable_let_scoring && config.enable_voxel_scoring &&
            !config.let_voxel_mhd_output_file.empty()) {
            std::cout << "Voxel LET_d MHD outputs: "
                      << config.let_voxel_mhd_output_file.string() << '\n';
        }
        if (config.enable_voxel_scoring) {
            std::cout << "Voxel MeV scorer output: "
                      << config.voxel_dose_output_file.string() << '\n';
            if (!config.voxel_dose_Gy_output_file.empty()) {
                std::cout << "Voxel dose scorer (Gy): "
                          << config.voxel_dose_Gy_output_file.string() << '\n';
            }
            if (!config.voxel_dose_mhd_output_file.empty()) {
                std::cout << "Voxel dose MHD (Gy): "
                          << config.voxel_dose_mhd_output_file.string() << '\n';
            }
        }
        if (config.enable_charged_origin_voxel_scoring) {
            std::cout << "Charged-origin MeV scorer output: "
                      << config.charged_origin_voxel_output_file.string() << '\n';
            if (!config.charged_origin_voxel_dose_Gy_output_file.empty()) {
                std::cout << "Charged-origin dose scorer (Gy): "
                          << config.charged_origin_voxel_dose_Gy_output_file.string()
                          << '\n';
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_mc: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
