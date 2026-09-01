#include "carbon/plan_run.hpp"

#include "carbon/ct_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

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

void accumulate_transport_result(carbon::TransportResult& total,
                                 const carbon::TransportResult& part) {
    add_vector_in_place(total.deposited_energy_MeV, part.deposited_energy_MeV);
    add_vector_in_place(total.voxel_deposited_energy_MeV, part.voxel_deposited_energy_MeV);
    add_vector_in_place(total.charged_origin_voxel_deposited_energy_MeV,
                        part.charged_origin_voxel_deposited_energy_MeV);
    add_vector_in_place(total.be_isotope_origin_voxel_deposited_energy_MeV,
                        part.be_isotope_origin_voxel_deposited_energy_MeV);
    add_vector_in_place(total.neutral_origin_voxel_deposited_energy_MeV,
                        part.neutral_origin_voxel_deposited_energy_MeV);
    add_vector_in_place(total.primary_deposited_energy_MeV,
                        part.primary_deposited_energy_MeV);
    add_vector_in_place(total.secondary_carbon_deposited_energy_MeV,
                        part.secondary_carbon_deposited_energy_MeV);
    add_vector_in_place(total.secondary_boron_deposited_energy_MeV, part.secondary_boron_deposited_energy_MeV);
    add_vector_in_place(total.secondary_beryllium_deposited_energy_MeV,
                        part.secondary_beryllium_deposited_energy_MeV);
    add_vector_in_place(total.secondary_lithium_deposited_energy_MeV, part.secondary_lithium_deposited_energy_MeV);
    add_vector_in_place(total.secondary_helium_deposited_energy_MeV, part.secondary_helium_deposited_energy_MeV);
    add_vector_in_place(total.secondary_proton_deposited_energy_MeV, part.secondary_proton_deposited_energy_MeV);
    add_vector_in_place(total.secondary_other_charged_deposited_energy_MeV,
                        part.secondary_other_charged_deposited_energy_MeV);
    add_vector_in_place(total.primary_letd_numerator,
                        part.primary_letd_numerator);
    add_vector_in_place(total.primary_letd_denominator,
                        part.primary_letd_denominator);
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
    add_vector_in_place(total.primary_voxel_letd_numerator,
                        part.primary_voxel_letd_numerator);
    add_vector_in_place(total.primary_voxel_letd_denominator,
                        part.primary_voxel_letd_denominator);
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
    total.fred_model_unassigned_MeV += part.fred_model_unassigned_MeV;
    total.nuclear_interactions += part.nuclear_interactions;
    for (std::size_t i = 0; i < total.cinel02_energy_ledger_MeV.size(); ++i) {
        total.cinel02_energy_ledger_MeV[i] += part.cinel02_energy_ledger_MeV[i];
    }
    for (std::size_t i = 0;
         i < total.cinel02_species_transport_ledger_MeV.size(); ++i) {
        total.cinel02_species_transport_ledger_MeV[i] +=
            part.cinel02_species_transport_ledger_MeV[i];
    }
    for (std::size_t i = 0; i < total.fred_isotope_counts.size(); ++i) {
        total.fred_isotope_counts[i] += part.fred_isotope_counts[i];
    }
    total.fred_inelastic_events += part.fred_inelastic_events;
    total.fred_retry_sum += part.fred_retry_sum;
    total.fred_energy_scaled_events += part.fred_energy_scaled_events;
    total.fred_projectile_az_open_events += part.fred_projectile_az_open_events;
    total.fred_leftover_target_a_sum += part.fred_leftover_target_a_sum;
    total.fred_leftover_target_z_sum += part.fred_leftover_target_z_sum;
    total.fred_leftover_projectile_a_sum += part.fred_leftover_projectile_a_sum;
    total.fred_leftover_projectile_z_sum += part.fred_leftover_projectile_z_sum;
    total.fred_model_residual_MeV += part.fred_model_residual_MeV;
    total.fred_q_MeV += part.fred_q_MeV;
    total.fred_neutron_ke_MeV += part.fred_neutron_ke_MeV;
    total.fred_remnant_local_MeV += part.fred_remnant_local_MeV;
    total.fred_resample_failed_events += part.fred_resample_failed_events;
    total.fred_resample_failed_energy_MeV += part.fred_resample_failed_energy_MeV;
    total.fred_product_capacity_overflow_events += part.fred_product_capacity_overflow_events;
    total.fred_product_capacity_overflow_energy_MeV +=
        part.fred_product_capacity_overflow_energy_MeV;
    total.fred_invert_error_proj_h =
        std::max(total.fred_invert_error_proj_h, part.fred_invert_error_proj_h);
    total.fred_invert_error_proj_o =
        std::max(total.fred_invert_error_proj_o, part.fred_invert_error_proj_o);
    total.fred_invert_error_tgt_h =
        std::max(total.fred_invert_error_tgt_h, part.fred_invert_error_tgt_h);
    total.fred_invert_error_tgt_o =
        std::max(total.fred_invert_error_tgt_o, part.fred_invert_error_tgt_o);
    total.primary_elastic_interactions += part.primary_elastic_interactions;
    total.elastic_local_deposited_energy_MeV +=
        part.elastic_local_deposited_energy_MeV;
    total.elastic_queued_charged_energy_MeV +=
        part.elastic_queued_charged_energy_MeV;
    total.elastic_queued_neutral_energy_MeV +=
        part.elastic_queued_neutral_energy_MeV;
    total.elastic_queue_overflow += part.elastic_queue_overflow;
    total.elastic_queue_overflow_energy_MeV +=
        part.elastic_queue_overflow_energy_MeV;
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
    total.cascade_selection_exact += part.cascade_selection_exact;
    total.cascade_selection_expanded += part.cascade_selection_expanded;
    total.cascade_selection_nearest += part.cascade_selection_nearest;
    total.cascade_selection_no_coverage +=
        part.cascade_selection_no_coverage;
    total.cascade_selection_energy_distance_sum_MeVu +=
        part.cascade_selection_energy_distance_sum_MeVu;
    total.cascade_selection_energy_distance_max_MeVu = std::max(
        total.cascade_selection_energy_distance_max_MeVu,
        part.cascade_selection_energy_distance_max_MeVu);
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
    total.neutral_unsupported_product_energy_MeV +=
        part.neutral_unsupported_product_energy_MeV;
    total.neutral_package_closure_residual_MeV +=
        part.neutral_package_closure_residual_MeV;
    total.queued_electrons += part.queued_electrons;
    total.electron_queue_overflow += part.electron_queue_overflow;
    total.transported_electrons += part.transported_electrons;
    total.transported_positrons += part.transported_positrons;
    total.electron_transport_steps += part.electron_transport_steps;
    total.queued_electron_energy_MeV += part.queued_electron_energy_MeV;
    total.electron_queue_overflow_energy_MeV +=
        part.electron_queue_overflow_energy_MeV;
    total.electron_deposited_energy_MeV += part.electron_deposited_energy_MeV;
    total.electron_escaped_energy_MeV += part.electron_escaped_energy_MeV;
    total.electron_radiative_energy_MeV += part.electron_radiative_energy_MeV;
    total.positron_annihilation_reserve_MeV +=
        part.positron_annihilation_reserve_MeV;
    total.electron_generated_gammas += part.electron_generated_gammas;
    total.electron_gamma_queue_overflow += part.electron_gamma_queue_overflow;
    total.electron_brems_gamma_energy_MeV +=
        part.electron_brems_gamma_energy_MeV;
    total.positron_annihilation_gamma_energy_MeV +=
        part.positron_annihilation_gamma_energy_MeV;
    total.electron_gamma_queue_overflow_energy_MeV +=
        part.electron_gamma_queue_overflow_energy_MeV;
    total.electromagnetic_generation_residual_MeV +=
        part.electromagnetic_generation_residual_MeV;
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
    total.minibeam.water_entrance_primary +=
        part.minibeam.water_entrance_primary;
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
    total.electron_kernel_seconds += part.electron_kernel_seconds;
    total.charged_after_neutral_kernel_seconds +=
        part.charged_after_neutral_kernel_seconds;
    if (total.backend.empty()) {
        total.backend = part.backend;
    }
}

carbon::SpotSourcePose transform_topas_pose_to_fixed_patient_lps(
    const carbon::SpotSourcePose& world_pose,
    const carbon::TransportConfig& config) {
    constexpr double kPi = 3.14159265358979323846;
    const auto angle = config.tps_gantry_angle_deg * kPi / 180.0;
    const auto c = std::cos(angle);
    const auto s = std::sin(angle);
    const auto rotate = [c, s](const double x, const double y, const double z) {
        return std::array<double, 3>{c * x - s * y, s * x + c * y, z};
    };
    const auto origin = rotate(
        world_pose.origin_x_mm - config.spots_patient_trans_x_mm,
        world_pose.origin_y_mm - config.spots_patient_trans_y_mm,
        world_pose.origin_z_mm - config.spots_patient_trans_z_mm);
    const auto ux = rotate(world_pose.ux_x, world_pose.ux_y, world_pose.ux_z);
    const auto uy = rotate(world_pose.uy_x, world_pose.uy_y, world_pose.uy_z);
    const auto uz = rotate(world_pose.uz_x, world_pose.uz_y, world_pose.uz_z);
    // TOPAS TsDicomPatient local coordinates are centered on the CT volume.
    // CCTG stores slices from z=0 internally, so shift patient Z by half the
    // native superior-inferior extent without permuting the voxel array.
    const auto patient_z_low_edge_mm = -0.5 * config.phantom_length_mm;
    return carbon::SpotSourcePose{
        origin[0], origin[1], origin[2] - patient_z_low_edge_mm,
        ux[0], ux[1], ux[2], uy[0], uy[1], uy[2], uz[0], uz[1], uz[2]};
}

double distance_to_patient_ct_entry(const carbon::SpotSourcePose& pose,
                                    const carbon::TransportConfig& config) {
    auto enter = 0.0;
    auto exit = std::numeric_limits<double>::infinity();
    const auto intersect = [&](const double position, const double direction,
                               const double low, const double high) {
        if (std::abs(direction) < 1.0e-12) {
            return position >= low && position < high;
        }
        auto first = (low - position) / direction;
        auto second = (high - position) / direction;
        if (first > second) {
            std::swap(first, second);
        }
        enter = std::max(enter, first);
        exit = std::min(exit, second);
        return exit >= enter;
    };
    const auto x_max = config.voxel_origin_x_mm +
                       static_cast<double>(config.voxel_bins_x) *
                           config.voxel_size_x_mm;
    const auto y_max = config.voxel_origin_y_mm +
                       static_cast<double>(config.voxel_bins_y) *
                           config.voxel_size_y_mm;
    if (!intersect(pose.origin_x_mm, pose.uz_x, config.voxel_origin_x_mm, x_max) ||
        !intersect(pose.origin_y_mm, pose.uz_y, config.voxel_origin_y_mm, y_max) ||
        !intersect(pose.origin_z_mm, pose.uz_z, 0.0, config.phantom_length_mm) ||
        !std::isfinite(enter)) {
        throw std::runtime_error("TPS beam does not intersect the native patient CT");
    }
    return enter;
}

void apply_spot_to_config(carbon::TransportConfig& config,
                          const carbon::TopasSpotPlan& plan,
                          const carbon::TopasSpot& spot,
                          std::size_t spot_index,
                          std::uint64_t base_seed,
                          const carbon::StoppingPowerTable* upstream_air_stopping_power,
                          UpstreamAirLossAudit* upstream_air_audit) {
    if (config.primary_mass_number <= 0) {
        throw std::invalid_argument(
            "primary_mass_number must be positive for spots plans");
    }
    config.number_of_histories = spot.number_of_histories;
    config.initial_energy_MeVu = spot.energy_MeV / static_cast<double>(config.primary_mass_number);
    // TOPAS BeamEnergySpread is percent (1.0 => 1%); GPU uses relative RMS.
    config.beam_energy_spread = spot.energy_spread_percent / 100.0;

    // A flat field is shared by every energy layer. Otherwise apply the spot's
    // TOPAS BiGaussian emittance values (zero values retain a pencil source).
    config.enable_emittance_source = !config.enable_flat_source;
    if (config.enable_emittance_source) {
        const auto sigma_scale = config.spots_emittance_sigma_scale;
        const auto prime_scale = config.spots_emittance_prime_scale;
        config.emittance_sigma_x_mm = spot.sigma_x_mm * sigma_scale;
        config.emittance_sigma_y_mm = spot.sigma_y_mm * sigma_scale;
        config.emittance_sigma_x_prime = spot.sigma_x_prime * prime_scale;
        config.emittance_sigma_y_prime = spot.sigma_y_prime * prime_scale;
        config.emittance_correlation_x = spot.correlation_x;
        config.emittance_correlation_y = spot.correlation_y;
    }

    if (config.enable_tps_coordinate_system) {
        auto pose = transform_topas_pose_to_fixed_patient_lps(
            plan.tps_zero_beam_pose_for_spot(spot), config);
        const auto distance_to_entrance_mm =
            distance_to_patient_ct_entry(pose, config);
        const auto entrance_energy =
            carbon::spot_entry_total_energy_after_optional_upstream_loss(
                spot.energy_MeV, config.primary_mass_number, distance_to_entrance_mm,
                upstream_air_stopping_power);
        config.initial_energy_MeVu =
            entrance_energy / static_cast<double>(config.primary_mass_number);
        if (upstream_air_audit != nullptr) {
            upstream_air_audit->distance_to_entrance_mm = distance_to_entrance_mm;
            upstream_air_audit->energy_loss_MeV = spot.energy_MeV - entrance_energy;
        }
        // Keep the source at SAD. The SYCL AABB entry path advances each sampled
        // ray to the fixed patient CT, including emittance divergence.
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
    } else if (config.spots_geometry_mode == "beam_plus_z") {
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
        const auto entrance_energy =
            carbon::spot_entry_total_energy_after_optional_upstream_loss(
                spot.energy_MeV, config.primary_mass_number, distance_to_entrance_mm,
                upstream_air_stopping_power);
        // With no table this is exactly the legacy spot.energy_MeV assignment.
        config.initial_energy_MeVu =
            entrance_energy / static_cast<double>(config.primary_mass_number);
        if (upstream_air_audit != nullptr) {
            upstream_air_audit->distance_to_entrance_mm = distance_to_entrance_mm;
            upstream_air_audit->energy_loss_MeV = spot.energy_MeV - entrance_energy;
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
    carbon::SyclTransportContext* sycl_context) {
    if (config.device == "serial") {
        if (config.enable_let_scoring) {
            throw std::invalid_argument(
                "scorerLET is currently implemented only by the SYCL backend");
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
                                  sycl_context);
#else
    (void)sycl_context;
    throw std::runtime_error(
        "This binary was built without SYCL. Reconfigure with CARBON_ENABLE_SYCL=ON and icpx.");
#endif
}

}  // namespace carbon
