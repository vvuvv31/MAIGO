#include "carbon/cascade_package.hpp"
#include "carbon/cli.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/elastic_package.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/io.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/plan_run.hpp"
#include "carbon/package_identity.hpp"
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    // Line-buffer stdout so progress is visible when piped (tee/logs) and during
    // long CUDA kernels that would otherwise freeze WSL with no feedback.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        carbon::CliState cli;
        carbon::parse_config_and_help(argc, argv, cli);
        if (cli.help) {
            carbon::print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        auto config = carbon::load_config(cli.config_path);
        carbon::apply_cli_overrides(argc, argv, config, cli);
        const auto histories_cli_override = cli.histories_overridden;
        const auto plan_only = cli.plan_only;
        const auto sequential_spots = cli.sequential_spots;
        config.validate();
        if (!cli.canonical_config_output_path.empty()) {
            const auto parent = cli.canonical_config_output_path.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            std::ofstream canonical_output(
                cli.canonical_config_output_path, std::ios::trunc);
            if (!canonical_output) {
                throw std::runtime_error(
                    "Cannot write canonical configuration: " +
                    cli.canonical_config_output_path.string());
            }
            canonical_output << config.canonical_config_text;
            if (!canonical_output) {
                throw std::runtime_error(
                    "Failed while writing canonical configuration: " +
                    cli.canonical_config_output_path.string());
            }
            std::cout << "Canonical configuration: "
                      << cli.canonical_config_output_path << '\n';
        }

        const auto primary_ion = config.primary_ion();
        if (!config.ion_physics_file.empty()) {
            std::cout << "Ion physics manifest: " << config.ion_physics_file
                      << '\n';
        }
        std::cout << "Primary ion: Z=" << primary_ion.atomic_number
                  << " A=" << primary_ion.mass_number
                  << "; rest mass=" << primary_ion.rest_mass_MeV << " MeV"
                  << (config.primary_rest_mass_MeV > 0.0
                          ? " (configured)\n"
                          : " (A * nucleon mass)\n");

        if (config.enable_primary_inelastic_xs_correction) {
            std::cout << "Primary inelastic XS correction: "
                      << config.primary_inelastic_xs_correction_file
                      << "; points="
                      << config.primary_inelastic_xs_correction_energies_MeVu.size()
                      << "; incident-energy range="
                      << config.primary_inelastic_xs_correction_energies_MeVu.front()
                      << ".."
                      << config.primary_inelastic_xs_correction_energies_MeVu.back()
                      << " MeV/u; linear interpolation with endpoint clamping\n";
        }

        const auto stopping_power = carbon::StoppingPowerTable::from_csv(config.primary_stopping_power_file);
        std::optional<carbon::StoppingPowerTable> upstream_air_stopping_power;
        if (config.spots_enable_upstream_air_energy_loss) {
            upstream_air_stopping_power = carbon::StoppingPowerTable::from_csv(
                config.spots_upstream_air_stopping_power_file);
        }
        const auto* upstream_air_stopping_power_ptr =
            upstream_air_stopping_power ? &*upstream_air_stopping_power : nullptr;
        const auto cross_section =
            carbon::CrossSectionTable::from_csv(config.primary_inelastic_cross_section_file);
        std::optional<carbon::CrossSectionTable> elastic_cross_section;
        std::optional<carbon::ElasticPackageTable> elastic_packages;
        if (config.enable_primary_elastic_interactions) {
            elastic_cross_section = carbon::CrossSectionTable::from_csv(
                config.primary_elastic_cross_section_file);
            const auto& elastic_energies = elastic_cross_section->energies();
            if (elastic_energies.empty() ||
                elastic_energies.front() > config.initial_energy_MeVu ||
                elastic_energies.back() < config.initial_energy_MeVu) {
                throw std::invalid_argument(
                    "primary elastic cross-section table must cover the "
                    "configured initial energy (and its zero-energy clamp)");
            }
            carbon::validate_package_identity(
                config.primary_elastic_package_file,
                {"elastic", config.primary_atomic_number,
                 config.primary_mass_number, "G4_WATER",
                 config.primary_elastic_package_physics_model, 0.0,
                 config.initial_energy_MeVu},
                config.package_identity_validation,
                config.package_identity_override_manifest_file);
            elastic_packages = carbon::ElasticPackageTable::from_binary(
                config.primary_elastic_package_file);
            std::cout << "Elastic cross section: "
                      << elastic_cross_section->energies().size()
                      << " bins, energy range="
                      << elastic_cross_section->energies().front() << ".."
                      << elastic_cross_section->energies().back()
                      << " MeV/u\n"
                      << "Elastic packages: bins="
                      << elastic_packages->energy_bins().size()
                      << "; events=" << elastic_packages->events().size()
                      << "; products=" << elastic_packages->products().size()
                      << "; model="
                      << config.primary_elastic_package_physics_model << '\n';
        }
        std::optional<carbon::ReactionPackageTable> reaction_packages;
        std::optional<carbon::CascadePackageTable> cascade_packages;
        std::optional<carbon::NeutralPackageTable> neutral_packages;
        if (config.enable_secondary_generation) {
            carbon::validate_package_identity(
                config.primary_reaction_package_file,
                {"reaction", config.primary_atomic_number, config.primary_mass_number,
                 "G4_WATER", config.primary_package_physics_model, 0.0,
                 config.initial_energy_MeVu},
                config.package_identity_validation,
                config.package_identity_override_manifest_file);
            reaction_packages =
                carbon::ReactionPackageTable::from_binary(config.primary_reaction_package_file);
            std::cout << "Reaction packages: " << reaction_packages->reactions().size()
                      << "; direct secondaries: " << reaction_packages->secondaries().size()
                      << "; local deposit: Geant4 package"
                      << '\n';
        }
        if (config.enable_fragment_cascade) {
            carbon::validate_package_identity(
                config.cascade_package_file,
                {"cascade", config.primary_atomic_number, config.primary_mass_number,
                 "G4_WATER", config.cascade_package_physics_model, 0.0,
                 config.initial_energy_MeVu},
                config.package_identity_validation,
                config.package_identity_override_manifest_file);
            cascade_packages =
                carbon::CascadePackageTable::from_binary(config.cascade_package_file);
            std::cout << "Cascade projectiles: " << cascade_packages->projectiles().size()
                      << "; interactions: " << cascade_packages->interactions().size()
                      << "; products: " << cascade_packages->products().size()
                      << "; local deposit: Geant4 package"
                      << '\n';
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
        if (plan_only && spots_files.empty() && !config.enable_tps_source) {
            std::cout << "Plan-only validation passed; transport not started\n";
            return EXIT_SUCCESS;
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
        std::cout << "Random seed: " << base_seed << '\n';

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
            if (!batch.empty() &&
                batch.back().history_end != config.number_of_histories) {
                config.number_of_histories = batch.back().history_end;
            }
            if (config.enable_ct_grid) {
                const auto patient_ct =
                    carbon::CtGrid::from_config(config);
                std::cout << "Patient CT (fixed DICOM LPS): DimSize="
                          << patient_ct.nx << "x" << patient_ct.ny << "x"
                          << patient_ct.nz << " spacing="
                          << patient_ct.spacing_x_mm << "x"
                          << patient_ct.spacing_y_mm << "x"
                          << patient_ct.spacing_z_mm << " mm origin=("
                          << patient_ct.origin_x_mm << ","
                          << patient_ct.origin_y_mm << ","
                          << patient_ct.origin_z_mm << ") mm\n";
            }
            std::cout << "TPS source: "
                      << (config.tps_spots_file.empty()
                              ? std::string{"single YAML spot"}
                              : config.tps_spots_file.string())
                      << '\n'
                      << "  spots: " << batch.size() << "/" << plan.spots.size()
                      << " active; total histories: " << config.number_of_histories
                      << "; total MU: " << plan.total_mu << '\n'
                      << "  beam/couch/collimator angle: "
                      << config.tps_gantry_angle_deg << "/"
                      << config.tps_couch_angle_deg << "/"
                      << config.tps_collimator_angle_deg << " deg; SAD: "
                      << config.tps_sad_mm << " mm; patient: "
                      << config.tps_patient_position << "; coordinates: "
                      << config.tps_angle_convention
                      << "; weight: " << config.tps_spot_weight_mode;
            if (config.tps_angle_convention == "dicom_lps" ||
                config.tps_angle_convention == "topas_patient_rot_z") {
                std::cout << " (+X left, +Y posterior, +Z superior; CT fixed)";
            }
            if (config.tps_virtual_scanning_magnet_x_mm > 0.0 &&
                config.tps_virtual_scanning_magnet_y_mm > 0.0) {
                const auto source_distance =
                    config.tps_virtual_source_to_isocenter_mm > 0.0
                        ? config.tps_virtual_source_to_isocenter_mm
                        : config.tps_sad_mm;
                std::cout << "\n  PBS virtual magnets: X="
                          << config.tps_virtual_scanning_magnet_x_mm << " mm Y="
                          << config.tps_virtual_scanning_magnet_y_mm
                          << " mm; source plane D=" << source_distance << " mm";
                if (!config.tps_beam_model_file.empty()) {
                    std::cout << "; beam model "
                              << config.tps_beam_model_file.string();
                }
                if (config.tps_apply_topas_patient_placement ||
                    config.spots_patient_rot_z_deg != 0.0) {
                    std::cout << "\n  TOPAS patient placement: RotZ="
                              << config.spots_patient_rot_z_deg
                              << " deg then Trans=("
                              << config.spots_patient_trans_x_mm << ", "
                              << config.spots_patient_trans_y_mm << ", "
                              << config.spots_patient_trans_z_mm
                              << ") mm; tps_90 packing ct_axis_min="
                              << config.spots_ct_axis_min_mm << " mm";
                }
            }
            std::cout << '\n';
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
            result = carbon::run_transport(batch_config, stopping_power, cross_section,
                                   reaction_packages, cascade_packages, neutral_packages,
                                   elastic_cross_section, elastic_packages,
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
            const auto total_histories = plan.total_histories();
            if (config.enable_tps_coordinate_system) {
                const auto patient_ct =
                    carbon::CtGrid::from_config(config);
                std::cout << "Patient CT (fixed DICOM LPS): DimSize="
                          << patient_ct.nx << "x" << patient_ct.ny << "x"
                          << patient_ct.nz << " spacing="
                          << patient_ct.spacing_x_mm << "x"
                          << patient_ct.spacing_y_mm << "x"
                          << patient_ct.spacing_z_mm << " mm origin=("
                          << patient_ct.origin_x_mm << ","
                          << patient_ct.origin_y_mm << ","
                          << patient_ct.origin_z_mm << ") mm\n";
            }
            std::cout << "TOPAS spots plan files:";
            for (const auto& path : spots_files) {
                std::cout << ' ' << path.string();
            }
            std::cout << '\n'
                      << "  spots: " << plan.spots.size()
                      << "  total histories: " << total_histories
                      << "  geometry: "
                      << (config.enable_tps_coordinate_system
                              ? "fixed DICOM LPS patient CT"
                              : config.spots_geometry_mode)
                      << "  SAD: " << plan.sad_mm << " mm\n"
                      << "  (history ranges preserve file order; no TimeFeature timeline)\n";
            if (upstream_air_stopping_power_ptr != nullptr) {
                std::cout << "  upstream air energy loss: enabled; table "
                          << config.spots_upstream_air_stopping_power_file.string()
                          << "\n";
            }
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
                double min_entry_z = min_entry_x;
                double max_entry_z = -min_entry_x;
                std::array<double, 3> first_direction{};
                auto has_direction = false;
                double min_air_path = std::numeric_limits<double>::infinity();
                double max_air_path = -min_air_path;
                double min_air_loss = min_air_path;
                double max_air_loss = -min_air_path;
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    const auto& spot = plan.spots[i];
                    min_histories = std::min(min_histories, spot.number_of_histories);
                    max_histories = std::max(max_histories, spot.number_of_histories);
                    auto transformed = config;
                    carbon::UpstreamAirLossAudit air_audit{};
                    carbon::apply_spot_to_config(transformed, plan, spot, i, base_seed,
                                         upstream_air_stopping_power_ptr,
                                         upstream_air_stopping_power_ptr != nullptr ? &air_audit : nullptr);
                    transformed.validate();
                    min_entry_x = std::min(min_entry_x, transformed.source_origin_x_mm);
                    max_entry_x = std::max(max_entry_x, transformed.source_origin_x_mm);
                    min_entry_y = std::min(min_entry_y, transformed.source_origin_y_mm);
                    max_entry_y = std::max(max_entry_y, transformed.source_origin_y_mm);
                    min_entry_z = std::min(min_entry_z, transformed.source_origin_z_mm);
                    max_entry_z = std::max(max_entry_z, transformed.source_origin_z_mm);
                    if (!has_direction) {
                        first_direction = {transformed.beam_uz_x,
                                           transformed.beam_uz_y,
                                           transformed.beam_uz_z};
                        has_direction = true;
                    }
                    if (upstream_air_stopping_power_ptr != nullptr) {
                        min_air_path = std::min(min_air_path, air_audit.distance_to_entrance_mm);
                        max_air_path = std::max(max_air_path, air_audit.distance_to_entrance_mm);
                        min_air_loss = std::min(min_air_loss, air_audit.energy_loss_MeV);
                        max_air_loss = std::max(max_air_loss, air_audit.energy_loss_MeV);
                    }
                }
                std::cout << "  plan-only validation passed; histories/active spot min="
                          << min_histories << " max=" << max_histories << '\n'
                          << "  source bounds in patient coordinates: x=["
                          << min_entry_x << ", "
                          << max_entry_x << "] mm y=[" << min_entry_y << ", "
                          << max_entry_y << "] mm z=[" << min_entry_z << ", "
                          << max_entry_z << "] mm\n"
                          << "  first propagation direction: ("
                          << first_direction[0] << ", " << first_direction[1]
                          << ", " << first_direction[2] << ")\n";
                if (upstream_air_stopping_power_ptr != nullptr) {
                    std::cout << "  upstream air path=[" << min_air_path << ", "
                              << max_air_path << "] mm; total primary-ion loss=["
                              << min_air_loss << ", " << max_air_loss << "] MeV\n";
                }
                return EXIT_SUCCESS;
            }

            if (config.device != "serial" && !sequential_spots) {
                auto batch_config = config;
                batch_config.primary_spot_batch.clear();
                batch_config.primary_spot_batch.reserve(plan.spots.size());
                std::uint64_t history_begin = 0;
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    // Start from plan-level config; do not copy batch_config
                    // (it accumulates primary_spot_batch).
                    auto spot_config = config;
                    carbon::apply_spot_to_config(spot_config, plan, plan.spots[i], i, base_seed,
                                         upstream_air_stopping_power_ptr);
                    spot_config.validate();
                    auto entry = carbon::make_spot_batch_entry(spot_config, history_begin);
                    history_begin = entry.history_end;
                    batch_config.primary_spot_batch.push_back(entry);
                }
                batch_config.number_of_histories = total_histories;
                batch_config.enable_emittance_source = !batch_config.enable_flat_source;
                batch_config.validate();
                std::cout << "  batched SYCL launch: "
                          << batch_config.primary_spot_batch.size() << " spots, "
                          << total_histories << " histories\n";
                result = carbon::run_transport(batch_config, stopping_power, cross_section,
                                       reaction_packages, cascade_packages, neutral_packages,
                                       elastic_cross_section, elastic_packages,
                                       sycl_context);
            } else {
                for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                    const auto& spot = plan.spots[i];
                    auto spot_config = config;
                    carbon::apply_spot_to_config(spot_config, plan, spot, i, base_seed,
                                         upstream_air_stopping_power_ptr);
                    spot_config.validate();
                    auto spot_result = carbon::run_transport(
                        spot_config, stopping_power, cross_section, reaction_packages,
                        cascade_packages, neutral_packages, elastic_cross_section,
                        elastic_packages, sycl_context);
                    if (i == 0) {
                        result = std::move(spot_result);
                    } else {
                        carbon::accumulate_transport_result(result, spot_result);
                    }
                }
            }
            // Output writers divide by number_of_histories → absolute MeV / total primaries.
            config.number_of_histories = total_histories;
            if (!plan.spots.empty()) {
                // Keep the final run summary aligned with the same entry energy
                // used by both the primary batch and sequential spot paths.
                auto first_spot_config = config;
                carbon::apply_spot_to_config(first_spot_config, plan, plan.spots.front(), 0,
                                     base_seed, upstream_air_stopping_power_ptr);
                config.initial_energy_MeVu = first_spot_config.initial_energy_MeVu;
                config.beam_energy_spread = first_spot_config.beam_energy_spread;
            }
        } else {
            result = carbon::run_transport(config, stopping_power, cross_section, reaction_packages,
                                   cascade_packages, neutral_packages,
                                   elastic_cross_section, elastic_packages, nullptr);
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
        if (config.validation_scorers()) {
            const auto dir = config.validation_output_directory.empty()
                                 ? std::filesystem::path{"benchmark/scorer/results"}
                                 : config.validation_output_directory;
            carbon::write_energy_ledger_json(dir / "energy_ledger.json", config,
                                             result);
            carbon::write_validation_scorer_csvs(dir, config, result);
            if (result.secondary_queue_overflow != 0 ||
                result.cascade_queue_overflow != 0) {
                throw std::runtime_error(
                    "validation scorer run had a non-zero secondary/cascade queue overflow");
            }
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
        if (config.enable_charged_origin_voxel_scoring &&
            !config.charged_origin_voxel_mhd_output_prefix.empty()) {
            carbon::write_dense_charged_origin_voxel_dose_mhd(
                config.charged_origin_voxel_mhd_output_prefix, config, result);
        }
        const auto histories_per_second =
            result.elapsed_seconds > 0.0 ? static_cast<double>(config.number_of_histories) /
                                               result.elapsed_seconds
                                         : 0.0;
        std::cout << std::setprecision(8)
                  << "Backend: " << result.backend << '\n'
                  << "Histories: " << config.number_of_histories << '\n'
                  << "Initial energy: " << config.initial_energy_MeVu << " MeV/u = "
                  << config.initial_total_energy_MeV() << " MeV per primary ion\n"
                  << "Steps: " << result.total_steps << '\n'
                  << "Elapsed: " << result.elapsed_seconds << " s\n"
                  << "Throughput: " << histories_per_second << " histories/s\n"
                  << "Kernel time: primary=" << result.primary_kernel_seconds
                  << " s secondary=" << result.secondary_kernel_seconds
                  << " s neutral=" << result.neutral_kernel_seconds
                  << " s charged-after-neutral="
                  << result.charged_after_neutral_kernel_seconds << " s\n"
                  << "Energy balance error: " << result.relative_energy_balance_error() << '\n'
                  << "Nuclear interactions: " << result.nuclear_interactions << '\n'
                  << "Primary elastic interactions: "
                  << result.primary_elastic_interactions
                  << " (local=" << result.elastic_local_deposited_energy_MeV
                  << " MeV, queued charged="
                  << result.elastic_queued_charged_energy_MeV
                  << " MeV, queued neutral="
                  << result.elastic_queued_neutral_energy_MeV
                  << " MeV, overflow=" << result.elastic_queue_overflow
                  << ", overflow energy="
                  << result.elastic_queue_overflow_energy_MeV << " MeV)\n";
        if (result.minibeam.enabled) {
            const auto count = static_cast<double>(
                result.minibeam.water_entrance_primary);
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
                << result.minibeam.water_entrance_primary << '\n'
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
            std::cout << "Minibeam water entrance primary ion by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .water_entrance_primary_by_slit[slit];
            }
            std::cout << '\n';
            std::cout
                << "Minibeam collimator entrance primary ion by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .collimator_entrance_primary_by_slit[slit];
            }
            std::cout << '\n';
            std::cout << "Minibeam direct-air primary ion by slit:";
            for (std::size_t slit = 0;
                 slit < carbon::MinibeamDiagnostics::slit_count; ++slit) {
                std::cout
                    << (slit == 0 ? ' ' : '/')
                    << result.minibeam
                           .direct_air_primary_by_slit[slit];
            }
            std::cout << '\n';
            const auto direct_primary_count =
                result.minibeam.direct_air_slit_histories;
            const auto copper_touched_primary_count =
                result.minibeam.water_entrance_primary -
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
                          << result.charged_from_neutral_energy_MeV << " MeV\n"
                          << "Neutral unsupported-product energy: "
                          << result.neutral_unsupported_product_energy_MeV << " MeV\n"
                          << "Neutral package closure residual: "
                          << result.neutral_package_closure_residual_MeV << " MeV\n";
            }
        }
        std::cout << "Untracked nuclear energy: " << result.untracked_nuclear_energy_MeV
                  << " MeV\n"
                  << "MeV scorer output: " << config.output_file.string() << '\n';
        if (config.validation_scorers()) {
            const auto dir = config.validation_output_directory.empty()
                                 ? std::filesystem::path{"benchmark/scorer/results"}
                                 : config.validation_output_directory;
            std::cout << "Validation scorer directory: " << dir.string() << '\n';
        }
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
            if (!config.charged_origin_voxel_mhd_output_prefix.empty()) {
                std::cout << "Charged-origin voxel dose MHD prefix: "
                          << config.charged_origin_voxel_mhd_output_prefix.string()
                          << '\n';
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_mc: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
