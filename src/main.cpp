#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/io.hpp"
#include "carbon/neutral_package.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/topas_spots.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--config FILE] [--device serial|cpu|gpu] [--histories N]"
                 " [--spots FILE] [--straggling-scale X] [--output FILE]"
                 " [--dose-output FILE]\n"
                 "  --spots FILE         TOPAS-format spots_*.txt (L0-L14); run spots in order\n"
                 "  --histories N        If --spots is set, overrides histories per spot\n"
                 "  --output FILE        MeV energy-deposition scorer CSV\n"
                 "  --dose-output FILE   Dose scorer CSV (Gy/primary); empty disables\n";
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
    add_vector_in_place(total.neutron_origin_deposited_energy_MeV,
                        part.neutron_origin_deposited_energy_MeV);
    add_vector_in_place(total.gamma_origin_deposited_energy_MeV,
                        part.gamma_origin_deposited_energy_MeV);

    total.initial_energy_MeV += part.initial_energy_MeV;
    total.total_deposited_energy_MeV += part.total_deposited_energy_MeV;
    total.escaped_energy_MeV += part.escaped_energy_MeV;
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
    total.total_steps += part.total_steps;
    total.elapsed_seconds += part.elapsed_seconds;
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

carbon::TransportResult run_transport(
    const carbon::TransportConfig& config,
    const carbon::StoppingPowerTable& stopping_power,
    const carbon::CrossSectionTable& cross_section,
    const std::optional<carbon::ReactionPackageTable>& reaction_packages,
    const std::optional<carbon::CascadePackageTable>& cascade_packages,
    const std::optional<carbon::NeutralPackageTable>& neutral_packages) {
    if (config.device == "serial") {
        if (config.enable_secondary_generation) {
            throw std::invalid_argument(
                "Secondary generation is currently implemented only by the SYCL backend");
        }
        return carbon::transport_serial(config, stopping_power, cross_section);
    }
#ifdef CARBON_HAS_SYCL
    if (config.device != "cpu" && config.device != "gpu" && config.device != "default") {
        throw std::invalid_argument("SYCL device must be cpu, gpu, or default");
    }
    return carbon::transport_sycl(config, stopping_power, cross_section, config.device,
                                  reaction_packages ? &*reaction_packages : nullptr,
                                  cascade_packages ? &*cascade_packages : nullptr,
                                  neutral_packages ? &*neutral_packages : nullptr);
#else
    throw std::runtime_error(
        "This binary was built without SYCL. Reconfigure with CARBON_ENABLE_SYCL=ON and icpx.");
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
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
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config") {
                ++index;
            } else if (argument == "--device" && index + 1 < argc) {
                config.device = argv[++index];
            } else if (argument == "--histories" && index + 1 < argc) {
                config.number_of_histories = std::stoull(argv[++index]);
                histories_cli_override = true;
            } else if (argument == "--spots" && index + 1 < argc) {
                config.topas_spots_file = argv[++index];
            } else if (argument == "--straggling-scale" && index + 1 < argc) {
                config.straggling_scale = std::stod(argv[++index]);
            } else if (argument == "--output" && index + 1 < argc) {
                config.output_file = argv[++index];
            } else if (argument == "--dose-output" && index + 1 < argc) {
                const std::string path = argv[++index];
                config.dose_output_file =
                    path.empty() ? std::filesystem::path{} : std::filesystem::path{path};
            } else if (argument != "--help" && argument != "-h") {
                throw std::invalid_argument("Unknown or incomplete argument: " + argument);
            }
        }
        config.validate();

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

        carbon::TransportResult result;
        const auto base_seed = config.random_seed;

        if (!config.topas_spots_file.empty()) {
            auto plan = carbon::TopasSpotPlan::from_file(config.topas_spots_file);
            plan.sad_mm = config.spots_sad_mm;
            if (histories_cli_override) {
                for (auto& spot : plan.spots) {
                    spot.number_of_histories = config.number_of_histories;
                }
            }
            const auto total_histories = plan.total_histories();
            std::cout << "TOPAS spots plan: " << config.topas_spots_file.string() << '\n'
                      << "  spots: " << plan.spots.size()
                      << "  total histories: " << total_histories
                      << "  geometry: " << config.spots_geometry_mode
                      << "  SAD: " << plan.sad_mm << " mm\n"
                      << "  (sequential spot order; no TimeFeature timeline)\n";

            for (std::size_t i = 0; i < plan.spots.size(); ++i) {
                const auto& spot = plan.spots[i];
                auto spot_config = config;
                apply_spot_to_config(spot_config, plan, spot, i, base_seed);
                spot_config.validate();

                std::cout << "  spot " << (i + 1) << "/" << plan.spots.size()
                          << " id=" << spot.spot_id
                          << " E=" << spot.energy_MeV << " MeV ("
                          << spot_config.initial_energy_MeVu << " MeV/u)"
                          << " N=" << spot.number_of_histories
                          << " spread=" << spot.energy_spread_percent << "%"
                          << " Tx=" << spot.trans_x_mm << " Tz=" << spot.trans_z_mm
                          << " Rx=" << spot.rot_x_deg << " Ry=" << spot.rot_y_deg
                          << '\n';

                auto spot_result =
                    run_transport(spot_config, stopping_power, cross_section, reaction_packages,
                                  cascade_packages, neutral_packages);
                if (i == 0) {
                    result = std::move(spot_result);
                } else {
                    accumulate_transport_result(result, spot_result);
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

        // MeV energy-deposition scorers (unchanged).
        carbon::write_depth_dose_csv(config.output_file, config, result);
        if (config.enable_secondary_transport) {
            carbon::write_fragment_species_csv(
                config.fragment_species_output_file, config, result);
        }
        if (config.enable_voxel_scoring) {
            carbon::write_sparse_voxel_dose_csv(config.voxel_dose_output_file, config, result);
        }
        if (config.enable_charged_origin_voxel_scoring) {
            carbon::write_sparse_charged_origin_voxel_dose_csv(
                config.charged_origin_voxel_output_file, config, result);
        }
        // Dose scorers (Gy/primary). Independent outputs; empty path skips.
        if (!config.dose_output_file.empty()) {
            carbon::write_depth_dose_Gy_csv(config.dose_output_file, config, result);
        }
        if (config.enable_secondary_transport &&
            !config.fragment_species_dose_output_file.empty()) {
            carbon::write_fragment_species_dose_Gy_csv(
                config.fragment_species_dose_output_file, config, result);
        }
        if (config.enable_voxel_scoring && !config.voxel_dose_Gy_output_file.empty()) {
            carbon::write_sparse_voxel_dose_Gy_csv(
                config.voxel_dose_Gy_output_file, config, result);
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
                  << "Energy balance error: " << result.relative_energy_balance_error() << '\n'
                  << "Nuclear interactions: " << result.nuclear_interactions << '\n';
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
                          << "Cascade queue overflow: " << result.cascade_queue_overflow << '\n'
                          << "Fragment species output: "
                          << config.fragment_species_output_file.string() << '\n';
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
        if (config.enable_secondary_transport &&
            !config.fragment_species_dose_output_file.empty()) {
            std::cout << "Fragment-species dose (Gy): "
                      << config.fragment_species_dose_output_file.string() << '\n';
        }
        if (config.enable_voxel_scoring) {
            std::cout << "Voxel MeV scorer output: "
                      << config.voxel_dose_output_file.string() << '\n';
            if (!config.voxel_dose_Gy_output_file.empty()) {
                std::cout << "Voxel dose scorer (Gy): "
                          << config.voxel_dose_Gy_output_file.string() << '\n';
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
