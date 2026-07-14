#include "carbon/cascade_package.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/io.hpp"
#include "carbon/reaction_package.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--config FILE] [--device serial|cpu|gpu] [--histories N]"
                 " [--straggling-scale X] [--output FILE]\n";
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
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--config") {
                ++index;
            } else if (argument == "--device" && index + 1 < argc) {
                config.device = argv[++index];
            } else if (argument == "--histories" && index + 1 < argc) {
                config.number_of_histories = std::stoull(argv[++index]);
            } else if (argument == "--straggling-scale" && index + 1 < argc) {
                config.straggling_scale = std::stod(argv[++index]);
            } else if (argument == "--output" && index + 1 < argc) {
                config.output_file = argv[++index];
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
        carbon::TransportResult result;
        if (config.device == "serial") {
            if (config.enable_secondary_generation) {
                throw std::invalid_argument(
                    "Secondary generation is currently implemented only by the SYCL backend");
            }
            result = carbon::transport_serial(config, stopping_power, cross_section);
        } else {
#ifdef CARBON_HAS_SYCL
            if (config.device != "cpu" && config.device != "gpu" && config.device != "default") {
                throw std::invalid_argument("SYCL device must be cpu, gpu, or default");
            }
            std::cout << "SYCL device: " << carbon::describe_sycl_device(config.device) << '\n';
            result = carbon::transport_sycl(config, stopping_power, cross_section, config.device,
                                            reaction_packages ? &*reaction_packages : nullptr,
                                            cascade_packages ? &*cascade_packages : nullptr);
#else
            throw std::runtime_error(
                "This binary was built without SYCL. Reconfigure with CARBON_ENABLE_SYCL=ON and icpx.");
#endif
        }

        carbon::write_depth_dose_csv(config.output_file, config, result);
        if (config.enable_secondary_transport) {
            carbon::write_fragment_species_csv(
                config.fragment_species_output_file, config, result);
        }
        if (config.enable_voxel_scoring) {
            carbon::write_sparse_voxel_dose_csv(config.voxel_dose_output_file, config, result);
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
        }
        std::cout << "Untracked nuclear energy: " << result.untracked_nuclear_energy_MeV
                  << " MeV\n"
                  << "Output: " << config.output_file.string() << '\n';
        if (config.enable_voxel_scoring) {
            std::cout << "Voxel dose output: " << config.voxel_dose_output_file.string() << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_mc: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
