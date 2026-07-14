#include "carbon/io.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
        carbon::TransportResult result;
        if (config.device == "serial") {
            result = carbon::transport_serial(config, stopping_power);
        } else {
#ifdef CARBON_HAS_SYCL
            if (config.device != "cpu" && config.device != "gpu" && config.device != "default") {
                throw std::invalid_argument("SYCL device must be cpu, gpu, or default");
            }
            std::cout << "SYCL device: " << carbon::describe_sycl_device(config.device) << '\n';
            result = carbon::transport_sycl(config, stopping_power, config.device);
#else
            throw std::runtime_error(
                "This binary was built without SYCL. Reconfigure with CARBON_ENABLE_SYCL=ON and icpx.");
#endif
        }

        carbon::write_depth_dose_csv(config.output_file, config, result);
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
                  << "Output: " << config.output_file.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "carbon_mc: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
