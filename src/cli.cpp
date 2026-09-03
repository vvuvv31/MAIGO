#include "carbon/cli.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace carbon {

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--config FILE] [--device DEVICE] [--histories N]"
                 " [--spots FILE] [--straggling-scale X] [--output FILE]"
                 " [--dose-output FILE] [--scorer-let|--no-scorer-let]"
                 " [--let-output FILE] [--write-canonical-config FILE]"
                 " [--plan-only] [--sequential-spots]\n"
                 "  --device DEVICE      serial | cpu | gpu | default |\n"
                 "                       cuda|nvidia | level_zero|intel|arc | opencl\n"
                 "                       (gpu respects ONEAPI_DEVICE_SELECTOR;\n"
                 "                        cuda/level_zero pin the SYCL backend)\n"
                 "  --spots FILE         TOPAS-format spots_*.txt; repeat to concatenate files\n"
                 "  --spot-weights FILE  One optimization weight per concatenated spot\n"
                 "  --histories N        With weights: total plan histories; otherwise per spot\n"
                 "  --random-seed N|auto Override the configured RNG seed\n"
                 "  --ct-grid FILE       Override the configured CCTG patient grid\n"
                 "  --ct-stopping-power-scale X  Override the CT mass stopping-power scale\n"
                 "  --secondary-queue-capacity N  Override charged secondary queue capacity\n"
                 "  --neutral-queue-capacity N  Override neutral queue capacity\n"
                 "  --write-canonical-config FILE  Write strict normalized YAML input\n"
                 "  --plan-only          Parse/allocate/transform plan without transport\n"
                 "  --sequential-spots   Disable batched SYCL plan launch\n"
                 "  --output FILE        MeV energy-deposition scorer CSV\n"
                 "  --dose-output FILE   Dose scorer CSV (total Gy); empty disables\n"
                 "  --scorer-let         Enable primary and all-hadron LET_d scoring\n"
                 "  --no-scorer-let      Disable LET_d scoring (overrides YAML scorerLET)\n"
                 "  --let-output FILE    LET_d CSV including raw numerator/denominator\n"
                 "  --voxel-dose-mhd FILE  Override dense voxel dose MHD output\n";
}

void parse_config_and_help(int argc, char** argv, CliState& state) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--config" && index + 1 < argc) {
            state.config_path = argv[++index];
        } else if (argument == "--write-canonical-config" && index + 1 < argc) {
            state.canonical_config_output_path = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            state.help = true;
        }
    }
}

void apply_cli_overrides(int argc, char** argv, TransportConfig& config, CliState& state) {
    bool spots_cli_override = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--config") {
            ++index;
        } else if (argument == "--write-canonical-config" && index + 1 < argc) {
            ++index;
        } else if (argument == "--device" && index + 1 < argc) {
            config.device = argv[++index];
        } else if (argument == "--histories" && index + 1 < argc) {
            config.number_of_histories = std::stoull(argv[++index]);
            state.histories_overridden = true;
        } else if (argument == "--random-seed" && index + 1 < argc) {
            config.random_seed = parse_random_seed(argv[++index]);
        } else if (argument == "--ct-grid" && index + 1 < argc) {
            config.ct_grid_file = argv[++index];
        } else if (argument == "--ct-stopping-power-scale" && index + 1 < argc) {
            config.ct_stopping_power_scale = std::stod(argv[++index]);
        } else if (argument == "--ct-schneider-stopping-power-file" && index + 1 < argc) {
            config.ct_schneider_stopping_power_file = argv[++index];
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
            config.straggling_scale_energies_MeVu.clear();
            config.straggling_scale_values.clear();
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
            state.plan_only = true;
        } else if (argument == "--sequential-spots") {
            state.sequential_spots = true;
        } else if (argument != "--help" && argument != "-h") {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
}

}  // namespace carbon
