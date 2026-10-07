#pragma once
#include "carbon/transport_config.hpp"
#include <filesystem>
namespace carbon {
struct CliState {
    std::filesystem::path config_path{"config/proton_water_broadbeam.yaml"};
    std::size_t batch_count{}, histories_per_batch{};
    std::filesystem::path batch_output{};
    bool help{}, histories_overridden{};
    std::size_t histories{};
    bool seed_overridden{};
    std::uint64_t seed{};
};
CliState parse_cli(int argc,char** argv);
void print_usage();
}
