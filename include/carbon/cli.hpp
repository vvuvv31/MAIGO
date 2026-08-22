#pragma once

#include "carbon/transport_config.hpp"

#include <filesystem>

namespace carbon {

struct CliState {
    std::filesystem::path config_path{"config/beam_200MeVu.yaml"};
    std::filesystem::path canonical_config_output_path{};
    bool help{false};
    bool histories_overridden{false};
    bool plan_only{false};
    bool sequential_spots{false};
};

void print_usage(const char* executable);
void parse_config_and_help(int argc, char** argv, CliState& state);
void apply_cli_overrides(int argc, char** argv, TransportConfig& config, CliState& state);

}  // namespace carbon
