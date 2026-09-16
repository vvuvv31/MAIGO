#pragma once

#include "carbon/transport_config.hpp"

#include <filesystem>
#include <vector>

namespace carbon {

struct CliState {
    std::filesystem::path config_path{"config/unified_water_production.yaml"};
    std::filesystem::path canonical_config_output_path{};
    std::filesystem::path plan_manifest{};
    bool help{false};
    bool histories_overridden{false};
    bool plan_only{false};
    bool plan_preflight{false};
    bool sequential_spots{false};
};

void print_usage(const char* executable);
void parse_config_and_help(int argc, char** argv, CliState& state);
void apply_cli_overrides(int argc, char** argv, TransportConfig& config, CliState& state);

// Reads one shard config path per line (blank lines and '#' comments ignored);
// relative paths resolve against the manifest's directory.
[[nodiscard]] std::vector<std::filesystem::path> load_plan_manifest(
    const std::filesystem::path& manifest_path);

}  // namespace carbon
