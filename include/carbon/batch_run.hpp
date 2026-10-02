#pragma once
#include "carbon/cli.hpp"
#include "carbon/immutable_input_cache.hpp"
#include <vector>
namespace carbon {
struct RunShard {
    std::filesystem::path config_path;
    TransportConfig config;
    bool histories_overridden{false}, plan_only{false}, sequential_spots{false};
    std::filesystem::path output_directory{};
};
// Batch count/size are total incident primaries, never histories per spot.
std::vector<RunShard> make_repeated_batches(const RunShard&, const CliState&,
                                          ImmutableInputCache&);
void write_batch_manifest(const std::vector<RunShard>&, const CliState&);
}
