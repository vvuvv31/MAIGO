#include "carbon/batch_run.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/tps_source.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace carbon {
namespace {
std::vector<std::size_t> allocate(const std::vector<long double> &weights, std::size_t total) {
    // Largest remainder: round down every spot allocation, then assign the
    // remaining histories by descending fractional part (stable for ties).
    // Apply this once to the entire campaign, not independently in each batch.
    for (const auto weight : weights)
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("Invalid batch source weight");
    const auto sum = std::accumulate(weights.begin(), weights.end(), 0.L);
    if (!(sum > 0) || !std::isfinite(sum))
        throw std::invalid_argument("Invalid batch source weights");
    std::vector<std::size_t> counts(weights.size()), order(weights.size());
    std::vector<long double> remainder(weights.size());
    std::size_t assigned = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto exact = weights[i] / sum * total;
        counts[i] = static_cast<std::size_t>(std::floor(exact));
        remainder[i] = exact - counts[i];
        assigned += counts[i];
        order[i] = i;
    }
    if (assigned > total || total - assigned > weights.size())
        throw std::runtime_error("Batch source allocation overflow");
    std::stable_sort(order.begin(), order.end(),
                     [&](auto a, auto b) { return remainder[a] > remainder[b]; });
    for (std::size_t i = 0; i < total - assigned; ++i)
        ++counts[order[i]];
    return counts;
}
void relocate_outputs(TransportConfig &c, const std::filesystem::path &dir) {
    if (!c.voxel_dose_mhd_output_file.empty())
        c.voxel_dose_mhd_output_file = dir / c.voxel_dose_mhd_output_file.filename();
}
}
std::vector<RunShard> make_repeated_batches(const RunShard &original, const CliState &cli,
                                            ImmutableInputCache &inputs) {
    const auto n = cli.batch_count, h = cli.histories_per_batch;
    if (!n || !h || cli.batch_output.empty())
        throw std::invalid_argument(
            "Use --batch-count, --histories-per-batch and --batch-output together");
    if (h > std::numeric_limits<std::size_t>::max() / n)
        throw std::invalid_argument("Total batch history count overflows");
    const auto total = h * n;
    auto base = original.config;
    if (!base.primary_spot_batch.empty())
        throw std::invalid_argument(
            "Repeated batches do not accept an existing primary/secondary replay batch");
    std::vector<PrimarySpotBatchEntry> prototypes;
    std::vector<long double> weights;
    const StoppingPowerTable *air = nullptr;
    if (base.spots_enable_upstream_air_energy_loss)
        air = &inputs.get<StoppingPowerTable>(
            input_cache_key(base.spots_upstream_air_stopping_power_file), [&] {
                return StoppingPowerTable::from_csv(base.spots_upstream_air_stopping_power_file);
            });
    if (base.enable_tps_source) {
        auto plan = TpsSourcePlan::from_config(base);
        for (auto &spot : plan.spots) {
            if (spot.mu_weight > 0) {
                weights.push_back(spot.mu_weight);
                spot.mu_weight = 1;
            }
        }
        auto template_config = base;
        template_config.tps_spot_weight_mode = "histories";
        template_config.tps_histories_scale = 1;
        prototypes = plan.make_primary_batch(template_config, air);
        base.enable_emittance_source =
            std::any_of(prototypes.begin(), prototypes.end(), [](const auto &p) {
                return p.floats[2] > 0 || p.floats[3] > 0 || p.floats[4] > 0 || p.floats[5] > 0;
            });
    }
    if (base.enable_tps_source && weights.empty())
        throw std::invalid_argument("Batch plan has no positive source weights");
    if (weights.size() != prototypes.size())
        throw std::runtime_error("Batch source/weight mismatch");
    // Separate seeds for every (batch, spot). Queue/lane reordering below the
    // source layer must preserve the resulting history and seed identities.
    constexpr std::uint64_t spot_stride = 1000003;
    const auto streams = std::max<std::size_t>(1, prototypes.size());
    const auto max = std::numeric_limits<std::uint64_t>::max();
    if (streams > max / spot_stride || n > max / (streams * spot_stride) ||
        base.random_seed > max - n * (streams * spot_stride))
        throw std::invalid_argument("Repeated batch RNG seed range overflows");
    const auto counts = weights.empty() ? std::vector<std::size_t>{} : allocate(weights, total);
    // Balanced remainder placement: every batch has exactly H histories and
    // each spot receives its whole-campaign largest-remainder allocation.
    // Distribute each spot's remainder over a cyclic interval of batch indices;
    // starts/cursor carry the next interval forward without changing totals.
    std::vector<std::size_t> starts(counts.size());
    std::size_t cursor = 0;
    for (std::size_t s = 0; s < counts.size(); ++s) {
        starts[s] = cursor;
        cursor =
            (counts[s] % n >= n - cursor ? counts[s] % n - (n - cursor) : cursor + counts[s] % n);
    }
    if (!counts.empty() && cursor != 0)
        throw std::runtime_error("Batch remainder allocation mismatch");
    const auto root = std::filesystem::absolute(cli.batch_output).lexically_normal();
    if (std::filesystem::exists(root) && !std::filesystem::is_empty(root))
        throw std::invalid_argument("Batch output directory must be new or empty: " +
                                    root.string());
    std::vector<RunShard> batches;
    for (std::size_t i = 0; i < n; ++i) {
        std::ostringstream name;
        name << "batch" << std::setfill('0') << std::setw(6) << i;
        auto shard = original;
        shard.config = base;
        shard.output_directory = root / name.str();
        auto &c = shard.config;
        c.number_of_histories = h;
        c.random_seed = base.random_seed + i * streams * spot_stride;
        std::uint64_t begin = 0;
        for (std::size_t s = 0; s < counts.size(); ++s) {
            const auto count =
                counts[s] / n +
                ((i >= starts[s] ? i - starts[s] : n - (starts[s] - i)) < counts[s] % n ? 1 : 0);
            if (!count)
                continue;
            auto entry = prototypes[s];
            entry.history_begin = begin;
            entry.history_end = begin + count;
            entry.random_seed = c.random_seed + s * spot_stride;
            begin = entry.history_end;
            c.primary_spot_batch.push_back(entry);
        }
        if (!counts.empty() && begin != h)
            throw std::runtime_error("Batch incident count mismatch");
        if (!c.primary_spot_batch.empty()) {
            c.initial_energy_MeVu =
                c.primary_spot_batch.front().initial_energy_MeV() / c.primary_mass_number;
            c.beam_energy_spread = c.primary_spot_batch.front().beam_energy_spread();
        }
        relocate_outputs(c, shard.output_directory);
        c.validate();
        batches.push_back(std::move(shard));
    }
    return batches;
}
void write_batch_manifest(const std::vector<RunShard> &shards, const CliState &cli) {
    std::filesystem::create_directories(cli.batch_output);
    // quoted strings below are paths in CSV, doubled quotes for CSV escaping.
    std::ofstream out(cli.batch_output / "batches.csv");
    out << "batch,histories,seed,output_directory\n";
    std::ofstream provenance(cli.batch_output / "base_config.txt");
    provenance
        << "# Input: " << std::filesystem::absolute(shards.front().config_path)
        << "\n# Base normalized configuration; batch overrides are in batches.csv and primary_batches.csv.\n"
        << shards.front().config.canonical_config_text;
    if (!provenance)
        throw std::runtime_error("Cannot write batch input provenance");
    for (std::size_t i = 0; i < shards.size(); ++i) {
        const auto &shard = shards[i];
        std::filesystem::create_directories(shard.output_directory);
        auto escaped = shard.output_directory.string();
        std::size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos) {
            escaped.insert(pos, "\"");
            pos += 2;
        }
        out << i << ',' << shard.config.number_of_histories << ',' << shard.config.random_seed
            << ",\"" << escaped << "\"\n";
        std::ofstream source(shard.output_directory / "primary_batches.csv");
        source << "history_begin,history_end,seed";
        for (int j = 0; j < PrimarySpotBatchEntry::k_float_count; ++j)
            source << ",source_float_" << j;
        source << '\n' << std::setprecision(9);
        for (const auto &spot : shard.config.primary_spot_batch) {
            source << spot.history_begin << ',' << spot.history_end << ',' << spot.random_seed;
            for (float v : spot.floats)
                source << ',' << v;
            source << '\n';
        }
        if (!source)
            throw std::runtime_error("Cannot write batch source manifest");
    }
    if (!out)
        throw std::runtime_error("Cannot write batch manifest");
}
}
