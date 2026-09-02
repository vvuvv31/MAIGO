#include "carbon/transport_config.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/straggling.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace carbon {
namespace {

std::uint64_t mix_seed(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t generate_auto_seed() {
    static std::atomic<std::uint64_t> sequence{0};
    auto seed = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= mix_seed(sequence.fetch_add(1, std::memory_order_relaxed));
    try {
        std::random_device entropy;
        seed ^= static_cast<std::uint64_t>(entropy()) << 32U;
        seed ^= static_cast<std::uint64_t>(entropy());
    } catch (const std::exception&) {
        // The clock and process-local sequence still produce a changing seed
        // on systems where std::random_device is unavailable.
    }
    return mix_seed(seed);
}

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

class ConfigValues
    : public std::unordered_map<std::string, std::string> {
public:
    using Base = std::unordered_map<std::string, std::string>;
    using Base::Base;
    using Base::end;

    std::pair<iterator, bool> add(std::string key, std::string value,
                                  const std::size_t line_number) {
        auto result = Base::emplace(std::move(key), std::move(value));
        if (result.second) {
            line_numbers_.emplace(result.first->first, line_number);
        }
        return result;
    }

    iterator find(const key_type& key) {
        auto iterator = Base::find(key);
        if (iterator != Base::end()) {
            consumed_.insert(iterator->first);
        }
        return iterator;
    }

    const_iterator find(const key_type& key) const {
        auto iterator = Base::find(key);
        if (iterator != Base::end()) {
            consumed_.insert(iterator->first);
        }
        return iterator;
    }

    bool contains(const key_type& key) const {
        return find(key) != Base::end();
    }

    [[nodiscard]] bool consumed(const key_type& key) const {
        return consumed_.contains(key);
    }

    [[nodiscard]] std::size_t line_number(const key_type& key) const {
        const auto iterator = line_numbers_.find(key);
        return iterator == line_numbers_.end() ? 0U : iterator->second;
    }

private:
    std::unordered_map<std::string, std::size_t> line_numbers_;
    mutable std::unordered_set<std::string> consumed_;
};

ConfigValues read_key_values(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open configuration file: " + path.string());
    }

    ConfigValues values;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("Expected 'key: value' at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        auto key = trim(line.substr(0, separator));
        auto value = trim(line.substr(separator + 1));
        // Empty value is allowed (disables optional outputs such as sparse voxel CSV).
        if (key.empty()) {
            throw std::runtime_error("Empty configuration key at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        const auto [existing, inserted] = values.add(
            key, std::move(value), line_number);
        if (!inserted) {
            throw std::runtime_error(
                "Duplicate configuration key '" + key + "' at " +
                path.string() + ":" + std::to_string(line_number) +
                " (first defined at line " +
                std::to_string(values.line_number(key)) + ")");
        }
    }
    return values;
}

std::filesystem::path resolve_input_path_from_config(
    const std::filesystem::path& configured,
    const std::filesystem::path& config_path) {
    if (configured.empty() || configured.is_absolute() ||
        std::filesystem::exists(configured)) {
        return configured;
    }
    auto directory = std::filesystem::absolute(config_path).parent_path();
    while (!directory.empty()) {
        const auto candidate = directory / configured;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        const auto parent = directory.parent_path();
        if (parent == directory) {
            break;
        }
        directory = parent;
    }
    return configured;
}

const std::unordered_set<std::string>& ion_physics_manifest_keys() {
    static const std::unordered_set<std::string> keys{
        "primary_atomic_number",
        "primary_mass_number",
        "primary_rest_mass_MeV",
        "energy_straggling_model",
        "use_particle_specific_stopping_power",
        "primary_stopping_power_file",
        "let_delta_electron_fraction_file",
        "particle_stopping_power_file",
    };
    return keys;
}

const std::unordered_set<std::string>& ion_physics_manifest_path_keys() {
    static const std::unordered_set<std::string> keys{
        "primary_stopping_power_file",
        "let_delta_electron_fraction_file",
        "particle_stopping_power_file",
    };
    return keys;
}

std::filesystem::path merge_ion_physics_manifest(
    const std::filesystem::path& config_path,
    ConfigValues& values) {
    const auto manifest_entry = values.find("ion_physics_file");
    if (manifest_entry == values.end()) {
        return {};
    }
    if (manifest_entry->second.empty()) {
        throw std::invalid_argument("ion_physics_file must not be empty");
    }
    const auto manifest_path = resolve_input_path_from_config(
        std::filesystem::path{manifest_entry->second}, config_path);
    const auto manifest = read_key_values(manifest_path);
    const auto& allowed = ion_physics_manifest_keys();
    for (const auto& [key, value] : manifest) {
        if (!allowed.contains(key)) {
            throw std::invalid_argument(
                "Ion physics manifest contains non-physics key '" + key +
                "': " + manifest_path.string());
        }
        if (values.contains(key)) {
            throw std::invalid_argument(
                "Ion-dependent key '" + key + "' is defined in both " +
                config_path.string() + " and " + manifest_path.string());
        }
        auto imported = value;
        if (ion_physics_manifest_path_keys().contains(key) && !value.empty()) {
            imported = resolve_input_path_from_config(
                           std::filesystem::path{value}, manifest_path)
                           .string();
        }
        values.add(key, std::move(imported), manifest.line_number(key));
    }
    for (const char* required : {
             "primary_atomic_number",
             "primary_mass_number",
             "energy_straggling_model",
             "use_particle_specific_stopping_power",
             "primary_stopping_power_file",
             "particle_stopping_power_file",
         }) {
        const auto entry = manifest.find(required);
        if (entry == manifest.end() || entry->second.empty()) {
            throw std::invalid_argument(
                "Ion physics manifest is missing required key '" +
                std::string(required) + "': " + manifest_path.string());
        }
    }
    return manifest_path;
}

std::filesystem::path scorer_output_path(const std::filesystem::path& config_path,
                                         const std::string& name,
                                         const bool include_extension) {
    if (name.empty()) {
        throw std::invalid_argument("Scorer output name must not be empty");
    }
    std::filesystem::path output{name};
    if (output.is_absolute() || output.has_parent_path()) {
        throw std::invalid_argument(
            "Scorer output name must be a basename without a directory");
    }
    if (include_extension) {
        if (output.extension().empty()) {
            output += ".mhd";
        } else if (output.extension() != ".mhd") {
            throw std::invalid_argument(
                "MHD scorer output name must have no extension or use .mhd");
        }
    } else if (output.has_extension()) {
        throw std::invalid_argument(
            "LET MHD output name is a prefix and must not have an extension");
    }
    const auto stem = config_path.stem().empty() ? std::filesystem::path{"run"}
                                                 : config_path.stem();
    return std::filesystem::path{"out"} / stem / output;
}

template <typename Number>
Number parse_number(const ConfigValues& values,
                    const std::string& key,
                    Number fallback) {
    const auto iterator = values.find(key);
    if (iterator == values.end()) {
        return fallback;
    }
    std::size_t parsed = 0;
    if constexpr (std::is_integral_v<Number>) {
        const auto number = std::stoull(iterator->second, &parsed);
        if (parsed != iterator->second.size()) {
            throw std::runtime_error("Invalid integer for '" + key + "': " + iterator->second);
        }
        return static_cast<Number>(number);
    } else {
        const auto number = std::stod(iterator->second, &parsed);
        if (parsed != iterator->second.size()) {
            throw std::runtime_error("Invalid number for '" + key + "': " + iterator->second);
        }
        return static_cast<Number>(number);
    }
}

std::filesystem::path parse_path(const ConfigValues& values,
                                 const std::string& key,
                                 const std::filesystem::path& fallback) {
    const auto iterator = values.find(key);
    return iterator == values.end() ? fallback : std::filesystem::path(iterator->second);
}

bool parse_bool(const ConfigValues& values,
                const std::string& key,
                bool fallback) {
    const auto iterator = values.find(key);
    if (iterator == values.end()) {
        return fallback;
    }
    auto value = iterator->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (value == "true" || value == "yes" || value == "1") {
        return true;
    }
    if (value == "false" || value == "no" || value == "0") {
        return false;
    }
    throw std::runtime_error("Invalid boolean for '" + key + "': " + iterator->second);
}

void reject_unknown_config_keys(const ConfigValues& values,
                                const std::filesystem::path& path) {
    const auto unknown = std::min_element(
        values.begin(), values.end(), [&values](const auto& left, const auto& right) {
            const auto left_line = values.consumed(left.first)
                                       ? std::numeric_limits<std::size_t>::max()
                                       : values.line_number(left.first);
            const auto right_line = values.consumed(right.first)
                                        ? std::numeric_limits<std::size_t>::max()
                                        : values.line_number(right.first);
            return left_line < right_line;
        });
    if (unknown != values.end() && !values.consumed(unknown->first)) {
        throw std::runtime_error(
            "Unknown configuration key '" + unknown->first + "' at " +
            path.string() + ":" +
            std::to_string(values.line_number(unknown->first)));
    }
}

std::string canonicalize_config(const ConfigValues& values,
                                const std::uint32_t schema_version) {
    std::vector<std::string> keys;
    keys.reserve(values.size());
    for (const auto& [key, value] : values) {
        (void)value;
        if (key != "config_schema_version") {
            keys.push_back(key);
        }
    }
    std::sort(keys.begin(), keys.end());

    std::ostringstream output;
    output << "config_schema_version: " << schema_version << '\n';
    for (const auto& key : keys) {
        output << key << ": " << values.at(key) << '\n';
    }
    return output.str();
}

std::vector<double> parse_double_list(const std::string& text, const std::string& key) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        std::size_t parsed = 0;
        const auto number = std::stod(token, &parsed);
        if (parsed != token.size()) {
            throw std::runtime_error("Invalid number in '" + key + "': " + token);
        }
        values.push_back(number);
    }
    return values;
}

std::vector<std::filesystem::path> parse_path_list(const std::string& text) {
    std::vector<std::filesystem::path> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        values.emplace_back(token);
    }
    return values;
}

}  // namespace

const char* run_mode_name(const RunMode mode) noexcept {
    switch (mode) {
    case RunMode::smoke: return "smoke";
    case RunMode::research: return "research";
    case RunMode::production: return "production";
    }
    return "unknown";
}

std::uint64_t parse_random_seed(const std::string_view value) {
    if (value == "auto") {
        return generate_auto_seed();
    }
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(), [](const unsigned char character) {
            return std::isdigit(character) != 0;
        })) {
        throw std::runtime_error(
            "Invalid random_seed: " + std::string(value) +
            " (expected an unsigned integer or 'auto')");
    }
    std::size_t parsed = 0;
    try {
        const auto seed = std::stoull(std::string(value), &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return seed;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Invalid random_seed: " + std::string(value) +
            " (expected an unsigned integer or 'auto')");
    }
}

void validate_slab_layers(const std::vector<SlabLayer>& layers, double phantom_length_mm) {
    if (layers.empty()) {
        throw std::invalid_argument("enable_layered_phantom requires at least one slab layer");
    }
    double previous_end = 0.0;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        if (layer.density_g_per_cm3 <= 0.0) {
            throw std::invalid_argument("slab layer density must be positive");
        }
        if (layer.z_end_mm <= previous_end) {
            throw std::invalid_argument(
                "slab_z_ends_mm must be strictly increasing from the entrance");
        }
        previous_end = layer.z_end_mm;
    }
    if (std::abs(layers.back().z_end_mm - phantom_length_mm) > 1.0e-6) {
        throw std::invalid_argument(
            "last slab_z_ends_mm entry must equal phantom_length_mm");
    }
}

void validate_hetero_insert(const HeteroInsert& insert, double phantom_length_mm) {
    if (!(insert.x_min_mm < insert.x_max_mm) || !(insert.y_min_mm < insert.y_max_mm) ||
        !(insert.z_min_mm < insert.z_max_mm)) {
        throw std::invalid_argument(
            "hetero insert bounds must satisfy min < max on each axis");
    }
    if (insert.z_min_mm < 0.0 || insert.z_max_mm > phantom_length_mm + 1.0e-6) {
        throw std::invalid_argument(
            "hetero insert z range must lie within [0, phantom_length_mm]");
    }
    if (insert.density_g_per_cm3 <= 0.0) {
        throw std::invalid_argument("hetero insert density must be positive");
    }
}

std::size_t TransportConfig::number_of_bins() const {
    return voxel_bins_z != 0 ? voxel_bins_z
                             : static_cast<std::size_t>(
                                   std::ceil(phantom_length_mm / depth_bin_width_mm));
}

std::size_t TransportConfig::number_of_voxels() const {
    const auto depth_bins = number_of_bins();
    if (voxel_bins_y != 0 &&
        (voxel_bins_x > std::numeric_limits<std::size_t>::max() / voxel_bins_y ||
         voxel_bins_x * voxel_bins_y >
             std::numeric_limits<std::size_t>::max() / depth_bins)) {
        throw std::overflow_error("Voxel grid size exceeds the size_t runtime limit");
    }
    return voxel_bins_x * voxel_bins_y * depth_bins;
}

bool TransportConfig::validation_scorers() const noexcept {
#ifdef CARBON_VALIDATION_SCORERS
    return scorer_mode == "validation";
#else
    return false;
#endif
}

void TransportConfig::validate() const {
    if (config_schema_version != 1U) {
        throw std::invalid_argument(
            "config_schema_version must be 1 for the current flat configuration schema");
    }
    if (!std::isfinite(quality_maximum_relative_energy_residual) ||
        quality_maximum_relative_energy_residual < 0.0 ||
        !std::isfinite(quality_maximum_absolute_energy_residual_MeV) ||
        quality_maximum_absolute_energy_residual_MeV < 0.0) {
        throw std::invalid_argument(
            "run quality energy-residual tolerances must be finite and non-negative");
    }
    if (run_mode == RunMode::production &&
        (!quality_reject_any_queue_overflow || !quality_reject_nan_or_inf)) {
        throw std::invalid_argument(
            "production run_mode requires queue-overflow and NaN/Inf rejection");
    }
    if (enable_csda_range_energy_loss &&
        (enable_ct_grid || enable_layered_phantom || enable_hetero_insert ||
         enable_minibeam || use_particle_specific_stopping_power)) {
        throw std::invalid_argument(
            "enable_csda_range_energy_loss requires homogeneous water primary "
            "transport without CT, layered, insert, minibeam, or particle-specific materials");
    }
    if (number_of_histories == 0) {
        throw std::invalid_argument("number_of_histories must be greater than zero");
    }
    if (scorer_mode != "production" && scorer_mode != "validation") {
        throw std::invalid_argument(
            "scorer_mode must be production or validation");
    }
    if (scorer_mode == "validation" && !validation_scorers()) {
        throw std::invalid_argument(
            "scorer_mode=validation requires a binary built with "
            "-DCARBON_VALIDATION_SCORERS=ON");
    }
    if (primary_atomic_number <= 0 || primary_mass_number < primary_atomic_number) {
        throw std::invalid_argument(
            "primary ion must satisfy primary_atomic_number > 0 and "
            "primary_mass_number >= primary_atomic_number");
    }
    if (!std::isfinite(primary_rest_mass_MeV) || primary_rest_mass_MeV < 0.0) {
        throw std::invalid_argument(
            "primary_rest_mass_MeV must be zero or finite and positive");
    }
    const auto energy_comes_from_spot_file =
        !topas_spots_file.empty() || !topas_spots_files.empty() ||
        !tps_spots_file.empty();
    if (!energy_comes_from_spot_file && initial_energy_MeVu <= 0.0) {
        throw std::invalid_argument(
            "initial_energy_MeVu must be positive when no spot file supplies energy");
    }
    if (beam_energy_spread < 0.0 || beam_energy_spread > 0.2) {
        throw std::invalid_argument("beam_energy_spread must be in [0, 0.2] (relative RMS)");
    }
    if (phantom_length_mm <= 0.0 || depth_bin_width_mm <= 0.0 || maximum_step_mm <= 0.0) {
        throw std::invalid_argument("phantom and step lengths must be positive");
    }
    if (maximum_relative_energy_loss <= 0.0 || maximum_relative_energy_loss > 1.0) {
        throw std::invalid_argument("maximum_relative_energy_loss must be in (0, 1]");
    }
    if (energy_cutoff_MeV < 0.0 || water_density_g_per_cm3 <= 0.0 || scorer_area_mm2 <= 0.0) {
        throw std::invalid_argument("cutoff must be nonnegative; density and scorer area must be positive");
    }
    if (!std::isfinite(dose_output_scale) || dose_output_scale <= 0.0) {
        throw std::invalid_argument("dose_output_scale must be finite and positive");
    }
    if (secondary_local_deposit_cutoff_MeV < 0.0) {
        throw std::invalid_argument(
            "secondary_local_deposit_cutoff_MeV must be non-negative");
    }
    if (secondary_heavy_local_deposit_z_min < 0 ||
        secondary_heavy_local_deposit_z_min > 20) {
        throw std::invalid_argument(
            "secondary_heavy_local_deposit_z_min must be in [0, 20] "
            "(0 disables heavy local deposit)");
    }
    if (secondary_condensed_step_mm < 0.0) {
        throw std::invalid_argument(
            "secondary_condensed_step_mm must be non-negative");
    }
    if (enable_layered_phantom) {
        validate_slab_layers(slab_layers, phantom_length_mm);
        const auto has_sp = !slab_stopping_power_files.empty();
        const auto has_xs = !slab_cross_section_files.empty();
        if (has_sp != has_xs) {
            throw std::invalid_argument(
                "slab_stopping_power_files and slab_cross_section_files must both be set "
                "or both omitted");
        }
        if (has_sp) {
            if (slab_stopping_power_files.size() != slab_layers.size() ||
                slab_cross_section_files.size() != slab_layers.size()) {
                throw std::invalid_argument(
                    "slab material table file lists must match slab layer count");
            }
        }
        if (!slab_radiation_lengths_g_per_cm2.empty()) {
            if (slab_radiation_lengths_g_per_cm2.size() != slab_layers.size()) {
                throw std::invalid_argument(
                    "slab radiation-length list must match slab layer count");
            }
            for (const auto radiation_length :
                 slab_radiation_lengths_g_per_cm2) {
                if (!std::isfinite(radiation_length) ||
                    radiation_length <= 0.0) {
                    throw std::invalid_argument(
                        "slab radiation lengths must be finite and positive");
                }
            }
        }
    } else if (!slab_layers.empty()) {
        throw std::invalid_argument(
            "slab_layers is set but enable_layered_phantom=false");
    }
    if (enable_hetero_insert) {
        validate_hetero_insert(hetero_insert, phantom_length_mm);
        if (!std::isfinite(insert_radiation_length_g_per_cm2) ||
            insert_radiation_length_g_per_cm2 <= 0.0) {
            throw std::invalid_argument(
                "insert radiation length must be finite and positive");
        }
        const auto has_sp = !insert_stopping_power_file.empty();
        const auto has_xs = !insert_cross_section_file.empty();
        if (has_sp != has_xs) {
            throw std::invalid_argument(
                "insert_stopping_power_file and insert_cross_section_file must both be set "
                "or both omitted");
        }
    }
    if (enable_hetero_insert && enable_layered_phantom) {
        throw std::invalid_argument(
            "enable_hetero_insert and enable_layered_phantom cannot both be true "
            "(use one heterogeneity model per run)");
    }
    if (enable_ct_grid) {
        if (ct_grid_file.empty()) {
            throw std::invalid_argument("enable_ct_grid requires ct_grid_file");
        }
        if (enable_layered_phantom || enable_hetero_insert) {
            throw std::invalid_argument(
                "enable_ct_grid cannot combine with layered phantom or hetero insert");
        }
    }
    if (!std::isfinite(straggling_scale) || straggling_scale < 0.0) {
        throw std::invalid_argument("straggling_scale must be finite and nonnegative");
    }
    if (!std::isfinite(straggling_sampling_length_mm) ||
        straggling_sampling_length_mm < 0.0) {
        throw std::invalid_argument(
            "straggling_sampling_length_mm must be finite and nonnegative");
    }
    if (enable_step_stable_straggling && straggling_sampling_length_mm <= 0.0) {
        throw std::invalid_argument(
            "enable_step_stable_straggling requires straggling_sampling_length_mm > 0");
    }
    if (energy_straggling_model != "gaussian_clamped" &&
        energy_straggling_model != "legacy_calibrated" &&
        energy_straggling_model != "moment_matched" &&
        energy_straggling_model != "vavilov_landau" &&
        energy_straggling_model != "packaged_fluctuation") {
        throw std::invalid_argument(
            "energy_straggling_model must be gaussian_clamped, "
            "legacy_calibrated, moment_matched, vavilov_landau, or packaged_fluctuation");
    }
    if (uses_packaged_fluctuation() && energy_straggling_package_file.empty()) {
        throw std::invalid_argument(
            "packaged_fluctuation requires energy_straggling_package_file");
    }
    if (nuclear_model != "geant4" && nuclear_model != "fred_paper" &&
        nuclear_model != "cinel02") {
        throw std::invalid_argument(
            "nuclear_model must be geant4, fred_paper, or cinel02");
    }
    if (nuclear_model == "cinel02" &&
        (primary_inelastic_package_v2_file.empty() ||
         primary_inelastic_rate_v2_file.empty())) {
        throw std::invalid_argument(
            "nuclear_model: cinel02 requires both primary_inelastic_package_v2_file "
            "and primary_inelastic_rate_v2_file");
    }
    if (cinel02_max_secondary_inelastic_generations > 2U) {
        throw std::invalid_argument(
            "cinel02_max_secondary_inelastic_generations must be between 0 and 2");
    }
    if (cinel02_topas_compatibility_mode && nuclear_model != "cinel02") {
        throw std::invalid_argument(
            "cinel02_topas_compatibility_mode requires nuclear_model: cinel02");
    }
    if (enable_nuclear_elastic && nuclear_model != "fred_paper") {
        throw std::invalid_argument(
            "enable_nuclear_elastic requires nuclear_model: fred_paper");
    }
    if (uses_moment_matched_straggling() &&
        std::abs(straggling_scale - 1.0) > 1.0e-12) {
        throw std::invalid_argument(
            "uncalibrated straggling models require straggling_scale: 1.0");
    }
    if (uses_moment_matched_straggling() &&
        enable_step_stable_straggling) {
        throw std::invalid_argument(
            "uncalibrated straggling models cannot use the legacy fixed-block sampler");
    }
    if (uses_moment_matched_straggling() &&
        !straggling_scale_energies_MeVu.empty()) {
        throw std::invalid_argument(
            "uncalibrated straggling models reject energy-dependent "
            "straggling_scale tables; use a scalar straggling_scale of 1");
    }
    if (straggling_scale_energies_MeVu.size() != straggling_scale_values.size()) {
        throw std::invalid_argument(
            "straggling scale energy and value tables must have the same length");
    }
    if (!straggling_scale_energies_MeVu.empty()) {
        if (straggling_scale_energies_MeVu.size() < 2) {
            throw std::invalid_argument(
                "straggling scale table requires at least two points");
        }
        if (straggling_scale_energies_MeVu.size() > max_straggling_scale_points) {
            throw std::invalid_argument(
                "straggling scale table has too many points (maximum is 16)");
        }
        for (std::size_t index = 0; index < straggling_scale_energies_MeVu.size(); ++index) {
            const auto energy = straggling_scale_energies_MeVu[index];
            const auto scale = straggling_scale_values[index];
            if (!std::isfinite(energy) || energy < 0.0) {
                throw std::invalid_argument(
                    "straggling scale table energies must be finite and nonnegative");
            }
            if (!std::isfinite(scale) || scale < 0.0) {
                throw std::invalid_argument(
                    "straggling scale table values must be finite and nonnegative");
            }
            if (index > 0 && energy <= straggling_scale_energies_MeVu[index - 1]) {
                throw std::invalid_argument(
                    "straggling scale table energies must be strictly increasing");
            }
        }
    }
    if (multiple_scattering_model != "highland" &&
        multiple_scattering_model != "fred_2gr") {
        throw std::invalid_argument(
            "multiple_scattering_model must be highland or fred_2gr");
    }
    if (uses_fred_2gr_mcs() && fred_2gr_mcs_file.empty()) {
        throw std::invalid_argument("fred_2gr requires fred_2gr_mcs_file");
    }
    if (fred_2gr_high_energy_mode != "zero" &&
        fred_2gr_high_energy_mode != "kinematic_extrapolation") {
        throw std::invalid_argument(
            "fred_2gr_high_energy_mode must be zero or kinematic_extrapolation");
    }
    if (!uses_fred_2gr_mcs() && fred_2gr_high_energy_mode != "zero") {
        throw std::invalid_argument(
            "fred_2gr_high_energy_mode requires multiple_scattering_model: fred_2gr");
    }

    if (!std::isfinite(multiple_scattering_scale) ||
        multiple_scattering_scale < 0.0 || multiple_scattering_scale > 3.0) {
        throw std::invalid_argument(
            "multiple_scattering_scale must be in [0, 3]");
    }

    if (!std::isfinite(spots_emittance_sigma_scale) ||
        spots_emittance_sigma_scale < 0.5 || spots_emittance_sigma_scale > 1.5) {
        throw std::invalid_argument(
            "spots_emittance_sigma_scale must be in [0.5, 1.5]");
    }
    if (!std::isfinite(spots_emittance_prime_scale) ||
        spots_emittance_prime_scale < 0.5 || spots_emittance_prime_scale > 1.5) {
        throw std::invalid_argument(
            "spots_emittance_prime_scale must be in [0.5, 1.5]");
    }
    if (ct_stopping_power_scale <= 0.0 || ct_stopping_power_scale > 2.0) {
        throw std::invalid_argument(
            "ct_stopping_power_scale must be in (0, 2]");
    }
    if (restrict_fragment_species_energy_deposit) {
        if (restrict_fragment_species_z_min < 1 ||
            restrict_fragment_species_z_max < restrict_fragment_species_z_min ||
            restrict_fragment_species_z_max > 8) {
            throw std::invalid_argument(
                "restrict_fragment_species_z_min/max must satisfy "
                "1 <= min <= max <= 8 when restrict_fragment_species_energy_deposit "
                "is true");
        }
    }
    if (enable_voxel_scoring &&
        (voxel_bins_x == 0 || voxel_bins_y == 0 || voxel_size_x_mm <= 0.0 ||
         voxel_size_y_mm <= 0.0 ||
         (voxel_bins_z != 0 && voxel_size_z_mm <= 0.0))) {
        throw std::invalid_argument(
            "Enabled voxel scoring requires positive x/y/z bin counts and voxel sizes");
    }
    if (voxel_bins_z != 0) {
        const auto scorer_z_extent_mm =
            static_cast<double>(voxel_bins_z) * voxel_size_z_mm;
        if (std::abs(voxel_size_z_mm - depth_bin_width_mm) > 1.0e-6 ||
            std::abs(scorer_z_extent_mm - phantom_length_mm) > 1.0e-6) {
            throw std::invalid_argument(
                "voxel z geometry must match the legacy phantom z aliases");
        }
    }
    if (enable_voxel_scoring) {
        static_cast<void>(number_of_voxels());
    }
    if (enable_charged_origin_voxel_scoring && !enable_voxel_scoring) {
        throw std::invalid_argument(
            "enable_charged_origin_voxel_scoring requires enable_voxel_scoring=true");
    }
    if (!charged_origin_voxel_mhd_output_prefix.empty() &&
        !enable_charged_origin_voxel_scoring) {
        throw std::invalid_argument(
            "charged_origin_voxel_mhd_output_prefix requires "
            "enable_charged_origin_voxel_scoring=true");
    }
    if (!let_voxel_mhd_output_file.empty() &&
        (!enable_let_scoring || !enable_voxel_scoring)) {
        throw std::invalid_argument(
            "let_voxel_mhd_output_file requires scorerLET=true and "
            "enable_voxel_scoring=true");
    }
    if (!fragment_species_let_output_file.empty() && !enable_let_scoring) {
        throw std::invalid_argument(
            "fragment_species_let_output_file requires scorerLET=true");
    }
    if (!light_isotope_let_output_file.empty() && !enable_let_scoring) {
        throw std::invalid_argument(
            "light_isotope_let_output_file requires scorerLET=true");
    }
    if (enable_flat_source &&
        (flat_source_half_width_x_mm <= 0.0 || flat_source_half_width_y_mm <= 0.0)) {
        throw std::invalid_argument(
            "enable_flat_source requires positive flat_source_half_width_x_mm and "
            "flat_source_half_width_y_mm");
    }
    if (enable_flat_source && enable_emittance_source) {
        throw std::invalid_argument(
            "enable_flat_source and enable_emittance_source cannot both be true");
    }
    if (enable_electron_transport) {
        if (enable_ct_grid || enable_layered_phantom || enable_hetero_insert ||
            enable_minibeam) {
            throw std::invalid_argument(
                "electron transport v1 supports homogeneous G4_WATER only; "
                "CT, layered, insert, and minibeam geometries are unsupported");
        }
        if (electron_transport_data_file.empty()) {
            throw std::invalid_argument(
                "enable_electron_transport requires electron_transport_data_file");
        }
        const auto table = ElectronTransportTable::from_csv(
            electron_transport_data_file);
        if (table.kinetic_energies_MeV().front() > 0.001 ||
            table.kinetic_energies_MeV().back() < 500.0) {
            throw std::invalid_argument(
                "electron transport table must cover at least 0.001 to 500 MeV");
        }
        if (device == "serial") {
            throw std::invalid_argument(
                "electron transport is not implemented by the serial CPU backend");
        }
    } else if (!electron_transport_data_file.empty() || electron_queue_capacity != 0) {
        throw std::invalid_argument(
            "electron transport data/queue fields require enable_electron_transport=true");
    }
    if (!std::isfinite(electron_kinetic_cutoff_MeV) ||
        electron_kinetic_cutoff_MeV <= 0.0) {
        throw std::invalid_argument(
            "electron_kinetic_cutoff_MeV must be finite and positive");
    }
    if (maximum_electromagnetic_generations == 0) {
        throw std::invalid_argument(
            "maximum_electromagnetic_generations must be greater than zero");
    }
    if (!std::isfinite(maximum_electron_step_mm) || maximum_electron_step_mm <= 0.0) {
        throw std::invalid_argument(
            "maximum_electron_step_mm must be finite and positive");
    }
    if (!std::isfinite(maximum_electron_relative_energy_loss) ||
        maximum_electron_relative_energy_loss <= 0.0 ||
        maximum_electron_relative_energy_loss > 1.0) {
        throw std::invalid_argument(
            "maximum_electron_relative_energy_loss must be in (0, 1]");
    }
    if (max_device_memory_fraction <= 0.05 || max_device_memory_fraction > 1.0) {
        throw std::invalid_argument(
            "max_device_memory_fraction must be in (0.05, 1.0]");
    }
    if (electronic_buildup_fraction < 0.0 || electronic_buildup_fraction > 0.5) {
        throw std::invalid_argument(
            "electronic_buildup_fraction must be in [0, 0.5]");
    }
    if (electronic_buildup_mfp_mm < 0.0) {
        throw std::invalid_argument(
            "electronic_buildup_mfp_mm must be nonnegative");
    }
    if (electronic_buildup_lateral_sigma_mm < 0.0) {
        throw std::invalid_argument(
            "electronic_buildup_lateral_sigma_mm must be nonnegative");
    }
    if (electronic_buildup_lateral_sigma_mm > 0.0 &&
        electronic_buildup_fraction <= 0.0) {
        throw std::invalid_argument(
            "electronic_buildup_lateral_sigma_mm > 0 requires "
            "electronic_buildup_fraction > 0");
    }
    if (electronic_buildup_fraction > 0.0 && electronic_buildup_mfp_mm <= 0.0) {
        throw std::invalid_argument(
            "electronic_buildup_fraction > 0 requires electronic_buildup_mfp_mm > 0");
    }
    if (enable_emittance_source) {
        if (emittance_sigma_x_mm < 0.0 || emittance_sigma_y_mm < 0.0 ||
            emittance_sigma_x_prime < 0.0 || emittance_sigma_y_prime < 0.0) {
            throw std::invalid_argument(
                "emittance sigmas must be non-negative");
        }
        if (emittance_correlation_x < -1.0 || emittance_correlation_x > 1.0 ||
            emittance_correlation_y < -1.0 || emittance_correlation_y > 1.0) {
            throw std::invalid_argument(
                "emittance correlations must be in [-1, 1]");
        }
    }
    if (spots_sad_mm <= 0.0) {
        throw std::invalid_argument("spots_sad_mm must be positive");
    }
    if (spots_geometry_mode != "topas" && spots_geometry_mode != "beam_plus_z" &&
        spots_geometry_mode != "tps_90" &&
        spots_geometry_mode != "tps_gantry_y" &&
        spots_geometry_mode != "minibeam_topas_y") {
        throw std::invalid_argument(
            "spots_geometry_mode must be topas, beam_plus_z, tps_90, "
            "tps_gantry_y, or minibeam_topas_y");
    }
    if (enable_minibeam) {
#if !defined(CARBON_ENABLE_MINIBEAM)
        throw std::invalid_argument(
            "minibeam=true requires a build configured with "
            "CARBON_ENABLE_MINIBEAM=ON");
#endif
        if (device == "serial") {
            throw std::invalid_argument(
                "minibeam=true currently requires a SYCL device");
        }
        if (primary_atomic_number != 6 || primary_mass_number != 12) {
            throw std::invalid_argument(
                "minibeam=true is currently calibrated only for a C-12 primary; "
                "disable minibeam for other configured ions");
        }
        if (minibeam_transport_mode != "absorbing_geometry" &&
            minibeam_transport_mode != "copper_em") {
            throw std::invalid_argument(
                "minibeam_transport_mode must be absorbing_geometry or copper_em");
        }
        if (minibeam_material != "Copper") {
            throw std::invalid_argument(
                "the current minibeam reference supports minibeam_material=Copper");
        }
        if (!(minibeam_radius_mm > 0.0 && minibeam_thickness_mm > 0.0 &&
              minibeam_exit_to_phantom_mm >= 0.0 &&
              minibeam_slit_width_mm > 0.0 && minibeam_slit_pitch_mm > 0.0 &&
              minibeam_slit_half_length_mm > 0.0)) {
            throw std::invalid_argument(
                "minibeam dimensions must be positive (exit gap may be zero)");
        }
        if (minibeam_slit_count <= 0 || minibeam_slit_count % 2 == 0) {
            throw std::invalid_argument(
                "minibeam_slit_count must be a positive odd number");
        }
        if (minibeam_slit_width_mm >= minibeam_slit_pitch_mm) {
            throw std::invalid_argument(
                "minibeam_slit_width_mm must be smaller than the pitch");
        }
        if (!std::isfinite(minibeam_collimator_angle_deg) ||
            !std::isfinite(minibeam_slit_offset_mm) ||
            !std::isfinite(minibeam_water_entrance_world_y_mm)) {
            throw std::invalid_argument(
                "minibeam angle, slit offset, and water-entrance coordinate "
                "must be finite");
        }
        if (minibeam_transport_mode == "copper_em") {
            if (minibeam_copper_stopping_power_file.empty()) {
                throw std::invalid_argument(
                    "minibeam copper_em requires "
                    "minibeam_copper_stopping_power_file");
            }
            if (minibeam_air_stopping_power_file.empty()) {
                throw std::invalid_argument(
                    "minibeam copper_em requires "
                    "minibeam_air_stopping_power_file");
            }
            if (minibeam_copper_enable_nuclear_attenuation &&
                minibeam_copper_cross_section_file.empty()) {
                throw std::invalid_argument(
                    "minibeam Copper nuclear attenuation requires "
                    "minibeam_copper_cross_section_file");
            }
            if (!(minibeam_copper_density_g_per_cm3 > 0.0 &&
                  minibeam_copper_radiation_length_g_per_cm2 > 0.0 &&
                  minibeam_copper_max_step_mm > 0.0 &&
                  minibeam_copper_mcs_scale > 0.0 &&
                  minibeam_copper_straggling_scale > 0.0 &&
                  minibeam_copper_survivor_energy_loss_scale > 0.0 &&
                  minibeam_copper_survivor_energy_loss_scale <= 2.0 &&
                  minibeam_water_primary_stopping_power_scale > 0.0 &&
                  minibeam_water_primary_stopping_power_scale <= 2.0 &&
                  minibeam_water_low_energy_mcs_transition_MeVu >= 0.0 &&
                  minibeam_water_primary_low_energy_mcs_scale > 0.0 &&
                  minibeam_water_primary_low_energy_mcs_scale <= 2.0 &&
                  minibeam_water_fragment_low_energy_mcs_scale > 0.0 &&
                  minibeam_water_fragment_low_energy_mcs_scale <= 2.0 &&
                  minibeam_water_touched_primary_surface_boost >= 0.0 &&
                  minibeam_water_touched_primary_surface_boost <= 0.5 &&
                  minibeam_water_touched_primary_surface_sigma_mm > 0.0 &&
                  minibeam_water_touched_primary_deficit >= 0.0 &&
                  minibeam_water_touched_primary_deficit <= 0.5 &&
                  minibeam_water_touched_primary_deficit_center_mm >= 0.0 &&
                  minibeam_water_touched_primary_deficit_sigma_mm > 0.0) ||
                !std::isfinite(minibeam_copper_mcs_scale) ||
                !std::isfinite(minibeam_copper_straggling_scale) ||
                !std::isfinite(
                    minibeam_copper_survivor_energy_loss_scale) ||
                !std::isfinite(
                    minibeam_water_primary_stopping_power_scale) ||
                !std::isfinite(
                    minibeam_water_low_energy_mcs_transition_MeVu) ||
                !std::isfinite(
                    minibeam_water_primary_low_energy_mcs_scale) ||
                !std::isfinite(
                    minibeam_water_fragment_low_energy_mcs_scale) ||
                !std::isfinite(
                    minibeam_water_touched_primary_surface_boost) ||
                !std::isfinite(
                    minibeam_water_touched_primary_surface_sigma_mm) ||
                !std::isfinite(
                    minibeam_water_touched_primary_deficit) ||
                !std::isfinite(
                    minibeam_water_touched_primary_deficit_center_mm) ||
                !std::isfinite(
                    minibeam_water_touched_primary_deficit_sigma_mm)) {
                throw std::invalid_argument(
                    "minibeam Copper density, radiation length, max step, and "
                    "MCS/straggling scales must be finite/positive and survivor energy-loss "
                    "and water stopping/MCS scales must be finite and in (0, 2]; "
                    "the water low-energy MCS transition must be nonnegative");
            }
            const auto& calibration_energies =
                minibeam_copper_survivor_energy_loss_energies_MeVu;
            const auto& calibration_scales =
                minibeam_copper_survivor_energy_loss_scales;
            if (calibration_energies.size() != calibration_scales.size() ||
                (!calibration_energies.empty() &&
                 calibration_energies.size() < 2) ||
                calibration_energies.size() > 16) {
                throw std::invalid_argument(
                    "minibeam Copper survivor energy-loss calibration requires "
                    "matching energy/scale lists with 2 to 16 entries");
            }
            for (std::size_t index = 0;
                 index < calibration_energies.size(); ++index) {
                if (!(std::isfinite(calibration_energies[index]) &&
                      calibration_energies[index] > 0.0 &&
                      std::isfinite(calibration_scales[index]) &&
                      calibration_scales[index] > 0.0 &&
                      calibration_scales[index] <= 2.0) ||
                    (index > 0 &&
                     calibration_energies[index] <=
                         calibration_energies[index - 1])) {
                    throw std::invalid_argument(
                        "minibeam Copper survivor energy-loss calibration "
                        "energies must be finite, positive, and strictly "
                        "increasing; scales must be finite and in (0, 2]");
                }
            }
        }
    }
    if (enable_tps_source) {
        if (!topas_spots_file.empty() || !topas_spots_files.empty() ||
            !spot_weights_file.empty()) {
            throw std::invalid_argument(
                "tpsSource=true is mutually exclusive with topas_spots_file(s) and "
                "spot_weights_file");
        }
        if (enable_flat_source) {
            throw std::invalid_argument(
                "tpsSource=true cannot combine with enable_flat_source");
        }
        if (!enable_voxel_scoring) {
            throw std::invalid_argument(
                "tpsSource=true requires enable_voxel_scoring=true so the finite "
                "patient-volume AABB is defined");
        }
        if (!(tps_sad_mm > 0.0)) {
            throw std::invalid_argument("tps_sad_mm must be positive");
        }
        if (!std::isfinite(tps_gantry_angle_deg) ||
            !std::isfinite(tps_couch_angle_deg) ||
            !std::isfinite(tps_collimator_angle_deg) ||
            !std::isfinite(tps_isocenter_x_mm) ||
            !std::isfinite(tps_isocenter_y_mm) ||
            !std::isfinite(tps_isocenter_z_mm) || !std::isfinite(tps_sad_mm)) {
            throw std::invalid_argument(
                "TPS angles, isocenter, and SAD must be finite");
        }
        if (tps_patient_position != "HFS" && tps_patient_position != "HFP" &&
            tps_patient_position != "FFS" && tps_patient_position != "FFP") {
            throw std::invalid_argument(
                "tps_patient_position must be HFS, HFP, FFS, or FFP");
        }
        if (tps_angle_convention != "dicom_lps" &&
            tps_angle_convention != "iec61217" &&
            tps_angle_convention != "topas_patient_rot_z") {
            throw std::invalid_argument(
                "tps_angle_convention must be dicom_lps (preferred) or "
                "iec61217; topas_patient_rot_z is a deprecated alias");
        }
        const auto magnets_x = tps_virtual_scanning_magnet_x_mm;
        const auto magnets_y = tps_virtual_scanning_magnet_y_mm;
        if ((magnets_x > 0.0) != (magnets_y > 0.0)) {
            throw std::invalid_argument(
                "tps_virtual_scanning_magnet_x_mm and "
                "tps_virtual_scanning_magnet_y_mm must both be positive or both unused");
        }
        if (magnets_x < 0.0 || magnets_y < 0.0 ||
            tps_virtual_source_to_isocenter_mm < 0.0) {
            throw std::invalid_argument(
                "TPS virtual magnet / source distances must be nonnegative");
        }
        if (tps_spot_weight_mode != "mu" && tps_spot_weight_mode != "histories") {
            throw std::invalid_argument(
                "tps_spot_weight_mode must be mu or histories");
        }
        if (!(tps_histories_scale > 0.0) || !std::isfinite(tps_histories_scale)) {
            throw std::invalid_argument("tps_histories_scale must be positive");
        }
    } else if (!tps_spots_file.empty()) {
        throw std::invalid_argument(
            "tps_spots_file is set but tpsSource=false");
    }
    if (enable_tps_coordinate_system) {
        if (!enable_ct_grid || !enable_voxel_scoring) {
            throw std::invalid_argument(
                "enable_tps_coordinate_system requires CT and voxel scoring");
        }
        if (tps_angle_convention != "dicom_lps") {
            throw std::invalid_argument(
                "enable_tps_coordinate_system requires DICOM LPS beam geometry");
        }
    }
    if (spots_enable_upstream_air_energy_loss) {
        if (!enable_tps_coordinate_system &&
            spots_geometry_mode != "tps_90" &&
            spots_geometry_mode != "tps_gantry_y") {
            throw std::invalid_argument(
                "spots_enable_upstream_air_energy_loss requires fixed TPS coordinates "
                "or a legacy TPS geometry mode");
        }
        if (spots_upstream_air_stopping_power_file.empty()) {
            throw std::invalid_argument(
                "spots_enable_upstream_air_energy_loss requires spots_upstream_air_stopping_power_file");
        }
    }
    if (!primary_spot_batch.empty()) {
        std::uint64_t expected_begin = 0;
        for (const auto& entry : primary_spot_batch) {
            if (entry.history_begin != expected_begin ||
                entry.history_end <= entry.history_begin) {
                throw std::invalid_argument(
                    "primary_spot_batch history ranges must be contiguous and non-empty");
            }
            expected_begin = entry.history_end;
        }
        if (expected_begin != number_of_histories) {
            throw std::invalid_argument(
                "primary_spot_batch ranges must cover number_of_histories exactly");
        }
    }
}

TransportConfig load_config(const std::filesystem::path& path) {
    auto values = read_key_values(path);
    const auto ion_physics_file = merge_ion_physics_manifest(path, values);
    TransportConfig config;
    config.config_schema_version = parse_number(
        values, "config_schema_version", config.config_schema_version);
    if (const auto mode = values.find("run_mode"); mode != values.end()) {
        auto name = mode->second;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](const unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                       });
        if (name == "smoke") {
            config.run_mode = RunMode::smoke;
        } else if (name == "research") {
            config.run_mode = RunMode::research;
        } else if (name == "production") {
            config.run_mode = RunMode::production;
        } else {
            throw std::invalid_argument(
                "run_mode must be smoke, research, or production");
        }
    }
    config.quality_maximum_relative_energy_residual = parse_number(
        values, "quality_maximum_relative_energy_residual",
        config.quality_maximum_relative_energy_residual);
    config.quality_maximum_absolute_energy_residual_MeV = parse_number(
        values, "quality_maximum_absolute_energy_residual_MeV",
        config.quality_maximum_absolute_energy_residual_MeV);
    config.quality_reject_any_queue_overflow = parse_bool(
        values, "quality_reject_any_queue_overflow",
        config.quality_reject_any_queue_overflow);
    config.quality_reject_nan_or_inf = parse_bool(
        values, "quality_reject_nan_or_inf", config.quality_reject_nan_or_inf);
    config.ion_physics_file = ion_physics_file;
    if (!ion_physics_file.empty()) {
        config.primary_stopping_power_file.clear();
        config.let_delta_electron_fraction_file.clear();
        config.particle_stopping_power_file.clear();
    }
    for (const char* removed : {
             "physics_profile",
             "spots_lateral_yz_skew",
             "spots_lateral_yz_skew_auto_pivot",
             "spots_lateral_yz_skew_pivot_mm",
             "spots_lateral_yz_rotation_deg",
             "spots_lateral_yz_rotation_pivot_y_mm",
             "mass_number",
             "stopping_power_file",
             "nuclear_cross_section_file",
             "reaction_package_file",
             "tps_particle_type",
         }) {
        if (values.contains(removed)) {
            throw std::invalid_argument(
                std::string(removed) +
                " was removed; transport uses numeric YAML fields only");
        }
    }
    config.number_of_histories = parse_number(values, "number_of_histories", config.number_of_histories);
    config.initial_energy_MeVu = parse_number(values, "initial_energy_MeVu", config.initial_energy_MeVu);
    config.beam_energy_spread =
        parse_number(values, "beam_energy_spread", config.beam_energy_spread);
    config.primary_atomic_number = static_cast<int>(parse_number(
        values, "primary_atomic_number", config.primary_atomic_number));
    config.primary_mass_number = static_cast<int>(parse_number(
        values, "primary_mass_number", config.primary_mass_number));
    config.primary_rest_mass_MeV = parse_number(
        values, "primary_rest_mass_MeV", config.primary_rest_mass_MeV);
    config.phantom_length_mm = parse_number(values, "phantom_length_mm", config.phantom_length_mm);
    config.depth_bin_width_mm = parse_number(values, "depth_bin_width_mm", config.depth_bin_width_mm);
    config.maximum_step_mm = parse_number(values, "maximum_step_mm", config.maximum_step_mm);
    config.maximum_relative_energy_loss =
        parse_number(values, "maximum_relative_energy_loss", config.maximum_relative_energy_loss);
    config.energy_cutoff_MeV = parse_number(values, "energy_cutoff_MeV", config.energy_cutoff_MeV);
    config.secondary_local_deposit_cutoff_MeV = parse_number(
        values, "secondary_local_deposit_cutoff_MeV",
        config.secondary_local_deposit_cutoff_MeV);
    config.secondary_heavy_local_deposit_z_min = static_cast<int>(parse_number(
        values, "secondary_heavy_local_deposit_z_min",
        static_cast<double>(config.secondary_heavy_local_deposit_z_min)));
    config.secondary_condensed_step_mm = parse_number(
        values, "secondary_condensed_step_mm",
        config.secondary_condensed_step_mm);
    config.water_density_g_per_cm3 =
        parse_number(values, "water_density_g_per_cm3", config.water_density_g_per_cm3);
    config.enable_layered_phantom =
        parse_bool(values, "enable_layered_phantom", config.enable_layered_phantom);
    {
        const auto z_it = values.find("slab_z_ends_mm");
        const auto d_it = values.find("slab_densities_g_per_cm3");
        if (z_it != values.end() || d_it != values.end()) {
            if (z_it == values.end() || d_it == values.end()) {
                throw std::runtime_error(
                    "slab_z_ends_mm and slab_densities_g_per_cm3 must both be set");
            }
            const auto z_ends = parse_double_list(z_it->second, "slab_z_ends_mm");
            const auto densities =
                parse_double_list(d_it->second, "slab_densities_g_per_cm3");
            if (z_ends.size() != densities.size()) {
                throw std::runtime_error(
                    "slab_z_ends_mm and slab_densities_g_per_cm3 length mismatch");
            }
            config.slab_layers.clear();
            config.slab_layers.reserve(z_ends.size());
            for (std::size_t index = 0; index < z_ends.size(); ++index) {
                config.slab_layers.push_back(
                    SlabLayer{z_ends[index], densities[index]});
            }
            // Presence of slab lists implies layered mode unless explicitly false.
            if (values.find("enable_layered_phantom") == values.end()) {
                config.enable_layered_phantom = true;
            }
        }
        const auto sp_it = values.find("slab_stopping_power_files");
        const auto xs_it = values.find("slab_cross_section_files");
        if (sp_it != values.end() || xs_it != values.end()) {
            if (sp_it == values.end() || xs_it == values.end()) {
                throw std::runtime_error(
                    "slab_stopping_power_files and slab_cross_section_files must both be set");
            }
            config.slab_stopping_power_files = parse_path_list(sp_it->second);
            config.slab_cross_section_files = parse_path_list(xs_it->second);
            if (values.find("enable_layered_phantom") == values.end()) {
                config.enable_layered_phantom = true;
            }
        }
        const auto radiation_it =
            values.find("slab_radiation_lengths_g_per_cm2");
        if (radiation_it != values.end()) {
            config.slab_radiation_lengths_g_per_cm2 = parse_double_list(
                radiation_it->second,
                "slab_radiation_lengths_g_per_cm2");
            if (values.find("enable_layered_phantom") == values.end()) {
                config.enable_layered_phantom = true;
            }
        }
    }
    config.enable_hetero_insert =
        parse_bool(values, "enable_hetero_insert", config.enable_hetero_insert);
    config.hetero_insert.x_min_mm =
        parse_number(values, "insert_x_min_mm", config.hetero_insert.x_min_mm);
    config.hetero_insert.x_max_mm =
        parse_number(values, "insert_x_max_mm", config.hetero_insert.x_max_mm);
    config.hetero_insert.y_min_mm =
        parse_number(values, "insert_y_min_mm", config.hetero_insert.y_min_mm);
    config.hetero_insert.y_max_mm =
        parse_number(values, "insert_y_max_mm", config.hetero_insert.y_max_mm);
    config.hetero_insert.z_min_mm =
        parse_number(values, "insert_z_min_mm", config.hetero_insert.z_min_mm);
    config.hetero_insert.z_max_mm =
        parse_number(values, "insert_z_max_mm", config.hetero_insert.z_max_mm);
    config.hetero_insert.density_g_per_cm3 = parse_number(
        values, "insert_density_g_per_cm3", config.hetero_insert.density_g_per_cm3);
    config.insert_stopping_power_file = parse_path(
        values, "insert_stopping_power_file", config.insert_stopping_power_file);
    config.insert_cross_section_file = parse_path(
        values, "insert_cross_section_file", config.insert_cross_section_file);
    config.insert_radiation_length_g_per_cm2 = parse_number(
        values, "insert_radiation_length_g_per_cm2",
        config.insert_radiation_length_g_per_cm2);
    if (values.find("insert_x_min_mm") != values.end() &&
        values.find("enable_hetero_insert") == values.end()) {
        config.enable_hetero_insert = true;
    }
    config.enable_ct_grid = parse_bool(values, "enable_ct_grid", config.enable_ct_grid);
    config.ct_grid_file = parse_path(values, "ct_grid_file", config.ct_grid_file);
    config.ct_schneider_file =
        parse_path(values, "ct_schneider_file", config.ct_schneider_file);
    {
        const auto it = values.find("ct_dicom_origin_mode");
        if (it != values.end() && !it->second.empty()) {
            config.ct_dicom_origin_mode = it->second;
            std::transform(config.ct_dicom_origin_mode.begin(),
                           config.ct_dicom_origin_mode.end(),
                           config.ct_dicom_origin_mode.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
        }
    }
    config.ct_skip_homogeneous_face_clamp = parse_bool(
        values, "ct_skip_homogeneous_face_clamp", config.ct_skip_homogeneous_face_clamp);
    config.ct_air_stopping_power_file = parse_path(
        values, "ct_air_stopping_power_file", config.ct_air_stopping_power_file);
    config.ct_lung_stopping_power_file = parse_path(
        values, "ct_lung_stopping_power_file", config.ct_lung_stopping_power_file);
    config.ct_water_stopping_power_file = parse_path(
        values, "ct_water_stopping_power_file", config.ct_water_stopping_power_file);
    config.ct_bone_stopping_power_file = parse_path(
        values, "ct_bone_stopping_power_file", config.ct_bone_stopping_power_file);
    config.ct_air_cross_section_file = parse_path(
        values, "ct_air_cross_section_file", config.ct_air_cross_section_file);
    config.ct_lung_cross_section_file = parse_path(
        values, "ct_lung_cross_section_file", config.ct_lung_cross_section_file);
    config.ct_water_cross_section_file = parse_path(
        values, "ct_water_cross_section_file", config.ct_water_cross_section_file);
    config.ct_bone_cross_section_file = parse_path(
        values, "ct_bone_cross_section_file", config.ct_bone_cross_section_file);
    config.ct_schneider_cross_section_file = parse_path(
        values, "ct_schneider_cross_section_file",
        config.ct_schneider_cross_section_file);
    config.ct_cinel02_rate_file = parse_path(
        values, "ct_cinel02_rate_file", config.ct_cinel02_rate_file);
    config.ct_hu_stopping_power_lut_file = parse_path(
        values, "ct_hu_stopping_power_lut_file",
        config.ct_hu_stopping_power_lut_file);
    config.ct_use_density_mass_spr = parse_bool(
        values, "ct_use_density_mass_spr", config.ct_use_density_mass_spr);
    config.ct_stopping_power_scale = parse_number(
        values, "ct_stopping_power_scale", config.ct_stopping_power_scale);
    if (values.find("ct_grid_file") != values.end() &&
        values.find("enable_ct_grid") == values.end()) {
        config.enable_ct_grid = true;
    }
    config.scorer_area_mm2 = parse_number(values, "scorer_area_mm2", config.scorer_area_mm2);
    config.dose_output_scale =
        parse_number(values, "dose_output_scale", config.dose_output_scale);
    config.enable_voxel_scoring =
        parse_bool(values, "enable_voxel_scoring", config.enable_voxel_scoring);
    config.enable_charged_origin_voxel_scoring = parse_bool(
        values, "enable_charged_origin_voxel_scoring",
        config.enable_charged_origin_voxel_scoring);
    config.voxel_scorer_clamps_transport = parse_bool(
        values, "voxel_scorer_clamps_transport",
        config.voxel_scorer_clamps_transport);
    config.voxel_bins_x = parse_number(values, "voxel_bins_x", config.voxel_bins_x);
    config.voxel_bins_y = parse_number(values, "voxel_bins_y", config.voxel_bins_y);
    config.voxel_bins_z = parse_number(values, "voxel_bins_z", config.voxel_bins_z);
    config.voxel_size_x_mm =
        parse_number(values, "voxel_size_x_mm", config.voxel_size_x_mm);
    config.voxel_size_y_mm =
        parse_number(values, "voxel_size_y_mm", config.voxel_size_y_mm);
    config.voxel_size_z_mm =
        parse_number(values, "voxel_size_z_mm", config.voxel_size_z_mm);
    if (config.enable_ct_grid) {
        const auto grid = CtGrid::load(
            resolve_input_path_from_config(config.ct_grid_file, path),
            config.ct_schneider_file.empty()
                ? std::filesystem::path{}
                : resolve_input_path_from_config(config.ct_schneider_file, path),
            config.ct_dicom_origin_mode);
        const auto close = [](const double left, const double right) {
            return std::abs(left - right) <=
                   1.0e-6 * std::max({1.0, std::abs(left), std::abs(right)});
        };
        const auto set_or_check_count = [&](const char* key,
                                            std::size_t& configured,
                                            const std::size_t from_ct) {
            if (values.contains(key) && configured != from_ct) {
                throw std::invalid_argument(
                    std::string(key) + " does not match the native CT grid header");
            }
            configured = from_ct;
        };
        const auto set_or_check_spacing = [&](const char* key,
                                              double& configured,
                                              const double from_ct) {
            if (values.contains(key) && !close(configured, from_ct)) {
                throw std::invalid_argument(
                    std::string(key) + " does not match the native CT grid header");
            }
            configured = from_ct;
        };

        // A CCTG file owns the patient image geometry. Keeping a second copy in
        // YAML made the slice axis look like beam depth and allowed the two
        // descriptions to drift. Explicit values remain accepted as assertions.
        set_or_check_count("voxel_bins_x", config.voxel_bins_x, grid.nx);
        set_or_check_count("voxel_bins_y", config.voxel_bins_y, grid.ny);
        set_or_check_count("voxel_bins_z", config.voxel_bins_z, grid.nz);
        set_or_check_spacing(
            "voxel_size_x_mm", config.voxel_size_x_mm, grid.spacing_x_mm);
        set_or_check_spacing(
            "voxel_size_y_mm", config.voxel_size_y_mm, grid.spacing_y_mm);
        set_or_check_spacing(
            "voxel_size_z_mm", config.voxel_size_z_mm, grid.spacing_z_mm);
        config.voxel_origin_x_mm = grid.origin_x_mm;
        config.voxel_origin_y_mm = grid.origin_y_mm;
        config.voxel_origin_z_mm = grid.origin_z_mm;

        const auto ct_z_extent_mm = static_cast<double>(grid.extent_z_mm());
        if (values.contains("depth_bin_width_mm") &&
            !close(config.depth_bin_width_mm, grid.spacing_z_mm)) {
            throw std::invalid_argument(
                "depth_bin_width_mm does not match CT spacing_z; use "
                "voxel_size_z_mm for patient CT geometry");
        }
        if (values.contains("phantom_length_mm") &&
            !close(config.phantom_length_mm, ct_z_extent_mm)) {
            throw std::invalid_argument(
                "phantom_length_mm does not match the CT z extent; use "
                "voxel_bins_z and voxel_size_z_mm for patient CT geometry");
        }
        config.depth_bin_width_mm = config.voxel_size_z_mm;
        config.phantom_length_mm = ct_z_extent_mm;
    } else {
        // Homogeneous phantom configurations retain the established z aliases.
        if (config.voxel_bins_z != 0 &&
            !values.contains("phantom_length_mm")) {
            config.phantom_length_mm =
                static_cast<double>(config.voxel_bins_z) *
                config.scorer_spacing_z_mm();
        }
        if (config.voxel_size_z_mm > 0.0 &&
            !values.contains("depth_bin_width_mm")) {
            config.depth_bin_width_mm = config.voxel_size_z_mm;
        }
    }
    config.enable_energy_straggling =
        parse_bool(values, "enable_energy_straggling", config.enable_energy_straggling);
    config.enable_csda_range_energy_loss = parse_bool(
        values, "enable_csda_range_energy_loss", config.enable_csda_range_energy_loss);
    config.enable_step_stable_straggling = parse_bool(
        values, "enable_step_stable_straggling", config.enable_step_stable_straggling);
    config.straggling_sampling_length_mm = parse_number(
        values, "straggling_sampling_length_mm", config.straggling_sampling_length_mm);
    config.enable_secondary_energy_straggling = parse_bool(
        values, "enable_secondary_energy_straggling",
        config.enable_secondary_energy_straggling);
    {
        const auto it = values.find("energy_straggling_model");
        if (it != values.end() && !it->second.empty()) {
            config.energy_straggling_model = it->second;
            std::transform(config.energy_straggling_model.begin(),
                           config.energy_straggling_model.end(),
                           config.energy_straggling_model.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
        }
    }
    config.energy_straggling_package_file = parse_path(
        values, "energy_straggling_package_file", config.energy_straggling_package_file);
    config.straggling_scale = parse_number(values, "straggling_scale", config.straggling_scale);
    if (const auto iterator = values.find("straggling_scale_energies_MeVu");
        iterator != values.end()) {
        config.straggling_scale_energies_MeVu = parse_double_list(
            iterator->second, "straggling_scale_energies_MeVu");
    }
    if (const auto iterator = values.find("straggling_scale_values");
        iterator != values.end()) {
        config.straggling_scale_values = parse_double_list(
            iterator->second, "straggling_scale_values");
    }
    config.enable_inelastic =
        parse_bool(values, "enable_inelastic", config.enable_inelastic);
    config.primary_inelastic_package_v2_file = parse_path(
        values, "primary_inelastic_package_v2_file",
        config.primary_inelastic_package_v2_file);
    config.primary_inelastic_rate_v2_file = parse_path(
        values, "primary_inelastic_rate_v2_file",
        config.primary_inelastic_rate_v2_file);
    config.cinel02_max_secondary_inelastic_generations = parse_number(
        values, "cinel02_max_secondary_inelastic_generations",
        config.cinel02_max_secondary_inelastic_generations);
    config.cinel02_topas_compatibility_mode = parse_bool(
        values, "cinel02_topas_compatibility_mode",
        config.cinel02_topas_compatibility_mode);
    config.cinel02_strict_match = parse_bool(
        values, "cinel02_strict_match", config.cinel02_strict_match);
    if (const auto it = values.find("nuclear_model"); it != values.end()) {
        config.nuclear_model = it->second;
        std::transform(config.nuclear_model.begin(), config.nuclear_model.end(),
                       config.nuclear_model.begin(), [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
    config.enable_nuclear_elastic =
        parse_bool(values, "enable_nuclear_elastic", config.enable_nuclear_elastic);
    config.fred_event_library_h_file = parse_path(
        values, "fred_event_library_h_file", config.fred_event_library_h_file);
    config.fred_event_library_o_file = parse_path(
        values, "fred_event_library_o_file", config.fred_event_library_o_file);
    if (const auto it = values.find("fred_event_library_h_files"); it != values.end())
        config.fred_event_library_h_files = parse_path_list(it->second);
    if (const auto it = values.find("fred_event_library_c_files"); it != values.end())
        config.fred_event_library_c_files = parse_path_list(it->second);
    if (const auto it = values.find("fred_event_library_o_files"); it != values.end())
        config.fred_event_library_o_files = parse_path_list(it->second);
    if (!config.fred_event_library_h_file.empty() &&
        !config.fred_event_library_h_files.empty())
        throw std::invalid_argument("use singular or plural H event-library key, not both");
    if (!config.fred_event_library_o_file.empty() &&
        !config.fred_event_library_o_files.empty())
        throw std::invalid_argument("use singular or plural O event-library key, not both");
    if (config.nuclear_model == "fred_paper") {
        if (config.fred_event_library_h_file.empty()) {
            config.fred_event_library_h_file =
                "data/packages/c12_H1_95MeVu_events.bin";
        }
        if (config.fred_event_library_o_file.empty()) {
            config.fred_event_library_o_file =
                "data/packages/c12_O16_95MeVu_events.bin";
        }
    }
    config.primary_inelastic_cross_section_file =
        parse_path(values, "primary_inelastic_cross_section_file",
                   config.primary_inelastic_cross_section_file);
    config.enable_secondary_transport =
        parse_bool(values, "enable_secondary_transport",
                   config.enable_secondary_transport);
    config.enable_multiple_scattering =
        parse_bool(values, "enable_multiple_scattering", config.enable_multiple_scattering);
    if (const auto it = values.find("multiple_scattering_model"); it != values.end()) {
        config.multiple_scattering_model = it->second;
        std::transform(config.multiple_scattering_model.begin(),
                       config.multiple_scattering_model.end(),
                       config.multiple_scattering_model.begin(), [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
    config.fred_2gr_mcs_file = parse_path(
        values, "fred_2gr_mcs_file", config.fred_2gr_mcs_file);
    if (const auto it = values.find("fred_2gr_high_energy_mode"); it != values.end()) {
        config.fred_2gr_high_energy_mode = it->second;
        std::transform(config.fred_2gr_high_energy_mode.begin(),
                       config.fred_2gr_high_energy_mode.end(),
                       config.fred_2gr_high_energy_mode.begin(), [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
    config.multiple_scattering_scale = parse_number(
        values, "multiple_scattering_scale", config.multiple_scattering_scale);
    config.enable_ct_material_mcs =
        parse_bool(values, "enable_ct_material_mcs", config.enable_ct_material_mcs);
    config.enable_flat_source =
        parse_bool(values, "enable_flat_source", config.enable_flat_source);
    config.flat_source_half_width_x_mm = parse_number(
        values, "flat_source_half_width_x_mm", config.flat_source_half_width_x_mm);
    config.flat_source_half_width_y_mm = parse_number(
        values, "flat_source_half_width_y_mm", config.flat_source_half_width_y_mm);
    config.enable_emittance_source =
        parse_bool(values, "enable_emittance_source", config.enable_emittance_source);
    config.emittance_sigma_x_mm =
        parse_number(values, "emittance_sigma_x_mm", config.emittance_sigma_x_mm);
    config.emittance_sigma_y_mm =
        parse_number(values, "emittance_sigma_y_mm", config.emittance_sigma_y_mm);
    config.emittance_sigma_x_prime =
        parse_number(values, "emittance_sigma_x_prime", config.emittance_sigma_x_prime);
    config.emittance_sigma_y_prime =
        parse_number(values, "emittance_sigma_y_prime", config.emittance_sigma_y_prime);
    config.emittance_correlation_x =
        parse_number(values, "emittance_correlation_x", config.emittance_correlation_x);
    config.emittance_correlation_y =
        parse_number(values, "emittance_correlation_y", config.emittance_correlation_y);
    config.enable_fragment_species_scoring = parse_bool(
        values, "enable_fragment_species_scoring",
        config.enable_fragment_species_scoring);
    {
        const auto it = values.find("scorer_mode");
        if (it != values.end() && !it->second.empty()) {
            config.scorer_mode = it->second;
            std::transform(config.scorer_mode.begin(), config.scorer_mode.end(),
                           config.scorer_mode.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
        }
    }
    config.validation_output_directory = parse_path(
        values, "validation_output_directory",
        config.validation_output_directory);
    if (config.scorer_mode == "validation") {
        config.enable_fragment_species_scoring = true;
        if (!config.validation_output_directory.empty()) {
            const auto dir = config.validation_output_directory;
            if (config.output_file.empty() ||
                config.output_file == "out/cpu_depth_dose.csv") {
                config.output_file = dir / "dose_MeV.csv";
            }
            if (config.dose_output_file.empty() ||
                config.dose_output_file == "out/cpu_depth_dose_Gy.csv") {
                config.dose_output_file = dir / "dose.csv";
            }
            if (config.fragment_species_output_file.empty()) {
                config.fragment_species_output_file = dir / "species_dose_MeV.csv";
            }
            if (config.fragment_species_dose_output_file.empty()) {
                config.fragment_species_dose_output_file = dir / "species_dose.csv";
            }
            if (config.enable_let_scoring && config.let_output_file.empty()) {
                config.let_output_file = dir / "letd.csv";
            }
        }
    }
    {
        const auto camel = values.find("scorerLET");
        const auto snake = values.find("enable_let_scoring");
        if (camel != values.end() && snake != values.end()) {
            const auto camel_value = parse_bool(values, "scorerLET", false);
            const auto snake_value = parse_bool(values, "enable_let_scoring", false);
            if (camel_value != snake_value) {
                throw std::runtime_error(
                    "scorerLET and enable_let_scoring are both set with different values");
            }
            config.enable_let_scoring = camel_value;
        } else if (camel != values.end()) {
            config.enable_let_scoring =
                parse_bool(values, "scorerLET", config.enable_let_scoring);
        } else if (snake != values.end()) {
            config.enable_let_scoring =
                parse_bool(values, "enable_let_scoring", config.enable_let_scoring);
        }
    }
    config.enable_electron_transport = parse_bool(
        values, "enable_electron_transport", config.enable_electron_transport);
    config.electron_queue_capacity = parse_number(
        values, "electron_queue_capacity", config.electron_queue_capacity);
    config.electron_kinetic_cutoff_MeV = parse_number(
        values, "electron_kinetic_cutoff_MeV", config.electron_kinetic_cutoff_MeV);
    config.maximum_electromagnetic_generations = parse_number(
        values, "maximum_electromagnetic_generations",
        config.maximum_electromagnetic_generations);
    config.maximum_electron_step_mm = parse_number(
        values, "maximum_electron_step_mm", config.maximum_electron_step_mm);
    config.maximum_electron_relative_energy_loss = parse_number(
        values, "maximum_electron_relative_energy_loss",
        config.maximum_electron_relative_energy_loss);
    config.electronic_buildup_fraction = parse_number(
        values, "electronic_buildup_fraction", config.electronic_buildup_fraction);
    config.electronic_buildup_mfp_mm = parse_number(
        values, "electronic_buildup_mfp_mm", config.electronic_buildup_mfp_mm);
    config.minibeam_electronic_buildup_primary_only = parse_bool(
        values, "minibeam_electronic_buildup_primary_only",
        config.minibeam_electronic_buildup_primary_only);
    config.electronic_buildup_lateral_sigma_mm = parse_number(
        values, "electronic_buildup_lateral_sigma_mm",
        config.electronic_buildup_lateral_sigma_mm);
    config.max_device_memory_fraction = parse_number(
        values, "max_device_memory_fraction", config.max_device_memory_fraction);
    config.history_chunk_size =
        parse_number(values, "history_chunk_size", config.history_chunk_size);
    config.robust_boundary_nudge = parse_bool(
        values, "robust_boundary_nudge", config.robust_boundary_nudge);
    config.source_origin_x_mm =
        parse_number(values, "source_origin_x_mm", config.source_origin_x_mm);
    config.source_origin_y_mm =
        parse_number(values, "source_origin_y_mm", config.source_origin_y_mm);
    config.source_origin_z_mm =
        parse_number(values, "source_origin_z_mm", config.source_origin_z_mm);
    config.beam_ux_x = parse_number(values, "beam_ux_x", config.beam_ux_x);
    config.beam_ux_y = parse_number(values, "beam_ux_y", config.beam_ux_y);
    config.beam_ux_z = parse_number(values, "beam_ux_z", config.beam_ux_z);
    config.beam_uy_x = parse_number(values, "beam_uy_x", config.beam_uy_x);
    config.beam_uy_y = parse_number(values, "beam_uy_y", config.beam_uy_y);
    config.beam_uy_z = parse_number(values, "beam_uy_z", config.beam_uy_z);
    config.beam_uz_x = parse_number(values, "beam_uz_x", config.beam_uz_x);
    config.beam_uz_y = parse_number(values, "beam_uz_y", config.beam_uz_y);
    config.beam_uz_z = parse_number(values, "beam_uz_z", config.beam_uz_z);
    config.enable_minibeam =
        parse_bool(values, "minibeam", config.enable_minibeam);
    config.enable_minibeam_diagnostics = parse_bool(
        values, "minibeam_diagnostics",
        config.enable_minibeam_diagnostics);
    {
        const auto it = values.find("minibeam_transport_mode");
        if (it != values.end() && !it->second.empty()) {
            config.minibeam_transport_mode = it->second;
        }
    }
    {
        const auto it = values.find("minibeam_material");
        if (it != values.end() && !it->second.empty()) {
            config.minibeam_material = it->second;
        }
    }
    config.minibeam_radius_mm = parse_number(
        values, "minibeam_radius_mm", config.minibeam_radius_mm);
    config.minibeam_thickness_mm = parse_number(
        values, "minibeam_thickness_mm", config.minibeam_thickness_mm);
    config.minibeam_exit_to_phantom_mm = parse_number(
        values, "minibeam_exit_to_phantom_mm",
        config.minibeam_exit_to_phantom_mm);
    config.minibeam_slit_count = parse_number(
        values, "minibeam_slit_count", config.minibeam_slit_count);
    config.minibeam_slit_width_mm = parse_number(
        values, "minibeam_slit_width_mm", config.minibeam_slit_width_mm);
    config.minibeam_slit_pitch_mm = parse_number(
        values, "minibeam_slit_pitch_mm", config.minibeam_slit_pitch_mm);
    config.minibeam_slit_half_length_mm = parse_number(
        values, "minibeam_slit_half_length_mm",
        config.minibeam_slit_half_length_mm);
    config.minibeam_slit_offset_mm = parse_number(
        values, "minibeam_slit_offset_mm",
        config.minibeam_slit_offset_mm);
    config.minibeam_collimator_angle_deg = parse_number(
        values, "minibeam_collimator_angle_deg",
        config.minibeam_collimator_angle_deg);
    config.minibeam_copper_stopping_power_file = parse_path(
        values, "minibeam_copper_stopping_power_file",
        config.minibeam_copper_stopping_power_file);
    config.minibeam_air_stopping_power_file = parse_path(
        values, "minibeam_air_stopping_power_file",
        config.minibeam_air_stopping_power_file);
    config.minibeam_copper_density_g_per_cm3 = parse_number(
        values, "minibeam_copper_density_g_per_cm3",
        config.minibeam_copper_density_g_per_cm3);
    config.minibeam_copper_radiation_length_g_per_cm2 = parse_number(
        values, "minibeam_copper_radiation_length_g_per_cm2",
        config.minibeam_copper_radiation_length_g_per_cm2);
    config.minibeam_copper_max_step_mm = parse_number(
        values, "minibeam_copper_max_step_mm",
        config.minibeam_copper_max_step_mm);
    config.minibeam_copper_enable_mcs = parse_bool(
        values, "minibeam_copper_enable_mcs",
        config.minibeam_copper_enable_mcs);
    config.minibeam_copper_enable_energy_straggling = parse_bool(
        values, "minibeam_copper_enable_energy_straggling",
        config.minibeam_copper_enable_energy_straggling);
    config.minibeam_copper_straggling_scale = parse_number(
        values, "minibeam_copper_straggling_scale",
        config.minibeam_copper_straggling_scale);
    config.minibeam_copper_mcs_scale = parse_number(
        values, "minibeam_copper_mcs_scale",
        config.minibeam_copper_mcs_scale);
    config.minibeam_copper_survivor_energy_loss_scale = parse_number(
        values, "minibeam_copper_survivor_energy_loss_scale",
        config.minibeam_copper_survivor_energy_loss_scale);
    if (const auto iterator = values.find(
            "minibeam_copper_survivor_energy_loss_energies_MeVu");
        iterator != values.end()) {
        config.minibeam_copper_survivor_energy_loss_energies_MeVu =
            parse_double_list(
                iterator->second,
                "minibeam_copper_survivor_energy_loss_energies_MeVu");
    }
    if (const auto iterator = values.find(
            "minibeam_copper_survivor_energy_loss_scales");
        iterator != values.end()) {
        config.minibeam_copper_survivor_energy_loss_scales =
            parse_double_list(
                iterator->second,
                "minibeam_copper_survivor_energy_loss_scales");
    }
    config.minibeam_water_primary_stopping_power_scale = parse_number(
        values, "minibeam_water_primary_stopping_power_scale",
        config.minibeam_water_primary_stopping_power_scale);
    config.minibeam_water_low_energy_mcs_transition_MeVu = parse_number(
        values, "minibeam_water_low_energy_mcs_transition_MeVu",
        config.minibeam_water_low_energy_mcs_transition_MeVu);
    config.minibeam_water_primary_low_energy_mcs_scale = parse_number(
        values, "minibeam_water_primary_low_energy_mcs_scale",
        config.minibeam_water_primary_low_energy_mcs_scale);
    config.minibeam_water_fragment_low_energy_mcs_scale = parse_number(
        values, "minibeam_water_fragment_low_energy_mcs_scale",
        config.minibeam_water_fragment_low_energy_mcs_scale);
    config.minibeam_water_touched_primary_surface_boost = parse_number(
        values, "minibeam_water_touched_primary_surface_boost",
        config.minibeam_water_touched_primary_surface_boost);
    config.minibeam_water_touched_primary_surface_sigma_mm =
        parse_number(
            values,
            "minibeam_water_touched_primary_surface_sigma_mm",
            config.minibeam_water_touched_primary_surface_sigma_mm);
    config.minibeam_water_touched_primary_deficit = parse_number(
        values, "minibeam_water_touched_primary_deficit",
        config.minibeam_water_touched_primary_deficit);
    config.minibeam_water_touched_primary_deficit_center_mm =
        parse_number(
            values,
            "minibeam_water_touched_primary_deficit_center_mm",
            config.minibeam_water_touched_primary_deficit_center_mm);
    config.minibeam_water_touched_primary_deficit_sigma_mm =
        parse_number(
            values,
            "minibeam_water_touched_primary_deficit_sigma_mm",
            config.minibeam_water_touched_primary_deficit_sigma_mm);
    config.minibeam_copper_enable_nuclear_attenuation = parse_bool(
        values, "minibeam_copper_enable_nuclear_attenuation",
        config.minibeam_copper_enable_nuclear_attenuation);
    config.minibeam_copper_cross_section_file = parse_path(
        values, "minibeam_copper_cross_section_file",
        config.minibeam_copper_cross_section_file);
    config.minibeam_copper_ion_stopping_power_file = parse_path(
        values, "minibeam_copper_ion_stopping_power_file",
        config.minibeam_copper_ion_stopping_power_file);
    config.minibeam_copper_ion_cross_section_file = parse_path(
        values, "minibeam_copper_ion_cross_section_file",
        config.minibeam_copper_ion_cross_section_file);
    config.minibeam_copper_neutral_cross_section_file = parse_path(
        values, "minibeam_copper_neutral_cross_section_file",
        config.minibeam_copper_neutral_cross_section_file);
    config.minibeam_water_entrance_world_y_mm = parse_number(
        values, "minibeam_water_entrance_world_y_mm",
        config.minibeam_water_entrance_world_y_mm);
    {
        const auto it = values.find("topas_spots_file");
        if (it != values.end() && !it->second.empty()) {
            config.topas_spots_file = it->second;
        }
    }
    {
        const auto it = values.find("topas_spots_files");
        if (it != values.end()) {
            config.topas_spots_files = parse_path_list(it->second);
        }
    }
    config.spot_weights_file =
        parse_path(values, "spot_weights_file", config.spot_weights_file);
    config.spots_sad_mm = parse_number(values, "spots_sad_mm", config.spots_sad_mm);
    {
        const auto it = values.find("spots_geometry_mode");
        if (it != values.end() && !it->second.empty()) {
            config.spots_geometry_mode = it->second;
        }
    }
    config.spots_patient_trans_x_mm = parse_number(
        values, "spots_patient_trans_x_mm", config.spots_patient_trans_x_mm);
    config.spots_patient_trans_y_mm = parse_number(
        values, "spots_patient_trans_y_mm", config.spots_patient_trans_y_mm);
    config.spots_patient_trans_z_mm = parse_number(
        values, "spots_patient_trans_z_mm", config.spots_patient_trans_z_mm);
    config.spots_patient_rot_z_deg = parse_number(
        values, "spots_patient_rot_z_deg", config.spots_patient_rot_z_deg);
    config.spots_ct_axis_min_mm = parse_number(
        values, "spots_ct_axis_min_mm", config.spots_ct_axis_min_mm);
    config.spots_emittance_sigma_scale = parse_number(
        values, "spots_emittance_sigma_scale",
        config.spots_emittance_sigma_scale);
    config.spots_emittance_prime_scale = parse_number(
        values, "spots_emittance_prime_scale",
        config.spots_emittance_prime_scale);
    config.spots_enable_upstream_air_energy_loss = parse_bool(
        values, "spots_enable_upstream_air_energy_loss",
        config.spots_enable_upstream_air_energy_loss);
    config.spots_upstream_air_stopping_power_file = parse_path(
        values, "spots_upstream_air_stopping_power_file",
        config.spots_upstream_air_stopping_power_file);
    {
        const auto public_switch = values.find("enable_tps_coordinate_system");
        const auto camel = values.find("tpsSource");
        const auto snake = values.find("tps_source");
        if (public_switch != values.end()) {
            config.enable_tps_coordinate_system = parse_bool(
                values, "enable_tps_coordinate_system", false);
        }
        if (camel != values.end() && snake != values.end()) {
            const auto camel_value = parse_bool(values, "tpsSource", false);
            const auto snake_value = parse_bool(values, "tps_source", false);
            if (camel_value != snake_value) {
                throw std::runtime_error(
                    "tpsSource and tps_source are both set with different values");
            }
            config.enable_tps_source = camel_value;
        } else if (camel != values.end()) {
            config.enable_tps_source = parse_bool(
                values, "tpsSource", config.enable_tps_source);
        } else if (snake != values.end()) {
            config.enable_tps_source = parse_bool(
                values, "tps_source", config.enable_tps_source);
        }
    }
    config.tps_spots_file = parse_path(values, "tps_spots_file", config.tps_spots_file);
    config.tps_beam_model_file =
        parse_path(values, "tps_beam_model_file", config.tps_beam_model_file);
    config.tps_virtual_scanning_magnet_x_mm = parse_number(
        values, "tps_virtual_scanning_magnet_x_mm",
        config.tps_virtual_scanning_magnet_x_mm);
    config.tps_virtual_scanning_magnet_y_mm = parse_number(
        values, "tps_virtual_scanning_magnet_y_mm",
        config.tps_virtual_scanning_magnet_y_mm);
    config.tps_virtual_source_to_isocenter_mm = parse_number(
        values, "tps_virtual_source_to_isocenter_mm",
        config.tps_virtual_source_to_isocenter_mm);
    config.tps_apply_topas_patient_placement = parse_bool(
        values, "tps_apply_topas_patient_placement",
        config.tps_apply_topas_patient_placement);
    if (const auto it = values.find("tps_spot_weight_mode");
        it != values.end() && !it->second.empty()) {
        config.tps_spot_weight_mode = it->second;
        std::transform(config.tps_spot_weight_mode.begin(),
                       config.tps_spot_weight_mode.end(),
                       config.tps_spot_weight_mode.begin(),
                       [](const unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                       });
    }
    config.tps_histories_scale = parse_number(
        values, "tps_histories_scale", config.tps_histories_scale);
    {
        const auto it = values.find("tps_angle_convention");
        if (it != values.end() && !it->second.empty()) {
            config.tps_angle_convention = it->second;
            std::transform(config.tps_angle_convention.begin(),
                           config.tps_angle_convention.end(),
                           config.tps_angle_convention.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
        }
        if (it == values.end() && config.enable_tps_coordinate_system) {
            config.tps_angle_convention = "dicom_lps";
        }
    }
    {
        const auto public_angle = values.find("tps_beam_angle_deg");
        const auto legacy_angle = values.find("tps_gantry_angle_deg");
        if (public_angle != values.end() && legacy_angle != values.end()) {
            const auto angle =
                parse_number(values, "tps_beam_angle_deg", 0.0);
            const auto legacy =
                parse_number(values, "tps_gantry_angle_deg", 0.0);
            if (std::abs(angle - legacy) > 1.0e-12) {
                throw std::runtime_error(
                    "tps_beam_angle_deg conflicts with tps_gantry_angle_deg");
            }
            config.tps_gantry_angle_deg = angle;
        } else if (public_angle != values.end()) {
            config.tps_gantry_angle_deg =
                parse_number(values, "tps_beam_angle_deg", 0.0);
        } else {
            config.tps_gantry_angle_deg = parse_number(
                values, "tps_gantry_angle_deg", config.tps_gantry_angle_deg);
        }
    }
    config.tps_couch_angle_deg = parse_number(
        values, "tps_couch_angle_deg", config.tps_couch_angle_deg);
    config.tps_collimator_angle_deg = parse_number(
        values, "tps_collimator_angle_deg", config.tps_collimator_angle_deg);
    {
        const auto vector = values.find("tps_isocenter_mm");
        const auto has_components = values.contains("tps_isocenter_x_mm") ||
                                    values.contains("tps_isocenter_y_mm") ||
                                    values.contains("tps_isocenter_z_mm");
        if (vector != values.end() && has_components) {
            throw std::runtime_error(
                "Use either tps_isocenter_mm or its x/y/z component keys, not both");
        }
        if (vector != values.end()) {
            auto text = trim(vector->second);
            if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
                text = text.substr(1, text.size() - 2);
            }
            const auto coordinates = parse_double_list(text, "tps_isocenter_mm");
            if (coordinates.size() != 3) {
                throw std::runtime_error(
                    "tps_isocenter_mm must contain exactly [x, y, z]");
            }
            config.tps_isocenter_x_mm = coordinates[0];
            config.tps_isocenter_y_mm = coordinates[1];
            config.tps_isocenter_z_mm = coordinates[2];
        } else {
            config.tps_isocenter_x_mm = parse_number(
                values, "tps_isocenter_x_mm", config.tps_isocenter_x_mm);
            config.tps_isocenter_y_mm = parse_number(
                values, "tps_isocenter_y_mm", config.tps_isocenter_y_mm);
            config.tps_isocenter_z_mm = parse_number(
                values, "tps_isocenter_z_mm", config.tps_isocenter_z_mm);
        }
    }
    config.tps_sad_mm = parse_number(values, "tps_sad_mm", config.tps_sad_mm);
    {
        const auto it = values.find("tps_patient_position");
        if (it != values.end() && !it->second.empty()) {
            config.tps_patient_position = it->second;
            std::transform(config.tps_patient_position.begin(),
                           config.tps_patient_position.end(),
                           config.tps_patient_position.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::toupper(character));
                           });
        }
    }
    if (const auto seed = values.find("random_seed"); seed != values.end()) {
        config.random_seed = parse_random_seed(seed->second);
    }
    config.primary_stopping_power_file = parse_path(values, "primary_stopping_power_file", config.primary_stopping_power_file);
    config.let_delta_electron_fraction_file = parse_path(
        values, "let_delta_electron_fraction_file",
        config.let_delta_electron_fraction_file);
    config.use_particle_specific_stopping_power = parse_bool(
        values, "use_particle_specific_stopping_power",
        config.use_particle_specific_stopping_power);
    config.particle_stopping_power_file = parse_path(
        values, "particle_stopping_power_file",
        config.particle_stopping_power_file);
    config.ct_lung_particle_stopping_power_file = parse_path(
        values, "ct_lung_particle_stopping_power_file",
        config.ct_lung_particle_stopping_power_file);
    config.ct_soft_tissue_particle_stopping_power_file = parse_path(
        values, "ct_soft_tissue_particle_stopping_power_file",
        config.ct_soft_tissue_particle_stopping_power_file);
    config.ct_bone_particle_stopping_power_file = parse_path(
        values, "ct_bone_particle_stopping_power_file",
        config.ct_bone_particle_stopping_power_file);
    config.electron_transport_data_file = parse_path(
        values, "electron_transport_data_file", config.electron_transport_data_file);
    {
        const auto it = values.find("output_file");
        if (it != values.end()) {
            config.output_file =
                it->second.empty() ? std::filesystem::path{} : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("fragment_species_output_file");
        if (it != values.end()) {
            config.fragment_species_output_file =
                it->second.empty() ? std::filesystem::path{} : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("let_output_file");
        if (it != values.end()) {
            config.let_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("let_voxel_mhd_output_file");
        if (it != values.end()) {
            config.let_voxel_mhd_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("fragment_species_let_output_file");
        if (it != values.end()) {
            config.fragment_species_let_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("light_isotope_let_output_file");
        if (it != values.end()) {
            config.light_isotope_let_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("fragment_birth_spectrum_output_file");
        if (it != values.end()) {
            config.fragment_birth_spectrum_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("voxel_dose_output_file");
        if (it != values.end()) {
            config.voxel_dose_output_file =
                it->second.empty() ? std::filesystem::path{} : std::filesystem::path{it->second};
        }
    }
    config.charged_origin_voxel_output_file = parse_path(
        values, "charged_origin_voxel_output_file",
        config.charged_origin_voxel_output_file);
    config.neutral_origin_voxel_output_file = parse_path(
        values, "neutral_origin_voxel_output_file",
        config.neutral_origin_voxel_output_file);
    // Dose (Gy) scorers. Empty string disables that Gy file (MeV scorers unchanged).
    {
        const auto it = values.find("dose_output_file");
        if (it != values.end()) {
            config.dose_output_file = it->second.empty()
                                         ? std::filesystem::path{}
                                         : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("fragment_species_dose_output_file");
        if (it != values.end()) {
            config.fragment_species_dose_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("voxel_dose_Gy_output_file");
        if (it != values.end()) {
            config.voxel_dose_Gy_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("charged_origin_voxel_dose_Gy_output_file");
        if (it != values.end()) {
            config.charged_origin_voxel_dose_Gy_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("charged_origin_voxel_mhd_output_prefix");
        if (it != values.end()) {
            config.charged_origin_voxel_mhd_output_prefix =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto it = values.find("voxel_dose_mhd_output_file");
        if (it != values.end()) {
            config.voxel_dose_mhd_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    {
        const auto enabled_it = values.find("dose_to_medium");
        const auto type_it = values.find("dose_to_medium_type");
        const auto name_it = values.find("dose_to_medium_name");
        const auto has_public_dose = enabled_it != values.end() ||
                                     type_it != values.end() ||
                                     name_it != values.end();
        if (has_public_dose) {
            if (values.contains("voxel_dose_mhd_output_file") ||
                values.contains("voxel_dose_Gy_output_file") ||
                values.contains("dose_output_file")) {
                throw std::runtime_error(
                    "dose_to_medium cannot be combined with legacy dose output paths");
            }
            if (enabled_it == values.end()) {
                throw std::runtime_error(
                    "dose_to_medium_type/name require dose_to_medium");
            }
            const auto enabled = parse_bool(values, "dose_to_medium", false);
            // The high-level scorer owns the production output set. Suppress
            // legacy default CSV files so users do not need empty YAML keys.
            config.output_file.clear();
            config.fragment_species_output_file.clear();
            config.fragment_species_dose_output_file.clear();
            config.voxel_dose_output_file.clear();
            config.dose_output_file.clear();
            config.voxel_dose_Gy_output_file.clear();
            config.voxel_dose_mhd_output_file.clear();
            if (enabled) {
                if (!config.enable_voxel_scoring) {
                    throw std::invalid_argument(
                        "dose_to_medium requires enable_voxel_scoring=true");
                }
                const auto type = type_it != values.end() ? type_it->second : "mhd";
                if (type != "mhd") {
                    throw std::invalid_argument(
                        "dose_to_medium_type currently supports only mhd");
                }
                if (name_it == values.end()) {
                    throw std::runtime_error(
                        "dose_to_medium=true requires dose_to_medium_name");
                }
                config.voxel_dose_mhd_output_file = scorer_output_path(
                    path, name_it->second, true);
            }
        }
    }
    {
        const auto enabled_it = values.find("LET");
        const auto type_it = values.find("LET_type");
        const auto name_it = values.find("LET_name");
        const auto has_public_let = enabled_it != values.end() ||
                                    type_it != values.end() ||
                                    name_it != values.end();
        if (has_public_let) {
            if (values.contains("scorerLET") ||
                values.contains("enable_let_scoring") ||
                values.contains("let_voxel_mhd_output_file") ||
                values.contains("let_output_file")) {
                throw std::runtime_error(
                    "LET cannot be combined with legacy LET scorer/output keys");
            }
            if (enabled_it == values.end()) {
                throw std::runtime_error("LET_type/name require LET");
            }
            const auto enabled = parse_bool(values, "LET", false);
            config.enable_let_scoring = enabled;
            config.let_output_file.clear();
            config.let_voxel_mhd_output_file.clear();
            config.fragment_species_let_output_file.clear();
            config.light_isotope_let_output_file.clear();
            if (enabled) {
                if (!config.enable_voxel_scoring) {
                    throw std::invalid_argument(
                        "LET requires enable_voxel_scoring=true");
                }
                const auto type = type_it != values.end() ? type_it->second : "mhd";
                if (type != "mhd") {
                    throw std::invalid_argument("LET_type currently supports only mhd");
                }
                if (name_it == values.end()) {
                    throw std::runtime_error("LET=true requires LET_name");
                }
                config.let_voxel_mhd_output_file = scorer_output_path(
                    path, name_it->second, false);
            }
        }
    }
    const auto device = values.find("device");
    if (device != values.end()) {
        config.device = device->second;
    }
    if (!config.electron_transport_data_file.empty()) {
        config.electron_transport_data_file = resolve_input_path_from_config(
            config.electron_transport_data_file, path);
    }
    reject_unknown_config_keys(values, path);
    config.canonical_config_text = canonicalize_config(
        values, config.config_schema_version);
    config.validate();
    return config;
}

}  // namespace carbon
