#include <initializer_list>
#include "carbon/transport_config.hpp"
#include "carbon/source_parameters.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/min_json.hpp"
#include "carbon/neutron_package.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/delta_moments_data.hpp"
#include "carbon/secondary_rate_table.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
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
        "primary_particle",
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
    if (!manifest.contains("primary_particle") &&
        (!manifest.contains("primary_atomic_number") || !manifest.contains("primary_mass_number")))
        throw std::invalid_argument("Ion physics manifest requires primary_particle or both primary_atomic_number/primary_mass_number");
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
    const auto& str = iterator->second;
    std::size_t parsed = 0;
    if constexpr (std::is_integral_v<Number>) {
        if (str.find('.') != std::string::npos || str.find('e') != std::string::npos || str.find('E') != std::string::npos) {
            throw std::runtime_error("Invalid integer for '" + key + "': " + str);
        }
        if constexpr (std::is_unsigned_v<Number>) {
            if (!str.empty() && str.front() == '-') {
                throw std::runtime_error("Negative value not allowed for unsigned integer '" + key + "': " + str);
            }
            const auto number = std::stoull(str, &parsed);
            if (parsed != str.size()) {
                throw std::runtime_error("Invalid integer for '" + key + "': " + str);
            }
            if (number > static_cast<unsigned long long>(std::numeric_limits<Number>::max())) {
                throw std::runtime_error("Integer out of range for '" + key + "': " + str);
            }
            return static_cast<Number>(number);
        } else {
            const auto number = std::stoll(str, &parsed);
            if (parsed != str.size()) {
                throw std::runtime_error("Invalid integer for '" + key + "': " + str);
            }
            if (number < static_cast<long long>(std::numeric_limits<Number>::min()) ||
                number > static_cast<long long>(std::numeric_limits<Number>::max())) {
                throw std::runtime_error("Integer out of range for '" + key + "': " + str);
            }
            return static_cast<Number>(number);
        }
    } else {
        const auto number = std::stod(str, &parsed);
        if (parsed != str.size()) {
            throw std::runtime_error("Invalid number for '" + key + "': " + str);
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

PrimarySourceParameters parse_primary_source_parameters(const ConfigValues& values) {
    PrimarySourceParameters out;
    int z = 6, a = 12;
    const bool has_z = values.contains("primary_atomic_number");
    const bool has_a = values.contains("primary_mass_number");
    if (has_z != has_a)
        throw std::invalid_argument("Specify both primary_atomic_number and primary_mass_number");
    if (has_z) {
        z = parse_number(values, "primary_atomic_number", z);
        a = parse_number(values, "primary_mass_number", a);
    }
    if (const auto it = values.find("primary_particle"); it != values.end()) {
        auto name = it->second;
        if (name.size() >= 2 && ((name.front() == '"' && name.back() == '"') ||
                                (name.front() == '\'' && name.back() == '\'')))
            name = name.substr(1, name.size() - 2);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        int named_z = 0, named_a = 0;
        if (name == "proton") { named_z = 1; named_a = 1; }
        else if (name == "deuteron") { named_z = 1; named_a = 2; }
        else if (name == "triton") { named_z = 1; named_a = 3; }
        else if (name == "he3") { named_z = 2; named_a = 3; }
        else if (name == "alpha" || name == "he4") { named_z = 2; named_a = 4; }
        else if (name == "c12" || name == "carbon12") { named_z = 6; named_a = 12; }
        else if (name == "ion") {
            if (!has_z) throw std::invalid_argument("primary_particle: ion requires explicit Z and A");
        } else throw std::invalid_argument("Unknown primary_particle: " + name);
        if (named_z > 0) {
            if (has_z && (z != named_z || a != named_a))
                throw std::invalid_argument("primary_particle conflicts with primary_atomic_number/primary_mass_number");
            z = named_z; a = named_a;
        }
    }
    out.configured_rest_mass_MeV = parse_number(values, "primary_rest_mass_MeV", 0.0);
    out.ion = make_primary_ion_definition(z, a, out.configured_rest_mass_MeV);
    if (values.contains("initial_energy_MeV") && values.contains("initial_energy_MeVu"))
        throw std::invalid_argument("Specify only one of initial_energy_MeV (total) and initial_energy_MeVu");
    out.initial_energy_MeVu = values.contains("initial_energy_MeV")
        ? parse_number(values, "initial_energy_MeV", 0.0) / a
        : parse_number(values, "initial_energy_MeVu", out.initial_energy_MeVu);
    if (!std::isfinite(out.initial_energy_MeVu) || !(out.initial_energy_MeVu > 0) ||
        !std::isfinite(out.initial_total_energy_MeV()))
        throw std::invalid_argument("Primary source energy must be finite and positive");
    if (values.contains("beam_energy_spread_percent") && values.contains("beam_energy_spread"))
        throw std::invalid_argument("Specify only one of beam_energy_spread_percent and beam_energy_spread (relative RMS)");
    out.beam_energy_spread = values.contains("beam_energy_spread_percent")
        ? parse_number(values, "beam_energy_spread_percent", 0.0) / 100.0
        : parse_number(values, "beam_energy_spread", 0.0);
    if (!std::isfinite(out.beam_energy_spread) || out.beam_energy_spread < 0 || out.beam_energy_spread > 0.2)
        throw std::invalid_argument("Beam energy spread must be finite and in [0,20] percent RMS");
    return out;
}

void normalize_config_alias(ConfigValues& values, const char* old_key, const char* new_key) {
    const auto old = values.find(old_key);
    if (old == values.end()) return;
    if (values.contains(new_key))
        throw std::invalid_argument(std::string("Do not specify both ") + old_key + " and " + new_key);
    const auto value = old->second;
    const auto line = values.line_number(old_key);
    values.erase(old_key);
    values.add(new_key, value, line);
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

const char* material_physics_mode_name(const MaterialPhysicsMode mode) noexcept {
    switch (mode) {
    case MaterialPhysicsMode::Water: return "Water";
    case MaterialPhysicsMode::SchneiderCt: return "SchneiderCt";
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
    if (!neutron_proton_birth_output_file.empty() &&
        (!enable_neutron_transport || neutron_proton_birth_capacity==0 ||
         neutron_proton_birth_capacity>std::numeric_limits<std::uint32_t>::max()))
        throw std::invalid_argument("neutron_proton_birth_output_file requires neutron transport and capacity in [1, UINT32_MAX]");
    if ((enable_neutron_transport || !neutron_birth_output_file.empty()) &&
        (neutron_birth_capacity == 0 || neutron_birth_capacity > std::numeric_limits<std::uint32_t>::max()))
        throw std::invalid_argument("neutron_birth_capacity must be in [1, UINT32_MAX]");
    if ((enable_neutron_transport || !neutron_birth_output_file.empty()) &&
        (!enable_minibeam || enable_ct_grid || enable_layered_phantom || device == "serial"))
        throw std::invalid_argument("neutron_birth_output_file currently requires SYCL minibeam with homogeneous water");
    if (enable_neutron_transport) {
        if (neutron_package_file.empty())
            throw std::invalid_argument("enable_neutron_transport requires neutron_package_file");
        if (run_mode != RunMode::research)
            throw std::invalid_argument("Experimental neutron transport requires run_mode: research");
        const auto package=NeutronPackage::load(neutron_package_file,false);
        if(package.macroscopic_rates.empty())
            throw std::invalid_argument("Experimental neutron transport requires a v2 independent-rate package");
        if(!std::isfinite(neutron_world_half_width_mm) || neutron_world_half_width_mm<=urban_water_half_width_mm ||
           neutron_world_half_width_mm<=phantom_length_mm || urban_water_half_width_mm<=0 || neutron_max_collisions==0)
            throw std::invalid_argument("Invalid experimental neutron world or collision limit");
    }

    if (nuclear_model == "cinel02") {
        throw std::invalid_argument("nuclear_model cinel02 is retired; use the current shared CINEL03 bundle with geant4 selector");
    }
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
        (!quality_reject_any_queue_overflow || !quality_reject_nan_or_inf ||
         !quality_reject_voxel_over_total || !quality_reject_grid_closure)) {
        throw std::invalid_argument(
            "production run_mode requires queue-overflow, NaN/Inf, "
            "voxel-over-total and grid-closure rejection");
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
    (void)primary_ion(); // validate the complete source definition, including mass
    validate_primary_urban_reference_identity();
    if (primary_rest_mass_MeV > 0.0 && enable_minibeam && minibeam_copper_mcs_model == "urban")
        throw std::invalid_argument("Explicit primary mass requires urban_v2 or another mass-aware MCS model; legacy copper urban is deprecated");
    for (double cut : {minibeam_copper_production_cut_mm, minibeam_water_production_cut_mm})
        if (!std::isfinite(cut) || !(cut > 0.0))
            throw std::invalid_argument("Minibeam production cuts must be finite and positive");
    // Non-C12 primaries use the package-driven GPU route. Legacy carbon
    // diagnostics remain explicitly scoped to their reference projectile.
    if (primary_atomic_number != 6 || primary_mass_number != 12) {
        if (em_model != "g4_material_joint_v1" || device != "cuda")
            throw std::invalid_argument("Non-C12 primary requires CUDA unified EM and matching packages");
        if (ct_elastic_diagnostic)
            throw std::invalid_argument("Non-C12 primary cannot use C12 elastic diagnostics");
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
    if (!std::isfinite(broadbeam_maximum_mean_loss_fraction) ||
        broadbeam_maximum_mean_loss_fraction < 0.01 || broadbeam_maximum_mean_loss_fraction > 0.05 ||
        (broadbeam_maximum_mean_loss_fraction != 0.01 &&
         (transport_profile != "broadbeam_fast" || run_mode != RunMode::research)))
        throw std::invalid_argument("broadbeam_maximum_mean_loss_fraction requires research broadbeam_fast and [0.01,0.05]");
    for(double scale : {em_primary_step_scale,em_secondary_step_scale})
        if(!std::isfinite(scale) || scale<1.0 || scale>1.5)
            throw std::invalid_argument("Unified EM step scales must be finite in [1,1.5]");
    if((em_primary_step_scale!=1.0 || em_secondary_step_scale!=1.0) &&
       (em_model!="g4_material_joint_v1" || run_mode!=RunMode::research))
        throw std::invalid_argument("EM step extension requires research unified EM");
    if (secondary_step_chunking &&
        (em_model != "g4_material_joint_v1" ||
         !enable_secondary_transport || !enable_inelastic))
        throw std::invalid_argument("secondary_step_chunking requires unified secondary EM and inelastic secondary transport");
    if (em_model != "g4_material_joint_v1")
        throw std::invalid_argument("Water/CT legacy EM has been removed; use em_model: g4_material_joint_v1");
    if (device == "serial")
        throw std::invalid_argument("The serial legacy EM backend has been removed; use a SYCL device");
    {
        if (em_package_file.empty())
            throw std::invalid_argument("Unified EM requires one em_package_file");
        if (run_mode == RunMode::production &&
            (!enable_energy_straggling || !enable_secondary_energy_straggling))
            throw std::invalid_argument("Production unified EM requires primary/secondary fluctuations");
        if ((!enable_ct_grid && !is_water_mode()) || !slab_layers.empty() || enable_hetero_insert || enable_let_scoring)
            throw std::invalid_argument("Unified EM requires unified water or a Schneider CT grid, with LET off");
        if (!ct_secondary_exact_faces)
            throw std::invalid_argument("Unified EM requires exact CT faces and native unscaled fluctuations");
        if (!material_electron_charged_index_file.empty() &&
            run_mode!=RunMode::research)
            throw std::invalid_argument("Independent electrons require research mode");
    }
    if (maximum_primary_steps == 0) {
        throw std::invalid_argument("maximum_primary_steps must be greater than zero");
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
    const_cast<TransportConfig*>(this)->resolve_material_physics_mode();
    if (unified_water_nuclear_transport) {
        if (!is_water_mode() || enable_ct_grid ||
            !ct_grid_file.empty() || enable_layered_phantom || enable_hetero_insert ||
            !enable_voxel_scoring ||
            nuclear_model != "geant4" || enable_nuclear_elastic ||
            initial_energy_MeVu > 430.0 ||
            maximum_step_mm > 1.0 || maximum_relative_energy_loss > 0.005 + 1.e-6)
            throw std::invalid_argument("Unified water requires native homogeneous water, 3D scoring, geant4 selector and CT step limits; legacy water routing is retired");
        if (!primary_inelastic_package_v2_file.empty() || !primary_inelastic_rate_v2_file.empty() ||
            !water_cinel_package_file.empty() || !water_reaction_rate_file.empty() || !ct_cinel02_rate_file.empty())
            throw std::invalid_argument("Unified water forbids CINEL02/water event and rate fallback keys");
        if (ct_schneider_physics_bundle_file.empty() || ct_schneider_primary_rate_file.empty() ||
            ct_schneider_secondary_rate_file.empty() || ct_schneider_primary_cinel03_file.empty() ||
            ct_schneider_secondary_cinel03_file.empty() ||
            unified_water_material_file.empty())
            throw std::invalid_argument("Unified water requires explicit v2.1 bundle paths and pinned G4_WATER material");
    }
    if(enable_minibeam && material_electron_charged_index_file.empty())
        throw std::invalid_argument("Minibeam requires the shared FP32 electron packages; spatial-response replay is retired");
    if(material_electron_retain_bank &&
       (device=="serial" || material_electron_charged_index_file.empty() || run_mode!=RunMode::research))
        throw std::invalid_argument("Electron cache requires SYCL research shared-electron transport");
    if(material_electron_queue_budget_MiB<1 || material_electron_queue_budget_MiB>4096 ||
       material_electron_queue_chunk<1 ||
       material_electron_queue_chunk>(material_electron_group_size==32 ? 1048576U : 65536U) ||
       material_electron_queue_lanes<1 || material_electron_queue_lanes>65536 ||
       material_electron_group_size<1 || material_electron_group_size>32 ||
       (material_electron_group_size & (material_electron_group_size-1)))
        throw std::invalid_argument("Invalid electron queue budget/chunk/lanes/group size");
    if(broadbeam_electron_region_metric!="box" && broadbeam_electron_region_metric!="euclidean")
        throw std::invalid_argument("broadbeam_electron_region_metric must be box or euclidean");
    if(broadbeam_electron_region_diagnostics && !broadbeam_independent_electrons)
        throw std::invalid_argument("Electron region diagnostics require enabled CT broadbeam tracking");
    if(broadbeam_electron_external_air_local && broadbeam_electron_tracking &&
       (!broadbeam_independent_electrons || broadbeam_electron_body_mask_file.empty() ||
        !std::isfinite(broadbeam_electron_external_air_margin_mm) || broadbeam_electron_external_air_margin_mm<0 ||
        broadbeam_electron_external_air_margin_mm>100))
        throw std::invalid_argument("External-air electron localization requires CT tracking, a pinned BODY mask and margin 0..100 mm");
    if(!std::isfinite(broadbeam_electron_low_density_survival) || broadbeam_electron_low_density_survival<=0 ||
       broadbeam_electron_low_density_survival>1 || !std::isfinite(broadbeam_electron_tissue_survival) ||
       broadbeam_electron_tissue_survival<=0 || broadbeam_electron_tissue_survival>1 ||
       !std::isfinite(broadbeam_electron_expected_min_energy_MeV) ||
       broadbeam_electron_expected_min_energy_MeV<0 || broadbeam_electron_expected_min_energy_MeV>1)
        throw std::invalid_argument("broadbeam electron low-density/tissue survival must be in (0,1] and expected min energy in [0,1] MeV");
    if((broadbeam_electron_low_density_survival!=1 || broadbeam_electron_tissue_survival!=1 ||
        broadbeam_electron_birth_order || broadbeam_electron_expected_scoring) &&
       (!broadbeam_electron_tracking || !broadbeam_independent_electrons || !enable_ct_grid))
        throw std::invalid_argument("Low-density electron variance reduction requires CT broadbeam shared-electron tracking");
    if(broadbeam_electron_tracking) {
        if(transport_profile!="broadbeam_fast" || enable_minibeam || run_mode!=RunMode::research ||
           !std::isfinite(broadbeam_electron_density_threshold) || broadbeam_electron_density_threshold<=0 ||
           !std::isfinite(broadbeam_electron_halo_mm) || broadbeam_electron_halo_mm<0 || broadbeam_electron_halo_mm>100)
            throw std::invalid_argument("Broadbeam electron tracking requires research broadbeam_fast and valid density/halo selection");
        if(enable_ct_grid && !broadbeam_independent_electrons)
            throw std::invalid_argument("CT electron tracking requires the CT shared-electron preset and packages; a water-to-CT CLI override cannot disable tracking implicitly");
        // This is an empty selection, not a Schneider surrogate for water. If
        // water becomes eligible, require real Water_75eV electron data first.
        if(!enable_ct_grid && (!is_water_mode() || enable_hetero_insert || !slab_layers.empty() ||
           !std::isfinite(water_density_g_per_cm3) || water_density_g_per_cm3<=broadbeam_electron_density_threshold))
            throw std::invalid_argument("Selected water electron tracking requires Water_75eV birth/transport data; only an empty uniform-water selection is supported");
    }
    const bool broadbeam_independent=broadbeam_independent_electrons && transport_profile=="broadbeam_fast";
    if(broadbeam_independent_electrons &&
       (!broadbeam_independent || enable_minibeam || !enable_ct_grid || !fermi_eyges_gaussian_only ||
        material_electron_charged_index_file.empty() ||
        !std::isfinite(broadbeam_electron_density_threshold) || broadbeam_electron_density_threshold<=0 ||
        !std::isfinite(broadbeam_electron_halo_mm) || broadbeam_electron_halo_mm<0 || broadbeam_electron_halo_mm>100))
        throw std::invalid_argument("Broadbeam independent electrons require FP32 Gaussian CT research queue and valid density/halo selection");
    const bool independent_projectile=(primary_atomic_number==1&&primary_mass_number==1)||(primary_atomic_number==6&&primary_mass_number==12);
    if(!material_electron_charged_index_file.empty() &&
       (material_electron_birth_file.empty() ||
        material_photon_response_index_file.empty() || material_electron_joint_pilot_file.empty() ||
        run_mode!=RunMode::research ||
        !independent_projectile || (!enable_minibeam && !broadbeam_independent) ||
        enable_let_scoring || enable_charged_origin_voxel_scoring ||
        !enable_voxel_scoring))
        throw std::invalid_argument("Independent charged mode requires projectile-matched birth/joint/photon minibeam research queue; without LET/component/short-range scoring");
    if(!material_photon_response_index_file.empty() &&
       (material_electron_birth_file.empty()))
        throw std::invalid_argument("Independent photon bank requires a index and independent electron birth/joint queue");
    if(!material_electron_birth_file.empty() &&
       (material_electron_joint_pilot_file.empty() || run_mode!=RunMode::research ||
        !independent_projectile || (!enable_minibeam && !broadbeam_independent)))
        throw std::invalid_argument("Independent birth bank requires projectile-matched research minibeam joint queue");
    if(!material_electron_joint_pilot_file.empty() &&
       (run_mode!=RunMode::research || !independent_projectile ||
        (!enable_minibeam && !broadbeam_independent) || device=="serial" ||
        material_electron_charged_index_file.empty() ||
        (!broadbeam_independent && maximum_step_mm>.25)))
        throw std::invalid_argument("Joint pilot requires research projectile-matched minibeam, queue, bank and <=0.25 mm steps");
    if(!material_electron_charged_index_file.empty() && !enable_ct_grid &&
       (!is_water_mode() || enable_hetero_insert || enable_layered_phantom || !slab_layers.empty() ||
        water_density_g_per_cm3!=1.0 || multiple_scattering_model!="urban_v2"))
        throw std::invalid_argument("Water shared electrons require homogeneous unit-density Water_75eV and global Urban");
    if (material_physics_mode == MaterialPhysicsMode::SchneiderCt) {
        if (ct_grid_file.empty()) {
            throw std::invalid_argument("MaterialPhysicsMode::SchneiderCt requires ct_grid_file");
        }
        if (enable_layered_phantom || enable_hetero_insert) {
            throw std::invalid_argument(
                "enable_ct_grid cannot combine with layered phantom or hetero insert");
        }
        if (!is_primary_attenuation_only_mode()) {
            if (!ct_air_cross_section_file.empty() || !ct_lung_cross_section_file.empty() ||
                !ct_water_cross_section_file.empty() || !ct_bone_cross_section_file.empty()) {
                throw std::invalid_argument(
                    "Four-class material tables (air/lung/water/bone) are strictly forbidden in production Schneider CT path");
            }
            // Ambiguous water/CINEL02 keys are forbidden in the full
            // Schneider CT path: water package/rate files and the CT
            // CINEL02 rate table must not be present, and the water
            // nuclear model must not be selected. Schneider CT runs must
            // use the ct_schneider_* tables exclusively.
            if (!primary_inelastic_package_v2_file.empty() ||
                !primary_inelastic_rate_v2_file.empty() ||
                !water_cinel_package_file.empty() ||
                !water_reaction_rate_file.empty() ||
                !ct_cinel02_rate_file.empty()) {
                throw std::invalid_argument(
                    "Schneider CT full mode forbids water/CINEL02 physics keys "
                    "(primary_inelastic_package_v2_file, primary_inelastic_rate_v2_file, "
                    "water_cinel_package_file, water_reaction_rate_file, ct_cinel02_rate_file); "
                    "use ct_schneider_* tables only");
            }
            if (nuclear_model == "cinel02") {
                throw std::invalid_argument(
                    "Schneider CT full mode forbids nuclear_model 'cinel02' (water correlated "
                    "final states); the Schneider rate + CINEL03 path is the only allowed "
                    "nuclear configuration on CT");
            }
        }
        // v3 bundle: the masked rate-binary hazard replaces the CSV XS table,
        // so the CSV key is intentionally empty (mixing refused elsewhere).
        const bool v3_bundle_mode = !ct_schneider_physics_bundle_file.empty();
        if (!ct_schneider_file.empty() && nuclear_model != "none" && ct_schneider_cross_section_file.empty() &&
            !v3_bundle_mode) {
            throw std::invalid_argument(
                "CT Schneider-25 mode with active nuclear model requires ct_schneider_cross_section_file; "
                "fallback to water or four-class XS is forbidden.");
        }
        if ((!ct_schneider_cross_section_file.empty() || v3_bundle_mode) && nuclear_model != "none") {
            if (!v3_bundle_mode && (primary_atomic_number != 6 || primary_mass_number != 12)) {
                throw std::invalid_argument(
                    "Schneider primary cross section is validated for C12 (Z=6, A=12) primaries only, got Z=" +
                    std::to_string(primary_atomic_number) + ", A=" + std::to_string(primary_mass_number));
            }
            if (enable_nuclear_elastic) {
                throw std::invalid_argument(
                    "Nuclear elastic scattering is not supported under Schneider primary cross section mode");
            }
            if (maximum_relative_energy_loss > 0.005 + 1e-6) {
                throw std::invalid_argument(
                    "Schneider primary cross section mode requires maximum_relative_energy_loss <= 0.005, got " +
                    std::to_string(maximum_relative_energy_loss));
            }
            if (maximum_step_mm > 1.0 + 1e-6) {
                throw std::invalid_argument(
                    "Schneider primary cross section mode requires maximum_step_mm <= 1.0, got " +
                    std::to_string(maximum_step_mm));
            }
        }
    }
    if (!ct_validation_mode.empty() && ct_validation_mode != "none") {
        if (ct_validation_mode != "primary-attenuation-only") {
            throw std::invalid_argument("Unknown ct_validation_mode: '" + ct_validation_mode +
                                        "'; supported modes: 'none', 'primary-attenuation-only'");
        }
        if (primary_atomic_number != 6 || primary_mass_number != 12) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires C12 projectile (Z=6, A=12)");
        }
        if (!enable_ct_grid) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires enable_ct_grid = true");
        }
        if (ct_schneider_cross_section_file.empty()) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires ct_schneider_cross_section_file");
        }
        if (!enable_inelastic) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires enable_inelastic = true");
        }
        if (enable_nuclear_elastic) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires enable_nuclear_elastic = false");
        }
        if (enable_secondary_transport) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires enable_secondary_transport = false");
        }
        if (enable_energy_straggling) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires enable_energy_straggling = false");
        }
        if (beam_energy_spread != 0.0) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires beam_energy_spread = 0.0");
        }
        if (initial_energy_MeVu > 430.0 + 1e-5) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires initial_energy_MeVu <= 430.0 MeV/u, got " +
                                        std::to_string(initial_energy_MeVu));
        }
        if (energy_cutoff_MeV < 6.0 - 1e-5) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' requires energy_cutoff_MeV >= 6.0 MeV (>= 0.5 MeV/u for C12), got " +
                                        std::to_string(energy_cutoff_MeV));
        }
        if (!primary_inelastic_package_v2_file.empty()) {
            throw std::invalid_argument("ct_validation_mode 'primary-attenuation-only' forbids CINEL package / final-state replay");
        }
    }
    if (enable_primary_loss_query_audit && run_mode != RunMode::smoke) {
        throw std::invalid_argument("enable_primary_loss_query_audit requires smoke mode");
    }
    if (nuclear_model != "geant4" &&
        nuclear_model != "cinel02" && nuclear_model != "none") {
        throw std::invalid_argument(
            "nuclear_model must be geant4, cinel02, or none");
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

    if (enable_nuclear_elastic) {
        throw std::invalid_argument(
            "enable_nuclear_elastic is retired; use the all-ion elastic configuration");
    }
    if(transport_profile!="reference" && transport_profile!="broadbeam_fast" && transport_profile!="minibeam_detailed")
        throw std::invalid_argument("transport_profile must be reference, broadbeam_fast or minibeam_detailed");
    if(transport_profile=="minibeam_detailed" && !enable_minibeam)
        throw std::invalid_argument("minibeam_detailed requires minibeam: true");
    if(transport_profile=="broadbeam_fast") {
        if(enable_minibeam || device=="serial" || run_mode!=RunMode::research ||
           em_model!="g4_material_joint_v1" ||
           multiple_scattering_model!="fermi_eyges" ||
           fermi_eyges_species!="all_charged" || fermi_eyges_parameter_set!="species_water" ||
           !enable_multiple_scattering || multiple_scattering_scale!=1.0 ||
           enable_neutron_transport ||
           !urban_mcs_package_file.empty() ||

           !neutron_package_file.empty() ||

           (enable_ct_grid && (!enable_ct_material_mcs || ct_secondary_mcs_off_diagnostic)))
            throw std::invalid_argument("broadbeam_fast requires research unified EM, FE for all charged species, CT material MCS, no beamline/neutron transport; only pinned research C12 CT joint electron response is supported");
    }
    if (fermi_eyges_sampling != "segmented" && fermi_eyges_sampling != "moments")
        throw std::invalid_argument("fermi_eyges_sampling must be segmented or moments");
    if (fermi_eyges_sampling == "moments" &&
        (!fermi_eyges_gaussian_only || enable_minibeam || run_mode != RunMode::research || device == "serial"))
        throw std::invalid_argument("FE moments requires research SYCL Gaussian FE without minibeam");
    if(fermi_eyges_gaussian_only && (multiple_scattering_model!="fermi_eyges" ||
                                   run_mode!=RunMode::research || device=="serial"))
        throw std::invalid_argument("fermi_eyges_gaussian_only requires research SYCL fermi_eyges MCS");
    if (multiple_scattering_model != "highland" &&
        multiple_scattering_model != "fermi_eyges" &&
        multiple_scattering_model != "urban_v2") {
        throw std::invalid_argument(
            "multiple_scattering_model must be highland, fermi_eyges or urban_v2");
    }


    if(urban_ct_density_scaling && (run_mode!=RunMode::research ||
       !enable_ct_grid || multiple_scattering_model!="urban_v2"))
        throw std::invalid_argument("urban_ct_density_scaling requires research Schneider CT with global Urban");
    if(multiple_scattering_model=="urban_v2") {
        if(device!="cuda")
            throw std::invalid_argument("Global urban_v2 currently requires the CUDA SYCL transport path");
        if(urban_mcs_package_file.empty())
            throw std::invalid_argument("Global urban_v2 requires a validated urban_mcs_package_file");
        if(!enable_multiple_scattering || multiple_scattering_scale!=1.0 ||
           em_model!="g4_material_joint_v1" ||
           ct_secondary_mcs_off_diagnostic || enable_hetero_insert ||
           !slab_layers.empty())
            throw std::invalid_argument("Global urban_v2 requires unscaled MCS, unified EM, and ordinary water or Schneider CT");
        if(enable_ct_grid && (!is_schneider_ct_mode() || !ct_secondary_exact_faces))
            throw std::invalid_argument("Global urban_v2 requires Schneider CT and exact secondary faces");
        if(energy_cutoff_MeV<0.05 || effective_secondary_local_deposit_cutoff_MeV()<0.05)
            throw std::invalid_argument("Global urban_v2 reference currently requires transport cutoffs >= 0.05 MeV");
        if(!enable_ct_grid && (!(urban_water_half_width_mm>0) || !std::isfinite(urban_water_half_width_mm)))
            throw std::invalid_argument("Global water urban_v2 requires the TOPAS physical urban_water_half_width_mm");
    }
    if (fermi_eyges_species != "c12" &&
        fermi_eyges_species != "c12_he4" &&
        fermi_eyges_species != "c12_he4_pdt" &&
        fermi_eyges_species != "all_charged") {
        throw std::invalid_argument(
            "fermi_eyges_species must be c12, c12_he4, "
            "c12_he4_pdt, or all_charged");
    }
    if (fermi_eyges_parameter_set != "shared_c12" &&
        fermi_eyges_parameter_set != "species_water") {
        throw std::invalid_argument(
            "fermi_eyges_parameter_set must be shared_c12 or species_water");
    }
    if (!std::isfinite(fermi_eyges_max_segment_mm) ||
        fermi_eyges_max_segment_mm <= 0.0) {
        throw std::invalid_argument(
            "fermi_eyges_max_segment_mm must be finite and positive");
    }
    if (multiple_scattering_model == "fermi_eyges" &&
        (primary_atomic_number != 6 || primary_mass_number != 12) &&
        fermi_eyges_parameter_set != "species_water") {
        throw std::invalid_argument(
            "Non-C12 primary FE requires species_water parameters; shared_c12 is a carbon reference");
    }

    if (!std::isfinite(multiple_scattering_scale) ||
        multiple_scattering_scale < 0.0 || multiple_scattering_scale > 3.0) {
        throw std::invalid_argument(
            "multiple_scattering_scale must be in [0, 3]");
    }
    if (multiple_scattering_model == "fermi_eyges" &&
        multiple_scattering_scale != 1.0) {
        throw std::invalid_argument(
            "multiple_scattering_scale applies only to Highland; use 1.0 "
            "with multiple_scattering_model=fermi_eyges");
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
        // CCTG stores spacing and its derived extent as float.  For resampled
        // grids whose spacing is not exactly representable, recomputing the
        // extent in double can differ by a few ulps of the float extent.
        const auto extent_tolerance_mm =
            1.0e-6 * std::max(1.0, std::abs(phantom_length_mm));
        if (std::abs(voxel_size_z_mm - depth_bin_width_mm) > 1.0e-6 ||
            std::abs(scorer_z_extent_mm - phantom_length_mm) >
                extent_tolerance_mm) {
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
    if (enable_minibeam_energy_band_roi_scoring &&
        (!enable_minibeam || !enable_voxel_scoring ||
         !enable_charged_origin_voxel_scoring ||
         charged_origin_voxel_mhd_output_prefix.empty())) {
        throw std::invalid_argument(
            "enable_minibeam_energy_band_roi_scoring requires minibeam, "
            "voxel/component scoring and charged_origin_voxel_mhd_output_prefix");
    }
    if (enable_minibeam_primary_roi_scoring &&
        (!enable_minibeam || !enable_voxel_scoring)) {
        throw std::invalid_argument(
            "enable_minibeam_primary_roi_scoring requires minibeam and "
            "enable_voxel_scoring");
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
    if (!primary_voxel_fluence_mhd_output_file.empty() &&
        !enable_voxel_scoring) {
        throw std::invalid_argument(
            "primary_voxel_fluence_mhd_output_file requires "
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
    if (max_device_memory_fraction <= 0.05 || max_device_memory_fraction > 1.0) {
        throw std::invalid_argument(
            "max_device_memory_fraction must be in (0.05, 1.0]");
    }
    if (!std::isfinite(device_memory_budget_gib) || device_memory_budget_gib < 0.0) {
        throw std::invalid_argument(
            "device_memory_budget_gib must be finite and nonnegative (0 = unlimited)");
    }
    if (device_memory_budget_gib >
        static_cast<double>(std::numeric_limits<std::size_t>::max()) /
            (1024.0 * 1024.0 * 1024.0)) {
        throw std::invalid_argument(
            "device_memory_budget_gib exceeds the addressable byte range");
    }
    if (secondary_queue_capacity == 0 ||
        secondary_queue_capacity > 4000000000ULL) {
        throw std::invalid_argument(
            "secondary_queue_capacity must be in (0, 4e9]");
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
        spots_geometry_mode != "minibeam_topas_y" &&
        spots_geometry_mode != "minibeam_pbs_y") {
        throw std::invalid_argument(
            "spots_geometry_mode must be topas, beam_plus_z, tps_90, "
            "tps_gantry_y, minibeam_topas_y, or minibeam_pbs_y");
    }
    if (spots_geometry_mode == "minibeam_pbs_y" &&
        (!enable_minibeam || enable_flat_source || device == "serial" ||
         uses_fixed_patient_coordinates())) {
        throw std::invalid_argument(
            "minibeam_pbs_y requires SYCL minibeam emittance in canonical coordinates");
    }



    if ((!minibeam_water_primary_plane_output_file.empty() ||
         !minibeam_water_primary_plane_depths_mm.empty()) &&
        !enable_minibeam) {
        throw std::invalid_argument(
            "minibeam water-plane diagnostics require minibeam=true");
    }
    if (minibeam_water_primary_plane_output_file.empty() !=
        minibeam_water_primary_plane_depths_mm.empty()) {
        throw std::invalid_argument(
            "minibeam_water_primary_plane_output_file and "
            "minibeam_water_primary_plane_depths_mm must be configured together");
    }
    if (minibeam_water_primary_plane_depths_mm.size() > 16) {
        throw std::invalid_argument(
            "minibeam water-plane diagnostics support at most 16 depths");
    }
    for (std::size_t index = 0;
         index < minibeam_water_primary_plane_depths_mm.size(); ++index) {
        const auto depth = minibeam_water_primary_plane_depths_mm[index];
        if (!(depth > 0.0 && depth < phantom_length_mm) ||
            (index != 0 &&
             !(depth > minibeam_water_primary_plane_depths_mm[index - 1]))) {
            throw std::invalid_argument(
                "minibeam water-plane depths must be strictly increasing and "
                "inside the water phantom");
        }
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
        if ((primary_atomic_number != 6 || primary_mass_number != 12) &&
            (multiple_scattering_model != "urban_v2" ||
             minibeam_copper_mcs_model != "urban_v2" ||
             minibeam_transport_mode != "copper_em")) {
            throw std::invalid_argument(
                "Non-C12 minibeam requires global urban_v2, copper urban_v2 "
                "and matching primary copper physics packages");
        }

        if (minibeam_transport_mode != "copper_em") {
            throw std::invalid_argument(
                "minibeam_transport_mode must be copper_em");
        }
        if (minibeam_material != "Copper") {
            throw std::invalid_argument(
                "the current minibeam reference supports minibeam_material=Copper");
        }
        if (!(minibeam_collimator_center_to_isocenter_mm >= 0.0 &&
              minibeam_collimator_width_mm > 0.0 &&
              minibeam_collimator_length_mm > 0.0 &&
              minibeam_radius_mm > 0.0 &&
              minibeam_collimator_thickness_mm > 0.0 &&
              minibeam_slit_length_mm > 0.0 &&
              minibeam_slit_thickness_mm > 0.0 &&
              minibeam_slit_width_mm > 0.0 && minibeam_slit_pitch_mm > 0.0 &&
              minibeam_slit_half_length_mm > 0.0 &&
              minibeam_slit_length_mm <= minibeam_collimator_length_mm &&
              minibeam_slit_thickness_mm <= minibeam_collimator_thickness_mm)) {
            throw std::invalid_argument(
                "minibeam dimensions must be positive and each slit dimension "
                "must fit inside the Copper collimator");
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
            if (minibeam_copper_enable_nuclear_attenuation &&
                minibeam_copper_inclxx_file.empty()) {
                throw std::invalid_argument(
                    "minibeam Copper INCLXX requires minibeam_copper_inclxx_file");
            }
            if (minibeam_copper_primary_native_mean_loss &&
                (run_mode == RunMode::production || primary_atomic_number != 1 ||
                 primary_mass_number != 1 || multiple_scattering_model != "urban_v2" ||
                 minibeam_copper_mcs_model != "urban_v2" || !minibeam_copper_enable_mcs ||
                 !minibeam_copper_enable_energy_straggling ||
                 minibeam_copper_proton_straggling_model != "g4_universal_delta" ||
                 minibeam_copper_survivor_energy_loss_scale != 1.0 ||
                 !minibeam_copper_survivor_energy_loss_energies_MeVu.empty()))
                throw std::invalid_argument("Native Cu primary mean requires research proton Urban v2, finite-cut fluctuations and no stopping calibration");
            if (minibeam_copper_proton_straggling_model != "legacy_clipped_gaussian" &&
                minibeam_copper_proton_straggling_model != "g4_universal_total" &&
                minibeam_copper_proton_straggling_model != "g4_universal_delta")
                throw std::invalid_argument("Unknown copper proton straggling model");
            if (minibeam_copper_proton_straggling_model == "g4_universal_delta" &&
                std::abs(minibeam_copper_density_g_per_cm3 - 8.96) > 1e-5)
                throw std::invalid_argument("Native copper proton cut is pinned to G4_Cu at 8.96 g/cm3");
            if (minibeam_copper_proton_straggling_model != "legacy_clipped_gaussian" &&
                (run_mode == RunMode::production || minibeam_copper_straggling_scale != 1.0 ||
                 minibeam_copper_fragment_straggling_scale != 1.0))
                throw std::invalid_argument("Native copper proton fluctuation requires research/smoke and unit straggling scales");
            if (minibeam_copper_enable_elastic &&
                minibeam_copper_elastic_file.empty()) {
                throw std::invalid_argument(
                    "minibeam copper_em requires minibeam_copper_elastic_file");
            }
            if (minibeam_copper_mcs_model == "urban_v2" &&
                multiple_scattering_model != "urban_v2" &&
                minibeam_copper_loss_range_file.empty()) {
                throw std::invalid_argument(
                    "minibeam urban_v2 requires "
                    "minibeam_copper_loss_range_file (restricted loss-range "
                    "table); the legacy E/stopping range is never used");
            }
            if (!(minibeam_copper_density_g_per_cm3 > 0.0 &&
                  minibeam_copper_radiation_length_g_per_cm2 > 0.0 &&
                  minibeam_copper_max_step_mm > 0.0 &&
                  minibeam_copper_mcs_scale > 0.0 &&
                  minibeam_copper_fragment_mcs_scale > 0.0 &&
                  minibeam_copper_fragment_cascade_generations <= 3U &&
                  minibeam_copper_straggling_scale > 0.0 &&
                  minibeam_copper_fragment_straggling_scale > 0.0 &&
                  minibeam_copper_survivor_energy_loss_scale > 0.0 &&
                  minibeam_copper_survivor_energy_loss_scale <= 2.0 &&
                  minibeam_water_primary_stopping_power_scale == 1.0) ||
                !std::isfinite(minibeam_copper_mcs_scale) ||
                !std::isfinite(minibeam_copper_fragment_mcs_scale) ||
                !std::isfinite(minibeam_copper_straggling_scale) ||
                !std::isfinite(minibeam_copper_fragment_straggling_scale) ||
                !std::isfinite(
                    minibeam_copper_survivor_energy_loss_scale) ||
                !std::isfinite(
                    minibeam_water_primary_stopping_power_scale)) {
                throw std::invalid_argument(
                    "minibeam Copper density, radiation length, max step, and "
                    "MCS/straggling scales must be finite/positive and survivor energy-loss "
                    "and water stopping/MCS scales must be finite and in (0, 2]; "
                    "the water low-energy MCS transition must be nonnegative");
            }
            if (minibeam_copper_mcs_model != "urban_v2") {
                throw std::invalid_argument("minibeam_copper_mcs_model must be urban_v2");
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
    if (ct_secondary_mcs_off_diagnostic &&
        (run_mode!=RunMode::smoke || !enable_ct_grid || ct_schneider_physics_bundle_file.empty()))
        throw std::invalid_argument("Secondary MCS-off requires smoke Schneider CT bundle");
    if (spots_enable_upstream_air_mcs &&
        (run_mode != RunMode::smoke || !enable_tps_source || !enable_ct_grid ||
         !spots_enable_upstream_air_energy_loss || !enable_multiple_scattering ||
         spots_upstream_air_mcs_file.empty())) {
        throw std::invalid_argument("upstream air MCS requires smoke TPS CT, air energy loss and MCS enabled");
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
        for (std::size_t s = 0; s < primary_spot_batch.size(); ++s) {
            const auto& entry = primary_spot_batch[s];
            if (entry.history_begin != expected_begin ||
                entry.history_end <= entry.history_begin) {
                throw std::invalid_argument(
                    "primary_spot_batch history ranges must be contiguous and non-empty");
            }
            expected_begin = entry.history_end;
            const float spot_energy_MeV = entry.floats[0];
            const float spot_spread = entry.floats[1];
            if (!std::isfinite(spot_energy_MeV) || spot_energy_MeV <= 0.0F) {
                throw std::invalid_argument(
                    "primary_spot_batch[" + std::to_string(s) + "] energy must be finite and positive");
            }
            if (!std::isfinite(spot_spread) || spot_spread < 0.0F) {
                throw std::invalid_argument(
                    "primary_spot_batch[" + std::to_string(s) + "] energy spread must be finite and nonnegative");
            }
            if (is_primary_attenuation_only_mode()) {
                const double spot_energy_MeVu =
                    static_cast<double>(spot_energy_MeV) / static_cast<double>(primary_mass_number);
                if (spot_energy_MeVu > 430.0 + 1e-5) {
                    throw std::invalid_argument(
                        "primary-attenuation-only mode requires all spot energies <= 430 MeV/u, spot " +
                        std::to_string(s) + " has " + std::to_string(spot_energy_MeVu) + " MeV/u");
                }
                if (spot_spread != 0.0F) {
                    throw std::invalid_argument(
                        "primary-attenuation-only mode requires all spot energy spreads == 0.0, spot " +
                        std::to_string(s) + " has " + std::to_string(spot_spread));
                }
            }
        }
        if (expected_begin != number_of_histories) {
            throw std::invalid_argument(
                "primary_spot_batch ranges must cover number_of_histories exactly");
        }
    }
}

// v2.1 physics bundle enforcement. Content-pinning (SHA equality), not
// path-pinning: any file with matching SHA is accepted, any v1/v3 mixing
// is refused. Water mode never reaches here (is_schneider_ct_mode gate).
void validate_schneider_physics_bundle(const TransportConfig& config) {
    const std::filesystem::path primary_rate =
        config.ct_schneider_primary_rate_file;
    const std::filesystem::path secondary_rate =
        config.ct_schneider_secondary_rate_file;

    auto peek_version = [](const std::filesystem::path& p) -> std::uint32_t {
        if (!std::filesystem::exists(p)) {
            return 0;
        }
        std::ifstream in(p, std::ios::binary);
        char magic[8]{};
        std::uint32_t version{0};
        in.read(magic, 8);
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!in) {
            return 0;
        }
        if (std::strncmp(magic, "SCHNRATE", 8) != 0 && std::memcmp(magic, "SCHN2RAT", 8) != 0) {
            return 0;
        }
        return version;
    };
    const std::uint32_t primary_ver = peek_version(primary_rate);
    const std::uint32_t secondary_ver = peek_version(secondary_rate);

    if (config.ct_schneider_physics_bundle_file.empty()) {
        if (primary_ver == 3 || secondary_ver == 3) {
            throw std::runtime_error(
                "Schneider CT startup failed: v3 rate file requires "
                "ct_schneider_physics_bundle_file (refusing unbundled v3)");
        }
        throw std::runtime_error("Current CINEL03 requires a v2.1 physics bundle; unbundled legacy rates are retired");
    }
    if (primary_ver != 3 || secondary_ver != 3) {
        throw std::runtime_error("Current CINEL03 requires primary and secondary v3 rate binaries");
    }

    const std::filesystem::path& bundle_path = config.ct_schneider_physics_bundle_file;
    if (!std::filesystem::exists(bundle_path)) {
        throw std::runtime_error("Schneider CT startup failed: bundle file missing: " +
                                 bundle_path.string());
    }
    std::ifstream bundle_in(bundle_path);
    std::string bundle_content((std::istreambuf_iterator<char>(bundle_in)),
                               std::istreambuf_iterator<char>());
    minjson::Parser parser(bundle_content);
    const minjson::Value bundle = parser.parse();
    if (minjson::require_uint(bundle.at("schema_version"), "schema_version") != 1) {
        throw std::runtime_error("Schneider CT startup failed: unsupported bundle schema_version");
    }
    if (bundle.contains("source_energy_scope_MeV")) {
        const auto& scope = bundle.at("source_energy_scope_MeV");
        if (scope.type != minjson::Value::Type::Array || scope.arr.size() != 2)
            throw std::runtime_error("Invalid primary bundle source energy scope");
        const double lo = minjson::require_number(scope.arr[0], "source energy min");
        const double hi = minjson::require_number(scope.arr[1], "source energy max");
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo <= 0 || hi < lo ||
            config.initial_total_energy_MeV() < lo || config.initial_total_energy_MeV() > hi)
            throw std::runtime_error("Primary source energy is outside the physics bundle extraction scope");
    }

    const std::filesystem::path c12_cinel =
        config.ct_schneider_primary_cinel03_file;
    const std::filesystem::path sec_cinel =
        config.ct_schneider_secondary_cinel03_file;
    for (const auto& required : {primary_rate, secondary_rate, c12_cinel, sec_cinel}) {
        if (!std::filesystem::is_regular_file(required))
            throw std::runtime_error("Missing nuclear physics package: " + required.string());
    }

    // v3 requires exact versions; any v1 file with a bundle is mixing.
    if (primary_ver != 3 || secondary_ver != 3) {
        throw std::runtime_error(
            "Schneider CT startup failed: bundle requires v3 primary AND secondary rate files "
            "(refusing v1/v3 mixing)");
    }
    // v3 has no CSV XS table; the key must be empty.
    if (!config.ct_schneider_cross_section_file.empty()) {
        throw std::runtime_error(
            "Schneider CT startup failed: bundle (v3) requires empty "
            "ct_schneider_cross_section_file (refusing CSV/v3 mixing)");
    }

    // Full rate loads: schema + masked closure + metadata verified by
    // construction (separate from the transport loads that follow).
    const SecondaryRateTable sec_table = SecondaryRateTable::from_binary(secondary_rate);
    const SchneiderRateTable pri_table = SchneiderRateTable::from_binary(primary_rate);
    if (pri_table.projectile_z() != config.primary_atomic_number ||
        pri_table.projectile_a() != config.primary_mass_number)
        throw std::runtime_error("Primary rate projectile does not match YAML source Z/A; new packages require projectile metadata");
    // Registry: bundle order == rate keys (order-sensitive, no aliasing).
    const minjson::Value& registry = bundle.at("projectile_registry");
    if (registry.type != minjson::Value::Type::Array ||
        registry.arr.size() != sec_table.projectiles().size()) {
        throw std::runtime_error("Schneider CT startup failed: bundle registry size != rate keys");
    }
    for (std::size_t i = 0; i < registry.arr.size(); ++i) {
        const int z = static_cast<int>(minjson::require_uint(registry.arr[i].at("z"), "registry.z"));
        const int a = static_cast<int>(minjson::require_uint(registry.arr[i].at("a"), "registry.a"));
        if (z != sec_table.projectiles()[i].z || a != sec_table.projectiles()[i].a) {
            throw std::runtime_error("Schneider CT startup failed: bundle registry mismatch at index " +
                                     std::to_string(i));
        }
    }
    const minjson::Value& targets = bundle.at("target_order");
    if (targets.type != minjson::Value::Type::Array || targets.arr.size() != 13) {
        throw std::runtime_error("Schneider CT startup failed: bundle target_order size != 13");
    }

    // Package sidecars: channels.json next to each package binary;
    // projectile set == registry, bounds == rate domains (no 400MB load).
    auto read_sidecar = [](const std::filesystem::path& pkg_path, const minjson::Value& pkg_section,
                           const std::string& role) {
        const std::string channels_name = minjson::require_string(
            pkg_section.at("channels_file"), role + ".channels_file");
        const std::filesystem::path sidecar =
            pkg_path.parent_path() / std::filesystem::path(channels_name).filename();
        if (!std::filesystem::exists(sidecar)) {
            throw std::runtime_error("Schneider CT startup failed: package channels sidecar missing: " +
                                     sidecar.string());
        }
        std::ifstream side_in(sidecar);
        std::string side_content((std::istreambuf_iterator<char>(side_in)),
                                 std::istreambuf_iterator<char>());
        minjson::Parser side_parser(side_content);
        return side_parser.parse();
    };

    const minjson::Value sec_side = read_sidecar(sec_cinel, bundle.at("secondary_package"), "sec");
    {
        std::map<std::pair<int, int>, int> proj_seen;
        for (const auto& c : sec_side.at("channels").arr) {
            const int pz = static_cast<int>(minjson::require_uint(c.at("projectile_z"), "ch.pz"));
            const int pa = static_cast<int>(minjson::require_uint(c.at("projectile_a"), "ch.pa"));
            const int tz = static_cast<int>(minjson::require_uint(c.at("target_element_z"), "ch.tz"));
            const double lo = minjson::require_number(c.at("energy_min_MeV_per_u"), "ch.lo");
            const double hi = minjson::require_number(c.at("energy_max_MeV_per_u"), "ch.hi");
            proj_seen[{pz, pa}]++;
            const int pi = sec_table.projectile_index(pz, pa);
            if (pi < 0) {
                throw std::runtime_error(
                    "Schneider CT startup failed: package channel projectile not in registry");
            }
            const std::size_t ti = SecondaryRateTable::target_index_from_z(tz);
            const auto& dom = sec_table.channel_domain(static_cast<std::size_t>(pi), ti);
            if (!dom.has_support || dom.energy_min_mevu != lo || dom.energy_max_mevu != hi) {
                throw std::runtime_error(
                    "Schneider CT startup failed: rate/package domain mismatch");
            }
        }
        if (proj_seen.size() != registry.arr.size()) {
            throw std::runtime_error(
                "Schneider CT startup failed: package/registry projectile count mismatch");
        }
    }
    const minjson::Value pri_side = read_sidecar(c12_cinel, bundle.at("primary_package"), "pri");
    {
        std::array<bool, kSchneiderNumTargets> seen_targets{};
        if (pri_side.at("channels").arr.size() != 13) {
            throw std::runtime_error(
                "Schneider CT startup failed: primary package must have 13 channels");
        }
        for (const auto& c : pri_side.at("channels").arr) {
            const int pz = static_cast<int>(minjson::require_uint(c.at("projectile_z"), "ch.pz"));
            const int pa = static_cast<int>(minjson::require_uint(c.at("projectile_a"), "ch.pa"));
            if (pz != config.primary_atomic_number || pa != config.primary_mass_number) {
                throw std::runtime_error(
                    "Schneider CT startup failed: primary package projectile does not match the YAML source");
            }
            const int tz = static_cast<int>(minjson::require_uint(c.at("target_element_z"), "ch.tz"));
            const double lo = minjson::require_number(c.at("energy_min_MeV_per_u"), "ch.lo");
            const double hi = minjson::require_number(c.at("energy_max_MeV_per_u"), "ch.hi");
            const std::size_t ti = SchneiderRateTable::target_index_from_z(tz);
            if (seen_targets[ti])
                throw std::runtime_error("Duplicate primary package target channel");
            seen_targets[ti] = true;
            const auto& dom = pri_table.channel_domain(ti);
            if (c.contains("has_support") &&
                c.at("has_support").type != minjson::Value::Type::Boolean)
                throw std::runtime_error("Primary channel has_support must be boolean");
            const bool has_support = !c.contains("has_support") || c.at("has_support").boolean;
            if (static_cast<bool>(dom.has_support) != has_support ||
                dom.energy_min_mevu != lo || dom.energy_max_mevu != hi) {
                throw std::runtime_error(
                    "Schneider CT startup failed: primary rate/package domain mismatch");
            }
            // An explicitly empty channel is allowed only when every stored
            // partial rate is zero. For example, p+H below pion threshold.
            if (!has_support) {
                for (std::size_t section = 0; section < kSchneiderNumSections; ++section)
                    for (std::size_t energy = 0; energy < pri_table.num_energies(); ++energy)
                        if (pri_table.mass_partial_rate(section, ti, energy) != 0.0)
                            throw std::runtime_error("Unsupported primary channel has nonzero rates");
            }
        }
    }

    std::cout << "[schneider-bundle] domains and projectile registry verified: "
              << registry.arr.size() << " projectiles\n";
}

void validate_schneider_ct_startup(const TransportConfig& config) {
    if (!config.is_schneider_ct_mode() && !config.unified_water_nuclear_transport) {
        return;
    }
    if (config.is_schneider_ct_mode() && config.ct_grid_file.empty()) {
        throw std::invalid_argument("MaterialPhysicsMode::SchneiderCt requires ct_grid_file");
    }
    if (config.is_primary_attenuation_only_mode()) {
        if (config.ct_schneider_cross_section_file.empty() ||
            !std::filesystem::exists(config.ct_schneider_cross_section_file)) {
            throw std::invalid_argument(
                "ct_validation_mode 'primary-attenuation-only' requires valid ct_schneider_cross_section_file");
        }
        return;
    }

    if (!config.enable_inelastic && config.nuclear_model == "none") {
        return;
    }

    // Explicit em_only policy on the mandatory current bundle path.
    if (!config.ct_schneider_physics_bundle_file.empty() &&
        config.secondary_out_of_scope_nuclear_policy != "em_only") {
        throw std::invalid_argument(
            "Schneider CT bundle mode requires secondary_out_of_scope_nuclear_policy: em_only "
            "(registry-unknown projectiles keep EM transport with nuclear reactions disabled; "
            "no other value or default exists)");
    }

    validate_schneider_physics_bundle(config);

    // Determine primary rate / cross section source
    std::filesystem::path primary_source = config.ct_schneider_primary_rate_file;
    if (primary_source.empty()) {
        throw std::runtime_error("Current CINEL03 requires an explicit primary rate binary; legacy fallback is retired");
    }

    if (!std::filesystem::exists(primary_source)) {
        throw std::runtime_error("Schneider CT startup failed: primary rate/XS table missing: " + primary_source.string());
    }

    // When secondary transport is active or in production mode, verify secondary rate and CINEL03 packages
    if (config.enable_secondary_transport || config.run_mode == RunMode::production) {
        std::filesystem::path c12_cinel = config.ct_schneider_primary_cinel03_file;
        std::filesystem::path sec_rate = config.ct_schneider_secondary_rate_file;
        std::filesystem::path sec_cinel = config.ct_schneider_secondary_cinel03_file;
        const std::vector<std::pair<std::string, std::filesystem::path>> required = {
            {"Schneider primary rate table", primary_source},
            {"Schneider primary CINEL03 package", c12_cinel},
            {"Schneider secondary rate table", sec_rate},
            {"Schneider secondary CINEL03 package", sec_cinel},
        };

        for (const auto& [name, path] : required) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("Schneider CT startup failed: " + name + " missing: " + path.string());
            }
            if (path.extension() == ".bin") {
                // Companion metadata existence check
                const auto meta_path = path.string() + ".metadata.json";
                const auto meta_path_alt = std::filesystem::path(path).replace_extension(".metadata.json");
                std::filesystem::path resolved_meta;
                if (std::filesystem::exists(meta_path)) {
                    resolved_meta = meta_path;
                } else if (std::filesystem::exists(meta_path_alt)) {
                    resolved_meta = meta_path_alt;
                } else {
                    throw std::runtime_error("Schneider CT startup failed: companion metadata missing for " + path.string());
                }


            }
        }


    }
}

PrimarySourceParameters load_primary_source_parameters(const std::filesystem::path& path) {
    auto values = read_key_values(path);
    merge_ion_physics_manifest(path, values);
    return parse_primary_source_parameters(values);
}

TransportConfig load_config(const std::filesystem::path& path) {
    auto values = read_key_values(path);
    const auto ion_physics_file = merge_ion_physics_manifest(path, values);
    values.try_emplace("em_model", "g4_material_joint_v1");
    // Broad-beam YAML defaults apply to every source ZA, including protons.
    // Insert before canonicalization so the effective defaults are recorded.
    // Explicit model, enable/disable and species selections remain authoritative.
    if (!parse_bool(values, "minibeam", false)) {
        values.try_emplace("multiple_scattering_model", "fermi_eyges");
        values.try_emplace("enable_multiple_scattering", "true");
        values.try_emplace("fermi_eyges_species", "all_charged");
    }
    // Expand the explicit fast profile before parsing paths, validation and
    // canonicalization. Unused packages are never opened/prefetched. Unknown
    // keys are still rejected; only named, supported inputs are removed.
    const auto profile_it=values.find("transport_profile");
    const std::string profile=profile_it==values.end()?"reference":profile_it->second;
    const bool tracking_declared=values.contains("broadbeam_electron_tracking");
    const bool tracking_requested=parse_bool(values,"broadbeam_electron_tracking",false);
    if(tracking_declared && profile!="broadbeam_fast")
        throw std::invalid_argument("broadbeam_electron_tracking is a broadbeam_fast setting");
    const bool independent_profile=tracking_requested && parse_bool(values,"enable_ct_grid",false);
    if(profile=="broadbeam_fast") {
        if(parse_bool(values,"minibeam",false))
            throw std::invalid_argument("broadbeam_fast requires minibeam: false");
        values.try_emplace("fermi_eyges_gaussian_only", "true");
        for(const char* key:{
            "urban_mcs_package_file","material_electron_charged_index_file", "material_photon_response_index_file", "material_electron_birth_file", "material_electron_joint_pilot_file",
            "neutron_package_file","neutron_recoil_stopping_manifest"}) {
            const std::string_view name(key);
            const bool shared=name.starts_with("material_electron_charged_") || name.starts_with("material_photon_response_") ||
                name.starts_with("material_electron_birth_") || name.starts_with("material_electron_joint_pilot_");
            if(!independent_profile || !shared)values.erase(key);
        }
        for(const auto& setting:std::initializer_list<std::pair<const char*,const char*>>{
            {"multiple_scattering_model","fermi_eyges"},{"enable_multiple_scattering","true"},
            {"fermi_eyges_species","all_charged"},{"fermi_eyges_parameter_set","species_water"},
            {"multiple_scattering_scale","1.0"},
            {"enable_ct_material_mcs","true"},
            {"water_primary_specialization","false"},
            {"urban_ct_density_scaling","false"},
            {"enable_neutron_transport","false"},{"neutron_water_recoil_urban","false"},
            {"material_electron_retain_bank","false"},
            {"material_electron_group_size","1"},
            {"material_electron_queue_chunk","65536"},

        }) {
            const std::string_view name(setting.first);
            if(independent_profile && (name=="material_electron_retain_bank" ||
               name=="material_electron_group_size" ||
               name=="material_electron_queue_chunk"))continue;
            values[setting.first]=setting.second;
        }
    }
    TransportConfig config;
    config.transport_profile=profile;
    config.broadbeam_electron_tracking=tracking_requested;
    config.broadbeam_independent_electrons=independent_profile;
    config.broadbeam_electron_density_threshold=parse_number(values,"broadbeam_electron_density_threshold",config.broadbeam_electron_density_threshold);
    config.broadbeam_electron_halo_mm=parse_number(values,"broadbeam_electron_halo_mm",config.broadbeam_electron_halo_mm);
    if(const auto it=values.find("broadbeam_electron_region_metric");it!=values.end())config.broadbeam_electron_region_metric=it->second;
    config.broadbeam_electron_region_diagnostics=parse_bool(values,"broadbeam_electron_region_diagnostics",false);
    config.broadbeam_electron_external_air_local=parse_bool(values,"broadbeam_electron_external_air_local",false);
    config.broadbeam_electron_external_air_margin_mm=parse_number(values,"broadbeam_electron_external_air_margin_mm",0.);
    config.broadbeam_electron_body_mask_file=parse_path(values,"broadbeam_electron_body_mask_file",{});
    if((config.broadbeam_electron_external_air_local || config.broadbeam_electron_region_diagnostics) && tracking_requested)
        config.broadbeam_electron_body_mask_file=resolve_input_path_from_config(config.broadbeam_electron_body_mask_file,path);
    else {config.broadbeam_electron_body_mask_file.clear();}
    config.broadbeam_electron_low_density_survival=parse_number(values,"broadbeam_electron_low_density_survival",1.);
    config.broadbeam_electron_tissue_survival=parse_number(values,"broadbeam_electron_tissue_survival",1.);
    config.broadbeam_electron_birth_order=parse_bool(values,"broadbeam_electron_birth_order",false);
    config.broadbeam_electron_expected_scoring=parse_bool(values,"broadbeam_electron_expected_scoring",false);
    config.broadbeam_electron_expected_min_energy_MeV=parse_number(values,"broadbeam_electron_expected_min_energy_MeV",0.);
    // Variance-reduction settings are inert when the master switch is off.
    if(!tracking_requested){config.broadbeam_electron_low_density_survival=1;config.broadbeam_electron_expected_scoring=false;
        config.broadbeam_electron_expected_min_energy_MeV=0;config.broadbeam_electron_tissue_survival=1;
        config.broadbeam_electron_birth_order=false;}
    // Eligible broad-beam full-physics runs benefit from the existing grouped
    // continuation scheduler. Preserve explicit opt-outs and EM-only inputs.
    if (profile == "broadbeam_fast" &&
        parse_bool(values, "enable_inelastic", config.enable_inelastic) &&
        parse_bool(values, "enable_secondary_transport", config.enable_secondary_transport)) {
        values.try_emplace("secondary_species_grouping", "true");
        values.try_emplace("secondary_step_chunking", "true");
    }
    if (const auto it=values.find("fermi_eyges_sampling"); it!=values.end())
        config.fermi_eyges_sampling=it->second;
    config.fermi_eyges_gaussian_only=parse_bool(values,"fermi_eyges_gaussian_only",false);

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
    config.quality_reject_voxel_over_total = parse_bool(
        values, "quality_reject_voxel_over_total",
        config.quality_reject_voxel_over_total);
    config.quality_reject_grid_closure = parse_bool(
        values, "quality_reject_grid_closure",
        config.quality_reject_grid_closure);
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
    const auto source = parse_primary_source_parameters(values);
    config.initial_energy_MeVu = source.initial_energy_MeVu;
    config.beam_energy_spread = source.beam_energy_spread;
    config.primary_atomic_number = source.ion.atomic_number;
    config.primary_mass_number = source.ion.mass_number;
    config.primary_rest_mass_MeV = source.configured_rest_mass_MeV;
    if (const auto it = values.find("primary_urban_reference_particle"); it != values.end())
        config.primary_urban_reference_particle = it->second;
    if (const auto it = values.find("primary_urban_ionisation_process"); it != values.end())
        config.primary_urban_ionisation_process = it->second;
    config.minibeam_copper_production_cut_mm = parse_number(values,
        "minibeam_copper_production_cut_mm", config.minibeam_copper_production_cut_mm);
    config.minibeam_water_production_cut_mm = parse_number(values,
        "minibeam_water_production_cut_mm", config.minibeam_water_production_cut_mm);
    config.phantom_length_mm = parse_number(values, "phantom_length_mm", config.phantom_length_mm);
    config.depth_bin_width_mm = parse_number(values, "depth_bin_width_mm", config.depth_bin_width_mm);
    if (const auto it = values.find("em_model"); it != values.end()) config.em_model=it->second;
    config.em_primary_step_scale=parse_number(values,"em_primary_step_scale",config.em_primary_step_scale);
    config.em_secondary_step_scale=parse_number(values,"em_secondary_step_scale",config.em_secondary_step_scale);
    config.broadbeam_maximum_mean_loss_fraction=parse_number(values,"broadbeam_maximum_mean_loss_fraction",config.broadbeam_maximum_mean_loss_fraction);
    config.em_package_file=parse_path(values,"em_package_file",config.em_package_file);
    if (!config.em_package_file.empty())
        config.em_package_file=resolve_input_path_from_config(config.em_package_file,path);
    config.em_delta_moments_file=parse_path(values,"em_delta_moments_file",config.em_delta_moments_file);
    if(!config.em_delta_moments_file.empty())
        config.em_delta_moments_file=resolve_input_path_from_config(config.em_delta_moments_file,path);
    config.maximum_step_mm = parse_number(values, "maximum_step_mm", config.maximum_step_mm);
    config.maximum_relative_energy_loss =
        parse_number(values, "maximum_relative_energy_loss", config.maximum_relative_energy_loss);
    config.maximum_primary_steps =
        parse_number(values, "maximum_primary_steps", config.maximum_primary_steps);
    config.energy_cutoff_MeV = parse_number(values, "energy_cutoff_MeV", config.energy_cutoff_MeV);
    config.secondary_local_deposit_cutoff_MeV = parse_number(
        values, "secondary_local_deposit_cutoff_MeV",
        config.secondary_local_deposit_cutoff_MeV);
    config.secondary_heavy_local_deposit_z_min = parse_number(
        values, "secondary_heavy_local_deposit_z_min",
        config.secondary_heavy_local_deposit_z_min);
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
    if(!config.ct_schneider_file.empty())
        config.ct_schneider_file=resolve_input_path_from_config(config.ct_schneider_file,path);
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
    if (!config.ct_schneider_cross_section_file.empty()) {
        config.ct_schneider_cross_section_file = resolve_input_path_from_config(
            config.ct_schneider_cross_section_file, path);
    }
    config.water_cinel_package_file = parse_path(
        values, "water_cinel_package_file", config.water_cinel_package_file);
    config.unified_water_nuclear_transport = parse_bool(values,
        "unified_water_nuclear_transport", config.unified_water_nuclear_transport);
    config.unified_water_material_file = parse_path(values,
        "unified_water_material_file", config.unified_water_material_file);
    if (!config.unified_water_material_file.empty())
        config.unified_water_material_file = resolve_input_path_from_config(config.unified_water_material_file, path);
    if (config.water_cinel_package_file.empty()) {
        config.water_cinel_package_file = config.primary_inelastic_package_v2_file;
    }
    config.water_reaction_rate_file = parse_path(
        values, "water_reaction_rate_file", config.water_reaction_rate_file);
    if (config.water_reaction_rate_file.empty()) {
        config.water_reaction_rate_file = config.primary_inelastic_rate_v2_file;
    }
    config.ct_schneider_primary_cinel03_file = parse_path(
        values, "ct_schneider_primary_cinel03_file", config.ct_schneider_primary_cinel03_file);
    if (!config.ct_schneider_primary_cinel03_file.empty()) {
        config.ct_schneider_primary_cinel03_file = resolve_input_path_from_config(
            config.ct_schneider_primary_cinel03_file, path);
    }
    config.ct_schneider_secondary_cinel03_file = parse_path(
        values, "ct_schneider_secondary_cinel03_file", config.ct_schneider_secondary_cinel03_file);
    if (!config.ct_schneider_secondary_cinel03_file.empty()) {
        config.ct_schneider_secondary_cinel03_file = resolve_input_path_from_config(
            config.ct_schneider_secondary_cinel03_file, path);
    }
    config.ct_schneider_primary_rate_file = parse_path(
        values, "ct_schneider_primary_rate_file", config.ct_schneider_primary_rate_file);
    if (!config.ct_schneider_primary_rate_file.empty()) {
        config.ct_schneider_primary_rate_file = resolve_input_path_from_config(
            config.ct_schneider_primary_rate_file, path);
    }
    config.ct_schneider_secondary_rate_file = parse_path(
        values, "ct_schneider_secondary_rate_file", config.ct_schneider_secondary_rate_file);
    config.ct_elastic_section_rate_file = parse_path(
        values, "ct_elastic_section_rate_file", config.ct_elastic_section_rate_file);
    config.ct_elastic_diagnostic = parse_bool(values,
        "ct_elastic_diagnostic", config.ct_elastic_diagnostic);
    if (!config.ct_elastic_section_rate_file.empty()) {
        if (!config.ct_elastic_diagnostic)
            throw std::invalid_argument("Elastic rate file requires ct_elastic_diagnostic=true");
        config.ct_elastic_section_rate_file = resolve_input_path_from_config(
            config.ct_elastic_section_rate_file, path);
    }
    config.ct_elastic_all_targets = parse_bool(values,
        "ct_elastic_all_targets", config.ct_elastic_all_targets);
    if (config.ct_elastic_diagnostic &&
        (config.run_mode != RunMode::smoke || !config.enable_ct_grid))
        throw std::invalid_argument("Full-section elastic requires smoke Schneider CT");
    if (config.ct_elastic_diagnostic && config.ct_elastic_section_rate_file.empty())
        throw std::invalid_argument("ct_elastic_diagnostic=true requires ct_elastic_section_rate_file");
    config.all_ion_elastic_file = parse_path(values,"all_ion_elastic_file",config.all_ion_elastic_file);
    config.elastic_recoil_stopping_file=parse_path(values,"elastic_recoil_stopping_file",config.elastic_recoil_stopping_file);
    if(!config.all_ion_elastic_file.empty()) {
        if(config.elastic_recoil_stopping_file.empty())
            throw std::invalid_argument("All-ion elastic requires recoil stopping data");
        config.elastic_recoil_stopping_file=resolve_input_path_from_config(config.elastic_recoil_stopping_file,path);
        config.all_ion_elastic_file=resolve_input_path_from_config(config.all_ion_elastic_file,path);
        if(config.ct_elastic_diagnostic||config.enable_nuclear_elastic)
            throw std::invalid_argument("All-ion elastic requires no legacy elastic mode");
        if(config.run_mode==RunMode::production)
            throw std::invalid_argument("All-ion elastic remains a validation candidate: use smoke/research");
    }
    if (!config.ct_schneider_secondary_rate_file.empty()) {
        config.ct_schneider_secondary_rate_file = resolve_input_path_from_config(
            config.ct_schneider_secondary_rate_file, path);
    }
    config.ct_schneider_physics_bundle_file = parse_path(
        values, "ct_schneider_physics_bundle_file", config.ct_schneider_physics_bundle_file);
    if (!config.ct_schneider_physics_bundle_file.empty()) {
        config.ct_schneider_physics_bundle_file = resolve_input_path_from_config(
            config.ct_schneider_physics_bundle_file, path);
    }
    config.ct_schneider_radiation_length_file = parse_path(
        values, "ct_schneider_radiation_length_file", config.ct_schneider_radiation_length_file);
    if (!config.ct_schneider_radiation_length_file.empty()) {
        config.ct_schneider_radiation_length_file = resolve_input_path_from_config(
            config.ct_schneider_radiation_length_file, path);
    }
    config.material_electron_charged_index_file=parse_path(values,"material_electron_charged_index_file",config.material_electron_charged_index_file);
    if(!config.material_electron_charged_index_file.empty())config.material_electron_charged_index_file=resolve_input_path_from_config(config.material_electron_charged_index_file,path);
    config.material_electron_birth_file=parse_path(values,"material_electron_birth_file",config.material_electron_birth_file);
    config.material_photon_response_index_file=parse_path(values,"material_photon_response_index_file",config.material_photon_response_index_file);
    if(!config.material_photon_response_index_file.empty())config.material_photon_response_index_file=resolve_input_path_from_config(config.material_photon_response_index_file,path);
    if(!config.material_electron_birth_file.empty())config.material_electron_birth_file=resolve_input_path_from_config(config.material_electron_birth_file,path);
    config.material_electron_joint_pilot_file=parse_path(values,"material_electron_joint_pilot_file",config.material_electron_joint_pilot_file);
    if(!config.material_electron_joint_pilot_file.empty())config.material_electron_joint_pilot_file=resolve_input_path_from_config(config.material_electron_joint_pilot_file,path);
    config.material_electron_retain_bank=parse_bool(values,"material_electron_retain_bank",config.material_electron_retain_bank);
    if(const auto it=values.find("material_electron_queue_budget_MiB");it!=values.end()) {
        std::size_t used=0;const auto value=std::stoull(it->second,&used);
        if(used!=it->second.size() || it->second.empty() || it->second[0]=='-' || value>65536)
            throw std::invalid_argument("Invalid material_electron_queue_budget_MiB");
        config.material_electron_queue_budget_MiB=static_cast<std::uint32_t>(value);
    }
    if(const auto it=values.find("material_electron_queue_chunk");it!=values.end()) {
        std::size_t used=0;const auto value=std::stoull(it->second,&used);
        if(used!=it->second.size() || it->second.empty() || it->second[0]=='-' || value>1048576)
            throw std::invalid_argument("Invalid material_electron_queue_chunk");
        config.material_electron_queue_chunk=static_cast<std::uint32_t>(value);
    }
    if(const auto it=values.find("material_electron_queue_lanes");it!=values.end()) {
        std::size_t used=0;const auto value=std::stoull(it->second,&used);
        if(used!=it->second.size() || it->second.empty() || it->second[0]=='-' || value>65536)
            throw std::invalid_argument("Invalid material_electron_queue_lanes");
        config.material_electron_queue_lanes=static_cast<std::uint32_t>(value);
    }
    if(const auto it=values.find("material_electron_group_size");it!=values.end()) {
        std::size_t used=0;const auto value=std::stoull(it->second,&used);
        if(used!=it->second.size() || it->second.empty() || it->second[0]=='-' || value>65536)
            throw std::invalid_argument("Invalid material_electron_group_size");
        config.material_electron_group_size=static_cast<std::uint32_t>(value);
    }
    config.electron_joint_diagnostics = parse_bool(values,"electron_joint_diagnostics",config.electron_joint_diagnostics);
    config.ct_cinel02_rate_file = parse_path(
        values, "ct_cinel02_rate_file", config.ct_cinel02_rate_file);
    if (const auto it = values.find("ct_validation_mode"); it != values.end()) {
        config.ct_validation_mode = it->second;
    }
    if (const auto it = values.find("secondary_out_of_scope_nuclear_policy");
        it != values.end()) {
        config.secondary_out_of_scope_nuclear_policy = it->second;
    }
    if (values.find("ct_grid_file") != values.end() &&
        values.find("enable_ct_grid") == values.end()) {
        config.enable_ct_grid = true;
    }
    config.resolve_material_physics_mode();
    config.scorer_area_mm2 = parse_number(values, "scorer_area_mm2", config.scorer_area_mm2);
    config.dose_output_scale =
        parse_number(values, "dose_output_scale", config.dose_output_scale);
    config.enable_voxel_scoring =
        parse_bool(values, "enable_voxel_scoring", config.enable_voxel_scoring);
    config.enable_charged_origin_voxel_scoring = parse_bool(
        values, "enable_charged_origin_voxel_scoring",
        config.enable_charged_origin_voxel_scoring);
    config.enable_minibeam_energy_band_roi_scoring = parse_bool(
        values, "enable_minibeam_energy_band_roi_scoring",
        config.enable_minibeam_energy_band_roi_scoring);
    config.enable_minibeam_primary_roi_scoring = parse_bool(
        values, "enable_minibeam_primary_roi_scoring",
        config.enable_minibeam_primary_roi_scoring);
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
    if (const auto it = values.find("nuclear_model"); it != values.end()) {
        config.nuclear_model = it->second;
        std::transform(config.nuclear_model.begin(), config.nuclear_model.end(),
                       config.nuclear_model.begin(), [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
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

        if (!config.ct_schneider_cross_section_file.empty()) {
            config.ct_schneider_cross_section_file = resolve_input_path_from_config(
                config.ct_schneider_cross_section_file, path);
        }

        const bool is_schneider_mode = (grid.file_version >= CtGrid::version_v2 && grid.mass_sp_za_rel.size() == 25) ||
                                       !config.ct_schneider_file.empty() ||
                                       !config.ct_schneider_cross_section_file.empty();

        if (is_schneider_mode) {
            std::cout << "[material-indexing] mode: schneider-25\n";
            if (config.nuclear_model != "none") {
                // v3 bundle: masked rate-binary hazard replaces the CSV XS
                // table (key intentionally empty, mixing refused at startup);
                // nothing to preload here.
                const bool v3_bundle = !config.ct_schneider_physics_bundle_file.empty();
                if (config.ct_schneider_cross_section_file.empty() && !v3_bundle) {
                    throw std::runtime_error(
                        "CT Schneider-25 mode with active nuclear model '" + config.nuclear_model +
                        "' requires ct_schneider_cross_section_file; fallback to water or four-class XS is forbidden.");
                }
                if (!config.ct_schneider_cross_section_file.empty()) {
                    if (!std::filesystem::exists(config.ct_schneider_cross_section_file)) {
                        throw std::runtime_error(
                            "ct_schneider_cross_section_file does not exist: " +
                            config.ct_schneider_cross_section_file.string());
                    }
                    const auto xs_tables = CrossSectionTable::from_schneider_csv(
                        config.ct_schneider_cross_section_file);
                    if (xs_tables.size() != SchneiderResampledCrossSectionGrid::kExpectedSections) {
                        throw std::runtime_error(
                            "ct_schneider_cross_section_file must contain exactly 25 sections (got " +
                            std::to_string(xs_tables.size()) + ")");
                    }
                }
            }
        } else {
            std::cout << "[material-indexing] mode: legacy-four-class\n";
        }
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
    config.enable_primary_loss_query_audit = parse_bool(
        values, "enable_primary_loss_query_audit", config.enable_primary_loss_query_audit);
    config.enable_secondary_energy_straggling = parse_bool(
        values, "enable_secondary_energy_straggling",
        config.enable_secondary_energy_straggling);
    config.secondary_species_grouping = parse_bool(
        values, "secondary_species_grouping", config.secondary_species_grouping);
    config.water_primary_specialization = parse_bool(
        values, "water_primary_specialization", config.water_primary_specialization);
    config.minibeam_primary_specialization = parse_bool(
        values, "minibeam_primary_specialization", config.minibeam_primary_specialization);
    {
        const auto it = values.find("minibeam_primary_pipeline");
        if (it != values.end()) config.minibeam_primary_pipeline = it->second;
        if (config.minibeam_primary_pipeline != "combined" &&
            config.minibeam_primary_pipeline != "split" &&
            config.minibeam_primary_pipeline != "compact")
            throw std::invalid_argument(
                "minibeam_primary_pipeline must be combined, split or compact");
    }
    config.secondary_step_chunking = parse_bool(
        values, "secondary_step_chunking", config.secondary_step_chunking);
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
    config.cinel02_strict_match = parse_bool(
        values, "cinel02_strict_match", config.cinel02_strict_match);
    config.enable_nuclear_elastic =
        parse_bool(values, "enable_nuclear_elastic", config.enable_nuclear_elastic);
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
        if (config.multiple_scattering_model == "fe") {
            config.multiple_scattering_model = "fermi_eyges";
        }
    }
    if(const auto it=values.find("urban_mcs_package_file");it!=values.end())
        config.urban_mcs_package_file=it->second;
    config.urban_water_half_width_mm=parse_number(values,"urban_water_half_width_mm",config.urban_water_half_width_mm);
    config.urban_ct_density_scaling=parse_bool(values,"urban_ct_density_scaling",config.urban_ct_density_scaling);
    if (const auto it = values.find("fermi_eyges_species");
        it != values.end()) {
        config.fermi_eyges_species = it->second;
        std::transform(config.fermi_eyges_species.begin(),
                       config.fermi_eyges_species.end(),
                       config.fermi_eyges_species.begin(), [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (config.fermi_eyges_species == "all") {
            config.fermi_eyges_species = "all_charged";
        }
    }
    if (const auto it = values.find("fermi_eyges_parameter_set");
        it != values.end()) {
        config.fermi_eyges_parameter_set = it->second;
        std::transform(config.fermi_eyges_parameter_set.begin(),
                       config.fermi_eyges_parameter_set.end(),
                       config.fermi_eyges_parameter_set.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
    config.fermi_eyges_max_segment_mm = parse_number(
        values, "fermi_eyges_max_segment_mm",
        config.fermi_eyges_max_segment_mm);
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
    config.neutron_birth_capacity = parse_number(
        values, "neutron_birth_capacity", config.neutron_birth_capacity);
    config.neutron_birth_output_file = parse_path(
        values, "neutron_birth_output_file", config.neutron_birth_output_file);
    config.enable_neutron_transport = parse_bool(
        values, "enable_neutron_transport", config.enable_neutron_transport);
    config.neutron_world_half_width_mm=parse_number(values,"neutron_world_half_width_mm",config.neutron_world_half_width_mm);
    config.neutron_max_collisions=parse_number(values,"neutron_max_collisions",config.neutron_max_collisions);
    config.neutron_proton_birth_output_file=parse_path(values,"neutron_proton_birth_output_file",config.neutron_proton_birth_output_file);
    config.neutron_proton_birth_capacity=parse_number(values,"neutron_proton_birth_capacity",config.neutron_proton_birth_capacity);
    config.neutron_exact_plane_boundaries=parse_bool(values,"neutron_exact_plane_boundaries",config.neutron_exact_plane_boundaries);
    config.neutron_water_recoil_urban=parse_bool(values,"neutron_water_recoil_urban",config.neutron_water_recoil_urban);
    config.neutron_elastic_two_body=parse_bool(values,"neutron_elastic_two_body",config.neutron_elastic_two_body);
    config.neutron_recoil_stopping_manifest=parse_path(values,"neutron_recoil_stopping_manifest",config.neutron_recoil_stopping_manifest);
    config.neutron_package_file = parse_path(
        values, "neutron_package_file", config.neutron_package_file);
    config.max_device_memory_fraction = parse_number(
        values, "max_device_memory_fraction", config.max_device_memory_fraction);
    config.device_memory_budget_gib = parse_number(
        values, "device_memory_budget_gib", config.device_memory_budget_gib);
    config.secondary_queue_capacity = parse_number(
        values, "secondary_queue_capacity", config.secondary_queue_capacity);
    config.auto_device_tuning = parse_bool(
        values, "auto_device_tuning", config.auto_device_tuning);
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
    config.minibeam_collimator_center_to_isocenter_mm = parse_number(
        values, "minibeam_collimator_center_to_isocenter_mm",
        config.minibeam_collimator_center_to_isocenter_mm);
    config.minibeam_collimator_width_mm = parse_number(
        values, "minibeam_collimator_width_mm",
        config.minibeam_collimator_width_mm);
    config.minibeam_collimator_length_mm = parse_number(
        values, "minibeam_collimator_length_mm",
        config.minibeam_collimator_length_mm);
    config.minibeam_collimator_thickness_mm = parse_number(
        values, "minibeam_collimator_thickness_mm",
        config.minibeam_collimator_thickness_mm);
    config.minibeam_slit_length_mm = parse_number(
        values, "minibeam_slit_length_mm", config.minibeam_slit_length_mm);
    config.minibeam_slit_thickness_mm = parse_number(
        values, "minibeam_slit_thickness_mm",
        config.minibeam_slit_thickness_mm);
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
    if (auto it = values.find("minibeam_air_mcs_model"); it != values.end())
        config.minibeam_air_mcs_model = it->second;
    config.minibeam_air_slit_max_step_mm = parse_number(values, "minibeam_air_slit_max_step_mm", config.minibeam_air_slit_max_step_mm);
    if (!(config.minibeam_air_slit_max_step_mm > 0.0) || !std::isfinite(config.minibeam_air_slit_max_step_mm))
        throw std::invalid_argument("Air slit max step must be finite and positive");
    config.minibeam_air_urban_package_file = parse_path(values,
        "minibeam_air_urban_package_file", config.minibeam_air_urban_package_file);
    if (config.minibeam_air_mcs_model != "none" && config.minibeam_air_mcs_model != "urban_v2")
        throw std::invalid_argument("minibeam_air_mcs_model must be none or urban_v2");
    config.minibeam_air_ionization_rate_file = parse_path(values,
        "minibeam_air_ionization_rate_file", config.minibeam_air_ionization_rate_file);
    if (!config.minibeam_air_ionization_rate_file.empty() &&
        (!config.enable_minibeam || config.minibeam_transport_mode!="copper_em" ||
         config.minibeam_air_mcs_model!="urban_v2" ||
         config.run_mode==RunMode::production))
        throw std::invalid_argument("Air ionization clock requires research/smoke copper_em, air urban_v2 and a pinned rate table");
    config.minibeam_copper_loss_range_file = parse_path(
        values, "minibeam_copper_loss_range_file",
        config.minibeam_copper_loss_range_file);
    config.minibeam_copper_density_g_per_cm3 = parse_number(
        values, "minibeam_copper_density_g_per_cm3",
        config.minibeam_copper_density_g_per_cm3);
    config.minibeam_copper_radiation_length_g_per_cm2 = parse_number(
        values, "minibeam_copper_radiation_length_g_per_cm2",
        config.minibeam_copper_radiation_length_g_per_cm2);
    config.minibeam_copper_max_step_mm = parse_number(
        values, "minibeam_copper_max_step_mm",
        config.minibeam_copper_max_step_mm);
    config.minibeam_copper_exact_material_boundaries = parse_bool(
        values, "minibeam_copper_exact_material_boundaries",
        config.minibeam_copper_exact_material_boundaries);
    config.minibeam_copper_enable_mcs = parse_bool(
        values, "minibeam_copper_enable_mcs",
        config.minibeam_copper_enable_mcs);
    if (const auto iterator = values.find("minibeam_copper_mcs_model");
        iterator != values.end()) {
        config.minibeam_copper_mcs_model = iterator->second;
        std::transform(config.minibeam_copper_mcs_model.begin(),
                       config.minibeam_copper_mcs_model.end(),
                       config.minibeam_copper_mcs_model.begin(),
                       [](unsigned char value) {
                           return static_cast<char>(std::tolower(value));
                       });
    }
    config.minibeam_copper_enable_elastic = parse_bool(
        values, "minibeam_copper_enable_elastic",
        config.minibeam_copper_enable_elastic);
    config.minibeam_copper_enable_energy_straggling = parse_bool(
        values, "minibeam_copper_enable_energy_straggling",
        config.minibeam_copper_enable_energy_straggling);
    config.minibeam_copper_primary_native_mean_loss = parse_bool(
        values, "minibeam_copper_primary_native_mean_loss",
        config.minibeam_copper_primary_native_mean_loss);
    if (const auto it = values.find("minibeam_copper_proton_straggling_model"); it != values.end())
        config.minibeam_copper_proton_straggling_model = it->second;
    config.minibeam_copper_straggling_scale = parse_number(
        values, "minibeam_copper_straggling_scale",
        config.minibeam_copper_straggling_scale);
    config.minibeam_copper_fragment_enable_energy_straggling = parse_bool(
        values, "minibeam_copper_fragment_enable_energy_straggling",
        config.minibeam_copper_fragment_enable_energy_straggling);
    config.minibeam_copper_fragment_straggling_scale = parse_number(
        values, "minibeam_copper_fragment_straggling_scale",
        config.minibeam_copper_fragment_straggling_scale);
    config.minibeam_copper_mcs_scale = parse_number(
        values, "minibeam_copper_mcs_scale",
        config.minibeam_copper_mcs_scale);
    config.minibeam_copper_fragment_mcs_scale = parse_number(
        values, "minibeam_copper_fragment_mcs_scale",
        config.minibeam_copper_fragment_mcs_scale);
    config.minibeam_copper_fragment_cascade_generations = parse_number(
        values, "minibeam_copper_fragment_cascade_generations",
        config.minibeam_copper_fragment_cascade_generations);
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
    config.minibeam_copper_enable_nuclear_attenuation = parse_bool(
        values, "minibeam_copper_enable_nuclear_attenuation",
        config.minibeam_copper_enable_nuclear_attenuation);
    config.minibeam_copper_cross_section_file = parse_path(
        values, "minibeam_copper_cross_section_file",
        config.minibeam_copper_cross_section_file);
    config.minibeam_copper_inclxx_file = parse_path(
        values, "minibeam_copper_inclxx_file",
        config.minibeam_copper_inclxx_file);
    config.minibeam_copper_elastic_file = parse_path(
        values, "minibeam_copper_elastic_file",
        config.minibeam_copper_elastic_file);
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
    config.spots_enable_upstream_air_mcs = parse_bool(
        values, "spots_enable_upstream_air_mcs", config.spots_enable_upstream_air_mcs);
    config.ct_secondary_exact_faces_diagnostic = parse_bool(values,
        "ct_secondary_exact_faces_diagnostic", config.ct_secondary_exact_faces_diagnostic);
    config.ct_secondary_exact_faces = parse_bool(values,
        "ct_secondary_exact_faces", config.ct_secondary_exact_faces);
    if (values.find("ct_secondary_exact_faces_diagnostic") != values.end()) {
        if (values.find("ct_secondary_exact_faces") != values.end() &&
            config.ct_secondary_exact_faces != config.ct_secondary_exact_faces_diagnostic)
            throw std::invalid_argument("Conflicting exact-faces formal and diagnostic keys");
        config.ct_secondary_exact_faces = config.ct_secondary_exact_faces_diagnostic;
    }
    config.ct_secondary_mcs_off_diagnostic = parse_bool(values,
        "ct_secondary_mcs_off_diagnostic", config.ct_secondary_mcs_off_diagnostic);
    config.spots_upstream_air_mcs_file = parse_path(values, "spots_upstream_air_mcs_file", config.spots_upstream_air_mcs_file);
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
    config.particle_stopping_power_file = parse_path(
        values, "particle_stopping_power_file",
        config.particle_stopping_power_file);
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
    config.minibeam_phase_space_output_file = parse_path(
        values, "minibeam_phase_space_output_file",
        config.minibeam_phase_space_output_file);
    config.minibeam_phase_space_at_copper_exit = parse_bool(
        values, "minibeam_phase_space_at_copper_exit",
        config.minibeam_phase_space_at_copper_exit);
    if (config.minibeam_phase_space_at_copper_exit &&
        (!config.enable_minibeam || config.minibeam_transport_mode != "copper_em" ||
         config.minibeam_air_mcs_model != "urban_v2" ||
         config.minibeam_phase_space_output_file.empty() || config.run_mode == RunMode::production)) {
        throw std::invalid_argument("Copper-exit phase scoring requires research/smoke copper_em, air urban_v2 and a phase output file");
    }
    config.minibeam_water_primary_plane_output_file = parse_path(
        values, "minibeam_water_primary_plane_output_file",
        config.minibeam_water_primary_plane_output_file);
    {
        const auto it = values.find(
            "minibeam_water_primary_plane_depths_mm");
        if (it != values.end()) {
            config.minibeam_water_primary_plane_depths_mm = parse_double_list(
                it->second, "minibeam_water_primary_plane_depths_mm");
        }
    }
    config.minibeam_fragment_phase_space_output_file = parse_path(
        values, "minibeam_fragment_phase_space_output_file",
        config.minibeam_fragment_phase_space_output_file);
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
        const auto it = values.find("primary_voxel_fluence_mhd_output_file");
        if (it != values.end()) {
            config.primary_voxel_fluence_mhd_output_file =
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

    if(!config.all_ion_elastic_file.empty()) {
        if(!config.enable_inelastic||!config.enable_secondary_transport||config.enable_let_scoring||
           (!config.enable_ct_grid&&!config.unified_water_nuclear_transport))
            throw std::invalid_argument("All-ion elastic requires Schneider CT/unified-water nuclear transport, secondary transport and LET off");
        if(config.enable_ct_grid && !config.ct_secondary_exact_faces)
            throw std::invalid_argument("All-ion elastic requires secondary exact faces");
    }
    reject_unknown_config_keys(values, path);
    config.canonical_config_text = canonicalize_config(
        values, config.config_schema_version);
    config.validate();

    return config;
}

}  // namespace carbon
