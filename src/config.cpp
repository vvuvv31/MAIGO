#include "carbon/transport_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace carbon {
namespace {

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::unordered_map<std::string, std::string> read_key_values(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open configuration file: " + path.string());
    }

    std::unordered_map<std::string, std::string> values;
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
        if (key.empty() || value.empty()) {
            throw std::runtime_error("Empty configuration key or value at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        values[key] = value;
    }
    return values;
}

template <typename Number>
Number parse_number(const std::unordered_map<std::string, std::string>& values,
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

std::filesystem::path parse_path(const std::unordered_map<std::string, std::string>& values,
                                 const std::string& key,
                                 const std::filesystem::path& fallback) {
    const auto iterator = values.find(key);
    return iterator == values.end() ? fallback : std::filesystem::path(iterator->second);
}

bool parse_bool(const std::unordered_map<std::string, std::string>& values,
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

}  // namespace

std::size_t TransportConfig::number_of_bins() const {
    return static_cast<std::size_t>(std::ceil(phantom_length_mm / depth_bin_width_mm));
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

void TransportConfig::validate() const {
    if (number_of_histories == 0) {
        throw std::invalid_argument("number_of_histories must be greater than zero");
    }
    if (mass_number <= 0 || initial_energy_MeVu <= 0.0) {
        throw std::invalid_argument("mass_number and initial_energy_MeVu must be positive");
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
    if (straggling_scale < 0.0) {
        throw std::invalid_argument("straggling_scale must be nonnegative");
    }
    if (enable_voxel_scoring &&
        (voxel_bins_x == 0 || voxel_bins_y == 0 || voxel_size_x_mm <= 0.0 ||
         voxel_size_y_mm <= 0.0)) {
        throw std::invalid_argument(
            "Enabled voxel scoring requires positive x/y bin counts and voxel sizes");
    }
    if (enable_voxel_scoring) {
        static_cast<void>(number_of_voxels());
    }
    if (enable_secondary_transport && !enable_secondary_generation) {
        throw std::invalid_argument(
            "enable_secondary_transport requires enable_secondary_generation=true");
    }
    if (enable_fragment_cascade &&
        (!enable_secondary_transport || maximum_cascade_generations == 0)) {
        throw std::invalid_argument(
            "enable_fragment_cascade requires secondary transport and at least one generation");
    }
}

TransportConfig load_config(const std::filesystem::path& path) {
    const auto values = read_key_values(path);
    TransportConfig config;
    config.number_of_histories = parse_number(values, "number_of_histories", config.number_of_histories);
    config.initial_energy_MeVu = parse_number(values, "initial_energy_MeVu", config.initial_energy_MeVu);
    config.mass_number = parse_number(values, "mass_number", config.mass_number);
    config.phantom_length_mm = parse_number(values, "phantom_length_mm", config.phantom_length_mm);
    config.depth_bin_width_mm = parse_number(values, "depth_bin_width_mm", config.depth_bin_width_mm);
    config.maximum_step_mm = parse_number(values, "maximum_step_mm", config.maximum_step_mm);
    config.maximum_relative_energy_loss =
        parse_number(values, "maximum_relative_energy_loss", config.maximum_relative_energy_loss);
    config.energy_cutoff_MeV = parse_number(values, "energy_cutoff_MeV", config.energy_cutoff_MeV);
    config.water_density_g_per_cm3 =
        parse_number(values, "water_density_g_per_cm3", config.water_density_g_per_cm3);
    config.scorer_area_mm2 = parse_number(values, "scorer_area_mm2", config.scorer_area_mm2);
    config.enable_voxel_scoring =
        parse_bool(values, "enable_voxel_scoring", config.enable_voxel_scoring);
    config.voxel_bins_x = parse_number(values, "voxel_bins_x", config.voxel_bins_x);
    config.voxel_bins_y = parse_number(values, "voxel_bins_y", config.voxel_bins_y);
    config.voxel_size_x_mm =
        parse_number(values, "voxel_size_x_mm", config.voxel_size_x_mm);
    config.voxel_size_y_mm =
        parse_number(values, "voxel_size_y_mm", config.voxel_size_y_mm);
    config.enable_energy_straggling =
        parse_bool(values, "enable_energy_straggling", config.enable_energy_straggling);
    config.straggling_scale = parse_number(values, "straggling_scale", config.straggling_scale);
    config.enable_multiple_scattering =
        parse_bool(values, "enable_multiple_scattering", config.enable_multiple_scattering);
    config.enable_primary_attenuation =
        parse_bool(values, "enable_primary_attenuation", config.enable_primary_attenuation);
    config.enable_secondary_generation =
        parse_bool(values, "enable_secondary_generation", config.enable_secondary_generation);
    config.enable_secondary_transport =
        parse_bool(values, "enable_secondary_transport", config.enable_secondary_transport);
    config.enable_fragment_cascade =
        parse_bool(values, "enable_fragment_cascade", config.enable_fragment_cascade);
    config.maximum_cascade_generations = parse_number(
        values, "maximum_cascade_generations", config.maximum_cascade_generations);
    config.secondary_queue_capacity =
        parse_number(values, "secondary_queue_capacity", config.secondary_queue_capacity);
    config.random_seed = parse_number(values, "random_seed", config.random_seed);
    config.stopping_power_file = parse_path(values, "stopping_power_file", config.stopping_power_file);
    config.nuclear_cross_section_file =
        parse_path(values, "nuclear_cross_section_file", config.nuclear_cross_section_file);
    config.reaction_package_file =
        parse_path(values, "reaction_package_file", config.reaction_package_file);
    config.cascade_package_file =
        parse_path(values, "cascade_package_file", config.cascade_package_file);
    config.output_file = parse_path(values, "output_file", config.output_file);
    config.fragment_species_output_file =
        parse_path(values, "fragment_species_output_file", config.fragment_species_output_file);
    config.voxel_dose_output_file =
        parse_path(values, "voxel_dose_output_file", config.voxel_dose_output_file);
    const auto device = values.find("device");
    if (device != values.end()) {
        config.device = device->second;
    }
    config.validate();
    return config;
}

}  // namespace carbon
