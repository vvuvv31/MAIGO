#include "carbon/transport_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

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
        // Empty value is allowed (disables optional outputs such as sparse voxel CSV).
        if (key.empty()) {
            throw std::runtime_error("Empty configuration key at " + path.string() + ":" +
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
    if (secondary_local_deposit_cutoff_MeV < 0.0) {
        throw std::invalid_argument(
            "secondary_local_deposit_cutoff_MeV must be non-negative");
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
    } else if (!slab_layers.empty()) {
        throw std::invalid_argument(
            "slab_layers is set but enable_layered_phantom=false");
    }
    if (enable_hetero_insert) {
        validate_hetero_insert(hetero_insert, phantom_length_mm);
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
    if (straggling_scale < 0.0) {
        throw std::invalid_argument("straggling_scale must be nonnegative");
    }
    if (ct_stopping_power_scale <= 0.0 || ct_stopping_power_scale > 2.0) {
        throw std::invalid_argument(
            "ct_stopping_power_scale must be in (0, 2]");
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
    if (enable_charged_origin_voxel_scoring && !enable_voxel_scoring) {
        throw std::invalid_argument(
            "enable_charged_origin_voxel_scoring requires enable_voxel_scoring=true");
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
    if (!fragment_birth_spectrum_output_file.empty() &&
        !enable_secondary_generation) {
        throw std::invalid_argument(
            "fragment_birth_spectrum_output_file requires "
            "enable_secondary_generation=true");
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
    if (enable_secondary_transport && !enable_secondary_generation) {
        throw std::invalid_argument(
            "enable_secondary_transport requires enable_secondary_generation=true");
    }
    if (enable_secondary_energy_sorting && !enable_secondary_transport) {
        throw std::invalid_argument(
            "enable_secondary_energy_sorting requires enable_secondary_transport=true");
    }
    if (enable_charged_origin_voxel_scoring && !enable_secondary_transport) {
        throw std::invalid_argument(
            "charged-origin voxel scoring requires secondary transport");
    }
    if (enable_fragment_cascade &&
        (!enable_secondary_transport || maximum_cascade_generations == 0)) {
        throw std::invalid_argument(
            "enable_fragment_cascade requires secondary transport and at least one generation");
    }
    if (!enable_fragment_species_scoring && enable_secondary_transport &&
        (!output_file.empty() || !dose_output_file.empty() ||
         !fragment_species_output_file.empty() ||
         !fragment_species_dose_output_file.empty())) {
        throw std::invalid_argument(
            "fragment species scoring disabled: depth-dose and fragment output "
            "paths must be empty (voxel dose remains available)");
    }
    if (!enable_fragment_species_scoring && neutral_local_kerma_fraction > 0.0) {
        throw std::invalid_argument(
            "neutral_local_kerma_fraction requires fragment species scoring");
    }
    if (enable_neutral_transport &&
        (!enable_secondary_generation || !enable_secondary_transport ||
         maximum_neutral_generations == 0)) {
        throw std::invalid_argument(
            "enable_neutral_transport requires secondary generation/transport and at "
            "least one neutral generation");
    }
    if (enable_neutral_transport && neutral_transport_mode != "first_interaction" &&
        neutral_transport_mode != "full") {
        throw std::invalid_argument(
            "neutral_transport_mode must be first_interaction or full");
    }
    if (enable_neutral_transport && neutral_transport_mode == "full" &&
        maximum_neutral_generations < 2) {
        throw std::invalid_argument(
            "neutral_transport_mode=full requires maximum_neutral_generations >= 2");
    }
    if (neutral_local_kerma_fraction < 0.0 || neutral_local_kerma_fraction > 1.0) {
        throw std::invalid_argument(
            "neutral_local_kerma_fraction must be in [0, 1]");
    }
    if (neutral_kerma_high_energy_scale < 1.0 || neutral_kerma_high_energy_scale > 2.0) {
        throw std::invalid_argument(
            "neutral_kerma_high_energy_scale must be in [1, 2]");
    }
    if (neutral_local_kerma_fraction * neutral_kerma_high_energy_scale > 1.0) {
        throw std::invalid_argument(
            "neutral_local_kerma_fraction * neutral_kerma_high_energy_scale must be <= 1");
    }
    if (neutral_kerma_mean_free_path_mm < 0.0) {
        throw std::invalid_argument(
            "neutral_kerma_mean_free_path_mm must be nonnegative (0 = local dump)");
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
    if (electronic_buildup_fraction > 0.0 && electronic_buildup_mfp_mm <= 0.0) {
        throw std::invalid_argument(
            "electronic_buildup_fraction > 0 requires electronic_buildup_mfp_mm > 0");
    }
    if (neutral_local_kerma_fraction > 0.0 && enable_neutral_transport) {
        throw std::invalid_argument(
            "neutral_local_kerma_fraction is only for neutral transport off "
            "(interim local kerma); disable enable_neutral_transport");
    }
    if (neutral_local_kerma_fraction > 0.0 && !enable_secondary_transport) {
        throw std::invalid_argument(
            "neutral_local_kerma_fraction requires enable_secondary_transport "
            "so deposits can be scored into the fragment IDD");
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
        spots_geometry_mode != "tps_90") {
        throw std::invalid_argument(
            "spots_geometry_mode must be topas, beam_plus_z, or tps_90");
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
        if (tps_particle_type != "carbon") {
            throw std::invalid_argument(
                "Only tps_particle_type=carbon is supported by the current physics tables");
        }
    } else if (!tps_spots_file.empty()) {
        throw std::invalid_argument(
            "tps_spots_file is set but tpsSource=false");
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
    const auto values = read_key_values(path);
    TransportConfig config;
    config.number_of_histories = parse_number(values, "number_of_histories", config.number_of_histories);
    config.initial_energy_MeVu = parse_number(values, "initial_energy_MeVu", config.initial_energy_MeVu);
    config.beam_energy_spread =
        parse_number(values, "beam_energy_spread", config.beam_energy_spread);
    config.mass_number = parse_number(values, "mass_number", config.mass_number);
    config.phantom_length_mm = parse_number(values, "phantom_length_mm", config.phantom_length_mm);
    config.depth_bin_width_mm = parse_number(values, "depth_bin_width_mm", config.depth_bin_width_mm);
    config.maximum_step_mm = parse_number(values, "maximum_step_mm", config.maximum_step_mm);
    config.maximum_relative_energy_loss =
        parse_number(values, "maximum_relative_energy_loss", config.maximum_relative_energy_loss);
    config.energy_cutoff_MeV = parse_number(values, "energy_cutoff_MeV", config.energy_cutoff_MeV);
    config.secondary_local_deposit_cutoff_MeV = parse_number(
        values, "secondary_local_deposit_cutoff_MeV",
        config.secondary_local_deposit_cutoff_MeV);
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
    if (values.find("insert_x_min_mm") != values.end() &&
        values.find("enable_hetero_insert") == values.end()) {
        config.enable_hetero_insert = true;
    }
    config.enable_ct_grid = parse_bool(values, "enable_ct_grid", config.enable_ct_grid);
    config.ct_grid_file = parse_path(values, "ct_grid_file", config.ct_grid_file);
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
    config.ct_stopping_power_scale = parse_number(
        values, "ct_stopping_power_scale", config.ct_stopping_power_scale);
    if (values.find("ct_grid_file") != values.end() &&
        values.find("enable_ct_grid") == values.end()) {
        config.enable_ct_grid = true;
    }
    config.scorer_area_mm2 = parse_number(values, "scorer_area_mm2", config.scorer_area_mm2);
    config.enable_voxel_scoring =
        parse_bool(values, "enable_voxel_scoring", config.enable_voxel_scoring);
    config.enable_charged_origin_voxel_scoring = parse_bool(
        values, "enable_charged_origin_voxel_scoring",
        config.enable_charged_origin_voxel_scoring);
    config.voxel_bins_x = parse_number(values, "voxel_bins_x", config.voxel_bins_x);
    config.voxel_bins_y = parse_number(values, "voxel_bins_y", config.voxel_bins_y);
    config.voxel_size_x_mm =
        parse_number(values, "voxel_size_x_mm", config.voxel_size_x_mm);
    config.voxel_size_y_mm =
        parse_number(values, "voxel_size_y_mm", config.voxel_size_y_mm);
    config.enable_energy_straggling =
        parse_bool(values, "enable_energy_straggling", config.enable_energy_straggling);
    config.enable_secondary_energy_straggling = parse_bool(
        values, "enable_secondary_energy_straggling",
        config.enable_secondary_energy_straggling);
    config.straggling_scale = parse_number(values, "straggling_scale", config.straggling_scale);
    config.enable_multiple_scattering =
        parse_bool(values, "enable_multiple_scattering", config.enable_multiple_scattering);
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
    config.enable_primary_attenuation =
        parse_bool(values, "enable_primary_attenuation", config.enable_primary_attenuation);
    config.enable_secondary_generation =
        parse_bool(values, "enable_secondary_generation", config.enable_secondary_generation);
    config.enable_secondary_transport =
        parse_bool(values, "enable_secondary_transport", config.enable_secondary_transport);
    config.enable_fragment_cascade =
        parse_bool(values, "enable_fragment_cascade", config.enable_fragment_cascade);
    config.enable_fragment_species_scoring = parse_bool(
        values, "enable_fragment_species_scoring",
        config.enable_fragment_species_scoring);
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
    config.enable_neutral_transport =
        parse_bool(values, "enable_neutral_transport", config.enable_neutral_transport);
    const auto neutral_mode = values.find("neutral_transport_mode");
    if (neutral_mode != values.end()) {
        config.neutral_transport_mode = neutral_mode->second;
    }
    config.neutral_local_kerma_fraction = parse_number(
        values, "neutral_local_kerma_fraction", config.neutral_local_kerma_fraction);
    config.neutral_kerma_high_energy_scale = parse_number(
        values, "neutral_kerma_high_energy_scale",
        config.neutral_kerma_high_energy_scale);
    config.neutral_kerma_mean_free_path_mm = parse_number(
        values, "neutral_kerma_mean_free_path_mm",
        config.neutral_kerma_mean_free_path_mm);
    config.electronic_buildup_fraction = parse_number(
        values, "electronic_buildup_fraction", config.electronic_buildup_fraction);
    config.electronic_buildup_mfp_mm = parse_number(
        values, "electronic_buildup_mfp_mm", config.electronic_buildup_mfp_mm);
    config.maximum_cascade_generations = parse_number(
        values, "maximum_cascade_generations", config.maximum_cascade_generations);
    config.maximum_neutral_generations = parse_number(
        values, "maximum_neutral_generations", config.maximum_neutral_generations);
    config.secondary_queue_capacity =
        parse_number(values, "secondary_queue_capacity", config.secondary_queue_capacity);
    config.neutral_queue_capacity =
        parse_number(values, "neutral_queue_capacity", config.neutral_queue_capacity);
    config.max_device_memory_fraction = parse_number(
        values, "max_device_memory_fraction", config.max_device_memory_fraction);
    config.history_chunk_size =
        parse_number(values, "history_chunk_size", config.history_chunk_size);
    config.secondary_batch_size =
        parse_number(values, "secondary_batch_size", config.secondary_batch_size);
    config.enable_secondary_energy_sorting = parse_bool(
        values, "enable_secondary_energy_sorting",
        config.enable_secondary_energy_sorting);
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
    {
        const auto camel = values.find("tpsSource");
        const auto snake = values.find("tps_source");
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
    config.tps_gantry_angle_deg = parse_number(
        values, "tps_gantry_angle_deg", config.tps_gantry_angle_deg);
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
    {
        const auto it = values.find("tps_particle_type");
        if (it != values.end() && !it->second.empty()) {
            config.tps_particle_type = it->second;
            std::transform(config.tps_particle_type.begin(),
                           config.tps_particle_type.end(),
                           config.tps_particle_type.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
        }
    }
    config.random_seed = parse_number(values, "random_seed", config.random_seed);
    config.stopping_power_file = parse_path(values, "stopping_power_file", config.stopping_power_file);
    config.let_delta_electron_fraction_file = parse_path(
        values, "let_delta_electron_fraction_file",
        config.let_delta_electron_fraction_file);
    config.use_particle_specific_stopping_power = parse_bool(
        values, "use_particle_specific_stopping_power",
        config.use_particle_specific_stopping_power);
    config.particle_stopping_power_file = parse_path(
        values, "particle_stopping_power_file",
        config.particle_stopping_power_file);
    config.nuclear_cross_section_file =
        parse_path(values, "nuclear_cross_section_file", config.nuclear_cross_section_file);
    config.reaction_package_file =
        parse_path(values, "reaction_package_file", config.reaction_package_file);
    config.cascade_package_file =
        parse_path(values, "cascade_package_file", config.cascade_package_file);
    config.neutral_package_file =
        parse_path(values, "neutral_package_file", config.neutral_package_file);
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
        const auto it = values.find("voxel_dose_mhd_output_file");
        if (it != values.end()) {
            config.voxel_dose_mhd_output_file =
                it->second.empty() ? std::filesystem::path{}
                                   : std::filesystem::path{it->second};
        }
    }
    const auto device = values.find("device");
    if (device != values.end()) {
        config.device = device->second;
    }
    config.validate();
    return config;
}

}  // namespace carbon
