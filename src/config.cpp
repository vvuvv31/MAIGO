#include <initializer_list>
#include "carbon/transport_config.hpp"
#include "carbon/preset_keys.hpp"
#include "carbon/source_parameters.hpp"
#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/min_json.hpp"
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
    if (!values.contains("primary_atomic_number") || !values.contains("primary_mass_number"))
        throw std::invalid_argument("Specify primary_atomic_number and primary_mass_number");
    const int z = parse_number(values, "primary_atomic_number", 0);
    const int a = parse_number(values, "primary_mass_number", 0);
    out.ion = make_primary_ion_definition(z, a);
    out.initial_energy_MeVu = parse_number(values, "initial_energy_MeVu", out.initial_energy_MeVu);
    if (!std::isfinite(out.initial_energy_MeVu) || !(out.initial_energy_MeVu > 0) ||
        !std::isfinite(out.initial_total_energy_MeV()))
        throw std::invalid_argument("Primary source energy must be finite and positive");
    out.beam_energy_spread = parse_number(values, "beam_energy_spread", 0.0);
    if (!std::isfinite(out.beam_energy_spread) || out.beam_energy_spread < 0 || out.beam_energy_spread > 0.2)
        throw std::invalid_argument("Beam energy spread must be finite and in [0,20] percent RMS");
    return out;
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

std::size_t TransportConfig::number_of_bins() const {
    return voxel_bins_z;
}

std::size_t TransportConfig::number_of_voxels() const {
    const auto depth_bins = number_of_bins();
    if (depth_bins == 0) throw std::invalid_argument("Voxel grid requires positive Z bin count");
    if (voxel_bins_y != 0 &&
        (voxel_bins_x > std::numeric_limits<std::size_t>::max() / voxel_bins_y ||
         voxel_bins_x * voxel_bins_y >
             std::numeric_limits<std::size_t>::max() / depth_bins)) {
        throw std::overflow_error("Voxel grid size exceeds the size_t runtime limit");
    }
    return voxel_bins_x * voxel_bins_y * depth_bins;
}

void TransportConfig::validate() const {
    const_cast<TransportConfig*>(this)->resolve_electron_transport();
    if (enable_electron_transport && material_electron_charged_index_file.empty())
        throw std::invalid_argument("enable_electron_transport requires explicit shared FP32 electron packages");
    if (device != "cuda") throw std::invalid_argument("device must be cuda");
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
    if (primary_atomic_number <= 0 || primary_mass_number < primary_atomic_number) {
        throw std::invalid_argument(
            "primary ion must satisfy primary_atomic_number > 0 and "
            "primary_mass_number >= primary_atomic_number");
    }
    (void)primary_ion(); // validate the complete source definition, including mass
    const auto energy_comes_from_spot_file =
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
    if (secondary_step_chunking &&
        (!enable_secondary_transport || !enable_inelastic))
        throw std::invalid_argument("secondary_step_chunking requires unified secondary EM and inelastic secondary transport");
    {
        if (em_package_file.empty())
            throw std::invalid_argument("Unified EM requires one em_package_file");
        if (run_mode == RunMode::production &&
            (!enable_energy_straggling || !enable_secondary_energy_straggling))
            throw std::invalid_argument("Production unified EM requires primary/secondary fluctuations");
        if ((!enable_ct_grid) && (!is_water_mode()))
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
    const_cast<TransportConfig*>(this)->resolve_material_physics_mode();
    if (unified_water_nuclear_transport) {
        if (!is_water_mode() || enable_ct_grid ||
            !ct_grid_file.empty() ||
            !enable_voxel_scoring ||
            nuclear_model != "geant4" ||
            initial_energy_MeVu > 430.0 ||
            maximum_step_mm > 1.0 || maximum_relative_energy_loss > 0.005 + 1.e-6)
            throw std::invalid_argument("Unified water requires native homogeneous water, 3D scoring, geant4 selector and CT step limits; legacy water routing is retired");
        if (ct_schneider_physics_bundle_file.empty() || ct_schneider_primary_rate_file.empty() ||
            ct_schneider_secondary_rate_file.empty() || ct_schneider_primary_cinel03_file.empty() ||
            ct_schneider_secondary_cinel03_file.empty() ||
            unified_water_material_file.empty())
            throw std::invalid_argument("Unified water requires explicit v2.1 bundle paths and pinned G4_WATER material");
    }
    if(material_electron_retain_bank &&
       (material_electron_charged_index_file.empty() || run_mode!=RunMode::research))
        throw std::invalid_argument("Electron cache requires SYCL research shared-electron transport");
    if(material_electron_queue_budget_MiB<1 || material_electron_queue_budget_MiB>4096 ||
       material_electron_queue_chunk<1 ||
       material_electron_queue_chunk>(material_electron_group_size==32 ? 1048576U : 65536U) ||
       material_electron_queue_lanes<1 || material_electron_queue_lanes>65536 ||
       material_electron_group_size<1 || material_electron_group_size>32 ||
       (material_electron_group_size & (material_electron_group_size-1)))
        throw std::invalid_argument("Invalid electron queue budget/chunk/lanes/group size");
    if(broadbeam_electron_region_diagnostics && !broadbeam_independent_electrons)
        throw std::invalid_argument("Electron region diagnostics require enabled CT broadbeam tracking");
    if(!std::isfinite(broadbeam_electron_low_density_survival) || broadbeam_electron_low_density_survival<=0 ||
       broadbeam_electron_low_density_survival>1 || !std::isfinite(broadbeam_electron_tissue_survival) ||
       broadbeam_electron_tissue_survival<=0 || broadbeam_electron_tissue_survival>1)
        throw std::invalid_argument("broadbeam electron low-density/tissue survival must be in (0,1]");
    if((broadbeam_electron_low_density_survival!=1 || broadbeam_electron_tissue_survival!=1 ||
        broadbeam_electron_birth_order) &&
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
        if(!enable_ct_grid && (!is_water_mode() ||
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
    if((!material_electron_charged_index_file.empty()) && ((material_electron_birth_file.empty()) || (material_photon_response_index_file.empty()) || (material_electron_joint_pilot_file.empty()) || (run_mode!=RunMode::research) || (!independent_projectile) || ((!enable_minibeam) && (!broadbeam_independent)) || (enable_charged_origin_voxel_scoring) || (!enable_voxel_scoring)))
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
        (!enable_minibeam && !broadbeam_independent) ||
        material_electron_charged_index_file.empty() ||
        (!broadbeam_independent && maximum_step_mm>.25)))
        throw std::invalid_argument("Joint pilot requires research projectile-matched minibeam, queue, bank and <=0.25 mm steps");
    if(!material_electron_charged_index_file.empty() && !enable_ct_grid &&
       (!is_water_mode() ||
        water_density_g_per_cm3!=1.0 || multiple_scattering_model!="urban_v2"))
        throw std::invalid_argument("Water shared electrons require homogeneous unit-density Water_75eV and global Urban");
    if (material_physics_mode == MaterialPhysicsMode::SchneiderCt) {
        if (ct_grid_file.empty()) {
            throw std::invalid_argument("MaterialPhysicsMode::SchneiderCt requires ct_grid_file");
        }
        const bool v3_bundle_mode = !ct_schneider_physics_bundle_file.empty();
        if (nuclear_model != "none" && !v3_bundle_mode)
            throw std::invalid_argument("Schneider CT requires the current physics bundle");
        if (v3_bundle_mode && nuclear_model != "none") {
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

    if (enable_primary_loss_query_audit && run_mode != RunMode::smoke) {
        throw std::invalid_argument("enable_primary_loss_query_audit requires smoke mode");
    }
    if (nuclear_model != "geant4" && nuclear_model != "none") {
        throw std::invalid_argument("nuclear_model must be geant4 or none");
    }
    if (max_secondary_inelastic_generations > 2U) {
        throw std::invalid_argument(
            "max_secondary_inelastic_generations must be between 0 and 2");
    }

    const bool pipeline_compiled =
        (transport_pipeline == "proton_broadbeam" && CARBON_ENABLE_PROTON_BROADBEAM) ||
        (transport_pipeline == "proton_minibeam" && CARBON_ENABLE_PROTON_MINIBEAM) ||
        (transport_pipeline == "carbon_broadbeam" && CARBON_ENABLE_CARBON_BROADBEAM) ||
        (transport_pipeline == "carbon_minibeam" && CARBON_ENABLE_CARBON_MINIBEAM) ||
        (transport_pipeline == "generic" && CARBON_ENABLE_GENERIC);
    if (transport_pipeline != "generic") {
        const bool proton = primary_atomic_number == 1 && primary_mass_number == 1;
        const bool carbon = primary_atomic_number == 6 && primary_mass_number == 12;
        const std::string expected = (proton ? "proton" : carbon ? "carbon" : "unsupported") +
            std::string(transport_profile == "broadbeam_fast" ? "_broadbeam" :
                        enable_minibeam ? "_minibeam" : "_reference");
        if (transport_pipeline != expected || (!proton && !carbon))
            throw std::invalid_argument("transport_pipeline must be generic or match the proton/C12 source and beam mode");
    }
    if (!pipeline_compiled)
        throw std::invalid_argument("Transport pipeline is not compiled: " + transport_pipeline);
    if(transport_profile!="reference" && transport_profile!="broadbeam_fast")
        throw std::invalid_argument("transport_profile must be reference or broadbeam_fast");
    if(transport_profile=="broadbeam_fast") {
        if(enable_minibeam || run_mode!=RunMode::research ||
           multiple_scattering_model!="fermi_eyges" || !enable_multiple_scattering ||
           !urban_mcs_package_file.empty() ||


           (enable_ct_grid && !enable_ct_material_mcs))
            throw std::invalid_argument("broadbeam_fast requires research unified EM, FE for all charged species, CT material MCS, no beamline transport; only pinned research C12 CT joint electron response is supported");
    }
    if (transport_profile == "broadbeam_fast" && !fermi_eyges_gaussian_only)
        throw std::invalid_argument("broadbeam_fast requires fermi_eyges_gaussian_only: true");
    if(fermi_eyges_gaussian_only && (multiple_scattering_model!="fermi_eyges" ||
                                   run_mode!=RunMode::research))
        throw std::invalid_argument("fermi_eyges_gaussian_only requires research SYCL fermi_eyges MCS");
    if (multiple_scattering_model != "fermi_eyges" &&
        multiple_scattering_model != "urban_v2") {
        throw std::invalid_argument(
            "multiple_scattering_model must be fermi_eyges or urban_v2");
    }

    if(urban_ct_density_scaling && (run_mode!=RunMode::research ||
       !enable_ct_grid || multiple_scattering_model!="urban_v2"))
        throw std::invalid_argument("urban_ct_density_scaling requires research Schneider CT with global Urban");
    if(multiple_scattering_model=="urban_v2") {
        if(urban_mcs_package_file.empty())
            throw std::invalid_argument("Global urban_v2 requires a validated urban_mcs_package_file");
        if(!enable_multiple_scattering)
            throw std::invalid_argument("Global urban_v2 requires MCS");
        if(enable_ct_grid && (!is_schneider_ct_mode() || !ct_secondary_exact_faces))
            throw std::invalid_argument("Global urban_v2 requires Schneider CT and exact secondary faces");
        if(energy_cutoff_MeV<0.05 || effective_secondary_local_deposit_cutoff_MeV()<0.05)
            throw std::invalid_argument("Global urban_v2 reference currently requires transport cutoffs >= 0.05 MeV");
        if(!enable_ct_grid && (!(urban_water_half_width_mm>0) || !std::isfinite(urban_water_half_width_mm)))
            throw std::invalid_argument("Global water urban_v2 requires the TOPAS physical urban_water_half_width_mm");
    }
    if (!std::isfinite(fermi_eyges_max_segment_mm) ||
        fermi_eyges_max_segment_mm <= 0.0) {
        throw std::invalid_argument(
            "fermi_eyges_max_segment_mm must be finite and positive");
    }
    if (enable_voxel_scoring &&
        (voxel_bins_x == 0 || voxel_bins_y == 0 || voxel_size_x_mm <= 0.0 ||
         voxel_size_y_mm <= 0.0 ||
         (voxel_bins_z == 0 || !std::isfinite(voxel_size_z_mm) || voxel_size_z_mm <= 0.0))) {
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
                "voxel z geometry must match phantom length and depth-bin spacing");
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

    if (!primary_voxel_fluence_mhd_output_file.empty() &&
        !enable_voxel_scoring) {
        throw std::invalid_argument(
            "primary_voxel_fluence_mhd_output_file requires "
            "enable_voxel_scoring=true");
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
        if (multiple_scattering_model != "urban_v2") {
            throw std::invalid_argument(
                "minibeam requires multiple_scattering_model: urban_v2");
        }
        if (!(minibeam_collimator_center_to_isocenter_mm >= 0.0 &&
              minibeam_radius_mm > 0.0 &&
              minibeam_collimator_thickness_mm > 0.0 &&
              minibeam_slit_length_mm > 0.0 &&
              minibeam_slit_thickness_mm > 0.0 &&
              minibeam_slit_width_mm > 0.0 && minibeam_slit_pitch_mm > 0.0 &&
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
        {
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
                (run_mode == RunMode::production || multiple_scattering_model != "urban_v2" ||
                 !minibeam_copper_enable_mcs ||
                 !minibeam_copper_enable_energy_straggling ||
                 (minibeam_copper_straggling_model != "g4_universal_delta" &&
                  minibeam_copper_straggling_model != "g4_ion_delta")))
                throw std::invalid_argument("Native Cu primary mean requires Urban v2 and finite-cut fluctuations");
            if (minibeam_copper_straggling_model != "clipped_gaussian" &&
                minibeam_copper_straggling_model != "g4_universal_delta" &&
                minibeam_copper_straggling_model != "g4_ion_delta")
                throw std::invalid_argument("Unknown copper straggling model");
            if (minibeam_copper_straggling_model == "g4_ion_delta" &&
                (primary_atomic_number != 6 || primary_mass_number != 12 ||
                 !minibeam_copper_primary_native_mean_loss ||
                 minibeam_copper_fragment_urban_package_file.empty() ||
                 !minibeam_copper_fragment_enable_energy_straggling ||
                 (minibeam_copper_enable_elastic && minibeam_copper_fragment_elastic_file.empty())))
                throw std::invalid_argument("Cu ion transport requires C12, native mean, fragment fluctuations and native fragment Urban data and a species-specific elastic bank");
            if (minibeam_copper_straggling_model != "clipped_gaussian" &&
                std::abs(minibeam_copper_density_g_per_cm3 - 8.96) > 1e-5)
                throw std::invalid_argument("Native copper cut is pinned to G4_Cu at 8.96 g/cm3");
            if (minibeam_copper_straggling_model != "clipped_gaussian" &&
                (run_mode == RunMode::production || minibeam_copper_straggling_scale != 1.0 ||
                 minibeam_copper_fragment_straggling_scale != 1.0))
                throw std::invalid_argument("Native copper fluctuation requires research/smoke and unit straggling scales");
            if (minibeam_copper_enable_elastic &&
                minibeam_copper_elastic_file.empty()) {
                throw std::invalid_argument(
                    "minibeam copper_em requires minibeam_copper_elastic_file");
            }
            if (!(minibeam_copper_density_g_per_cm3 > 0.0 &&
                  minibeam_copper_radiation_length_g_per_cm2 > 0.0 &&
                  minibeam_copper_max_step_mm > 0.0 &&
                  minibeam_copper_mcs_scale > 0.0 &&
                  minibeam_copper_fragment_mcs_scale > 0.0 &&
                  minibeam_copper_fragment_cascade_generations <= 3U &&
                  minibeam_copper_straggling_scale > 0.0 &&
                  minibeam_copper_fragment_straggling_scale > 0.0) ||
                !std::isfinite(minibeam_copper_mcs_scale) ||
                !std::isfinite(minibeam_copper_fragment_mcs_scale) ||
                !std::isfinite(minibeam_copper_straggling_scale) ||
                !std::isfinite(minibeam_copper_fragment_straggling_scale)) {
                throw std::invalid_argument(
                    "minibeam Copper density, radiation length, max step, and "
                    "MCS/straggling scales must be finite and positive");
            }
        }
    }
    if (enable_tps_source) {

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
        if (!std::isfinite(tps_beam_angle_deg) ||
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
            tps_angle_convention != "iec61217") {
            throw std::invalid_argument(
                "tps_angle_convention must be dicom_lps (preferred) or "
                "iec61217");
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
        if (!enable_tps_coordinate_system) {
            throw std::invalid_argument(
                "spots_enable_upstream_air_energy_loss requires fixed TPS coordinates");
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

        }
        if (expected_begin != number_of_histories) {
            throw std::invalid_argument(
                "primary_spot_batch ranges must cover number_of_histories exactly");
        }
    }
}

// Current physics bundle enforces rate/package identity, grids, domains and
// registry ordering for native water and Schneider CT. Nuclear rates require v3.
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
            if (!c.contains("has_support") ||
                c.at("has_support").type != minjson::Value::Type::Boolean)
                throw std::runtime_error("Primary channel requires explicit boolean has_support");
            const bool has_support = c.at("has_support").boolean;
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
                // The current rate readers use only basename.metadata.json.
                const auto metadata = std::filesystem::path(path).replace_extension(".metadata.json");
                if (!std::filesystem::exists(metadata)) {
                    throw std::runtime_error("Schneider CT startup failed: companion metadata missing for " + path.string());
                }
            }
        }

    }
}

TransportConfig load_config(const std::filesystem::path& path) {
    auto values = read_key_values(path);
    for (const auto& [key,value]: values)
        if (!is_preset_key(key)) throw std::invalid_argument("Unsupported preset configuration key: "+key);
    // Record the mode-specific ion step default in the canonical config;
    // an explicit maximum_step_mm remains authoritative.
    values.try_emplace("maximum_step_mm",
        parse_bool(values, "minibeam", false) &&
        !parse_bool(values, "enable_ct_grid", false) && !values.contains("ct_grid_file")
            ? "0.1" : "0.5");
    // Broad-beam YAML defaults apply to every source ZA, including protons.
    // Insert before canonicalization so the effective defaults are recorded.
    // Explicit model, enable/disable and species selections remain authoritative.
    if (!parse_bool(values, "minibeam", false)) {
        values.try_emplace("multiple_scattering_model", "fermi_eyges");
        values.try_emplace("enable_multiple_scattering", "true");
    }
    // Expand the explicit fast profile before parsing paths, validation and
    // canonicalization. Unused packages are never opened/prefetched. Unknown
    // keys are still rejected; only named, supported inputs are removed.
    const auto profile_it=values.find("transport_profile");
    const std::string profile=profile_it==values.end()?"reference":profile_it->second;
    const bool tracking_declared=values.contains("broadbeam_electron_tracking");
    values.try_emplace("enable_electron_transport", "false");
    const bool electrons_requested=parse_bool(values,"enable_electron_transport",false);
    const bool tracking_requested=electrons_requested && parse_bool(values,"broadbeam_electron_tracking",false);
    if(tracking_declared && profile!="broadbeam_fast")
        throw std::invalid_argument("broadbeam_electron_tracking is a broadbeam_fast setting");
    const bool independent_profile=tracking_requested && parse_bool(values,"enable_ct_grid",false);
    if(profile=="broadbeam_fast") {
        if(parse_bool(values,"minibeam",false))
            throw std::invalid_argument("broadbeam_fast requires minibeam: false");
        values.try_emplace("fermi_eyges_gaussian_only", "true");
        for(const char* key:{
            "urban_mcs_package_file","material_electron_charged_index_file", "material_photon_response_index_file", "material_electron_birth_file", "material_electron_joint_pilot_file"}) {
            const std::string_view name(key);
            const bool shared=name.starts_with("material_electron_charged_") || name.starts_with("material_photon_response_") ||
                name.starts_with("material_electron_birth_") || name.starts_with("material_electron_joint_pilot_");
            if(!independent_profile || !shared)values.erase(key);
        }
        for(const auto& setting:std::initializer_list<std::pair<const char*,const char*>>{
            {"multiple_scattering_model","fermi_eyges"},{"enable_multiple_scattering","true"},
            {"enable_ct_material_mcs","true"},
            {"urban_ct_density_scaling","false"},
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
    config.enable_electron_transport=electrons_requested;
    config.transport_profile=profile;
    if (auto pipeline=values.find("transport_pipeline"); pipeline!=values.end())
        config.transport_pipeline=pipeline->second;
    config.broadbeam_electron_tracking=tracking_requested;
    config.broadbeam_independent_electrons=independent_profile;
    config.broadbeam_electron_density_threshold=parse_number(values,"broadbeam_electron_density_threshold",config.broadbeam_electron_density_threshold);
    config.broadbeam_electron_halo_mm=parse_number(values,"broadbeam_electron_halo_mm",config.broadbeam_electron_halo_mm);
    config.broadbeam_electron_low_density_survival=parse_number(values,"broadbeam_electron_low_density_survival",1.);
    config.broadbeam_electron_tissue_survival=parse_number(values,"broadbeam_electron_tissue_survival",1.);
    config.broadbeam_electron_birth_order=parse_bool(values,"broadbeam_electron_birth_order",false);
    // Variance-reduction settings are inert when the master switch is off.
    if(!tracking_requested){config.broadbeam_electron_low_density_survival=1;config.broadbeam_electron_tissue_survival=1;
        config.broadbeam_electron_birth_order=false;}
    // Eligible broad-beam full-physics runs benefit from the existing grouped
    // continuation scheduler. Preserve explicit opt-outs and EM-only inputs.
    if (profile == "broadbeam_fast" &&
        parse_bool(values, "enable_inelastic", config.enable_inelastic) &&
        parse_bool(values, "enable_secondary_transport", config.enable_secondary_transport)) {
        values.try_emplace("secondary_species_grouping", "true");
        values.try_emplace("secondary_step_chunking", "true");
    }
    config.fermi_eyges_gaussian_only=parse_bool(values,"fermi_eyges_gaussian_only",false);

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

    config.number_of_histories = parse_number(values, "number_of_histories", config.number_of_histories);
    const auto source = parse_primary_source_parameters(values);
    config.initial_energy_MeVu = source.initial_energy_MeVu;
    config.beam_energy_spread = source.beam_energy_spread;
    config.primary_atomic_number = source.ion.atomic_number;
    config.primary_mass_number = source.ion.mass_number;
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
    config.water_density_g_per_cm3 =
        parse_number(values, "water_density_g_per_cm3", config.water_density_g_per_cm3);
    config.enable_ct_grid = parse_bool(values, "enable_ct_grid", config.enable_ct_grid);
    config.ct_grid_file = parse_path(values, "ct_grid_file", config.ct_grid_file);
    config.ct_schneider_file =
        parse_path(values, "ct_schneider_file", config.ct_schneider_file);
    if(!config.ct_schneider_file.empty())
        config.ct_schneider_file=resolve_input_path_from_config(config.ct_schneider_file,path);

    config.ct_skip_homogeneous_face_clamp = parse_bool(
        values, "ct_skip_homogeneous_face_clamp", config.ct_skip_homogeneous_face_clamp);
    config.unified_water_nuclear_transport = parse_bool(values,
        "unified_water_nuclear_transport", config.unified_water_nuclear_transport);
    config.unified_water_material_file = parse_path(values,
        "unified_water_material_file", config.unified_water_material_file);
    if (!config.unified_water_material_file.empty())
        config.unified_water_material_file = resolve_input_path_from_config(config.unified_water_material_file, path);
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
    config.all_ion_elastic_file = parse_path(values,"all_ion_elastic_file",config.all_ion_elastic_file);
    config.elastic_recoil_stopping_file=parse_path(values,"elastic_recoil_stopping_file",config.elastic_recoil_stopping_file);
    if(!config.all_ion_elastic_file.empty()) {
        if(config.elastic_recoil_stopping_file.empty())
            throw std::invalid_argument("All-ion elastic requires recoil stopping data");
        config.elastic_recoil_stopping_file=resolve_input_path_from_config(config.elastic_recoil_stopping_file,path);
        config.all_ion_elastic_file=resolve_input_path_from_config(config.all_ion_elastic_file,path);
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
    if (!config.ct_schneider_radiation_length_file.empty()) {
        config.ct_schneider_radiation_length_file = resolve_input_path_from_config(
            config.ct_schneider_radiation_length_file, path);
    }
    config.material_electron_charged_index_file=parse_path(values,"material_electron_charged_index_file",config.material_electron_charged_index_file);
    config.material_electron_birth_file=parse_path(values,"material_electron_birth_file",config.material_electron_birth_file);
    config.material_photon_response_index_file=parse_path(values,"material_photon_response_index_file",config.material_photon_response_index_file);
    config.material_electron_joint_pilot_file=parse_path(values,"material_electron_joint_pilot_file",config.material_electron_joint_pilot_file);
    if (electrons_requested) {
        if(!config.material_electron_charged_index_file.empty())config.material_electron_charged_index_file=resolve_input_path_from_config(config.material_electron_charged_index_file,path);
        if(!config.material_photon_response_index_file.empty())config.material_photon_response_index_file=resolve_input_path_from_config(config.material_photon_response_index_file,path);
        if(!config.material_electron_birth_file.empty())config.material_electron_birth_file=resolve_input_path_from_config(config.material_electron_birth_file,path);
        if(!config.material_electron_joint_pilot_file.empty())config.material_electron_joint_pilot_file=resolve_input_path_from_config(config.material_electron_joint_pilot_file,path);
    }
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

    if (const auto it = values.find("secondary_out_of_scope_nuclear_policy");
        it != values.end()) {
        config.secondary_out_of_scope_nuclear_policy = it->second;
    }
    if (values.find("ct_grid_file") != values.end() &&
        values.find("enable_ct_grid") == values.end()) {
        config.enable_ct_grid = true;
    }
    config.resolve_material_physics_mode();
    config.resolve_electron_transport();
    config.scorer_area_mm2 = parse_number(values, "scorer_area_mm2", config.scorer_area_mm2);
    config.dose_output_scale =
        parse_number(values, "dose_output_scale", config.dose_output_scale);
    config.enable_voxel_scoring =
        parse_bool(values, "enable_voxel_scoring", config.enable_voxel_scoring);
    if (parse_bool(values,"enable_charged_origin_voxel_scoring",false))
        throw std::invalid_argument("Additional research scorers are not part of the eight presets");
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
        const auto grid_owner = CtGrid::load(
            resolve_input_path_from_config(config.ct_grid_file, path));
        const auto& grid = *grid_owner;
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
        config.depth_bin_width_mm = config.voxel_size_z_mm;
        config.phantom_length_mm = ct_z_extent_mm;

        std::cout << "[material-indexing] mode: schneider-25\n";
        if (config.nuclear_model != "none" && config.ct_schneider_physics_bundle_file.empty())
            throw std::runtime_error("Schneider CT requires the current physics bundle");
    } else {
        if (!values.contains("voxel_bins_z") || !values.contains("voxel_size_z_mm")) {
            throw std::invalid_argument("Water geometry requires explicit voxel_bins_z and voxel_size_z_mm");
        }
        config.phantom_length_mm = static_cast<double>(config.voxel_bins_z) * config.voxel_size_z_mm;
        config.depth_bin_width_mm = config.voxel_size_z_mm;
    }
    config.enable_energy_straggling =
        parse_bool(values, "enable_energy_straggling", config.enable_energy_straggling);
    config.enable_secondary_energy_straggling = parse_bool(
        values, "enable_secondary_energy_straggling",
        config.enable_secondary_energy_straggling);
    config.secondary_species_grouping = parse_bool(
        values, "secondary_species_grouping", config.secondary_species_grouping);
    config.minibeam_primary_specialization = parse_bool(
        values, "minibeam_primary_specialization", config.minibeam_primary_specialization);
    {
        const auto it = values.find("minibeam_primary_pipeline");
        if (it != values.end()) config.minibeam_primary_pipeline = it->second;
        if (config.minibeam_primary_pipeline != "compact")
            throw std::invalid_argument(
                "minibeam_primary_pipeline must be compact");
    }
    config.secondary_step_chunking = parse_bool(
        values, "secondary_step_chunking", config.secondary_step_chunking);
    config.enable_inelastic =
        parse_bool(values, "enable_inelastic", config.enable_inelastic);

    config.max_secondary_inelastic_generations = parse_number(
        values, "max_secondary_inelastic_generations",
        config.max_secondary_inelastic_generations);
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
    if(const auto it=values.find("urban_mcs_package_file");it!=values.end())
        config.urban_mcs_package_file=it->second;
    config.urban_water_half_width_mm=parse_number(values,"urban_water_half_width_mm",config.urban_water_half_width_mm);
    config.urban_ct_density_scaling=parse_bool(values,"urban_ct_density_scaling",config.urban_ct_density_scaling);
    config.fermi_eyges_max_segment_mm = parse_number(
        values, "fermi_eyges_max_segment_mm",
        config.fermi_eyges_max_segment_mm);
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

    config.max_device_memory_fraction = parse_number(
        values, "max_device_memory_fraction", config.max_device_memory_fraction);
    config.device_memory_budget_gib = parse_number(
        values, "device_memory_budget_gib", config.device_memory_budget_gib);
    config.secondary_queue_capacity = parse_number(
        values, "secondary_queue_capacity", config.secondary_queue_capacity);
    config.history_chunk_size =
        parse_number(values, "history_chunk_size", config.history_chunk_size);
    config.source_origin_z_mm =
        parse_number(values, "source_origin_z_mm", config.source_origin_z_mm);
    config.enable_minibeam =
        parse_bool(values, "minibeam", config.enable_minibeam);
    config.minibeam_radius_mm = parse_number(
        values, "minibeam_radius_mm", config.minibeam_radius_mm);
    config.minibeam_collimator_center_to_isocenter_mm = parse_number(
        values, "minibeam_collimator_center_to_isocenter_mm",
        config.minibeam_collimator_center_to_isocenter_mm);
    config.minibeam_collimator_thickness_mm = parse_number(
        values, "minibeam_collimator_thickness_mm",
        config.minibeam_collimator_thickness_mm);
    config.minibeam_slit_length_mm = parse_number(
        values, "minibeam_slit_length_mm", config.minibeam_slit_length_mm);
    config.minibeam_slit_thickness_mm = parse_number(
        values, "minibeam_slit_thickness_mm",
        config.minibeam_slit_thickness_mm);
    config.minibeam_slit_count = parse_number(
        values, "minibeam_slit_count", config.minibeam_slit_count);
    config.minibeam_slit_width_mm = parse_number(
        values, "minibeam_slit_width_mm", config.minibeam_slit_width_mm);
    config.minibeam_slit_pitch_mm = parse_number(
        values, "minibeam_slit_pitch_mm", config.minibeam_slit_pitch_mm);
    config.minibeam_collimator_angle_deg = parse_number(
        values, "minibeam_collimator_angle_deg",
        config.minibeam_collimator_angle_deg);
    config.minibeam_copper_stopping_power_file = parse_path(
        values, "minibeam_copper_stopping_power_file",
        config.minibeam_copper_stopping_power_file);
    config.minibeam_air_stopping_power_file = parse_path(
        values, "minibeam_air_stopping_power_file",
        config.minibeam_air_stopping_power_file);
    config.minibeam_air_slit_max_step_mm = parse_number(values, "minibeam_air_slit_max_step_mm", config.minibeam_air_slit_max_step_mm);
    if (!(config.minibeam_air_slit_max_step_mm > 0.0) || !std::isfinite(config.minibeam_air_slit_max_step_mm))
        throw std::invalid_argument("Air slit max step must be finite and positive");
    config.minibeam_air_urban_package_file = parse_path(values,
        "minibeam_air_urban_package_file", config.minibeam_air_urban_package_file);
    config.minibeam_air_ionization_rate_file = parse_path(values,
        "minibeam_air_ionization_rate_file", config.minibeam_air_ionization_rate_file);
    if (!config.minibeam_air_ionization_rate_file.empty() &&
        (!config.enable_minibeam || config.run_mode==RunMode::production))
        throw std::invalid_argument("Air ionization clock requires a research/smoke minibeam and a pinned rate table");
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
    config.minibeam_copper_enable_elastic = parse_bool(
        values, "minibeam_copper_enable_elastic",
        config.minibeam_copper_enable_elastic);
    config.minibeam_copper_enable_energy_straggling = parse_bool(
        values, "minibeam_copper_enable_energy_straggling",
        config.minibeam_copper_enable_energy_straggling);
    config.minibeam_copper_primary_native_mean_loss = parse_bool(
        values, "minibeam_copper_primary_native_mean_loss",
        config.minibeam_copper_primary_native_mean_loss);
    if (const auto it = values.find("minibeam_copper_straggling_model"); it != values.end())
        config.minibeam_copper_straggling_model = it->second;
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
    config.minibeam_copper_fragment_urban_package_file = parse_path(
        values, "minibeam_copper_fragment_urban_package_file",
        config.minibeam_copper_fragment_urban_package_file);
    config.minibeam_copper_fragment_elastic_file = parse_path(
        values, "minibeam_copper_fragment_elastic_file",
        config.minibeam_copper_fragment_elastic_file);
    config.minibeam_copper_ion_stopping_power_file = parse_path(
        values, "minibeam_copper_ion_stopping_power_file",
        config.minibeam_copper_ion_stopping_power_file);
    config.minibeam_copper_ion_cross_section_file = parse_path(
        values, "minibeam_copper_ion_cross_section_file",
        config.minibeam_copper_ion_cross_section_file);
    config.minibeam_water_entrance_world_y_mm = parse_number(
        values, "minibeam_water_entrance_world_y_mm",
        config.minibeam_water_entrance_world_y_mm);

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
    config.spots_enable_upstream_air_energy_loss = parse_bool(
        values, "spots_enable_upstream_air_energy_loss",
        config.spots_enable_upstream_air_energy_loss);
    config.ct_secondary_exact_faces = parse_bool(values,
        "ct_secondary_exact_faces", config.ct_secondary_exact_faces);
    config.spots_upstream_air_stopping_power_file = parse_path(
        values, "spots_upstream_air_stopping_power_file",
        config.spots_upstream_air_stopping_power_file);
    config.enable_tps_coordinate_system = parse_bool(values,
        "enable_tps_coordinate_system", config.enable_tps_coordinate_system);
    config.enable_tps_source = parse_bool(values, "tpsSource", config.enable_tps_source);
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
    config.tps_beam_angle_deg = parse_number(values,
        "tps_beam_angle_deg", config.tps_beam_angle_deg);
    config.tps_isocenter_z_mm = parse_number(
        values, "tps_isocenter_z_mm", config.tps_isocenter_z_mm);
    config.tps_sad_mm = parse_number(values, "tps_sad_mm", config.tps_sad_mm);

    if (const auto seed = values.find("random_seed"); seed != values.end()) {
        config.random_seed = parse_random_seed(seed->second);
    }
    config.primary_stopping_power_file = parse_path(values, "primary_stopping_power_file", config.primary_stopping_power_file);
    config.particle_stopping_power_file = parse_path(
        values, "particle_stopping_power_file",
        config.particle_stopping_power_file);

    config.minibeam_phase_space_at_copper_exit = parse_bool(
        values, "minibeam_phase_space_at_copper_exit",
        config.minibeam_phase_space_at_copper_exit);
    if (config.minibeam_phase_space_at_copper_exit &&
        (!config.enable_minibeam ||
         config.minibeam_phase_space_output_file.empty() || config.run_mode == RunMode::production)) {
        throw std::invalid_argument("Copper-exit phase scoring requires a research/smoke minibeam and a phase output file");
    }
    {
        const auto it = values.find(
            "minibeam_water_primary_plane_depths_mm");
        if (it != values.end()) {
            config.minibeam_water_primary_plane_depths_mm = parse_double_list(
                it->second, "minibeam_water_primary_plane_depths_mm");
        }
    }

    // Dose (Gy) scorers. Empty string disables that Gy file (MeV scorers unchanged).

    {
        const auto enabled_it = values.find("dose_to_medium");
        const auto type_it = values.find("dose_to_medium_type");
        const auto name_it = values.find("dose_to_medium_name");
        const auto has_public_dose = enabled_it != values.end() ||
                                     type_it != values.end() ||
                                     name_it != values.end();
        if (has_public_dose) {

            if (enabled_it == values.end()) {
                throw std::runtime_error(
                    "dose_to_medium_type/name require dose_to_medium");
            }
            const auto enabled = parse_bool(values, "dose_to_medium", false);
            // The high-level scorer owns the production output set. Suppress
            // debug CSV sinks for the production MHD output set.
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
    if (parse_bool(values,"LET",false))
        throw std::invalid_argument("The eight presets use dose scoring with LET off");

    const auto device = values.find("device");
    if (device != values.end()) {
        config.device = device->second;
    }

    if(!config.all_ion_elastic_file.empty()) {
        if((!config.enable_inelastic) || (!config.enable_secondary_transport) || ((!config.enable_ct_grid) && (!config.unified_water_nuclear_transport)))
            throw std::invalid_argument("All-ion elastic requires Schneider CT/unified-water nuclear transport, secondary transport and LET off");
        if(config.enable_ct_grid && !config.ct_secondary_exact_faces)
            throw std::invalid_argument("All-ion elastic requires secondary exact faces");
    }
    if (!((config.primary_atomic_number==1 && config.primary_mass_number==1) ||
          (config.primary_atomic_number==6 && config.primary_mass_number==12)))
        throw std::invalid_argument("Preset source must be proton or C12");
    reject_unknown_config_keys(values, path);
    config.canonical_config_text = canonicalize_config(
        values, config.config_schema_version);
    config.validate();

    return config;
}

}  // namespace carbon
