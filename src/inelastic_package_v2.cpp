#include "carbon/inelastic_package_v2.hpp"
#include "carbon/inelastic_identity.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace carbon {
namespace {

constexpr std::array<char, 8> package_magic{'C', 'I', 'N', 'P', 'K', 'G', '0', '3'};
constexpr std::uint32_t package_version = 3;
constexpr std::uint32_t endian_marker = 0x01020304U;
constexpr std::uint32_t required_flags = 0x0FU;
constexpr std::uint32_t known_flags = required_flags | cinel02_global_energy_index_flag;

template <typename Record>
void read_records(std::ifstream& input, std::vector<Record>& records,
                  std::uint64_t count, const std::filesystem::path& path,
                  const char* label) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                    sizeof(Record)) {
        throw std::runtime_error(std::string("CINPKG03 ") + label +
                                 " count is too large: " + path.string());
    }
    records.resize(static_cast<std::size_t>(count));
    const auto bytes = static_cast<std::streamsize>(records.size() * sizeof(Record));
    if (bytes > 0 && !input.read(reinterpret_cast<char*>(records.data()), bytes)) {
        throw std::runtime_error(std::string("Truncated CINPKG03 ") + label +
                                 " table: " + path.string());
    }
}

bool unit_direction(float x, float y, float z) {
    const auto norm = static_cast<double>(x) * x + static_cast<double>(y) * y +
                       static_cast<double>(z) * z;
    return std::isfinite(norm) && std::abs(norm - 1.0) <= 2.0e-3;
}

bool c_string_present(const char* value, std::size_t size) {
    return std::find(value, value + size, '\0') != value + size && value[0] != '\0';
}

bool canonical_uuid(const char* value, std::size_t size) {
    if (size != 36U || value[8] != '-' || value[13] != '-' ||
        value[18] != '-' || value[23] != '-') {
        return false;
    }
    for (std::size_t index = 0; index < size; ++index) {
        if (index == 8U || index == 13U || index == 18U || index == 23U) {
            continue;
        }
        if (!std::isxdigit(static_cast<unsigned char>(value[index]))) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, ',')) {
        const auto first = field.find_first_not_of(" \t\r\n");
        const auto last = field.find_last_not_of(" \t\r\n");
        fields.push_back(first == std::string::npos
                             ? std::string{}
                             : field.substr(first, last - first + 1));
    }
    return fields;
}

double parse_rate_number(const std::string& value,
                         const std::filesystem::path& path,
                         const std::size_t line_number) {
    std::size_t consumed = 0;
    double result = 0.0;
    try {
        result = std::stod(value, &consumed);
    } catch (...) {
        throw std::runtime_error("Invalid CINEL02 rate value at " + path.string() + ":" +
                                 std::to_string(line_number));
    }
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::runtime_error("Invalid CINEL02 rate value at " + path.string() + ":" +
                                 std::to_string(line_number));
    }
    return result;
}

struct RateRow {
    int projectile_z{0};
    int projectile_a{0};
    int target_z{0};
    int target_a{0};
    double energy{0.0};
    double cross_section{0.0};
};

auto energy_node_key(const Cinel02EnergyNode& node) {
    return std::tuple{static_cast<int>(node.projectile_z),
                      static_cast<int>(node.projectile_a),
                      static_cast<int>(node.target_z),
                      static_cast<int>(node.target_a),
                      node.collision_energy_MeV_per_u};
}

void build_energy_index(
    const std::vector<Cinel02InteractionRecord>& interactions,
    std::vector<Cinel02EnergyNode>& nodes,
    std::vector<std::uint64_t>& offsets,
    std::vector<std::uint64_t>& indices) {
    std::vector<std::uint64_t> order(interactions.size());
    std::iota(order.begin(), order.end(), std::uint64_t{0});
    std::sort(order.begin(), order.end(), [&](const std::uint64_t lhs,
                                              const std::uint64_t rhs) {
        const auto lhs_key = std::tuple{
            static_cast<int>(interactions[static_cast<std::size_t>(lhs)].projectile_z),
            static_cast<int>(interactions[static_cast<std::size_t>(lhs)].projectile_a),
            static_cast<int>(interactions[static_cast<std::size_t>(lhs)].target_z),
            static_cast<int>(interactions[static_cast<std::size_t>(lhs)].target_a),
            interactions[static_cast<std::size_t>(lhs)].collision_energy_MeV_per_u};
        const auto rhs_key = std::tuple{
            static_cast<int>(interactions[static_cast<std::size_t>(rhs)].projectile_z),
            static_cast<int>(interactions[static_cast<std::size_t>(rhs)].projectile_a),
            static_cast<int>(interactions[static_cast<std::size_t>(rhs)].target_z),
            static_cast<int>(interactions[static_cast<std::size_t>(rhs)].target_a),
            interactions[static_cast<std::size_t>(rhs)].collision_energy_MeV_per_u};
        return lhs_key < rhs_key ||
               (lhs_key == rhs_key && lhs < rhs);
    });

    nodes.clear();
    offsets.clear();
    indices.clear();
    offsets.push_back(0U);
    for (const auto interaction_index : order) {
        const auto& interaction = interactions[static_cast<std::size_t>(interaction_index)];
        const Cinel02EnergyNode node{
            interaction.projectile_z, interaction.projectile_a,
            interaction.target_z, interaction.target_a,
            interaction.collision_energy_MeV_per_u};
        if (nodes.empty() || energy_node_key(nodes.back()) != energy_node_key(node)) {
            nodes.push_back(node);
            offsets.push_back(offsets.back());
        }
        indices.push_back(interaction_index);
        ++offsets.back();
    }
}

void validate_energy_index(
    const std::vector<Cinel02EnergyNode>& nodes,
    const std::vector<std::uint64_t>& offsets,
    const std::vector<std::uint64_t>& indices,
    const std::vector<Cinel02InteractionRecord>& interactions,
    const std::filesystem::path& path) {
    if (nodes.empty() || offsets.size() != nodes.size() + 1U ||
        indices.size() != interactions.size() || offsets.front() != 0U ||
        offsets.back() != interactions.size()) {
        throw std::runtime_error("Invalid CINPKG03 global energy index: " + path.string());
    }
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const auto& node = nodes[node_index];
        if (node.projectile_z <= 0 || node.projectile_a < node.projectile_z ||
            node.target_z <= 0 || node.target_a < node.target_z ||
            !std::isfinite(node.collision_energy_MeV_per_u) ||
            node.collision_energy_MeV_per_u < 0.0F ||
            offsets[node_index] >= offsets[node_index + 1U] ||
            offsets[node_index + 1U] > interactions.size() ||
            (node_index > 0U && energy_node_key(nodes[node_index - 1U]) >=
                                    energy_node_key(node))) {
            throw std::runtime_error("Invalid CINPKG03 global energy node: " + path.string());
        }
    }

    std::vector<std::uint8_t> seen(interactions.size(), 0U);
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const auto& node = nodes[node_index];
        for (std::uint64_t offset = offsets[node_index];
             offset < offsets[node_index + 1U]; ++offset) {
            const auto interaction_index = indices[static_cast<std::size_t>(offset)];
            if (interaction_index >= interactions.size() ||
                seen[static_cast<std::size_t>(interaction_index)] != 0U) {
                throw std::runtime_error(
                    "CINPKG03 global energy index is not a permutation: " + path.string());
            }
            seen[static_cast<std::size_t>(interaction_index)] = 1U;
            const auto& interaction = interactions[static_cast<std::size_t>(interaction_index)];
            if (interaction.projectile_z != node.projectile_z ||
                interaction.projectile_a != node.projectile_a ||
                interaction.target_z != node.target_z ||
                interaction.target_a != node.target_a ||
                interaction.collision_energy_MeV_per_u !=
                    node.collision_energy_MeV_per_u) {
                throw std::runtime_error(
                    "CINPKG03 global energy node does not match event: " + path.string());
            }
        }
    }
    if (std::any_of(seen.begin(), seen.end(), [](const std::uint8_t value) {
            return value == 0U;
        })) {
        throw std::runtime_error("CINPKG03 global energy index has a missing event: " +
                                 path.string());
    }
}

}  // namespace

InelasticRateV2Table InelasticRateV2Table::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open CINEL02 rate table: " + path.string());
    }
    constexpr std::array<const char*, 6> expected_header{
        "projectile_z", "projectile_a", "target_z", "target_a",
        "energy_MeV_per_u", "macroscopic_cross_section_per_mm"};
    bool header_seen = false;
    std::vector<RateRow> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const auto fields = split_csv(line);
        if (!header_seen) {
            if (fields.size() != expected_header.size()) {
                throw std::runtime_error("CINEL02 rate CSV has an invalid header: " +
                                         path.string());
            }
            for (std::size_t index = 0; index < expected_header.size(); ++index) {
                if (fields[index] != expected_header[index]) {
                    throw std::runtime_error("CINEL02 rate CSV header mismatch: " +
                                             path.string());
                }
            }
            header_seen = true;
            continue;
        }
        if (fields.size() != expected_header.size()) {
            throw std::runtime_error("Incomplete CINEL02 rate row at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        const auto integer = [&](const std::size_t index) {
            const auto value = parse_rate_number(fields[index], path, line_number);
            if (value < 0.0 ||
                value > static_cast<double>(std::numeric_limits<std::int16_t>::max()) ||
                std::floor(value) != value) {
                throw std::runtime_error("Invalid CINEL02 rate identity at " + path.string() + ":" +
                                         std::to_string(line_number));
            }
            return static_cast<int>(value);
        };
        RateRow row{integer(0), integer(1), integer(2), integer(3),
                    parse_rate_number(fields[4], path, line_number),
                    parse_rate_number(fields[5], path, line_number)};
        if (row.projectile_z <= 0 || row.projectile_a < row.projectile_z ||
            row.target_z <= 0 || row.target_a < row.target_z || row.energy < 0.0 ||
            row.cross_section < 0.0) {
            throw std::runtime_error("CINEL02 rate row is outside its physical domain at " +
                                     path.string() + ":" + std::to_string(line_number));
        }
        const auto energy_float = static_cast<float>(row.energy);
        const auto cross_section_float = static_cast<float>(row.cross_section);
        if (!std::isfinite(energy_float) || !std::isfinite(cross_section_float)) {
            throw std::runtime_error("CINEL02 rate value exceeds device float range at " +
                                     path.string() + ":" + std::to_string(line_number));
        }
        rows.push_back(row);
    }
    if (!header_seen || rows.empty()) {
        throw std::runtime_error("CINEL02 rate CSV has no samples: " + path.string());
    }
    std::sort(rows.begin(), rows.end(), [](const RateRow& lhs, const RateRow& rhs) {
        return std::tie(lhs.projectile_z, lhs.projectile_a, lhs.target_z, lhs.target_a,
                        lhs.energy) <
               std::tie(rhs.projectile_z, rhs.projectile_a, rhs.target_z, rhs.target_a,
                        rhs.energy);
    });
    for (std::size_t index = 1; index < rows.size(); ++index) {
        const auto& previous = rows[index - 1];
        const auto& current = rows[index];
        if (std::tie(previous.projectile_z, previous.projectile_a, previous.target_z,
                     previous.target_a, previous.energy) ==
            std::tie(current.projectile_z, current.projectile_a, current.target_z,
                     current.target_a, current.energy)) {
            throw std::runtime_error("Duplicate CINEL02 rate sample in " + path.string());
        }
    }

    InelasticRateV2Table table;
    std::size_t offset = 0;
    while (offset < rows.size()) {
        const auto key = std::tie(rows[offset].projectile_z, rows[offset].projectile_a,
                                  rows[offset].target_z, rows[offset].target_a);
        const auto begin = offset;
        while (offset < rows.size() &&
               std::tie(rows[offset].projectile_z, rows[offset].projectile_a,
                        rows[offset].target_z, rows[offset].target_a) == key) {
            if (offset - begin >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("CINEL02 rate group exceeds uint32 range: " +
                                         path.string());
            }
            table.samples_.push_back(Cinel02RateSample{
                static_cast<float>(rows[offset].energy),
                static_cast<float>(rows[offset].cross_section)});
            ++offset;
        }
        table.groups_.push_back(Cinel02RateGroup{
            static_cast<std::int16_t>(rows[begin].projectile_z),
            static_cast<std::int16_t>(rows[begin].projectile_a),
            static_cast<std::int16_t>(rows[begin].target_z),
            static_cast<std::int16_t>(rows[begin].target_a),
            static_cast<std::uint32_t>(begin),
            static_cast<std::uint32_t>(offset - begin),
            0.0F});
    }
    return table;
}

const std::vector<Cinel02RateGroup>& InelasticRateV2Table::groups() const noexcept {
    return groups_;
}

const std::vector<Cinel02RateSample>& InelasticRateV2Table::samples() const noexcept {
    return samples_;
}

void InelasticRateV2Table::bind_reference_number_densities(
    const std::vector<Cinel02MaterialTarget>& targets) {
    if (targets.empty()) {
        throw std::invalid_argument(
            "CINEL02 material target table is empty");
    }
    for (auto& group : groups_) {
        const auto iterator = std::find_if(
            targets.begin(), targets.end(), [&](const Cinel02MaterialTarget& target) {
                return target.material_section < 0 &&
                       target.target_z == group.target_z &&
                       target.target_a == group.target_a;
            });
        if (iterator == targets.end() ||
            !std::isfinite(iterator->reference_number_density_per_mm3) ||
            iterator->reference_number_density_per_mm3 <= 0.0F) {
            throw std::invalid_argument(
                "CINEL02 rate group has no positive homogeneous reference number density");
        }
        group.reference_number_density_per_mm3 =
            iterator->reference_number_density_per_mm3;
    }
    reference_number_densities_bound_ = true;
}

bool InelasticRateV2Table::has_reference_number_densities() const noexcept {
    return reference_number_densities_bound_;
}

InelasticRateLookup InelasticRateV2Table::lookup(
    const int projectile_z, const int projectile_a, const int target_z,
    const int target_a, const double energy) const noexcept {
    if (!std::isfinite(energy)) {
        return {};
    }
    const auto iterator = std::find_if(
        groups_.begin(), groups_.end(), [&](const Cinel02RateGroup& group) {
            return group.projectile_z == projectile_z && group.projectile_a == projectile_a &&
                   group.target_z == target_z && group.target_a == target_a;
        });
    if (iterator == groups_.end() || iterator->sample_count == 0U) {
        return {};
    }
    const auto begin = samples_.begin() + iterator->sample_offset;
    const auto end = begin + iterator->sample_count;
    // Out-of-campaign energies are uncovered. Returning an endpoint value here
    // would make the GPU silently use a high-energy TOPAS hazard at the end of
    // the range, which is exactly the rate/transport mismatch this table is
    // intended to expose.
    if (energy < begin->energy_MeV_per_u ||
        energy > (end - 1)->energy_MeV_per_u) {
        return {};
    }
    if (energy == begin->energy_MeV_per_u) {
        return {true, begin->macroscopic_cross_section_per_mm};
    }
    if (energy == (end - 1)->energy_MeV_per_u) {
        return {true, (end - 1)->macroscopic_cross_section_per_mm};
    }
    const auto upper = std::upper_bound(
        begin, end, static_cast<float>(energy),
        [](const float value, const Cinel02RateSample& sample) {
            return value < sample.energy_MeV_per_u;
        });
    const auto lower = upper - 1;
    const auto fraction = (energy - lower->energy_MeV_per_u) /
                          (upper->energy_MeV_per_u - lower->energy_MeV_per_u);
    return {true, lower->macroscopic_cross_section_per_mm +
                       fraction * (upper->macroscopic_cross_section_per_mm -
                                   lower->macroscopic_cross_section_per_mm)};
}

double InelasticRateV2Table::interpolate(const int projectile_z,
                                         const int projectile_a,
                                         const int target_z, const int target_a,
                                         const double energy) const noexcept {
    return lookup(projectile_z, projectile_a, target_z, target_a, energy).value_per_mm;
}

InelasticPackageV2Table InelasticPackageV2Table::from_binary(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open CINPKG03 package: " + path.string());
    }
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("Cannot determine CINPKG03 size: " + path.string());
    }
    const auto actual_size = static_cast<std::uint64_t>(end);
    input.seekg(0);
    Cinel02PackageHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        !std::equal(package_magic.begin(), package_magic.end(), header.magic) ||
        header.version != package_version || header.header_size != sizeof(header) ||
        header.endian_marker != endian_marker || header.index_record_size != sizeof(Cinel02CellIndex) ||
        header.interaction_record_size != sizeof(Cinel02InteractionRecord) ||
        header.product_record_size != sizeof(Cinel02ProductRecord) ||
        (header.flags & required_flags) != required_flags ||
        (header.flags & ~known_flags) != 0U || header.cell_count == 0 ||
        header.interaction_count == 0 || header.energy_bin_width_MeV_per_u <= 0.0F ||
        !std::isfinite(header.minimum_energy_MeV_per_u) ||
        !std::isfinite(header.energy_bin_width_MeV_per_u)) {
        throw std::runtime_error("Unsupported or non-authoritative CINPKG03 header: " +
                                 path.string());
    }
    const auto expected_size = static_cast<std::uint64_t>(sizeof(header)) +
                               header.cell_count * sizeof(Cinel02CellIndex);
    if (header.file_size != actual_size || header.file_size < expected_size) {
        throw std::runtime_error("CINPKG03 file-size mismatch: " + path.string());
    }

    InelasticPackageV2Table table;
    table.minimum_energy_MeV_per_u_ = header.minimum_energy_MeV_per_u;
    table.energy_bin_width_MeV_per_u_ = header.energy_bin_width_MeV_per_u;
    table.minimum_events_per_bin_ = header.minimum_events_per_bin;
    const auto uuid_end = std::find(
        std::begin(header.campaign_uuid), std::end(header.campaign_uuid), '\0');
    if (uuid_end == std::end(header.campaign_uuid) ||
        !canonical_uuid(header.campaign_uuid,
                        static_cast<std::size_t>(uuid_end - std::begin(header.campaign_uuid)))) {
        throw std::runtime_error("Invalid CINPKG03 campaign UUID: " + path.string());
    }
    table.campaign_uuid_.assign(header.campaign_uuid, uuid_end);
    read_records(input, table.cells_, header.cell_count, path, "cell");

    std::uint64_t previous_interaction_end = 0;
    std::tuple<int, int, int, int, std::uint32_t> previous_key{
        -1, -1, -1, -1, 0};
    for (const auto& cell : table.cells_) {
        const auto key = std::tuple{static_cast<int>(cell.projectile_z),
                                    static_cast<int>(cell.projectile_a),
                                    static_cast<int>(cell.target_z),
                                    static_cast<int>(cell.target_a), cell.energy_bin};
        if (key <= previous_key || cell.projectile_z <= 0 || cell.projectile_a < cell.projectile_z ||
            cell.target_z <= 0 || cell.target_a < cell.target_z || cell.interaction_count == 0 ||
            cell.interaction_offset != previous_interaction_end ||
            cell.interaction_offset + cell.interaction_count > header.interaction_count ||
            !std::isfinite(cell.energy_lower_MeV_per_u) ||
            !std::isfinite(cell.energy_upper_MeV_per_u) ||
            cell.energy_upper_MeV_per_u <= cell.energy_lower_MeV_per_u) {
            throw std::runtime_error("Invalid CINPKG03 cell index: " + path.string());
        }
        previous_key = key;
        previous_interaction_end = cell.interaction_offset + cell.interaction_count;
    }
    if (previous_interaction_end != header.interaction_count) {
        throw std::runtime_error("CINPKG03 cell ranges do not close: " + path.string());
    }

    read_records(input, table.interactions_, header.interaction_count, path, "interaction");
    table.product_offsets_.resize(table.interactions_.size());
    std::uint64_t expected_product_count = 0;
    for (std::size_t index = 0; index < table.interactions_.size(); ++index) {
        const auto& interaction = table.interactions_[index];
        const auto projectile_identity = classify_inelastic_species(
            interaction.projectile_pdg, interaction.projectile_z,
            interaction.projectile_a, interaction.projectile_charge,
            interaction.projectile_rest_mass, interaction.projectile_excitation);
        const auto parent_identity = classify_inelastic_species(
            interaction.parent_pdg, interaction.parent_z, interaction.parent_a,
            interaction.parent_charge, interaction.parent_rest_mass,
            interaction.parent_excitation);
        if (interaction.projectile_z <= 0 || interaction.projectile_a < interaction.projectile_z ||
            interaction.target_z <= 0 || interaction.target_a < interaction.target_z ||
            !projectile_identity.valid() ||
            projectile_identity.kind != InelasticSpeciesKind::ion ||
            !parent_identity.valid() || parent_identity.kind != InelasticSpeciesKind::ion ||
            !std::isfinite(interaction.collision_energy_MeV) ||
            interaction.collision_energy_MeV < 0.0F ||
            !std::isfinite(interaction.collision_energy_MeV_per_u) ||
            interaction.collision_energy_MeV_per_u < 0.0F ||
            !std::isfinite(interaction.process_local_deposit_MeV) ||
            interaction.process_local_deposit_MeV < 0.0F ||
            !std::isfinite(interaction.nonionizing_deposit_MeV) ||
            interaction.nonionizing_deposit_MeV < 0.0F ||
            !std::isfinite(interaction.track_weight) ||
            std::abs(interaction.track_weight - 1.0F) > 1.0e-6F ||
            !std::isfinite(interaction.parent_weight) ||
            std::abs(interaction.parent_weight - 1.0F) > 1.0e-6F ||
            interaction.parent_pdg != interaction.projectile_pdg ||
            interaction.parent_z != interaction.projectile_z ||
            interaction.parent_a != interaction.projectile_a ||
            !std::isfinite(interaction.projectile_charge) ||
            !std::isfinite(interaction.projectile_rest_mass) ||
            !std::isfinite(interaction.projectile_excitation) ||
            !std::isfinite(interaction.parent_charge) ||
            !std::isfinite(interaction.parent_rest_mass) ||
            !std::isfinite(interaction.parent_excitation) ||
            !std::isfinite(interaction.parent_energy_MeV) ||
            interaction.parent_energy_MeV < 0.0F ||
            (interaction.parent_status != 0 && interaction.parent_status != 2) ||
            interaction.parent_energy_MeV > interaction.collision_energy_MeV + 1.0e-3F ||
            (interaction.parent_status == 2 &&
             interaction.parent_energy_MeV > 1.0e-4F) ||
            (interaction.parent_status == 0 &&
             (std::abs(interaction.parent_charge - interaction.projectile_charge) >
                  1.0e-3F * std::max(1.0F, std::abs(interaction.projectile_charge)) ||
              std::abs(interaction.parent_rest_mass - interaction.projectile_rest_mass) >
                  1.0e-3F * std::max(1.0F, std::abs(interaction.projectile_rest_mass)))) ||
            !unit_direction(interaction.collision_direction_x,
                            interaction.collision_direction_y,
                            interaction.collision_direction_z) ||
            !unit_direction(interaction.parent_direction_x,
                            interaction.parent_direction_y,
                            interaction.parent_direction_z) ||
            !c_string_present(interaction.material_name, sizeof(interaction.material_name)) ||
            !c_string_present(interaction.process_name, sizeof(interaction.process_name)) ||
            !c_string_present(interaction.model_name, sizeof(interaction.model_name))) {
            throw std::runtime_error("Invalid CINPKG03 interaction: " + path.string());
        }
        table.product_offsets_[index] = static_cast<std::uint32_t>(expected_product_count);
        expected_product_count += interaction.direct_product_count;
    }
    if (expected_product_count != header.product_count) {
        throw std::runtime_error("CINPKG03 product count does not close: " + path.string());
    }
    read_records(input, table.products_, header.product_count, path, "product");
    for (std::size_t product_index = 0; product_index < table.products_.size();
         ++product_index) {
        const auto& product = table.products_[product_index];
        const auto identity = classify_inelastic_species(
            product.pdg, product.z, product.a, product.charge,
            product.rest_mass, product.excitation);
        const bool supported_identity = identity.runtime_supported();
        if (!identity.valid() ||
            !std::isfinite(product.kinetic_energy_MeV) || product.kinetic_energy_MeV < 0.0F ||
            !std::isfinite(product.weight) || std::abs(product.weight - 1.0F) > 1.0e-6F ||
            !unit_direction(product.direction_x, product.direction_y, product.direction_z) ||
            !unit_direction(product.local_direction_x, product.local_direction_y,
                            product.local_direction_z) ||
            product.role < 0 || product.role > 2 ||
            (product.role == 2 && supported_identity) ||
            (!supported_identity && product.role != 2)) {
            throw std::runtime_error(
                "Invalid CINPKG03 product/unsupported ledger at product " +
                std::to_string(product_index) + ": pdg=" +
                std::to_string(product.pdg) + ", z=" +
                std::to_string(product.z) + ", a=" +
                std::to_string(product.a) + ", role=" +
                std::to_string(product.role) + ", kind=" +
                std::to_string(static_cast<int>(identity.kind)) + ": " +
                path.string());
        }
    }

    if ((header.flags & cinel02_global_energy_index_flag) != 0U) {
        if (header.reserved == 0U ||
            header.reserved > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) -
                                  1U) {
            throw std::runtime_error("Invalid CINPKG03 global energy-node count: " +
                                     path.string());
        }
        read_records(input, table.energy_nodes_, header.reserved, path, "energy node");
        read_records(input, table.event_offsets_, header.reserved + 1U, path,
                     "energy-node offset");
        read_records(input, table.event_indices_, header.interaction_count, path,
                     "energy event index");
    } else {
        if (header.reserved != 0U) {
            throw std::runtime_error("CINPKG03 legacy header has reserved index metadata: " +
                                     path.string());
        }
        build_energy_index(table.interactions_, table.energy_nodes_, table.event_offsets_,
                           table.event_indices_);
    }
    validate_energy_index(table.energy_nodes_, table.event_offsets_, table.event_indices_,
                          table.interactions_, path);
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("CINPKG03 has trailing bytes: " + path.string());
    }
    return table;
}

const std::vector<Cinel02CellIndex>& InelasticPackageV2Table::cells() const noexcept {
    return cells_;
}
const std::vector<Cinel02InteractionRecord>& InelasticPackageV2Table::interactions() const noexcept {
    return interactions_;
}
const std::vector<Cinel02ProductRecord>& InelasticPackageV2Table::products() const noexcept {
    return products_;
}

const std::vector<Cinel02EnergyNode>& InelasticPackageV2Table::energy_nodes() const noexcept {
    return energy_nodes_;
}

const std::vector<std::uint64_t>& InelasticPackageV2Table::event_offsets() const noexcept {
    return event_offsets_;
}

const std::vector<std::uint64_t>& InelasticPackageV2Table::event_indices() const noexcept {
    return event_indices_;
}

Cinel02DeviceTables InelasticPackageV2Table::make_device_tables() const {
    if (interactions_.size() > std::numeric_limits<std::uint32_t>::max() ||
        products_.size() > std::numeric_limits<std::uint32_t>::max() ||
        energy_nodes_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("CINEL02 device table exceeds uint32 range");
    }

    const auto parent_local_direction = [](const Cinel02InteractionRecord& event) {
        const auto normalize = [](const std::array<float, 3>& value,
                                  const char* label) {
            const auto norm = std::sqrt(value[0] * value[0] +
                                        value[1] * value[1] +
                                        value[2] * value[2]);
            if (!std::isfinite(norm) || norm <= 1.0e-8F) {
                throw std::runtime_error(std::string("CINEL02 invalid ") + label);
            }
            return std::array<float, 3>{value[0] / norm, value[1] / norm,
                                        value[2] / norm};
        };
        const auto axis = normalize(
            {event.collision_direction_x, event.collision_direction_y,
             event.collision_direction_z}, "collision direction");
        const auto global = normalize(
            {event.parent_direction_x, event.parent_direction_y,
             event.parent_direction_z}, "parent direction");
        const std::array<float, 3> reference =
            std::abs(axis[0]) < 0.9F ? std::array<float, 3>{1.0F, 0.0F, 0.0F}
                                    : std::array<float, 3>{0.0F, 1.0F, 0.0F};
        const auto projection = reference[0] * axis[0] +
                                reference[1] * axis[1] +
                                reference[2] * axis[2];
        const auto local_x = normalize(
            {reference[0] - projection * axis[0],
             reference[1] - projection * axis[1],
             reference[2] - projection * axis[2]},
            "projectile-local x axis");
        const std::array<float, 3> local_y{
            axis[1] * local_x[2] - axis[2] * local_x[1],
            axis[2] * local_x[0] - axis[0] * local_x[2],
            axis[0] * local_x[1] - axis[1] * local_x[0]};
        return std::array<float, 3>{
            global[0] * local_x[0] + global[1] * local_x[1] +
                global[2] * local_x[2],
            global[0] * local_y[0] + global[1] * local_y[1] +
                global[2] * local_y[2],
            global[0] * axis[0] + global[1] * axis[1] +
                global[2] * axis[2]};
    };

    Cinel02DeviceTables tables;
    tables.interactions.reserve(interactions_.size());
    for (std::size_t index = 0; index < interactions_.size(); ++index) {
        const auto& event = interactions_[index];
        const auto local = parent_local_direction(event);
        tables.interactions.push_back(Cinel02DeviceInteraction{
            event.collision_energy_MeV_per_u,
            event.process_local_deposit_MeV,
            event.nonionizing_deposit_MeV,
            product_offsets_[index],
            event.direct_product_count,
            event.parent_energy_MeV,
            event.parent_pdg,
            event.parent_z,
            event.parent_a,
            event.parent_charge,
            event.parent_rest_mass,
            event.parent_excitation,
            event.parent_weight,
            local[0], local[1], local[2],
            event.parent_status});
    }
    tables.products.reserve(products_.size());
    for (const auto& product : products_) {
        tables.products.push_back(Cinel02DeviceProduct{
            product.pdg, product.z, product.a, product.charge,
            product.rest_mass, product.excitation, product.weight,
            product.kinetic_energy_MeV,
            product.local_direction_x, product.local_direction_y,
            product.local_direction_z,
            product.direction_x, product.direction_y, product.direction_z,
            product.role});
    }
    tables.energy_nodes = energy_nodes_;
    tables.event_offsets.reserve(event_offsets_.size());
    for (const auto offset : event_offsets_) {
        if (offset > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("CINEL02 device event offset exceeds uint32 range");
        }
        tables.event_offsets.push_back(static_cast<std::uint32_t>(offset));
    }
    tables.event_indices.reserve(event_indices_.size());
    for (const auto index : event_indices_) {
        if (index > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("CINEL02 device event index exceeds uint32 range");
        }
        tables.event_indices.push_back(static_cast<std::uint32_t>(index));
    }
    return tables;
}

Cinel02FixedReplayView InelasticPackageV2Table::fixed_replay(
    const std::uint64_t event_index,
    const Cinel02DeviceTables& compact) const noexcept {
    if (event_index >= interactions_.size() ||
        event_index >= compact.interactions.size()) {
        return {};
    }
    const auto index = static_cast<std::size_t>(event_index);
    const auto& serialized = interactions_[index];
    const auto& device = compact.interactions[index];
    const auto raw_offset = static_cast<std::uint64_t>(product_offsets_[index]);
    const auto compact_offset = static_cast<std::uint64_t>(device.product_offset);
    const auto raw_end = raw_offset + serialized.direct_product_count;
    const auto compact_end = compact_offset + device.product_count;
    if (serialized.direct_product_count != device.product_count ||
        raw_offset != compact_offset || raw_end > products_.size() ||
        compact_end > compact.products.size()) {
        return {};
    }
    return Cinel02FixedReplayView{
        &serialized, &device,
        serialized.direct_product_count == 0U ? nullptr : &products_[raw_offset],
        device.product_count == 0U ? nullptr : &compact.products[compact_offset],
        device.product_count};
}

std::uint64_t InelasticPackageV2Table::find_event(
    const int projectile_z, const int projectile_a, const int target_z, const int target_a,
    const float energy, const float tolerance, const float u01) const noexcept {
    constexpr auto invalid = std::numeric_limits<std::uint64_t>::max();
    if (!std::isfinite(energy) || !std::isfinite(tolerance) || tolerance < 0.0F ||
        !std::isfinite(u01) || u01 < 0.0F || u01 > 1.0F || energy_nodes_.empty() ||
        event_offsets_.size() != energy_nodes_.size() + 1U) {
        return invalid;
    }
    const auto lower_energy = energy - tolerance;
    const auto upper_energy = energy + tolerance;
    if (!std::isfinite(lower_energy) || !std::isfinite(upper_energy)) {
        return invalid;
    }
    const auto lower_key = std::tuple{projectile_z, projectile_a, target_z, target_a,
                                      lower_energy};
    const auto upper_key = std::tuple{projectile_z, projectile_a, target_z, target_a,
                                      upper_energy};
    const auto first = std::lower_bound(
        energy_nodes_.begin(), energy_nodes_.end(), lower_key,
        [](const Cinel02EnergyNode& node, const auto& key) {
            return energy_node_key(node) < key;
        });
    const auto last = std::upper_bound(
        first, energy_nodes_.end(), upper_key,
        [](const auto& key, const Cinel02EnergyNode& node) {
            return key < energy_node_key(node);
        });
    const auto first_node = static_cast<std::size_t>(first - energy_nodes_.begin());
    const auto last_node = static_cast<std::size_t>(last - energy_nodes_.begin());
    if (first_node >= last_node || last_node >= event_offsets_.size()) {
        return invalid;
    }
    const auto first_event = event_offsets_[first_node];
    const auto last_event = event_offsets_[last_node];
    if (first_event >= last_event || last_event > event_indices_.size()) {
        return invalid;
    }
    const auto count = last_event - first_event;
    const auto pick = u01 >= 1.0F
                          ? count - 1U
                          : static_cast<std::uint64_t>(static_cast<double>(u01) * count);
    if (pick >= count) {
        return invalid;
    }
    const auto event_index = event_indices_[static_cast<std::size_t>(first_event + pick)];
    if (event_index >= interactions_.size()) {
        return invalid;
    }
    const auto& event_record = interactions_[static_cast<std::size_t>(event_index)];
    if (event_record.projectile_z != projectile_z || event_record.projectile_a != projectile_a ||
        event_record.target_z != target_z || event_record.target_a != target_a ||
        std::abs(event_record.collision_energy_MeV_per_u - energy) > tolerance) {
        return invalid;
    }
    return event_index;
}

const Cinel02CellIndex* InelasticPackageV2Table::find_cell(
    int projectile_z, int projectile_a, int target_z, int target_a,
    float energy, float tolerance) const noexcept {
    if (!std::isfinite(energy) || !std::isfinite(tolerance) || tolerance < 0.0F) {
        return nullptr;
    }
    const auto event_index = find_event(projectile_z, projectile_a, target_z, target_a,
                                         energy, tolerance, 0.0F);
    if (event_index == std::numeric_limits<std::uint64_t>::max()) {
        return nullptr;
    }
    const auto iterator = std::upper_bound(
        cells_.begin(), cells_.end(), event_index,
        [](const std::uint64_t index, const Cinel02CellIndex& cell) {
            return index < cell.interaction_offset;
        });
    if (iterator == cells_.begin()) {
        return nullptr;
    }
    const auto& cell = *(iterator - 1);
    if (event_index >= cell.interaction_offset &&
        event_index < cell.interaction_offset + cell.interaction_count &&
        cell.projectile_z == projectile_z && cell.projectile_a == projectile_a &&
        cell.target_z == target_z && cell.target_a == target_a) {
        return &cell;
    }
    return nullptr;
}

const Cinel02InteractionRecord* InelasticPackageV2Table::interaction(
    const Cinel02CellIndex& cell, std::uint64_t offset) const noexcept {
    if (offset >= cell.interaction_count || cell.interaction_offset + offset >= interactions_.size()) {
        return nullptr;
    }
    return &interactions_[static_cast<std::size_t>(cell.interaction_offset + offset)];
}

const Cinel02ProductRecord* InelasticPackageV2Table::products_for(
    const Cinel02InteractionRecord& event) const noexcept {
    const auto index = static_cast<std::size_t>(&event - interactions_.data());
    if (index >= product_offsets_.size()) {
        return nullptr;
    }
    const auto offset = product_offsets_[index];
    if (event.direct_product_count == 0) {
        return nullptr;
    }
    if (offset >= products_.size() ||
        static_cast<std::uint64_t>(offset) + event.direct_product_count > products_.size()) {
        return nullptr;
    }
    return products_.data() + offset;
}

std::uint32_t InelasticPackageV2Table::product_offset(
    const Cinel02InteractionRecord& event) const noexcept {
    const auto index = static_cast<std::size_t>(&event - interactions_.data());
    return index < product_offsets_.size() ? product_offsets_[index] : 0U;
}

float InelasticPackageV2Table::minimum_energy_MeV_per_u() const noexcept {
    return minimum_energy_MeV_per_u_;
}
const std::string& InelasticPackageV2Table::campaign_uuid() const noexcept {
    return campaign_uuid_;
}
float InelasticPackageV2Table::energy_bin_width_MeV_per_u() const noexcept {
    return energy_bin_width_MeV_per_u_;
}
std::uint64_t InelasticPackageV2Table::minimum_events_per_bin() const noexcept {
    return minimum_events_per_bin_;
}

int InelasticPackageV2Table::select_target_z(
    float u01, float hydrogen_xs, float oxygen_xs) noexcept {
    if (!(hydrogen_xs > 0.0F) && !(oxygen_xs > 0.0F)) {
        return 0;
    }
    const auto total = std::max(0.0F, hydrogen_xs) + std::max(0.0F, oxygen_xs);
    const auto probability_hydrogen = std::max(0.0F, hydrogen_xs) / total;
    return u01 < probability_hydrogen ? 1 : 8;
}

}  // namespace carbon
