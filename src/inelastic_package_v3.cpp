#include "carbon/inelastic_package_v3.hpp"
#include "carbon/inelastic_identity.hpp"
#include "carbon/sha256.hpp"

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

constexpr std::array<char, 8> package_magic{'C', 'I', 'N', 'P', 'K', 'G', '0', '4'};
constexpr std::uint32_t package_version = 4;
constexpr std::uint32_t endian_marker = 0x01020304U;
constexpr std::uint32_t required_flags = 0x0FU;
constexpr std::uint32_t known_flags = required_flags | cinel03_global_energy_index_flag | cinel03_crc32_checksum_flag;

// Standard IEEE 802.3 CRC32
std::uint32_t compute_crc32(const void* data, std::size_t length, std::uint32_t previous_crc = 0) noexcept {
    static constexpr auto table = []() constexpr {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();

    std::uint32_t c = ~previous_crc;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < length; ++i) {
        c = table[(c ^ bytes[i]) & 0xFF] ^ (c >> 8);
    }
    return ~c;
}

template <typename Record>
void read_records(std::ifstream& input, std::vector<Record>& records,
                  std::uint64_t count, const std::filesystem::path& path,
                  const char* label, std::uint32_t* crc_accumulator = nullptr) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                    sizeof(Record)) {
        throw std::runtime_error(std::string("CINPKG04 ") + label +
                                 " count is too large: " + path.string());
    }
    records.resize(static_cast<std::size_t>(count));
    const auto bytes = static_cast<std::streamsize>(records.size() * sizeof(Record));
    if (bytes > 0) {
        if (!input.read(reinterpret_cast<char*>(records.data()), bytes)) {
            throw std::runtime_error(std::string("Truncated CINPKG04 ") + label +
                                     " table: " + path.string());
        }
        if (crc_accumulator != nullptr) {
            *crc_accumulator = compute_crc32(records.data(), static_cast<std::size_t>(bytes), *crc_accumulator);
        }
    }
}

template <typename Record>
void write_records(std::ofstream& output, const std::vector<Record>& records,
                   std::uint32_t* crc_accumulator = nullptr) {
    const auto bytes = static_cast<std::streamsize>(records.size() * sizeof(Record));
    if (bytes > 0) {
        output.write(reinterpret_cast<const char*>(records.data()), bytes);
        if (crc_accumulator != nullptr) {
            *crc_accumulator = compute_crc32(records.data(), static_cast<std::size_t>(bytes), *crc_accumulator);
        }
    }
}

bool unit_direction(float x, float y, float z) noexcept {
    const auto norm = static_cast<double>(x) * x + static_cast<double>(y) * y +
                       static_cast<double>(z) * z;
    return std::isfinite(norm) && std::abs(norm - 1.0) <= 2.0e-3;
}

bool c_string_present(const char* value, std::size_t size) noexcept {
    return std::find(value, value + size, '\0') != value + size && value[0] != '\0';
}

bool canonical_uuid(const char* value, std::size_t size) noexcept {
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

auto energy_node_key(const Cinel03EnergyNode& node) noexcept {
    return std::tuple{static_cast<int>(node.projectile_z),
                      static_cast<int>(node.projectile_a),
                      static_cast<int>(node.target_element_z),
                      node.collision_energy_MeV_per_u};
}

auto interaction_node_key(const Cinel03InteractionRecord& event) noexcept {
    return std::tuple{static_cast<int>(event.projectile_z),
                      static_cast<int>(event.projectile_a),
                      static_cast<int>(event.target_element_z),
                      event.collision_energy_MeV_per_u};
}

void build_energy_index(
    const std::vector<Cinel03InteractionRecord>& interactions,
    std::vector<Cinel03EnergyNode>& energy_nodes,
    std::vector<std::uint64_t>& event_offsets,
    std::vector<std::uint64_t>& event_indices) {
    energy_nodes.clear();
    event_offsets.clear();
    event_indices.resize(interactions.size());
    std::iota(event_indices.begin(), event_indices.end(), 0U);
    std::stable_sort(
        event_indices.begin(), event_indices.end(),
        [&interactions](const std::uint64_t lhs, const std::uint64_t rhs) {
            const auto& l = interactions[static_cast<std::size_t>(lhs)];
            const auto& r = interactions[static_cast<std::size_t>(rhs)];
            return interaction_node_key(l) < interaction_node_key(r);
        });

    std::size_t cursor = 0;
    while (cursor < event_indices.size()) {
        const auto& first =
            interactions[static_cast<std::size_t>(event_indices[cursor])];
        energy_nodes.push_back(Cinel03EnergyNode{
            first.projectile_z, first.projectile_a, first.target_element_z, 0,
            first.collision_energy_MeV_per_u});
        event_offsets.push_back(static_cast<std::uint64_t>(cursor));
        const auto key = interaction_node_key(first);
        while (cursor < event_indices.size() &&
               interaction_node_key(
                   interactions[static_cast<std::size_t>(event_indices[cursor])]) == key) {
            ++cursor;
        }
    }
    event_offsets.push_back(static_cast<std::uint64_t>(event_indices.size()));
}

void validate_energy_index(
    const std::vector<Cinel03EnergyNode>& energy_nodes,
    const std::vector<std::uint64_t>& event_offsets,
    const std::vector<std::uint64_t>& event_indices,
    const std::vector<Cinel03InteractionRecord>& interactions,
    const std::filesystem::path& path) {
    if (energy_nodes.empty() || event_offsets.size() != energy_nodes.size() + 1U ||
        event_indices.size() != interactions.size() || event_offsets.front() != 0U ||
        event_offsets.back() != event_indices.size()) {
        throw std::runtime_error("Malformed CINPKG04 global energy index: " + path.string());
    }
    std::tuple<int, int, int, float> previous_key{-1, -1, -1, -1.0F};
    for (std::size_t index = 0; index < energy_nodes.size(); ++index) {
        const auto& node = energy_nodes[index];
        const auto key = energy_node_key(node);
        if (key <= previous_key || node.projectile_z <= 0 ||
            node.projectile_a < node.projectile_z ||
            !is_valid_elemental_target(node.target_element_z) ||
            !std::isfinite(node.collision_energy_MeV_per_u) ||
            node.collision_energy_MeV_per_u < 0.0F) {
            throw std::runtime_error("Non-monotonic CINPKG04 energy node at " +
                                     std::to_string(index) + ": " + path.string());
        }
        previous_key = key;
        const auto start = event_offsets[index];
        const auto finish = event_offsets[index + 1U];
        if (start >= finish || finish > event_indices.size()) {
            throw std::runtime_error("Invalid CINPKG04 energy offset at " +
                                     std::to_string(index) + ": " + path.string());
        }
        for (auto event_cursor = start; event_cursor < finish; ++event_cursor) {
            const auto event_index = event_indices[static_cast<std::size_t>(event_cursor)];
            if (event_index >= interactions.size()) {
                throw std::runtime_error("CINPKG04 event index out of range: " + path.string());
            }
            const auto& event = interactions[static_cast<std::size_t>(event_index)];
            if (event.projectile_z != node.projectile_z ||
                event.projectile_a != node.projectile_a ||
                event.target_element_z != node.target_element_z ||
                event.collision_energy_MeV_per_u != node.collision_energy_MeV_per_u) {
                throw std::runtime_error("CINPKG04 energy node event mismatch: " + path.string());
            }
        }
    }
}

} // namespace

InelasticPackageV3Table InelasticPackageV3Table::from_binary(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open CINPKG04 package: " + path.string());
    }
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("Cannot determine CINPKG04 size: " + path.string());
    }
    const auto actual_size = static_cast<std::uint64_t>(end);
    input.seekg(0);

    Cinel03PackageHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("Truncated CINPKG04 header: " + path.string());
    }

    if (!std::equal(package_magic.begin(), package_magic.end(), header.magic) ||
        header.version != package_version || header.header_size != sizeof(header) ||
        header.endian_marker != endian_marker || header.index_record_size != sizeof(Cinel03CellIndex) ||
        header.interaction_record_size != sizeof(Cinel03InteractionRecord) ||
        header.product_record_size != sizeof(Cinel03ProductRecord) ||
        (header.flags & required_flags) != required_flags ||
        (header.flags & ~known_flags) != 0U || header.cell_count == 0 ||
        header.interaction_count == 0 || header.energy_bin_width_MeV_per_u <= 0.0F ||
        !std::isfinite(header.minimum_energy_MeV_per_u) ||
        !std::isfinite(header.energy_bin_width_MeV_per_u)) {
        throw std::runtime_error("Unsupported or non-authoritative CINPKG04 header: " +
                                 path.string());
    }

    const auto min_expected_size = static_cast<std::uint64_t>(sizeof(header)) +
                                   header.cell_count * sizeof(Cinel03CellIndex);
    if (header.file_size != actual_size || header.file_size < min_expected_size) {
        throw std::runtime_error("CINPKG04 file-size mismatch: " + path.string());
    }

    InelasticPackageV3Table table;
    table.minimum_energy_MeV_per_u_ = header.minimum_energy_MeV_per_u;
    table.energy_bin_width_MeV_per_u_ = header.energy_bin_width_MeV_per_u;
    table.minimum_events_per_bin_ = header.minimum_events_per_bin;

    const auto uuid_end = std::find(
        std::begin(header.campaign_uuid), std::end(header.campaign_uuid), '\0');
    if (uuid_end == std::end(header.campaign_uuid) ||
        !canonical_uuid(header.campaign_uuid,
                        static_cast<std::size_t>(uuid_end - std::begin(header.campaign_uuid)))) {
        throw std::runtime_error("Invalid CINPKG04 campaign UUID: " + path.string());
    }
    table.campaign_uuid_.assign(header.campaign_uuid, uuid_end);

    std::uint32_t running_crc = 0;
    read_records(input, table.cells_, header.cell_count, path, "cell", &running_crc);

    std::uint64_t previous_interaction_end = 0;
    std::tuple<int, int, int, std::uint32_t> previous_key{-1, -1, -1, 0};
    for (const auto& cell : table.cells_) {
        const auto key = std::tuple{static_cast<int>(cell.projectile_z),
                                    static_cast<int>(cell.projectile_a),
                                    static_cast<int>(cell.target_element_z),
                                    cell.energy_bin};
        if (key <= previous_key || cell.projectile_z <= 0 || cell.projectile_a < cell.projectile_z ||
            !is_valid_elemental_target(cell.target_element_z) || cell.interaction_count == 0 ||
            cell.interaction_offset != previous_interaction_end ||
            cell.interaction_offset + cell.interaction_count > header.interaction_count ||
            !std::isfinite(cell.energy_lower_MeV_per_u) ||
            !std::isfinite(cell.energy_upper_MeV_per_u) ||
            cell.energy_upper_MeV_per_u <= cell.energy_lower_MeV_per_u) {
            throw std::runtime_error("Invalid or duplicate CINPKG04 cell index: " + path.string());
        }
        previous_key = key;
        previous_interaction_end = cell.interaction_offset + cell.interaction_count;
    }
    if (previous_interaction_end != header.interaction_count) {
        throw std::runtime_error("CINPKG04 cell ranges do not close: " + path.string());
    }

    read_records(input, table.interactions_, header.interaction_count, path, "interaction", &running_crc);
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
            !is_valid_elemental_target(interaction.target_element_z) ||
            (interaction.target_a > 0 && interaction.target_a < interaction.target_element_z) ||
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
            !std::isfinite(interaction.parent_energy_MeV) ||
            interaction.parent_energy_MeV < 0.0F ||
            (interaction.parent_status != 0 && interaction.parent_status != 2) ||
            interaction.parent_energy_MeV > interaction.collision_energy_MeV + 1.0e-3F ||
            (interaction.parent_status == 2 && interaction.parent_energy_MeV > 1.0e-4F) ||
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
            throw std::runtime_error("Invalid CINPKG04 interaction at index " +
                                     std::to_string(index) + ": " + path.string());
        }

        if (interaction.direct_product_count > 64) {
            throw std::runtime_error("CINPKG04 event direct_product_count exceeds limit (64): " +
                                     std::to_string(interaction.direct_product_count) + " at index " + std::to_string(index));
        }
        if (expected_product_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("CINPKG04 product offset exceeds uint32 max at index " + std::to_string(index));
        }
        table.product_offsets_[index] = static_cast<std::uint32_t>(expected_product_count);
        expected_product_count += interaction.direct_product_count;
    }

    if (expected_product_count != header.product_count) {
        throw std::runtime_error("CINPKG04 product count does not close: " + path.string());
    }

    read_records(input, table.products_, header.product_count, path, "product", &running_crc);

    for (std::size_t product_index = 0; product_index < table.products_.size(); ++product_index) {
        const auto& product = table.products_[product_index];
        const auto identity = classify_inelastic_species(
            product.pdg, product.z, product.a, product.charge,
            product.rest_mass, product.excitation);
        const bool supported_identity = identity.valid() && identity.runtime_supported();
        if (!std::isfinite(product.kinetic_energy_MeV) || product.kinetic_energy_MeV < 0.0F ||
            !std::isfinite(product.weight) || std::abs(product.weight - 1.0F) > 1.0e-6F ||
            !unit_direction(product.direction_x, product.direction_y, product.direction_z) ||
            !unit_direction(product.local_direction_x, product.local_direction_y,
                            product.local_direction_z) ||
            product.role < 0 || product.role > 2 ||
            (product.role == 2 && supported_identity) ||
            (product.role != 2 && !supported_identity)) {
            throw std::runtime_error("Invalid CINPKG04 product at index " +
                                     std::to_string(product_index) + ": " + path.string());
        }
    }

    // Energy closure validation per interaction
    for (std::size_t index = 0; index < table.interactions_.size(); ++index) {
        const auto& ev = table.interactions_[index];
        const auto prod_start = table.product_offsets_[index];
        const auto prod_count = ev.direct_product_count;
        double total_product_ke = 0.0;
        for (std::uint32_t p = 0; p < prod_count; ++p) {
            total_product_ke += table.products_[prod_start + p].kinetic_energy_MeV;
        }
        const double total_out = total_product_ke + ev.parent_energy_MeV +
                                 ev.process_local_deposit_MeV + ev.unsupported_product_energy_MeV;
        // Conservative upper bound allowing target disintegration / nuclear release
        const double upper_bound = ev.collision_energy_MeV + std::max(200.0, 0.20 * ev.collision_energy_MeV);
        if (total_out > upper_bound) {
            throw std::runtime_error("CINPKG04 energy closure violation at interaction " +
                                     std::to_string(index) + ": E_out=" + std::to_string(total_out) +
                                     " > limit=" + std::to_string(upper_bound) + " (" + path.string() + ")");
        }
    }

    if ((header.flags & cinel03_global_energy_index_flag) != 0U) {
        if (header.energy_node_count == 0U) {
            throw std::runtime_error("Invalid CINPKG04 global energy-node count: " + path.string());
        }
        read_records(input, table.energy_nodes_, header.energy_node_count, path, "energy node", &running_crc);
        read_records(input, table.event_offsets_, header.energy_node_count + 1U, path, "energy-node offset", &running_crc);
        read_records(input, table.event_indices_, header.interaction_count, path, "energy event index", &running_crc);
    } else {
        build_energy_index(table.interactions_, table.energy_nodes_, table.event_offsets_, table.event_indices_);
    }

    validate_energy_index(table.energy_nodes_, table.event_offsets_, table.event_indices_,
                          table.interactions_, path);

    // Checksum verification
    if ((header.flags & cinel03_crc32_checksum_flag) != 0U) {
        if (header.checksum_crc32 != running_crc) {
            throw std::runtime_error("CINPKG04 checksum mismatch: header 0x" +
                                     std::to_string(header.checksum_crc32) + " vs computed 0x" +
                                     std::to_string(running_crc) + " in " + path.string());
        }
    }

    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("CINPKG04 has trailing bytes: " + path.string());
    }

    // Companion metadata SHA-256 validation
    std::filesystem::path sidecar = path.string() + ".metadata.json";
    if (!std::filesystem::exists(sidecar)) {
        std::string s = path.string();
        if (s.size() > 4 && s.substr(s.size() - 4) == ".bin") {
            sidecar = s.substr(0, s.size() - 4) + ".metadata.json";
        }
    }
    if (std::filesystem::exists(sidecar)) {
        std::ifstream meta_file(sidecar);
        if (meta_file) {
            std::string meta_content((std::istreambuf_iterator<char>(meta_file)),
                                     std::istreambuf_iterator<char>());
            const auto sha_pos = meta_content.find("\"data_sha256\"");
            if (sha_pos != std::string::npos) {
                const auto colon = meta_content.find(':', sha_pos);
                const auto q1 = (colon != std::string::npos) ? meta_content.find('"', colon) : std::string::npos;
                const auto q2 = (q1 != std::string::npos) ? meta_content.find('"', q1 + 1) : std::string::npos;
                if (q1 != std::string::npos && q2 != std::string::npos) {
                    const std::string expected_sha = meta_content.substr(q1 + 1, q2 - q1 - 1);
                    const std::string actual_sha = compute_file_sha256_hex(path);
                    if (actual_sha != expected_sha) {
                        throw std::runtime_error("CINEL03 package SHA-256 mismatch for " + path.string() +
                                                 ": expected " + expected_sha + ", got " + actual_sha);
                    }
                }
            }
        }
    }

    return table;
}

void InelasticPackageV3Table::to_binary(const std::filesystem::path& path) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create CINPKG04 package: " + path.string());
    }

    Cinel03PackageHeader header{};
    std::memcpy(header.magic, package_magic.data(), 8);
    header.version = package_version;
    header.header_size = sizeof(header);
    header.endian_marker = endian_marker;
    header.index_record_size = sizeof(Cinel03CellIndex);
    header.interaction_record_size = sizeof(Cinel03InteractionRecord);
    header.product_record_size = sizeof(Cinel03ProductRecord);
    header.flags = required_flags | cinel03_global_energy_index_flag | cinel03_crc32_checksum_flag;
    header.cell_count = cells_.size();
    header.interaction_count = interactions_.size();
    header.product_count = products_.size();
    header.minimum_energy_MeV_per_u = minimum_energy_MeV_per_u_;
    header.energy_bin_width_MeV_per_u = energy_bin_width_MeV_per_u_;
    header.minimum_events_per_bin = minimum_events_per_bin_;
    header.energy_node_count = energy_nodes_.size();

    std::size_t uuid_len = std::min<std::size_t>(campaign_uuid_.size(), sizeof(header.campaign_uuid) - 1);
    std::memcpy(header.campaign_uuid, campaign_uuid_.data(), uuid_len);
    header.campaign_uuid[uuid_len] = '\0';

    const std::uint64_t total_file_size =
        sizeof(Cinel03PackageHeader) +
        cells_.size() * sizeof(Cinel03CellIndex) +
        interactions_.size() * sizeof(Cinel03InteractionRecord) +
        products_.size() * sizeof(Cinel03ProductRecord) +
        energy_nodes_.size() * sizeof(Cinel03EnergyNode) +
        event_offsets_.size() * sizeof(std::uint64_t) +
        event_indices_.size() * sizeof(std::uint64_t);
    header.file_size = total_file_size;

    // Compute CRC32 over all payload tables
    std::uint32_t payload_crc = 0;
    if (!cells_.empty()) {
        payload_crc = compute_crc32(cells_.data(), cells_.size() * sizeof(Cinel03CellIndex), payload_crc);
    }
    if (!interactions_.empty()) {
        payload_crc = compute_crc32(interactions_.data(), interactions_.size() * sizeof(Cinel03InteractionRecord), payload_crc);
    }
    if (!products_.empty()) {
        payload_crc = compute_crc32(products_.data(), products_.size() * sizeof(Cinel03ProductRecord), payload_crc);
    }
    if (!energy_nodes_.empty()) {
        payload_crc = compute_crc32(energy_nodes_.data(), energy_nodes_.size() * sizeof(Cinel03EnergyNode), payload_crc);
    }
    if (!event_offsets_.empty()) {
        payload_crc = compute_crc32(event_offsets_.data(), event_offsets_.size() * sizeof(std::uint64_t), payload_crc);
    }
    if (!event_indices_.empty()) {
        payload_crc = compute_crc32(event_indices_.data(), event_indices_.size() * sizeof(std::uint64_t), payload_crc);
    }
    header.checksum_crc32 = payload_crc;

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_records(output, cells_);
    write_records(output, interactions_);
    write_records(output, products_);
    write_records(output, energy_nodes_);
    write_records(output, event_offsets_);
    write_records(output, event_indices_);
}

const std::vector<Cinel03CellIndex>& InelasticPackageV3Table::cells() const noexcept {
    return cells_;
}
const std::vector<Cinel03InteractionRecord>& InelasticPackageV3Table::interactions() const noexcept {
    return interactions_;
}
const std::vector<Cinel03ProductRecord>& InelasticPackageV3Table::products() const noexcept {
    return products_;
}
const std::vector<Cinel03EnergyNode>& InelasticPackageV3Table::energy_nodes() const noexcept {
    return energy_nodes_;
}
const std::vector<std::uint64_t>& InelasticPackageV3Table::event_offsets() const noexcept {
    return event_offsets_;
}
const std::vector<std::uint64_t>& InelasticPackageV3Table::event_indices() const noexcept {
    return event_indices_;
}

InelasticPackageV3Table::ChannelDomain InelasticPackageV3Table::channel_domain(
    const int projectile_z, const int projectile_a, const int target_element_z) const noexcept {
    ChannelDomain domain{};
    if (energy_nodes_.empty() || event_offsets_.size() != energy_nodes_.size() + 1U) {
        return domain;
    }
    const auto proj_begin = std::lower_bound(
        energy_nodes_.begin(), energy_nodes_.end(),
        std::tuple{projectile_z, projectile_a, -1000, -1.0F},
        [](const Cinel03EnergyNode& node, const auto& key) {
            return energy_node_key(node) < key;
        });
    const auto proj_end = std::upper_bound(
        proj_begin, energy_nodes_.end(),
        std::tuple{projectile_z, projectile_a, 1000, 1.0e30F},
        [](const auto& key, const Cinel03EnergyNode& node) {
            return key < energy_node_key(node);
        });
    if (proj_begin >= proj_end) {
        return domain;
    }
    domain.found_projectile = true;
    const auto tgt_begin = std::lower_bound(
        proj_begin, proj_end, target_element_z,
        [](const Cinel03EnergyNode& node, const int tz) {
            return node.target_element_z < tz;
        });
    const auto tgt_end = std::upper_bound(
        tgt_begin, proj_end, target_element_z,
        [](const int tz, const Cinel03EnergyNode& node) {
            return tz < node.target_element_z;
        });
    if (tgt_begin >= tgt_end) {
        return domain;
    }
    domain.found_target = true;
    domain.energy_min_MeV_per_u = tgt_begin->collision_energy_MeV_per_u;
    domain.energy_max_MeV_per_u = (tgt_end - 1)->collision_energy_MeV_per_u;
    domain.node_count = static_cast<std::size_t>(tgt_end - tgt_begin);
    float max_gap = 0.0F;
    for (auto it = tgt_begin + 1; it != tgt_end; ++it) {
        const float gap = it->collision_energy_MeV_per_u - (it - 1)->collision_energy_MeV_per_u;
        if (gap > max_gap) {
            max_gap = gap;
        }
    }
    domain.maximum_node_gap_MeV_per_u = max_gap;
    return domain;
}

Cinel03LookupResult InelasticPackageV3Table::lookup_event(
    const int projectile_z, const int projectile_a, const int target_element_z,
    const float energy, const float u_bracket, const float u_event,
    const float max_allowed_gap_MeV_per_u) const noexcept {
    Cinel03LookupResult out{};
    out.query_energy_MeV_per_u = energy;
    if (!std::isfinite(energy) || !std::isfinite(u_bracket) || !std::isfinite(u_event) ||
        energy_nodes_.empty() || event_offsets_.size() != energy_nodes_.size() + 1U) {
        out.status = Cinel03LookupStatus::MissingProjectile;
        return out;
    }

    const auto proj_begin = std::lower_bound(
        energy_nodes_.begin(), energy_nodes_.end(),
        std::tuple{projectile_z, projectile_a, -1000, -1.0F},
        [](const Cinel03EnergyNode& node, const auto& key) {
            return energy_node_key(node) < key;
        });
    const auto proj_end = std::upper_bound(
        proj_begin, energy_nodes_.end(),
        std::tuple{projectile_z, projectile_a, 1000, 1.0e30F},
        [](const auto& key, const Cinel03EnergyNode& node) {
            return key < energy_node_key(node);
        });
    if (proj_begin >= proj_end) {
        out.status = Cinel03LookupStatus::MissingProjectile;
        return out;
    }
    const auto tgt_begin = std::lower_bound(
        proj_begin, proj_end, target_element_z,
        [](const Cinel03EnergyNode& node, const int tz) {
            return node.target_element_z < tz;
        });
    const auto tgt_end = std::upper_bound(
        tgt_begin, proj_end, target_element_z,
        [](const int tz, const Cinel03EnergyNode& node) {
            return tz < node.target_element_z;
        });
    if (tgt_begin >= tgt_end) {
        out.status = Cinel03LookupStatus::MissingTarget;
        return out;
    }

    const auto channel_first = static_cast<std::size_t>(tgt_begin - energy_nodes_.begin());
    const auto channel_last = static_cast<std::size_t>(tgt_end - energy_nodes_.begin());
    const float channel_min = tgt_begin->collision_energy_MeV_per_u;
    const float channel_max = (tgt_end - 1)->collision_energy_MeV_per_u;
    if (!(energy >= channel_min)) {
        out.status = Cinel03LookupStatus::BelowEnergyDomain;
        out.selected_energy_MeV_per_u = channel_min;
        out.absolute_energy_mismatch_MeV_per_u = channel_min - energy;
        return out;
    }
    if (energy > channel_max) {
        out.status = Cinel03LookupStatus::AboveEnergyDomain;
        out.selected_energy_MeV_per_u = channel_max;
        out.absolute_energy_mismatch_MeV_per_u = energy - channel_max;
        return out;
    }

    const auto e1_it = std::lower_bound(
        tgt_begin, tgt_end, energy,
        [](const Cinel03EnergyNode& node, const float value) {
            return node.collision_energy_MeV_per_u < value;
        });
    auto e1 = static_cast<std::size_t>(e1_it - energy_nodes_.begin());
    if (e1 >= channel_last) {
        e1 = channel_last - 1;
    }
    auto e0 = e1;
    if (e1 > channel_first && energy_nodes_[e1].collision_energy_MeV_per_u > energy) {
        e0 = e1 - 1;
    }
    const float e0_energy = energy_nodes_[e0].collision_energy_MeV_per_u;
    const float e1_energy = energy_nodes_[e1].collision_energy_MeV_per_u;

    std::size_t chosen_node = e0;
    if (e1_energy > e0_energy) {
        const float gap = e1_energy - e0_energy;
        if (gap > max_allowed_gap_MeV_per_u) {
            out.status = Cinel03LookupStatus::EnergyGapTooLarge;
            out.selected_energy_MeV_per_u =
                (energy - e0_energy <= e1_energy - energy) ? e0_energy : e1_energy;
            out.absolute_energy_mismatch_MeV_per_u =
                (energy - e0_energy <= e1_energy - energy) ? (energy - e0_energy) : (e1_energy - energy);
            return out;
        }
        const float p_up = (energy - e0_energy) / gap;
        const float ub = u_bracket < 0.0F ? 0.0F : (u_bracket >= 1.0F ? 0.9999999F : u_bracket);
        chosen_node = (ub < p_up) ? e1 : e0;
    }

    const auto first_event = event_offsets_[chosen_node];
    const auto last_event = event_offsets_[chosen_node + 1];
    if (first_event >= last_event || last_event > event_indices_.size()) {
        out.status = Cinel03LookupStatus::EmptyNode;
        out.energy_node_index = static_cast<std::uint32_t>(chosen_node);
        out.selected_energy_MeV_per_u = energy_nodes_[chosen_node].collision_energy_MeV_per_u;
        out.absolute_energy_mismatch_MeV_per_u =
            std::abs(energy - out.selected_energy_MeV_per_u);
        return out;
    }
    const auto count = last_event - first_event;
    const float ue = u_event < 0.0F ? 0.0F : (u_event >= 1.0F ? 0.9999999F : u_event);
    auto pick = static_cast<std::uint64_t>(static_cast<double>(ue) * static_cast<double>(count));
    if (pick >= count) {
        pick = count - 1;
    }
    const auto event_index = event_indices_[static_cast<std::size_t>(first_event + pick)];
    if (event_index >= interactions_.size()) {
        out.status = Cinel03LookupStatus::EmptyNode;
        out.energy_node_index = static_cast<std::uint32_t>(chosen_node);
        return out;
    }
    out.event_index = static_cast<std::uint32_t>(event_index);
    out.energy_node_index = static_cast<std::uint32_t>(chosen_node);
    out.status = Cinel03LookupStatus::Hit;
    out.selected_energy_MeV_per_u = energy_nodes_[chosen_node].collision_energy_MeV_per_u;
    out.absolute_energy_mismatch_MeV_per_u = std::abs(energy - out.selected_energy_MeV_per_u);
    return out;
}

namespace {
const char* cinel03_status_name(const Cinel03LookupStatus status) noexcept {
    switch (status) {
    case Cinel03LookupStatus::Hit: return "Hit";
    case Cinel03LookupStatus::MissingProjectile: return "MissingProjectile";
    case Cinel03LookupStatus::MissingTarget: return "MissingTarget";
    case Cinel03LookupStatus::BelowEnergyDomain: return "BelowEnergyDomain";
    case Cinel03LookupStatus::AboveEnergyDomain: return "AboveEnergyDomain";
    case Cinel03LookupStatus::EnergyGapTooLarge: return "EnergyGapTooLarge";
    case Cinel03LookupStatus::EmptyNode: return "EmptyNode";
    }
    return "Unknown";
}
} // namespace

std::uint64_t InelasticPackageV3Table::find_event(
    const int projectile_z, const int projectile_a, const int target_element_z,
    const float energy, const float tolerance, const float u01,
    const bool audit_mode, std::uint64_t* missing_target_counter) const {
    constexpr auto invalid = std::numeric_limits<std::uint64_t>::max();

    if (!is_valid_elemental_target(target_element_z)) {
        if (audit_mode) {
            if (missing_target_counter != nullptr) (*missing_target_counter)++;
            return invalid;
        }
        throw std::runtime_error("CINEL03: Invalid target element Z=" + std::to_string(target_element_z));
    }

    if (!std::isfinite(energy) || !std::isfinite(tolerance) || tolerance < 0.0F ||
        !std::isfinite(u01) || u01 < 0.0F || u01 > 1.0F || energy_nodes_.empty() ||
        event_offsets_.size() != energy_nodes_.size() + 1U) {
        if (audit_mode) {
            if (missing_target_counter != nullptr) (*missing_target_counter)++;
            return invalid;
        }
        throw std::runtime_error("CINEL03: Invalid lookup arguments or empty table");
    }

    // Strict exact-target + bounded-domain semantics shared with the device.
    // The legacy tolerance window no longer widens the match; the fixed
    // channel gap rule governs. The single uniform drives both the bracket
    // choice and the intra-node pick.
    const auto result = lookup_event(projectile_z, projectile_a, target_element_z,
                                     energy, u01, u01);
    if (result.status == Cinel03LookupStatus::Hit) {
        return result.event_index;
    }
    if (audit_mode) {
        if (missing_target_counter != nullptr) (*missing_target_counter)++;
        return invalid;
    }
    if (result.status == Cinel03LookupStatus::MissingTarget) {
        throw std::runtime_error(
            "CINEL03: Missing target element Z=" + std::to_string(target_element_z) +
            " for projectile Z=" + std::to_string(projectile_z) +
            " A=" + std::to_string(projectile_a) +
            " at E=" + std::to_string(energy) + " MeV/u");
    }
    throw std::runtime_error(
        std::string("CINEL03: ") + cinel03_status_name(result.status) +
        " for projectile Z=" + std::to_string(projectile_z) +
        " A=" + std::to_string(projectile_a) +
        " target Z=" + std::to_string(target_element_z) +
        " at E=" + std::to_string(energy) + " MeV/u");
}

const Cinel03CellIndex* InelasticPackageV3Table::find_cell(
    int projectile_z, int projectile_a, int target_element_z,
    float energy, float tolerance) const noexcept {
    if (!std::isfinite(energy) || !std::isfinite(tolerance) || tolerance < 0.0F) {
        return nullptr;
    }
    std::uint64_t dummy_counter = 0;
    const auto event_index = find_event(projectile_z, projectile_a, target_element_z,
                                        energy, tolerance, 0.0F, true, &dummy_counter);
    if (event_index == std::numeric_limits<std::uint64_t>::max()) {
        return nullptr;
    }
    const auto iterator = std::upper_bound(
        cells_.begin(), cells_.end(), event_index,
        [](const std::uint64_t index, const Cinel03CellIndex& cell) {
            return index < cell.interaction_offset;
        });
    if (iterator == cells_.begin()) {
        return nullptr;
    }
    const auto cell = iterator - 1;
    if (event_index >= cell->interaction_offset + cell->interaction_count) {
        return nullptr;
    }
    return &*cell;
}

const Cinel03InteractionRecord* InelasticPackageV3Table::interaction(
    const Cinel03CellIndex& cell, const std::uint64_t offset) const noexcept {
    if (offset >= cell.interaction_count) {
        return nullptr;
    }
    const auto index = cell.interaction_offset + offset;
    if (index >= interactions_.size()) {
        return nullptr;
    }
    return &interactions_[static_cast<std::size_t>(index)];
}

Cinel03DeviceTables InelasticPackageV3Table::make_device_tables() const {
    auto parent_local_direction = [](const Cinel03InteractionRecord& event) {
        auto normalize = [](std::array<float, 3> value, const char* label) {
            const auto norm = std::sqrt(value[0] * value[0] + value[1] * value[1] +
                                        value[2] * value[2]);
            if (!std::isfinite(norm) || norm <= 1.0e-12F) {
                throw std::runtime_error(std::string("Cannot normalize ") + label);
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

    if (interactions_.size() > std::numeric_limits<std::uint32_t>::max() ||
        products_.size() > std::numeric_limits<std::uint32_t>::max() ||
        energy_nodes_.size() > std::numeric_limits<std::uint32_t>::max() ||
        event_offsets_.size() > std::numeric_limits<std::uint32_t>::max() ||
        event_indices_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("CINEL03 table counts exceed uint32 bounds for device upload");
    }

    Cinel03DeviceTables tables;
    tables.interactions.reserve(interactions_.size());
    for (std::size_t index = 0; index < interactions_.size(); ++index) {
        const auto& event = interactions_[index];
        const std::uint64_t count = event.direct_product_count;
        const std::uint64_t offset = product_offsets_[index];
        if (count > 64) {
            throw std::runtime_error("CINEL03 event direct_product_count exceeds limit (64): " + std::to_string(count));
        }
        if (offset > UINT32_MAX - count || offset + count > products_.size()) {
            throw std::runtime_error("CINEL03 product range overflow or out of bounds: offset=" +
                                     std::to_string(offset) + ", count=" + std::to_string(count) +
                                     ", total_products=" + std::to_string(products_.size()));
        }
        const auto local = parent_local_direction(event);
        tables.interactions.push_back(Cinel03DeviceInteraction{
            event.collision_energy_MeV_per_u,
            event.process_local_deposit_MeV,
            event.nonionizing_deposit_MeV,
            static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(count),
            event.parent_energy_MeV,
            event.parent_pdg,
            event.parent_z,
            event.parent_a,
            event.parent_charge,
            event.parent_rest_mass,
            event.parent_excitation,
            event.parent_weight,
            local[0], local[1], local[2],
            event.target_element_z,
            event.target_a});
    }

    tables.products.reserve(products_.size());
    for (const auto& prod : products_) {
        tables.products.push_back(Cinel03DeviceProduct{
            prod.kinetic_energy_MeV,
            prod.local_direction_x,
            prod.local_direction_y,
            prod.local_direction_z,
            prod.charge,
            prod.rest_mass,
            prod.excitation,
            prod.weight,
            prod.pdg,
            prod.z,
            prod.a,
            prod.role});
    }

    tables.energy_nodes = energy_nodes_;
    tables.event_offsets.assign(event_offsets_.begin(), event_offsets_.end());
    tables.event_indices.assign(event_indices_.begin(), event_indices_.end());
    return tables;
}

Cinel03FixedReplayView InelasticPackageV3Table::fixed_replay(
    const std::uint64_t event_index,
    const Cinel03DeviceTables& compact) const noexcept {
    if (event_index >= interactions_.size() || event_index >= compact.interactions.size()) {
        return {};
    }
    const auto& serialized = interactions_[static_cast<std::size_t>(event_index)];
    const auto& device = compact.interactions[static_cast<std::size_t>(event_index)];
    const auto serialized_offset = product_offsets_[static_cast<std::size_t>(event_index)];
    const auto compact_offset = device.product_offset;

    return Cinel03FixedReplayView{
        &serialized,
        &device,
        serialized.direct_product_count == 0U ? nullptr : &products_[serialized_offset],
        device.direct_product_count == 0U ? nullptr : &compact.products[compact_offset],
        device.direct_product_count};
}

const Cinel03ProductRecord* InelasticPackageV3Table::products_for(
    const Cinel03InteractionRecord& interaction) const noexcept {
    const auto index = static_cast<std::size_t>(&interaction - interactions_.data());
    if (index >= product_offsets_.size() || interaction.direct_product_count == 0U) {
        return nullptr;
    }
    return &products_[product_offsets_[index]];
}

std::uint32_t InelasticPackageV3Table::product_offset(
    const Cinel03InteractionRecord& interaction) const noexcept {
    const auto index = static_cast<std::size_t>(&interaction - interactions_.data());
    if (index >= product_offsets_.size()) {
        return 0;
    }
    return product_offsets_[index];
}

float InelasticPackageV3Table::minimum_energy_MeV_per_u() const noexcept {
    return minimum_energy_MeV_per_u_;
}
float InelasticPackageV3Table::energy_bin_width_MeV_per_u() const noexcept {
    return energy_bin_width_MeV_per_u_;
}
std::uint64_t InelasticPackageV3Table::minimum_events_per_bin() const noexcept {
    return minimum_events_per_bin_;
}
const std::string& InelasticPackageV3Table::campaign_uuid() const noexcept {
    return campaign_uuid_;
}

void InelasticPackageV3Table::set_metadata(
    float min_energy_MeV_per_u, float bin_width_MeV_per_u,
    std::uint64_t min_events_per_bin, const std::string& uuid) {
    minimum_energy_MeV_per_u_ = min_energy_MeV_per_u;
    energy_bin_width_MeV_per_u_ = bin_width_MeV_per_u;
    minimum_events_per_bin_ = min_events_per_bin;
    campaign_uuid_ = uuid;
}

void InelasticPackageV3Table::add_event(
    const Cinel03InteractionRecord& interaction,
    const std::vector<Cinel03ProductRecord>& products) {
    interactions_.push_back(interaction);
    product_offsets_.push_back(static_cast<std::uint32_t>(products_.size()));
    for (const auto& p : products) {
        products_.push_back(p);
    }
}

void InelasticPackageV3Table::finalize() {
    // Build cells and energy nodes
    cells_.clear();
    energy_nodes_.clear();
    event_offsets_.clear();
    event_indices_.clear();

    if (interactions_.empty()) {
        return;
    }

    build_energy_index(interactions_, energy_nodes_, event_offsets_, event_indices_);

    // Build contiguous cells based on energy bins
    std::size_t offset = 0;
    while (offset < interactions_.size()) {
        const auto& first = interactions_[offset];
        const auto p_z = first.projectile_z;
        const auto p_a = first.projectile_a;
        const auto t_z = first.target_element_z;
        const auto e_bin = static_cast<std::uint32_t>(
            std::floor((first.collision_energy_MeV_per_u - minimum_energy_MeV_per_u_) /
                       energy_bin_width_MeV_per_u_));
        const auto begin_offset = offset;

        while (offset < interactions_.size()) {
            const auto& cur = interactions_[offset];
            const auto cur_bin = static_cast<std::uint32_t>(
                std::floor((cur.collision_energy_MeV_per_u - minimum_energy_MeV_per_u_) /
                           energy_bin_width_MeV_per_u_));
            if (cur.projectile_z != p_z || cur.projectile_a != p_a ||
                cur.target_element_z != t_z || cur_bin != e_bin) {
                break;
            }
            ++offset;
        }

        Cinel03CellIndex cell{};
        cell.projectile_z = p_z;
        cell.projectile_a = p_a;
        cell.target_element_z = t_z;
        cell.energy_bin = e_bin;
        cell.interaction_offset = begin_offset;
        cell.interaction_count = offset - begin_offset;
        cell.energy_lower_MeV_per_u = minimum_energy_MeV_per_u_ + e_bin * energy_bin_width_MeV_per_u_;
        cell.energy_upper_MeV_per_u = cell.energy_lower_MeV_per_u + energy_bin_width_MeV_per_u_;
        cells_.push_back(cell);
    }
}

} // namespace carbon
