#include "carbon/inelastic_package_v3.hpp"
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

constexpr std::array<char, 8> package_magic{'C', 'I', 'N', 'P', 'K', 'G', '0', '4'};
constexpr std::uint32_t package_version = 4;
constexpr std::uint32_t endian_marker = 0x01020304U;
constexpr std::uint32_t required_flags = 0x0FU;
constexpr std::uint32_t known_flags = required_flags | cinel03_global_energy_index_flag | cinel03_crc32_checksum_flag;

// Standard IEEE 802.3 CRC32, slicing-by-8. Identical polynomial (0xEDB88320),
// initial value (0), final inversion and incremental concatenation semantics as
// the previous byte-at-a-time version; only the number of table lookups per
// byte changes. `previous_crc` is the finalized CRC of the preceding chunks.
std::uint32_t compute_crc32(const void* data, std::size_t length, std::uint32_t previous_crc = 0) noexcept {
    static constexpr auto tables = []() constexpr {
        std::array<std::array<std::uint32_t, 256>, 8> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            t[0][i] = c;
        }
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = t[0][i];
            for (int k = 1; k < 8; ++k) {
                c = t[0][c & 0xFF] ^ (c >> 8);
                t[k][i] = c;
            }
        }
        return t;
    }();

    std::uint32_t c = ~previous_crc;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    while (length >= 8) {
        const std::uint32_t lo =
            static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8) |
            (static_cast<std::uint32_t>(bytes[2]) << 16) |
            (static_cast<std::uint32_t>(bytes[3]) << 24);
        const std::uint32_t hi =
            static_cast<std::uint32_t>(bytes[4]) |
            (static_cast<std::uint32_t>(bytes[5]) << 8) |
            (static_cast<std::uint32_t>(bytes[6]) << 16) |
            (static_cast<std::uint32_t>(bytes[7]) << 24);
        const std::uint32_t x = c ^ lo;
        c = tables[7][x & 0xFF] ^ tables[6][(x >> 8) & 0xFF] ^
            tables[5][(x >> 16) & 0xFF] ^ tables[4][(x >> 24) & 0xFF] ^
            tables[3][hi & 0xFF] ^ tables[2][(hi >> 8) & 0xFF] ^
            tables[1][(hi >> 16) & 0xFF] ^ tables[0][(hi >> 24) & 0xFF];
        bytes += 8;
        length -= 8;
    }
    while (length > 0) {
        c = tables[0][(c ^ *bytes) & 0xFF] ^ (c >> 8);
        ++bytes;
        --length;
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
        (header.flags & (required_flags | cinel03_global_energy_index_flag)) !=
            (required_flags | cinel03_global_energy_index_flag) ||
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
        // products_ contains supported and unsupported direct products.  The
        // interaction's unsupported_product_energy_MeV is the subtotal of the
        // latter, so adding it here would count those products twice.
        const double total_out = total_product_ke + ev.parent_energy_MeV +
                                 ev.process_local_deposit_MeV;
        // Conservative upper bound allowing target disintegration / nuclear release
        const double upper_bound = ev.collision_energy_MeV + std::max(200.0, 0.20 * ev.collision_energy_MeV);
        if (total_out > upper_bound) {
            throw std::runtime_error("CINPKG04 energy closure violation at interaction " +
                                     std::to_string(index) + ": E_out=" + std::to_string(total_out) +
                                     " > limit=" + std::to_string(upper_bound) + " (" + path.string() + ")");
        }
    }

        if (header.energy_node_count == 0U) {
            throw std::runtime_error("Invalid CINPKG04 global energy-node count: " + path.string());
        }
        read_records(input, table.energy_nodes_, header.energy_node_count, path, "energy node", &running_crc);
        read_records(input, table.event_offsets_, header.energy_node_count + 1U, path, "energy-node offset", &running_crc);
        read_records(input, table.event_indices_, header.interaction_count, path, "energy event index", &running_crc);

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

    return table;
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

Cinel03DeviceTables InelasticPackageV3Table::make_device_tables() const {
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

const std::string& InelasticPackageV3Table::campaign_uuid() const noexcept {
    return campaign_uuid_;
}

} // namespace carbon
