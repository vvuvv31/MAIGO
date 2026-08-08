#include "carbon/reaction_package.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carbon {
namespace {

constexpr std::array<char, 8> expected_magic{'C', 'R', 'P', 'K', 'G', '0', '1', '\0'};

struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t energy_bin_record_size;
    std::uint32_t reaction_record_size;
    std::uint32_t secondary_record_size;
    std::uint32_t energy_bin_count;
    float minimum_energy_MeV_per_u;
    float energy_bin_width_MeV_per_u;
    std::uint64_t reaction_count;
    std::uint64_t secondary_count;
    std::uint64_t expected_file_size;
};
struct ReactionSecondaryV1 {
    std::int32_t pdg_id;
    std::int16_t atomic_number;
    std::int16_t mass_number;
    float kinetic_energy_MeV;
    float direction_z;
};

static_assert(sizeof(ReactionSecondaryV1) == 16);

static_assert(sizeof(BinaryHeader) == 64);
static_assert(sizeof(ReactionEnergyBin) == 8);
static_assert(sizeof(ReactionPackage) == 16);
static_assert(sizeof(ReactionSecondary) == 24);
static_assert(std::is_trivially_copyable_v<ReactionEnergyBin>);
static_assert(std::is_trivially_copyable_v<ReactionPackage>);
static_assert(std::is_trivially_copyable_v<ReactionSecondary>);

template <typename Record>
void read_records(std::ifstream& input,
                  std::vector<Record>& records,
                  std::uint64_t count,
                  const std::filesystem::path& path,
                  const char* label) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                    sizeof(Record)) {
        throw std::runtime_error(std::string("Reaction package ") + label +
                                 " count is too large: " + path.string());
    }
    records.resize(static_cast<std::size_t>(count));
    const auto bytes = static_cast<std::streamsize>(records.size() * sizeof(Record));
    if (bytes > 0 && !input.read(reinterpret_cast<char*>(records.data()), bytes)) {
        throw std::runtime_error(std::string("Truncated reaction package ") + label +
                                 " table: " + path.string());
    }
}

}  // namespace

ReactionPackageTable ReactionPackageTable::from_binary(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Reaction packages require a little-endian host");
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open reaction package: " + path.string());
    }
    const auto file_size_position = input.tellg();
    if (file_size_position < 0) {
        throw std::runtime_error("Cannot determine reaction package size: " + path.string());
    }
    const auto file_size = static_cast<std::uint64_t>(file_size_position);
    input.seekg(0);

    BinaryHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("Truncated reaction package header: " + path.string());
    }
    if (!std::equal(expected_magic.begin(), expected_magic.end(), header.magic)) {
        throw std::runtime_error("Invalid reaction package magic: " + path.string());
    }
    const auto legacy_v1 = header.version == 1;
    const auto expected_secondary_size =
        legacy_v1 ? sizeof(ReactionSecondaryV1) : sizeof(ReactionSecondary);
    if ((header.version != 1 && header.version != 2) ||
        header.header_size != sizeof(BinaryHeader) ||
        header.energy_bin_record_size != sizeof(ReactionEnergyBin) ||
        header.reaction_record_size != sizeof(ReactionPackage) ||
        header.secondary_record_size != expected_secondary_size) {
        throw std::runtime_error("Unsupported reaction package layout: " + path.string());
    }
    if (header.energy_bin_count == 0 || header.reaction_count == 0 ||
        header.energy_bin_width_MeV_per_u <= 0.0F ||
        !std::isfinite(header.minimum_energy_MeV_per_u) ||
        !std::isfinite(header.energy_bin_width_MeV_per_u)) {
        throw std::runtime_error("Invalid reaction package header values: " + path.string());
    }
    const auto computed_file_size =
        static_cast<std::uint64_t>(sizeof(BinaryHeader)) +
        static_cast<std::uint64_t>(header.energy_bin_count) * sizeof(ReactionEnergyBin) +
        header.reaction_count * sizeof(ReactionPackage) +
        header.secondary_count * header.secondary_record_size;
    if (header.expected_file_size != computed_file_size || file_size != computed_file_size) {
        throw std::runtime_error("Reaction package file-size mismatch: " + path.string());
    }

    ReactionPackageTable table;
    table.minimum_energy_MeV_per_u_ = header.minimum_energy_MeV_per_u;
    table.energy_bin_width_MeV_per_u_ = header.energy_bin_width_MeV_per_u;
    read_records(input, table.energy_bins_, header.energy_bin_count, path, "energy-bin");
    read_records(input, table.reactions_, header.reaction_count, path, "reaction");
    if (legacy_v1) {
        std::vector<ReactionSecondaryV1> legacy_secondaries;
        read_records(input, legacy_secondaries, header.secondary_count, path, "secondary");
        table.secondaries_.reserve(legacy_secondaries.size());
        const auto missing = std::numeric_limits<float>::quiet_NaN();
        for (const auto& secondary : legacy_secondaries) {
            table.secondaries_.push_back(ReactionSecondary{
                secondary.pdg_id, secondary.atomic_number, secondary.mass_number,
                secondary.kinetic_energy_MeV, missing, missing, secondary.direction_z});
        }
    } else {
        read_records(input, table.secondaries_, header.secondary_count, path, "secondary");
    }

    std::uint64_t expected_reaction_offset = 0;
    for (std::size_t bin_index = 0; bin_index < table.energy_bins_.size(); ++bin_index) {
        const auto& bin = table.energy_bins_[bin_index];
        if (bin.reaction_count == 0 || bin.reaction_offset != expected_reaction_offset ||
            static_cast<std::uint64_t>(bin.reaction_offset) + bin.reaction_count >
                table.reactions_.size()) {
            throw std::runtime_error("Invalid reaction energy-bin range: " + path.string());
        }
        const auto lower_energy = table.minimum_energy_MeV_per_u_ +
                                  static_cast<float>(bin_index) *
                                      table.energy_bin_width_MeV_per_u_;
        const auto upper_energy = lower_energy + table.energy_bin_width_MeV_per_u_;
        for (std::uint32_t offset = 0; offset < bin.reaction_count; ++offset) {
            const auto energy = table.reactions_[bin.reaction_offset + offset]
                                    .incident_energy_MeV_per_u;
            if (!std::isfinite(energy) || energy < lower_energy || energy >= upper_energy) {
                throw std::runtime_error("Reaction is outside its energy bin: " + path.string());
            }
        }
        expected_reaction_offset += bin.reaction_count;
    }
    if (expected_reaction_offset != table.reactions_.size()) {
        throw std::runtime_error("Reaction energy bins do not cover the table: " + path.string());
    }

    std::uint64_t expected_secondary_offset = 0;
    for (const auto& reaction : table.reactions_) {
        // Depth may be negative for long low-density material packages (lung)
        // when the TOPAS n-tuple records world coordinates. Offsets still must
        // form a contiguous secondary range.
        if (!std::isfinite(reaction.reaction_depth_mm) ||
            reaction.secondary_offset != expected_secondary_offset ||
            static_cast<std::uint64_t>(reaction.secondary_offset) + reaction.secondary_count >
                table.secondaries_.size()) {
            throw std::runtime_error("Invalid reaction secondary range: " + path.string());
        }
        expected_secondary_offset += reaction.secondary_count;
    }
    if (expected_secondary_offset != table.secondaries_.size()) {
        throw std::runtime_error("Reaction ranges do not cover the secondary table: " +
                                 path.string());
    }
    for (const auto& secondary : table.secondaries_) {
        const auto has_x = std::isfinite(secondary.direction_x);
        const auto has_y = std::isfinite(secondary.direction_y);
        const auto direction_norm_squared =
            secondary.direction_x * secondary.direction_x +
            secondary.direction_y * secondary.direction_y +
            secondary.direction_z * secondary.direction_z;
        if (secondary.atomic_number < 0 || secondary.mass_number < 0 ||
            !std::isfinite(secondary.kinetic_energy_MeV) ||
            secondary.kinetic_energy_MeV < 0.0F || !std::isfinite(secondary.direction_z) ||
            secondary.direction_z < -1.0001F || secondary.direction_z > 1.0001F ||
            has_x != has_y ||
            (has_x && (!std::isfinite(direction_norm_squared) ||
                       std::abs(direction_norm_squared - 1.0F) > 2.0e-3F))) {
            throw std::runtime_error("Invalid reaction secondary value: " + path.string());
        }
    }
    return table;
}

const std::vector<ReactionEnergyBin>& ReactionPackageTable::energy_bins() const noexcept {
    return energy_bins_;
}

const std::vector<ReactionPackage>& ReactionPackageTable::reactions() const noexcept {
    return reactions_;
}

const std::vector<ReactionSecondary>& ReactionPackageTable::secondaries() const noexcept {
    return secondaries_;
}

float ReactionPackageTable::minimum_energy_MeV_per_u() const noexcept {
    return minimum_energy_MeV_per_u_;
}

float ReactionPackageTable::energy_bin_width_MeV_per_u() const noexcept {
    return energy_bin_width_MeV_per_u_;
}

std::size_t ReactionPackageTable::energy_bin_index(float energy_MeV_per_u) const noexcept {
    if (!std::isfinite(energy_MeV_per_u) || energy_MeV_per_u <= minimum_energy_MeV_per_u_) {
        return 0;
    }
    const auto floating_index =
        (energy_MeV_per_u - minimum_energy_MeV_per_u_) / energy_bin_width_MeV_per_u_;
    const auto index = static_cast<std::size_t>(floating_index);
    return std::min(index, energy_bins_.size() - 1);
}

const ReactionEnergyBin& ReactionPackageTable::energy_bin(float energy_MeV_per_u) const noexcept {
    return energy_bins_[energy_bin_index(energy_MeV_per_u)];
}

}  // namespace carbon
