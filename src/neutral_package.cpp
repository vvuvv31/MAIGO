#include "carbon/neutral_package.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {

struct NeutralFileHeader {
    std::array<char, 8> magic;
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t projectile_size;
    std::uint32_t cross_section_size;
    std::uint32_t interaction_size;
    std::uint32_t product_size;
    std::uint32_t projectile_count;
    std::uint32_t reserved;
    std::uint64_t cross_section_count;
    std::uint64_t interaction_count;
    std::uint64_t product_count;
    std::uint64_t file_size;
};

static_assert(sizeof(NeutralFileHeader) == 72);
constexpr std::array<char, 8> neutral_magic{'C', 'N', 'P', 'K', '0', '0', '1', '\0'};

template <typename T>
void read_records(std::ifstream& input,
                  std::vector<T>& records,
                  std::uint64_t count,
                  const std::filesystem::path& path,
                  const char* label) {
    if (count > static_cast<std::uint64_t>(std::vector<T>().max_size())) {
        throw std::runtime_error(std::string("Neutral ") + label + " count is too large: " +
                                 path.string());
    }
    records.resize(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char*>(records.data()),
               static_cast<std::streamsize>(records.size() * sizeof(T)));
    if (!input) {
        throw std::runtime_error(std::string("Truncated neutral ") + label + " table: " +
                                 path.string());
    }
}

bool is_supported_neutral_pdg(const std::int32_t pdg_id) noexcept {
    return pdg_id == 22 || pdg_id == 2112;
}

}  // namespace

NeutralPackageTable NeutralPackageTable::from_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open neutral package: " + path.string());
    }
    const auto actual_size = input.tellg();
    input.seekg(0);
    NeutralFileHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.magic != neutral_magic || header.version != 1 ||
        header.header_size != sizeof(header) ||
        header.projectile_size != sizeof(NeutralProjectile) ||
        header.cross_section_size != sizeof(NeutralCrossSectionSample) ||
        header.interaction_size != sizeof(NeutralInteraction) ||
        header.product_size != sizeof(ReactionSecondary) || header.projectile_count == 0) {
        throw std::runtime_error("Unsupported neutral package layout: " + path.string());
    }
    if (actual_size < 0 || header.file_size != static_cast<std::uint64_t>(actual_size)) {
        throw std::runtime_error("Neutral package file-size mismatch: " + path.string());
    }

    NeutralPackageTable table;
    read_records(input, table.projectiles_, header.projectile_count, path, "projectile");
    read_records(input, table.cross_sections_, header.cross_section_count, path, "cross-section");
    read_records(input, table.interactions_, header.interaction_count, path, "interaction");
    read_records(input, table.products_, header.product_count, path, "product");

    std::uint64_t expected_xs_offset = 0;
    std::uint64_t expected_interaction_offset = 0;
    std::int32_t previous_pdg = 0;
    for (const auto& projectile : table.projectiles_) {
        if (!is_supported_neutral_pdg(projectile.pdg_id) ||
            (previous_pdg != 0 && projectile.pdg_id <= previous_pdg) ||
            projectile.cross_section_count == 0 || projectile.interaction_count == 0 ||
            projectile.cross_section_offset != expected_xs_offset ||
            projectile.interaction_offset != expected_interaction_offset ||
            static_cast<std::uint64_t>(projectile.cross_section_offset) +
                    projectile.cross_section_count > table.cross_sections_.size() ||
            static_cast<std::uint64_t>(projectile.interaction_offset) +
                    projectile.interaction_count > table.interactions_.size()) {
            throw std::runtime_error("Invalid neutral projectile range: " + path.string());
        }
        float previous_energy = -1.0F;
        for (std::uint32_t index = 0; index < projectile.cross_section_count; ++index) {
            const auto& sample =
                table.cross_sections_[projectile.cross_section_offset + index];
            if (!std::isfinite(sample.energy_MeV) ||
                !std::isfinite(sample.macroscopic_total_per_mm) ||
                sample.energy_MeV < previous_energy ||
                sample.macroscopic_total_per_mm <= 0.0F) {
                throw std::runtime_error("Invalid neutral cross-section sample: " +
                                         path.string());
            }
            previous_energy = sample.energy_MeV;
        }
        previous_energy = -1.0F;
        for (std::uint32_t index = 0; index < projectile.interaction_count; ++index) {
            const auto& interaction =
                table.interactions_[projectile.interaction_offset + index];
            const auto continuation_norm_squared =
                interaction.continuation_direction_x * interaction.continuation_direction_x +
                interaction.continuation_direction_y * interaction.continuation_direction_y +
                interaction.continuation_direction_z * interaction.continuation_direction_z;
            if (!std::isfinite(interaction.incident_energy_MeV) ||
                !std::isfinite(interaction.continuation_energy_MeV) ||
                !std::isfinite(interaction.local_deposit_MeV) ||
                interaction.incident_energy_MeV <= 0.0F ||
                interaction.continuation_energy_MeV < 0.0F ||
                interaction.local_deposit_MeV < 0.0F ||
                interaction.incident_energy_MeV < previous_energy ||
                !std::isfinite(continuation_norm_squared) ||
                std::abs(continuation_norm_squared - 1.0F) > 2.0e-3F ||
                static_cast<std::uint64_t>(interaction.product_offset) +
                        interaction.product_count > table.products_.size()) {
                throw std::runtime_error("Invalid neutral interaction: " + path.string());
            }
            previous_energy = interaction.incident_energy_MeV;
        }
        expected_xs_offset += projectile.cross_section_count;
        expected_interaction_offset += projectile.interaction_count;
        previous_pdg = projectile.pdg_id;
    }
    if (expected_xs_offset != table.cross_sections_.size() ||
        expected_interaction_offset != table.interactions_.size()) {
        throw std::runtime_error("Neutral projectile ranges do not close: " + path.string());
    }

    std::uint64_t expected_product_offset = 0;
    for (const auto& interaction : table.interactions_) {
        if (interaction.product_offset != expected_product_offset) {
            throw std::runtime_error("Neutral product offsets are not contiguous: " +
                                     path.string());
        }
        expected_product_offset += interaction.product_count;
    }
    if (expected_product_offset != table.products_.size()) {
        throw std::runtime_error("Neutral product ranges do not close: " + path.string());
    }
    for (const auto& product : table.products_) {
        const auto direction_norm_squared = product.direction_x * product.direction_x +
                                            product.direction_y * product.direction_y +
                                            product.direction_z * product.direction_z;
        if (!std::isfinite(product.kinetic_energy_MeV) ||
            product.kinetic_energy_MeV < 0.0F ||
            !std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.0F) > 2.0e-3F) {
            throw std::runtime_error("Invalid neutral product: " + path.string());
        }
    }
    return table;
}

const std::vector<NeutralProjectile>& NeutralPackageTable::projectiles() const noexcept {
    return projectiles_;
}
const std::vector<NeutralCrossSectionSample>& NeutralPackageTable::cross_sections()
    const noexcept {
    return cross_sections_;
}
const std::vector<NeutralInteraction>& NeutralPackageTable::interactions() const noexcept {
    return interactions_;
}
const std::vector<ReactionSecondary>& NeutralPackageTable::products() const noexcept {
    return products_;
}
const NeutralProjectile* NeutralPackageTable::find_projectile(int pdg_id) const noexcept {
    const auto iterator = std::lower_bound(
        projectiles_.begin(), projectiles_.end(), pdg_id,
        [](const NeutralProjectile& projectile, const int key) {
            return projectile.pdg_id < key;
        });
    if (iterator == projectiles_.end() || iterator->pdg_id != pdg_id) {
        return nullptr;
    }
    return &*iterator;
}

}  // namespace carbon
