#include "carbon/cascade_package.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {

struct CascadeFileHeader {
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

struct ReactionSecondaryV1 {
    std::int32_t pdg_id;
    std::int16_t atomic_number;
    std::int16_t mass_number;
    float kinetic_energy_MeV;
    float direction_z;
};

struct CascadeInteractionV2 {
    float incident_energy_MeV_per_u;
    std::uint32_t product_offset;
    std::uint32_t product_count;
};

static_assert(sizeof(ReactionSecondaryV1) == 16);
static_assert(sizeof(CascadeInteractionV2) == 12);

static_assert(sizeof(CascadeFileHeader) == 72);
constexpr std::array<char, 8> cascade_magic{'C', 'C', 'A', 'S', '0', '0', '1', '\0'};

template <typename T>
void read_records(std::ifstream& input,
                  std::vector<T>& records,
                  std::uint64_t count,
                  const std::filesystem::path& path,
                  const char* label) {
    if (count > static_cast<std::uint64_t>(std::vector<T>().max_size())) {
        throw std::runtime_error(std::string("Cascade ") + label + " count is too large: " +
                                 path.string());
    }
    records.resize(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char*>(records.data()),
               static_cast<std::streamsize>(records.size() * sizeof(T)));
    if (!input) {
        throw std::runtime_error(std::string("Truncated cascade ") + label + " table: " +
                                 path.string());
    }
}

}  // namespace

CascadePackageTable CascadePackageTable::from_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open cascade package: " + path.string());
    }
    const auto actual_size = input.tellg();
    input.seekg(0);
    CascadeFileHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    const auto legacy_v1 = header.version == 1;
    const auto legacy_interactions = header.version <= 2;
    const auto expected_product_size =
        legacy_v1 ? sizeof(ReactionSecondaryV1) : sizeof(ReactionSecondary);
    const auto expected_interaction_size =
        legacy_interactions ? sizeof(CascadeInteractionV2)
                            : sizeof(CascadeInteraction);
    if (!input || header.magic != cascade_magic ||
        (header.version != 1 && header.version != 2 && header.version != 3) ||
        header.header_size != sizeof(header) ||
        header.projectile_size != sizeof(CascadeProjectile) ||
        header.cross_section_size != sizeof(CascadeCrossSectionSample) ||
        header.interaction_size != expected_interaction_size ||
        header.product_size != expected_product_size || header.projectile_count == 0) {
        throw std::runtime_error("Unsupported cascade package layout: " + path.string());
    }
    if (actual_size < 0 || header.file_size != static_cast<std::uint64_t>(actual_size)) {
        throw std::runtime_error("Cascade package file-size mismatch: " + path.string());
    }

    CascadePackageTable table;
    read_records(input, table.projectiles_, header.projectile_count, path, "projectile");
    read_records(input, table.cross_sections_, header.cross_section_count, path, "cross-section");
    if (legacy_interactions) {
        std::vector<CascadeInteractionV2> legacy_records;
        read_records(input, legacy_records, header.interaction_count, path,
                     "interaction");
        table.interactions_.reserve(legacy_records.size());
        const auto missing = std::numeric_limits<float>::quiet_NaN();
        for (const auto& interaction : legacy_records) {
            table.interactions_.push_back(CascadeInteraction{
                interaction.incident_energy_MeV_per_u, missing,
                interaction.product_offset, interaction.product_count});
        }
    } else {
        read_records(input, table.interactions_, header.interaction_count, path,
                     "interaction");
    }
    if (legacy_v1) {
        std::vector<ReactionSecondaryV1> legacy_products;
        read_records(input, legacy_products, header.product_count, path, "product");
        table.products_.reserve(legacy_products.size());
        const auto missing = std::numeric_limits<float>::quiet_NaN();
        for (const auto& product : legacy_products) {
            table.products_.push_back(ReactionSecondary{
                product.pdg_id, product.atomic_number, product.mass_number,
                product.kinetic_energy_MeV, missing, missing, product.direction_z});
        }
    } else {
        read_records(input, table.products_, header.product_count, path, "product");
    }

    std::uint64_t expected_xs_offset = 0;
    std::uint64_t expected_interaction_offset = 0;
    int previous_z = -1;
    int previous_a = -1;
    for (const auto& projectile : table.projectiles_) {
        const bool ordered = projectile.atomic_number > previous_z ||
                             (projectile.atomic_number == previous_z &&
                              projectile.mass_number > previous_a);
        if (!ordered || projectile.atomic_number <= 0 || projectile.mass_number <= 0 ||
            projectile.cross_section_count == 0 || projectile.interaction_count == 0 ||
            projectile.cross_section_offset != expected_xs_offset ||
            projectile.interaction_offset != expected_interaction_offset ||
            static_cast<std::uint64_t>(projectile.cross_section_offset) +
                    projectile.cross_section_count > table.cross_sections_.size() ||
            static_cast<std::uint64_t>(projectile.interaction_offset) +
                    projectile.interaction_count > table.interactions_.size()) {
            throw std::runtime_error("Invalid cascade projectile range: " + path.string());
        }
        float previous_energy = -1.0F;
        for (std::uint32_t index = 0; index < projectile.cross_section_count; ++index) {
            const auto& sample = table.cross_sections_[projectile.cross_section_offset + index];
            if (!std::isfinite(sample.energy_MeV_per_u) ||
                !std::isfinite(sample.macroscopic_cross_section_per_mm) ||
                sample.energy_MeV_per_u < previous_energy ||
                sample.macroscopic_cross_section_per_mm <= 0.0F) {
                throw std::runtime_error("Invalid cascade cross-section sample: " + path.string());
            }
            previous_energy = sample.energy_MeV_per_u;
        }
        previous_energy = -1.0F;
        auto previous_energy_bin = -1;
        auto previous_depth_bin = -1;
        for (std::uint32_t index = 0; index < projectile.interaction_count; ++index) {
            const auto& interaction = table.interactions_[projectile.interaction_offset + index];
            const auto energy_bin = static_cast<int>(
                interaction.incident_energy_MeV_per_u / 2.0F);
            const auto depth_bin = std::isfinite(interaction.depth_mm)
                                       ? static_cast<int>(interaction.depth_mm / 10.0F)
                                       : -1;
            const auto conditioned_ordered =
                legacy_interactions ||
                energy_bin > previous_energy_bin ||
                (energy_bin == previous_energy_bin &&
                 depth_bin >= previous_depth_bin);
            // Depth may be negative for long low-density phantoms (e.g. lung)
            // when the cascade n-tuple records world coordinates upstream of
            // the geometric midplane. Energy-only cascade sampling still uses
            // these packages; only finite energy/depth are required.
            if (!std::isfinite(interaction.incident_energy_MeV_per_u) ||
                (!legacy_interactions && !std::isfinite(interaction.depth_mm)) ||
                (legacy_interactions &&
                 interaction.incident_energy_MeV_per_u < previous_energy) ||
                !conditioned_ordered ||
                static_cast<std::uint64_t>(interaction.product_offset) +
                        interaction.product_count > table.products_.size()) {
                throw std::runtime_error("Invalid cascade interaction: " + path.string());
            }
            previous_energy = interaction.incident_energy_MeV_per_u;
            previous_energy_bin = energy_bin;
            previous_depth_bin = depth_bin;
        }
        expected_xs_offset += projectile.cross_section_count;
        expected_interaction_offset += projectile.interaction_count;
        previous_z = projectile.atomic_number;
        previous_a = projectile.mass_number;
    }
    if (expected_xs_offset != table.cross_sections_.size() ||
        expected_interaction_offset != table.interactions_.size()) {
        throw std::runtime_error("Cascade projectile ranges do not close: " + path.string());
    }

    std::uint64_t expected_product_offset = 0;
    for (const auto& interaction : table.interactions_) {
        if (interaction.product_offset != expected_product_offset) {
            throw std::runtime_error("Cascade product offsets are not contiguous: " + path.string());
        }
        expected_product_offset += interaction.product_count;
    }
    if (expected_product_offset != table.products_.size()) {
        throw std::runtime_error("Cascade product ranges do not close: " + path.string());
    }
    for (const auto& product : table.products_) {
        const auto has_x = std::isfinite(product.direction_x);
        const auto has_y = std::isfinite(product.direction_y);
        const auto direction_norm_squared = product.direction_x * product.direction_x +
                                            product.direction_y * product.direction_y +
                                            product.direction_z * product.direction_z;
        if (!std::isfinite(product.kinetic_energy_MeV) ||
            !std::isfinite(product.direction_z) || product.kinetic_energy_MeV < 0.0F ||
            product.direction_z < -1.0001F || product.direction_z > 1.0001F ||
            has_x != has_y ||
            (has_x && (!std::isfinite(direction_norm_squared) ||
                       std::abs(direction_norm_squared - 1.0F) > 2.0e-3F))) {
            throw std::runtime_error("Invalid cascade product: " + path.string());
        }
    }
    return table;
}

const std::vector<CascadeProjectile>& CascadePackageTable::projectiles() const noexcept {
    return projectiles_;
}
const std::vector<CascadeCrossSectionSample>& CascadePackageTable::cross_sections() const noexcept {
    return cross_sections_;
}
const std::vector<CascadeInteraction>& CascadePackageTable::interactions() const noexcept {
    return interactions_;
}
const std::vector<ReactionSecondary>& CascadePackageTable::products() const noexcept {
    return products_;
}
const CascadeProjectile* CascadePackageTable::find_projectile(int atomic_number,
                                                              int mass_number) const noexcept {
    const auto iterator = std::lower_bound(
        projectiles_.begin(), projectiles_.end(), std::pair{atomic_number, mass_number},
        [](const CascadeProjectile& projectile, const std::pair<int, int>& key) {
            return std::pair{static_cast<int>(projectile.atomic_number),
                             static_cast<int>(projectile.mass_number)} < key;
        });
    if (iterator == projectiles_.end() || iterator->atomic_number != atomic_number ||
        iterator->mass_number != mass_number) {
        return nullptr;
    }
    return &*iterator;
}

}  // namespace carbon
