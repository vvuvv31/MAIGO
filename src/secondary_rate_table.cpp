#include "carbon/secondary_rate_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace carbon {

SecondaryRateTable SecondaryRateTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    (void)metadata_path;
    std::ifstream file(binary_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open secondary rate binary: " + binary_path.string());
    }

    SecondaryRateHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(SecondaryRateHeader));
    if (!file) {
        throw std::runtime_error("Failed to read header from: " + binary_path.string());
    }

    if (std::memcmp(header.magic, "SCHN2RAT", 8) != 0) {
        throw std::runtime_error("Invalid magic in secondary rate binary: " + binary_path.string());
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported version in secondary rate binary: " + std::to_string(header.version));
    }

    SecondaryRateTable table;
    table.num_projectiles_ = header.num_projectiles;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    table.projectiles_.resize(header.num_projectiles);
    file.read(reinterpret_cast<char*>(table.projectiles_.data()), header.num_projectiles * sizeof(SecondaryProjectileKey));

    file.read(reinterpret_cast<char*>(table.canonical_targets_.data()), 13 * sizeof(int32_t));

    const std::size_t num_partial = header.num_projectiles * kSecondaryNumSections * kSecondaryNumTargets * kSecondaryNumEnergies;
    table.mass_partial_rates_.resize(num_partial);
    file.read(reinterpret_cast<char*>(table.mass_partial_rates_.data()), num_partial * sizeof(double));

    const std::size_t num_total = header.num_projectiles * kSecondaryNumSections * kSecondaryNumEnergies;
    table.mass_total_rates_.resize(num_total);
    file.read(reinterpret_cast<char*>(table.mass_total_rates_.data()), num_total * sizeof(double));

    if (!file) {
        throw std::runtime_error("Failed to read full payload from secondary rate binary: " + binary_path.string());
    }

    return table;
}

int SecondaryRateTable::projectile_index(int proj_z, int proj_a) const noexcept {
    for (std::size_t i = 0; i < projectiles_.size(); ++i) {
        if (projectiles_[i].z == proj_z && projectiles_[i].a == proj_a) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::size_t SecondaryRateTable::target_index_from_z(int target_z) {
    constexpr std::array<int32_t, 13> canonical_z = {
        1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
    };
    for (std::size_t i = 0; i < 13; ++i) {
        if (canonical_z[i] == target_z) {
            return i;
        }
    }
    throw std::invalid_argument("Unsupported target element Z: " + std::to_string(target_z));
}

double SecondaryRateTable::mass_partial_rate(std::size_t proj_idx, std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const {
    const std::size_t stride_proj = kSecondaryNumSections * kSecondaryNumTargets * kSecondaryNumEnergies;
    const std::size_t stride_sec = kSecondaryNumTargets * kSecondaryNumEnergies;
    const std::size_t stride_tgt = kSecondaryNumEnergies;
    return mass_partial_rates_[proj_idx * stride_proj + section_id * stride_sec + target_idx * stride_tgt + energy_idx];
}

double SecondaryRateTable::mass_total_rate(std::size_t proj_idx, std::size_t section_id, std::size_t energy_idx) const {
    const std::size_t stride_proj = kSecondaryNumSections * kSecondaryNumEnergies;
    const std::size_t stride_sec = kSecondaryNumEnergies;
    return mass_total_rates_[proj_idx * stride_proj + section_id * stride_sec + energy_idx];
}

double SecondaryRateTable::interpolate_mass_total(std::size_t proj_idx, std::size_t section_id, double energy_mevu) const {
    if (energy_mevu <= energy_min_mevu_) {
        return mass_total_rate(proj_idx, section_id, 0);
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_total_rate(proj_idx, section_id, kSecondaryNumEnergies - 1);
    }
    const double node_flt = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const std::size_t idx0 = static_cast<std::size_t>(node_flt);
    const std::size_t idx1 = std::min(idx0 + 1, kSecondaryNumEnergies - 1);
    const double alpha = node_flt - static_cast<double>(idx0);

    const double y0 = mass_total_rate(proj_idx, section_id, idx0);
    const double y1 = mass_total_rate(proj_idx, section_id, idx1);
    return (1.0 - alpha) * y0 + alpha * y1;
}

}  // namespace carbon
