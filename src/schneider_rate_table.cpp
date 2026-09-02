#include "carbon/schneider_rate_table.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace carbon {

std::size_t SchneiderRateTable::target_index_from_z(int target_z) {
    for (std::size_t i = 0; i < kSchneiderCanonicalZ.size(); ++i) {
        if (kSchneiderCanonicalZ[i] == target_z) {
            return i;
        }
    }
    throw std::invalid_argument("Unsupported Schneider canonical target Z: " + std::to_string(target_z));
}

SchneiderRateTable SchneiderRateTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider rate binary file: " + binary_path.string());
    }

    SchneiderRateHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || in.gcount() != sizeof(header)) {
        throw std::runtime_error("Truncated header in Schneider rate binary: " + binary_path.string());
    }

    if (std::strncmp(header.magic, "SCHNRATE", 8) != 0) {
        throw std::runtime_error("Invalid magic in Schneider rate binary: " + binary_path.string());
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported binary version in: " + binary_path.string());
    }
    if (header.num_sections != kSchneiderNumSections) {
        throw std::runtime_error("Invalid section count in: " + binary_path.string());
    }
    if (header.num_targets != kSchneiderNumTargets) {
        throw std::runtime_error("Invalid target count in: " + binary_path.string());
    }
    if (header.num_energies != kSchneiderNumEnergies) {
        throw std::runtime_error("Invalid energy count in: " + binary_path.string());
    }
    if (std::abs(header.energy_min_mevu - 0.5) > 1e-6 ||
        std::abs(header.energy_max_mevu - 430.0) > 1e-6 ||
        std::abs(header.energy_step_mevu - 0.5) > 1e-6) {
        throw std::runtime_error("Invalid energy grid bounds in: " + binary_path.string());
    }

    for (std::size_t i = 0; i < kSchneiderNumTargets; ++i) {
        if (header.target_z[i] != kSchneiderCanonicalZ[i]) {
            throw std::runtime_error("Canonical target Z order mismatch in: " + binary_path.string());
        }
    }

    SchneiderRateTable table;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    const std::size_t partial_elements = kSchneiderNumSections * kSchneiderNumTargets * kSchneiderNumEnergies;
    table.mass_partial_rates_.resize(partial_elements);
    in.read(reinterpret_cast<char*>(table.mass_partial_rates_.data()),
            partial_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(partial_elements * sizeof(double))) {
        throw std::runtime_error("Truncated partial rate payload in: " + binary_path.string());
    }

    const std::size_t total_elements = kSchneiderNumSections * kSchneiderNumEnergies;
    table.mass_total_rates_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.mass_total_rates_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated total rate payload in: " + binary_path.string());
    }

    // Optional metadata validation
    if (!metadata_path.empty() && std::filesystem::exists(metadata_path)) {
        std::ifstream meta_in(metadata_path);
        std::string meta_content((std::istreambuf_iterator<char>(meta_in)),
                                 std::istreambuf_iterator<char>());
        if (meta_content.find("schneider_inelastic_rates_v1.bin") == std::string::npos) {
            throw std::runtime_error("Metadata file does not reference binary: " + metadata_path.string());
        }
    }

    return table;
}

double SchneiderRateTable::mass_partial_rate(
    std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const {
    if (section_id >= kSchneiderNumSections || target_idx >= kSchneiderNumTargets || energy_idx >= kSchneiderNumEnergies) {
        throw std::out_of_range("SchneiderRateTable index out of range");
    }
    const std::size_t idx = section_id * (kSchneiderNumTargets * kSchneiderNumEnergies) +
                            target_idx * kSchneiderNumEnergies +
                            energy_idx;
    return mass_partial_rates_[idx];
}

double SchneiderRateTable::mass_total_rate(
    std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderNumSections || energy_idx >= kSchneiderNumEnergies) {
        throw std::out_of_range("SchneiderRateTable index out of range");
    }
    const std::size_t idx = section_id * kSchneiderNumEnergies + energy_idx;
    return mass_total_rates_[idx];
}

double SchneiderRateTable::interpolate_mass_partial(
    std::size_t section_id, std::size_t target_idx, double energy_mevu) const {
    if (section_id >= kSchneiderNumSections || target_idx >= kSchneiderNumTargets) {
        throw std::out_of_range("SchneiderRateTable section or target index out of range");
    }
    if (energy_mevu <= energy_min_mevu_) {
        return mass_partial_rate(section_id, target_idx, 0);
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_partial_rate(section_id, target_idx, kSchneiderNumEnergies - 1);
    }

    const double frac_idx = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const auto lower_idx = static_cast<std::size_t>(std::floor(frac_idx));
    const auto upper_idx = std::min(lower_idx + 1, kSchneiderNumEnergies - 1);
    const double alpha = frac_idx - static_cast<double>(lower_idx);

    const double y0 = mass_partial_rate(section_id, target_idx, lower_idx);
    const double y1 = mass_partial_rate(section_id, target_idx, upper_idx);
    return (1.0 - alpha) * y0 + alpha * y1;
}

double SchneiderRateTable::interpolate_mass_total(
    std::size_t section_id, double energy_mevu) const {
    if (section_id >= kSchneiderNumSections) {
        throw std::out_of_range("SchneiderRateTable section index out of range");
    }
    if (energy_mevu <= energy_min_mevu_) {
        return mass_total_rate(section_id, 0);
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_total_rate(section_id, kSchneiderNumEnergies - 1);
    }

    const double frac_idx = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const auto lower_idx = static_cast<std::size_t>(std::floor(frac_idx));
    const auto upper_idx = std::min(lower_idx + 1, kSchneiderNumEnergies - 1);
    const double alpha = frac_idx - static_cast<double>(lower_idx);

    const double y0 = mass_total_rate(section_id, lower_idx);
    const double y1 = mass_total_rate(section_id, upper_idx);
    return (1.0 - alpha) * y0 + alpha * y1;
}

}  // namespace carbon
