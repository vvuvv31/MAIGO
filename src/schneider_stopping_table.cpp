#include "carbon/schneider_stopping_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace carbon {

SchneiderStoppingTable SchneiderStoppingTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider stopping binary file: " + binary_path.string());
    }

    SchneiderStoppingHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || in.gcount() != sizeof(header)) {
        throw std::runtime_error("Truncated header in Schneider stopping binary: " + binary_path.string());
    }

    if (std::strncmp(header.magic, "SCHNSTOP", 8) != 0) {
        throw std::runtime_error("Invalid magic in Schneider stopping binary: " + binary_path.string());
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported binary version in: " + binary_path.string());
    }
    if (header.num_sections != kSchneiderStoppingNumSections) {
        throw std::runtime_error("Invalid section count in: " + binary_path.string());
    }
    if (header.num_energies != kSchneiderStoppingNumEnergies) {
        throw std::runtime_error("Invalid energy count in: " + binary_path.string());
    }
    if (std::abs(header.energy_min_mevu - kSchneiderStoppingEnergyMin) > 1e-6 ||
        std::abs(header.energy_max_mevu - kSchneiderStoppingEnergyMax) > 1e-6 ||
        std::abs(header.energy_step_mevu - kSchneiderStoppingEnergyStep) > 1e-6) {
        throw std::runtime_error("Invalid energy grid bounds in: " + binary_path.string());
    }

    SchneiderStoppingTable table;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    table.densities_.resize(kSchneiderStoppingNumSections);
    in.read(reinterpret_cast<char*>(table.densities_.data()),
            kSchneiderStoppingNumSections * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(kSchneiderStoppingNumSections * sizeof(double))) {
        throw std::runtime_error("Truncated densities in: " + binary_path.string());
    }

    const std::size_t total_elements = kSchneiderStoppingNumSections * kSchneiderStoppingNumEnergies;
    table.mass_stopping_powers_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.mass_stopping_powers_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated mass stopping powers in: " + binary_path.string());
    }

    table.csda_ranges_mm_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.csda_ranges_mm_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated CSDA ranges in: " + binary_path.string());
    }

    // Validate metadata reference if supplied
    if (!metadata_path.empty() && std::filesystem::exists(metadata_path)) {
        std::ifstream meta_in(metadata_path);
        std::string meta_content((std::istreambuf_iterator<char>(meta_in)),
                                 std::istreambuf_iterator<char>());
        if (meta_content.find("schneider_stopping_v1.bin") == std::string::npos) {
            throw std::runtime_error("Metadata file does not reference binary: " + metadata_path.string());
        }
        if (meta_content.find("\"section_manifest\"") == std::string::npos) {
            throw std::runtime_error("Metadata file lacks section_manifest schema: " + metadata_path.string());
        }
    }

    return table;
}

SchneiderStoppingTable SchneiderStoppingTable::from_csv(
    const std::filesystem::path& csv_path) {
    std::ifstream in(csv_path);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider stopping CSV file: " + csv_path.string());
    }

    SchneiderStoppingTable table;
    table.densities_.assign(kSchneiderStoppingNumSections, 0.0);
    const std::size_t total_elements = kSchneiderStoppingNumSections * kSchneiderStoppingNumEnergies;
    table.mass_stopping_powers_.assign(total_elements, 0.0);
    table.csda_ranges_mm_.assign(total_elements, 0.0);

    std::string line;
    std::size_t line_num = 0;
    while (std::getline(in, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#' || line.rfind("energy_mevu", 0) == 0) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double energy_mevu = 0.0;
        std::size_t sec_id = 0;
        std::string mat_name;
        double density = 0.0;
        double mass_sp = 0.0;
        double linear_sp = 0.0;
        double csda_r = 0.0;

        if (!(iss >> energy_mevu >> sec_id >> mat_name >> density >> mass_sp >> linear_sp >> csda_r)) {
            throw std::runtime_error("Malformed row in " + csv_path.string() + ":" + std::to_string(line_num));
        }

        if (sec_id >= kSchneiderStoppingNumSections) {
            throw std::out_of_range("Section id out of range in " + csv_path.string() + ":" + std::to_string(line_num));
        }

        const auto e_idx = static_cast<std::size_t>(
            std::round((energy_mevu - kSchneiderStoppingEnergyMin) / kSchneiderStoppingEnergyStep));
        if (e_idx >= kSchneiderStoppingNumEnergies) {
            continue;
        }

        table.densities_[sec_id] = density;
        const std::size_t offset = sec_id * kSchneiderStoppingNumEnergies + e_idx;
        table.mass_stopping_powers_[offset] = mass_sp;
        table.csda_ranges_mm_[offset] = csda_r;
    }

    return table;
}

double SchneiderStoppingTable::density(std::size_t section_id) const {
    if (section_id >= kSchneiderStoppingNumSections) {
        throw std::out_of_range("Schneider section_id out of range: " + std::to_string(section_id));
    }
    return densities_[section_id];
}

double SchneiderStoppingTable::mass_stopping_power(std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderStoppingNumSections || energy_idx >= kSchneiderStoppingNumEnergies) {
        throw std::out_of_range("Schneider stopping index out of range");
    }
    return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies + energy_idx];
}

double SchneiderStoppingTable::csda_range_mm(std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderStoppingNumSections || energy_idx >= kSchneiderStoppingNumEnergies) {
        throw std::out_of_range("Schneider stopping index out of range");
    }
    return csda_ranges_mm_[section_id * kSchneiderStoppingNumEnergies + energy_idx];
}

double SchneiderStoppingTable::interpolate_mass_stopping(std::size_t section_id, double energy_mevu) const {
    if (section_id >= kSchneiderStoppingNumSections) {
        throw std::out_of_range("Schneider section_id out of range: " + std::to_string(section_id));
    }
    if (energy_mevu <= energy_min_mevu_) {
        return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies];
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies + (kSchneiderStoppingNumEnergies - 1)];
    }

    const double frac_idx = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const auto idx = static_cast<std::size_t>(std::floor(frac_idx));
    const double frac = frac_idx - static_cast<double>(idx);

    const auto base = section_id * kSchneiderStoppingNumEnergies;
    return mass_stopping_powers_[base + idx] + frac * (mass_stopping_powers_[base + idx + 1] - mass_stopping_powers_[base + idx]);
}

std::vector<float> SchneiderStoppingTable::to_flat_mass_stopping_float() const {
    std::vector<float> result(mass_stopping_powers_.size());
    for (std::size_t i = 0; i < mass_stopping_powers_.size(); ++i) {
        result[i] = static_cast<float>(mass_stopping_powers_[i]);
    }
    return result;
}

}  // namespace carbon
