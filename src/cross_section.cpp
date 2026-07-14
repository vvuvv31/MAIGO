#include "carbon/cross_section.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {
namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

double parse_number(const std::string& field,
                    const std::filesystem::path& path,
                    std::size_t line_number) {
    std::size_t parsed = 0;
    const auto value = std::stod(field, &parsed);
    if (parsed != field.size()) {
        throw std::runtime_error("Invalid cross-section value at " + path.string() + ":" +
                                 std::to_string(line_number));
    }
    return value;
}

}  // namespace

CrossSectionTable::CrossSectionTable(
    std::vector<double> energies_MeVu,
    std::vector<double> macroscopic_cross_sections_per_mm)
    : energies_MeVu_(std::move(energies_MeVu)),
      macroscopic_cross_sections_per_mm_(std::move(macroscopic_cross_sections_per_mm)) {
    if (energies_MeVu_.size() < 2 ||
        energies_MeVu_.size() != macroscopic_cross_sections_per_mm_.size()) {
        throw std::invalid_argument("Cross-section table must contain at least two paired samples");
    }
    for (std::size_t index = 0; index < energies_MeVu_.size(); ++index) {
        if (energies_MeVu_[index] <= 0.0 || macroscopic_cross_sections_per_mm_[index] < 0.0) {
            throw std::invalid_argument(
                "Cross-section energies must be positive and values must be nonnegative");
        }
        if (index > 0 && energies_MeVu_[index] <= energies_MeVu_[index - 1]) {
            throw std::invalid_argument("Cross-section energies must be strictly increasing");
        }
    }
}

CrossSectionTable CrossSectionTable::from_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open cross-section table: " + path.string());
    }

    constexpr const char* energy_name = "energy_MeV_per_u";
    constexpr const char* cross_section_name = "water_macroscopic_cross_section_per_mm";
    std::size_t energy_column = 0;
    std::size_t cross_section_column = 0;
    bool found_header = false;
    std::vector<double> energies;
    std::vector<double> cross_sections;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const auto fields = split_csv(line);
        if (!found_header) {
            const auto energy = std::find(fields.begin(), fields.end(), energy_name);
            const auto cross_section =
                std::find(fields.begin(), fields.end(), cross_section_name);
            if (energy == fields.end() || cross_section == fields.end()) {
                throw std::runtime_error("Cross-section CSV is missing required columns: " +
                                         path.string());
            }
            energy_column = static_cast<std::size_t>(energy - fields.begin());
            cross_section_column = static_cast<std::size_t>(cross_section - fields.begin());
            found_header = true;
            continue;
        }
        if (fields.size() <= std::max(energy_column, cross_section_column)) {
            throw std::runtime_error("Incomplete cross-section row at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        energies.push_back(parse_number(fields[energy_column], path, line_number));
        cross_sections.push_back(
            parse_number(fields[cross_section_column], path, line_number));
    }
    if (!found_header) {
        throw std::runtime_error("Cross-section CSV header not found: " + path.string());
    }
    return CrossSectionTable(std::move(energies), std::move(cross_sections));
}

double CrossSectionTable::interpolate(double energy_MeVu) const noexcept {
    if (energy_MeVu <= energies_MeVu_.front()) {
        return macroscopic_cross_sections_per_mm_.front();
    }
    if (energy_MeVu >= energies_MeVu_.back()) {
        return macroscopic_cross_sections_per_mm_.back();
    }
    const auto upper = std::upper_bound(energies_MeVu_.begin(), energies_MeVu_.end(), energy_MeVu);
    const auto upper_index = static_cast<std::size_t>(upper - energies_MeVu_.begin());
    const auto lower_index = upper_index - 1;
    const auto fraction = (energy_MeVu - energies_MeVu_[lower_index]) /
                          (energies_MeVu_[upper_index] - energies_MeVu_[lower_index]);
    return macroscopic_cross_sections_per_mm_[lower_index] +
           fraction * (macroscopic_cross_sections_per_mm_[upper_index] -
                       macroscopic_cross_sections_per_mm_[lower_index]);
}

const std::vector<double>& CrossSectionTable::energies() const noexcept { return energies_MeVu_; }
const std::vector<double>& CrossSectionTable::values() const noexcept {
    return macroscopic_cross_sections_per_mm_;
}

}  // namespace carbon
