#include "carbon/cross_section.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
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
        while (!field.empty() &&
               std::isspace(static_cast<unsigned char>(field.front()))) {
            field.erase(field.begin());
        }
        while (!field.empty() &&
               std::isspace(static_cast<unsigned char>(field.back()))) {
            field.pop_back();
        }
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
        if (!std::isfinite(energies_MeVu_[index]) || energies_MeVu_[index] < 0.0) {
            throw std::invalid_argument(
                "Cross-section energies must be finite and nonnegative");
        }
        if (!std::isfinite(macroscopic_cross_sections_per_mm_[index]) ||
            macroscopic_cross_sections_per_mm_[index] < 0.0) {
            throw std::invalid_argument(
                "Cross-section values must be finite and nonnegative");
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
    constexpr const char* water_cross_section_name =
        "water_macroscopic_cross_section_per_mm";
    constexpr const char* copper_cross_section_name =
        "c12_copper_macroscopic_inelastic_cross_section_per_mm";
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
            auto energy = std::find(fields.begin(), fields.end(), energy_name);
            if (energy == fields.end()) {
                energy = std::find(fields.begin(), fields.end(), "energy_MeVu");
            }
            if (energy == fields.end()) {
                energy = std::find(fields.begin(), fields.end(), "energy_MeV");
            }
            auto cross_section =
                std::find(fields.begin(), fields.end(), water_cross_section_name);
            if (cross_section == fields.end()) {
                cross_section =
                    std::find(fields.begin(), fields.end(), copper_cross_section_name);
            }
            if (cross_section == fields.end()) {
                cross_section =
                    std::find(fields.begin(), fields.end(), "macroscopic_cross_section_per_mm");
            }
            if (cross_section == fields.end()) {
                cross_section =
                    std::find(fields.begin(), fields.end(), "macro_total_per_mm");
            }
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

std::vector<CrossSectionTable> CrossSectionTable::from_schneider_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open Schneider cross-section table: " +
                                 path.string());
    }
    std::string line;
    std::size_t line_number = 0;
    std::vector<std::string> header;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && line[first] != '#') {
            header = split_csv(line);
            break;
        }
    }
    if (header.size() < 2 || header.front() != "energy_MeV_per_u") {
        throw std::runtime_error(
            "Schneider cross-section CSV must start with energy_MeV_per_u and "
            "at least one section column: " +
            path.string());
    }
    for (std::size_t section = 0; section + 1 < header.size(); ++section) {
        const auto expected = "section_" + (section < 10 ? std::string{"0"} : std::string{}) +
                              std::to_string(section) +
                              "_mass_xs_per_mm_at_1g_cm3";
        if (header[section + 1] != expected) {
            throw std::runtime_error("Unexpected Schneider cross-section column '" +
                                     header[section + 1] + "' in " + path.string());
        }
    }

    std::vector<double> energies;
    std::vector<std::vector<double>> section_values(header.size() - 1);
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const auto fields = split_csv(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("Incomplete Schneider cross-section row at " +
                                     path.string() + ":" + std::to_string(line_number));
        }
        energies.push_back(parse_number(fields[0], path, line_number));
        for (std::size_t section = 0; section < section_values.size(); ++section) {
            section_values[section].push_back(
                parse_number(fields[section + 1], path, line_number));
        }
    }

    std::vector<CrossSectionTable> tables;
    tables.reserve(section_values.size());
    for (auto& values : section_values) {
        tables.emplace_back(energies, std::move(values));
    }
    return tables;
}

double CrossSectionTable::interpolate(double energy_MeVu) const noexcept {
    if (!std::isfinite(energy_MeVu) || energy_MeVu <= energies_MeVu_.front()) {
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

IonCrossSectionTables IonCrossSectionTables::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Cannot open ion cross-section table: " + path.string());
    }
    struct Row {
        int z{};
        int a{};
        double energy{};
        double value{};
    };
    std::vector<Row> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#' ||
            std::isalpha(static_cast<unsigned char>(line[first]))) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        Row row;
        double mean_free_path = 0.0;
        if (!(parser >> row.z >> row.a >> row.energy >> row.value >>
              mean_free_path)) {
            throw std::runtime_error(
                "Invalid ion cross-section row at " + path.string() +
                ":" + std::to_string(line_number));
        }
        if (row.z <= 0 ||
            row.z >= static_cast<int>(atomic_number_slots) ||
            row.a <= 0 || row.a >= static_cast<int>(mass_stride) ||
            !(row.energy > 0.0) || row.value < 0.0 ||
            !std::isfinite(row.energy) || !std::isfinite(row.value)) {
            throw std::runtime_error(
                "Ion cross-section row outside supported range at " +
                path.string() + ":" + std::to_string(line_number));
        }
        rows.push_back(row);
    }
    if (rows.size() < 2) {
        throw std::runtime_error(
            "Ion cross-section table is empty: " + path.string());
    }

    IonCrossSectionTables result;
    const auto first_z = rows.front().z;
    const auto first_a = rows.front().a;
    while (result.energy_grid_size_ < rows.size() &&
           rows[result.energy_grid_size_].z == first_z &&
           rows[result.energy_grid_size_].a == first_a) {
        ++result.energy_grid_size_;
    }
    if (result.energy_grid_size_ < 2) {
        throw std::runtime_error(
            "Ion cross-section grid needs at least two points");
    }
    result.minimum_energy_MeVu_ =
        static_cast<float>(rows[0].energy);
    result.energy_step_MeVu_ =
        static_cast<float>(rows[1].energy - rows[0].energy);
    result.values_.assign(
        species_slots * result.energy_grid_size_, 0.0F);
    result.species_present_.assign(species_slots, 0);
    std::vector<std::size_t> counts(species_slots, 0);
    for (const auto& row : rows) {
        const auto species =
            static_cast<std::size_t>(row.z) * mass_stride +
            static_cast<std::size_t>(row.a);
        const auto energy_index = counts[species]++;
        if (energy_index >= result.energy_grid_size_) {
            throw std::runtime_error(
                "Too many ion cross-section samples for one isotope");
        }
        const auto expected =
            static_cast<double>(result.minimum_energy_MeVu_) +
            static_cast<double>(energy_index) *
                static_cast<double>(result.energy_step_MeVu_);
        if (std::abs(row.energy - expected) > 1.0e-5) {
            throw std::runtime_error(
                "Nonuniform ion cross-section energy grid");
        }
        result.values_[
            species * result.energy_grid_size_ + energy_index] =
            static_cast<float>(row.value);
    }
    for (std::size_t species = 0; species < species_slots;
         ++species) {
        if (counts[species] == 0) {
            continue;
        }
        if (counts[species] != result.energy_grid_size_) {
            throw std::runtime_error(
                "Incomplete isotope in ion cross-section table");
        }
        result.species_present_[species] = 1;
    }
    return result;
}

const std::vector<float>& IonCrossSectionTables::values() const noexcept {
    return values_;
}

const std::vector<std::uint8_t>&
IonCrossSectionTables::species_present() const noexcept {
    return species_present_;
}

std::size_t IonCrossSectionTables::energy_grid_size() const noexcept {
    return energy_grid_size_;
}

float IonCrossSectionTables::minimum_energy_MeVu() const noexcept {
    return minimum_energy_MeVu_;
}

float IonCrossSectionTables::energy_step_MeVu() const noexcept {
    return energy_step_MeVu_;
}

NeutralCrossSectionTables NeutralCrossSectionTables::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Cannot open neutral cross-section table: " + path.string());
    }
    struct Row {
        int pdg_id{};
        double energy{};
        double value{};
    };
    std::vector<Row> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#' ||
            std::isalpha(static_cast<unsigned char>(line[first]))) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        Row row;
        double mean_free_path = 0.0;
        if (!(parser >> row.pdg_id >> row.energy >> row.value >>
              mean_free_path)) {
            throw std::runtime_error(
                "Invalid neutral cross-section row at " + path.string() +
                ":" + std::to_string(line_number));
        }
        if ((row.pdg_id != 22 && row.pdg_id != 2112) ||
            !(row.energy > 0.0) || row.value < 0.0 ||
            !std::isfinite(row.energy) || !std::isfinite(row.value)) {
            throw std::runtime_error(
                "Neutral cross-section row outside supported range at " +
                path.string() + ":" + std::to_string(line_number));
        }
        rows.push_back(row);
    }
    if (rows.size() < 4) {
        throw std::runtime_error(
            "Neutral cross-section table is empty: " + path.string());
    }

    NeutralCrossSectionTables result;
    const auto first_pdg = rows.front().pdg_id;
    while (result.energy_grid_size_ < rows.size() &&
           rows[result.energy_grid_size_].pdg_id == first_pdg) {
        ++result.energy_grid_size_;
    }
    if (first_pdg != 22 || result.energy_grid_size_ < 2 ||
        rows.size() != species_count * result.energy_grid_size_) {
        throw std::runtime_error(
            "Neutral cross-section table must contain rectangular gamma "
            "then neutron grids");
    }
    result.minimum_log_energy_ =
        static_cast<float>(std::log(rows[0].energy));
    result.log_energy_step_ = static_cast<float>(
        (std::log(rows[result.energy_grid_size_ - 1].energy) -
         std::log(rows[0].energy)) /
        static_cast<double>(result.energy_grid_size_ - 1));
    if (!(result.log_energy_step_ > 0.0F)) {
        throw std::runtime_error(
            "Neutral cross-section log-energy step must be positive");
    }
    result.values_.resize(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto species = index / result.energy_grid_size_;
        const auto energy_index = index % result.energy_grid_size_;
        const auto expected_pdg = species == 0 ? 22 : 2112;
        const auto expected_log =
            static_cast<double>(result.minimum_log_energy_) +
            static_cast<double>(energy_index) *
                static_cast<double>(result.log_energy_step_);
        if (rows[index].pdg_id != expected_pdg ||
            std::abs(std::log(rows[index].energy) - expected_log) >
                3.0e-5) {
            throw std::runtime_error(
                "Neutral cross-section energy grid is not uniform in log(E)");
        }
        result.values_[index] = static_cast<float>(rows[index].value);
    }
    return result;
}

const std::vector<float>& NeutralCrossSectionTables::values()
    const noexcept {
    return values_;
}

std::size_t NeutralCrossSectionTables::energy_grid_size() const noexcept {
    return energy_grid_size_;
}

float NeutralCrossSectionTables::minimum_log_energy() const noexcept {
    return minimum_log_energy_;
}

float NeutralCrossSectionTables::log_energy_step() const noexcept {
    return log_energy_step_;
}

}  // namespace carbon
