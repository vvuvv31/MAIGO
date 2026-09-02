#include "carbon/cross_section.hpp"
#include "carbon/detail/fred_fragmentation_data.hpp"

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
    std::size_t target_h_column = static_cast<std::size_t>(-1);
    std::size_t macro_h_column = static_cast<std::size_t>(-1);
    std::size_t macro_o_column = static_cast<std::size_t>(-1);
    bool found_header = false;
    std::vector<double> energies;
    std::vector<double> cross_sections;
    std::vector<double> target_h;
    std::vector<double> macro_h;
    std::vector<double> macro_o;
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
            const auto hfrac = std::find(fields.begin(), fields.end(), "target_h_fraction");
            if (hfrac != fields.end()) {
                target_h_column = static_cast<std::size_t>(hfrac - fields.begin());
            }
            const auto mh = std::find(fields.begin(), fields.end(), "macro_h_per_mm");
            const auto mo = std::find(fields.begin(), fields.end(), "macro_o_per_mm");
            if (mh != fields.end()) {
                macro_h_column = static_cast<std::size_t>(mh - fields.begin());
            }
            if (mo != fields.end()) {
                macro_o_column = static_cast<std::size_t>(mo - fields.begin());
            }
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
        if (target_h_column != static_cast<std::size_t>(-1) && fields.size() > target_h_column) {
            target_h.push_back(parse_number(fields[target_h_column], path, line_number));
        }
        if (macro_h_column != static_cast<std::size_t>(-1) && fields.size() > macro_h_column) {
            macro_h.push_back(parse_number(fields[macro_h_column], path, line_number));
        }
        if (macro_o_column != static_cast<std::size_t>(-1) && fields.size() > macro_o_column) {
            macro_o.push_back(parse_number(fields[macro_o_column], path, line_number));
        }
    }
    if (!found_header) {
        throw std::runtime_error("Cross-section CSV header not found: " + path.string());
    }
    auto table = CrossSectionTable(std::move(energies), std::move(cross_sections));
    if (target_h.size() == table.energies().size()) {
        table.set_target_h_fractions(std::move(target_h));
    }
    if (macro_h.size() == table.energies().size() && macro_o.size() == table.energies().size()) {
        table.set_partial_macros(std::move(macro_h), std::move(macro_o));
    }
    return table;
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
    if (header.empty() || header.front() != "energy_MeV_per_u") {
        throw std::runtime_error(
            "Schneider cross-section CSV must start with energy_MeV_per_u and have exactly 25 section columns: " +
            path.string());
    }
    constexpr std::size_t kExpectedSections = 25;
    constexpr std::size_t kExpectedColumns = 1 + kExpectedSections;
    for (std::size_t section = 0; section < kExpectedSections && section + 1 < header.size(); ++section) {
        const auto expected = "section_" + (section < 10 ? std::string{"0"} : std::string{}) +
                              std::to_string(section) +
                              "_mass_xs_per_mm_at_1g_cm3";
        if (header[section + 1] != expected) {
            throw std::runtime_error("Unexpected Schneider cross-section column '" +
                                     header[section + 1] + "' (expected '" + expected + "') in " + path.string());
        }
    }
    if (header.size() != kExpectedColumns) {
        throw std::runtime_error(
            "Schneider cross-section CSV must have exactly 25 section columns (got " +
            std::to_string(header.size() > 0 ? header.size() - 1 : 0) + "): " +
            path.string());
    }

    std::vector<double> energies;
    std::vector<std::vector<double>> section_values(kExpectedSections);
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
        const double energy = parse_number(fields[0], path, line_number);
        if (!std::isfinite(energy) || energy <= 0.0) {
            throw std::runtime_error("Non-positive or non-finite energy " + fields[0] +
                                     " at " + path.string() + ":" + std::to_string(line_number));
        }
        if (!energies.empty() && energy <= energies.back()) {
            throw std::runtime_error("Non-monotonic energy sequence (" +
                                     std::to_string(energy) + " <= " + std::to_string(energies.back()) +
                                     ") at " + path.string() + ":" + std::to_string(line_number));
        }
        energies.push_back(energy);
        for (std::size_t section = 0; section < kExpectedSections; ++section) {
            const double val = parse_number(fields[section + 1], path, line_number);
            if (!std::isfinite(val) || val < 0.0) {
                throw std::runtime_error("Negative or non-finite Schneider mass cross section (" +
                                         fields[section + 1] + ") at " + path.string() + ":" +
                                         std::to_string(line_number));
            }
            section_values[section].push_back(val);
        }
    }
    if (energies.empty()) {
        throw std::runtime_error("Empty Schneider cross-section table in " + path.string());
    }

    std::vector<CrossSectionTable> tables;
    tables.reserve(section_values.size());
    for (auto& values : section_values) {
        tables.emplace_back(energies, std::move(values));
    }
    return tables;
}

CrossSectionTable CrossSectionTable::from_fred_paper_water(double density_g_per_cm3) {
    if (!(density_g_per_cm3 > 0.0) || !std::isfinite(density_g_per_cm3)) {
        throw std::invalid_argument("fred paper water density must be positive");
    }
    std::vector<double> energies;
    std::vector<double> macros;
    std::vector<double> hfrac;
    std::vector<double> mh;
    std::vector<double> mo;
    energies.reserve(401);
    for (int i = 0; i <= 400; ++i) {
        const float e = static_cast<float>(i);
        const float incident = e * 12.0F;
        const float sig_h = calculate_icru_sigma_H(e);
        const float sig_o = calculate_sigma_nonel_mb(12.0F, 6.0F, 16.0F, 8.0F, e, incident);
        const float p_h = calculate_target_prob_H(sig_h, sig_o);
        const float tot = water_macroscopic_xs_per_mm(sig_h, sig_o,
                                                     static_cast<float>(density_g_per_cm3));
        constexpr float n_a = 6.02214076e23F;
        constexpr float m_water = 18.01528F;
        constexpr float mb_to_cm2 = 1.0e-27F;
        const float n_mol = static_cast<float>(density_g_per_cm3) * n_a / m_water;
        const float macro_h = 2.0F * n_mol * sig_h * mb_to_cm2 * 0.1F;
        const float macro_o = n_mol * sig_o * mb_to_cm2 * 0.1F;
        energies.push_back(static_cast<double>(e));
        macros.push_back(static_cast<double>(tot));
        hfrac.push_back(static_cast<double>(p_h));
        mh.push_back(static_cast<double>(macro_h));
        mo.push_back(static_cast<double>(macro_o));
    }
    CrossSectionTable table(std::move(energies), std::move(macros));
    table.set_target_h_fractions(std::move(hfrac));
    table.set_partial_macros(std::move(mh), std::move(mo));
    return table;
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

double CrossSectionTable::interpolate_target_h_fraction(double energy_MeVu) const noexcept {
    if (target_h_fractions_.size() != energies_MeVu_.size() || target_h_fractions_.empty()) {
        return 0.5;
    }
    if (!std::isfinite(energy_MeVu) || energy_MeVu <= energies_MeVu_.front()) {
        return target_h_fractions_.front();
    }
    if (energy_MeVu >= energies_MeVu_.back()) {
        return target_h_fractions_.back();
    }
    const auto upper = std::upper_bound(energies_MeVu_.begin(), energies_MeVu_.end(), energy_MeVu);
    const auto upper_index = static_cast<std::size_t>(upper - energies_MeVu_.begin());
    const auto lower_index = upper_index - 1;
    const auto fraction = (energy_MeVu - energies_MeVu_[lower_index]) /
                          (energies_MeVu_[upper_index] - energies_MeVu_[lower_index]);
    return target_h_fractions_[lower_index] +
           fraction * (target_h_fractions_[upper_index] - target_h_fractions_[lower_index]);
}

void CrossSectionTable::set_target_h_fractions(std::vector<double> fractions) {
    target_h_fractions_ = std::move(fractions);
}

void CrossSectionTable::set_partial_macros(std::vector<double> macro_h,
                                           std::vector<double> macro_o) {
    macro_h_per_mm_ = std::move(macro_h);
    macro_o_per_mm_ = std::move(macro_o);
}

const std::vector<double>& CrossSectionTable::energies() const noexcept { return energies_MeVu_; }
const std::vector<double>& CrossSectionTable::values() const noexcept {
    return macroscopic_cross_sections_per_mm_;
}
const std::vector<double>& CrossSectionTable::target_h_fractions() const noexcept {
    return target_h_fractions_;
}

ResampledCrossSectionGrid resample_cross_section_grid(
    const CrossSectionTable& cross_section,
    const std::vector<double>& transport_energies_MeVu) {
    ResampledCrossSectionGrid result;
    result.macroscopic_per_mm.reserve(transport_energies_MeVu.size());
    result.target_h_fraction.reserve(transport_energies_MeVu.size());
    const bool has_target_fraction =
        cross_section.target_h_fractions().size() == cross_section.energies().size() &&
        !cross_section.target_h_fractions().empty();
    for (const double energy_MeVu : transport_energies_MeVu) {
        result.macroscopic_per_mm.push_back(
            static_cast<float>(cross_section.interpolate(energy_MeVu)));
        result.target_h_fraction.push_back(
            has_target_fraction
                ? static_cast<float>(
                      cross_section.interpolate_target_h_fraction(energy_MeVu))
                : 0.5F);
    }
    return result;
}

SchneiderResampledCrossSectionGrid resample_schneider_cross_section_grid(
    const std::vector<CrossSectionTable>& tables,
    const std::vector<double>& transport_energies_MeVu) {
    if (tables.size() != SchneiderResampledCrossSectionGrid::kExpectedSections) {
        throw std::invalid_argument(
            "resample_schneider_cross_section_grid requires exactly 25 section tables, got " +
            std::to_string(tables.size()));
    }
    if (transport_energies_MeVu.empty()) {
        throw std::invalid_argument("transport_energies_MeVu cannot be empty");
    }

    const auto& ref_energies = tables.front().energies();
    if (ref_energies.empty()) {
        throw std::invalid_argument("Schneider cross-section tables contain empty energy grid");
    }

    // Verify all 25 tables share identical energy nodes
    for (std::size_t s = 1; s < tables.size(); ++s) {
        const auto& e_s = tables[s].energies();
        if (e_s.size() != ref_energies.size()) {
            throw std::invalid_argument(
                "Schneider cross-section section " + std::to_string(s) +
                " has differing energy node count (" + std::to_string(e_s.size()) +
                " vs reference " + std::to_string(ref_energies.size()) + ")");
        }
        for (std::size_t i = 0; i < ref_energies.size(); ++i) {
            if (std::abs(e_s[i] - ref_energies[i]) > 1e-6) {
                throw std::invalid_argument(
                    "Schneider cross-section section " + std::to_string(s) +
                    " has differing energy grid node at index " + std::to_string(i));
            }
        }
    }

    // Energy coverage check: transport energy range must be fully covered by validated table
    constexpr double kCoverageTolerance = 1e-4;
    const double t_min = transport_energies_MeVu.front();
    const double t_max = transport_energies_MeVu.back();
    const double table_min = ref_energies.front();
    const double table_max = ref_energies.back();

    if (t_min < table_min - kCoverageTolerance || t_max > table_max + kCoverageTolerance) {
        throw std::runtime_error(
            "Transport energy range [" + std::to_string(t_min) + ", " + std::to_string(t_max) +
            "] exceeds validated Schneider cross-section coverage [" + std::to_string(table_min) +
            ", " + std::to_string(table_max) + "] without clamping policy.");
    }

    SchneiderResampledCrossSectionGrid result;
    result.transport_energies_MeVu = transport_energies_MeVu;
    const std::size_t n_nodes = transport_energies_MeVu.size();
    result.mass_xs_per_mm_at_1g_cm3.resize(
        SchneiderResampledCrossSectionGrid::kExpectedSections * n_nodes);

    for (std::size_t s = 0; s < SchneiderResampledCrossSectionGrid::kExpectedSections; ++s) {
        const auto& table = tables[s];
        const auto& xs_vals = table.values();
        const std::size_t offset = s * n_nodes;

        for (std::size_t i = 0; i < n_nodes; ++i) {
            const double e = transport_energies_MeVu[i];
            // Linear interpolation within [table_min, table_max]
            if (e <= table_min) {
                result.mass_xs_per_mm_at_1g_cm3[offset + i] = static_cast<float>(xs_vals.front());
            } else if (e >= table_max) {
                result.mass_xs_per_mm_at_1g_cm3[offset + i] = static_cast<float>(xs_vals.back());
            } else {
                auto it = std::lower_bound(ref_energies.begin(), ref_energies.end(), e);
                const auto idx = static_cast<std::size_t>(std::distance(ref_energies.begin(), it));
                const double x0 = ref_energies[idx - 1];
                const double x1 = ref_energies[idx];
                const double y0 = xs_vals[idx - 1];
                const double y1 = xs_vals[idx];
                const double frac = (e - x0) / (x1 - x0);
                result.mass_xs_per_mm_at_1g_cm3[offset + i] = static_cast<float>(y0 + frac * (y1 - y0));
            }
        }
    }

    return result;
}


const std::vector<double>& CrossSectionTable::macro_h_per_mm() const noexcept {
    return macro_h_per_mm_;
}
const std::vector<double>& CrossSectionTable::macro_o_per_mm() const noexcept {
    return macro_o_per_mm_;
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
