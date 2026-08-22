#include "carbon/electron_transport.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace carbon {
namespace {

constexpr std::string_view expected_header =
    "kinetic_energy_MeV,"
    "electron_collisional_stopping_power_MeV_per_mm,"
    "electron_radiative_stopping_power_MeV_per_mm,"
    "electron_total_stopping_power_MeV_per_mm,"
    "positron_collisional_stopping_power_MeV_per_mm,"
    "positron_radiative_stopping_power_MeV_per_mm,"
    "positron_total_stopping_power_MeV_per_mm";

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

double integrate_linear_inverse_stopping_power(
    const double e0, const double e1, const double s0, const double s1) {
    const auto delta_energy = e1 - e0;
    if (delta_energy <= 0.0) return 0.0;
    const auto slope = (s1 - s0) / delta_energy;
    if (std::abs(slope) < 1.0e-14 * std::max({1.0, s0, s1})) {
        return delta_energy / s0;
    }
    return std::log(s1 / s0) / slope;
}

std::vector<double> cumulative_ranges(
    const std::vector<double>& energies,
    const std::vector<double>& stopping_powers) {
    std::vector<double> ranges(energies.size(), 0.0);
    ranges.front() = energies.front() / stopping_powers.front();
    for (std::size_t index = 1; index < energies.size(); ++index) {
        ranges[index] = ranges[index - 1] + integrate_linear_inverse_stopping_power(
            energies[index - 1], energies[index], stopping_powers[index - 1],
            stopping_powers[index]);
    }
    return ranges;
}

double interpolate(const std::vector<double>& energies,
                   const std::vector<double>& values,
                   const double energy) noexcept {
    if (energy <= energies.front()) return values.front();
    if (energy >= energies.back()) return values.back();
    const auto upper = std::upper_bound(energies.begin(), energies.end(), energy);
    const auto upper_index = static_cast<std::size_t>(upper - energies.begin());
    const auto lower_index = upper_index - 1;
    const auto fraction = (energy - energies[lower_index]) /
                          (energies[upper_index] - energies[lower_index]);
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index]);
}

}  // namespace

ElectronTransportTable::ElectronTransportTable(
    std::vector<double> kinetic_energies_MeV,
    std::vector<double> electron_collisional_stopping_power_MeV_per_mm,
    std::vector<double> electron_radiative_stopping_power_MeV_per_mm,
    std::vector<double> electron_total_stopping_power_MeV_per_mm,
    std::vector<double> positron_collisional_stopping_power_MeV_per_mm,
    std::vector<double> positron_radiative_stopping_power_MeV_per_mm,
    std::vector<double> positron_total_stopping_power_MeV_per_mm)
    : kinetic_energies_MeV_(std::move(kinetic_energies_MeV)),
      electron_collisional_stopping_power_MeV_per_mm_(
          std::move(electron_collisional_stopping_power_MeV_per_mm)),
      electron_radiative_stopping_power_MeV_per_mm_(
          std::move(electron_radiative_stopping_power_MeV_per_mm)),
      electron_total_stopping_power_MeV_per_mm_(
          std::move(electron_total_stopping_power_MeV_per_mm)),
      positron_collisional_stopping_power_MeV_per_mm_(
          std::move(positron_collisional_stopping_power_MeV_per_mm)),
      positron_radiative_stopping_power_MeV_per_mm_(
          std::move(positron_radiative_stopping_power_MeV_per_mm)),
      positron_total_stopping_power_MeV_per_mm_(
          std::move(positron_total_stopping_power_MeV_per_mm)) {
    const auto count = kinetic_energies_MeV_.size();
    if (count < 2 || electron_collisional_stopping_power_MeV_per_mm_.size() != count ||
        electron_radiative_stopping_power_MeV_per_mm_.size() != count ||
        electron_total_stopping_power_MeV_per_mm_.size() != count ||
        positron_collisional_stopping_power_MeV_per_mm_.size() != count ||
        positron_radiative_stopping_power_MeV_per_mm_.size() != count ||
        positron_total_stopping_power_MeV_per_mm_.size() != count) {
        throw std::invalid_argument(
            "Electron transport table must contain at least two complete rows");
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto energy = kinetic_energies_MeV_[index];
        if (!std::isfinite(energy) || energy <= 0.0 ||
            (index > 0 && energy <= kinetic_energies_MeV_[index - 1])) {
            throw std::invalid_argument(
                "Electron transport energies must be finite, positive, and strictly increasing");
        }
        for (const auto* values : {
                 &electron_collisional_stopping_power_MeV_per_mm_,
                 &electron_total_stopping_power_MeV_per_mm_,
                 &positron_collisional_stopping_power_MeV_per_mm_,
                 &positron_total_stopping_power_MeV_per_mm_,
             }) {
            if (!std::isfinite((*values)[index]) || (*values)[index] <= 0.0) {
                throw std::invalid_argument(
                    "Electron transport collision and total stopping powers must be "
                    "finite and positive");
            }
        }
        for (const auto* values : {
                 &electron_radiative_stopping_power_MeV_per_mm_,
                 &positron_radiative_stopping_power_MeV_per_mm_,
             }) {
            if (!std::isfinite((*values)[index]) || (*values)[index] < 0.0) {
                throw std::invalid_argument(
                    "Electron transport radiative stopping powers must be finite and "
                    "nonnegative");
            }
        }
        const auto validate_total = [index](const std::vector<double>& collision,
                                            const std::vector<double>& radiation,
                                            const std::vector<double>& total) {
            const auto expected = collision[index] + radiation[index];
            if (std::abs(total[index] - expected) > 1.0e-6 * expected) {
                throw std::invalid_argument(
                    "Electron transport total stopping power must equal collision plus radiation");
            }
        };
        validate_total(electron_collisional_stopping_power_MeV_per_mm_,
                       electron_radiative_stopping_power_MeV_per_mm_,
                       electron_total_stopping_power_MeV_per_mm_);
        validate_total(positron_collisional_stopping_power_MeV_per_mm_,
                       positron_radiative_stopping_power_MeV_per_mm_,
                       positron_total_stopping_power_MeV_per_mm_);
    }
    electron_cumulative_range_mm_ = cumulative_ranges(
        kinetic_energies_MeV_, electron_total_stopping_power_MeV_per_mm_);
    positron_cumulative_range_mm_ = cumulative_ranges(
        kinetic_energies_MeV_, positron_total_stopping_power_MeV_per_mm_);
}

ElectronTransportTable ElectronTransportTable::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open electron transport table: " + path.string());
    }
    std::vector<std::vector<double>> columns(7);
    bool found_header = false;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto cleaned = trim(line);
        if (cleaned.empty() || cleaned.front() == '#') continue;
        if (!found_header) {
            if (cleaned != expected_header) {
                throw std::runtime_error(
                    "Invalid electron transport table header at " + path.string() + ":" +
                    std::to_string(line_number));
            }
            found_header = true;
            continue;
        }
        auto row = cleaned;
        std::replace(row.begin(), row.end(), ',', ' ');
        std::istringstream parser(row);
        for (auto& column : columns) {
            double value = 0.0;
            if (!(parser >> value)) {
                throw std::runtime_error(
                    "Invalid electron transport row at " + path.string() + ":" +
                    std::to_string(line_number));
            }
            column.push_back(value);
        }
        std::string trailing;
        if (parser >> trailing) {
            throw std::runtime_error(
                "Too many electron transport columns at " + path.string() + ":" +
                std::to_string(line_number));
        }
    }
    if (!found_header) {
        throw std::runtime_error("Electron transport table has no header: " + path.string());
    }
    return ElectronTransportTable(
        std::move(columns[0]), std::move(columns[1]), std::move(columns[2]),
        std::move(columns[3]), std::move(columns[4]), std::move(columns[5]),
        std::move(columns[6]));
}

const std::vector<double>& ElectronTransportTable::collisional_stopping_powers(
    const LeptonSpecies species) const noexcept {
    return species == LeptonSpecies::electron
               ? electron_collisional_stopping_power_MeV_per_mm_
               : positron_collisional_stopping_power_MeV_per_mm_;
}

const std::vector<double>& ElectronTransportTable::radiative_stopping_powers(
    const LeptonSpecies species) const noexcept {
    return species == LeptonSpecies::electron
               ? electron_radiative_stopping_power_MeV_per_mm_
               : positron_radiative_stopping_power_MeV_per_mm_;
}

const std::vector<double>& ElectronTransportTable::total_stopping_powers(
    const LeptonSpecies species) const noexcept {
    return species == LeptonSpecies::electron
               ? electron_total_stopping_power_MeV_per_mm_
               : positron_total_stopping_power_MeV_per_mm_;
}

const std::vector<double>& ElectronTransportTable::cumulative_ranges_mm(
    const LeptonSpecies species) const noexcept {
    return species == LeptonSpecies::electron ? electron_cumulative_range_mm_
                                               : positron_cumulative_range_mm_;
}

double ElectronTransportTable::collisional_stopping_power(
    const LeptonSpecies species, const double kinetic_energy_MeV) const noexcept {
    return interpolate(kinetic_energies_MeV_, collisional_stopping_powers(species),
                       kinetic_energy_MeV);
}

double ElectronTransportTable::radiative_stopping_power(
    const LeptonSpecies species, const double kinetic_energy_MeV) const noexcept {
    return interpolate(kinetic_energies_MeV_, radiative_stopping_powers(species),
                       kinetic_energy_MeV);
}

double ElectronTransportTable::total_stopping_power(
    const LeptonSpecies species, const double kinetic_energy_MeV) const noexcept {
    return interpolate(kinetic_energies_MeV_, total_stopping_powers(species),
                       kinetic_energy_MeV);
}

double ElectronTransportTable::csda_range_mm(
    const LeptonSpecies species, const double kinetic_energy_MeV) const {
    if (!std::isfinite(kinetic_energy_MeV) || kinetic_energy_MeV < 0.0) {
        throw std::invalid_argument("Electron CSDA energy must be finite and nonnegative");
    }
    if (kinetic_energy_MeV <= kinetic_energies_MeV_.front()) {
        return kinetic_energy_MeV / total_stopping_powers(species).front();
    }
    if (kinetic_energy_MeV >= kinetic_energies_MeV_.back()) {
        return cumulative_ranges_mm(species).back() +
               (kinetic_energy_MeV - kinetic_energies_MeV_.back()) /
                   total_stopping_powers(species).back();
    }
    const auto upper = std::upper_bound(
        kinetic_energies_MeV_.begin(), kinetic_energies_MeV_.end(),
        kinetic_energy_MeV);
    const auto lower_index = static_cast<std::size_t>(
        upper - kinetic_energies_MeV_.begin() - 1);
    const auto& stopping = total_stopping_powers(species);
    return cumulative_ranges_mm(species)[lower_index] +
           integrate_linear_inverse_stopping_power(
               kinetic_energies_MeV_[lower_index], kinetic_energy_MeV,
               stopping[lower_index],
               interpolate(kinetic_energies_MeV_, stopping, kinetic_energy_MeV));
}

}  // namespace carbon
