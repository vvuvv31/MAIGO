#include "carbon/stopping_power.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace carbon {

double ion_effective_charge(int atomic_number, double energy_MeVu) {
    if (atomic_number <= 0 || energy_MeVu <= 0.0) {
        throw std::invalid_argument("Effective charge requires positive Z and energy per nucleon");
    }
    constexpr double nucleon_mass_MeV = 931.49410242;
    const auto gamma = 1.0 + energy_MeVu / nucleon_mass_MeV;
    const auto beta_squared = std::max(0.0, 1.0 - 1.0 / (gamma * gamma));
    const auto beta = std::sqrt(beta_squared);
    const auto charge = static_cast<double>(atomic_number);
    return charge *
           (1.0 - std::exp(-125.0 * beta * std::pow(charge, -2.0 / 3.0)));
}

double stopping_power_scale_from_carbon(int atomic_number, double energy_MeVu) {
    const auto ion_charge = ion_effective_charge(atomic_number, energy_MeVu);
    const auto carbon_charge = ion_effective_charge(6, energy_MeVu);
    const auto ratio = ion_charge / carbon_charge;
    return ratio * ratio;
}

StoppingPowerTable::StoppingPowerTable(std::vector<double> energies_MeVu,
                                       std::vector<double> stopping_powers_MeV_per_mm)
    : energies_MeVu_(std::move(energies_MeVu)),
      stopping_powers_MeV_per_mm_(std::move(stopping_powers_MeV_per_mm)) {
    if (energies_MeVu_.size() < 2 || energies_MeVu_.size() != stopping_powers_MeV_per_mm_.size()) {
        throw std::invalid_argument("Stopping-power table must contain at least two paired samples");
    }
    for (std::size_t index = 0; index < energies_MeVu_.size(); ++index) {
        if (energies_MeVu_[index] <= 0.0 || stopping_powers_MeV_per_mm_[index] <= 0.0) {
            throw std::invalid_argument("Stopping-power energies and values must be positive");
        }
        if (index > 0 && energies_MeVu_[index] <= energies_MeVu_[index - 1]) {
            throw std::invalid_argument("Stopping-power energies must be strictly increasing");
        }
    }
}

StoppingPowerTable StoppingPowerTable::from_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open stopping-power table: " + path.string());
    }

    std::vector<double> energies;
    std::vector<double> stopping_powers;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(line[first]))) {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        double energy = 0.0;
        double stopping_power = 0.0;
        if (!(parser >> energy >> stopping_power)) {
            throw std::runtime_error("Invalid stopping-power row at " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        energies.push_back(energy);
        stopping_powers.push_back(stopping_power);
    }
    return StoppingPowerTable(std::move(energies), std::move(stopping_powers));
}

double StoppingPowerTable::interpolate(double energy_MeVu) const noexcept {
    if (energy_MeVu <= energies_MeVu_.front()) {
        return stopping_powers_MeV_per_mm_.front();
    }
    if (energy_MeVu >= energies_MeVu_.back()) {
        return stopping_powers_MeV_per_mm_.back();
    }
    const auto upper = std::upper_bound(energies_MeVu_.begin(), energies_MeVu_.end(), energy_MeVu);
    const auto upper_index = static_cast<std::size_t>(upper - energies_MeVu_.begin());
    const auto lower_index = upper_index - 1;
    const auto fraction = (energy_MeVu - energies_MeVu_[lower_index]) /
                          (energies_MeVu_[upper_index] - energies_MeVu_[lower_index]);
    return stopping_powers_MeV_per_mm_[lower_index] +
           fraction * (stopping_powers_MeV_per_mm_[upper_index] -
                       stopping_powers_MeV_per_mm_[lower_index]);
}

double StoppingPowerTable::minimum_energy_MeVu() const noexcept { return energies_MeVu_.front(); }
double StoppingPowerTable::maximum_energy_MeVu() const noexcept { return energies_MeVu_.back(); }
const std::vector<double>& StoppingPowerTable::energies() const noexcept { return energies_MeVu_; }
const std::vector<double>& StoppingPowerTable::values() const noexcept {
    return stopping_powers_MeV_per_mm_;
}

}  // namespace carbon
