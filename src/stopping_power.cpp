#include "carbon/stopping_power.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace carbon {

namespace {

double integrate_linear_inverse_stopping_power(
    const double e0, const double e1, const double s0, const double s1) {
    const auto delta_e = e1 - e0;
    if (delta_e <= 0.0) return 0.0;
    const auto slope = (s1 - s0) / delta_e;
    if (std::abs(slope) < 1.0e-14 * std::max({1.0, std::abs(s0), std::abs(s1)})) {
        return delta_e / std::max(s0, 1.0e-300);
    }
    return std::log(s1 / s0) / slope;
}

}  // namespace

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

double stopping_power_scale_from_reference_ion(
    int atomic_number, int reference_atomic_number, double energy_MeVu) {
    const auto ion_charge = ion_effective_charge(atomic_number, energy_MeVu);
    const auto reference_charge =
        ion_effective_charge(reference_atomic_number, energy_MeVu);
    const auto ratio = ion_charge / reference_charge;
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
    cumulative_ranges_mm_.assign(energies_MeVu_.size(), 0.0);
    cumulative_ranges_mm_.front() = energies_MeVu_.front() /
                                     stopping_powers_MeV_per_mm_.front();
    for (std::size_t index = 1; index < energies_MeVu_.size(); ++index) {
        cumulative_ranges_mm_[index] = cumulative_ranges_mm_[index - 1] +
            integrate_linear_inverse_stopping_power(
                energies_MeVu_[index - 1], energies_MeVu_[index],
                stopping_powers_MeV_per_mm_[index - 1],
                stopping_powers_MeV_per_mm_[index]);
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

double StoppingPowerTable::csda_range_mm(const double energy_MeVu,
                                         const int mass_number) const {
    if (!std::isfinite(energy_MeVu) || energy_MeVu < 0.0) {
        throw std::invalid_argument("CSDA energy must be finite and nonnegative");
    }
    if (mass_number <= 0) {
        throw std::invalid_argument("CSDA mass number must be positive");
    }
    double range_a1 = 0.0;
    if (energy_MeVu <= energies_MeVu_.front()) {
        range_a1 = energy_MeVu / stopping_powers_MeV_per_mm_.front();
    } else if (energy_MeVu >= energies_MeVu_.back()) {
        range_a1 = cumulative_ranges_mm_.back() +
            (energy_MeVu - energies_MeVu_.back()) /
                stopping_powers_MeV_per_mm_.back();
    } else {
        const auto upper = std::upper_bound(
            energies_MeVu_.begin(), energies_MeVu_.end(), energy_MeVu);
        const auto index = static_cast<std::size_t>(upper - energies_MeVu_.begin() - 1);
        const auto s0 = stopping_powers_MeV_per_mm_[index];
        const auto s1 = stopping_powers_MeV_per_mm_[index + 1];
        range_a1 = cumulative_ranges_mm_[index] +
            integrate_linear_inverse_stopping_power(
                energies_MeVu_[index], energy_MeVu, s0,
                s0 + (s1 - s0) * (energy_MeVu - energies_MeVu_[index]) /
                                  (energies_MeVu_[index + 1] - energies_MeVu_[index]));
    }
    return static_cast<double>(mass_number) * range_a1;
}

double StoppingPowerTable::csda_energy_after_distance_MeVu(
    const double initial_energy_MeVu, const double distance_mm, const int mass_number) const {
    if (!std::isfinite(distance_mm) || distance_mm < 0.0) {
        throw std::invalid_argument("CSDA distance must be finite and nonnegative");
    }
    const auto initial_range = csda_range_mm(initial_energy_MeVu, mass_number);
    if (distance_mm >= initial_range) return 0.0;
    const auto target_a1 = (initial_range - distance_mm) / mass_number;
    if (target_a1 <= cumulative_ranges_mm_.front()) {
        return target_a1 * stopping_powers_MeV_per_mm_.front();
    }
    if (target_a1 >= cumulative_ranges_mm_.back()) {
        return energies_MeVu_.back() +
            (target_a1 - cumulative_ranges_mm_.back()) *
                stopping_powers_MeV_per_mm_.back();
    }
    const auto upper = std::upper_bound(
        cumulative_ranges_mm_.begin(), cumulative_ranges_mm_.end(), target_a1);
    const auto index = static_cast<std::size_t>(upper - cumulative_ranges_mm_.begin() - 1);
    const auto e0 = energies_MeVu_[index];
    const auto e1 = energies_MeVu_[index + 1];
    const auto s0 = stopping_powers_MeV_per_mm_[index];
    const auto slope = (stopping_powers_MeV_per_mm_[index + 1] - s0) / (e1 - e0);
    const auto integral = target_a1 - cumulative_ranges_mm_[index];
    const auto delta_e = std::abs(slope) <
            1.0e-14 * std::max({1.0, std::abs(s0),
                                std::abs(stopping_powers_MeV_per_mm_[index + 1])})
        ? s0 * integral
        : s0 * std::expm1(slope * integral) / slope;
    return e0 + std::clamp(delta_e, 0.0, e1 - e0);
}

const std::vector<double>& StoppingPowerTable::cumulative_ranges_mm() const noexcept {
    return cumulative_ranges_mm_;
}

double StoppingPowerTable::minimum_energy_MeVu() const noexcept { return energies_MeVu_.front(); }
double StoppingPowerTable::maximum_energy_MeVu() const noexcept { return energies_MeVu_.back(); }
const std::vector<double>& StoppingPowerTable::energies() const noexcept { return energies_MeVu_; }
const std::vector<double>& StoppingPowerTable::values() const noexcept {
    return stopping_powers_MeV_per_mm_;
}

IonStoppingPowerTables IonStoppingPowerTables::from_csv(
    const std::filesystem::path& path,
    const StoppingPowerTable& carbon_stopping_power,
    const bool normalize_to_file_carbon) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open ion stopping-power table: " + path.string());
    }

    IonStoppingPowerTables result;
    result.energy_grid_size_ = carbon_stopping_power.energies().size();
    const auto value_count = species_slots * result.energy_grid_size_;
    result.ratios_to_carbon_.assign(value_count, 0.0F);
    result.delta_electron_fractions_.assign(value_count, 0.0F);
    result.species_present_.assign(species_slots, 0);
    std::vector<std::size_t> samples_per_species(species_slots, 0);

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
        int atomic_number = 0;
        int mass_number = 0;
        double energy = 0.0;
        double stopping_power = 0.0;
        double restricted = 0.0;
        double nuclear = 0.0;
        double raw_delta = 0.0;
        double delta_fraction = 0.0;
        if (!(parser >> atomic_number >> mass_number >> energy >> stopping_power >>
              restricted >> nuclear >> raw_delta >> delta_fraction)) {
            throw std::runtime_error("Invalid ion stopping-power row at " +
                                     path.string() + ":" +
                                     std::to_string(line_number));
        }
        (void)restricted;
        (void)nuclear;
        (void)raw_delta;
        if (atomic_number <= 0 ||
            atomic_number >= static_cast<int>(atomic_number_slots) ||
            mass_number <= 0 || mass_number >= static_cast<int>(mass_stride)) {
            throw std::runtime_error("Ion species outside supported Z/A table range at " +
                                     path.string() + ":" +
                                     std::to_string(line_number));
        }
        const auto species = static_cast<std::size_t>(atomic_number) * mass_stride +
                             static_cast<std::size_t>(mass_number);
        const auto energy_index = samples_per_species[species]++;
        if (energy_index >= result.energy_grid_size_ ||
            std::abs(energy - carbon_stopping_power.energies()[energy_index]) >
                1.0e-9) {
            throw std::runtime_error(
                "Ion stopping-power energy grid differs from C-12 grid at " +
                path.string() + ":" + std::to_string(line_number));
        }
        if (!std::isfinite(stopping_power) || stopping_power <= 0.0 ||
            !std::isfinite(delta_fraction) || delta_fraction < 0.0 ||
            delta_fraction >= 1.0) {
            throw std::runtime_error("Invalid ion stopping-power value at " +
                                     path.string() + ":" +
                                     std::to_string(line_number));
        }
        const auto offset = species * result.energy_grid_size_ + energy_index;
        result.ratios_to_carbon_[offset] = static_cast<float>(
            stopping_power / carbon_stopping_power.values()[energy_index]);
        result.delta_electron_fractions_[offset] =
            static_cast<float>(delta_fraction);
    }

    for (std::size_t species = 0; species < species_slots; ++species) {
        if (samples_per_species[species] == 0) {
            continue;
        }
        if (samples_per_species[species] != result.energy_grid_size_) {
            throw std::runtime_error("Incomplete isotope in ion stopping-power table: " +
                                     path.string());
        }
        result.species_present_[species] = 1;
    }
    if (normalize_to_file_carbon) {
        const auto carbon12_species =
            std::size_t{6} * mass_stride + std::size_t{12};
        if (result.species_present_[carbon12_species] == 0) {
            throw std::runtime_error(
                "Material ion stopping-power table lacks C-12: " +
                path.string());
        }
        const auto carbon12_base =
            carbon12_species * result.energy_grid_size_;
        for (std::size_t species = 0; species < species_slots; ++species) {
            if (result.species_present_[species] == 0) {
                continue;
            }
            const auto base = species * result.energy_grid_size_;
            for (std::size_t energy = 0;
                 energy < result.energy_grid_size_; ++energy) {
                const auto material_carbon_ratio =
                    result.ratios_to_carbon_[carbon12_base + energy];
                if (!std::isfinite(material_carbon_ratio) ||
                    material_carbon_ratio <= 0.0F) {
                    throw std::runtime_error(
                        "Invalid material C-12 stopping-power ratio in " +
                        path.string());
                }
                result.ratios_to_carbon_[base + energy] /=
                    material_carbon_ratio;
            }
        }
    }
    return result;
}

const std::vector<float>& IonStoppingPowerTables::ratios_to_carbon() const noexcept {
    return ratios_to_carbon_;
}

const std::vector<float>&
IonStoppingPowerTables::delta_electron_fractions() const noexcept {
    return delta_electron_fractions_;
}

const std::vector<std::uint8_t>&
IonStoppingPowerTables::species_present() const noexcept {
    return species_present_;
}

std::size_t IonStoppingPowerTables::energy_grid_size() const noexcept {
    return energy_grid_size_;
}

std::vector<float> load_hu_stopping_power_lut(
    const std::filesystem::path& path,
    const std::size_t n_sections,
    const std::size_t table_size,
    const float scale) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open HU stopping power LUT: " + path.string());
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(line[first]))) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        std::vector<float> row;
        float value = 0.0F;
        while (parser >> value) {
            row.push_back(value);
        }
        if (row.size() != table_size) {
            throw std::runtime_error(
                "HU stopping power LUT file " + path.string() +
                " has a row with " + std::to_string(row.size()) +
                " columns; expected " + std::to_string(table_size));
        }
        rows.push_back(std::move(row));
    }
    std::size_t first_section_row = 0;
    if (rows.size() == n_sections + 1U) {
        // Generated TOPAS calibration files carry the transport energy grid
        // in their first numeric row. It is metadata, not Schneider section 0.
        first_section_row = 1;
    } else if (rows.size() != n_sections) {
        throw std::runtime_error(
            "HU stopping power LUT file " + path.string() + " contains " +
            std::to_string(rows.size()) + " numeric rows; expected " +
            std::to_string(n_sections) + " section rows, optionally preceded "
            "by one energy-grid row");
    }
    std::vector<float> lut(n_sections * table_size, 0.0F);
    for (std::size_t section = 0; section < n_sections; ++section) {
        const auto& row = rows[first_section_row + section];
        std::transform(row.begin(), row.end(),
                       lut.begin() + static_cast<std::ptrdiff_t>(section * table_size),
                       [scale](const float value) { return value * scale; });
    }
    return lut;
}


DensityMassSprLut build_density_mass_spr_lut(
    const StoppingPowerTable& water,
    const StoppingPowerTable& air,
    const StoppingPowerTable& lung,
    const StoppingPowerTable& bone,
    const float scale) {
    constexpr std::uint32_t n_rho = 48;
    constexpr float rho_min = 0.00120479F;
    constexpr float rho_max = 3.0F;
    constexpr float rho_air = 0.00120479F;
    constexpr float rho_lung_table = 1.04F;
    constexpr float rho_bone = 1.85F;
    constexpr float rho_lung_knot = 0.26F;
    constexpr float rho_soft_lo = 0.90F;
    constexpr float rho_soft_hi = 1.20F;

    DensityMassSprLut out;
    out.n_rho = n_rho;
    out.log_rho_min = std::log(rho_min);
    const auto log_span = std::log(rho_max) - out.log_rho_min;
    out.inv_dlog = static_cast<float>(n_rho - 1U) / log_span;

    const auto& energies = water.energies();
    const auto n_energy = energies.size();
    out.factors.assign(static_cast<std::size_t>(n_rho) * n_energy, scale);

    auto mass_spr = [&](const StoppingPowerTable& table, const float rho_ref,
                        const double energy) {
        const auto water_sp = water.interpolate(energy);
        const auto table_sp = table.interpolate(energy);
        if (!(water_sp > 0.0) || !(rho_ref > 0.0F)) {
            return 1.0F;
        }
        return static_cast<float>(table_sp / (static_cast<double>(rho_ref) * water_sp));
    };

    auto lerp = [](const float x, const float x0, const float x1, const float y0,
                   const float y1) {
        if (x1 <= x0) {
            return y1;
        }
        const auto t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    };

    for (std::uint32_t i = 0; i < n_rho; ++i) {
        const auto rho = std::exp(out.log_rho_min + static_cast<float>(i) / out.inv_dlog);
        for (std::size_t e = 0; e < n_energy; ++e) {
            const auto energy = energies[e];
            const auto fs_air = mass_spr(air, rho_air, energy);
            const auto fs_lung = mass_spr(lung, rho_lung_table, energy);
            const auto fs_water = 1.0F;
            const auto fs_bone = mass_spr(bone, rho_bone, energy);
            float fs = fs_water;
            if (rho <= rho_lung_knot) {
                fs = lerp(rho, rho_air, rho_lung_knot, fs_air, fs_lung);
            } else if (rho < rho_soft_lo) {
                fs = lerp(rho, rho_lung_knot, rho_soft_lo, fs_lung, fs_water);
            } else if (rho <= rho_soft_hi) {
                fs = fs_water;
            } else if (rho < rho_bone) {
                fs = lerp(rho, rho_soft_hi, rho_bone, fs_water, fs_bone);
            } else {
                fs = fs_bone;
            }
            if (fs < 0.5F) {
                fs = 0.5F;
            }
            if (fs > 1.5F) {
                fs = 1.5F;
            }
            out.factors[static_cast<std::size_t>(i) * n_energy + e] = scale * fs;
        }
    }
    return out;
}

std::vector<float> load_ion_species_stopping_power_lut(
    const std::filesystem::path& path,
    const std::size_t table_size,
    const float scale) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Ion stopping-power file does not exist: " + path.string());
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ion stopping-power file: " + path.string());
    }

    // 18 charged species (indices 0..17 from get_charged_species_idx).
    std::vector<float> lut(18 * table_size, 0.0F);
    std::vector<std::size_t> samples_count(18, 0);

    const auto get_species_idx = [](int z, int a) -> int {
        if (z == 1) {
            if (a == 1) return 0;  // 1H (p)
            if (a == 2) return 1;  // 2H (d)
            if (a == 3) return 2;  // 3H (t)
            return -1;
        }
        if (z == 2) {
            if (a == 3) return 3;  // 3He
            if (a == 4) return 4;  // 4He (alpha)
            if (a == 6) return 5;  // 6He
            return -1;
        }
        if (z == 3) {
            if (a == 6) return 6;  // 6Li
            if (a == 7) return 7;  // 7Li
            return -1;
        }
        if (z == 4) {
            if (a == 7) return 8;  // 7Be
            if (a == 9) return 9;  // 9Be
            if (a == 10) return 10; // 10Be
            if (a == 6) return 17; // 6Be
            return -1;
        }
        if (z == 5) {
            if (a == 8) return 11;  // 8B
            if (a == 10) return 12; // 10B
            if (a == 11) return 13; // 11B
            return -1;
        }
        if (z == 6) {
            if (a == 10) return 14; // 10C
            if (a == 11) return 15; // 11C
            if (a == 12) return 16; // 12C
            return -1;
        }
        return -1;
    };

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '#') continue;
        if (line.rfind("atomic_number", 0) == 0) continue;
        std::stringstream ss(line);
        std::string token;
        int atomic_number = 0, mass_number = 0;
        double energy_MeVu = 0.0, stopping_power = 0.0;
        if (std::getline(ss, token, ',')) atomic_number = std::stoi(token);
        if (std::getline(ss, token, ',')) mass_number = std::stoi(token);
        if (std::getline(ss, token, ',')) energy_MeVu = std::stod(token);
        if (std::getline(ss, token, ',')) stopping_power = std::stod(token);

        const auto sp_idx = get_species_idx(atomic_number, mass_number);
        if (sp_idx >= 0 && sp_idx < 18) {
            const auto e_idx = samples_count[sp_idx];
            if (e_idx < table_size) {
                const double expected_e = 0.01 + static_cast<double>(e_idx) * 0.1;
                if (std::abs(energy_MeVu - expected_e) > 0.005) {
                    throw std::runtime_error(
                        "Ion species (Z=" + std::to_string(atomic_number) +
                        ", A=" + std::to_string(mass_number) + ") grid mismatch at index " +
                        std::to_string(e_idx) + ": found " + std::to_string(energy_MeVu) +
                        ", expected " + std::to_string(expected_e));
                }
                lut[static_cast<std::size_t>(sp_idx) * table_size + e_idx] =
                    static_cast<float>(stopping_power * scale);
                samples_count[sp_idx]++;
            } else {
                throw std::runtime_error("Ion species (Z=" + std::to_string(atomic_number) +
                                         ", A=" + std::to_string(mass_number) +
                                         ") has excess samples in " + path.string());
            }
        }
    }

    for (std::size_t sp = 0; sp < 18; ++sp) {
        if (samples_count[sp] != table_size) {
            throw std::runtime_error("Ion species index " + std::to_string(sp) +
                                     " in " + path.string() + " has " +
                                     std::to_string(samples_count[sp]) +
                                     " samples, expected " + std::to_string(table_size));
        }
    }
    return lut;
}

UrbanLossRangeTable::UrbanLossRangeTable(std::vector<double> e_total_mev,
                                         std::vector<double> range_mm,
                                         std::vector<double> dedx_mev_per_mm,
                                         double max_inverse_residual_mev,
                                         double zeff, double radlen_mm,
                                         double density_g_per_cm3,
                                         std::string oracle_status,
                                         std::string physics_list,
                                         std::string particle,
                                         std::string material,
                                         std::string range_source,
                                         std::string inverse_source,
                                         std::string dedx_source,
                                         double production_cut_mm)
    : e_total_mev_(std::move(e_total_mev)),
      range_mm_(std::move(range_mm)),
      dedx_mev_per_mm_(std::move(dedx_mev_per_mm)),
      max_inverse_residual_mev_(max_inverse_residual_mev),
      zeff_(zeff),
      radlen_mm_(radlen_mm),
      density_g_per_cm3_(density_g_per_cm3),
      oracle_status_(std::move(oracle_status)),
      physics_list_(std::move(physics_list)),
      particle_(std::move(particle)),
      material_(std::move(material)),
      range_source_(std::move(range_source)),
      inverse_source_(std::move(inverse_source)),
      dedx_source_(std::move(dedx_source)),
      production_cut_mm_(production_cut_mm) {}

UrbanLossRangeTable UrbanLossRangeTable::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open loss-range table: " + path.string());
    }
    std::vector<double> energies;
    std::vector<double> ranges;
    std::vector<double> dedx;
    double max_residual = 0.0;
    double zeff = std::numeric_limits<double>::quiet_NaN();
    double radlen = std::numeric_limits<double>::quiet_NaN();
    double density = std::numeric_limits<double>::quiet_NaN();
    double production_cut = std::numeric_limits<double>::quiet_NaN();
    std::string oracle_status;
    std::string physics_list;
    std::string particle;
    std::string material;
    std::string range_source;
    std::string inverse_source;
    std::string dedx_source;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            const auto metadata_line = first == std::string::npos
                                           ? std::string{}
                                           : line.substr(first + 1);
            std::istringstream meta(metadata_line);
            std::string key;
            if (meta >> key) {
                std::string value;
                std::getline(meta, value);
                const auto value_first = value.find_first_not_of(" \t");
                if (value_first != std::string::npos) {
                    value.erase(0, value_first);
                } else {
                    value.clear();
                }
                if (key == "ORACLE_STATUS") oracle_status = value;
                if (key == "physics_list") physics_list = value;
                if (key == "particle") particle = value;
                if (key == "material") material = value;
                if (key == "range_source") range_source = value;
                if (key == "inverse_source") inverse_source = value;
                if (key == "dedx_source") dedx_source = value;
                if (key == "zeff") zeff = std::stod(value);
                if (key == "radlen_mm") radlen = std::stod(value);
                if (key == "density_g_per_cm3") density = std::stod(value);
                if (key == "production_cut_mm") production_cut = std::stod(value);
            }
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(line[first]))) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        double e_u = 0.0, e_tot = 0.0, r = 0.0, d = 0.0, res = 0.0;
        if (!(parser >> e_u >> e_tot >> r >> d >> res)) {
            throw std::runtime_error("Invalid loss-range row at " + path.string() +
                                     ":" + std::to_string(line_number));
        }
        (void)e_u;
        energies.push_back(e_tot);
        ranges.push_back(r);
        dedx.push_back(d);
        max_residual = std::max(max_residual, std::fabs(res));
    }
    if (energies.size() < 2) {
        throw std::runtime_error("Loss-range table too short: " + path.string());
    }
    for (std::size_t i = 1; i < energies.size(); ++i) {
        if (!(energies[i] > energies[i - 1]) || !(ranges[i] > ranges[i - 1])) {
            throw std::runtime_error("Loss-range table not strictly increasing at row " +
                                     std::to_string(i) + ": " + path.string());
        }
    }
    return UrbanLossRangeTable(std::move(energies), std::move(ranges),
                               std::move(dedx), max_residual, zeff, radlen,
                               density, std::move(oracle_status),
                               std::move(physics_list), std::move(particle),
                               std::move(material), std::move(range_source),
                               std::move(inverse_source), std::move(dedx_source),
                               production_cut);
}

const std::vector<double>& UrbanLossRangeTable::energies_total_mev() const noexcept {
    return e_total_mev_;
}

const std::vector<double>& UrbanLossRangeTable::ranges_mm() const noexcept {
    return range_mm_;
}

const std::vector<double>& UrbanLossRangeTable::dedx_values() const noexcept {
    return dedx_mev_per_mm_;
}

double UrbanLossRangeTable::max_inverse_residual_mev() const noexcept {
    return max_inverse_residual_mev_;
}

double UrbanLossRangeTable::zeff() const noexcept {
    return zeff_;
}

double UrbanLossRangeTable::radlen_mm() const noexcept {
    return radlen_mm_;
}

double UrbanLossRangeTable::density_g_per_cm3() const noexcept {
    return density_g_per_cm3_;
}

const std::string& UrbanLossRangeTable::oracle_status() const noexcept {
    return oracle_status_;
}

const std::string& UrbanLossRangeTable::physics_list() const noexcept {
    return physics_list_;
}

const std::string& UrbanLossRangeTable::particle() const noexcept {
    return particle_;
}

const std::string& UrbanLossRangeTable::material() const noexcept {
    return material_;
}

const std::string& UrbanLossRangeTable::range_source() const noexcept {
    return range_source_;
}

const std::string& UrbanLossRangeTable::inverse_source() const noexcept {
    return inverse_source_;
}

const std::string& UrbanLossRangeTable::dedx_source() const noexcept {
    return dedx_source_;
}

double UrbanLossRangeTable::production_cut_mm() const noexcept {
    return production_cut_mm_;
}

bool UrbanLossRangeTable::is_active_c12_reference(
    const std::string& expected_material, const double expected_cut_mm) const noexcept {
    constexpr double cut_tolerance_mm = 1.0e-9;
    return oracle_status_ == "VALID_ACTIVE_C12_STEP_CONTEXT" &&
           physics_list_ == "G4EmStandardPhysics_option4" &&
           particle_ == "C12_Z6_A12_charge6" &&
           material_ == expected_material &&
           range_source_ ==
               "active_UrbanMsc_GetRange_bound_ionIoni_GetRange" &&
           inverse_source_ ==
               "active_UrbanMsc_GetEnergy_and_ionIoni_GetKineticEnergy" &&
           dedx_source_ ==
               "active_UrbanMsc_GetDEDX_bound_ionIoni_GetDEDX_restricted" &&
           std::isfinite(production_cut_mm_) &&
           std::fabs(production_cut_mm_ - expected_cut_mm) <= cut_tolerance_mm;
}

}  // namespace carbon
