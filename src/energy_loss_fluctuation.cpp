#include "carbon/energy_loss_fluctuation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace carbon {
namespace {

struct DistributionRow {
    int atomic_number{};
    int mass_number{};
    std::string material;
    double energy_MeVu{};
    double areal_density_g_per_cm2{};
    std::vector<double> quantiles;
};

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(trim(field));
    return fields;
}

double parse_number(const std::string& field,
                    const std::filesystem::path& path,
                    const std::size_t line_number,
                    const char* label) {
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(field, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Invalid " + std::string(label) + " at " + path.string() + ":" +
            std::to_string(line_number));
    }
    if (parsed != field.size() || !std::isfinite(value)) {
        throw std::runtime_error(
            "Invalid " + std::string(label) + " at " + path.string() + ":" +
            std::to_string(line_number));
    }
    return value;
}

int parse_positive_integer(const std::string& field,
                           const std::filesystem::path& path,
                           const std::size_t line_number,
                           const char* label) {
    const auto value = parse_number(field, path, line_number, label);
    if (value < 1.0 || value > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value) != value) {
        throw std::runtime_error(
            std::string(label) + " must be a positive integer at " +
            path.string() + ":" + std::to_string(line_number));
    }
    return static_cast<int>(value);
}

}  // namespace

EnergyLossFluctuationTable EnergyLossFluctuationTable::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Cannot open energy-loss fluctuation package: " + path.string());
    }

    constexpr std::size_t fixed_columns = 5;
    constexpr const char* expected_prefix =
        "projectile_Z,projectile_A,material,energy_MeV_per_u,"
        "areal_density_g_per_cm2";
    std::vector<double> probabilities;
    std::vector<DistributionRow> rows;
    bool found_header = false;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (!found_header) {
            if (fields.size() < fixed_columns + 3) {
                throw std::runtime_error(
                    "Energy-loss fluctuation CSV requires at least three quantiles: " +
                    path.string());
            }
            const auto prefix_fields = split_csv(expected_prefix);
            if (!std::equal(prefix_fields.begin(), prefix_fields.end(), fields.begin())) {
                throw std::runtime_error(
                    "Energy-loss fluctuation CSV has an unexpected identity/grid header: " +
                    path.string());
            }
            for (std::size_t index = fixed_columns; index < fields.size(); ++index) {
                if (!fields[index].starts_with("q_")) {
                    throw std::runtime_error(
                        "Energy-loss fluctuation quantile columns must use q_<probability>: " +
                        path.string());
                }
                probabilities.push_back(parse_number(
                    fields[index].substr(2), path, line_number, "quantile probability"));
            }
            if (probabilities.front() != 0.0 || probabilities.back() != 1.0) {
                throw std::runtime_error(
                    "Energy-loss fluctuation probabilities must include exact 0 and 1 endpoints: " +
                    path.string());
            }
            for (std::size_t index = 1; index < probabilities.size(); ++index) {
                if (probabilities[index] <= probabilities[index - 1] ||
                    probabilities[index] > 1.0) {
                    throw std::runtime_error(
                        "Energy-loss fluctuation probabilities must be strictly increasing in [0,1]: " +
                        path.string());
                }
            }
            found_header = true;
            continue;
        }
        if (fields.size() != fixed_columns + probabilities.size()) {
            throw std::runtime_error(
                "Incomplete energy-loss fluctuation row at " + path.string() + ":" +
                std::to_string(line_number));
        }
        DistributionRow row;
        row.atomic_number = parse_positive_integer(
            fields[0], path, line_number, "projectile_Z");
        row.mass_number = parse_positive_integer(
            fields[1], path, line_number, "projectile_A");
        row.material = fields[2];
        row.energy_MeVu = parse_number(
            fields[3], path, line_number, "energy_MeV_per_u");
        row.areal_density_g_per_cm2 = parse_number(
            fields[4], path, line_number, "areal_density_g_per_cm2");
        if (row.mass_number < row.atomic_number || row.material.empty() ||
            row.energy_MeVu <= 0.0 || row.areal_density_g_per_cm2 <= 0.0) {
            throw std::runtime_error(
                "Invalid fluctuation identity or grid coordinate at " + path.string() + ":" +
                std::to_string(line_number));
        }
        for (std::size_t index = fixed_columns; index < fields.size(); ++index) {
            const auto quantile = parse_number(
                fields[index], path, line_number, "loss-ratio quantile");
            if (quantile < 0.0 ||
                (!row.quantiles.empty() && quantile < row.quantiles.back())) {
                throw std::runtime_error(
                    "Loss-ratio quantiles must be finite, nonnegative, and nondecreasing at " +
                    path.string() + ":" + std::to_string(line_number));
            }
            row.quantiles.push_back(quantile);
        }
        double integrated_mean = 0.0;
        for (std::size_t index = 1; index < probabilities.size(); ++index) {
            integrated_mean += 0.5 * (row.quantiles[index - 1] + row.quantiles[index]) *
                               (probabilities[index] - probabilities[index - 1]);
        }
        if (std::abs(integrated_mean - 1.0) > 0.001) {
            throw std::runtime_error(
                "Loss-ratio quantiles must integrate to unit mean within 0.1% at " +
                path.string() + ":" + std::to_string(line_number));
        }
        rows.push_back(std::move(row));
    }
    if (!found_header || rows.empty()) {
        throw std::runtime_error(
            "Energy-loss fluctuation CSV has no data rows: " + path.string());
    }

    const auto atomic_number = rows.front().atomic_number;
    const auto mass_number = rows.front().mass_number;
    const auto material = rows.front().material;
    std::vector<double> energies;
    std::vector<double> areal_densities;
    for (const auto& row : rows) {
        if (row.atomic_number != atomic_number || row.mass_number != mass_number ||
            row.material != material) {
            throw std::runtime_error(
                "Energy-loss fluctuation package must contain one projectile/material identity: " +
                path.string());
        }
        energies.push_back(row.energy_MeVu);
        areal_densities.push_back(row.areal_density_g_per_cm2);
    }
    std::sort(energies.begin(), energies.end());
    energies.erase(std::unique(energies.begin(), energies.end()), energies.end());
    std::sort(areal_densities.begin(), areal_densities.end());
    areal_densities.erase(
        std::unique(areal_densities.begin(), areal_densities.end()),
        areal_densities.end());
    if (energies.size() < 2 || areal_densities.size() < 2) {
        throw std::runtime_error(
            "Energy-loss fluctuation package must provide at least two energies and two areal densities: " +
            path.string());
    }

    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.energy_MeVu, lhs.areal_density_g_per_cm2) <
               std::tie(rhs.energy_MeVu, rhs.areal_density_g_per_cm2);
    });
    for (std::size_t index = 1; index < rows.size(); ++index) {
        if (rows[index].energy_MeVu == rows[index - 1].energy_MeVu &&
            rows[index].areal_density_g_per_cm2 ==
                rows[index - 1].areal_density_g_per_cm2) {
            throw std::runtime_error(
                "Energy-loss fluctuation package contains a duplicate grid point: " +
                path.string());
        }
    }

    EnergyLossFluctuationTable result;
    result.projectile_atomic_number_ = atomic_number;
    result.projectile_mass_number_ = mass_number;
    result.material_name_ = material;
    result.energies_MeVu_ = std::move(energies);
    result.areal_densities_g_per_cm2_ = std::move(areal_densities);
    result.probabilities_ = std::move(probabilities);
    result.loss_ratio_quantiles_.reserve(
        result.energies_MeVu_.size() *
        result.areal_densities_g_per_cm2_.size() *
        result.probabilities_.size());

    // Different energies may use different sampled thicknesses. Expand each
    // local density grid onto their union so the existing rectangular device
    // sampler has the same result as a ragged lookup.
    std::size_t begin = 0;
    for (const auto energy : result.energies_MeVu_) {
        const auto end = std::find_if(
            rows.begin() + static_cast<std::ptrdiff_t>(begin), rows.end(),
            [energy](const auto& row) { return row.energy_MeVu != energy; });
        const auto end_index = static_cast<std::size_t>(end - rows.begin());
        if (end_index - begin < 2) {
            throw std::runtime_error(
                "Each fluctuation energy requires at least two areal-density points: " +
                path.string());
        }
        for (const auto target_density : result.areal_densities_g_per_cm2_) {
            const auto local_begin =
                rows.begin() + static_cast<std::ptrdiff_t>(begin);
            const auto upper = std::upper_bound(
                local_begin, end, target_density,
                [](const double density, const DistributionRow& row) {
                    return density < row.areal_density_g_per_cm2;
                });
            const DistributionRow* lower_row = nullptr;
            const DistributionRow* upper_row = nullptr;
            if (upper == local_begin) {
                lower_row = &*upper;
                upper_row = &*upper;
            } else if (upper == end) {
                lower_row = &*(end - 1);
                upper_row = &*(end - 1);
            } else {
                lower_row = &*(upper - 1);
                upper_row = &*upper;
            }
            const auto fraction = lower_row == upper_row
                ? 0.0
                : (target_density - lower_row->areal_density_g_per_cm2) /
                      (upper_row->areal_density_g_per_cm2 -
                       lower_row->areal_density_g_per_cm2);
            for (std::size_t probability = 0;
                 probability < result.probabilities_.size(); ++probability) {
                result.loss_ratio_quantiles_.push_back(
                    lower_row->quantiles[probability] +
                    fraction * (upper_row->quantiles[probability] -
                                lower_row->quantiles[probability]));
            }
        }
        begin = end_index;
    }
    return result;
}

double EnergyLossFluctuationTable::sample_loss_ratio(
    const double energy_MeVu,
    const double areal_density_g_per_cm2,
    const double uniform) const noexcept {
    return sample_energy_loss_ratio_from_grid(
        energies_MeVu_.data(), energies_MeVu_.size(),
        areal_densities_g_per_cm2_.data(),
        areal_densities_g_per_cm2_.size(), probabilities_.data(),
        probabilities_.size(), loss_ratio_quantiles_.data(), energy_MeVu,
        areal_density_g_per_cm2, uniform);
}

int EnergyLossFluctuationTable::projectile_atomic_number() const noexcept {
    return projectile_atomic_number_;
}

int EnergyLossFluctuationTable::projectile_mass_number() const noexcept {
    return projectile_mass_number_;
}

const std::string& EnergyLossFluctuationTable::material_name() const noexcept {
    return material_name_;
}

const std::vector<double>& EnergyLossFluctuationTable::energies_MeVu() const noexcept {
    return energies_MeVu_;
}

const std::vector<double>&
EnergyLossFluctuationTable::areal_densities_g_per_cm2() const noexcept {
    return areal_densities_g_per_cm2_;
}

const std::vector<double>& EnergyLossFluctuationTable::probabilities() const noexcept {
    return probabilities_;
}

const std::vector<double>&
EnergyLossFluctuationTable::loss_ratio_quantiles() const noexcept {
    return loss_ratio_quantiles_;
}

}  // namespace carbon
