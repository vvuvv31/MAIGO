#include "carbon/schneider_delta_tail.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    return fields;
}

}  // namespace

SchneiderDeltaTailTable SchneiderDeltaTailTable::from_csv(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open Schneider delta-tail table: " + path.string());

    SchneiderDeltaTailTable out;
    std::string line;
    bool saw_header = false;
    float previous_energy = -1.0F;
    float current_energy = -1.0F;
    float current_fraction = -1.0F;
    std::vector<float> row_quantiles;
    std::vector<float> row_radii;
    auto flush_row = [&]() {
        if (row_quantiles.empty()) return;
        if (out.quantiles_.empty()) {
            out.quantiles_ = row_quantiles;
        } else if (row_quantiles != out.quantiles_) {
            throw std::invalid_argument("Schneider delta-tail quantile grids are not rectangular");
        }
        out.energies_MeV_per_u_.push_back(current_energy);
        out.moved_fractions_.push_back(current_fraction);
        out.radii_mm_.insert(out.radii_mm_.end(), row_radii.begin(), row_radii.end());
        previous_energy = current_energy;
        row_quantiles.clear();
        row_radii.clear();
    };

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!saw_header) {
            if (line != "energy_MeV_per_u,moved_fraction,quantile,radius_mm") {
                throw std::invalid_argument("Unexpected Schneider delta-tail CSV header");
            }
            saw_header = true;
            continue;
        }
        const auto fields = split_csv(line);
        if (fields.size() != 4) throw std::invalid_argument("Malformed Schneider delta-tail CSV row");
        const auto energy = std::stof(fields[0]);
        const auto fraction = std::stof(fields[1]);
        const auto quantile = std::stof(fields[2]);
        const auto radius = std::stof(fields[3]);
        if (!std::isfinite(energy) || !std::isfinite(fraction) ||
            !std::isfinite(quantile) || !std::isfinite(radius) || energy <= 0.0F ||
            fraction <= 0.0F || fraction >= 0.5F || quantile < 0.0F ||
            quantile > 1.0F || radius < 0.0F) {
            throw std::invalid_argument("Nonphysical Schneider delta-tail CSV value");
        }
        if (row_quantiles.empty() || energy != current_energy) {
            flush_row();
            if (energy <= previous_energy) {
                throw std::invalid_argument("Schneider delta-tail energies must increase");
            }
            current_energy = energy;
            current_fraction = fraction;
        } else if (fraction != current_fraction) {
            throw std::invalid_argument("Moved fraction changes within a delta-tail energy row");
        }
        if (!row_quantiles.empty() && quantile <= row_quantiles.back()) {
            throw std::invalid_argument("Schneider delta-tail quantiles must increase");
        }
        if (!row_radii.empty() && radius < row_radii.back()) {
            throw std::invalid_argument("Schneider delta-tail radii must be nondecreasing");
        }
        row_quantiles.push_back(quantile);
        row_radii.push_back(radius);
    }
    flush_row();
    if (!saw_header || out.energy_count() == 0 || out.quantile_count() < 2 ||
        out.quantiles_.front() != 0.0F || out.quantiles_.back() != 1.0F) {
        throw std::invalid_argument("Incomplete Schneider delta-tail table");
    }
    for (std::size_t i = 0; i < out.quantile_count(); ++i) {
        const auto expected = static_cast<float>(i) /
                              static_cast<float>(out.quantile_count() - 1);
        if (std::abs(out.quantiles_[i] - expected) > 1.0e-6F) {
            throw std::invalid_argument(
                "Schneider delta-tail quantiles must use a uniform [0,1] grid");
        }
    }
    return out;
}

}  // namespace carbon
