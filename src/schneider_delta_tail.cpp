#include "carbon/schneider_delta_tail.hpp"
#include "carbon/sha256.hpp"
#include "carbon/min_json.hpp"

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

SchneiderLongitudinalTable SchneiderLongitudinalTable::from_csv(
    const std::filesystem::path& path) {
    // Immutable candidate pins, NOT an upgrade of the validated CT stack.
    const auto metadata = path.parent_path() / (path.stem().string() + ".metadata.json");
    const auto manifest = path.parent_path() / (path.stem().string() + ".candidate.json");
    const std::string data_sha =
        "f42140bc6a99ca90ef9a7fc267c22be9a6d8002192644dbe45b7a40c4e10820b";
    const std::string metadata_sha =
        "32a37a8235d75954ea020685cd659be2ed6c7d766fad5725eaabdae91035339a";
    if (compute_file_sha256_hex(path) != data_sha ||
        compute_file_sha256_hex(metadata) != metadata_sha ||
        std::filesystem::file_size(path) != 128U) {
        throw std::invalid_argument("Longitudinal candidate CSV/metadata SHA or size mismatch");
    }
    std::ifstream manifest_input(manifest);
    if (!manifest_input) throw std::invalid_argument("Missing longitudinal candidate manifest");
    const auto contract = minjson::Parser(std::string(
        std::istreambuf_iterator<char>(manifest_input), std::istreambuf_iterator<char>())).parse();
    if (contract.at("schema_version").number != 1 ||
        contract.at("status").str != "unvalidated_diagnostic" ||
        contract.at("data_sha256").str != data_sha ||
        contract.at("metadata_sha256").str != metadata_sha ||
        contract.at("scale").number != 1 ||
        contract.at("energy_min_MeV_u").number != 150 ||
        contract.at("energy_max_MeV_u").number != 225 ||
        contract.at("reference_density_g_cm3").number != kLongitudinalReferenceDensityGPerCm3 ||
        contract.at("kernel").str != "homogeneous_only_exact_voxel_segments_v1") {
        throw std::invalid_argument("Invalid longitudinal candidate contract");
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Cannot open Schneider longitudinal table: " + path.string());
    }
    SchneiderLongitudinalTable out;
    std::string line;
    bool saw_header = false;
    float previous_energy = -1.0F;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!saw_header) {
            if (line != "energy_MeV_per_u,forward_fraction,lambda_mm") {
                throw std::invalid_argument(
                    "Unexpected Schneider longitudinal CSV header");
            }
            saw_header = true;
            continue;
        }
        const auto fields = split_csv(line);
        if (fields.size() != 3) {
            throw std::invalid_argument("Malformed Schneider longitudinal CSV row");
        }
        const auto energy = std::stof(fields[0]);
        const auto fraction = std::stof(fields[1]);
        const auto lambda = std::stof(fields[2]);
        if (!std::isfinite(energy) || !std::isfinite(fraction) ||
            !std::isfinite(lambda) || energy <= 0.0F || fraction <= 0.0F ||
            fraction >= 0.5F || lambda <= 0.0F) {
            throw std::invalid_argument(
                "Nonphysical Schneider longitudinal CSV value");
        }
        if (energy <= previous_energy) {
            throw std::invalid_argument(
                "Schneider longitudinal energies must increase");
        }
        previous_energy = energy;
        out.energies_MeV_per_u_.push_back(energy);
        out.forward_fractions_.push_back(fraction);
        out.lambdas_mm_.push_back(lambda);
    }
    if (!saw_header || out.energy_count() == 0) {
        throw std::invalid_argument("Incomplete Schneider longitudinal table");
    }
    return out;
}

}  // namespace carbon
