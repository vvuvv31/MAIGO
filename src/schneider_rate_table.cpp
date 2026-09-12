#include "carbon/schneider_rate_table.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace carbon {

std::uint32_t schneider_rate_binary_version(const std::filesystem::path& binary_path) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider rate binary file: " + binary_path.string());
    }
    char magic[8]{};
    std::uint32_t version{0};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || in.gcount() != static_cast<std::streamsize>(sizeof(version))) {
        throw std::runtime_error("Truncated header in Schneider rate binary: " + binary_path.string());
    }
    if (std::strncmp(magic, "SCHNRATE", 8) != 0) {
        throw std::runtime_error("Invalid magic in Schneider rate binary: " + binary_path.string());
    }
    return version;
}

std::size_t SchneiderRateTable::target_index_from_z(int target_z) {
    for (std::size_t i = 0; i < kSchneiderCanonicalZ.size(); ++i) {
        if (kSchneiderCanonicalZ[i] == target_z) {
            return i;
        }
    }
    throw std::invalid_argument("Unsupported Schneider canonical target Z: " + std::to_string(target_z));
}

SchneiderRateTable SchneiderRateTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path,
    const std::string& expected_magic,
    std::uint32_t expected_version) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider rate binary file: " + binary_path.string());
    }

    SchneiderRateHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || in.gcount() != sizeof(header)) {
        throw std::runtime_error("Truncated header in Schneider rate binary: " + binary_path.string());
    }

    if (std::strncmp(header.magic, expected_magic.c_str(), 8) != 0 ||
        std::strlen(expected_magic.c_str()) != 8) {
        throw std::runtime_error("Invalid magic in Schneider rate binary: " + binary_path.string());
    }
    // Only the current domain-masked rate schema is supported.
    if (header.version != expected_version) {
        throw std::runtime_error("Schneider rate version mismatch in: " + binary_path.string());
    }
    if (header.num_sections != kSchneiderNumSections) {
        throw std::runtime_error("Invalid section count in: " + binary_path.string());
    }
    if (header.num_targets != kSchneiderNumTargets) {
        throw std::runtime_error("Invalid target count in: " + binary_path.string());
    }
    const std::size_t expect_energies = 921;
    const double expect_emin = 0.1;
    const double expect_emax = 460.1;
    if (header.num_energies != expect_energies) {
        throw std::runtime_error("Invalid energy count in: " + binary_path.string());
    }
    if (std::abs(header.energy_min_mevu - expect_emin) > 1e-6 ||
        std::abs(header.energy_max_mevu - expect_emax) > 1e-6 ||
        std::abs(header.energy_step_mevu - 0.5) > 1e-6) {
        throw std::runtime_error("Invalid energy grid bounds in: " + binary_path.string());
    }

    for (std::size_t i = 0; i < kSchneiderNumTargets; ++i) {
        if (header.target_z[i] != kSchneiderCanonicalZ[i]) {
            throw std::runtime_error("Canonical target Z order mismatch in: " + binary_path.string());
        }
    }

    SchneiderRateTable table;
    table.binary_version_ = header.version;
    table.num_energies_ = header.num_energies;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    const std::size_t ne = table.num_energies_;
    const std::size_t partial_elements = kSchneiderNumSections * kSchneiderNumTargets * ne;
    table.mass_partial_rates_.resize(partial_elements);
    in.read(reinterpret_cast<char*>(table.mass_partial_rates_.data()),
            partial_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(partial_elements * sizeof(double))) {
        throw std::runtime_error("Truncated partial rate payload in: " + binary_path.string());
    }

    const std::size_t total_elements = kSchneiderNumSections * ne;
    table.mass_total_rates_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.mass_total_rates_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated total rate payload in: " + binary_path.string());
    }

    {
        table.channel_domains_.resize(kSchneiderNumTargets);
        in.read(reinterpret_cast<char*>(table.channel_domains_.data()),
                kSchneiderNumTargets * sizeof(SchneiderRateDomainEntry));
        if (!in ||
            in.gcount() != static_cast<std::streamsize>(kSchneiderNumTargets * sizeof(SchneiderRateDomainEntry))) {
            throw std::runtime_error("Truncated v3 domain block in: " + binary_path.string());
        }
        for (std::size_t t = 0; t < kSchneiderNumTargets; ++t) {
            const auto& d = table.channel_domains_[t];
            if (!std::isfinite(d.energy_min_mevu) || !std::isfinite(d.energy_max_mevu)) {
                throw std::runtime_error("Non-finite v3 domain bounds at target index " + std::to_string(t));
            }
            if (d.has_support != 0 && d.has_support != 1) {
                throw std::runtime_error("Invalid v3 has_support flag at target index " + std::to_string(t));
            }
            if (d.energy_min_mevu > d.energy_max_mevu) {
                throw std::runtime_error("Inverted v3 domain bounds at target index " + std::to_string(t));
            }
            if (d.has_support == 1 &&
                (d.energy_min_mevu < table.energy_min_mevu_ - 1e-6 ||
                 d.energy_max_mevu > table.energy_max_mevu_ + 1e-6)) {
                throw std::runtime_error("v3 supported domain outside global grid at target index " +
                                         std::to_string(t));
            }
        }
        // v3 closure: stored totals are masked sums; require bitwise equality
        // with the ascending-target masked sum (same order as the compiler).
        for (std::size_t s = 0; s < kSchneiderNumSections; ++s) {
            for (std::size_t e = 0; e < ne; ++e) {
                double part_sum = 0.0;
                for (std::size_t t = 0; t < kSchneiderNumTargets; ++t) {
                    part_sum += table.masked_partial_at_node(s, t, e);
                }
                const double tot = table.mass_total_rate(s, e);
                if (part_sum != tot) {
                    throw std::runtime_error("Schneider rate v3 masked-total mismatch for sec=" +
                                             std::to_string(s) + " e=" + std::to_string(e));
                }
            }
        }
        // Exact file size: header + partials + totals + domain block.
        const std::size_t expected_v3 =
            sizeof(SchneiderRateHeader) +
            partial_elements * sizeof(double) + total_elements * sizeof(double) +
            kSchneiderNumTargets * sizeof(SchneiderRateDomainEntry);
        if (std::filesystem::file_size(binary_path) != expected_v3) {
            throw std::runtime_error("Schneider rate v3 file size mismatch (truncated or trailing bytes)");
        }
    }

    // Companion metadata requires the full schema with binary<->metadata equality on
    // filename, SHA, magic, version, target order, grid, and domains.
    std::filesystem::path resolved_meta = metadata_path;
    if (resolved_meta.empty()) {
        const std::string s = binary_path.string();
        if (s.size() > 4 && s.substr(s.size() - 4) == ".bin") {
            resolved_meta = s.substr(0, s.size() - 4) + ".metadata.json";
        }
    }
    if (!std::filesystem::exists(resolved_meta)) {
        throw std::runtime_error("SchneiderRateTable: companion metadata file is required but missing for: " +
                                 binary_path.string());
    }
    {
        std::ifstream meta_in(resolved_meta);
        if (!meta_in) {
            throw std::runtime_error("SchneiderRateTable: cannot open metadata file: " + resolved_meta.string());
        }
        std::string meta_content((std::istreambuf_iterator<char>(meta_in)),
                                 std::istreambuf_iterator<char>());
        {
            minjson::Parser parser(meta_content);
            const minjson::Value meta = parser.parse();
            const std::string data_filename =
                minjson::require_string(meta.at("data_filename"), "data_filename");
            if (data_filename != binary_path.filename().string()) {
                throw std::runtime_error("SchneiderRateTable: metadata data_filename '" + data_filename +
                                         "' does not match binary basename '" +
                                         binary_path.filename().string() + "'");
            }
            const std::string expected_sha =
                minjson::require_string(meta.at("data_sha256"), "data_sha256");
            const std::string actual_sha = compute_file_sha256_hex(binary_path);
            if (actual_sha != expected_sha) {
                throw std::runtime_error("SchneiderRateTable: SHA-256 mismatch for " + binary_path.string());
            }
            const std::string magic = minjson::require_string(meta.at("binary_magic"), "binary_magic");
            if (magic != expected_magic) {
                throw std::runtime_error("SchneiderRateTable: metadata binary_magic mismatch: " + magic);
            }
            if (minjson::require_uint(meta.at("binary_version"), "binary_version") != header.version) {
                throw std::runtime_error("SchneiderRateTable: metadata binary_version mismatch");
            }
            const minjson::Value& targets = meta.at("target_order");
            if (targets.type != minjson::Value::Type::Array ||
                targets.arr.size() != kSchneiderNumTargets) {
                throw std::runtime_error("SchneiderRateTable: metadata target_order size mismatch");
            }
            for (std::size_t i = 0; i < kSchneiderNumTargets; ++i) {
                const int z = static_cast<int>(minjson::require_uint(targets.arr[i], "target_order[]"));
                if (z != kSchneiderCanonicalZ[i]) {
                    throw std::runtime_error("SchneiderRateTable: metadata target_order mismatch at index " +
                                             std::to_string(i));
                }
            }
            const minjson::Value& grid = meta.at("energy_grid");
            if (std::abs(minjson::require_number(grid.at("emin"), "energy_grid.emin") -
                         table.energy_min_mevu_) > 1e-9 ||
                std::abs(minjson::require_number(grid.at("emax"), "energy_grid.emax") -
                         table.energy_max_mevu_) > 1e-9 ||
                std::abs(minjson::require_number(grid.at("step"), "energy_grid.step") -
                         table.energy_step_mevu_) > 1e-9 ||
                minjson::require_uint(grid.at("count"), "energy_grid.count") != table.num_energies_) {
                throw std::runtime_error("SchneiderRateTable: metadata energy_grid mismatch");
            }
            const minjson::Value& doms = meta.at("channel_domains");
            if (doms.type != minjson::Value::Type::Array ||
                doms.arr.size() != kSchneiderNumTargets) {
                throw std::runtime_error("SchneiderRateTable: metadata channel_domains size mismatch");
            }
            for (std::size_t t = 0; t < kSchneiderNumTargets; ++t) {
                const auto& e = doms.arr[t];
                const int tz = static_cast<int>(minjson::require_uint(e.at("target_z"), "domain.tz"));
                const bool has = e.at("has_support").type == minjson::Value::Type::Boolean
                                     ? e.at("has_support").boolean
                                     : throw std::runtime_error(
                                           "SchneiderRateTable: domain has_support must be boolean");
                const double emin = minjson::require_number(e.at("energy_min_mevu"), "domain.emin");
                const double emax = minjson::require_number(e.at("energy_max_mevu"), "domain.emax");
                const auto& bin = table.channel_domains_[t];
                if (tz != kSchneiderCanonicalZ[t] || (has ? 1 : 0) != bin.has_support ||
                    emin != bin.energy_min_mevu || emax != bin.energy_max_mevu) {
                    throw std::runtime_error("SchneiderRateTable: metadata channel_domains mismatch at index " +
                                             std::to_string(t));
                }
            }
            for (const char* field :
                 {"physics_list", "topas_version", "geant4_version", "schneider_sha256", "compiler_commit"}) {
                if (!meta.contains(field)) {
                    throw std::runtime_error(std::string("SchneiderRateTable: metadata missing '") + field +
                                             "': " + resolved_meta.string());
                }
            }
        }
    }

    return table;
}

double SchneiderRateTable::mass_partial_rate(
    std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const {
    if (section_id >= kSchneiderNumSections || target_idx >= kSchneiderNumTargets || energy_idx >= num_energies_) {
        throw std::out_of_range("SchneiderRateTable index out of range");
    }
    const std::size_t idx = section_id * (kSchneiderNumTargets * num_energies_) +
                            target_idx * num_energies_ +
                            energy_idx;
    return mass_partial_rates_[idx];
}

double SchneiderRateTable::mass_total_rate(
    std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderNumSections || energy_idx >= num_energies_) {
        throw std::out_of_range("SchneiderRateTable index out of range");
    }
    const std::size_t idx = section_id * num_energies_ + energy_idx;
    return mass_total_rates_[idx];
}

double SchneiderRateTable::interpolate_mass_partial(
    std::size_t section_id, std::size_t target_idx, double energy_mevu) const {
    if (section_id >= kSchneiderNumSections || target_idx >= kSchneiderNumTargets) {
        throw std::out_of_range("SchneiderRateTable section or target index out of range");
    }
    // v3: mask before interpolation; exactly 0 outside the channel domain.
    const SchneiderRateDomainEntry& dom = channel_domain(target_idx);
    if (dom.has_support == 0 || !(energy_mevu >= energy_min_mevu_) ||
        !(energy_mevu <= energy_max_mevu_) || energy_mevu < dom.energy_min_mevu ||
        energy_mevu > dom.energy_max_mevu) {
        return 0.0;
    }
    return interpolate_masked_partial(section_id, target_idx, energy_mevu);
}

double SchneiderRateTable::interpolate_mass_total(
    std::size_t section_id, double energy_mevu) const {
    if (section_id >= kSchneiderNumSections) {
        throw std::out_of_range("SchneiderRateTable section index out of range");
    }
    // v3: host mirror of the device mask; total = sum of masked partials.
    if (!(energy_mevu >= energy_min_mevu_) || !(energy_mevu <= energy_max_mevu_)) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t t = 0; t < kSchneiderNumTargets; ++t) {
        sum += interpolate_mass_partial(section_id, t, energy_mevu);
    }
    // Hazard/sampler unity with the device (float-sliver guard).
    if (!(sum > 1e-12)) {
        return 0.0;
    }
    return sum;
}

double SchneiderRateTable::interpolate_masked_partial(std::size_t section_id, std::size_t target_idx,
                                                      double energy_mevu) const {
    const double frac_idx = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    std::size_t lower_idx = static_cast<std::size_t>(std::floor(frac_idx));
    if (lower_idx >= num_energies_ - 1) {
        lower_idx = num_energies_ - 2;
    }
    const std::size_t upper_idx = lower_idx + 1;
    const double alpha = frac_idx - static_cast<double>(lower_idx);
    const double y0 = mass_partial_rate(section_id, target_idx, lower_idx);
    const double y1 = mass_partial_rate(section_id, target_idx, upper_idx);
    double v = (1.0 - alpha) * y0 + alpha * y1;
    if (v < 0.0) {
        v = 0.0;
    }
    return v;
}

}  // namespace carbon
