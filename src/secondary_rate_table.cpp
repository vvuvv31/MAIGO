#include "carbon/secondary_rate_table.hpp"
#include "carbon/min_json.hpp"
#include "carbon/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace carbon {

SecondaryRateTable SecondaryRateTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    if (!std::filesystem::exists(binary_path)) {
        throw std::runtime_error("Secondary rate binary file does not exist: " + binary_path.string());
    }

    const auto file_sz = std::filesystem::file_size(binary_path);
    if (file_sz < sizeof(SecondaryRateHeader)) {
        throw std::runtime_error("Secondary rate binary file too small for header: " + binary_path.string());
    }

    std::ifstream file(binary_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open secondary rate binary: " + binary_path.string());
    }

    SecondaryRateHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(SecondaryRateHeader));
    if (!file) {
        throw std::runtime_error("Failed to read header from: " + binary_path.string());
    }

    if (std::memcmp(header.magic, "SCHN2RAT", 8) != 0) {
        throw std::runtime_error("Invalid magic in secondary rate binary: " + binary_path.string());
    }
    // Version 1: frozen [0.5, 430.0]/860 grid, 13 projectiles, legacy
    // endpoint clamp. Version 3: v2.1 data product (variable projectile
    // count, data-driven registry) with an appended per-(projectile,target)
    // valid-domain block and masked totals. Version 2 was a retired
    // intermediate that never shipped; it is rejected explicitly.
    if (header.version == 2) {
        throw std::runtime_error("Retired version 2 in secondary rate binary (never shipped): " + binary_path.string());
    }
    const bool is_v3 = (header.version == 3);
    if (header.version != 1 && !is_v3) {
        throw std::runtime_error("Unsupported version in secondary rate binary: " + std::to_string(header.version));
    }
    if (!is_v3 && header.num_projectiles != kSecondaryNumProjectiles) {
        throw std::runtime_error("Invalid num_projectiles in secondary rate binary: " + std::to_string(header.num_projectiles));
    }
    if (is_v3 && (header.num_projectiles == 0 || header.num_projectiles > 64)) {
        throw std::runtime_error("Invalid num_projectiles in v3 secondary rate binary: " + std::to_string(header.num_projectiles));
    }
    if (header.num_sections != kSecondaryNumSections) {
        throw std::runtime_error("Invalid num_sections in secondary rate binary: " + std::to_string(header.num_sections));
    }
    if (header.num_targets != kSecondaryNumTargets) {
        throw std::runtime_error("Invalid num_targets in secondary rate binary: " + std::to_string(header.num_targets));
    }
    const std::size_t expect_energies = is_v3 ? kSecondaryV2NumEnergies : kSecondaryNumEnergies;
    const double expect_emin = is_v3 ? kSecondaryV2EnergyMin : kSecondaryV1EnergyMin;
    const double expect_emax = is_v3 ? kSecondaryV2EnergyMax : kSecondaryV1EnergyMax;
    if (header.num_energies != expect_energies) {
        throw std::runtime_error("Invalid num_energies in secondary rate binary: " + std::to_string(header.num_energies));
    }
    if (!std::isfinite(header.energy_min_mevu) || !std::isfinite(header.energy_max_mevu) || !std::isfinite(header.energy_step_mevu)) {
        throw std::runtime_error("Non-finite energy grid specification in header: " + binary_path.string());
    }
    if (std::abs(header.energy_min_mevu - expect_emin) > 1e-5 ||
        std::abs(header.energy_max_mevu - expect_emax) > 1e-5 ||
        std::abs(header.energy_step_mevu - kSecondaryEnergyStep) > 1e-5) {
        throw std::runtime_error("Unexpected energy grid range/step in header: " + binary_path.string());
    }

    // Overflow-safe size computation
    const std::uint64_t np64 = header.num_projectiles;
    const std::uint64_t ns64 = header.num_sections;
    const std::uint64_t nt64 = header.num_targets;
    const std::uint64_t ne64 = header.num_energies;

    const std::uint64_t num_partial64 = np64 * ns64 * nt64 * ne64;
    const std::uint64_t num_total64 = np64 * ns64 * ne64;
    constexpr std::uint64_t max_sz = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (num_partial64 > max_sz || num_total64 > max_sz) {
        throw std::overflow_error("Secondary rate table dimensions exceed size_t bounds");
    }

    const std::size_t num_partial = static_cast<std::size_t>(num_partial64);
    const std::size_t num_total = static_cast<std::size_t>(num_total64);
    std::size_t expected_size = sizeof(SecondaryRateHeader) +
                                    header.num_projectiles * sizeof(SecondaryProjectileKey) +
                                    header.num_targets * sizeof(int32_t) +
                                    num_partial * sizeof(double) +
                                    num_total * sizeof(double);
    if (is_v3) {
        constexpr std::uint64_t max_dom =
            std::numeric_limits<std::size_t>::max() / sizeof(SecondaryRateDomainEntry);
        if (static_cast<std::uint64_t>(header.num_projectiles) *
                static_cast<std::uint64_t>(header.num_targets) >
            max_dom) {
            throw std::overflow_error("Secondary rate v3 domain block exceeds size_t bounds");
        }
        expected_size += static_cast<std::size_t>(header.num_projectiles) *
                         static_cast<std::size_t>(header.num_targets) *
                         sizeof(SecondaryRateDomainEntry);
    }
    if (file_sz != expected_size) {
        throw std::runtime_error("SecondaryRateTable: file size mismatch (truncated or trailing bytes): expected " +
                                 std::to_string(expected_size) + " bytes, got " + std::to_string(file_sz));
    }

    SecondaryRateTable table;
    table.num_projectiles_ = header.num_projectiles;
    table.binary_version_ = header.version;
    table.num_energies_ = header.num_energies;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    table.projectiles_.resize(header.num_projectiles);
    file.read(reinterpret_cast<char*>(table.projectiles_.data()), header.num_projectiles * sizeof(SecondaryProjectileKey));
    if (!file) {
        throw std::runtime_error("Failed to read projectiles from secondary rate binary: " + binary_path.string());
    }

    for (std::size_t i = 0; i < header.num_projectiles; ++i) {
        const auto& p = table.projectiles_[i];
        if (p.z <= 0 || p.a <= 0 || p.z > 100 || p.a > 250) {
            throw std::runtime_error("Invalid projectile Z/A at index " + std::to_string(i));
        }
        for (std::size_t j = i + 1; j < header.num_projectiles; ++j) {
            if (p.z == table.projectiles_[j].z && p.a == table.projectiles_[j].a) {
                throw std::runtime_error("Duplicate projectile in table: Z=" + std::to_string(p.z) + " A=" + std::to_string(p.a));
            }
        }
    }

    file.read(reinterpret_cast<char*>(table.canonical_targets_.data()), 13 * sizeof(int32_t));
    if (!file) {
        throw std::runtime_error("Failed to read canonical targets from secondary rate binary: " + binary_path.string());
    }
    constexpr std::array<int32_t, 13> expected_targets = {
        1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
    };
    for (std::size_t i = 0; i < 13; ++i) {
        if (table.canonical_targets_[i] != expected_targets[i]) {
            throw std::runtime_error("Canonical target ordering mismatch at index " + std::to_string(i));
        }
    }

    table.mass_partial_rates_.resize(num_partial);
    file.read(reinterpret_cast<char*>(table.mass_partial_rates_.data()), num_partial * sizeof(double));

    table.mass_total_rates_.resize(num_total);
    file.read(reinterpret_cast<char*>(table.mass_total_rates_.data()), num_total * sizeof(double));

    if (!file) {
        throw std::runtime_error("Failed to read full payload from secondary rate binary: " + binary_path.string());
    }

    if (is_v3) {
        const std::size_t num_domains =
            static_cast<std::size_t>(header.num_projectiles) *
            static_cast<std::size_t>(header.num_targets);
        table.channel_domains_.resize(num_domains);
        file.read(reinterpret_cast<char*>(table.channel_domains_.data()),
                  num_domains * sizeof(SecondaryRateDomainEntry));
        if (!file) {
            throw std::runtime_error("Failed to read v3 domain block from secondary rate binary: " +
                                     binary_path.string());
        }
        // v3 domain validity: finite bounds, ordered, inside the global grid
        // (1e-6 tolerance), support flag binary.
        for (std::size_t i = 0; i < num_domains; ++i) {
            const auto& d = table.channel_domains_[i];
            if (!std::isfinite(d.energy_min_mevu) || !std::isfinite(d.energy_max_mevu)) {
                throw std::runtime_error("Non-finite v3 domain bounds at index " + std::to_string(i));
            }
            if (d.has_support != 0 && d.has_support != 1) {
                throw std::runtime_error("Invalid v3 has_support flag at index " + std::to_string(i));
            }
            if (d.energy_min_mevu > d.energy_max_mevu) {
                throw std::runtime_error("Inverted v3 domain bounds at index " + std::to_string(i));
            }
            if (d.has_support == 1 &&
                (d.energy_min_mevu < table.energy_min_mevu_ - 1e-6 ||
                 d.energy_max_mevu > table.energy_max_mevu_ + 1e-6)) {
                throw std::runtime_error("v3 supported domain outside global grid at index " +
                                         std::to_string(i));
            }
        }
    }

    // Physical validity: all rates must be non-negative and finite
    for (std::size_t i = 0; i < num_partial; ++i) {
        const double r = table.mass_partial_rates_[i];
        if (!std::isfinite(r) || r < 0.0) {
            throw std::runtime_error("Non-finite or negative partial rate at index " + std::to_string(i));
        }
    }
    for (std::size_t i = 0; i < num_total; ++i) {
        const double r = table.mass_total_rates_[i];
        if (!std::isfinite(r) || r < 0.0) {
            throw std::runtime_error("Non-finite or negative total rate at index " + std::to_string(i));
        }
    }

    // Consistency check: sum(partial) vs total. For v3 the stored totals
    // are masked sums, so the check sums masked node partials (ascending
    // target order, same as the compiler) and requires bitwise equality.
    for (std::size_t p = 0; p < header.num_projectiles; ++p) {
        for (std::size_t s = 0; s < header.num_sections; ++s) {
            for (std::size_t e = 0; e < header.num_energies; ++e) {
                double part_sum = 0.0;
                for (std::size_t t = 0; t < header.num_targets; ++t) {
                    part_sum += is_v3 ? table.masked_partial_at_node(p, s, t, e)
                                      : table.mass_partial_rate(p, s, t, e);
                }
                const double tot = table.mass_total_rate(p, s, e);
                if (is_v3) {
                    if (part_sum != tot) {
                        throw std::runtime_error(
                            "Secondary rate v3 masked-total mismatch for proj=" + std::to_string(p) +
                            " sec=" + std::to_string(s) + " e=" + std::to_string(e) +
                            ": sum(masked)=" + std::to_string(part_sum) +
                            " total=" + std::to_string(tot));
                    }
                    continue;
                }
                const double diff = std::abs(part_sum - tot);
                if (diff > 1e-4 * (tot + 1e-6)) {
                    throw std::runtime_error("Secondary rate consistency failure for proj=" + std::to_string(p) +
                                             " sec=" + std::to_string(s) + " e=" + std::to_string(e) +
                                             ": sum(partial)=" + std::to_string(part_sum) + " total=" + std::to_string(tot));
                }
            }
        }
    }

    // Companion metadata validation
    std::filesystem::path resolved_meta = metadata_path;
    if (resolved_meta.empty()) {
        std::string s = binary_path.string();
        if (s.size() > 4 && s.substr(s.size() - 4) == ".bin") {
            resolved_meta = s.substr(0, s.size() - 4) + ".metadata.json";
        }
        if (!std::filesystem::exists(resolved_meta)) {
            resolved_meta = binary_path.string() + ".metadata.json";
        }
    }

    // Companion metadata: REQUIRED (no silent loads). Strict schema:
    // data_filename == actual basename, data_sha256 verified, binary_magic
    // and binary_version equal the binary header, projectile list / target
    // order / energy grid / channel domains equal the binary content, plus
    // provenance presence (physics list, TOPAS/Geant4 versions, Schneider
    // hash, compiler commit). v1 metadata predates this schema: v1 loads
    // require only data_sha256 presence + match (frozen path unchanged).
    if (resolved_meta.empty() || !std::filesystem::exists(resolved_meta)) {
        throw std::runtime_error("SecondaryRateTable: companion metadata file is required but missing for: " +
                                 binary_path.string());
    }
    {
        std::ifstream meta_file(resolved_meta);
        if (!meta_file) {
            throw std::runtime_error("SecondaryRateTable: cannot open metadata file: " + resolved_meta.string());
        }
        std::string meta_content((std::istreambuf_iterator<char>(meta_file)),
                                 std::istreambuf_iterator<char>());
        if (!is_v3) {
            if (meta_content.find("\"data_sha256\"") == std::string::npos) {
                throw std::runtime_error("SecondaryRateTable: v1 metadata missing data_sha256: " +
                                         resolved_meta.string());
            }
        } else {
            minjson::Parser parser(meta_content);
            const minjson::Value meta = parser.parse();
            const std::string data_filename =
                minjson::require_string(meta.at("data_filename"), "data_filename");
            if (data_filename != binary_path.filename().string()) {
                throw std::runtime_error("SecondaryRateTable: metadata data_filename '" + data_filename +
                                         "' does not match binary basename '" +
                                         binary_path.filename().string() + "'");
            }
            const std::string expected_sha =
                minjson::require_string(meta.at("data_sha256"), "data_sha256");
            const std::string actual_sha = compute_file_sha256_hex(binary_path);
            if (actual_sha != expected_sha) {
                throw std::runtime_error("SecondaryRateTable: SHA-256 mismatch for " + binary_path.string() +
                                         ": expected " + expected_sha + ", got " + actual_sha);
            }
            const std::string magic = minjson::require_string(meta.at("binary_magic"), "binary_magic");
            if (magic != "SCHN2RAT") {
                throw std::runtime_error("SecondaryRateTable: metadata binary_magic mismatch: " + magic);
            }
            const auto meta_version = minjson::require_uint(meta.at("binary_version"), "binary_version");
            if (meta_version != header.version) {
                throw std::runtime_error("SecondaryRateTable: metadata binary_version mismatch");
            }
            const minjson::Value& projs = meta.at("projectiles");
            if (projs.type != minjson::Value::Type::Array ||
                projs.arr.size() != table.projectiles_.size()) {
                throw std::runtime_error("SecondaryRateTable: metadata projectile list size mismatch");
            }
            for (std::size_t i = 0; i < projs.arr.size(); ++i) {
                const int z = static_cast<int>(minjson::require_uint(projs.arr[i].at("z"), "projectiles[].z"));
                const int a = static_cast<int>(minjson::require_uint(projs.arr[i].at("a"), "projectiles[].a"));
                if (z != table.projectiles_[i].z || a != table.projectiles_[i].a) {
                    throw std::runtime_error("SecondaryRateTable: metadata projectile mismatch at index " +
                                             std::to_string(i));
                }
            }
            const minjson::Value& targets = meta.at("target_order");
            if (targets.type != minjson::Value::Type::Array || targets.arr.size() != 13) {
                throw std::runtime_error("SecondaryRateTable: metadata target_order size mismatch");
            }
            for (std::size_t i = 0; i < 13; ++i) {
                const int z = static_cast<int>(minjson::require_uint(targets.arr[i], "target_order[]"));
                if (z != table.canonical_targets_[i]) {
                    throw std::runtime_error("SecondaryRateTable: metadata target_order mismatch at index " +
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
                throw std::runtime_error("SecondaryRateTable: metadata energy_grid mismatch");
            }
            const minjson::Value& doms = meta.at("channel_domains");
            if (doms.type != minjson::Value::Type::Array ||
                doms.arr.size() != table.channel_domains_.size()) {
                throw std::runtime_error("SecondaryRateTable: metadata channel_domains size mismatch");
            }
            for (std::size_t i = 0; i < doms.arr.size(); ++i) {
                const std::size_t p = i / static_cast<std::size_t>(kSecondaryNumTargets);
                const std::size_t t = i % static_cast<std::size_t>(kSecondaryNumTargets);
                const auto& e = doms.arr[i];
                const int pz = static_cast<int>(minjson::require_uint(e.at("projectile_z"), "domain.pz"));
                const int pa = static_cast<int>(minjson::require_uint(e.at("projectile_a"), "domain.pa"));
                const int tz = static_cast<int>(minjson::require_uint(e.at("target_z"), "domain.tz"));
                const bool has = e.at("has_support").type == minjson::Value::Type::Boolean
                                     ? e.at("has_support").boolean
                                     : throw std::runtime_error(
                                           "SecondaryRateTable: domain has_support must be boolean");
                const double emin = minjson::require_number(e.at("energy_min_mevu"), "domain.emin");
                const double emax = minjson::require_number(e.at("energy_max_mevu"), "domain.emax");
                const auto& bin = table.channel_domains_[i];
                if (pz != table.projectiles_[p].z || pa != table.projectiles_[p].a ||
                    tz != table.canonical_targets_[t] ||
                    (has ? 1 : 0) != bin.has_support || emin != bin.energy_min_mevu ||
                    emax != bin.energy_max_mevu) {
                    throw std::runtime_error("SecondaryRateTable: metadata channel_domains mismatch at index " +
                                             std::to_string(i));
                }
            }
            for (const char* field :
                 {"physics_list", "topas_version", "geant4_version", "schneider_sha256", "compiler_commit"}) {
                if (!meta.contains(field)) {
                    throw std::runtime_error(std::string("SecondaryRateTable: metadata missing '") + field +
                                             "': " + resolved_meta.string());
                }
            }
        }
    }

    return table;
}

std::uint32_t secondary_rate_binary_version(const std::filesystem::path& binary_path) {
    std::ifstream file(binary_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open secondary rate binary file: " + binary_path.string());
    }
    char magic[8]{};
    std::uint32_t version{0};
    file.read(magic, 8);
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file || file.gcount() != static_cast<std::streamsize>(sizeof(version))) {
        throw std::runtime_error("Truncated header in secondary rate binary: " + binary_path.string());
    }
    if (std::memcmp(magic, "SCHN2RAT", 8) != 0) {
        throw std::runtime_error("Invalid magic in secondary rate binary: " + binary_path.string());
    }
    return version;
}

int SecondaryRateTable::projectile_index(int proj_z, int proj_a) const noexcept {
    for (std::size_t i = 0; i < projectiles_.size(); ++i) {
        if (projectiles_[i].z == proj_z && projectiles_[i].a == proj_a) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::size_t SecondaryRateTable::target_index_from_z(int target_z) {
    constexpr std::array<int32_t, 13> canonical_z = {
        1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
    };
    for (std::size_t i = 0; i < 13; ++i) {
        if (canonical_z[i] == target_z) {
            return i;
        }
    }
    throw std::invalid_argument("Unsupported target element Z: " + std::to_string(target_z));
}

double SecondaryRateTable::mass_partial_rate(std::size_t proj_idx, std::size_t section_id, std::size_t target_idx, std::size_t energy_idx) const {
    if (proj_idx >= num_projectiles_ || section_id >= kSecondaryNumSections ||
        target_idx >= kSecondaryNumTargets || energy_idx >= num_energies_) {
        throw std::out_of_range("SecondaryRateTable::mass_partial_rate: index out of range");
    }
    const std::size_t stride_proj = kSecondaryNumSections * kSecondaryNumTargets * num_energies_;
    const std::size_t stride_sec = kSecondaryNumTargets * num_energies_;
    const std::size_t stride_tgt = num_energies_;
    return mass_partial_rates_[proj_idx * stride_proj + section_id * stride_sec + target_idx * stride_tgt + energy_idx];
}

double SecondaryRateTable::mass_total_rate(std::size_t proj_idx, std::size_t section_id, std::size_t energy_idx) const {
    if (proj_idx >= num_projectiles_ || section_id >= kSecondaryNumSections || energy_idx >= num_energies_) {
        throw std::out_of_range("SecondaryRateTable::mass_total_rate: index out of range");
    }
    const std::size_t stride_proj = kSecondaryNumSections * num_energies_;
    const std::size_t stride_sec = num_energies_;
    return mass_total_rates_[proj_idx * stride_proj + section_id * stride_sec + energy_idx];
}

double SecondaryRateTable::interpolate_mass_total(std::size_t proj_idx, std::size_t section_id, double energy_mevu) const {
    if (proj_idx >= num_projectiles_ || section_id >= kSecondaryNumSections) {
        throw std::out_of_range("SecondaryRateTable::interpolate_mass_total: index out of range");
    }
    // v3 (binary version 3): host mirror of the device mask. The channel
    // domain is applied per target BEFORE interpolation; the total is the
    // sum of masked partials. Strict zero outside the global grid; no clamp,
    // no extrapolation. v1 keeps the legacy endpoint clamp EXACTLY.
    if (binary_version_ == 3) {
        if (!(energy_mevu >= energy_min_mevu_) || !(energy_mevu <= energy_max_mevu_)) {
            return 0.0;
        }
        const double node_flt = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
        std::size_t idx0 = static_cast<std::size_t>(node_flt);
        if (idx0 >= num_energies_ - 1) {
            idx0 = num_energies_ - 2;
        }
        const std::size_t idx1 = idx0 + 1;
        const double alpha = node_flt - static_cast<double>(idx0);
        double sum = 0.0;
        for (std::size_t t = 0; t < kSecondaryNumTargets; ++t) {
            const SecondaryRateDomainEntry& dom = channel_domain(proj_idx, t);
            if (dom.has_support == 0 || energy_mevu < dom.energy_min_mevu ||
                energy_mevu > dom.energy_max_mevu) {
                continue;
            }
            const double y0 = mass_partial_rate(proj_idx, section_id, t, idx0);
            const double y1 = mass_partial_rate(proj_idx, section_id, t, idx1);
            double v = (1.0 - alpha) * y0 + alpha * y1;
            if (v < 0.0) {
                v = 0.0;
            }
            sum += v;
        }
        // Hazard/sampler unity with the device (float-sliver guard).
        if (!(sum > 1e-12)) {
            return 0.0;
        }
        return sum;
    }
    if (energy_mevu <= energy_min_mevu_) {
        return mass_total_rate(proj_idx, section_id, 0);
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_total_rate(proj_idx, section_id, num_energies_ - 1);
    }
    const double node_flt = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const std::size_t idx0 = static_cast<std::size_t>(node_flt);
    const std::size_t idx1 = std::min(idx0 + 1, num_energies_ - 1);
    const double alpha = node_flt - static_cast<double>(idx0);

    const double y0 = mass_total_rate(proj_idx, section_id, idx0);
    const double y1 = mass_total_rate(proj_idx, section_id, idx1);
    return (1.0 - alpha) * y0 + alpha * y1;
}

}  // namespace carbon
