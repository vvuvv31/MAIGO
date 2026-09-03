#include "carbon/schneider_stopping_table.hpp"
#include "carbon/sha256.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace carbon {

namespace {

constexpr std::array<std::string_view, 25> kCanonicalSchneiderCompositionHashes = {{
    "9a5e655bc67c888f1ba3ad3532a6114dcb69e1054738c46bb86996e571e32a43", // sec 0: PatientTissueFromHUNegative975
    "d85f60883acd556b0d54a7b05eda07dae753c5d7d3d49c5e777eff2228c01acb", // sec 1: PatientTissueFromHUNegative535
    "c979ebe99d85d61ee6e6dd657bcd594ee5aa17a95c86af8d383e23182fc34fa8", // sec 2: PatientTissueFromHUNegative102
    "b754d221929076adf060245ea9df9fc802f8a1cc850f4784ffacdf35fcbc96f9", // sec 3: PatientTissueFromHUNegative68
    "180fbed0d50a66b0e680d4fa0a77b34b359418c08d1d86af5395893bb9d5282b", // sec 4: PatientTissueFromHUNegative38
    "1340176e48a2ee30bb74de2bc44277716aa73eb89daa3869c505f36f3c1a9f4c", // sec 5: PatientTissueFromHUNegative8
    "4feda65b1b00754db48b3c90ab5a2eddf48163209fcb838ce4fb27645a2ac9d9", // sec 6: PatientTissueFromHU12
    "3224284d45c417ea1a1153495a5de59800e6f4f31a0b93e1cfaa833533a8ff5a", // sec 7: PatientTissueFromHU49
    "bb0aa41d37f7f932c7e79d1c6aecde17efe27b28e6c28bd959dc2c0d6b9d1ffe", // sec 8: PatientTissueFromHU100
    "5cf10aa24ee479b9c76b3ee379e974bafebdf1365b961eb94df0d2c63db02876", // sec 9: PatientTissueFromHU160
    "4cfe354b748faaff62581c2e2c77500e68317f8e2df30764eb196d21928bb850", // sec 10: PatientTissueFromHU250
    "274aa82d5a4eeb248f472707e7edf9e375be74c4a51772f3cd3ebe23adcee114", // sec 11: PatientTissueFromHU350
    "84596ab01c4e118e60b57712082c25d7d667b15b5f9b78535071eb2ee4abce45", // sec 12: PatientTissueFromHU450
    "b6c5e7b66fa67d76de71dbc46633b6aabf34656eb40f5f0eb88a69a3b1bacfce", // sec 13: PatientTissueFromHU550
    "1986e360a3f3151c014a848d06cbab774e13eb2481b068b2b3dd613404b35838", // sec 14: PatientTissueFromHU650
    "e5eff8bf427eed45554ff967d3593377a04d854c9813c92c37b83bf8e0cf5161", // sec 15: PatientTissueFromHU750
    "4c05cb552f9c4a3d0925ead691e00c3856faf61f27c39d841b7756219429fd7c", // sec 16: PatientTissueFromHU850
    "ebecc0310944118f64263e9a91b183dbfbcf00fe845e583cbcf4901831570d50", // sec 17: PatientTissueFromHU950
    "27a8942e42e3d45c11b65efee97fd33e1ede99e76a7574618a3a40b79d270a0a", // sec 18: PatientTissueFromHU1050
    "6b1d932fcff7507eb0c79f9e23f09aefc22c4d1b208fc6bfdfcb9140791024d4", // sec 19: PatientTissueFromHU1150
    "1947d9de30f6e1b8723ddcd38cca594782857cd8e8c08552fc055d795de7a66f", // sec 20: PatientTissueFromHU1250
    "0b3645352552795265980ed78fa9541d63071c0c886088c0b3ffa4a7e5b79bbd", // sec 21: PatientTissueFromHU1350
    "515d7e2f04bce4e6c7e217a7bb2e3d2d991dc94a038709346cf6b21d4f736ffa", // sec 22: PatientTissueFromHU1450
    "bdef38f6ba2806369cff1d54bcd467fb6f9cd08446852dbd6da04530e6230572", // sec 23: PatientTissueFromHU2247
    "4ed07b9386302fddbdbed3b43bcd979ffd64077ec5c23560d778bd1719d32e58"  // sec 24: PatientTissueFromHU2995
}};

std::string extract_json_field(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\": \"";
    const auto pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    const auto start = pos + pattern.size();
    const auto end = json.find('"', start);
    if (end == std::string::npos) return {};
    return json.substr(start, end - start);
}

}  // namespace

SchneiderStoppingTable SchneiderStoppingTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider stopping binary file: " + binary_path.string());
    }

    SchneiderStoppingHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || in.gcount() != sizeof(header)) {
        throw std::runtime_error("Truncated header in Schneider stopping binary: " + binary_path.string());
    }

    if (std::strncmp(header.magic, "SCHNSTOP", 8) != 0) {
        throw std::runtime_error("Invalid magic in Schneider stopping binary: " + binary_path.string());
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported binary version in: " + binary_path.string());
    }
    if (header.num_sections != kSchneiderStoppingNumSections) {
        throw std::runtime_error("Invalid section count in: " + binary_path.string());
    }
    if (header.num_energies != kSchneiderStoppingNumEnergies) {
        throw std::runtime_error("Invalid energy count in: " + binary_path.string());
    }
    if (std::abs(header.energy_min_mevu - kSchneiderStoppingEnergyMin) > 1e-6 ||
        std::abs(header.energy_max_mevu - kSchneiderStoppingEnergyMax) > 1e-6 ||
        std::abs(header.energy_step_mevu - kSchneiderStoppingEnergyStep) > 1e-6) {
        throw std::runtime_error("Invalid energy grid bounds in: " + binary_path.string());
    }

    SchneiderStoppingTable table;
    table.energy_min_mevu_ = header.energy_min_mevu;
    table.energy_max_mevu_ = header.energy_max_mevu;
    table.energy_step_mevu_ = header.energy_step_mevu;

    table.densities_.resize(kSchneiderStoppingNumSections);
    in.read(reinterpret_cast<char*>(table.densities_.data()),
            kSchneiderStoppingNumSections * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(kSchneiderStoppingNumSections * sizeof(double))) {
        throw std::runtime_error("Truncated densities in: " + binary_path.string());
    }

    const std::size_t total_elements = kSchneiderStoppingNumSections * kSchneiderStoppingNumEnergies;
    table.mass_stopping_powers_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.mass_stopping_powers_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated mass stopping powers in: " + binary_path.string());
    }

    table.csda_ranges_mm_.resize(total_elements);
    in.read(reinterpret_cast<char*>(table.csda_ranges_mm_.data()),
            total_elements * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total_elements * sizeof(double))) {
        throw std::runtime_error("Truncated CSDA ranges in: " + binary_path.string());
    }

    // Strict structural metadata validation
    const auto effective_meta_path = metadata_path.empty()
        ? (binary_path.parent_path() / (binary_path.stem().string() + ".metadata.json"))
        : metadata_path;

    if (!std::filesystem::exists(effective_meta_path)) {
        throw std::runtime_error(
            "Schneider stopping binary requires companion metadata sidecar: " + effective_meta_path.string());
    }

    std::ifstream meta_in(effective_meta_path);
    if (!meta_in) {
        throw std::runtime_error("Cannot open metadata sidecar: " + effective_meta_path.string());
    }
    const std::string meta_content((std::istreambuf_iterator<char>(meta_in)),
                                   std::istreambuf_iterator<char>());

    const auto recorded_sha = extract_json_field(meta_content, "data_sha256");
    if (recorded_sha.empty()) {
        throw std::runtime_error("Missing data_sha256 in metadata sidecar: " + effective_meta_path.string());
    }
    const auto actual_bin_sha = compute_file_sha256_hex(binary_path);
    if (recorded_sha != actual_bin_sha) {
        throw std::runtime_error("Schneider stopping binary SHA256 mismatch: recorded=" + recorded_sha +
                                 ", actual=" + actual_bin_sha);
    }

    // Section Identity & Composition Hash Contract: verify all 25 sections in sequence
    for (std::size_t s = 0; s < kSchneiderStoppingNumSections; ++s) {
        const std::string sec_pattern = "\"section_id\": " + std::to_string(s) + ",";
        const auto sec_pos = meta_content.find(sec_pattern);
        if (sec_pos == std::string::npos) {
            throw std::runtime_error("Metadata missing section_id " + std::to_string(s) + " in " + effective_meta_path.string());
        }

        const auto comp_hash = kCanonicalSchneiderCompositionHashes[s];
        const std::string hash_pattern = "\"composition_sha256\": \"" + std::string(comp_hash) + "\"";
        const auto hash_pos = meta_content.find(hash_pattern, sec_pos);
        if (hash_pos == std::string::npos || hash_pos > sec_pos + 400) {
            throw std::runtime_error(
                "Schneider stopping table section identity / composition permutation detected at section " +
                std::to_string(s) + " (expected composition hash " + std::string(comp_hash) + ")");
        }
    }

    return table;
}

SchneiderStoppingTable SchneiderStoppingTable::from_csv(
    const std::filesystem::path& csv_path) {
    std::ifstream in(csv_path);
    if (!in) {
        throw std::runtime_error("Cannot open Schneider stopping CSV file: " + csv_path.string());
    }

    SchneiderStoppingTable table;
    table.densities_.assign(kSchneiderStoppingNumSections, 0.0);
    const std::size_t total_elements = kSchneiderStoppingNumSections * kSchneiderStoppingNumEnergies;
    table.mass_stopping_powers_.assign(total_elements, 0.0);
    table.csda_ranges_mm_.assign(total_elements, 0.0);

    std::string line;
    std::size_t line_num = 0;
    while (std::getline(in, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#' || line.rfind("energy_mevu", 0) == 0) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double energy_mevu = 0.0;
        std::size_t sec_id = 0;
        std::string mat_name;
        double density = 0.0;
        double mass_sp = 0.0;
        double linear_sp = 0.0;
        double csda_r = 0.0;

        if (!(iss >> energy_mevu >> sec_id >> mat_name >> density >> mass_sp >> linear_sp >> csda_r)) {
            throw std::runtime_error("Malformed row in " + csv_path.string() + ":" + std::to_string(line_num));
        }

        if (sec_id >= kSchneiderStoppingNumSections) {
            throw std::out_of_range("Section id out of range in " + csv_path.string() + ":" + std::to_string(line_num));
        }

        const auto e_idx = static_cast<std::size_t>(
            std::round((energy_mevu - kSchneiderStoppingEnergyMin) / kSchneiderStoppingEnergyStep));
        if (e_idx >= kSchneiderStoppingNumEnergies) {
            continue;
        }

        table.densities_[sec_id] = density;
        const std::size_t offset = sec_id * kSchneiderStoppingNumEnergies + e_idx;
        table.mass_stopping_powers_[offset] = mass_sp;
        table.csda_ranges_mm_[offset] = csda_r;
    }

    return table;
}

double SchneiderStoppingTable::density(std::size_t section_id) const {
    if (section_id >= kSchneiderStoppingNumSections) {
        throw std::out_of_range("Schneider section_id out of range: " + std::to_string(section_id));
    }
    return densities_[section_id];
}

double SchneiderStoppingTable::mass_stopping_power(std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderStoppingNumSections || energy_idx >= kSchneiderStoppingNumEnergies) {
        throw std::out_of_range("Schneider stopping index out of range");
    }
    return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies + energy_idx];
}

double SchneiderStoppingTable::csda_range_mm(std::size_t section_id, std::size_t energy_idx) const {
    if (section_id >= kSchneiderStoppingNumSections || energy_idx >= kSchneiderStoppingNumEnergies) {
        throw std::out_of_range("Schneider stopping index out of range");
    }
    return csda_ranges_mm_[section_id * kSchneiderStoppingNumEnergies + energy_idx];
}

double SchneiderStoppingTable::interpolate_mass_stopping(std::size_t section_id, double energy_mevu) const {
    if (section_id >= kSchneiderStoppingNumSections) {
        throw std::out_of_range("Schneider section_id out of range: " + std::to_string(section_id));
    }
    if (energy_mevu <= energy_min_mevu_) {
        return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies];
    }
    if (energy_mevu >= energy_max_mevu_) {
        return mass_stopping_powers_[section_id * kSchneiderStoppingNumEnergies + (kSchneiderStoppingNumEnergies - 1)];
    }

    const double frac_idx = (energy_mevu - energy_min_mevu_) / energy_step_mevu_;
    const auto idx = static_cast<std::size_t>(std::floor(frac_idx));
    const double frac = frac_idx - static_cast<double>(idx);

    const auto base = section_id * kSchneiderStoppingNumEnergies;
    return mass_stopping_powers_[base + idx] + frac * (mass_stopping_powers_[base + idx + 1] - mass_stopping_powers_[base + idx]);
}

std::vector<float> SchneiderStoppingTable::to_flat_mass_stopping_float() const {
    std::vector<float> result(mass_stopping_powers_.size());
    for (std::size_t i = 0; i < mass_stopping_powers_.size(); ++i) {
        result[i] = static_cast<float>(mass_stopping_powers_[i]);
    }
    return result;
}

}  // namespace carbon
