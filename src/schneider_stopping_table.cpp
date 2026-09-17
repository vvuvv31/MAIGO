#include "carbon/schneider_stopping_table.hpp"
#include "carbon/sha256.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

namespace {

constexpr std::array<std::string_view, 25> kCanonicalSchneiderMaterialNames = {{
    "PatientTissueFromHUNegative975",
    "PatientTissueFromHUNegative535",
    "PatientTissueFromHUNegative102",
    "PatientTissueFromHUNegative68",
    "PatientTissueFromHUNegative38",
    "PatientTissueFromHUNegative8",
    "PatientTissueFromHU12",
    "PatientTissueFromHU49",
    "PatientTissueFromHU100",
    "PatientTissueFromHU160",
    "PatientTissueFromHU250",
    "PatientTissueFromHU350",
    "PatientTissueFromHU450",
    "PatientTissueFromHU550",
    "PatientTissueFromHU650",
    "PatientTissueFromHU750",
    "PatientTissueFromHU850",
    "PatientTissueFromHU950",
    "PatientTissueFromHU1050",
    "PatientTissueFromHU1150",
    "PatientTissueFromHU1250",
    "PatientTissueFromHU1350",
    "PatientTissueFromHU1450",
    "PatientTissueFromHU2247",
    "PatientTissueFromHU2995"
}};

constexpr std::array<std::string_view, 25> kCanonicalSchneiderCompositionHashes = {{
    "9a5e655bc67c888f1ba3ad3532a6114dcb69e1054738c46bb86996e571e32a43", // sec 0
    "d85f60883acd556b0d54a7b05eda07dae753c5d7d3d49c5e777eff2228c01acb", // sec 1
    "c979ebe99d85d61ee6e6dd657bcd594ee5aa17a95c86af8d383e23182fc34fa8", // sec 2
    "b754d221929076adf060245ea9df9fc802f8a1cc850f4784ffacdf35fcbc96f9", // sec 3
    "180fbed0d50a66b0e680d4fa0a77b34b359418c08d1d86af5395893bb9d5282b", // sec 4
    "1340176e48a2ee30bb74de2bc44277716aa73eb89daa3869c505f36f3c1a9f4c", // sec 5
    "4feda65b1b00754db48b3c90ab5a2eddf48163209fcb838ce4fb27645a2ac9d9", // sec 6
    "3224284d45c417ea1a1153495a5de59800e6f4f31a0b93e1cfaa833533a8ff5a", // sec 7
    "bb0aa41d37f7f932c7e79d1c6aecde17efe27b28e6c28bd959dc2c0d6b9d1ffe", // sec 8
    "5cf10aa24ee479b9c76b3ee379e974bafebdf1365b961eb94df0d2c63db02876", // sec 9
    "4cfe354b748faaff62581c2e2c77500e68317f8e2df30764eb196d21928bb850", // sec 10
    "274aa82d5a4eeb248f472707e7edf9e375be74c4a51772f3cd3ebe23adcee114", // sec 11
    "84596ab01c4e118e60b57712082c25d7d667b15b5f9b78535071eb2ee4abce45", // sec 12
    "b6c5e7b66fa67d76de71dbc46633b6aabf34656eb40f5f0eb88a69a3b1bacfce", // sec 13
    "1986e360a3f3151c014a848d06cbab774e13eb2481b068b2b3dd613404b35838", // sec 14
    "e5eff8bf427eed45554ff967d3593377a04d854c9813c92c37b83bf8e0cf5161", // sec 15
    "4c05cb552f9c4a3d0925ead691e00c3856faf61f27c39d841b7756219429fd7c", // sec 16
    "ebecc0310944118f64263e9a91b183dbfbcf00fe845e583cbcf4901831570d50", // sec 17
    "27a8942e42e3d45c11b65efee97fd33e1ede99e76a7574618a3a40b79d270a0a", // sec 18
    "6b1d932fcff7507eb0c79f9e23f09aefc22c4d1b208fc6bfdfcb9140791024d4", // sec 19
    "1947d9de30f6e1b8723ddcd38cca594782857cd8e8c08552fc055d795de7a66f", // sec 20
    "0b3645352552795265980ed78fa9541d63071c0c886088c0b3ffa4a7e5b79bbd", // sec 21
    "515d7e2f04bce4e6c7e217a7bb2e3d2d991dc94a038709346cf6b21d4f736ffa", // sec 22
    "bdef38f6ba2806369cff1d54bcd467fb6f9cd08446852dbd6da04530e6230572", // sec 23
    "4ed07b9386302fddbdbed3b43bcd979ffd64077ec5c23560d778bd1719d32e58"  // sec 24
}};

struct JsonVal {
    enum Type { Null, String, Number, Boolean, Array, Object } type{Null};
    std::string str_val{};
    double num_val{0.0};
    bool bool_val{false};
    std::vector<JsonVal> arr{};
    std::map<std::string, JsonVal> obj{};

    bool has(const std::string& key) const {
        return obj.find(key) != obj.end();
    }
    const JsonVal& operator[](const std::string& key) const {
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("JSON key not found: '" + key + "'");
        return it->second;
    }
    const JsonVal& operator[](std::size_t idx) const {
        if (idx >= arr.size()) throw std::runtime_error("JSON array index out of bounds: " + std::to_string(idx));
        return arr[idx];
    }
};

std::size_t get_strict_uint(const JsonVal& val, const std::string& field_name) {
    if (val.type != JsonVal::Number) {
        throw std::runtime_error(field_name + " must be a number");
    }
    const double d = val.num_val;
    if (!std::isfinite(d) || d < 0.0 || std::floor(d) != d) {
        throw std::runtime_error(field_name + " must be a non-negative integer, got " + std::to_string(d));
    }
    constexpr double kMaxDoubleForSizeT = static_cast<double>(std::numeric_limits<std::size_t>::max());
    if (d > kMaxDoubleForSizeT) {
        throw std::runtime_error(field_name + " exceeds maximum size_t range");
    }
    return static_cast<std::size_t>(d);
}

int get_strict_int(const JsonVal& val, const std::string& field_name) {
    if (val.type != JsonVal::Number) {
        throw std::runtime_error(field_name + " must be a number");
    }
    const double d = val.num_val;
    if (!std::isfinite(d) || std::floor(d) != d) {
        throw std::runtime_error(field_name + " must be an integer, got " + std::to_string(d));
    }
    constexpr double kMinDoubleForInt = static_cast<double>(std::numeric_limits<int>::min());
    constexpr double kMaxDoubleForInt = static_cast<double>(std::numeric_limits<int>::max());
    if (d < kMinDoubleForInt || d > kMaxDoubleForInt) {
        throw std::runtime_error(field_name + " exceeds integer range");
    }
    return static_cast<int>(d);
}

class JsonParser {
    std::string src_;
    std::size_t pos_{0};

    void skip_whitespace() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }
    char peek() {
        skip_whitespace();
        return (pos_ < src_.size()) ? src_[pos_] : '\0';
    }
    char get() {
        skip_whitespace();
        if (pos_ >= src_.size()) throw std::runtime_error("Unexpected EOF in JSON");
        return src_[pos_++];
    }

    std::string parse_string() {
        if (get() != '"') throw std::runtime_error("Expected '\"' in string");
        std::string s;
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '"') return s;
            if (c == '\\') {
                if (pos_ >= src_.size()) throw std::runtime_error("Truncated escape in JSON string");
                char esc = src_[pos_++];
                if (esc == '"' || esc == '\\' || esc == '/') s += esc;
                else if (esc == 'b') s += '\b';
                else if (esc == 'f') s += '\f';
                else if (esc == 'n') s += '\n';
                else if (esc == 'r') s += '\r';
                else if (esc == 't') s += '\t';
                else throw std::runtime_error(std::string("Invalid escape sequence '\\") + esc + "' in JSON string");
            } else if (static_cast<unsigned char>(c) < 0x20) {
                throw std::runtime_error("Unescaped control character in JSON string");
            } else {
                s += c;
            }
        }
        throw std::runtime_error("Unterminated string in JSON");
    }

    JsonVal parse_number() {
        skip_whitespace();
        std::size_t start = pos_;
        if (src_[pos_] == '-') ++pos_;
        if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            throw std::runtime_error("Malformed number in JSON");
        }
        if (src_[pos_] == '0') {
            ++pos_;
            if (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw std::runtime_error("Leading zero in JSON number");
            }
        } else {
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw std::runtime_error("Expected digit after decimal point in JSON number");
            }
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw std::runtime_error("Expected digit in exponent in JSON number");
            }
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        std::string num_str = src_.substr(start, pos_ - start);
        JsonVal v;
        v.type = JsonVal::Number;
        v.num_val = std::stod(num_str);
        return v;
    }

public:
    explicit JsonParser(std::string src) : src_(std::move(src)) {}

    JsonVal parse_value() {
        skip_whitespace();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonVal v;
            v.type = JsonVal::String;
            v.str_val = parse_string();
            return v;
        }
        if (c == 't') {
            if (src_.compare(pos_, 4, "true") != 0) {
                throw std::runtime_error("Invalid literal in JSON (expected 'true')");
            }
            pos_ += 4;
            if (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
                throw std::runtime_error("Invalid token starting with 'true'");
            }
            JsonVal v;
            v.type = JsonVal::Boolean;
            v.bool_val = true;
            return v;
        }
        if (c == 'f') {
            if (src_.compare(pos_, 5, "false") != 0) {
                throw std::runtime_error("Invalid literal in JSON (expected 'false')");
            }
            pos_ += 5;
            if (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
                throw std::runtime_error("Invalid token starting with 'false'");
            }
            JsonVal v;
            v.type = JsonVal::Boolean;
            v.bool_val = false;
            return v;
        }
        if (c == 'n') {
            if (src_.compare(pos_, 4, "null") != 0) {
                throw std::runtime_error("Invalid literal in JSON (expected 'null')");
            }
            pos_ += 4;
            if (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
                throw std::runtime_error("Invalid token starting with 'null'");
            }
            JsonVal v;
            v.type = JsonVal::Null;
            return v;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
            return parse_number();
        }
        throw std::runtime_error(std::string("Unexpected character in JSON: '") + c + "'");
    }

    JsonVal parse_object() {
        if (get() != '{') throw std::runtime_error("Expected '{'");
        JsonVal v;
        v.type = JsonVal::Object;
        skip_whitespace();
        if (peek() == '}') { get(); return v; }
        while (true) {
            std::string key = parse_string();
            if (v.obj.find(key) != v.obj.end()) {
                throw std::runtime_error("Duplicate key in JSON object: '" + key + "'");
            }
            skip_whitespace();
            if (get() != ':') throw std::runtime_error("Expected ':' after object key");
            JsonVal val = parse_value();
            v.obj[key] = std::move(val);
            skip_whitespace();
            char c = get();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("Expected ',' or '}' in object");
        }
        return v;
    }

    JsonVal parse_array() {
        if (get() != '[') throw std::runtime_error("Expected '['");
        JsonVal v;
        v.type = JsonVal::Array;
        skip_whitespace();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_whitespace();
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("Expected ',' or ']' in array");
        }
        return v;
    }

    JsonVal parse() {
        auto val = parse_value();
        skip_whitespace();
        if (pos_ < src_.size()) throw std::runtime_error("Trailing characters after JSON root");
        return val;
    }
};

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

    // Check for trailing data in binary file
    char trailing_byte;
    in.read(&trailing_byte, 1);
    if (in.gcount() > 0) {
        throw std::runtime_error("Unexpected trailing data after payload in: " + binary_path.string());
    }

    // Physical payload validation (fail-fast before GPU upload)
    for (std::size_t s = 0; s < kSchneiderStoppingNumSections; ++s) {
        const double rho = table.densities_[s];
        if (!std::isfinite(rho) || rho <= 0.0) {
            throw std::runtime_error("Invalid non-positive or non-finite density at section " +
                                     std::to_string(s) + " in: " + binary_path.string());
        }
    }

    for (std::size_t i = 0; i < total_elements; ++i) {
        const double sp = table.mass_stopping_powers_[i];
        if (!std::isfinite(sp) || sp <= 0.0) {
            const std::size_t s = i / kSchneiderStoppingNumEnergies;
            const std::size_t e = i % kSchneiderStoppingNumEnergies;
            throw std::runtime_error("Invalid non-positive or non-finite mass stopping power at section " +
                                     std::to_string(s) + ", energy index " + std::to_string(e) +
                                     " in: " + binary_path.string());
        }
    }

    for (std::size_t s = 0; s < kSchneiderStoppingNumSections; ++s) {
        const std::size_t base = s * kSchneiderStoppingNumEnergies;
        double prev_r = 0.0;
        for (std::size_t e = 0; e < kSchneiderStoppingNumEnergies; ++e) {
            const double r = table.csda_ranges_mm_[base + e];
            if (!std::isfinite(r) || r <= 0.0) {
                throw std::runtime_error("Invalid non-positive or non-finite CSDA range at section " +
                                         std::to_string(s) + ", energy index " + std::to_string(e) +
                                         " in: " + binary_path.string());
            }
            if (e > 0 && r <= prev_r) {
                throw std::runtime_error("Non-monotonically increasing CSDA range at section " +
                                         std::to_string(s) + ", energy index " + std::to_string(e) +
                                         " in: " + binary_path.string());
            }
            prev_r = r;
        }
    }

    // Strict structural metadata validation using recursive descent JSON parser
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

    JsonVal root;
    try {
        JsonParser parser(meta_content);
        root = parser.parse();
    } catch (const std::exception& e) {
        throw std::runtime_error("Malformed JSON in metadata sidecar " + effective_meta_path.string() + ": " + e.what());
    }

    if (root.type != JsonVal::Object) {
        throw std::runtime_error("Metadata root must be a JSON object: " + effective_meta_path.string());
    }

    // 1. Schema version & format
    if (!root.has("schema_version")) {
        throw std::runtime_error("Missing schema_version in metadata");
    }
    const auto schema_version = get_strict_uint(root["schema_version"], "schema_version");
    if (schema_version < 1) {
        throw std::runtime_error("Invalid schema_version in metadata");
    }
    if (!root.has("format") || root["format"].str_val != "binary") {
        throw std::runtime_error("Metadata format must be 'binary', got: " + (root.has("format") ? root["format"].str_val : "none"));
    }
    if (!root.has("data_filename") || root["data_filename"].str_val != "schneider_stopping_v1.bin") {
        throw std::runtime_error("Metadata data_filename must be 'schneider_stopping_v1.bin'");
    }

    // 2. Binary SHA256 match
    const auto actual_bin_sha = compute_file_sha256_hex(binary_path);
    if (!root.has("data_sha256") || root["data_sha256"].str_val != actual_bin_sha) {
        throw std::runtime_error("Schneider stopping binary SHA256 mismatch: recorded=" +
                                 (root.has("data_sha256") ? root["data_sha256"].str_val : "none") +
                                 ", actual=" + actual_bin_sha);
    }

    // 3. Projectile validation: GenericIon(6,12)
    if (!root.has("projectile") || root["projectile"].type != JsonVal::Object) {
        throw std::runtime_error("Missing projectile object in metadata");
    }
    const auto& proj = root["projectile"];
    if (!proj.has("z") || get_strict_int(proj["z"], "projectile.z") != 6 ||
        !proj.has("a") || get_strict_int(proj["a"], "projectile.a") != 12) {
        throw std::runtime_error("Metadata projectile must have z=6, a=12");
    }

    // 4. Dimensions
    if (!root.has("sections_count") || get_strict_uint(root["sections_count"], "sections_count") != kSchneiderStoppingNumSections) {
        throw std::runtime_error("Invalid sections_count in metadata");
    }
    if (!root.has("energies_count") || get_strict_uint(root["energies_count"], "energies_count") != kSchneiderStoppingNumEnergies) {
        throw std::runtime_error("Invalid energies_count in metadata");
    }

    // 5. Section Manifest verification
    if (!root.has("section_manifest") || root["section_manifest"].type != JsonVal::Array) {
        throw std::runtime_error("Missing section_manifest array in metadata");
    }
    const auto& manifest = root["section_manifest"].arr;
    if (manifest.size() != kSchneiderStoppingNumSections) {
        throw std::runtime_error("section_manifest count mismatch: " + std::to_string(manifest.size()) +
                                 " (expected 25)");
    }

    for (std::size_t s = 0; s < kSchneiderStoppingNumSections; ++s) {
        const auto& sec = manifest[s];
        if (sec.type != JsonVal::Object) {
            throw std::runtime_error("section_manifest entry " + std::to_string(s) + " must be an object");
        }
        if (!sec.has("section_id") || get_strict_uint(sec["section_id"], "section_manifest.section_id") != s) {
            throw std::runtime_error("Section ID mismatch, non-integer, or duplicate at index " + std::to_string(s));
        }

        const auto expected_name = kCanonicalSchneiderMaterialNames[s];
        if (!sec.has("material_name") || sec["material_name"].str_val != expected_name) {
            throw std::runtime_error("Section material name mismatch at section " + std::to_string(s) +
                                     ": expected '" + std::string(expected_name) + "', got '" +
                                     (sec.has("material_name") ? sec["material_name"].str_val : "none") + "'");
        }

        if (!sec.has("nominal_density_g_cm3") ||
            std::abs(sec["nominal_density_g_cm3"].num_val - table.densities_[s]) > 1e-4) {
            throw std::runtime_error("Section nominal density mismatch at section " + std::to_string(s));
        }

        const auto expected_hash = kCanonicalSchneiderCompositionHashes[s];
        if (!sec.has("composition_sha256") || sec["composition_sha256"].str_val != expected_hash) {
            throw std::runtime_error("Section composition hash mismatch at section " + std::to_string(s) +
                                     ": expected " + std::string(expected_hash) + ", got " +
                                     (sec.has("composition_sha256") ? sec["composition_sha256"].str_val : "none"));
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
