#include "carbon/schneider_ion_stopping_table.hpp"
#include "carbon/sha256.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace carbon {

namespace {

struct JsonVal2 {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type{Type::Null};
    bool bool_val{false};
    double num_val{0.0};
    std::string str_val{};
    std::vector<std::pair<std::string, JsonVal2>> obj{};
    std::vector<JsonVal2> arr{};
    bool has(const std::string& k) const {
        if (type != Type::Object) return false;
        for (auto& p : obj) if (p.first == k) return true;
        return false;
    }
    const JsonVal2& operator[](const std::string& k) const {
        for (auto& p : obj) if (p.first == k) return p.second;
        throw std::runtime_error("Missing JSON key: " + k);
    }
};

class JsonParser2 {
public:
    explicit JsonParser2(const std::string& s) : s_(s) {}
    JsonVal2 parse() {
        skip();
        JsonVal2 v = value();
        skip();
        if (pos_ != s_.size()) throw std::runtime_error("Trailing JSON content");
        return v;
    }
private:
    const std::string& s_;
    std::size_t pos_{0};
    void skip() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
               s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }
    char peek() {
        if (pos_ >= s_.size()) throw std::runtime_error("Truncated JSON");
        return s_[pos_];
    }
    JsonVal2 value() {
        skip();
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { JsonVal2 v; v.type = JsonVal2::Type::String; v.str_val = str(); return v; }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') { expect("null"); return JsonVal2{}; }
        return number();
    }
    JsonVal2 object() {
        JsonVal2 v; v.type = JsonVal2::Type::Object; ++pos_;
        skip();
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            skip();
            if (peek() != '"') throw std::runtime_error("Expected JSON key");
            std::string k = str();
            skip();
            if (peek() != ':') throw std::runtime_error("Expected ':'");
            ++pos_;
            v.obj.emplace_back(k, value());
            skip();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return v; }
            throw std::runtime_error("Expected ',' or '}'");
        }
    }
    JsonVal2 array() {
        JsonVal2 v; v.type = JsonVal2::Type::Array; ++pos_;
        skip();
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.arr.push_back(value());
            skip();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return v; }
            throw std::runtime_error("Expected ',' or ']'");
        }
    }
    std::string str() {
        ++pos_;
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) throw std::runtime_error("Unterminated string");
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("Bad escape");
                char e = s_[pos_++];
                out += (e == 'n') ? '\n' : (e == 't' ? '\t' : e);
            } else out += c;
        }
    }
    JsonVal2 boolean() {
        JsonVal2 v; v.type = JsonVal2::Type::Bool;
        if (s_.compare(pos_, 4, "true") == 0) { v.bool_val = true; pos_ += 4; return v; }
        if (s_.compare(pos_, 5, "false") == 0) { v.bool_val = false; pos_ += 5; return v; }
        throw std::runtime_error("Bad boolean");
    }
    void expect(const char* lit) {
        std::string l(lit);
        if (s_.compare(pos_, l.size(), l) != 0) throw std::runtime_error("Bad literal");
        pos_ += l.size();
    }
    JsonVal2 number() {
        JsonVal2 v; v.type = JsonVal2::Type::Number;
        std::size_t n = 0;
        try { v.num_val = std::stod(s_.substr(pos_), &n); }
        catch (...) { throw std::runtime_error("Bad number"); }
        if (!n) throw std::runtime_error("Bad number");
        pos_ += n;
        return v;
    }
};

std::size_t strict_uint(const JsonVal2& v, const std::string& name) {
    if (v.type != JsonVal2::Type::Number || !std::isfinite(v.num_val) || v.num_val < 0 ||
        std::floor(v.num_val) != v.num_val)
        throw std::runtime_error("Expected uint field: " + name);
    return static_cast<std::size_t>(v.num_val);
}

}  // namespace

SchneiderIonStoppingTable SchneiderIonStoppingTable::from_binary(
    const std::filesystem::path& binary_path,
    const std::filesystem::path& metadata_path) {
    std::ifstream in(binary_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open ion stopping binary: " + binary_path.string());
    SchneiderIonStoppingHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || in.gcount() != sizeof(header))
        throw std::runtime_error("Truncated header in: " + binary_path.string());
    if (std::strncmp(header.magic, "SCHNIOSP", 8) != 0)
        throw std::runtime_error("Invalid magic in: " + binary_path.string());
    if (header.version != 1)
        throw std::runtime_error("Unsupported version in: " + binary_path.string());
    if (header.num_sections != kSchneiderIonSections ||
        header.num_species != kSchneiderIonSpecies ||
        header.num_energies != kSchneiderIonEnergies)
        throw std::runtime_error("Dimension mismatch in: " + binary_path.string());
    if (!std::isfinite(header.energy_min_mevu) || !std::isfinite(header.energy_max_mevu) ||
        !std::isfinite(header.energy_step_mevu) || std::abs(header.energy_min_mevu - kSchneiderIonEnergyMin) > 1e-6 ||
        std::abs(header.energy_max_mevu - kSchneiderIonEnergyMax) > 1e-6 ||
        std::abs(header.energy_step_mevu - kSchneiderIonEnergyStep) > 1e-6)
        throw std::runtime_error("Grid mismatch in: " + binary_path.string());

    SchneiderIonStoppingTable table;
    table.densities_.resize(kSchneiderIonSections);
    in.read(reinterpret_cast<char*>(table.densities_.data()),
            kSchneiderIonSections * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(kSchneiderIonSections * sizeof(double)))
        throw std::runtime_error("Truncated densities in: " + binary_path.string());

    const std::size_t total = kSchneiderIonSections * kSchneiderIonSpecies * kSchneiderIonEnergies;
    table.values_.resize(total);
    in.read(reinterpret_cast<char*>(table.values_.data()), total * sizeof(double));
    if (!in || in.gcount() != static_cast<std::streamsize>(total * sizeof(double)))
        throw std::runtime_error("Truncated values in: " + binary_path.string());
    char trailing = 0;
    in.read(&trailing, 1);
    if (in.gcount() > 0)
        throw std::runtime_error("Trailing data in: " + binary_path.string());

    for (double rho : table.densities_)
        if (!std::isfinite(rho) || rho <= 0) throw std::runtime_error("Invalid reference density");
    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(table.values_[i]) || table.values_[i] <= 0.0)
            throw std::runtime_error("Non-positive value in ion stopping table");
    }

    const auto effective_meta = metadata_path.empty()
        ? (binary_path.parent_path() / (binary_path.stem().string() + ".metadata.json"))
        : metadata_path;
    if (!std::filesystem::exists(effective_meta))
        throw std::runtime_error("Missing metadata sidecar: " + effective_meta.string());
    std::ifstream meta_in(effective_meta);
    if (!meta_in) throw std::runtime_error("Cannot open metadata: " + effective_meta.string());
    const std::string content((std::istreambuf_iterator<char>(meta_in)),
                              std::istreambuf_iterator<char>());
    JsonParser2 parser(content);
    JsonVal2 root = parser.parse();
    if (root.type != JsonVal2::Type::Object)
        throw std::runtime_error("Metadata root must be object");
    if (strict_uint(root["schema_version"], "schema_version") != 1)
        throw std::runtime_error("Bad schema_version");
    if (!root.has("format") || root["format"].str_val != "binary")
        throw std::runtime_error("Metadata format must be 'binary'");
    if (!root.has("data_filename") ||
        root["data_filename"].str_val != binary_path.filename().string())
        throw std::runtime_error("Metadata data_filename mismatch");
    if (!root.has("data_sha256") ||
        root["data_sha256"].str_val != compute_file_sha256_hex(binary_path))
        throw std::runtime_error("Metadata data_sha256 mismatch");
    if (strict_uint(root["sections_count"], "sections_count") != kSchneiderIonSections ||
        strict_uint(root["species_count"], "species_count") != kSchneiderIonSpecies ||
        strict_uint(root["energies_count"], "energies_count") != kSchneiderIonEnergies)
        throw std::runtime_error("Metadata dimension mismatch");
    if (!root.has("provenance") || root["provenance"].str_val.empty())
        throw std::runtime_error("Metadata provenance required");
    if (root["quantity"].str_val != "unrestricted electronic dE/dx, linear at unit density")
        throw std::runtime_error("Wrong stopping quantity");
    const auto& grid = root["energy_grid_MeV_per_u"];
    for (const auto& entry : std::array<std::pair<const char*,double>,3>{{
            {"minimum",kSchneiderIonEnergyMin},{"maximum",kSchneiderIonEnergyMax},{"step",kSchneiderIonEnergyStep}}}) {
        const auto& value = grid[entry.first];
        if (value.type != JsonVal2::Type::Number || !std::isfinite(value.num_val) ||
            std::abs(value.num_val-entry.second)>1e-8) throw std::runtime_error("Metadata energy grid mismatch");
    }
    constexpr std::array<std::pair<int,int>,18> species{{{1,1},{1,2},{1,3},{2,3},{2,4},{2,6},{3,6},{3,7},
        {4,7},{4,9},{4,10},{5,8},{5,10},{5,11},{6,10},{6,11},{6,12},{4,6}}};
    const auto& registry=root["species_za"];
    if (registry.type != JsonVal2::Type::Array || registry.arr.size()!=18) throw std::runtime_error("Missing species registry");
    for (std::size_t i=0;i<18;++i) {
        const auto& row=registry.arr[i];
        if(row.type!=JsonVal2::Type::Array || row.arr.size()!=2 ||
            strict_uint(row.arr[0],"Z")!=static_cast<std::size_t>(species[i].first) ||
            strict_uint(row.arr[1],"A")!=static_cast<std::size_t>(species[i].second)) throw std::runtime_error("Species registry mismatch");
    }
    return table;
}

double SchneiderIonStoppingTable::linear_stopping_at_unit_density(
    std::size_t species_idx, std::size_t section_id, double energy_mevu) const noexcept {
    return schneider_ion_stopping_lookup(values_.empty() ? nullptr : values_.data(),
                                         species_idx, section_id, energy_mevu);
}

std::vector<float> SchneiderIonStoppingTable::to_flat_float() const {
    std::vector<float> out(values_.size());
    for (std::size_t i = 0; i < values_.size(); ++i)
        out[i] = static_cast<float>(values_[i]);
    return out;
}

}  // namespace carbon
