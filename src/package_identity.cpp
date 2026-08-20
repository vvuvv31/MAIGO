#include "carbon/package_identity.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

namespace carbon {
namespace {

// Small, self-contained JSON reader for package sidecars. It accepts the JSON
// grammar used by our manifests and rejects duplicate keys and trailing bytes.
struct Json {
    using object = std::map<std::string, Json, std::less<>>;
    using array = std::vector<Json>;
    std::variant<std::nullptr_t, bool, double, std::string, object, array> value;
};

class JsonReader {
public:
    explicit JsonReader(std::string_view text) : text_(text) {}
    Json parse() {
        auto value = parse_value();
        space();
        if (pos_ != text_.size()) fail("trailing input");
        return value;
    }
private:
    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(std::string("Invalid package identity JSON: ") + message);
    }
    void space() { while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
    char take() { if (pos_ == text_.size()) fail("unexpected end of input"); return text_[pos_++]; }
    bool consume(char expected) { space(); if (pos_ < text_.size() && text_[pos_] == expected) { ++pos_; return true; } return false; }
    Json parse_value() {
        space(); if (pos_ == text_.size()) fail("missing value");
        switch (text_[pos_]) {
        case '{': return Json{parse_object()}; case '[': return Json{parse_array()};
        case '"': return Json{parse_string()}; case 't': literal("true"); return Json{true};
        case 'f': literal("false"); return Json{false}; case 'n': literal("null"); return Json{nullptr};
        default: if (text_[pos_] == '-' || std::isdigit(static_cast<unsigned char>(text_[pos_]))) return Json{parse_number()}; fail("invalid value");
        }
    }
    void literal(std::string_view word) { if (text_.substr(pos_, word.size()) != word) fail("invalid literal"); pos_ += word.size(); }
    Json::object parse_object() {
        take(); Json::object result; space(); if (consume('}')) return result;
        for (;;) {
            space();
            if (take() != '"') fail("object key must be a string");
            --pos_;
            auto key = parse_string();
            if (!consume(':')) fail("missing object colon");
            auto [it, inserted] = result.emplace(std::move(key), parse_value());
            if (!inserted) fail("duplicate object key");
            if (consume('}')) return result;
            if (!consume(',')) fail("missing object comma");
        }
    }
    Json::array parse_array() {
        take(); Json::array result; if (consume(']')) return result;
        for (;;) { result.push_back(parse_value()); if (consume(']')) return result; if (!consume(',')) fail("missing array comma"); }
    }
    std::uint32_t parse_hex_quad() {
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const char digit = take();
            value <<= 4U;
            if (digit >= '0' && digit <= '9') value |= digit - '0';
            else if (digit >= 'a' && digit <= 'f') value |= digit - 'a' + 10U;
            else if (digit >= 'A' && digit <= 'F') value |= digit - 'A' + 10U;
            else fail("invalid Unicode escape");
        }
        return value;
    }
    void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) output += static_cast<char>(codepoint);
        else if (codepoint <= 0x7ffU) {
            output += static_cast<char>(0xc0U | (codepoint >> 6U));
            output += static_cast<char>(0x80U | (codepoint & 0x3fU));
        } else if (codepoint <= 0xffffU) {
            output += static_cast<char>(0xe0U | (codepoint >> 12U));
            output += static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU));
            output += static_cast<char>(0x80U | (codepoint & 0x3fU));
        } else {
            output += static_cast<char>(0xf0U | (codepoint >> 18U));
            output += static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU));
            output += static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU));
            output += static_cast<char>(0x80U | (codepoint & 0x3fU));
        }
    }
    std::string parse_string() {
        if (take() != '"') fail("invalid string");
        std::string result;
        while (pos_ < text_.size()) { const char c = take(); if (c == '"') return result; if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') { result += c; continue; } const char escaped = take();
            switch (escaped) {
            case '"': result += '"'; break; case '\\': result += '\\'; break;
            case '/': result += '/'; break; case 'b': result += '\b'; break;
            case 'f': result += '\f'; break; case 'n': result += '\n'; break;
            case 'r': result += '\r'; break; case 't': result += '\t'; break;
            case 'u': {
                auto codepoint = parse_hex_quad();
                if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                    if (take() != '\\' || take() != 'u') fail("missing low surrogate");
                    const auto low = parse_hex_quad();
                    if (low < 0xdc00U || low > 0xdfffU) fail("invalid low surrogate");
                    codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                                (low - 0xdc00U);
                } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                    fail("unpaired low surrogate");
                }
                append_utf8(result, codepoint);
                break;
            }
            default: fail("unsupported string escape");
            }
        } fail("unterminated string");
    }
    double parse_number() {
        const auto start = pos_; if (text_[pos_] == '-') ++pos_; if (pos_ == text_.size()) fail("bad number");
        if (text_[pos_] == '0') ++pos_; else { if (!std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("bad number"); while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
        if (pos_ < text_.size() && text_[pos_] == '.') { ++pos_; if (pos_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("bad fraction"); while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) { ++pos_; if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_; if (pos_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("bad exponent"); while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
        try { return std::stod(std::string(text_.substr(start, pos_ - start))); } catch (...) { fail("bad number"); }
    }
    std::string_view text_; std::size_t pos_{0};
};

const Json::object& object(const Json& value, const char* name) { if (const auto* result = std::get_if<Json::object>(&value.value)) return *result; throw std::runtime_error(std::string("Package identity field is not an object: ") + name); }
const Json& required(const Json::object& value, const char* key) { const auto it = value.find(key); if (it == value.end()) throw std::runtime_error(std::string("Package identity is missing field: ") + key); return it->second; }
std::string string(const Json& value, const char* name) { if (const auto* result = std::get_if<std::string>(&value.value)) return *result; throw std::runtime_error(std::string("Package identity field is not a string: ") + name); }
double number(const Json& value, const char* name) { if (const auto* result = std::get_if<double>(&value.value)) return *result; throw std::runtime_error(std::string("Package identity field is not a number: ") + name); }
std::string read_text(const std::filesystem::path& path) { std::ifstream in(path); if (!in) throw std::runtime_error("Cannot open package identity sidecar: " + path.string()); std::ostringstream out; out << in.rdbuf(); return out.str(); }

// FIPS 180-4 SHA-256, kept local to make package validation build offline.
std::string sha256(const std::filesystem::path& path) {
    constexpr std::array<std::uint32_t, 64> k{0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
    auto rot = [](std::uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); };
    std::array<std::uint32_t,8> h{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    auto block = [&](const std::uint8_t* data) { std::array<std::uint32_t,64> w{}; for (int i=0;i<16;++i) w[i]=(std::uint32_t(data[4*i])<<24)|(std::uint32_t(data[4*i+1])<<16)|(std::uint32_t(data[4*i+2])<<8)|data[4*i+3]; for(int i=16;i<64;++i) w[i]=(rot(w[i-15],7)^rot(w[i-15],18)^(w[i-15]>>3))+w[i-16]+(rot(w[i-2],17)^rot(w[i-2],19)^(w[i-2]>>10))+w[i-7]; auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7]; for(int i=0;i<64;++i){ auto s1=rot(e,6)^rot(e,11)^rot(e,25); auto choice=(e&f)^((~e)&g); auto t1=hh+s1+choice+k[i]+w[i]; auto s0=rot(a,2)^rot(a,13)^rot(a,22); auto majority=(a&b)^(a&c)^(b&c); auto t2=s0+majority; hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;} h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh; };
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("Cannot hash package: " + path.string()); std::array<std::uint8_t,64> buffer{}; std::uint64_t bytes=0; while (in.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) { block(buffer.data()); bytes += 64; } const auto remainder=static_cast<std::size_t>(in.gcount()); bytes += remainder; buffer[remainder]=0x80; if (remainder >= 56) { std::fill(buffer.begin()+remainder+1,buffer.end(),0); block(buffer.data()); buffer.fill(0); } else std::fill(buffer.begin()+remainder+1,buffer.end(),0); const auto bits=bytes*8; for(int i=0;i<8;++i) buffer[56+i]=static_cast<std::uint8_t>(bits >> (56-8*i)); block(buffer.data()); std::ostringstream out; for(auto word:h) out<<std::hex<<std::setfill('0')<<std::setw(8)<<word; return out.str();
}

Json sidecar_or_override(const std::filesystem::path& package_path,
                         const std::filesystem::path& manifest_path) {
    const auto sibling = package_path.parent_path() /
                         (package_path.stem().string() + ".compiled.json");
    const auto source = std::filesystem::exists(sibling) ? std::filesystem::path(sibling) : manifest_path;
    const auto text = read_text(source);
    // Avoid parsing old arbitrary-provenance sidecars: v1 has no identity
    // contract and can contain fields not relevant to runtime validation.
    if (source != manifest_path && text.find("\"kind\"") == std::string::npos) {
        auto manifest = JsonReader(read_text(manifest_path)).parse();
        const auto& entries = object(required(object(manifest, "override manifest"), "packages"), "packages");
        const auto key = package_path.filename().string();
        const auto it = entries.find(key);
        if (it == entries.end()) throw std::runtime_error("No override identity for package: " + package_path.string());
        return it->second;
    }
    auto root = JsonReader(text).parse();
    const auto& root_object = object(root, "sidecar");
    // v1 sidecars intentionally lack identity fields; they need an explicit
    // repository manifest rather than being treated as self-identifying.
    if (source == manifest_path || !root_object.contains("kind")) {
        if (source != manifest_path) root = JsonReader(read_text(manifest_path)).parse();
        const auto& entries = object(required(object(root, "override manifest"), "packages"), "packages");
        const auto key = package_path.filename().string();
        const auto it = entries.find(key);
        if (it == entries.end()) throw std::runtime_error("No sidecar or override identity for package: " + package_path.string());
        return it->second;
    }
    return root;
}

void mismatch(const std::string& message, PackageIdentityValidation mode) {
    if (mode == PackageIdentityValidation::warn) { std::cerr << "warning: package identity: " << message << '\n'; return; }
    throw std::runtime_error("Package identity validation failed: " + message);
}

}  // namespace

PackageIdentityValidation parse_package_identity_validation(const std::string& value) {
    if (value == "strict") return PackageIdentityValidation::strict;
    if (value == "warn") return PackageIdentityValidation::warn;
    if (value == "off") return PackageIdentityValidation::off;
    throw std::invalid_argument("package_identity_validation must be strict, warn, or off");
}

void validate_package_identity(const std::filesystem::path& package_path,
                               const PackageIdentityExpectation& expected,
                               PackageIdentityValidation mode,
                               const std::filesystem::path& override_manifest_path) {
    if (mode == PackageIdentityValidation::off) return;
    try {
        const auto root = sidecar_or_override(package_path, override_manifest_path);
        const auto& values = object(root, "sidecar");
        const auto& output = object(required(values, "output"), "output");
        const auto bytes = static_cast<std::uintmax_t>(number(required(output, "bytes"), "output.bytes"));
        if (bytes != std::filesystem::file_size(package_path)) mismatch("byte size differs for " + package_path.string(), mode);
        if (sha256(package_path) != string(required(output, "sha256"), "output.sha256")) mismatch("SHA-256 differs for " + package_path.string(), mode);
        if (string(required(values, "kind"), "kind") != expected.kind) mismatch("kind differs for " + package_path.string(), mode);
        const auto& projectile = object(required(values, "projectile"), "projectile");
        if (static_cast<int>(number(required(projectile, "Z"), "projectile.Z")) != expected.projectile_atomic_number || static_cast<int>(number(required(projectile, "A"), "projectile.A")) != expected.projectile_mass_number) mismatch("projectile differs for " + package_path.string(), mode);
        if (string(required(values, "material"), "material") != expected.material) mismatch("material differs for " + package_path.string(), mode);
        if (string(required(values, "physics_model"), "physics_model") != expected.physics_model) mismatch("physics model differs for " + package_path.string(), mode);
        const auto& energy = object(required(values, "energy_range_MeV_per_u"), "energy_range_MeV_per_u");
        const auto minimum = number(required(energy, "minimum"), "energy.minimum"); const auto maximum = number(required(energy, "maximum"), "energy.maximum");
        if (minimum > expected.minimum_energy_MeV_per_u || maximum < expected.maximum_energy_MeV_per_u) mismatch("energy range does not cover configured primary energy for " + package_path.string(), mode);
    } catch (const std::exception& error) { if (mode == PackageIdentityValidation::warn) { std::cerr << "warning: package identity: " << error.what() << '\n'; } else throw; }
}

}  // namespace carbon
