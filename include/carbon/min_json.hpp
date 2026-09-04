#pragma once

// Minimal strict JSON reader for metadata/bundle validation.
// Supports objects, arrays, strings, numbers, booleans, null.
// Independent of the file-local parser in schneider_stopping_table.cpp.
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon::minjson {

struct Value {
    enum class Type { Null, Boolean, Number, String, Array, Object };
    Type type{Type::Null};
    bool boolean{false};
    double number{0.0};
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    [[nodiscard]] const Value& at(const std::string& key) const {
        if (type != Type::Object) {
            throw std::runtime_error("minjson: not an object");
        }
        const auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("minjson: missing key '" + key + "'");
        }
        return it->second;
    }
    [[nodiscard]] bool contains(const std::string& key) const {
        return type == Type::Object && obj.find(key) != obj.end();
    }
    [[nodiscard]] const Value& at(std::size_t idx) const {
        if (type != Type::Array || idx >= arr.size()) {
            throw std::runtime_error("minjson: array index out of bounds");
        }
        return arr[idx];
    }
};

class Parser {
public:
    explicit Parser(std::string src) : src_(std::move(src)) {}

    Value parse() {
        Value v = parse_value();
        skip_ws();
        if (pos_ != src_.size()) {
            throw std::runtime_error("minjson: trailing bytes after document");
        }
        return v;
    }

private:
    std::string src_;
    std::size_t pos_{0};

    void skip_ws() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }
    char peek() {
        skip_ws();
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }
    char take() {
        skip_ws();
        if (pos_ >= src_.size()) {
            throw std::runtime_error("minjson: unexpected EOF");
        }
        return src_[pos_++];
    }
    std::string parse_string() {
        if (take() != '"') {
            throw std::runtime_error("minjson: expected string");
        }
        std::string s;
        while (true) {
            if (pos_ >= src_.size()) {
                throw std::runtime_error("minjson: unterminated string");
            }
            const char c = src_[pos_++];
            if (c == '"') {
                return s;
            }
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    throw std::runtime_error("minjson: truncated escape");
                }
                const char e = src_[pos_++];
                if (e == '"' || e == '\\' || e == '/') {
                    s += e;
                } else if (e == 'n') {
                    s += '\n';
                } else if (e == 't') {
                    s += '\t';
                } else if (e == 'r') {
                    s += '\r';
                } else if (e == 'b') {
                    s += '\b';
                } else if (e == 'f') {
                    s += '\f';
                } else if (e == 'u') {
                    if (pos_ + 4 > src_.size()) {
                        throw std::runtime_error("minjson: truncated \\u escape");
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = src_[pos_++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code += static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code += static_cast<unsigned>(h - 'a') + 10;
                        } else if (h >= 'A' && h <= 'F') {
                            code += static_cast<unsigned>(h - 'A') + 10;
                        } else {
                            throw std::runtime_error("minjson: bad \\u escape");
                        }
                    }
                    if (code < 0x80) {
                        s += static_cast<char>(code);
                    } else if (code < 0x800) {
                        s += static_cast<char>(0xC0 | (code >> 6));
                        s += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        s += static_cast<char>(0xE0 | (code >> 12));
                        s += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        s += static_cast<char>(0x80 | (code & 0x3F));
                    }
                } else {
                    throw std::runtime_error("minjson: invalid escape");
                }
            } else {
                s += c;
            }
        }
    }
    Value parse_number() {
        skip_ws();
        const std::size_t start = pos_;
        if (pos_ < src_.size() && src_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= src_.size() ||
            !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            throw std::runtime_error("minjson: malformed number");
        }
        while (pos_ < src_.size() &&
               std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            if (pos_ >= src_.size() ||
                !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw std::runtime_error("minjson: malformed fraction");
            }
            while (pos_ < src_.size() &&
                   std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= src_.size() ||
                !std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                throw std::runtime_error("minjson: malformed exponent");
            }
            while (pos_ < src_.size() &&
                   std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        Value v;
        v.type = Value::Type::Number;
        v.number = std::stod(src_.substr(start, pos_ - start));
        return v;
    }
    Value parse_value() {
        const char c = peek();
        if (c == '{') {
            take();
            Value v;
            v.type = Value::Type::Object;
            if (peek() == '}') {
                take();
                return v;
            }
            while (true) {
                std::string key = parse_string();
                if (take() != ':') {
                    throw std::runtime_error("minjson: expected ':'");
                }
                v.obj.emplace(std::move(key), parse_value());
                const char sep = take();
                if (sep == '}') {
                    return v;
                }
                if (sep != ',') {
                    throw std::runtime_error("minjson: expected ',' or '}'");
                }
            }
        }
        if (c == '[') {
            take();
            Value v;
            v.type = Value::Type::Array;
            if (peek() == ']') {
                take();
                return v;
            }
            while (true) {
                v.arr.push_back(parse_value());
                const char sep = take();
                if (sep == ']') {
                    return v;
                }
                if (sep != ',') {
                    throw std::runtime_error("minjson: expected ',' or ']'");
                }
            }
        }
        if (c == '"') {
            Value v;
            v.type = Value::Type::String;
            v.str = parse_string();
            return v;
        }
        if (c == 't') {
            if (src_.compare(pos_, 4, "true") != 0) {
                throw std::runtime_error("minjson: bad literal");
            }
            pos_ += 4;
            Value v;
            v.type = Value::Type::Boolean;
            v.boolean = true;
            return v;
        }
        if (c == 'f') {
            if (src_.compare(pos_, 5, "false") != 0) {
                throw std::runtime_error("minjson: bad literal");
            }
            pos_ += 5;
            Value v;
            v.type = Value::Type::Boolean;
            return v;
        }
        if (c == 'n') {
            if (src_.compare(pos_, 4, "null") != 0) {
                throw std::runtime_error("minjson: bad literal");
            }
            pos_ += 4;
            return Value{};
        }
        return parse_number();
    }
};

inline std::string require_string(const Value& v, const std::string& field) {
    if (v.type != Value::Type::String) {
        throw std::runtime_error("minjson: '" + field + "' must be a string");
    }
    return v.str;
}

inline double require_number(const Value& v, const std::string& field) {
    if (v.type != Value::Type::Number || !std::isfinite(v.number)) {
        throw std::runtime_error("minjson: '" + field + "' must be a finite number");
    }
    return v.number;
}

inline std::uint64_t require_uint(const Value& v, const std::string& field) {
    const double d = require_number(v, field);
    if (d < 0.0 || std::floor(d) != d ||
        d > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error("minjson: '" + field + "' must be a non-negative integer");
    }
    return static_cast<std::uint64_t>(d);
}

}  // namespace carbon::minjson
