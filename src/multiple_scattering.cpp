#include "carbon/multiple_scattering.hpp"

#include <array>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace carbon {
Fred2GrMcsTable Fred2GrMcsTable::from_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open FRED 2GR MCS package: " + path.string());
    std::array<char, 4> magic{};
    std::uint32_t version = 0, nx = 0, ny = 0, np = 0;
    in.read(magic.data(), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&nx), 4);
    in.read(reinterpret_cast<char*>(&ny), 4);
    in.read(reinterpret_cast<char*>(&np), 4);
    if (!in || std::memcmp(magic.data(), "M2GR", 4) != 0 || version != 1 ||
        nx != energy_bins || ny != thickness_bins || np != parameter_count)
        throw std::runtime_error("Invalid FRED 2GR MCS package header: " + path.string());
    Fred2GrMcsTable result;
    result.values.resize(energy_bins * thickness_bins * parameter_count);
    in.read(reinterpret_cast<char*>(result.values.data()),
            static_cast<std::streamsize>(result.values.size() * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("Truncated or trailing FRED 2GR MCS package: " + path.string());
    return result;
}
}  // namespace carbon
