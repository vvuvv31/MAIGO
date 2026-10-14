#pragma once
#include <cstddef>
#include <filesystem>
namespace carbon {
inline constexpr char delta_moments_filename[]="delta_moments.bin";
// EMDMOMT2 v2: 16-byte header followed by interleaved float M1/M2 per EM node.
void validate_delta_moments(const std::filesystem::path& file,
                                    std::size_t nodes);
}
