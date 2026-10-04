#pragma once
#include <cstddef>
#include <filesystem>
#include <vector>
namespace carbon {
inline constexpr char delta_moments_filename[]="unified_em_delta_moments_v2.bin";
// EMDMOMT2 v2: 16-byte header followed by interleaved float M1/M2 per EM node.
std::vector<float> load_delta_moments(const std::filesystem::path& file,
                                    std::size_t nodes);
}
