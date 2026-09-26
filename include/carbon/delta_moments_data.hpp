#pragma once
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <vector>
namespace carbon {
inline constexpr char delta_moments_filename[]="unified_em_delta_moments_v2.bin";
inline constexpr char delta_moments_sha256[]="c551bc52fa30ff7e3ea229c8b89792e8ecd2b0fb12f18fcad50d1c6f206bc7cc";
inline constexpr char delta_moments_source_sha256[]="8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855";
// EMDMOMT2 v2: 16-byte header followed by interleaved float M1/M2 per EM node.
std::vector<float> load_delta_moments(const std::filesystem::path& file,
                                    std::string_view source_sha256,std::size_t nodes,
                                    std::string_view moments_sha256=delta_moments_sha256,
                                    std::string_view moments_source_sha256=delta_moments_source_sha256);
}
