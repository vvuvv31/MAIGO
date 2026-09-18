#include "carbon/delta_moments_data.hpp"
#include "carbon/sha256.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
namespace carbon {
std::vector<float> load_delta_moments(const std::filesystem::path& file,
                                    std::string_view source_sha256,std::size_t nodes){
    if(source_sha256!=delta_moments_source_sha256)
        throw std::runtime_error("Delta moments source-package SHA mismatch");
    if(!std::filesystem::exists(file))
        throw std::runtime_error("Missing delta moments table: "+file.string()+
            "; run python3 tools/build_delta_moments.py (no water/older-data fallback)");
    if(std::endian::native!=std::endian::little)
        throw std::runtime_error("Delta moments require little-endian float data");
    if(nodes==0 || nodes>std::numeric_limits<std::uint32_t>::max() ||
       std::filesystem::file_size(file)!=16+std::uint64_t{8}*nodes)
        throw std::runtime_error("Delta moments size/node-count mismatch");
    if(!file_sha256_matches(file,delta_moments_sha256))
        throw std::runtime_error("Delta moments SHA256 mismatch");
    std::ifstream in(file,std::ios::binary);char magic[8]{};std::uint32_t header[2]{};
    in.read(magic,8);in.read(reinterpret_cast<char*>(header),8);
    if(!in || std::string_view(magic,8)!="EMDMOMT2" || header[0]!=2 || header[1]!=nodes)
        throw std::runtime_error("Delta moments header mismatch");
    std::vector<float> data(2*nodes);
    in.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
    if(!in)throw std::runtime_error("Delta moments truncated");
    for(float value:data)if(!(value>=0 && std::isfinite(value)))
        throw std::runtime_error("Delta moments contain invalid values");
    return data;
}
} // namespace carbon
