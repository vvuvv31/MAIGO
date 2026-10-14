#include "carbon/delta_moments_data.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
namespace carbon {
void validate_delta_moments(const std::filesystem::path& file,
                                    std::size_t nodes){
    if(!std::filesystem::exists(file))
        throw std::runtime_error("Missing delta moments table: "+file.string());
    if(std::endian::native!=std::endian::little)
        throw std::runtime_error("Delta moments require little-endian float data");
    if(nodes==0 || nodes>std::numeric_limits<std::uint32_t>::max() ||
       std::filesystem::file_size(file)>4ULL*1024*1024*1024)
        throw std::runtime_error("Delta moments size/node-count mismatch");
    std::ifstream in(file,std::ios::binary);char magic[8]{};std::uint32_t header[2]{};
    in.read(magic,8);in.read(reinterpret_cast<char*>(header),8);
    if(std::filesystem::file_size(file)!=16+std::uint64_t{8}*nodes ||
       !in || std::string_view(magic,8)!="EMDMOMT2" || header[0]!=2 || header[1]!=nodes)
        throw std::runtime_error("Delta moments header mismatch");
    // This sidecar remains a validated dependency. Its values are not used
    // by continuous26 transport, so validate in bounded chunks without caching.
    std::array<float,4096> values;
    std::size_t remaining=2*nodes;
    while(remaining) {
        const auto count=std::min(remaining,values.size());
        in.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
        if(!in)throw std::runtime_error("Delta moments truncated");
        for(std::size_t i=0;i<count;++i)
            if(!(values[i]>=0 && std::isfinite(values[i])))
                throw std::runtime_error("Delta moments contain invalid values");
        remaining-=count;
    }
}
} // namespace carbon
