#include "carbon/delta_moments_data.hpp"
#include "carbon/em_density_factorization.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
namespace carbon {
std::vector<float> load_delta_moments(const std::filesystem::path& file,
                                    std::size_t nodes){
    if(!std::filesystem::exists(file))
        throw std::runtime_error("Missing delta moments table: "+file.string()+
            "; run python3 tools/build_delta_moments.py (no water/older-data fallback)");
    if(std::endian::native!=std::endian::little)
        throw std::runtime_error("Delta moments require little-endian float data");
    if(nodes==0 || nodes>std::numeric_limits<std::uint32_t>::max() ||
       std::filesystem::file_size(file)>4ULL*1024*1024*1024)
        throw std::runtime_error("Delta moments size/node-count mismatch");
    std::ifstream in(file,std::ios::binary);char magic[8]{};std::uint32_t header[2]{};
    in.read(magic,8);in.read(reinterpret_cast<char*>(header),8);
    if(in && std::string_view(magic,8)=="EMDFAC01") {
        namespace f=em_density_factorization;
        const unsigned nr=header[1];unsigned nn{};f::read(in,&nn);
        if(header[0]!=1 || nn!=nodes || nr<26*18 || nr>256*18 || nr%18)
            throw std::runtime_error("Factorized delta registry mismatch");
        struct Block {unsigned offset{},count{},id{},base{};float ratio{};std::array<f::Field,2> fields;};
        std::vector<Block> blocks(nr);std::vector<float> data(2*nodes);std::uint64_t offset=0;
        for(unsigned i=0;i<nr;++i) {
            auto& b=blocks[i];f::read(in,&b.offset);f::read(in,&b.count);f::read(in,&b.id);f::read(in,&b.base);f::read(in,&b.ratio);
            if(b.id!=i || b.offset!=offset || b.count<2 || offset+b.count>nodes ||
               b.base>=nr || b.base%18!=i%18 || !(b.ratio>0 && std::isfinite(b.ratio)))
                throw std::runtime_error("Factorized delta directory mismatch");
            offset+=b.count;
            if(b.base==i)f::read(in,data.data()+2*b.offset,2*b.count);
            else for(auto& field:b.fields)field=f::Field::load(in,b.count);
        }
        if(offset!=nodes)throw std::runtime_error("Factorized delta node coverage mismatch");
        f::require_end(in);
        for(unsigned i=0;i<nr;++i) {
            const auto& b=blocks[i];if(b.base==i)continue;const auto& base=blocks[b.base];
            if(base.base!=b.base || base.count!=b.count)throw std::runtime_error("Invalid factorized delta base");
            for(unsigned column=0;column<2;++column)b.fields[column].expand(data.data()+2*b.offset+column,2,
                data.data()+2*base.offset+column,2,b.count,b.ratio,true);
        }
        for(float v:data)if(!(v>=0 && std::isfinite(v)))throw std::runtime_error("Invalid expanded delta moment");
        return data;
    }
    if(std::filesystem::file_size(file)!=16+std::uint64_t{8}*nodes ||
       !in || std::string_view(magic,8)!="EMDMOMT2" || header[0]!=2 || header[1]!=nodes)
        throw std::runtime_error("Delta moments header mismatch");
    std::vector<float> data(2*nodes);
    in.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
    if(!in)throw std::runtime_error("Delta moments truncated");
    for(float value:data)if(!(value>=0 && std::isfinite(value)))
        throw std::runtime_error("Delta moments contain invalid values");
    return data;
}
} // namespace carbon
