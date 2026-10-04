#pragma once
#include "carbon/sha256.hpp"
#include <cstring>
#include <cmath>

namespace carbon {
// Research BODY mask input. Pin both this derived mask and its exact packed CT;
// matching dimensions alone must never allow masks from a different patient.
inline std::vector<std::uint8_t> load_broadbeam_body_mask(
    const std::filesystem::path& path,const std::string& pin,const std::filesystem::path& ct_path,
    std::array<unsigned,3> dims,std::array<float,3> origin,std::array<float,3> spacing) {
    if(compute_file_sha256_hex(path)!=pin)throw std::runtime_error("Broadbeam BODY mask SHA256 mismatch");
    std::ifstream in(path,std::ios::binary);std::array<char,8> magic{};std::array<unsigned,3> shape{};
    std::array<float,6> geometry{};std::array<char,64> ct_sha{};
    in.read(magic.data(),8);in.read(reinterpret_cast<char*>(shape.data()),12);
    in.read(reinterpret_cast<char*>(geometry.data()),24);in.read(ct_sha.data(),64);
    if(!in || std::string(magic.data(),8)!="MAIBODY1" || shape!=dims)
        throw std::runtime_error("Broadbeam BODY mask header/shape mismatch");
    for(unsigned a=0;a<3;++a)if(geometry[a]!=origin[a] || geometry[a+3]!=spacing[a])
        throw std::runtime_error("Broadbeam BODY mask geometry mismatch");
    if(std::string(ct_sha.data(),64)!=compute_file_sha256_hex(ct_path))
        throw std::runtime_error("Broadbeam BODY mask CT identity mismatch");
    const auto n=std::size_t(dims[0])*dims[1]*dims[2];std::vector<std::uint8_t> mask(n);
    in.read(reinterpret_cast<char*>(mask.data()),n);
    if(!in || in.peek()!=std::char_traits<char>::eof() ||
       std::any_of(mask.begin(),mask.end(),[](auto x){return x>1;}) ||
       std::none_of(mask.begin(),mask.end(),[](auto x){return x!=0;}))
        throw std::runtime_error("Broadbeam BODY mask payload invalid");
    return mask;
}
}
