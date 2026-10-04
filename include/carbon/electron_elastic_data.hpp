#pragma once
#include "carbon/electron_elastic.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <array>
#include <cstring>
namespace carbon {
class ElectronElasticData {
public:
    std::array<double,31*38> rows{};double mass{},bc{},xc2{};
    ElectronElasticData(const std::filesystem::path& path) {
        if(std::filesystem::file_size(path)!=9464)
            throw std::runtime_error("Elastic kernel size mismatch");
        std::ifstream in(path,std::ios::binary);char magic[8];std::uint32_t version,count;
        in.read(magic,8);in.read(reinterpret_cast<char*>(&version),4);in.read(reinterpret_cast<char*>(&count),4);
        in.read(reinterpret_cast<char*>(&mass),8);in.read(reinterpret_cast<char*>(&bc),8);in.read(reinterpret_cast<char*>(&xc2),8);
        in.read(reinterpret_cast<char*>(rows.data()),8*rows.size());
        if(!in||std::memcmp(magic,"MAIELS01",8)||version!=1||count!=31||
           !std::isfinite(mass)||mass<=0||!std::isfinite(bc)||bc<=0||!std::isfinite(xc2)||xc2<=0)
            throw std::runtime_error("Elastic kernel header/state invalid");
        for(double value:rows)if(!std::isfinite(value))throw std::runtime_error("Nonfinite elastic correction");
        for(unsigned i=0;i<31;++i)for(unsigned j=0;j<3;++j)if(rows[38*i+j]<=0)throw std::runtime_error("Invalid elastic correction factor");
    }
    ElectronElasticView view()const{return {rows.data(),mass,bc,xc2};}
};
} // namespace carbon
