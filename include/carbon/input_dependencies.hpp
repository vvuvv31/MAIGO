#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/immutable_input_cache.hpp"
#include "carbon/ct_grid.hpp"
#include <cctype>
#include <cmath>
#include <fstream>
#include <string_view>

namespace carbon {
// Production indices describe exactly26 immutable composition payloads. Include every
// payload's file identity so a repeated batch never reuses a replaced operator.
inline std::string electron_bank_input_key(const std::filesystem::path& index,
                                           std::string_view expected_magic) {
    auto key=input_cache_key(index);
    std::ifstream in(index);
    std::string magic;
    unsigned count{};
    if(!(in>>magic>>count) || magic!=expected_magic ||
       (magic!="MAIEC026" && magic!="MAIPH026") || count!=26)
        throw std::runtime_error("Invalid electron bank index: "+index.string());
    for(unsigned i=0;i<count;++i) {
        int section;
        double density;
        std::string name;
        if(!(in>>section>>density>>name) || section != int(i) ||
           !std::isfinite(density) || density<=0)
            throw std::runtime_error("Invalid electron bank material row");
        const std::filesystem::path relative(name);
        if(relative.is_absolute())throw std::runtime_error("Absolute electron payload path");
        for(const auto& part:relative)if(part=="..")
            throw std::runtime_error("Electron payload path traversal");
        key+="|"+input_cache_key(index.parent_path()/relative);
    }
    std::string extra;
    if(in>>extra)throw std::runtime_error("Trailing electron bank index fields");
    if(magic=="MAIEC026")for(const char* file:{"scalar26.bin","penelope26.bin","atomic26.bin","brems26.bin","rates26.bin","proton26.bin","carbon26.bin"})key+="|"+input_cache_key(index.parent_path()/file);
    return key;
}

// Packed CCTG v3 only; slice conversion is outside this runtime.
inline std::string ct_input_key(const std::filesystem::path& path) {
    return input_cache_key(path);
}
}
