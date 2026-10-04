#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/immutable_input_cache.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/min_json.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string_view>

namespace carbon {
// Electron indices describe immutable material-node payloads. Include every
// payload's file identity so a repeated batch never reuses a replaced operator.
inline std::string electron_bank_input_key(const std::filesystem::path& index,
                                           std::string_view expected_magic) {
    auto key=input_cache_key(index);
    std::ifstream in(index);
    std::string magic;
    unsigned count{};
    if(!(in>>magic>>count) || magic!=expected_magic || count==0 || count>4096)
        throw std::runtime_error("Invalid electron bank index: "+index.string());
    for(unsigned i=0;i<count;++i) {
        int section;
        double density;
        std::string name;
        if(!(in>>section>>density>>name) || section < 0 || section>=int(kElectronMaterialCount) ||
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
    return key;
}

// Include companion payload identities, not only the small manifest. Reading
// metadata/statting files is cheap; the validated payload loader runs on misses.
inline std::string companion_input_key(const std::filesystem::path& manifest,
                                      std::initializer_list<std::string_view> fields) {
    auto key=input_cache_key(manifest);
    std::ifstream in(manifest);
    if(!in)throw std::runtime_error("Cannot read input manifest: "+manifest.string());
    const auto root=minjson::Parser(std::string(std::istreambuf_iterator<char>(in),{})).parse();
    const auto visit=[&](auto&& self,const minjson::Value& v)->void {
        for(const auto& [name,child]:v.obj) {
            if(std::find(fields.begin(),fields.end(),name)!=fields.end()) {
                if(child.type!=minjson::Value::Type::String || child.str.empty())
                    throw std::invalid_argument("Invalid companion input path");
                const std::filesystem::path relative(child.str);
                if(relative.is_absolute())throw std::invalid_argument("Absolute companion input path");
                for(const auto& part:relative)if(part=="..")
                    throw std::invalid_argument("Companion input path traversal");
                key+="|"+input_cache_key(manifest.parent_path()/relative);
            } else self(self,child);
        }
        for(const auto& child:v.arr)self(self,child);
    };
    visit(visit,root);
    return key;
}

// Match CtGrid::load's packed-file / DICOM-series dispatch. Slice additions,
// removals, edits, material mapping and origin policy all invalidate reuse.
inline std::string ct_input_key(const std::filesystem::path& path,
                                const std::filesystem::path& schneider,
                                std::string_view origin) {
    if(!std::filesystem::is_directory(path) && !std::filesystem::is_regular_file(path))
        throw std::runtime_error("CT grid path does not exist: "+path.string());
    const auto lower_extension=[](const auto& p) {
        auto s=p.extension().string();
        std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return std::tolower(c);});
        return s;
    };
    bool packed=false;
    if(!std::filesystem::is_directory(path)) {
        const auto ext=lower_extension(path);
        if(ext!=".dcm" && ext!=".ima" && ext!=".dicom") {
            std::ifstream in(path,std::ios::binary);std::uint32_t magic=0;
            in.read(reinterpret_cast<char*>(&magic),sizeof(magic));
            packed=in && magic==CtGrid::magic_value;
        }
    }
    std::string key;
    if(packed)key=input_cache_key(path);
    else {
        const auto dir=std::filesystem::is_directory(path)?path:path.parent_path();
        key=std::filesystem::absolute(dir).lexically_normal().string();
        std::vector<std::filesystem::path> files;
        for(const auto& entry:std::filesystem::directory_iterator(dir)) {
            const auto ext=lower_extension(entry.path());
            if(entry.is_regular_file() && (ext.empty() || ext==".dcm" || ext==".ima" || ext==".dicom"))
                files.push_back(entry.path());
        }
        std::sort(files.begin(),files.end());
        for(const auto& file:files)key+="|"+input_cache_key(file);
    }
    if(!schneider.empty())key+="|"+input_cache_key(schneider);
    return key+"|origin="+std::string(origin);
}
}
