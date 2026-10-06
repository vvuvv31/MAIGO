#pragma once
#include "carbon/electron_cut_birth.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
namespace carbon {
// Parent-dependent physics only; all material/charged/photon laws are shared.
inline ElectronProjectileBirthParameters load_electron_projectile26(
    const std::filesystem::path& path,unsigned expected_a) {
    std::ifstream f(path,std::ios::binary);
    auto read=[&]<class T>(T& x){if(!f.read(reinterpret_cast<char*>(&x),sizeof(x)))throw std::runtime_error("Projectile26 truncated");};
    char magic[8]{};if(!f.read(magic,8))throw std::runtime_error("Projectile26 missing header");
    unsigned version,z,a,spin;float mass,bragg,floor,form,magnetic;
    read(version);read(z);read(a);read(spin);read(mass);read(bragg);read(floor);read(form);read(magnetic);
    if(std::string_view(magic,8)!="MAIEPR26"||version!=1||a!=expected_a||(a!=1&&a!=12)||z!=(a==1?1u:6u)||spin>1||
       !std::isfinite(mass)||!(mass>0.f)||!std::isfinite(bragg)||bragg<0.f||!std::isfinite(floor)||floor<0.f||
       !std::isfinite(form)||form<0.f||!std::isfinite(magnetic)||magnetic<0.f)
        throw std::runtime_error("Projectile26 invalid species/parameters");
    char extra;if(f.read(&extra,1))throw std::runtime_error("Projectile26 trailing bytes");
    return {z,a,mass,bragg,floor,form,magnetic,bool(spin)};
}
}
