#pragma once
#include "carbon/electron_atomic_cascade26.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>
namespace carbon {
class ElectronAtomic26Data {
    template<class T>static void read(std::ifstream& f,T& v){if(!f.read(reinterpret_cast<char*>(&v),sizeof(v)))throw std::runtime_error("Atomic26 truncated");}
public:
    std::vector<electron_composition26::AtomicVacancy26> vacancies;
    std::vector<std::vector<electron_composition26::AtomicTransition26>> radiative,auger;
    bool active{},auger_enabled{},cascade_enabled{};
    ElectronAtomic26Data(const ElectronAtomic26Data&)=delete;
    ElectronAtomic26Data& operator=(const ElectronAtomic26Data&)=delete;
    ElectronAtomic26Data(ElectronAtomic26Data&&)=default;
    ElectronAtomic26Data& operator=(ElectronAtomic26Data&&)=default;
    explicit ElectronAtomic26Data(const std::filesystem::path& path){
        std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);unsigned version,n,a,b,c;read(f,version);read(f,n);read(f,a);read(f,b);read(f,c);
        if(std::string_view(magic,8)!="MAIAT026"||version!=1||!n||n>4096||a>1||b>1||c>1)throw std::runtime_error("Atomic26 header");
        active=a;auger_enabled=b;cascade_enabled=c;vacancies.resize(n);radiative.resize(n);auger.resize(n);unsigned previous=0;
        for(unsigned i=0;i<n;++i){auto& v=vacancies[i];read(f,v.z);read(f,v.id);read(f,v.radiative_count);read(f,v.auger_count);read(f,v.binding);read(f,v.max_photon);read(f,v.max_electron);
            unsigned key=(v.z<<16)|v.id;
            if(v.z<1||v.z>104||!v.id||key<=previous||v.radiative_count>4096||v.auger_count>4096||!(v.binding>0.f)||!std::isfinite(v.binding))throw std::runtime_error("Atomic26 vacancy");previous=key;
            for(unsigned k=0;k<2;++k){auto& data=k?auger[i]:radiative[i];data.resize(k?v.auger_count:v.radiative_count);std::uint64_t last=0;float maximum=0;
                for(auto& t:data){read(f,t);if(t.cdf<last||!(t.energy>0.f)||!std::isfinite(t.energy)||t.target<=v.id||(t.ejected&&t.ejected<=v.id))throw std::runtime_error("Atomic26 transition");last=t.cdf;maximum=std::max(maximum,t.energy);}
                if(k&&!data.empty()&&last!=UINT64_MAX)throw std::runtime_error("Atomic26 Auger CDF");
                if(maximum!=(k?v.max_electron:v.max_photon))throw std::runtime_error("Atomic26 maximum");}
            v.radiative=radiative[i].data();v.auger=auger[i].data();
        }
        char extra;if(f.read(&extra,1))throw std::runtime_error("Atomic26 trailing data");
    }
    electron_composition26::AtomicBank26 view()const{return {vacancies.data(),unsigned(vacancies.size()),active,auger_enabled,cascade_enabled};}
};
static_assert(sizeof(electron_composition26::AtomicTransition26)==24);
}
