#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/photon_response.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
namespace carbon {
// Production26 MAIPHF01 metadata. FP32 outcomes are validated by the uploader.
class PhotonResponseMetadata26 {
    static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
    template<class T>static void read(std::ifstream& f,T* p,std::size_t n=1){if(n&&!f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("Truncated photon response");}
public:
    std::vector<PhotonRateRow> rates;std::vector<double> edges;
    std::vector<PhotonChannel> channels;
    int section{};double density{};std::uint64_t bytes{},outcome_count{};
    PhotonResponseMetadata26(const std::filesystem::path& path) {
        bytes=std::filesystem::file_size(path);std::ifstream f(path,std::ios::binary);
        char magic[8];std::uint32_t version,nrate,nbin;std::uint64_t nout;
        read(f,magic,8);read(f,&version);read(f,&section);read(f,&density);read(f,&nrate);read(f,&nbin);read(f,&nout);
        static_assert(sizeof(PhotonRateRow)==32&&sizeof(PhotonOutcome)==32&&sizeof(PhotonChannel)==16);
        outcome_count=nout;
        require(!std::memcmp(magic,"MAIPHF01",8)&&version==1&&section>=0&&section<26&&std::isfinite(density)&&density>0,"Photon response header");
        require(nrate>=2&&nrate<=1000000&&nbin>0&&nbin<=4096&&nout>0&&nout<=bytes/24,"Photon response counts");
        require(bytes==40ull+32ull*nrate+8ull*(nbin+1)+16ull*3*nbin+24ull*nout,"Photon response size");
        rates.resize(nrate);edges.resize(nbin+1);channels.resize(3*nbin);
        read(f,rates.data(),rates.size());read(f,edges.data(),edges.size());read(f,channels.data(),channels.size());f.seekg(nout*24ull,std::ios::cur);
        require(f.peek()==std::char_traits<char>::eof(),"Photon response trailing bytes");
        double previous=0;
        for(const auto& r:rates){require(std::isfinite(r.energy)&&r.energy>previous,"Photon rate energy");previous=r.energy;for(double x:r.rate)require(std::isfinite(x)&&x>=0,"Photon rate value");}
        previous=0;for(double e:edges){require(std::isfinite(e)&&e>previous,"Photon channel energy");previous=e;}
        std::uint64_t offset=0;
        for(const auto& c:channels){require(c.offset==offset&&c.offset<=nout&&c.count<=nout-c.offset,"Photon channel bounds");offset+=c.count;
        }
        require(offset==nout,"Photon unreferenced outcomes");
    }

};
} // namespace carbon
