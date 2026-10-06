#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/photon_response.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
namespace carbon {
// Host-owned immutable operator; the view can be uploaded without any legacy
// material-electron response, birth reference or trajectory graph.
class PhotonResponseData {
    static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
    template<class T>static void read(std::ifstream& f,T* p,std::size_t n=1){if(n&&!f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("Truncated photon response");}
public:
    std::vector<PhotonRateRow> rates;std::vector<double> edges;
    std::vector<PhotonChannel> channels;std::vector<PhotonOutcome> outcomes;
    int section{};double density{};std::uint64_t bytes{},outcome_count{};bool compact_outcomes{};
    PhotonResponseData(const std::filesystem::path& path,bool load_outcomes=true) {
        bytes=std::filesystem::file_size(path);std::ifstream f(path,std::ios::binary);
        char magic[8];std::uint32_t version,nrate,nbin;std::uint64_t nout;
        read(f,magic,8);read(f,&version);read(f,&section);read(f,&density);read(f,&nrate);read(f,&nbin);read(f,&nout);
        static_assert(sizeof(PhotonRateRow)==32&&sizeof(PhotonOutcome)==32&&sizeof(PhotonChannel)==16);
        compact_outcomes=!std::memcmp(magic,"MAIPHF01",8);outcome_count=nout;
        require((compact_outcomes||!std::memcmp(magic,"MAIPHN01",8))&&version==1&&section>=0&&section<kElectronMaterialCount&&std::isfinite(density)&&density>0,"Photon response header");
        require(nrate>=2&&nrate<=1000000&&nbin>0&&nbin<=4096&&nout>0&&nout<=bytes/(compact_outcomes?24:32),"Photon response counts");
        require(bytes==40ull+32ull*nrate+8ull*(nbin+1)+16ull*3*nbin+(compact_outcomes?24ull:32ull)*nout,"Photon response size");
        rates.resize(nrate);edges.resize(nbin+1);channels.resize(3*nbin);if(load_outcomes){require(!compact_outcomes,"Compact photon outcomes require FP32 loader");outcomes.resize(nout);}
        read(f,rates.data(),rates.size());read(f,edges.data(),edges.size());read(f,channels.data(),channels.size());if(load_outcomes)read(f,outcomes.data(),outcomes.size());else f.seekg(nout*(compact_outcomes?24ull:32ull),std::ios::cur);
        require(f.peek()==std::char_traits<char>::eof(),"Photon response trailing bytes");
        double previous=0;
        for(const auto& r:rates){require(std::isfinite(r.energy)&&r.energy>previous,"Photon rate energy");previous=r.energy;for(double x:r.rate)require(std::isfinite(x)&&x>=0,"Photon rate value");}
        previous=0;for(double e:edges){require(std::isfinite(e)&&e>previous,"Photon channel energy");previous=e;}
        std::uint64_t offset=0;
        for(const auto& c:channels){require(c.offset==offset&&c.offset<=nout&&c.count<=nout-c.offset,"Photon channel bounds");double cdf=0;
            for(auto i=c.offset;load_outcomes&&i<c.offset+c.count;++i){const auto& s=outcomes[i];
                require(std::isfinite(s.cdf)&&s.cdf>=cdf&&s.cdf<=1,"Photon outcome CDF");cdf=s.cdf;
                require(std::isfinite(s.fraction)&&s.fraction>0&&s.fraction<=1&&std::isfinite(s.cosine)&&std::abs(s.cosine)<=1&&(s.pdg==0||s.pdg==11||s.pdg==22),"Photon outcome state");
            }
            require(!load_outcomes||!c.count||cdf==1,"Photon channel final CDF");offset+=c.count;
        }
        require(offset==nout,"Photon unreferenced outcomes");
    }
    PhotonResponseView view()const{return {rates.data(),edges.data(),channels.data(),outcomes.data(),static_cast<std::uint32_t>(rates.size()),static_cast<std::uint32_t>(edges.size()-1),outcomes.size(),section,density};}
};
} // namespace carbon
