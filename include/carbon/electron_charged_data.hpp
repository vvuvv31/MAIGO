#pragma once
#include "carbon/electron_charged_response.hpp"
#include "carbon/sha256.hpp"
#include <array>
#include <vector>
#include <cstring>
namespace carbon {
class ChargedResponseData {
    static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
    template<class T>static void read(std::ifstream& f,T* p,std::size_t n=1){if(n&&!f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("Truncated charged response");}
public:
    std::vector<ChargedParameter> parameters;std::vector<ChargedRatePiece> rate[2];
    std::vector<double> edges;std::vector<ChargedChannel> channels;std::vector<ChargedOutcome> outcomes;
    std::array<double,31*38> elastic_rows{};double mass{},bc{},xc2{};
    ElectronUrbanMaterial urban{};int section{};double density{},electron_cut{},photon_cut{},lowest{},linear{};std::uint64_t bytes{};
    ChargedResponseData(const std::filesystem::path& path,const std::string& pin){
        require(pin.size()==64&&compute_file_sha256_hex(path)==pin,"Charged response SHA mismatch");
        bytes=std::filesystem::file_size(path);std::ifstream f(path,std::ios::binary);
        char magic[8];unsigned version,npar,nbin,nrate[2];std::uint64_t nout;
        read(f,magic,8);read(f,&version);read(f,&section);read(f,&density);read(f,&npar);read(f,&nbin);read(f,nrate,2);read(f,&nout);
        read(f,&electron_cut);read(f,&photon_cut);read(f,&lowest);read(f,&linear);
        require(!std::memcmp(magic,"MAIECH02",8)&&version==2&&section>=0&&section<25&&std::isfinite(density)&&density>0,"Charged response header");
        require(npar>=2&&npar<=1000000&&nbin>0&&nbin<=4096&&nrate[0]>0&&nrate[0]<=1000000&&nrate[1]>0&&nrate[1]<=1000000&&nout>0&&nout<=bytes/32,"Charged response counts");
        require(bytes==9600ull+48ull*(npar+nrate[0]+nrate[1])+8ull*(nbin+1)+32ull*nbin+32ull*nout,"Charged response size");
        require(std::isfinite(electron_cut)&&electron_cut>0&&std::isfinite(photon_cut)&&photon_cut>0&&std::isfinite(lowest)&&std::abs(lowest-.0001)<1e-18&&linear==.01,"Charged response cuts/native terminal rules");
        unsigned elastic_version,elastic_count;read(f,magic,8);read(f,&elastic_version);read(f,&elastic_count);read(f,&mass);read(f,&bc);read(f,&xc2);read(f,elastic_rows.data(),elastic_rows.size());read(f,&urban);
        require(!std::memcmp(magic,"MAIELS01",8)&&elastic_version==1&&elastic_count==31&&std::isfinite(mass)&&mass>0&&std::isfinite(bc)&&bc>0&&std::isfinite(xc2)&&xc2>0,"Charged elastic header");
        for(double x:elastic_rows)require(std::isfinite(x),"Charged elastic row");
        const double uv[]={urban.mass,urban.excitation,urban.e0,urban.e1,urban.e2,urban.f1,urban.f2};
        for(unsigned j=0;j<7;++j)require(std::isfinite(uv[j])&&(j<5?uv[j]>0:uv[j]>=0),"Charged Urban constants");
        parameters.resize(npar);for(unsigned p=0;p<2;++p)rate[p].resize(nrate[p]);edges.resize(nbin+1);channels.resize(2*nbin);outcomes.resize(nout);
        read(f,parameters.data(),npar);for(unsigned p=0;p<2;++p)read(f,rate[p].data(),nrate[p]);read(f,edges.data(),edges.size());read(f,channels.data(),channels.size());read(f,outcomes.data(),outcomes.size());
        require(f.peek()==std::char_traits<char>::eof(),"Charged response trailing data");
        double previous=0,range=0;
        for(const auto& p:parameters){require(std::isfinite(p.energy)&&p.energy>previous&&std::isfinite(p.range)&&p.range>range&&std::isfinite(p.elastic_mfp)&&p.elastic_mfp>0&&std::isfinite(p.dedx)&&p.dedx>0,"Charged parameter row");previous=p.energy;range=p.range;for(double r:p.diagnostic_raw_rate)require(std::isfinite(r),"Charged raw diagnostic rate");}
        previous=0;for(double e:edges){require(std::isfinite(e)&&e>previous,"Charged channel edge");previous=e;}
        require(edges.front()==parameters.front().energy&&edges.back()==parameters.back().energy,"Charged energy coverage");
        for(unsigned p=0;p<2;++p){previous=edges.front();for(const auto& r:rate[p]){require(r.low==previous&&std::isfinite(r.high)&&r.high>r.low,"Charged cubic interval");previous=r.high;for(double x:r.value)require(std::isfinite(x),"Charged cubic value");}require(previous==edges.back(),"Charged cubic coverage");}
        std::uint64_t offset=0;
        for(const auto& ch:channels){require(ch.offset==offset&&ch.offset<=nout&&ch.count<=nout-ch.offset,"Charged channel bounds");double cdf=0;
            for(auto i=ch.offset;i<ch.offset+ch.count;++i){const auto& s=outcomes[i];require(std::isfinite(s.cdf)&&s.cdf>=cdf&&s.cdf<=1,"Charged outcome CDF");cdf=s.cdf;
                require(std::isfinite(s.fraction)&&s.fraction>0&&s.fraction<=1&&std::isfinite(s.cosine)&&std::abs(s.cosine)<=1&&(s.pdg==0||s.pdg==11||s.pdg==22),"Charged outcome state");}
            require(!ch.count||cdf==1,"Charged final CDF");offset+=ch.count;}
        require(offset==nout,"Charged unreferenced outcomes");
    }
    ChargedResponseView view()const{
        ChargedResponseView v;v.parameters=parameters.data();v.parameter_count=parameters.size();
        for(unsigned p=0;p<2;++p){v.rate[p]=rate[p].data();v.rate_count[p]=rate[p].size();}
        v.edges=edges.data();v.bins=edges.size()-1;v.channels=channels.data();v.outcomes=outcomes.data();v.outcome_count=outcomes.size();v.elastic={elastic_rows.data(),mass,bc,xc2};v.urban=urban;
        v.section=section;v.density=density;v.electron_cut=electron_cut;v.photon_cut=photon_cut;v.lowest_energy=lowest;v.linear_loss_limit=linear;return v;
    }
};
static_assert(sizeof(ChargedParameter)==48&&sizeof(ChargedRatePiece)==48&&sizeof(ChargedOutcome)==32&&sizeof(ChargedChannel)==16&&sizeof(ElectronUrbanMaterial)==56);
} // namespace carbon
