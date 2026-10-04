#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_birth_bank.hpp"
#include <sycl/sycl.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace carbon {
class ElectronBirthBankDevice {
    sycl::queue queue_;std::vector<void*> allocations_;
    static void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
    template<class T>static void read(std::ifstream& f,T* p,std::size_t n=1){if(n&&!f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("Truncated independent birth bank");}
    template<class T>T* upload(const std::vector<T>& host){
        if(host.empty())return nullptr;
        auto p=sycl::malloc_device<T>(host.size(),queue_);if(!p)throw std::bad_alloc();allocations_.push_back(p);
        queue_.copy(host.data(),p,host.size()).wait_and_throw();return p;
    }
public:
    ElectronBirthBankView view{};std::vector<ElectronBirthNodeView> host_nodes;std::uint64_t bytes{};
    ElectronBirthBankDevice(sycl::queue q,const std::filesystem::path& path,
                           unsigned projectile_z=1,unsigned projectile_a=1):queue_(q){
        try{
            bytes=std::filesystem::file_size(path);std::ifstream f(path,std::ios::binary);char magic[8];unsigned version,count;
            read(f,magic,8);read(f,&version);read(f,&count);
            require(!std::memcmp(magic,"MAIBIR02",8)&&version==2&&count>0&&count<=4096,"Independent birth bank header");
            std::uint64_t consumed=24;
            unsigned bank_z=0,bank_a=0;
            read(f,&bank_z);read(f,&bank_a);
            require(bank_z==projectile_z&&bank_a==projectile_a,"Independent birth projectile mismatch");
            require((bank_z==1&&bank_a==1)||(bank_z==6&&bank_a==12),"Independent birth projectile scope");
            const unsigned expected_channels=bank_z==6?51:50;
            for(unsigned i=0;i<count;++i){
                ElectronBirthNodeView n;read(f,&n.section);read(f,&n.density);read(f,&n.sample_count);read(f,&n.channel_count);
                require(n.section>=0&&n.section<kElectronMaterialCount&&std::isfinite(n.density)&&n.density>0&&n.channel_count==(n.section==int(kElectronWaterSection)&&bank_z==6?60:expected_channels),"Independent birth node scope");
                require(n.sample_count<=bytes/24,"Independent birth sample count");
                const auto size=24ull+40ull*n.channel_count+24ull*n.sample_count;
                require(consumed<=bytes&&size<=bytes-consumed,"Independent birth node size");consumed+=size;
                std::vector<ElectronBirthChannel> channels(n.channel_count);std::vector<ElectronBirthStateSample> samples(n.sample_count);
                static_assert(sizeof(ElectronBirthChannel)==40&&sizeof(ElectronBirthStateSample)==24);
                read(f,channels.data(),channels.size());read(f,samples.data(),samples.size());
                std::uint64_t offset=0;
                for(unsigned b=0;b<n.channel_count;++b){const auto& c=channels[b];
                    require(c.low==5.*b&&c.high==5.*(b+1)&&std::isfinite(c.fraction)&&c.fraction>=0&&c.fraction<=1,"Independent birth channel");
                    require(c.offset==offset&&offset<=n.sample_count&&c.count<=n.sample_count-offset&&((c.fraction>0)==(c.count>0)),"Independent birth offsets/yield");
                    double cdf=0;
                    for(auto k=c.offset;k<c.offset+c.count;++k){const auto& s=samples[k];require(std::isfinite(s.cdf)&&s.cdf>cdf&&s.cdf<=1&&std::isfinite(s.energy)&&s.energy>0&&s.energy<=(n.section==int(kElectronWaterSection)?.8:.65)&&std::isfinite(s.cosine)&&std::abs(s.cosine)<=1,"Independent birth joint sample");cdf=s.cdf;}
                    require(!c.count||cdf==1,"Independent birth final CDF");offset+=c.count;
                }
                require(offset==n.sample_count,"Independent birth unreferenced samples");
                n.channels=upload(channels);n.samples=upload(samples);host_nodes.push_back(n);
            }
            require(consumed==bytes&&f.peek()==std::char_traits<char>::eof(),"Independent birth trailing bytes");
            std::sort(host_nodes.begin(),host_nodes.end(),[](const auto& a,const auto& b){return a.section<b.section||(a.section==b.section&&a.density<b.density);});
            for(unsigned i=1;i<count;++i)require(host_nodes[i].section!=host_nodes[i-1].section||host_nodes[i].density>host_nodes[i-1].density,"Independent birth duplicate node");
            std::vector<unsigned> offsets(kElectronMaterialCount+1);unsigned pos=0;for(unsigned s=0;s<=kElectronMaterialCount;++s){while(pos<count&&host_nodes[pos].section<int(s))++pos;offsets[s]=pos;}
            view.section_offsets=upload(offsets);
            view.nodes=upload(host_nodes);view.count=count;
        }catch(...){for(auto p:allocations_)sycl::free(p,queue_);throw;}
    }
    ~ElectronBirthBankDevice(){for(auto p:allocations_)sycl::free(p,queue_);}
    ElectronBirthBankDevice(const ElectronBirthBankDevice&)=delete;
};
} // namespace carbon
