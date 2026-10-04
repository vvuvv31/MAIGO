#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/photon_response_data.hpp"
#include "carbon/photon_response_bank.hpp"
#include <sycl/sycl.hpp>
#include <memory>
namespace carbon {
class PhotonResponseBankDevice {
    sycl::queue queue_;std::vector<void*> allocations_;
    template<class T>T* upload(const std::vector<T>& host,bool mapped=false) {
        if(host.empty())return nullptr;
        auto* p=mapped?sycl::malloc_host<T>(host.size(),queue_):sycl::malloc_device<T>(host.size(),queue_);
        if(!p)throw std::bad_alloc();allocations_.push_back(p);
        queue_.copy(host.data(),p,host.size()).wait_and_throw();return p;
    }
    static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
public:
    PhotonResponseBankView view{};std::vector<PhotonResponseView> host_nodes;
    std::uint64_t bytes{},mapped_bytes{};
    PhotonResponseBankDevice(sycl::queue queue,const std::filesystem::path& index,bool mapped=true):queue_(queue) {
        try {
            std::ifstream input(index);std::string magic;unsigned count;
            require(bool(input>>magic>>count)&&magic=="MAIPHB02"&&count>0&&count<=4096,"Photon bank index header");
            for(unsigned i=0;i<count;++i) {
                int section;double density;std::string file;
                require(bool(input>>section>>density>>file),"Photon bank index row");
                PhotonResponseData data(index.parent_path()/file);auto node=data.view();
                require(node.section==section&&node.density==density,"Photon bank identity");
                node.rates=upload(data.rates);node.edges=upload(data.edges);node.channels=upload(data.channels);node.outcomes=upload(data.outcomes,mapped);
                host_nodes.push_back(node);bytes+=data.bytes;if(mapped)mapped_bytes+=data.outcomes.size()*sizeof(PhotonOutcome);
            }
            std::string extra;require(!(input>>extra),"Photon bank trailing fields");
            std::sort(host_nodes.begin(),host_nodes.end(),[](auto a,auto b){return a.section<b.section||(a.section==b.section&&a.density<b.density);});
            for(unsigned i=1;i<count;++i)require(host_nodes[i].section!=host_nodes[i-1].section||host_nodes[i].density>host_nodes[i-1].density,"Photon duplicate node");
            std::vector<unsigned> offsets(kElectronMaterialCount+1);unsigned pos=0;for(unsigned s=0;s<=kElectronMaterialCount;++s){while(pos<count&&host_nodes[pos].section<int(s))++pos;offsets[s]=pos;}
            view.nodes=upload(host_nodes);view.section_offsets=upload(offsets);view.count=count;
        }catch(...){for(auto* p:allocations_)sycl::free(p,queue_);throw;}
    }
    ~PhotonResponseBankDevice(){for(auto* p:allocations_)sycl::free(p,queue_);}
    PhotonResponseBankDevice(const PhotonResponseBankDevice&)=delete;
};
} // namespace carbon
