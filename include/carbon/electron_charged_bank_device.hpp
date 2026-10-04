#pragma once
#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_charged_bank.hpp"
#include <sycl/sycl.hpp>
namespace carbon {
class ChargedBankDevice {
    sycl::queue queue_;std::vector<void*> allocations_;
    template<class T>T* upload(const T* source,std::size_t n){if(!n)return nullptr;auto* p=sycl::malloc_device<T>(n,queue_);if(!p)throw std::bad_alloc();allocations_.push_back(p);queue_.copy(source,p,n).wait_and_throw();return p;}
public:
    ChargedBankView view{};std::vector<ChargedResponseView> host_nodes;std::uint64_t bytes{};
    ChargedBankDevice(sycl::queue q,const std::filesystem::path& index,const std::string& pin):queue_(q){try{
        if(pin.size()!=64||compute_file_sha256_hex(index)!=pin)throw std::runtime_error("Charged bank index SHA");
        std::ifstream input(index);std::string magic;unsigned n;input>>magic>>n;
        if(!input||magic!="MAIECB01"||!n||n>4096)throw std::runtime_error("Charged bank header");
        std::vector<std::array<double,2>> majorants;
        for(unsigned i=0;i<n;++i){int section;double density;std::string file,sha;input>>section>>density>>file>>sha;if(!input)throw std::runtime_error("Charged bank row");
            ChargedResponseData d(index.parent_path()/file,sha);auto v=d.view();if(section!=v.section||density!=v.density)throw std::runtime_error("Charged bank identity");
            for(const auto& old:host_nodes)if(old.section==section&&old.density==density)throw std::runtime_error("Duplicate charged node");
            v.parameters=upload(v.parameters,v.parameter_count);std::array<double,2> bound{};
            // On [0,1], the sum of absolute Lagrange basis coefficients for
            // nodes 1/8,3/8,5/8,7/8 is <=6. Thus this is a conservative global
            // majorant, including negative raw spline lobes before max(.,0).
            for(unsigned p=0;p<2;++p){for(const auto& r:d.rate[p])for(double x:r.value)bound[p]=std::max(bound[p],6*std::abs(x));v.rate[p]=upload(v.rate[p],v.rate_count[p]);}
            v.edges=upload(v.edges,v.bins+1);v.channels=upload(v.channels,2*v.bins);v.outcomes=upload(v.outcomes,v.outcome_count);v.elastic.rows=upload(v.elastic.rows,31*38);
            host_nodes.push_back(v);majorants.push_back(bound);bytes+=d.bytes;
        }
        std::string extra;if(input>>extra)throw std::runtime_error("Charged bank trailing fields");
        view.nodes=upload(host_nodes.data(),host_nodes.size());view.majorants=upload(majorants.data(),majorants.size());view.count=n;
        std::vector<unsigned> order(n),offsets(26);for(unsigned i=0;i<n;++i)order[i]=i;
        std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){const auto x=host_nodes[a],y=host_nodes[b];return x.section<y.section||(x.section==y.section&&x.density<y.density);});
        unsigned pos=0;for(unsigned sec=0;sec<=25;++sec){while(pos<n&&host_nodes[order[pos]].section<int(sec))++pos;offsets[sec]=pos;}
        view.node_order=upload(order.data(),order.size());view.section_offsets=upload(offsets.data(),offsets.size());
    }catch(...){for(auto* p:allocations_)sycl::free(p,queue_);throw;}}
    ~ChargedBankDevice(){for(auto* p:allocations_)sycl::free(p,queue_);}
    ChargedBankDevice(const ChargedBankDevice&)=delete;
};
} // namespace carbon
