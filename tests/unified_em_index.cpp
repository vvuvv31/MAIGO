#include "carbon/unified_em_view.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
struct Query {unsigned record,kind;float x;};
int main(int argc,char** argv) {
    if(argc!=3)throw std::runtime_error("usage: unified_em_index package sha");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);
    auto index=carbon::build_unified_em_index(p);std::vector<Query> queries;
    for(unsigned j=0;j<p.records.size();++j) {
        const auto& r=p.records[j];
        for(unsigned kind=0;kind<5;++kind) {
            unsigned n=kind==0?r.node_count:r.counts[kind-1];
            for(unsigned i=0;i<n;++i) {
                float x=kind==0?p.nodes[r.node_offset+i].energy:p.segments[r.offsets[kind-1]+i].lower;
                for(float e:{std::nextafter(x,0.f),x,std::nextafter(x,std::numeric_limits<float>::infinity())})queries.push_back({j,kind,e});
            }
        }
    }
    sycl::queue q(sycl::gpu_selector_v);
    auto upload=[&]<class T>(const std::vector<T>& v){auto* d=sycl::malloc_device<T>(v.size(),q);if(!d)throw std::bad_alloc();q.copy(v.data(),d,v.size()).wait_and_throw();return d;};
    auto* records=upload(p.records);auto* nodes=upload(p.nodes);auto* segments=upload(p.segments);auto* indices=upload(index);auto* inputs=upload(queries);
    auto* errors=sycl::malloc_shared<unsigned>(1,q);*errors=0;
    q.parallel_for(sycl::range<1>(queries.size()),[=](sycl::id<1> id){
        auto in=inputs[id];const auto* r=records+in.record;
        carbon::UnifiedEmPoint original{r,nodes+r->node_offset,segments,1};
        auto indexed=original;indexed.index=indices+in.record*carbon::unified_em_index_stride;
        float a,b;
        if(in.kind==0) {a=original.at(in.x).factor;b=indexed.at(in.x).factor;}
        else {a=original.raw(in.kind-1,in.x);b=indexed.raw(in.kind-1,in.x);}
        if(std::bit_cast<unsigned>(a)!=std::bit_cast<unsigned>(b)) {
            sycl::atomic_ref<unsigned,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> count(*errors);count.fetch_add(1);
        }
    }).wait_and_throw();
    const auto failed=*errors;std::cout<<"GPU exact-index queries="<<queries.size()<<" failures="<<failed<<"\n";
    sycl::free(records,q);sycl::free(nodes,q);sycl::free(segments,q);sycl::free(indices,q);sycl::free(inputs,q);sycl::free(errors,q);
    return failed?1:0;
}
