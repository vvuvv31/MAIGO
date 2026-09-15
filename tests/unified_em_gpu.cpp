#include "carbon/unified_em_view.hpp"
#include <sycl/sycl.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
struct Query {std::uint32_t record;float kinetic,length,expected;};
int main(int argc,char** argv){
    if(argc!=4)throw std::runtime_error("usage: unified_em_gpu package sha export-root");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);std::vector<Query> queries;
    int previous=-2,index=0;
    for(unsigned m=0;m<p.materials.size();++m){
        int section=p.materials[m].section;if(section!=previous)index=0;else ++index;previous=section;
        std::ostringstream name;if(section<0)name<<"JEM_W";else name<<"JEM_S"<<std::setw(2)<<std::setfill('0')<<section<<"D"<<index;
        for(unsigned j=0;j<18;++j){
            auto r=p.records[m*18+j];auto path=std::filesystem::path(argv[3])/("ion_"+std::to_string(r.z)+"_"+std::to_string(r.a))/"mean_probe_v1"/name.str()/"mean_probe.csv";
            std::ifstream in(path);if(!in)throw std::runtime_error("Missing probe");std::string line;std::getline(in,line);
            while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);float e,h,mean;row>>e>>h>>mean;queries.push_back({m*18+j,e*r.a,h,mean});}
        }
    }
    sycl::queue q(sycl::gpu_selector_v);
    auto upload=[&]<class T>(const std::vector<T>& v){auto* d=sycl::malloc_device<T>(v.size(),q);if(!d)throw std::bad_alloc();q.copy(v.data(),d,v.size()).wait_and_throw();return d;};
    auto* records=upload(p.records);auto* nodes=upload(p.nodes);auto* segments=upload(p.segments);auto* inputs=upload(queries);
    auto* values=sycl::malloc_shared<float>(queries.size(),q);if(!values)throw std::bad_alloc();
    q.parallel_for(sycl::range<1>(queries.size()),[=](sycl::id<1> idx){
        auto input=inputs[idx];const auto* r=records+input.record;
        carbon::UnifiedEmPoint view{r,nodes+r->node_offset,segments,1};
        float value=view.native_mean(input.kinetic,input.length);if(input.kinetic-value<=r->lowest_kinetic)value=input.kinetic;
        values[idx]=value;
    }).wait_and_throw();
    unsigned failures=0;float maximum=0;
    for(unsigned i=0;i<queries.size();++i){float expected=queries[i].expected,error=std::abs(values[i]-expected);maximum=std::max(maximum,error/std::max(expected,1e-5f));if(!std::isfinite(values[i]) || (error>1e-5f && error>.002f*expected))++failures;}
    std::cerr<<"device="<<q.get_device().get_info<sycl::info::device::name>()<<" queries="<<queries.size()<<" maximum_relative="<<maximum<<" failures="<<failures<<"\n";
    sycl::free(records,q);sycl::free(nodes,q);sycl::free(segments,q);sycl::free(inputs,q);sycl::free(values,q);
    return failures?1:0;
}
