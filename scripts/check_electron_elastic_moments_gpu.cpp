#include "carbon/electron_fp32.hpp"
#include "carbon/electron_charged_data.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
struct Case {unsigned node;float energy;};
int main(int argc,char** argv){try{
    if(argc!=5)throw std::runtime_error("usage: charged-index cases samples output-float-q");
    namespace f=carbon::electron_fp32;
    const std::filesystem::path index=argv[1];

    std::ifstream in(index);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB02"||(nodes<82||nodes>83))throw std::runtime_error("index header");
    sycl::queue queue{sycl::gpu_selector_v};
    auto* rows=sycl::malloc_shared<float>(nodes*31*38,queue);
    auto* views=sycl::malloc_shared<f::ElectronElasticView>(nodes,queue);
    if(!rows||!views)throw std::bad_alloc();
    for(unsigned i=0;i<nodes;++i){int section;double density;std::string path;
        in>>section>>density>>path;if(!in)throw std::runtime_error("index row");
        carbon::ChargedResponseData data(index.parent_path()/path);
        if(data.section!=section||data.density!=density)throw std::runtime_error("node identity");
        for(unsigned k=0;k<31*38;++k)rows[i*31*38+k]=float(data.elastic_rows[k]);
        views[i]={rows+i*31*38,float(data.mass),float(data.bc),float(data.xc2)};
    }
    std::ifstream cases_file(argv[2]);unsigned nc;cases_file>>nc;
    const unsigned samples=std::stoul(argv[3]);if(!nc||!samples||samples>1000000)throw std::runtime_error("counts");
    auto* cases=sycl::malloc_shared<Case>(nc,queue);auto* score=sycl::malloc_shared<float>(nc,queue);
    const std::size_t count=std::size_t(nc)*samples;auto* output=sycl::malloc_shared<float>(count,queue);
    if(!cases||!score||!output)throw std::bad_alloc();
    for(unsigned i=0;i<nc;++i){cases_file>>cases[i].node>>cases[i].energy;if(!cases_file||cases[i].node>=nodes)throw std::runtime_error("case row");score[i]=0;}
    queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
        const auto i=id[0];const unsigned c=i/samples;
        // Decorrelate histories before feeding the production LCG. A linear
        // seed progression can stratify its first draws and invalidate an
        // independent-history variance interpretation of the moment test.
        std::uint64_t rng=0x44149abcdefULL+i+0x9e3779b97f4a7c15ULL;
        rng=(rng^(rng>>30))*0xbf58476d1ce4e5b9ULL;
        rng=(rng^(rng>>27))*0x94d049bb133111ebULL;rng^=rng>>31;
        auto uniform=[&](){return f::random_draw(rng);};
        auto r=f::sample_electron_elastic(views[cases[c].node],cases[c].energy,uniform);
        output[i]=r.valid?r.one_minus_cosine:NAN;
        if(r.valid){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> tally(score[c]);tally.fetch_add(1.f);}
    }).wait_and_throw();
    for(unsigned i=0;i<nc;++i)if(score[i]!=samples)throw std::runtime_error("invalid/exhausted sampler or FP32 count mismatch");
    std::ofstream out(argv[4],std::ios::binary);out.write(reinterpret_cast<const char*>(output),count*sizeof(float));if(!out)throw std::runtime_error("output write");
    std::cout<<"{\"passed\":true,\"cases\":"<<nc<<",\"samples_per_case\":"<<samples<<",\"samples\":"<<count<<",\"scorer\":\"FP32\"}\n";
    sycl::free(output,queue);sycl::free(score,queue);sycl::free(cases,queue);sycl::free(views,queue);sycl::free(rows,queue);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
