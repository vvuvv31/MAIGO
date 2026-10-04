#include "carbon/electron_elastic_data.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
struct Reference {std::uint64_t seed;double energy,cosine;std::uint64_t calls;};
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: index");
    const std::filesystem::path path=argv[1];
    std::ifstream index(path);std::string magic;unsigned nodes;index>>magic>>nodes;
    if(magic!="MAIELB02"||!nodes||nodes>4096)throw std::runtime_error("Index format");
    sycl::queue q{sycl::gpu_selector_v};std::uint64_t total=0;double maxCosineError=0;
    for(unsigned node=0;node<nodes;++node){
        std::string file,refFile;unsigned count;index>>file>>refFile>>count;
        if(!index||!count||count>10000000)throw std::runtime_error("Index row");
        carbon::ElectronElasticData data(path.parent_path()/file);
        const auto reference=path.parent_path()/refFile;
        if(std::filesystem::file_size(reference)!=32ull*count)throw std::runtime_error("Reference size");
        auto* refs=sycl::malloc_shared<Reference>(count,q);auto* rows=sycl::malloc_shared<double>(31*38,q);
        auto* output=sycl::malloc_shared<carbon::ElectronElasticDraw>(count,q);auto* calls=sycl::malloc_shared<std::uint64_t>(count,q);
        auto* score=sycl::malloc_shared<float>(1,q);if(!refs||!rows||!output||!calls||!score)throw std::bad_alloc();
        std::ifstream input(reference,std::ios::binary);input.read(reinterpret_cast<char*>(refs),32ull*count);if(!input)throw std::runtime_error("Reference read");
        std::copy(data.rows.begin(),data.rows.end(),rows);*score=0;auto view=data.view();view.rows=rows;
        q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
            const auto i=id[0];auto rng=refs[i].seed;std::uint64_t n=0;
            auto uniform=[&](){++n;rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
            output[i]=carbon::sample_electron_elastic(view,refs[i].energy,uniform);calls[i]=n;
            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(*score);sum.fetch_add(1.f);
        }).wait_and_throw();
        for(unsigned i=0;i<count;++i){
            if(!output[i].valid||calls[i]!=refs[i].calls)throw std::runtime_error("GPU/native elastic RNG/cap mismatch");
            maxCosineError=std::max(maxCosineError,std::abs(output[i].cosine-refs[i].cosine));
        }
        if(*score!=count||maxCosineError>1e-10)throw std::runtime_error("GPU/native elastic state/FP32 score mismatch");
        total+=count;sycl::free(refs,q);sycl::free(rows,q);sycl::free(output,q);sycl::free(calls,q);sycl::free(score,q);
    }
    std::string extra;if(index>>extra)throw std::runtime_error("Trailing index fields");
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"nodes\":"<<nodes<<",\"samples\":"<<total<<",\"maximum_cosine_error\":"<<maxCosineError<<",\"paired_random_streams_equal\":true,\"scorer\":\"FP32\"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
