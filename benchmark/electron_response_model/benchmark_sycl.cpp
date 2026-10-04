#include "sampler.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>

#if !defined(CARBON_DOSE_FP32) || defined(CARBON_DOSE_FP64)
#error "Diagnostic scoring must be FP32"
#endif
using namespace electron_model_diagnostic;
struct Sample {float x,y,z;std::uint32_t cell;};
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("Usage: sampler TABLE.bin OFFSET COUNT SAMPLES OUTPUT_PREFIX");
        const auto start=std::chrono::steady_clock::now();
        const auto offset=std::stoull(argv[2]);
        const auto count=std::stoul(argv[3]);
        const auto n=std::stoull(argv[4]);
        if(!count || count>UINT32_MAX || !n || n>16000000)throw std::runtime_error("Invalid sampler dimensions");
        if((offset+count)>std::filesystem::file_size(argv[1])/sizeof(Cell))throw std::runtime_error("Truncated sampler table");
        std::vector<Cell> cells(count);std::ifstream input(argv[1],std::ios::binary);
        input.seekg(offset*sizeof(Cell));input.read(reinterpret_cast<char*>(cells.data()),count*sizeof(Cell));
        if(!input)throw std::runtime_error("Cannot read sampler table");
        double last=0;
        for(auto c:cells) {
            if(!std::isfinite(c.cdf) || c.cdf<=last || c.cdf>1 || !std::isfinite(c.u0) ||
               !std::isfinite(c.u1) || !std::isfinite(c.v0) || !std::isfinite(c.v1) ||
               c.u0<0 || c.u1<=c.u0 || c.v1<=c.v0)throw std::runtime_error("Invalid sampler cell");
            last=c.cdf;
        }
        if(last!=1.)throw std::runtime_error("Incomplete sampler CDF");
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::enable_profiling{});
        auto* device=sycl::malloc_device<Cell>(count,q);
        auto* output=sycl::malloc_device<Sample>(n,q);
        auto* scores=sycl::malloc_device<float>(128,q);
        if(!device || !output || !scores)throw std::bad_alloc();
        q.copy(cells.data(),device,count).wait_and_throw();
        const auto load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        for(unsigned pass=0;pass<4;++pass) {
            q.fill(scores,0.f,128).wait_and_throw();
            auto event=q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> item) {
                const auto i=static_cast<std::uint32_t>(item[0]);
                const auto base=mix(i+1002200001u+pass*0x9e3779b9u);
                const auto a=mix(base),b=mix(base+1),c=mix(base+2),d=mix(base+3);
                const auto sample=electron_model_diagnostic::sample(device,count,(double(a)+0.5)*0x1p-32,
                    uniform(b),uniform(c),uniform(d));
                output[i]={sample.x,sample.y,sample.z,sample.cell};
                // FP32 scoring smoke; this fixed diagnostic grid is NOT a CT dose grid.
                const unsigned bin=static_cast<unsigned>(sycl::clamp((sample.x+5.f)*12.8f,0.f,127.f));
                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(scores[bin]).fetch_add(1.f);
            });
            event.wait_and_throw();
            const auto ns=event.get_profiling_info<sycl::info::event_profiling::command_end>()-
                event.get_profiling_info<sycl::info::event_profiling::command_start>();
            std::cout<<"pass="<<pass<<" samples="<<n<<" kernel_seconds="<<double(ns)*1e-9
                     <<" samples_per_second="<<double(n)/(double(ns)*1e-9)<<'\n';
        }
        std::vector<Sample> host(n);q.copy(output,host.data(),n).wait_and_throw();
        float bins[128];q.copy(scores,bins,128).wait_and_throw();
        double sum=0;for(float v:bins)sum+=v;
        if(sum!=double(n))throw std::runtime_error("FP32 scoring count mismatch");
        std::ofstream raw(std::string(argv[5])+".samples",std::ios::binary);
        raw.write(reinterpret_cast<const char*>(host.data()),n*sizeof(Sample));raw.close();
        if(!raw)throw std::runtime_error("Sample output failed");
        std::cout<<"device="<<q.get_device().get_info<sycl::info::device::name>()
                 <<" load_to_device_seconds="<<load_seconds<<" model_bytes="<<count*sizeof(Cell)
                 <<" fp32_score_sum="<<sum<<'\n';
        sycl::free(scores,q);sycl::free(output,q);sycl::free(device,q);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
