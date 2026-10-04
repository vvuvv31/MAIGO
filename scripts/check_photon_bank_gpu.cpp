#include "carbon/photon_response_bank_device.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: index");
        sycl::queue q{sycl::gpu_selector_v};
        const auto start=std::chrono::steady_clock::now();
        carbon::PhotonResponseBankDevice data(q,argv[1]);
        const double load=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const auto bank=data.view;constexpr unsigned n=262400;
        auto* results=sycl::malloc_shared<carbon::IndependentPhotonResult>(n,q);
        auto* score=sycl::malloc_shared<float>(5,q);
        if(!results||!score)throw std::bad_alloc();
        for(unsigned k=0;k<5;++k)score[k]=0;
        q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            const unsigned i=id[0],node=i%bank.count;
            std::uint64_t rng=0x123456789abcdefULL+i*747796405ULL;
            auto uniform=[&](){rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (double(rng>>12)+.5)*0x1p-52;};
            float densities[3];std::uint8_t sections[3];
            for(unsigned j=0;j<3;++j){const auto m=bank.nodes[(node+j)%bank.count];densities[j]=m.density;sections[j]=m.section;}
            const carbon::ElectronCtGeometry geometry{{3,1,1},{0,-100,-100},{30,200,200},densities,sections};
            const double energy=std::exp(std::log(1e-5)+uniform()*std::log(.65/1e-5));
            const auto r=carbon::transport_independent_photon(bank,geometry,{15,0,0},{1,0,0},energy,uniform);
            results[i]=r;
            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[unsigned(r.status)]);
            sum.fetch_add(1.f); // every packet carries the same unit energy weight
        }).wait_and_throw();
        std::uint64_t counts[5]{},interactions=0,crossings=0;unsigned diagnostics=0;
        for(unsigned i=0;i<n;++i){const auto& r=results[i];++counts[unsigned(r.status)];interactions+=r.interactions;crossings+=r.voxel_crossings;
            if((r.status==carbon::IndependentPhotonStatus::invalid||r.status==carbon::IndependentPhotonStatus::coverage_missing)&&diagnostics++<20)
                std::cerr<<"failure i="<<i<<" node="<<i%bank.count<<" status="<<unsigned(r.status)<<" energy="<<std::setprecision(17)<<r.energy<<" interactions="<<r.interactions<<'\n';
        }
        float total=0;for(unsigned k=0;k<5;++k){if(score[k]!=counts[k])throw std::runtime_error("FP32 packet weight mismatch");total+=score[k];}
        if(total!=n)throw std::runtime_error("FP32 total weight mismatch");
        const bool pass=!counts[0]&&!counts[4];
        std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"packets\":"<<n<<",\"nodes\":"<<bank.count<<",\"bytes\":"<<data.bytes<<",\"mapped_bytes\":"<<data.mapped_bytes<<",\"load_seconds\":"<<load<<",\"invalid\":"<<counts[0]<<",\"deposited\":"<<counts[1]<<",\"escaped\":"<<counts[2]<<",\"electron_handoff\":"<<counts[3]<<",\"coverage_missing\":"<<counts[4]<<",\"interactions\":"<<interactions<<",\"crossings\":"<<crossings<<",\"scorer\":\"FP32\"}\n";
        sycl::free(results,q);sycl::free(score,q);return pass?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
