#include "carbon/photon_optical_transport.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: lung_rate_per_mm soft_rate_per_mm");
        const double a=std::stod(argv[1]),b=std::stod(argv[2]);
        if(!(a>0&&b>0))throw std::runtime_error("invalid rates");
        sycl::queue q{sycl::gpu_selector_v};constexpr unsigned n=1000000;
        auto* density=sycl::malloc_shared<float>(200,q);auto* section=sycl::malloc_shared<std::uint8_t>(200,q);
        auto* output=sycl::malloc_shared<carbon::PhotonFlight>(n,q);auto* score=sycl::malloc_shared<float>(3,q);
        if(!density||!section||!output||!score)throw std::bad_alloc();
        for(unsigned i=0;i<200;++i){density[i]=i<100?.2994157f:.9989110f;section[i]=i<100?1:6;}
        for(unsigned i=0;i<3;++i)score[i]=0;
        const carbon::ElectronCtGeometry g{{200,1,1},{0,-10,-10},{1,20,20},density,section};
        q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            const auto i=id[0];const double tau=-sycl::log((double(i)+.5)/n);
            const auto rates=[=](unsigned s,float,double){return carbon::PhotonAttenuation{true,s==1?a:b};};
            const auto r=carbon::transport_photon_optical_depth(g,{0,0,0},{1,0,0},.1,tau,rates);output[i]=r;
            const unsigned bin=r.status==carbon::PhotonFlightStatus::escaped?2:(r.section==1?0:1);
            sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> sum(score[bin]);sum.fetch_add(1.f);
        }).wait_and_throw();
        double max_error=0;
        for(unsigned i=0;i<n;++i){const double tau=-std::log((double(i)+.5)/n);const auto& r=output[i];
            const bool escape=tau>=100*(a+b);
            if(r.status!=(escape?carbon::PhotonFlightStatus::escaped:carbon::PhotonFlightStatus::interaction))throw std::runtime_error("GPU photon state mismatch");
            const double expected=escape?200:(tau<100*a?tau/a:100+(tau-100*a)/b);
            max_error=std::max(max_error,std::abs(r.position[0]-expected));
        }
        if(max_error>1e-9||score[0]+score[1]+score[2]!=n)throw std::runtime_error("GPU analytical flight/scoring mismatch");
        std::cout<<std::setprecision(17)<<"{\"status\":\"PASS\",\"histories\":"<<n<<",\"scorer\":\"FP32\",\"max_position_error_mm\":"<<max_error<<",\"layer1\":"<<score[0]<<",\"layer2\":"<<score[1]<<",\"escaped\":"<<score[2]<<"}\n";
        sycl::free(density,q);sycl::free(section,q);sycl::free(output,q);sycl::free(score,q);
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
