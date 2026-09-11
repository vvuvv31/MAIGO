#include "carbon/schneider_ion_stopping_table.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    const auto table=carbon::SchneiderIonStoppingTable::from_binary(argv[1]);
    const auto flat=table.to_flat_float();
    sycl::queue q{sycl::gpu_selector_v};
    const std::array<float,10> energies{.01f,.06f,1.06f,5.06f,100.06f,438.569916f,6000.11f,-1.f,7000.f,NAN};
    constexpr std::size_t n=25*18*10;
    auto* data=sycl::malloc_device<float>(flat.size(),q);
    auto* result=sycl::malloc_shared<float>(n,q);
    q.copy(flat.data(),data,flat.size()).wait_and_throw();
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id) {
        std::size_t i=id[0];result[i]=carbon::schneider_ion_stopping_lookup(data,(i/10)%18,i/180,energies[i%10]);
    }).wait_and_throw();
    double worst=0;
    for(std::size_t i=0;i<n;++i) {
        const float e=energies[i%10];
        const float host=carbon::schneider_ion_stopping_lookup(flat.data(),(i/10)%18,i/180,e);
        if(host<0) {if(result[i]!=-1) return 3;continue;}
        if(!std::isfinite(result[i]) || result[i]<=0) return 4;
        const double rel=std::abs(result[i]-host)/host;worst=std::max(worst,rel);
        if(rel>2e-6) return 5;
    }
    sycl::free(data,q);sycl::free(result,q);
    std::cout<<"host/device queries="<<n<<" max_relative_difference="<<worst<<" PASS\n";
}
