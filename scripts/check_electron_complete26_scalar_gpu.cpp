#include <sycl/sycl.hpp>
#include "carbon/electron_composition26_data.hpp"
#include <iostream>
int main(int argc,char** argv){if(argc!=2)return 2;sycl::queue q;carbon::ElectronComposition26Data data(argv[1]);auto* views=sycl::malloc_shared<carbon::ElectronContinuousScalarView>(26,q);auto* rho=sycl::malloc_shared<float>(52,q);auto* result=sycl::malloc_shared<int>(52,q);auto* logs=sycl::malloc_shared<float>(52,q);
for(unsigned s=0;s<26;++s){auto& e=data.entries[s];views[s]=e.scalar_view();auto cp=[&](std::vector<float>& v){auto* p=sycl::malloc_shared<float>(v.size(),q);std::copy(v.begin(),v.end(),p);return p;};views[s].energy=cp(e.scalar_energy);views[s].scalar=cp(e.scalar_coeff);views[s].cuts=cp(e.cut_coeff);rho[s*2]=e.density_low;rho[s*2+1]=e.density_high;}
q.parallel_for(sycl::range<1>(52),[=](sycl::id<1> i){auto v=views[i/2];logs[i]=std::log(rho[i]);result[i]=carbon::electron_continuous_scalar(v,rho[i],.0697666108608f).valid;}).wait();unsigned bad=0;
for(unsigned i=0;i<52;++i)if(!result[i]){++bad;std::cout<<"FAIL section="<<i/2<<" endpoint="<<i%2<<" rho="<<rho[i]<<" log="<<std::hexfloat<<logs[i]<<" bounds="<<views[i/2].log_density_min<<","<<views[i/2].log_density_max<<std::defaultfloat<<"\n";}std::cout<<"bad="<<bad<<"/52\n";return bad?1:0;}
