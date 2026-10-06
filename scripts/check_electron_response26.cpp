#include "carbon/electron_response26.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdint>
using namespace carbon::electron_response26;
struct Uniform {
    std::uint64_t state{20261005};Draw operator()(){state^=state<<13;state^=state>>7;state^=state<<17;return {state};}
};
struct Zero {Draw operator()(){return {0};}};
int main() {
    const float totals[]={.1f,.3f};const std::uint64_t cdf[]={UINT64_MAX/4,UINT64_MAX,UINT64_MAX/2,UINT64_MAX};
    const float energy[]={.1f,.2f,.3f,.4f},cosine[]={.2f,.4f,.6f,.8f};const std::int32_t pdg[]={11,22};
    const Domain domain{1.f,2.f,1};const JointView channel{cdf,energy,cosine,pdg,totals,2};
    Uniform uniform;unsigned right=0;double maximum_partition=0;
    for(unsigned i=0;i<200000;++i) {
        const auto sample=sample_birth(domain,1.25f,channel,uniform);
        if(!sample.valid)throw std::runtime_error("birth sampling");right+=sample.control==1;
        const float f=birth_fraction(domain,1.25f,channel);
        if(std::abs(f-.15f)>1e-6f)throw std::runtime_error("birth yield");
        const float loss=(float(uniform().bits>>32)+1.f)*0x1p-28f;
        const auto p=partition(loss,f);if(!p.valid)throw std::runtime_error("partition validity");
        const double error=std::abs(double(p.local)+p.electron-loss)/loss;
        maximum_partition=std::max(maximum_partition,error);
    }
    if(std::abs(double(right)/200000-.5)>.006||maximum_partition>2e-7)
        throw std::runtime_error("weighted control law or energy conservation");
    // Tests probe rare intervals directly; ordinary sampling cannot establish
    // preservation of1e-9-probability support in only200k draws.
    Zero zero;Domain tiny{0.1f,1.1f,1};
    if(choose(tiny,.10000001f,zero,[](unsigned){return 1.f;})!=1)
        throw std::runtime_error("small new-control support lost");
    if(choose(tiny,1.0999999f,zero,[](unsigned){return 1.f;})!=0)
        throw std::runtime_error("small old-control support lost");
    if(partition(1.f,-1.f).valid||partition(-1.f,.1f).valid||std::isfinite(birth_fraction(domain,3.f,channel)))
        throw std::runtime_error("unsupported input accepted");
    const float edges[]={.01f,1.f};const float values[]={1.f,1.f,1.f,1.f,3.f,3.f,3.f,3.f};
    const RateView rates{edges,values,1,1,4.f};
    if(std::abs(rate(domain,1.25f,rates,.5f)-1.875f)>1e-6f)
        throw std::runtime_error("actual-energy rate interpolation");
    right=0;
    for(unsigned i=0;i<200000;++i){const auto s=sample_collision(domain,1.25f,rates,.5f,channel,uniform);
        if(!s.valid)throw std::runtime_error("collision sampling");right+=s.control==1;}
    if(std::abs(double(right)/200000-.5)>.006)throw std::runtime_error("rate-weighted collision law");
    const Domain partitioned{.1f,5.f,32};
    for(unsigned i=0;i<20000;++i){
        const float rho=.1f+4.9f*float(uniform());
        const float actual=mean(partitioned,rho,[](unsigned j){return std::exp(std::log(.1f)+j*std::log(50.f)/32.f)/10.f;});
        if(std::abs(actual-rho/10.f)>2e-6f)throw std::runtime_error("partition coordinate law");
    }
    std::cout<<"{\"samples\":400000,\"maximum_partition_relative\":"<<maximum_partition
             <<",\"rare_control_checks\":true,\"component_pass\":true}\n";
}
