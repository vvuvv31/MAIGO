#include "carbon/electron_analytic_birth.hpp"
#include <cstdint>
#include <iostream>
struct Uniform {
    std::uint64_t state{20261005};
    float operator()(){state^=state<<13;state^=state>>7;state^=state<<17;return (float(state>>41)+.5f)*0x1p-23f;}
};
int main() {
    Uniform random;
    std::cout.precision(12);
    for(unsigned a:{1u,12u})for(float parent:{50.f,150.f,250.f})for(float cut:{.001f,.01f,.05f,.1f}) {
        double e=0.,mu=0.,ee=0.,mm=0.;unsigned count=0,invalid=0,attempts=0;
        for(unsigned n=0;n<200000;++n) {
            const auto birth=carbon::sample_analytic_electron_birth(parent,a,cut,random);
            if(!birth.valid){++invalid;continue;}
            if(birth.energy<cut||birth.cosine<0.f||birth.cosine>1.f)return 2;
            ++count;attempts+=birth.attempts;e+=birth.energy;mu+=birth.cosine;
            ee+=double(birth.energy)*birth.energy;mm+=double(birth.cosine)*birth.cosine;
        }
        std::cout<<a<<' '<<parent<<' '<<cut<<' '<<count<<' '<<invalid<<' '<<e<<' '<<mu<<' '<<ee<<' '<<mm<<' '<<attempts<<'\n';
    }
    for(float unsupported:{0.f,49.f,INFINITY,NAN})
        if(carbon::sample_analytic_electron_birth(unsupported,1,.001f,random).valid)return 3;
    if(carbon::sample_analytic_electron_birth(150.f,6,.001f,random).valid||
       carbon::sample_analytic_electron_birth(150.f,1,0.f,random).valid)return 3;
}
