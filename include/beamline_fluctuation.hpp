#pragma once
#include "beamline_em.hpp"
#include "energy_loss_fluctuation.hpp"
namespace carbon {
// One full electronic draw, never restricted loss plus explicit delta energy.
// The caller supplies the same full response used by its mean/range limiter.
template<bool NativeMath=false,class Uniform>
EnergyLossFluctuationDraw<float> sample_beamline_fluctuation(
    float kinetic,float mass,int z,int a,float mean,float path,
    const BeamlineEmPoint& point,bool enabled,Uniform& uniform) {
    if(!point.valid() || !(kinetic>0 && mass>0 && z>0 && a>=z && mean>=0 && path>=0) ||
       !std::isfinite(kinetic+mass+mean+path))return {};
    const auto response=point.response(kinetic);
    if(!std::isfinite(response.stopping+response.ion_variance+response.universal_variance))return {};
    if(mean>=kinetic)return {kinetic,true}; // range termination never fluctuates
    if(!enabled || mean==0 || path==0)return {mean,true};
    constexpr float me=.51099891f;
    const float tau=kinetic/mass,re=me/mass;
    const float tmax=2*me*tau*(tau+2)/(1+2*(tau+1)*re+re*re);
    EnergyLossFluctuationInput<float> input{kinetic,mass,mean,
        response.ion_variance*path,response.universal_variance*path,
        tmax,point.material.excitation_MeV,point.material.e0_MeV};
    EnergyLossFluctuationSampler<float,Uniform,NativeMath,NativeMath> sampler(uniform);
    return sampler.sample(input,!(z==1&&a==1),float(z));
}
}
