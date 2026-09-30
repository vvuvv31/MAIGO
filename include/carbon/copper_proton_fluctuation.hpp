#pragma once
#include "carbon/restricted_fluctuation_candidate.hpp"
#include "carbon/discrete_delta_candidate.hpp"
#include <cmath>

namespace carbon {
// G4_Cu constants queried from Geant4 11.3.2, not dose-fit coefficients.
// Proton-only G4UniversalFluctuation with either cut=tmax (diagnostic) or
// the native 0.05 mm Cu production cut plus compound-Poisson delta transfers.
// Delta energies are removed from the proton; electrons are not transported.
// Finite-cut validation: evidence/copper_loss_20260927/finite_slab_comparison.json.
template<class Uniform>
RestrictedFluctuationDraw<float> copper_proton_total_fluctuation(
    float kinetic, float mass, float mean, float length, float density,
    Uniform& uniform, bool finite_cut = false, float restricted_mean = -1.f) {
    // An optional native AlongStep mean is already restricted to the cut.
    // Add the separately sampled delta mean only for the bookkeeping below.
    if (restricted_mean >= 0.f) mean = restricted_mean;
    if (!(kinetic>0 && mass>0 && mean>=0 && length>=0 && density>0) ||
        !std::isfinite(kinetic+mass+mean+length+density)) return {};
    if (finite_cut && std::abs(density-8.96f)>1e-5f) return {};
    if (mean==0 || length==0) return {0.f,true};
    constexpr float electron_mass=.51099891f;
    const float tau=kinetic/mass;
    const float beta2=tau*(tau+2)/((tau+1)*(tau+1));
    const float ratio=electron_mass/mass;
    const float tmax=2*electron_mass*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);
    // 2*pi*r_e^2*m_e times native electron density (1/mm^3).
    constexpr float classical_electron_radius_mm=2.8179403262e-12f;
    constexpr float electron_density_per_mm3=2.4624709586328329e21f;
    const float cut=finite_cut ? std::min(tmax,.16144811353831331f) : tmax;
    const float variance=(tmax/beta2-.5f*cut)*
        (6.283185307179586f*electron_mass*classical_electron_radius_mm*
         classical_electron_radius_mm*electron_density_per_mm3)*length*(density/8.96f);
    float delta_mean=0.f, delta_count=0.f;
    if (cut<tmax) {
        // Bethe-Bloch spin-1/2 differential transfer cross section integrated
        // above the native 0.05 mm G4_Cu electron production cut. No fitted rate.
        // For <=250 MeV protons the native form-factor veto is inactive.
        constexpr float form_factor=2*electron_mass/(842.6f*842.6f);
        if(tmax*form_factor>1e-6f)return {}; // outside this validated candidate
        const float total2=(kinetic+mass)*(kinetic+mass);
        const float factor=(6.283185307179586f*electron_mass*
            classical_electron_radius_mm*classical_electron_radius_mm*
            electron_density_per_mm3)*length*(density/8.96f)/beta2;
        const float logarithm=std::log(tmax/cut);
        delta_count=factor*(1/cut-1/tmax-beta2*logarithm/tmax+.5f*(tmax-cut)/total2);
        delta_mean=factor*(logarithm-beta2*(1-cut/tmax)+.25f*(tmax*tmax-cut*cut)/total2);
        if (restricted_mean >= 0.f) mean += delta_mean;
        if(!(delta_count>=0 && delta_count<=8 && delta_mean>=0 && delta_mean<=mean))return {};
    }
    RestrictedFluctuationInput<float> input{kinetic,mass,mean-delta_mean,variance,variance,
        cut,tmax,.000322f,.00001f};
    RestrictedFluctuationSampler<float,Uniform> sampler(uniform);
    auto draw=sampler.sample(input,false,1.f);
    if(!draw.valid || delta_count==0)return draw;
    // Draw a compound Poisson sum, retaining the zero-event atom and rare tail.
    const float threshold=std::exp(-delta_count);float product=1;int count=0;
    do {product*=uniform();if(++count>128)return {};} while(product>threshold);
    for(int i=1;i<count;++i) {
        auto delta=sample_charged_delta_candidate(cut,tmax,beta2,0.f,mass,kinetic,.5f,0.f,uniform);
        if(delta.status!=DeltaDrawStatus::accepted)return {};
        draw.loss+=delta.energy_MeV;
    }
    return {draw.loss,std::isfinite(draw.loss) && draw.loss>=0};
}
}
