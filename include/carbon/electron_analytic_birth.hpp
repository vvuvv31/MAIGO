#pragma once
#include <algorithm>
#include <cmath>

namespace carbon {
// Experimental energy-weighted delta-birth kernel. Cut and birth yield must
// come from an independently validated composition model; this kernel does
// not approximate the yield or substitute local deposition for missing data.
// Current native component probes cover parent energies >=50MeV/u only.
struct ElectronAnalyticBirth {
    bool valid{};
    float energy{}, cosine{};
    unsigned attempts{};
};
template<class Uniform> inline ElectronAnalyticBirth sample_analytic_electron_birth(
    float parent_mevu,unsigned projectile_a,float electron_cut,Uniform& uniform) {
    ElectronAnalyticBirth out;
    if((projectile_a!=1&&projectile_a!=12)||!std::isfinite(parent_mevu)||parent_mevu<50.f||
       !std::isfinite(electron_cut)||electron_cut<=0.f)return out;
    constexpr float electron_mass=.51099895f;
    const float mass=projectile_a==1?938.27208816f:11174.862f/12.f;
    const float gamma=1.f+parent_mevu/mass;
    const float beta2=1.f-1.f/(gamma*gamma);
    const float ratio=electron_mass/(projectile_a*mass);
    const float maximum=2.f*electron_mass*(gamma*gamma-1.f)/(1.f+2.f*gamma*ratio+ratio*ratio);
    if(!std::isfinite(maximum)||maximum<=electron_cut)return out;
    const float total=projectile_a*(mass+parent_mevu);
    const float momentum=projectile_a*std::sqrt(parent_mevu*(parent_mevu+2.f*mass));
    const float log_width=std::log(maximum/electron_cut);
    const float spin_max=projectile_a==1?maximum*maximum/(2.f*total*total):0.f;
    // Log-uniform proposals target T*d(sigma)/dT, not the number spectrum.
    // Rejection acts on the joint event; the cosine is derived from the sameT.
    for(unsigned attempt=1;attempt<=128;++attempt) {
        out.attempts=attempt;
        const float first=uniform(),second=uniform();
        if(!std::isfinite(first)||first<=0.f||first>=1.f||
           !std::isfinite(second)||second<=0.f||second>=1.f)return out;
        const float transfer=electron_cut*std::exp(first*log_width);
        const float spin=projectile_a==1?transfer*transfer/(2.f*total*total):0.f;
        if(second*(1.f+spin_max)>1.f-beta2*transfer/maximum+spin)continue;
        const float cosine=std::sqrt(transfer/(transfer+2.f*electron_mass))*(total+electron_mass)/momentum;
        if(!std::isfinite(cosine)||cosine<0.f||cosine>1.000002f)return out;
        out.energy=transfer;out.cosine=std::min(1.f,cosine);out.valid=true;return out;
    }
    // An exhausted rejection loop is explicitly invalid, never deposition.
    return out;
}
} // namespace carbon
