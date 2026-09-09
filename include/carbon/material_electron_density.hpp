#pragma once
#include "carbon/material_electron_response.hpp"

namespace carbon {
struct MaterialElectronDensityBracket {
    bool valid{};std::size_t low{},high{};double upper_weight{};
};
// Same-section interpolation ONLY. No density clamp or material substitution.
// Candidate law: mix measured conditional path distributions, without scaling
// coordinates, KE or yields. It is not a claim of exact microscopic physics.
inline MaterialElectronDensityBracket material_electron_density_bracket(
    int section,double density,const MaterialElectronResponseView* tables,std::size_t count) {
    MaterialElectronDensityBracket out;
    if(!tables || !std::isfinite(density) || density<=0)return out;
    std::size_t low=count,high=count;
    for(std::size_t i=0;i<count;++i) {
        if(tables[i].section!=section)continue;
        const double r=tables[i].density_g_cm3;
        if(!std::isfinite(r) || r<=0)return out;
        if(r<=density && (low==count || r>tables[low].density_g_cm3))low=i;
        if(r>=density && (high==count || r<tables[high].density_g_cm3))high=i;
    }
    if(low==count || high==count)return out;
    out.low=low;out.high=high;
    out.upper_weight=low==high?0:(density-tables[low].density_g_cm3)/
        (tables[high].density_g_cm3-tables[low].density_g_cm3);
    out.valid=std::isfinite(out.upper_weight) && out.upper_weight>=0 && out.upper_weight<=1;
    return out;
}
struct MaterialElectronBirthDraw {ElectronBirthDraw birth{};std::size_t table{};};
inline MaterialElectronBirthDraw sample_material_electron_birth(
    int section,double density,double energy,double mix_u,double birth_u,
    const MaterialElectronResponseView* tables,std::size_t count) {
    MaterialElectronBirthDraw out;
    const auto b=material_electron_density_bracket(section,density,tables,count);
    if(!b.valid || !std::isfinite(mix_u) || mix_u<0 || mix_u>=1)return out;
    auto draw=[&](std::size_t i) {
        const auto t=tables[i];return sample_electron_birth(energy,birth_u,t.birth_channels,
            t.channel_count,t.birth_samples,t.birth_sample_count);
    };
    const auto low=draw(b.low),high=b.low==b.high?low:draw(b.high);
    if(!low.valid || !high.valid)return out;
    const double a=(1-b.upper_weight)*low.fraction,c=b.upper_weight*high.fraction,total=a+c;
    const bool upper=c>0 && (a==0 || mix_u>=a/total);
    out.table=upper?b.high:b.low;out.birth=upper?high:low;
    out.birth.fraction=total;return out;
}
} // namespace carbon
