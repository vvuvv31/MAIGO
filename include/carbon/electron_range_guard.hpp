#pragma once
// Diagnostic eligibility only. CSDA range is NOT a deterministic range bound.
// No runtime packet is stopped or reweighted by this helper.
#include "carbon/electron_fp32.hpp"
namespace carbon::electron_range_guard_research {
namespace f=electron_fp32;
inline std::uint64_t scoring_index(const f::ElectronCtGeometry& g,f::Position position){
    std::uint64_t index=0,stride=1;
    for(unsigned k=0;k<3;++k){
        const float cell=std::floor((position[k]-g.origin[k])/g.spacing[k]);
        if(!std::isfinite(cell)||cell<0||cell>=g.bins[k])return UINT64_MAX;
        index+=stride*std::uint64_t(cell);stride*=g.bins[k];
    }
    return index;
}
inline std::uint64_t eligible(const f::ElectronEnergyPacket& packet,f::ChargedBankView bank,
    const f::ElectronCtGeometry& material,const f::ElectronCtGeometry& scoring,float factor=1.5f){
    const auto position=packet.cursor.position;
    const auto face=f::next_charged_face(material,position,packet.cursor.direction);
    if(!face.valid||face.escaped)return UINT64_MAX;
    const auto target=scoring_index(scoring,position);if(target==UINT64_MAX)return target;
    const auto bracket=f::charged_density_bracket(bank,face.section,face.density);if(!bracket.valid)return UINT64_MAX;
    const auto model=f::charged_density_model(bank.nodes+bracket.low,bank.nodes+bracket.high,face.density);
    if(packet.cursor.energy_MeV<=model.lowest_energy)return UINT64_MAX;
    const auto parameters=f::charged_parameters_certified(model,packet.cursor.energy_MeV);
    if(!parameters.valid)return UINT64_MAX;
    float safety=INFINITY,rounding=0;
    // Protect all six material faces AND all six scoring faces. Neither final
    // direction nor a forward ray distance is sufficient for lateral spread.
    for(unsigned kind=0;kind<2;++kind){
        const auto& g=kind?scoring:material;auto cell=kind?target:face.voxel;
        for(unsigned k=0;k<3;++k){
            const auto coordinate=cell%g.bins[k];cell/=g.bins[k];
            const float low=g.origin[k]+coordinate*g.spacing[k],high=g.origin[k]+(coordinate+1)*g.spacing[k];
            safety=std::min(safety,std::min(position[k]-low,high-position[k]));
            rounding=std::max(rounding,4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(position[k])));
        }
    }
    return safety>factor*parameters.range+rounding?target:UINT64_MAX;
}
} // namespace carbon::electron_range_guard_research
