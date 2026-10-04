#pragma once
// Observational samples from unchanged charged trajectories. No RNG draws,
// step changes, range rejection or proposed runtime containment proof.
#include "carbon/electron_range_guard.hpp"
namespace carbon::electron_segment_diagnostics {
namespace f=electron_fp32;
constexpr std::uint32_t capacity=200000;
constexpr std::uint64_t selection_mask=16383;
struct Sample {
    std::uint64_t history{},birth_rng{};
    std::uint32_t birth_step{},electron_step{},valid{},node{};
    float energy{},density{},elastic_rate{},dedx{},range{},hard0{},hard1{};
    float material_safety{},scoring_safety{},rounding{},position[3]{};
    std::uint32_t reserved{};
};
static_assert(sizeof(Sample)==88);
inline bool selected(std::uint64_t birth_rng,unsigned step){
    auto x=birth_rng+0x9e3779b97f4a7c15ULL*(std::uint64_t(step)+1);
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return ((x^(x>>31))&selection_mask)==0;
}
inline float safety(const f::ElectronCtGeometry& g,f::Position position,std::uint64_t cell){
    float value=INFINITY;
    for(unsigned k=0;k<3;++k){
        const auto coordinate=cell%g.bins[k];cell/=g.bins[k];
        const float low=g.origin[k]+coordinate*g.spacing[k];
        const float high=g.origin[k]+(coordinate+1)*g.spacing[k];
        value=std::min(value,std::min(position[k]-low,high-position[k]));
    }
    return value;
}
inline Sample observe(const f::ElectronEnergyPacket& packet,f::ChargedBankView bank,
    const f::ElectronCtGeometry& material,const f::ElectronCtGeometry& scoring){
    Sample s;s.energy=packet.cursor.energy_MeV;
    for(unsigned k=0;k<3;++k){
        s.position[k]=packet.cursor.position[k];
        s.rounding=std::max(s.rounding,4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(s.position[k])));
    }
    const auto face=f::next_charged_face(material,packet.cursor.position,packet.cursor.direction);
    if(!face.valid)return s;
    if(face.escaped){s.valid=2;return s;}
    const auto target=electron_range_guard_research::scoring_index(scoring,packet.cursor.position);
    if(target==UINT64_MAX){s.valid=3;return s;}
    const auto bracket=f::charged_density_bracket(bank,face.section,face.density);
    if(!bracket.valid)return s;
    const auto model=f::charged_density_model(bank.nodes+bracket.low,bank.nodes+bracket.high,face.density);
    if(s.energy<=model.lowest_energy){s.valid=4;return s;}
    const auto p=f::charged_parameters_certified(model,s.energy);
    if(!p.valid)return s;
    const auto hard=f::charged_density_majorant(bank,bracket,face.density);
    s.valid=1;s.node=bracket.low;s.density=face.density;
    s.elastic_rate=p.elastic_rate;s.dedx=p.dedx;s.range=p.range;s.hard0=hard[0];s.hard1=hard[1];
    s.material_safety=safety(material,packet.cursor.position,face.voxel);
    s.scoring_safety=safety(scoring,packet.cursor.position,target);
    return s;
}
} // namespace carbon::electron_segment_diagnostics
