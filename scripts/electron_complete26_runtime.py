"""Reviewed generator additions for the explicit independent26 dose candidate."""
def extend(s):
 def replace(a,b,n=-1):
  nonlocal s
  if a not in s:raise ValueError('Complete26 generator anchor absent: '+a[:90])
  s=s.replace(a,b,n)
 replace('#include "carbon/electron_composition_kinematics.hpp"','#include "carbon/electron_composition_kinematics.hpp"\n#include "carbon/electron_collision_composition26.hpp"')
 replace('    const electron_composition26::NativeGridRateView* rate26{};', '    const electron_composition26::CollisionComposition26* composition26{};\n    const electron_composition26::NativeGridRateView* rate26{};')
 replace('    int section{-1};float density{},electron_cut{},photon_cut{},lowest_energy{},linear_loss_limit{};', '    int section{-1};float density{},electron_cut{},photon_cut{},lowest_energy{},linear_loss_limit{};\n    float density_low{},density_high{};',1)
 replace('    const auto& low=*lower;const auto& high=*upper;', '''    const auto& low=*lower;const auto& high=*upper;
    if(low.composition26){
        if(low.section!=high.section||!std::isfinite(density)||density<low.density_low||density>low.density_high)return out;
        const auto scalar=electron_continuous_scalar(low.composition26->scalar,density,low.composition26->scalar.energy[0]);
        if(!scalar.valid)return out;
        out.lower=lower;out.upper=lower;out.section=low.section;out.density=density;out.composition26=low.composition26;
        out.rate26=low.rate26;out.moller26=low.moller26;out.urban=low.urban;out.elastic=low.elastic;
        out.lowest_energy=low.lowest_energy;out.linear_loss_limit=low.linear_loss_limit;
        // Explicit cut-fit approximation constrained to the extracted domain.
        out.electron_cut=std::clamp(scalar.electron_cut,out.rate26[0].cut_low,out.rate26[0].cut_high);
        out.photon_cut=std::clamp(scalar.photon_cut,out.rate26[1].cut_low,out.rate26[1].cut_high);return out;
    }''',1)
 replace('if(m.rate26&&process<2&&(process==1||energy>.1f))', 'if(m.rate26&&process<2&&(m.composition26||process==1||energy>.1f))')
 for fn in ['charged_parameters','charged_parameters_certified']:
  anchor='inline ChargedParameters '+fn+'(const ChargedDensityModel& m,float energy) {\n'
  replace(anchor,anchor+'''    if(m.composition26){
        const auto scalar=electron_continuous_scalar(m.composition26->scalar,m.density,energy);if(!scalar.valid)return {};
        ChargedParameters out;out.valid=true;out.elastic_rate=scalar.elastic_rate;out.dedx=scalar.stopping;out.range=scalar.range;
'''+('        for(unsigned p=0;p<2;++p){out.hard_rate[p]=charged_raw_rate(m,p,energy);if(!std::isfinite(out.hard_rate[p]))return {};}\n' if fn=='charged_parameters' else '')+'''        return out;
    }
''')
 replace('inline float charged_inverse_range(const ChargedDensityModel& m,float range) {','inline float charged_inverse_range(const ChargedDensityModel& m,float range) {\n    if(m.composition26)return electron_composition26::continuous_inverse_range26(m.composition26->scalar,m.density,range);')
 replace('template<class Uniform>inline ChargedCollision sample_charged_collision_rng(const ChargedDensityModel& m,float energy,unsigned process,Uniform& uniform){', '''template<class Uniform>inline ChargedCollision sample_charged_collision_rng(const ChargedDensityModel& m,float energy,unsigned process,Uniform& uniform){
    if(m.composition26){
        const auto draw=electron_composition26::sample_collision_composition26(*m.composition26,energy,m.electron_cut,m.photon_cut,process,uniform);
        return {draw.valid,draw.pdg,draw.energy,draw.cosine,process};
    }''')
 anchor='    const auto first=bank.section_offsets[section],end=bank.section_offsets[section+1];'
 replace(anchor,anchor+'''
    if(bank.count==26&&first+1==end){const auto index=bank.node_order[first];const auto& v=bank.nodes[index];
        if(v.composition26)return {density>=v.density_low&&density<=v.density_high,index,index,0.f};}
''',1)
 replace('    if(model.rate26){for(unsigned p=0;p<2;++p)out[p]=', '''    if(model.composition26){for(unsigned p=0;p<2;++p)out[p]=electron_composition26::native_grid_cut_majorant(model.rate26[p],density,p?model.photon_cut:model.electron_cut);return out;}
    if(model.rate26){for(unsigned p=0;p<2;++p)out[p]=''',1)
 replace('    int section{-1};float density{};\n};\nstruct PhotonRates', '    int section{-1};float density{},density_low{},density_high{};\n};\nstruct PhotonRates')
 # The calibrated photon conditional law is an explicit candidate approximation.
 # Attenuation already uses continuous density mass scaling.
 start=s.index('inline PhotonDensityBracket photon_density_bracket(')
 head=s[:start];tail=s[start:]
 anchor='    const auto first=bank.section_offsets[section],end=bank.section_offsets[section+1];'
 if anchor not in tail:raise ValueError('photon bracket anchor')
 tail=tail.replace(anchor,anchor+'''
    if(bank.count==26&&first+1==end&&bank.nodes[first].density_low>0.f){const auto& v=bank.nodes[first];
        return {density>=v.density_low&&density<=v.density_high,first,first,0.f};}
''',1);s=head+tail
 return s
