#pragma once
#include "carbon/electron_material_registry.hpp"
#include "carbon/electron_charged_density.hpp"
namespace carbon {
struct ChargedBankView {
    const ChargedResponseView* nodes{};const std::array<double,2>* majorants{};unsigned count{};
    const unsigned* section_offsets{};const unsigned* node_order{};
};
struct ChargedDensityBracket {bool valid{};unsigned low{},high{};double fraction{};};
inline ChargedDensityBracket charged_density_bracket(ChargedBankView bank,int section,double density){
    if(!bank.nodes||!bank.node_order||!bank.section_offsets||section<0||section>=kElectronMaterialCount||!std::isfinite(density)||density<=0)return {};
    const auto first=bank.section_offsets[section],end=bank.section_offsets[section+1];
    if(first>=end||end>bank.count||density<bank.nodes[bank.node_order[first]].density||density>bank.nodes[bank.node_order[end-1]].density)return {};
    unsigned hi=first;while(hi+1<end&&bank.nodes[bank.node_order[hi]].density<density)++hi;
    const unsigned lo=bank.nodes[bank.node_order[hi]].density==density?hi:hi-1;
    const auto l=bank.node_order[lo],h=bank.node_order[hi];return {true,l,h,l==h?0:(density-bank.nodes[l].density)/(bank.nodes[h].density-bank.nodes[l].density)};
}
inline std::array<double,2> charged_density_majorant(ChargedBankView bank,ChargedDensityBracket b,double density){
    std::array<double,2> out{};if(!b.valid||!bank.majorants)return out;
    for(unsigned p=0;p<2;++p)out[p]=density*((1-b.fraction)*bank.majorants[b.low][p]/bank.nodes[b.low].density+b.fraction*bank.majorants[b.high][p]/bank.nodes[b.high].density);
    return out;
}
} // namespace carbon
