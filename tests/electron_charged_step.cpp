#include "carbon/electron_charged_step.hpp"
#include <iostream>
#include <stdexcept>
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    float densities[30];std::uint8_t sections[30];for(unsigned i=0;i<30;++i){densities[i]=1;sections[i]=0;}
    // Decimal CT spacing can reconstruct an exact physical face whose
    // coordinate quotient lies just below its integer index.
    for(double origin:{-17.3,-.03,0.,19.2})for(double spacing:{.02,.1,.3})for(unsigned i=1;i<30;++i){
        carbon::ElectronCtGeometry g{{30,1,1},{origin,-1,-1},{spacing,2,2},densities,sections};
        const double face=origin+i*spacing;
        for(double direction:{-1.,1.}){const auto r=carbon::next_charged_face(g,{face,0,0},{direction,0,0});
            require(r.valid&&!r.escaped,"exact internal face invalid");require(r.voxel==(direction>0?i:i-1),"exact face ownership");require(r.distance>0,"repeated zero boundary step");}
    }
    const carbon::ElectronCtGeometry g{{1,1,1},{0,0,0},{1,1,1},densities,sections};
    require(carbon::next_charged_face(g,{0,.5,.5},{-1,0,0}).escaped,"outer face outward escape");
    require(carbon::next_charged_face(g,{1,.5,.5},{0,1,0}).valid,"outer face tangent");
    const double inv=1/std::sqrt(3.);const auto corner=carbon::next_charged_face(g,{.5,.5,.5},{inv,inv,inv});require(corner.valid&&corner.position==std::array<double,3>{1,1,1},"corner ties");
    auto uniform=[](){return .5;};carbon::ChargedResponseView v;v.lowest_energy=.0001;
    for(double energy:{1e-9,1e-5,.0001}){const auto r=carbon::sample_charged_loss(v,energy,0,uniform);require(r.valid&&r.stopped&&r.deposit==energy&&r.remaining==0,"physical subtable termination");}
    require(!carbon::sample_charged_loss(v,.4,.250001,uniform).valid,"step cap");
    require(!carbon::sample_charged_loss(v,.4,.01,uniform).valid,"missing table cannot deposit");
    require(!carbon::sample_charged_collision(v,.4,0,.5).valid,"missing branch cannot deposit");
    std::cout<<"charged geometry and terminal accounting PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
