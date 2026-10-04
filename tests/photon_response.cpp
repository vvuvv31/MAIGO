#include "carbon/photon_response.hpp"
#include <cassert>
#include <iostream>
using namespace carbon;
int main(){
    PhotonRateRow rates[]={{.01,{1,2,1}},{.65,{1,2,1}}};
    double edges[]={.01,.65};PhotonChannel channels[]={{0,1},{1,2},{3,1}};
    PhotonOutcome rows[]={{1,1,0,0,0},{.25,.25,.6,11,0},{1,.75,.8,22,0},{1,1,.95,22,0}};
    PhotonResponseView model{rates,edges,channels,rows,2,1,4,6,1};
    auto r=photon_rates(model,6,2,.4);assert(r.valid&&r.total==8&&r.rate[1]==4);
    assert(!photon_rates(model,5,1,.4).valid&&!photon_rates(model,6,1,.001).valid);
    assert(!photon_rates(model,6,-1,.4).valid);
    unsigned counts[3]{};
    for(unsigned i=0;i<4;++i)for(unsigned j=0;j<10000;++j){
        auto d=sample_photon_collision(model,6,1,.4,{0,0,1},(i+.5)/4,(j+.5)/10000,.31);
        assert(d.valid);++counts[d.pdg==0?0:(d.pdg==11?1:2)];
        if(d.pdg==11)assert(d.energy==.1&&std::abs(d.direction.z-.6)<1e-14);
        if(d.pdg!=0)assert(std::abs(electron_joint_research::dot(d.direction,d.direction)-1)<1e-12);
    }
    assert(counts[0]==10000&&counts[1]==5000&&counts[2]==25000);
    assert(!sample_photon_collision(model,6,1,.4,{0,0,2},.5,.5,.5).valid);
    assert(!sample_photon_collision(model,6,2,.4,{0,0,1},.5,.5,.5).valid); // no unvalidated outcome-density interpolation
    channels[1].count=0;
    assert(!sample_photon_collision(model,6,1,.4,{0,0,1},.5,.5,.5).valid); // coverage, not local scoring
    assert(!sample_photon_collision(model,6,1,.4,{0,0,1},1.,.5,.5).valid);
    std::cout<<"photon response: rate mixing, all energy-weighted branches, correlated angle/energy, density/support/empty-channel guards PASS\n";
}
