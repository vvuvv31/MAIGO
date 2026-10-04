#include "carbon/electron_birth_bank.hpp"
#include "carbon/electron_birth_task.hpp"
#include <cassert>
#include <iostream>
using namespace carbon;
int main(){
    ElectronBirthStateSample samples[]={{.25,.1,0},{1.,.3,1.}};
    ElectronBirthChannel low[]={{0,250,.2,0,2}},high[]={{0,250,.6,0,2}};
    ElectronBirthNodeView nodes[]={{6,1.,low,samples,2,1},{6,3.,high,samples,2,1}};
    unsigned offsets[kElectronMaterialCount+1]{};for(unsigned i=7;i<=kElectronMaterialCount;++i)offsets[i]=2;
    ElectronBirthBankView bank{nodes,2,offsets};
    auto a=prepare_independent_electron_birth(bank,6,2,100,.1);
    assert(a.valid&&a.node==0&&std::abs(a.fraction-.4)<1e-14);
    a=prepare_independent_electron_birth(bank,6,2,100,.5);assert(a.valid&&a.node==1);
    assert(!prepare_independent_electron_birth(bank,5,2,100,.5).valid);
    assert(!prepare_independent_electron_birth(bank,6,.9,100,.5).valid);
    assert(!prepare_independent_electron_birth(bank,6,2,251,.5).valid);
    assert(sample_independent_electron_birth(nodes[0],100,.25).energy==.3);
    assert(sample_independent_electron_birth(nodes[0],250,0).energy==.1);
    assert(sample_independent_electron_birth(nodes[0],250,1).energy==0);
    double mean=0;unsigned upper=0;
    for(unsigned i=0;i<10000;++i){const auto u=(i+.5)/10000.;auto b=prepare_independent_electron_birth(bank,6,2,100,u);upper+=b.node==1;mean+=sample_independent_electron_birth(nodes[0],100,u).energy;}
    assert(upper==7500&&std::abs(mean/10000-.25)<1e-12);
    for(double spin:{0.,.25,.5,.75}){const auto d=independent_electron_direction({0,0,1},.6,spin);assert(std::abs(d.z-.6)<1e-14&&std::abs(electron_joint_research::dot(d,d)-1)<1e-14);}
    low[0].fraction=0;low[0].count=0;
    a=prepare_independent_electron_birth(bank,6,1,10,.5);assert(a.valid&&a.fraction==0);
    a=prepare_independent_electron_birth(bank,6,2,10,0);assert(a.valid&&a.node==1&&std::abs(a.fraction-.3)<1e-14);
    static_assert(sizeof(QueuedElectronBirth)==24);
    std::cout<<"independent birth: energy-weighted density mixing, correlated CDF, support, zero yield, rotation and independent queue state PASS\n";
}
