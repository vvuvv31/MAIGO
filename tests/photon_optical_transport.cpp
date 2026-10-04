#include "carbon/photon_optical_transport.hpp"
#include <cassert>
#include <iostream>
using namespace carbon;
int main() {
    float rho[]={1,2,3};std::uint8_t section[]={1,2,3};
    ElectronCtGeometry g{{3,1,1},{0,0,0},{1,1,1},rho,section};
    const std::array<double,3> p{0,.5,.5},d{1,0,0};
    auto rates=[](unsigned,float r,double e){return PhotonAttenuation{e==.4,double(r)};};
    auto t=transport_photon_optical_depth(g,p,d,.4,.5,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.position[0]==.5 && t.section==1);
    t=transport_photon_optical_depth(g,p,d,.4,2.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.position[0]==1.5 && t.section==2 && t.voxel_crossings==1);
    t=transport_photon_optical_depth(g,p,d,.4,1.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.position[0]==1 && t.section==2);
    t=transport_photon_optical_depth(g,p,d,.4,7.,rates);
    assert(t.status==PhotonFlightStatus::escaped && t.position[0]==3 && t.remaining_optical_depth==1);
    t=transport_photon_optical_depth(g,{2,.5,.5},{-1,0,0},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.position[0]==1.5 && t.section==2);
    t=transport_photon_optical_depth(g,{3,.5,.5},{1,0,0},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::escaped && t.distance_mm==0 && t.remaining_optical_depth==1);
    t=transport_photon_optical_depth(g,{0,.5,.5},{-1,0,0},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::escaped && t.distance_mm==0);
    t=transport_photon_optical_depth(g,{3,.5,.5},{0,1,0},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.section==3);
    t=transport_photon_optical_depth(g,{3,.5,.5},{2,0,0},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::invalid);
    t=transport_photon_optical_depth(g,{3,.5,.5},{-1,0,0},.4,4.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.position[0]==1.5 && t.section==2);
    // Analytical layered survival, including an optically transparent cell.
    auto transparent=[](unsigned s,float r,double){return PhotonAttenuation{true,s==2?0.:double(r)};};
    unsigned exits=0;
    constexpr unsigned n=1000000;
    for(unsigned i=0;i<n;++i) {
        const double tau=-std::log((i+.5)/n);
        t=transport_photon_optical_depth(g,p,d,.4,tau,transparent);
        if(tau<1)assert(t.status==PhotonFlightStatus::interaction && std::abs(t.position[0]-tau)<1e-12);
        else if(tau<4)assert(t.status==PhotonFlightStatus::interaction && std::abs(t.position[0]-(2+(tau-1)/3))<1e-12);
        else {assert(t.status==PhotonFlightStatus::escaped);++exits;}
    }
    assert(std::abs(double(exits)/n-std::exp(-4.))<=1./n);
    auto missing=[](unsigned s,float,double){return PhotonAttenuation{s!=2,1};};
    t=transport_photon_optical_depth(g,p,d,.4,2.,missing);
    assert(t.status==PhotonFlightStatus::coverage_missing && t.position[0]==1 && t.remaining_optical_depth==1);
    float uniform_rho[8];std::uint8_t uniform_section[8];
    for(int i=0;i<8;++i){uniform_rho[i]=1;uniform_section[i]=1;}
    ElectronCtGeometry cube{{2,2,2},{-1,-1,-1},{1,1,1},uniform_rho,uniform_section};
    const double v=1/std::sqrt(3.);
    t=transport_photon_optical_depth(cube,{-.5,-.5,-.5},{v,v,v},.4,1.,rates);
    assert(t.status==PhotonFlightStatus::interaction && t.voxel_crossings==1);
    for(double x:t.position)assert(std::abs(x-(-.5+v))<1e-14);
    assert(transport_photon_optical_depth(g,p,{2,0,0},.4,1.,rates).status==PhotonFlightStatus::invalid);
    assert(transport_photon_optical_depth(g,p,d,.4,-1.,rates).status==PhotonFlightStatus::invalid);
    assert(transport_photon_optical_depth(g,p,d,-1.,1.,rates).status==PhotonFlightStatus::invalid);
    std::cout<<"photon optical depth: 1M layered analytical flights, faces/corners, reverse rays, transparent layer, coverage/invalid states PASS\n";
}
