#include "carbon/electron_fp32.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace carbon::electron_fp32;
int main(){
 std::vector<float> rho(160*160*320,.988223493f);std::vector<std::uint8_t> sec(rho.size(),5);
 ElectronCtGeometry geom{{160,160,320},{-40,-40,0},{.5f,.5f,.5f},rho.data(),sec.data()};
 Position p{.530400276f,13.6407509f,58.9928551f};std::array<float,3>d{.873296559f,.0395948626f,-.485577345f};
 const float exact=(40-p[0])/d[0];float accumulated=(1-p[0])/d[0];for(int i=0;i<78;++i)accumulated+=.5f/d[0];
 std::cout<<"canonical_exit="<<exact<<" accumulated_exit="<<accumulated<<" drift_mm="<<(accumulated-exact)*d[0]<<'\n';
 if(!(accumulated>exact))throw std::runtime_error("Counterexample no longer reproduces old outward face drift");
 unsigned checks=0;
 for(float rate:{.001f,.003f,.01f,.02f,.03f,.05f,.1f,.2f,.4f,1.f,2.f,5.f}){
  float center=rate*exact,low=center;for(int i=0;i<128;++i)low=std::nextafter(low,0.f);
  for(int i=0;i<257;++i,low=std::nextafter(low,INFINITY)){
   auto rates=[&](int,float,float){return PhotonAttenuation{true,rate};};
   auto r=transport_photon_optical_depth(geom,p,d,.0647296757f,low,rates);
   if(r.status!=PhotonFlightStatus::interaction&&r.status!=PhotonFlightStatus::escaped){std::cerr<<"rate="<<rate<<" tau="<<low<<" reason="<<r.failure_reason<<'\n';throw std::runtime_error("Valid near-exit optical flight rejected");}
   for(int k=0;k<3;++k){const float hi=geom.origin[k]+geom.bins[k]*geom.spacing[k];if(r.position[k]<geom.origin[k]||r.position[k]>hi)throw std::runtime_error("Photon flight placed outside represented CT");}
   ++checks;
  }
 }
 // Optical-depth equality crosses a face, retaining the downstream material.
 std::vector<float> two_rho(2,1);std::vector<std::uint8_t> two_sec{5,6};ElectronCtGeometry two{{2,1,1},{0,0,0},{.5f,.5f,.5f},two_rho.data(),two_sec.data()};
 auto r=transport_photon_optical_depth(two,{.25f,.25f,.25f},{1,0,0},1,.025f,[](int,float,float){return PhotonAttenuation{true,.1f};});
 if(r.status!=PhotonFlightStatus::interaction||r.section!=6||r.position[0]!=.5f)throw std::runtime_error("Exact hazard tie did not retain downstream face ownership");
 std::cout<<"PASS "<<checks+1<<" photon geometry checks\n";
}
