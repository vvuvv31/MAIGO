// CPU checks for the integer sampling representation and exact represented
// CT-face ownership. GPU trig/packet checks live in the companion programs.
#include "carbon/electron_fp32.hpp"
#include <bit>
#include <iostream>
#include <vector>
int main(){
 namespace f=carbon::electron_fp32;
 std::uint64_t state=99191,probabilities=0,mismatch=0,remaps=0;
 for(unsigned exponent=0;exponent<127;++exponent)for(unsigned i=0;i<65536;++i){
  const auto bits=f::random_draw(state).bits;
  const float p=std::bit_cast<float>((exponent<<23)|(unsigned(bits>>32)&0x7fffffU));
  const auto reference=static_cast<std::uint64_t>(std::ldexp(p,64));
  mismatch+=f::probability_threshold(p)!=reference;++probabilities;
 }
 for(float p:{0.f,1.f,0x1p-64f,std::nextafter(1.f,0.f),std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min()}){
  const auto reference=p>=1?UINT64_MAX:static_cast<std::uint64_t>(std::ldexp(p,64));
  mismatch+=f::probability_threshold(p)!=reference;++probabilities;
 }
 mismatch+=f::Draw{UINT64_MAX}>=1;
 mismatch+=!(f::Draw{UINT64_MAX}<f::Probability{UINT64_MAX});
 mismatch+=f::Draw{0}<f::Probability{0};
 for(unsigned i=0;i<100000;++i){
  const auto a=f::random_draw(state).bits,b=f::random_draw(state).bits;
  const auto lo=std::min(a,b),hi=std::max(a,b);if(lo==hi)continue;
  const auto bits=lo+f::random_draw(state).bits%(hi-lo);
  const auto expected=static_cast<std::uint64_t>((__uint128_t(bits-lo)<<64)/(hi-lo));
  mismatch+=f::Draw{bits}.remap(lo,hi).bits!=expected;++remaps;
 }
 std::vector<float> density(505,1.f);std::vector<std::uint8_t> section(505,1);
 f::ElectronCtGeometry g{{505,1,1},{-169.1015f,-1,-1},{.5f,2,2},density.data(),section.data()};
 unsigned faces=0,face_errors=0;
 for(int i=1;i<505;++i){const float face=g.origin[0]+i*g.spacing[0];
  for(int side=-1;side<=1;++side)for(int direction:{-1,1}){
   const float p=side?std::nextafter(face,side<0?-INFINITY:INFINITY):face;
   const auto r=f::next_charged_face(g,{p,0,0},{float(direction),0,0});
   const unsigned expected=i-((side<0)||(side==0&&direction<0));
   face_errors+=!r.valid||r.escaped||r.voxel!=expected||!(r.distance>0);++faces;
  }
 }
 const bool pass=!mismatch&&!face_errors;
 std::cout<<"{\"passed\":"<<(pass?"true":"false")<<",\"probability_samples\":"<<probabilities<<",\"remap_samples\":"<<remaps<<",\"sampling_mismatches\":"<<mismatch<<",\"face_cases\":"<<faces<<",\"face_errors\":"<<face_errors<<"}\n";
 return pass?0:1;
}
