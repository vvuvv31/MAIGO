#pragma once
#include "carbon/primary_em_mean_candidate.hpp"
#include "carbon/restricted_fluctuation_candidate.hpp"
#include "carbon/discrete_delta_candidate.hpp"
namespace carbon {
struct JointNode {float e,full,restricted,sp,range,lambda,q2,correction,disp,uni;};
struct JointView {
 const JointNode* nodes{};const EmCubicSegmentCandidate<float>* vectors{};int offsets[4]{},counts[4]{},node_count{};
 float mass=11174.863387984389f,ratio=938.272013f/11174.863387984389f,cut=.057023483021082615f,ff=5.5102926240330359e-6f,peak{};
 float field(float e,int col) const {
  int lo=0,hi=node_count-2;while(lo<hi){int m=(lo+hi+1)/2;if(nodes[m].e<=e)lo=m;else hi=m-1;}
  int i=lo;float w=std::clamp((e-nodes[i].e)/(nodes[i+1].e-nodes[i].e),0.f,1.f);
  // Explicit fields avoid aliasing struct storage as a scalar array.
  auto pick=[&](int j){const auto& n=nodes[j];switch(col){case 0:return n.q2;case 1:return n.correction;case 2:return n.full;case 3:return n.disp;default:return n.uni;}};
  return pick(i)+w*(pick(i+1)-pick(i));
 }
 float raw(int k,float x)const {
  auto* v=vectors+offsets[k];int a=0,b=counts[k]-1;
  if(x<v[0].lower){if(k==3)return 0;if(k==2)return v[0].y0*(x/v[0].lower)*(x/v[0].lower);if(k==1)return v[0].y0*std::sqrt(x/v[0].lower);}
  while(a<b){int m=(a+b+1)/2;if(v[m].lower<=x)a=m;else b=m-1;}
  return v[a].value(x);
 }
 float rate(float e,float q)const{return std::max(0.f,q*raw(3,e*12*ratio));}
 float mean(float e,float h,float q)const {
  float t=12*e,range=raw(1,t*ratio)/(q*ratio),sp=q*raw(0,t*ratio);
  return primary_restricted_mean_candidate(t,h,sp,range,[&](float r){return raw(2,r*q*ratio)/ratio;},[&](float mid,float loss,float step){
   float eu=mid/12;
   // C12 ICRU replacement below scaled 2 MeV; additive LS above.
   return mid*ratio<=2 ? field(eu,2)*step : loss+field(eu,1)*step;
  }).energy_loss;
 }
};
struct JointRateCache {
 float threshold=1e30f,rate=0;
 float update(float e,float q,const JointView& v){
  if(e<=v.peak){if(e*1.25f<threshold){rate=v.rate(e,q);threshold=rate>0?e:0;}}
  else if(e<threshold){threshold=std::max(v.peak,e*.8f);rate=v.rate(threshold,q);}
  return rate;
 }
};
}
