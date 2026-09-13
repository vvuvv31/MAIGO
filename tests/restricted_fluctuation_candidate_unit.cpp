#ifdef NDEBUG
#undef NDEBUG
#endif
#include "carbon/restricted_fluctuation_candidate.hpp"
#include <cassert>
#include <random>
int main(){
 std::mt19937_64 engine(20260913);auto rng=[&](){return std::generate_canonical<double,53>(engine);};
 carbon::RestrictedFluctuationSampler<double,decltype(rng)> sampler(rng);
 for(double sn : {.05,.5,4.}){
  double mean=.01,var=mean*mean/(sn*sn);
  carbon::RestrictedFluctuationInput<double> p{100.,11174.,mean,var,var,.01,.01,.000075,.00001};
  double sum=0,sq=0;constexpr int n=300000;
  for(int i=0;i<n;++i){auto r=sampler.sample(p);assert(r.valid && r.loss>=0);if(sn<=.1 || sn>=2)assert(r.loss<=2*mean);sum+=r.loss;sq+=r.loss*r.loss;}
  double expected_var=sn<=.1?mean*mean/3:var;
  if(sn>=2){double phi=std::exp(-.5*sn*sn)/std::sqrt(2*3.141592653589793);expected_var*=1-2*sn*phi/std::erf(sn/std::sqrt(2.));}
  assert(std::abs(sum/n/mean-1)<.015);
  assert(std::abs((sq/n-(sum/n)*(sum/n))/expected_var-1)<.025);
 }
 // Regression: the explicit Glandz Poisson remainder must remain <=8
 // across finite-step means, including the first integration failure region.
 for(float mean=3.8f;mean<4.f;mean+=.0001f){
  auto uf=[&](){return std::generate_canonical<float,24>(engine);};
  carbon::RestrictedFluctuationSampler<float,decltype(uf)> f(uf);
  float g=1+720.f/11174.863f;
  carbon::RestrictedFluctuationInput<float> in{720.f,11174.863f,mean,.003f*mean,.003f*mean,.057023483f,1.02199782f*(g*g-1),.000075f,.00001f};
  assert(f.sample(in).valid);
 }
 auto bad=[](){return 1.;};carbon::RestrictedFluctuationSampler<double,decltype(bad)> broken(bad);
 carbon::RestrictedFluctuationInput<double> p{100,11174,.01,.0001,.0001,.01,.01,.000075,.00001};assert(!broken.sample(p).valid);
 p.mean_MeV=0;assert(sampler.sample(p).loss==0);
 p.cut_MeV=0;assert(!sampler.sample(p).valid);
}
