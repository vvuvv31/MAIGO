#ifdef NDEBUG
#undef NDEBUG
#endif
#include "carbon/primary_em_mean_candidate.hpp"
#include <cassert>
#include <cmath>
int main(){
 auto cubic=[](double x){return 2*x*x*x-3*x*x+.5*x+4;};
 carbon::EmCubicSegmentCandidate<double> s{2,5,cubic(2),cubic(3),cubic(4),cubic(5)};
 for(int i=0;i<=1000;++i){double x=2+.003*i;assert(std::abs(s.value(x)-cubic(x))<1e-12);}
 // Native range inverse deliberately differs from integrating S=2. Retain it.
 auto inverse=[](double r){return r*r;};
 auto unchanged=[](double,double loss,double){return loss;};
 auto short_step=carbon::primary_restricted_mean_candidate(100.,1.,2.,10.,inverse,unchanged);
 assert(!short_step.range_branch && short_step.energy_loss==2.);
 auto long_step=carbon::primary_restricted_mean_candidate(100.,1.1,2.,10.,inverse,unchanged);
 assert(long_step.range_branch && std::abs(long_step.energy_loss-20.79)<1e-12);
 auto stop=carbon::primary_restricted_mean_candidate(100.,10.,2.,10.,inverse,unchanged);
 assert(stop.stopped && stop.energy_loss==100.);
 auto low=[](double,double,double){return .001;};
 auto fallback=carbon::primary_restricted_mean_candidate(100.,1.,2.,10.,inverse,low);
 assert(fallback.energy_loss==2.);
 auto corrected=[](double middle,double basic,double h){assert(middle==99.);return basic+.1*h;};
 auto after=carbon::primary_restricted_mean_candidate(100.,1.,2.,10.,inverse,corrected);
 assert(after.energy_loss==2.1);
}
