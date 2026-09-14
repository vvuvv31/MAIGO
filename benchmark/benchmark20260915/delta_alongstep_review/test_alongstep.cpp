#include "carbon/unified_em_view.hpp"
#include <iostream>
#include <cmath>
int main(){
 auto poly=[](double x){return 2+3*x-2*x*x+.3*x*x*x;};auto deriv=[](double x){return 3-4*x+.9*x*x;};
 carbon::EmCubicSegmentCandidate<double> v{2,5,poly(2),poly(3),poly(4),poly(5)};
 for(double x:{2.,2.4,3.,3.6,4.8,5.})if(std::abs(v.derivative(x)-deriv(x))>1e-12)return 1;
 if(v.derivative(1)!=0 || v.derivative(6)!=0)return 5;
 carbon::UnifiedEmRecord r{};r.is_ion=0;r.linear_limit=.02;r.ratio=1;
 carbon::UnifiedEmPoint p{};p.record=&r;carbon::UnifiedEmPointStep pre{};pre.factor=1;pre.range=10000;pre.stopping=20;pre.stopping_slope=.1f;
 double prev=0;
 for(float h:{.2f,.1f,.05f}){double exact=200*(-std::expm1(-.1*double(h)));double candidate=p.mean(1000,h,pre);double error=std::abs(candidate-exact);std::cout<<"h="<<h<<" exact="<<exact<<" candidate="<<candidate<<" error="<<error<<"\n";if(prev && prev/error<6)return 2;prev=error;}
 pre.stopping_slope=0;if(p.mean(1000,.1f,pre)!=2)return 3;
 pre.stopping_slope=.1f;pre.delta_stopping=5;float loss=p.mean(1000,.1f,pre);if(std::abs(loss-1.9875f)>1e-6)return 4;
 std::cout<<"Cubic derivatives, second-order convergence, zero slope, total-loss coupling PASS\n";
}
