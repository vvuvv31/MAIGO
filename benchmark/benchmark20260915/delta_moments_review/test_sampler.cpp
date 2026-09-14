#include "carbon/delta_moments_candidate.hpp"
#include <random>
#include <iostream>
int main(){
 std::mt19937 rng(20260915);auto u=[&](){return float((rng()>>8)*0x1p-24);};
 constexpr int n=2000000;
 for(float k:{.001f,.01f,.1f,1.f,10.f,100.f,1000.f}){
  double sum=0,sq=0;for(int i=0;i<n;++i){float x=carbon::delta_moment_draw(k,k,u);if(!(x>=0 && std::isfinite(x)))return 1;sum+=x;sq+=double(x)*x;}
  double m=sum/n,v=sq/n-m*m,dm=std::abs(m/k-1),dv=std::abs(v/k-1);
  std::cout<<"shape="<<k<<" mean="<<m<<" variance="<<v<<" mean_rel="<<dm<<" var_rel="<<dv<<"\n";
  if(dm>6/std::sqrt(n*double(k))+.0001 || dv>6*std::sqrt((6/double(k)+2)/n)+.0001)return 2;
 }
 if(carbon::delta_moment_draw(0,0,u)!=0 || carbon::delta_moment_draw(2,0,u)!=2 || std::isfinite(carbon::delta_moment_draw(-1,1,u)))return 3;
}
