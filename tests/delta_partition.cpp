#include "carbon/delta_partition.hpp"
#include <random>
#include <iostream>
int main(){
 std::mt19937 rng(20260915);auto u=[&](){return (double(rng())+.5)/4294967296.;};
 for(float x:{0.f,1e-6f,.001f,.05f,.1f,.5f,1.f,5.f,20.f,100.f}){
  long double y=x,ref=x>0?1-2/y-2*std::expm1(-y)/(y*y):0;double got=carbon::delta_partition_fraction(x);
  if(std::abs(got-double(ref))>3e-6 || got<0 || got>1)return 1;
  std::cout<<"x="<<x<<" F="<<got<<" analytic="<<double(ref)<<"\n";
 }
 const int n=300000;
 for(double rate:{.01,.1,1.,5.,20.}){
  double sum=0,sq=0,jump=0,jump2=0;
  for(int i=0;i<n;++i){double last=0,t=0,widths2=0,weightedjumps=0;
   while(true){t+=-std::log(u())/rate;if(t>=1)break;double d=t-last;widths2+=d*d;last=t;weightedjumps+=1-t;}
   widths2+=(1-last)*(1-last);sum+=widths2;sq+=widths2*widths2;jump+=weightedjumps;jump2+=weightedjumps*weightedjumps;
  }
  double mean=sum/n,se=std::sqrt((sq/n-mean*mean)/n),ref=1-carbon::delta_partition_fraction(rate),jm=jump/n,jse=std::sqrt((jump2/n-jm*jm)/n);
  std::cout<<"rate="<<rate<<" E_sum_width_squared="<<mean<<" reference="<<ref<<" se="<<se<<" jump_integral="<<jm<<" expected="<<rate/2<<"\n";
  if(std::abs(mean-ref)>6*se+3e-6 || std::abs(jm-rate/2)>6*jse)return 2;
 }
 std::cout<<"Analytic Poisson partition and unscaled jump drift PASS\n";
}
