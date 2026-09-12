#include "carbon/nuclear_elastic_scattering.hpp"
#include <cstdio>
#include <cmath>
int main(){
 int failed=0,queries=0;
 for(double m1:{938.27208816,1875.61294257,3727.3794066,11174.86323534})
 for(double m2:{938.27208816,11174.86323534,14895.080645})
 for(float T:{.01f,1.f,120.f,4200.f,72000.f})
 for(float c:{-1.f,-.99f,-.74f,0.f,.74f,.99999f,1.f}){
  auto o=carbon::elastic_two_body(T,0,0,1,c,.31f,m1,m2);
  double p=std::sqrt(T*(T+2*m1));
  double p1=std::sqrt(o.projectile_ke_MeV*(o.projectile_ke_MeV+2*m1));
  double p2=std::sqrt(o.recoil_ke_MeV*(o.recoil_ke_MeV+2*m2));
  double x=p1*o.proj_dir_x+p2*o.recoil_dir_x,y=p1*o.proj_dir_y+p2*o.recoil_dir_y;
  double z=p1*o.proj_dir_z+p2*o.recoil_dir_z-p;
  if(!std::isfinite(x+y+z)||std::sqrt(x*x+y*y+z*z)>2e-6*p||
     std::abs(double(o.projectile_ke_MeV)+o.recoil_ke_MeV-T)>2e-6*T||
     o.projectile_ke_MeV<0||o.recoil_ke_MeV<0)++failed;
  ++queries;
 }
 auto max=carbon::elastic_two_body(4200,0,0,1,-1,0,11174.86323534,938.27208816);
 if(std::abs(max.recoil_ke_MeV-1353.42f)>.1f)++failed;
 auto invalid=carbon::elastic_two_body(100,0,0,1,0,0,0,1);
 if(invalid.recoil_ke_MeV!=0||invalid.projectile_ke_MeV!=0)++failed;
 std::printf("relativistic elastic: %d four-momentum checks, failures=%d, C12-H Tmax=%g MeV\n",queries,failed,max.recoil_ke_MeV);
 return failed?1:0;
}
