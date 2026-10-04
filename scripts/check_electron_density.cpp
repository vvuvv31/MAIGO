#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_charged_density.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
int main(int argc,char** argv){try{
 if(argc!=7)throw std::runtime_error("usage: lower sha upper sha density native.csv");
 carbon::ChargedResponseData a(argv[1],argv[2]),b(argv[3],argv[4]);const auto low=a.view(),high=b.view();auto m=carbon::charged_density_model(&low,&high,std::stod(argv[5]));
 std::ifstream f(argv[6]);std::string line;std::getline(f,line);std::cout<<"energy,elastic_rate,dedx,range,ionization,brem,native_elastic_rate,native_dedx,native_range,native_ionization,native_brem\n"<<std::setprecision(17);
 while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);double x[12];for(auto& v:x)in>>v;if(!in)throw std::runtime_error("native row");
  auto p=carbon::charged_parameters(m,x[0]);if(!p.valid)throw std::runtime_error("interpolated parameters");
  for(unsigned proc=0;proc<2;++proc)if(p.hard_rate[proc]>0)for(double u:{0.,.123,.5,.999999999999}){auto d=carbon::sample_charged_collision(m,x[0],proc,u);if(!d.valid||d.energy>x[0])throw std::runtime_error("mixed collision support");}
  std::cout<<x[0]<<','<<p.elastic_rate<<','<<p.dedx<<','<<p.range<<','<<p.hard_rate[0]<<','<<p.hard_rate[1]<<','<<1/x[1]<<','<<x[6]<<','<<x[7]<<','<<std::max(0.,x[10])<<','<<std::max(0.,x[11])<<'\n';
 }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
