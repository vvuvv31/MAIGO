#include "carbon/electron_charged_data.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: index sha");const std::filesystem::path path=argv[1];if(carbon::compute_file_sha256_hex(path)!=argv[2])throw std::runtime_error("SHA");std::ifstream in(path);std::string magic;unsigned count;in>>magic>>count;
 std::uint64_t samples=0,invalid=0,terminal_changes=0;double sum_a=0,sum_b=0,max_cell=0,max_closure=0;
 for(unsigned node=0;node<count;++node){int section;double density;std::string file,pin;in>>section>>density>>file>>pin;carbon::ChargedResponseData data(path.parent_path()/file,pin);auto model=data.view();
  for(double energy:{.00010001,.001,.01,.1,.4,.6499999})for(double fraction:{.0001,.001,.005}){
   auto par=carbon::charged_parameters(model,energy);if(!par.valid)throw std::runtime_error("parameters");const double step=std::min(.05,fraction*energy/par.dedx);double a_sum=0,b_sum=0;
   for(unsigned i=0;i<4096;++i){std::uint64_t a=0x836429aceULL+node*0x9e3779b97f4a7c15ULL+i*747796405ULL,b=a;
    auto ua=[&](){a=a*6364136223846793005ULL+1442695040888963407ULL;return (double(a>>12)+.5)*0x1p-52;};auto ub=[&](){b=b*6364136223846793005ULL+1442695040888963407ULL;return (double(b>>12)+.5)*0x1p-52;};
    auto x=carbon::sample_charged_loss<false>(model,energy,step,ua);auto y=carbon::sample_charged_loss<true>(model,energy,step,ub);++samples;invalid+=!x.valid||!y.valid;if(!x.valid||!y.valid)continue;terminal_changes+=x.stopped!=y.stopped;a_sum+=x.deposit;b_sum+=y.deposit;max_closure=std::max(max_closure,std::abs(y.deposit+y.remaining-energy));
   }
   if(a_sum>0&&std::abs(b_sum/a_sum-1)>max_cell){max_cell=std::abs(b_sum/a_sum-1);std::cerr<<std::setprecision(17)<<"max node="<<node<<" energy="<<energy<<" fraction="<<fraction<<" mean="<<step*par.dedx<<" cut="<<model.electron_cut<<" error="<<b_sum/a_sum-1<<" sums="<<a_sum<<","<<b_sum<<"\n";}sum_a+=a_sum;sum_b+=b_sum;
  }
 }
 const double relative=sum_b/sum_a-1;bool pass=!invalid&&max_closure<1e-12&&std::abs(relative)<.001&&max_cell<.01;
 std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"samples\":"<<samples<<",\"invalid\":"<<invalid<<",\"terminal_changes\":"<<terminal_changes<<",\"mean_deposit_relative_change\":"<<relative<<",\"max_cell_mean_relative_change\":"<<max_cell<<",\"maximum_energy_closure_MeV\":"<<max_closure<<"}\n";return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
