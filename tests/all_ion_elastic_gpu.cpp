#include "carbon/all_ion_elastic.hpp"
#include "carbon/sha256.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 auto bank=carbon::AllIonElasticTable::load(argv[1],carbon::compute_file_sha256_hex(argv[1]));
 auto host=bank.view();sycl::queue q{sycl::gpu_selector_v};
 auto* e=sycl::malloc_device<float>(bank.energies.size(),q);auto* r=sycl::malloc_device<float>(bank.rates.size(),q);
 auto* s=sycl::malloc_device<carbon::ElasticSample>(bank.samples.size(),q);auto* m=sycl::malloc_device<double>(18,q);
 q.copy(bank.energies.data(),e,bank.energies.size());q.copy(bank.rates.data(),r,bank.rates.size());q.copy(bank.samples.data(),s,bank.samples.size());q.copy(bank.masses.data(),m,18).wait_and_throw();
 carbon::AllIonElasticView v{e,r,s,m,host.ne,host.nq};constexpr int n=18*26*6;
 auto* result=sycl::malloc_shared<carbon::ElasticDraw>(n,q);
 q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){int i=id[0],p=i/156,section=(i/6)%26;
  constexpr float energies[6]={.1f,1.f,50.f,350.f,1000.f,5999.f};float T=energies[i%6]*carbon::elastic_projectiles[p][1];
  result[i]=v.draw(p,section,T,0,0,1,.51f,.36f,.71f,.25f);
 }).wait_and_throw();
 int failed=0,valid=0;
 for(int i=0;i<n;++i){int p=i/156,section=(i/6)%26;constexpr float energies[6]={.1f,1.f,50.f,350.f,1000.f,5999.f};
  auto a=host.draw(p,section,energies[i%6]*carbon::elastic_projectiles[p][1],0,0,1,.51f,.36f,.71f,.25f);auto b=result[i];
  if(a.valid!=b.valid||a.target_z!=b.target_z||a.target_a!=b.target_a){++failed;continue;}
  if(a.valid){++valid;float ref=a.outcome.projectile_ke_MeV;if(std::abs(ref-b.outcome.projectile_ke_MeV)>1e-5f*std::max(ref,1.f)||std::abs(a.outcome.recoil_dir_z-b.outcome.recoil_dir_z)>1e-5f)++failed;}
 }
 sycl::free(e,q);sycl::free(r,q);sycl::free(s,q);sycl::free(m,q);sycl::free(result,q);
 std::cout<<"Elastic host/device: "<<n<<" queries, "<<valid<<" valid, "<<failed<<" failed\n";return failed?1:0;
}
