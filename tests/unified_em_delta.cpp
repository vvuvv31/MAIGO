#include "carbon/unified_em_view.hpp"
#include <iostream>
#include <random>
int main(int argc,char** argv){
    if(argc!=3)throw std::runtime_error("usage: unified_em_delta package sha");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);std::mt19937_64 engine(20260914);
    auto uniform=[&](){return std::generate_canonical<double,53>(engine);};unsigned checks=0,failed=0;
    for(unsigned s=0;s<18;++s)for(double eu:{30.,100.,300.}){
        const auto& r=p.records[s];double t=eu*r.a,m=r.mass,tau=t/m,ratio=.51099891/m;
        double tm=2*.51099891*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio),b=tau*(tau+2)/((tau+1)*(tau+1)),cut=r.cut;
        if(tm<=cut)continue;
        // Independent log-space integration of the Geant4 differential proposal
        // and magnetic/form-factor veto. Include nulls in both moments.
        double norm=0,mom[5]={};constexpr int bins=20000;double dx=std::log(tm/cut)/bins;
        for(int i=0;i<bins;++i){double e=cut*std::exp((i+.5)*dx),spin=r.spin>0?.5*e*e/((t+m)*(t+m)):0;
            double f=1-b*e/tm+spin,w=f/e*dx;norm+=w;
            double accept=1,x=r.form_factor*e;
            if(x>1e-6){accept=1/((1+x)*(1+x));if(r.spin>0){double x2=.5*.51099891*e/(m*m);accept*=1+r.magnetic_moment2*(x2-spin/f)/(1+x2);}}
            accept=std::clamp(accept,0.,1.);double power=1;for(int k=0;k<5;++k){mom[k]+=w*accept*power;power*=e;}
        }
        for(double& v:mom)v/=norm;
        double sums[3]={};constexpr int n=200000;
        for(int k=0;k<n;++k){auto v=carbon::sample_charged_delta_candidate(cut,tm,b,double(r.form_factor),m,t,double(r.spin),double(r.magnetic_moment2),uniform);
            if(v.status!=carbon::DeltaDrawStatus::accepted && v.status!=carbon::DeltaDrawStatus::form_factor_null)throw std::runtime_error("Bad delta draw");
            sums[0]+=v.status==carbon::DeltaDrawStatus::accepted;sums[1]+=v.energy_MeV;sums[2]+=v.energy_MeV*v.energy_MeV;
        }
        for(int k=0;k<3;++k){double var=(k==0?mom[0]:mom[2*k])-mom[k]*mom[k];double err=std::abs(sums[k]/n-mom[k]);if(err>7*std::sqrt(std::max(0.,var)/n)+1e-5*std::max(1.,mom[k])){++failed;std::cerr<<"Z="<<r.z<<" A="<<r.a<<" E="<<eu<<" moment="<<k<<'\n';}++checks;}
    }
    std::cout<<"checks="<<checks<<" failures="<<failed<<'\n';return failed?1:0;
}
