#include "carbon/unified_em_view.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <map>
int main(int argc,char** argv){
    if(argc!=4)throw std::runtime_error("usage: unified_em_distribution package sha probe-root");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);
    unsigned queries=0,failures=0;double worst=0;
    std::mt19937_64 engine(20260913);auto uniform=[&](){return std::generate_canonical<double,53>(engine);};
    // Match native probe materials by density, not by density-node ordinal:
    // refinement may renumber nodes without changing the reference material.
    std::map<std::string,std::pair<int,float>> reference_materials;
    auto catalog=std::filesystem::path(argv[3])/"ion_1_1"/"dump_v4"/"catalog.csv";
    if(!std::filesystem::exists(catalog))catalog=std::filesystem::path(argv[3])/"ion_1_1"/"dump_v5"/"catalog.csv";
    std::ifstream materials(catalog);if(!materials)throw std::runtime_error("Missing native material catalog");
    std::string line;std::getline(materials,line);
    while(std::getline(materials,line)){
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::string name;float z,a,mass,rho;row>>name>>z>>a>>mass>>rho;
        if(name=="JEM_W" || name=="JEM_S00D1" || name=="JEM_S12D1" || name=="JEM_S24D0")reference_materials[name]={name=="JEM_W"?-1:std::stoi(name.substr(5,2)),rho};
    }
    for(unsigned mi=0;mi<p.materials.size();++mi){
        std::string name;
        for(const auto& [key,value]:reference_materials)if(p.materials[mi].section==value.first && std::abs(p.materials[mi].density-value.second)<1e-7f*value.second)name=key;
        if(name.empty())continue;
        for(unsigned j=0;j<18;++j){
            const auto& r=p.records[mi*18+j];carbon::UnifiedEmPoint point{&r,p.nodes.data()+r.node_offset,p.segments.data(),1};
            auto path=std::filesystem::path(argv[3])/("ion_"+std::to_string(r.z)+"_"+std::to_string(r.a))/"distribution_probe_v2"/name/"distribution_probe.csv";
            std::ifstream in(path);if(!in)throw std::runtime_error("Missing distribution probe "+path.string());std::string line;std::getline(in,line);
            while(std::getline(in,line)){
                std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
                double e,h,mean,n,ref1,ref2,ref4;row>>e>>h>>mean>>n>>ref1>>ref2>>ref4;
                double t=e*r.a,tau=t/r.mass,ratio=.51099891/r.mass;
                double tmax=2*.51099891*tau*(tau+2)/(1+2*(tau+1)*ratio+ratio*ratio);auto node=point.at(e);
                carbon::RestrictedFluctuationInput<double> input{t,r.mass,mean,node.dispersion*h,node.universal_dispersion*h,std::min(double(r.cut),tmax),tmax,r.excitation,r.e0};
                carbon::RestrictedFluctuationSampler<double,decltype(uniform)> sampler(uniform);
                double s1=0,s2=0,s4=0;constexpr int count=100000;
                for(int k=0;k<count;++k){auto draw=sampler.sample(input,node.fluctuation==1,double(r.z));if(!draw.valid)throw std::runtime_error("Invalid sample");double x=draw.loss;s1+=x;s2+=x*x;s4+=x*x*x*x;}
                double m1=s1/count,m2=s2/count,m4=s4/count;
                double se1=std::sqrt(std::max(0.,ref2-ref1*ref1)/n+std::max(0.,m2-m1*m1)/count);
                double se2=std::sqrt(std::max(0.,ref4-ref2*ref2)/n+std::max(0.,m4-m2*m2)/count);
                // Independent random samples: 7-sigma family-wide guard plus 0.1% table interpolation allowance.
                double score=std::max(std::abs(m1-ref1)/(7*se1+.001*ref1+1e-12),std::abs(m2-ref2)/(7*se2+.001*ref2+1e-12));worst=std::max(worst,score);++queries;
                if(score>1){++failures;std::cerr<<name<<" Z="<<r.z<<" A="<<r.a<<" E="<<e<<" mean_ratio="<<m1/ref1<<" moment2_ratio="<<m2/ref2<<" gate_ratio="<<score<<'\n';}
            }
        }
    }
    std::cout<<"queries="<<queries<<" failures="<<failures<<" worst_gate_ratio="<<worst<<'\n';return failures || queries!=432;
}
