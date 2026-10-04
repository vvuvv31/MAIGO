#include "carbon/electron_charged_data.hpp"
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: operator");
    carbon::ChargedResponseData data(argv[1]);const auto v=data.view();
    for(unsigned i=0;i<10001;++i){const double energy=v.edges[0]*std::pow(v.edges[v.bins]/v.edges[0],i/10000.);
        const double e=i==10000?v.edges[v.bins]:energy;const auto p=carbon::charged_parameters(v,e);if(!p.valid)throw std::runtime_error("Parameter coverage");
        for(unsigned process=0;process<2;++process)if(p.hard_rate[process]>0){const auto r=carbon::sample_charged_collision(v,e,process,.371828);if(!r.valid||r.energy>e)throw std::runtime_error("Positive-rate collision coverage");}
    }
    std::cout<<"charged operator load/coverage PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
