#include "carbon/electron_cut_rates.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    if(argc!=3)return 2;std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
    auto read=[&]<class T>(T& v){if(!in.read(reinterpret_cast<char*>(&v),sizeof(v)))throw std::runtime_error("truncated fixture");};
    auto array=[&](unsigned n){std::vector<float> x(n);if(!in.read(reinterpret_cast<char*>(x.data()),n*sizeof(float)))throw std::runtime_error("truncated array");return x;};
    unsigned models;read(models);std::uint64_t queries=0;
    for(unsigned m=0;m<models;++m){
        unsigned intervals,pieces,coeffs,count;float low,high,majorant;read(intervals);read(pieces);read(coeffs);read(count);read(low);read(high);read(majorant);
        auto energies=array(intervals+1),bounds=array(pieces+1),values=array(pieces*coeffs*intervals*4),slope=array(intervals*4);
        const carbon::ElectronCutRateView v{energies.data(),bounds.data(),values.data(),slope.data(),intervals,pieces,coeffs,low,high,majorant};
        for(unsigned j=0;j<count;++j){float density,cut,energy;read(density);read(cut);read(energy);
            const float r=carbon::electron_cut_rate(v,density,cut,energy);
            if(!std::isfinite(r)||r<0.f||r>density*majorant){std::cerr<<"invalid model="<<m<<" rho="<<density<<" cut="<<cut<<" E="<<energy<<" r="<<r<<"\n";return 3;}
            out.write(reinterpret_cast<const char*>(&r),sizeof(r));++queries;
        }
    }
    char extra;if(in.read(&extra,1)||!out)return 4;
    std::cout<<"{\"models\":"<<models<<",\"queries\":"<<queries<<",\"runtime_pass\":true}\n";
}
