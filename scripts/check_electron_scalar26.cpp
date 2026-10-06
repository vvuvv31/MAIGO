#include "carbon/electron_continuous_scalar.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> void read(std::ifstream& f,T* p,std::size_t n=1) {
    if(!f.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("truncated scalar fixture");
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream f(argv[1],std::ios::binary);
    char magic[8];read(f,magic,8);
    if(std::string(magic,8)!="ECS26T01")throw std::runtime_error("fixture schema");
    unsigned count;read(f,&count);if(count!=26)throw std::runtime_error("composition count");
    std::array<double,5> errors{};unsigned total=0;
    for(unsigned section=0;section<count;++section) {
        std::array<unsigned,5> sizes;std::array<float,2> domain;
        read(f,sizes.data(),sizes.size());read(f,domain.data(),domain.size());
        const auto [ne,ns,np,nc,nq]=sizes;
        if(ne<2||ne>4096||!ns||ns>129||!np||np>32||!nc||nc>129||!nq||nq>1000000)
            throw std::runtime_error("fixture dimensions");
        std::vector<float> energy(ne),scalar(ne*ns*3),cut(np*nc*2);
        read(f,energy.data(),energy.size());read(f,scalar.data(),scalar.size());read(f,cut.data(),cut.size());
        carbon::ElectronContinuousScalarView v{energy.data(),scalar.data(),cut.data(),ne,ns,np,nc,domain[0],domain[1]};
        for(unsigned q=0;q<nq;++q) {
            std::array<float,2> input;std::array<double,5> expected;
            read(f,input.data(),input.size());read(f,expected.data(),expected.size());
            const auto r=carbon::electron_continuous_scalar(v,input[0],input[1]);
            if(!r.valid)throw std::runtime_error("valid query rejected: section="+std::to_string(section));
            const float actual[]={r.elastic_rate,r.stopping,r.range,r.electron_cut,r.photon_cut};
            for(unsigned k=0;k<5;++k)errors[k]=std::max(errors[k],std::abs(actual[k]/expected[k]-1));
            ++total;
        }
        for(float invalid:{-1.f,0.f,INFINITY,NAN})
            if(carbon::electron_continuous_scalar(v,invalid,energy[ne/2]).valid)
                throw std::runtime_error("invalid density accepted");
        if(carbon::electron_continuous_scalar(v,1.f,energy[0]*.5f).valid||
           carbon::electron_continuous_scalar(v,1.f,energy.back()*2.f).valid)
            throw std::runtime_error("invalid energy accepted");
    }
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("fixture trailing bytes");
    std::cout.precision(12);
    std::cout<<"{\"queries\":"<<total<<",\"maximum_relative\":[";
    for(unsigned k=0;k<5;++k)std::cout<<(k?",":"")<<errors[k];
    std::cout<<"],\"invalid_density_energy_checks\":true}\n";
    return *std::max_element(errors.begin(),errors.end())<=.0026?0:1;
}
