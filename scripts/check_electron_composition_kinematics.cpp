#include "carbon/electron_composition_kinematics.hpp"
#include "carbon/electron_fp32.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::ifstream input(argv[1],std::ios::binary);std::ofstream output(argv[2],std::ios::binary);
    auto read=[&]<class T>(T& x){input.read(reinterpret_cast<char*>(&x),sizeof(x));if(!input)throw std::runtime_error("truncated fixture");};
    unsigned cases;read(cases);
    for(unsigned i=0;i<cases;++i){
        float energy,cut;unsigned count,shells;read(energy);read(cut);read(count);read(shells);
        std::vector<carbon::ElectronBirthShell> table(shells);
        input.read(reinterpret_cast<char*>(table.data()),shells*sizeof(table[0]));
        if(!input||table.empty()||table.back().cdf!=UINT64_MAX)return 5;
        carbon::ElectronCutBirthComposition composition{table.data(),shells,0.f,10.f};
        std::uint64_t rng=20261006+1009ULL*i;
        auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
        for(unsigned j=0;j<count;++j){
            const auto s=carbon::electron_composition26::sample_moller_pair(composition,energy,cut,uniform);
            if(!s.valid||s.secondary_energy<cut*.999999f||s.secondary_energy>energy*.500001f||
               std::abs(s.primary_energy+s.secondary_energy-energy)>energy*2e-7f)return 3;
            const float data[]={s.primary_energy,s.primary_cosine,s.secondary_energy,s.secondary_cosine};
            output.write(reinterpret_cast<const char*>(data),sizeof(data));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 4;
}
