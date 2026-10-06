#include "carbon/electron_cut_birth.hpp"
#include "carbon/electron_projectile26_data.hpp"
#include "carbon/electron_fp32.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    if(argc!=3&&argc!=5)return 2;
    std::ifstream input(argv[1],std::ios::binary);std::ofstream output(argv[2],std::ios::binary);
    auto read=[&]<class T>(T& x){input.read(reinterpret_cast<char*>(&x),sizeof(x));if(!input)throw std::runtime_error("fixture truncated");};
    unsigned cases;read(cases);
    for(unsigned i=0;i<cases;++i){
        unsigned a,shell_count,n;float energy,cut;read(a);read(energy);read(cut);read(shell_count);read(n);
        std::vector<carbon::ElectronBirthShell> shells(shell_count);input.read(reinterpret_cast<char*>(shells.data()),shell_count*sizeof(shells[0]));
        if(!input||shells.empty()||shells.back().cdf!=UINT64_MAX)return 3;
        const auto parameters=argc==5?carbon::load_electron_projectile26(argv[a==1?3:4],a):carbon::electron_birth_projectile_parameters(a);
        std::uint64_t rng=20261005+i*1009;auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
        carbon::ElectronCutBirthComposition composition{shells.data(),shell_count,0.f,10.f};
        for(unsigned j=0;j<n;++j){
            const auto sample=carbon::sample_cut_electron_birth(composition,energy,parameters,cut,uniform);
            if(!sample.valid||sample.energy<cut||sample.energy>carbon::electron_birth_maximum(energy,a)*1.000002f||std::abs(sample.cosine)>1.f){
                std::cerr<<"invalid case "<<i<<" a="<<a<<" E="<<energy<<" cut="<<cut<<"\n";return 4;}
            output.write(reinterpret_cast<const char*>(&sample.energy),sizeof(float));output.write(reinterpret_cast<const char*>(&sample.cosine),sizeof(float));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 5;
}
