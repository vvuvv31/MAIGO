#include "carbon/electron_brems26.hpp"
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
        float energy,cut;unsigned count,ne,nx,nz;read(energy);read(cut);read(count);read(ne);read(nx);read(nz);
        std::vector<float> energies(ne),fractions(nx),integral(ne*nx);
        input.read(reinterpret_cast<char*>(energies.data()),ne*sizeof(float));
        input.read(reinterpret_cast<char*>(fractions.data()),nx*sizeof(float));
        input.read(reinterpret_cast<char*>(integral.data()),ne*nx*sizeof(float));
        std::vector<unsigned> z(nz);std::vector<float> elements(nz*ne*nx);
        input.read(reinterpret_cast<char*>(z.data()),nz*sizeof(unsigned));
        input.read(reinterpret_cast<char*>(elements.data()),nz*ne*nx*sizeof(float));
        if(!input||ne<2||nx<2||!nz)return 5;
        carbon::electron_composition26::BremsSpectrum26 spectrum{energies.data(),fractions.data(),integral.data(),ne,nx,z.data(),elements.data(),nz};
        std::uint64_t rng=20261006+1009ULL*i;
        auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
        for(unsigned j=0;j<count;++j){
            const auto s=carbon::electron_composition26::sample_brems_pair26(spectrum,energy,cut,uniform);
            if(!s.valid||s.photon_energy<cut*.999999f||s.photon_energy>energy*1.000001f||
               std::abs(s.primary_energy+s.photon_energy-energy)>energy*2e-7f)return 3;
            const float data[]={s.primary_energy,s.primary_cosine,s.photon_energy,s.photon_cosine};
            output.write(reinterpret_cast<const char*>(data),sizeof(data));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 4;
}
