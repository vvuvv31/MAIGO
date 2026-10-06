#include "carbon/electron_penelope_composition.hpp"
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
        float energy,cut;unsigned count,shells,ndelta;read(energy);read(cut);read(count);read(shells);read(ndelta);
        std::vector<carbon::electron_composition26::PenelopeOscillator> table(shells);std::vector<float> delta(ndelta);
        input.read(reinterpret_cast<char*>(table.data()),shells*sizeof(table[0]));input.read(reinterpret_cast<char*>(delta.data()),ndelta*sizeof(float));
        if(!input||table.empty())return 3;
        carbon::electron_composition26::PenelopeComposition composition{table.data(),shells,delta.data(),ndelta};
        std::uint64_t rng=20261006+1009ULL*i;auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
        for(unsigned j=0;j<count;++j){
            const auto s=carbon::electron_composition26::sample_penelope_event(composition,energy,cut,uniform);
            if(!s.valid||std::abs(s.primary_energy+s.secondary_energy+s.relaxation_energy+s.local-energy)>energy*3e-7f){
                std::cerr<<"invalid Penelope case="<<i<<" sample="<<j<<" E="<<energy<<" cut="<<cut<<'\n';return 4;}
            const float data[]={s.primary_energy,s.primary_cosine,s.secondary_energy,s.secondary_cosine,s.relaxation_energy,s.local,float(s.null)};
            output.write(reinterpret_cast<const char*>(data),sizeof(data));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 5;
}
