#include "carbon/electron_collision_composition26.hpp"
#include "carbon/electron_atomic26_data.hpp"
#include "carbon/electron_fp32.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    if(argc!=4)return 2;
    carbon::ElectronAtomic26Data atomic(argv[3]);
    std::ifstream input(argv[1],std::ios::binary);std::ofstream output(argv[2],std::ios::binary);
    auto read=[&]<class T>(T& x){input.read(reinterpret_cast<char*>(&x),sizeof(x));if(!input)throw std::runtime_error("truncated fixture");};
    unsigned cases;read(cases);
    for(unsigned i=0;i<cases;++i){
        float energy,cut,photon_cut;unsigned count,no,nd;read(energy);read(cut);read(photon_cut);read(count);read(no);read(nd);
        std::vector<carbon::electron_composition26::PenelopeOscillator> oscillators(no);std::vector<float> delta(nd);
        input.read(reinterpret_cast<char*>(oscillators.data()),no*sizeof(oscillators[0]));input.read(reinterpret_cast<char*>(delta.data()),nd*sizeof(float));
        if(!input)return 3;
        carbon::electron_composition26::CollisionComposition26 composition;
        composition.penelope={oscillators.data(),no,delta.data(),nd};composition.atomic=atomic.view();
        std::uint64_t rng=20261006+1009ULL*i;auto uniform=[&](){return carbon::electron_fp32::random_draw(rng);};
        for(unsigned j=0;j<count;++j){
            // Complete event, up to16 branches; never silently truncate.
            float data[16][3]{};unsigned n=0;float total=0.f;
            auto emit=[&](int pdg,float e,float mu){if(e<=0.f)return;total+=e;if(n<16){data[n][0]=float(pdg);data[n][1]=e;data[n][2]=mu;}++n;};
            if(!carbon::electron_composition26::sample_collision_event26(composition,energy,cut,photon_cut,0,uniform,emit)||n>16||std::abs(total-energy)>energy*1e-6f){
                std::cerr<<"invalid full event case="<<i<<" draw="<<j<<" energy="<<energy<<" total="<<total<<" branches="<<n<<'\n';return 4;}
            output.write(reinterpret_cast<const char*>(data),sizeof(data));
        }
    }
    char extra;if(input.read(&extra,1)||!output)return 5;
}
