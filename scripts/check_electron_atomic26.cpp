#include "carbon/electron_atomic26_data.hpp"
#include "carbon/electron_fp32.hpp"
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("usage: atomic26.bin queries.txt output.bin");
    carbon::ElectronAtomic26Data data(argv[1]);std::ifstream q(argv[2]);std::ofstream f(argv[3],std::ios::binary);
    unsigned z,id,n,auger,cascade;float ecut,pcut,budget;
    std::uint64_t seed=20261006;auto uniform=[&](){return carbon::electron_fp32::random_draw(seed);};
    while(q>>z>>id>>budget>>ecut>>pcut>>n>>auger>>cascade){auto bank=data.view();bank.auger=auger;bank.cascade=cascade;
        for(unsigned i=0;i<n;++i){float photon=0,electron=0;unsigned photons=0,electrons=0;
            auto emit=[&](int pdg,float e,float cosine){if(!std::isfinite(cosine)||std::abs(cosine)>1.f)throw std::runtime_error("atomic angle");if(pdg==22){photon+=e;++photons;}else{electron+=e;++electrons;}};
            const auto r=carbon::electron_composition26::sample_atomic_cascade26(bank,z,id,budget,ecut,pcut,uniform,emit);
            if(!r.valid||std::abs(r.local+r.photon+r.electron-budget)>budget*5e-7f||r.photon!=photon||r.electron!=electron||r.photons!=photons||r.electrons!=electrons)throw std::runtime_error("Atomic26 closure");
            float row[]={r.photon,r.electron,r.local,float(r.photons),float(r.electrons)};f.write(reinterpret_cast<char*>(row),sizeof(row));
        }
    }
    if(!q.eof()||!f)throw std::runtime_error("atomic fixture IO");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
