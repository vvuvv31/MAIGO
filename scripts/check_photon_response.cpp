#include "carbon/photon_response_data.hpp"
#include "carbon/rng.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("usage: operator sha histories energy output");
        carbon::PhotonResponseData data(argv[1],argv[2]);const auto model=data.view();
        const auto n=std::stoull(argv[3]);const double energy=std::stod(argv[4]);
        if(!n)throw std::runtime_error("zero samples");
        std::uint64_t count[4]{},processes[3]{};double physical[4]{},cosine[4]{};
        std::ofstream file(argv[5],std::ios::binary);
        for(std::uint64_t i=0;i<n;++i) {
            auto u=[&](unsigned k){return (double(carbon::rng::random_u32(114420001,i,0,k))+.5)*0x1p-32;};
            const auto draw=carbon::sample_photon_collision(model,model.section,model.density,energy,{0,0,1},u(0),u(1),u(2));
            const unsigned k=!draw.valid?3:(draw.pdg==0?0:(draw.pdg==11?1:2));
            ++count[k];if(draw.valid){++processes[draw.process];physical[k]+=draw.energy;cosine[k]+=draw.direction.z;}
            const double row[]={double(k),draw.energy,draw.direction.x,draw.direction.y,draw.direction.z,double(draw.process)};
            file.write(reinterpret_cast<const char*>(row),sizeof(row));
        }
        if(!file)throw std::runtime_error("sample output failed");
        std::cout<<std::setprecision(17)<<"{\"histories\":"<<n<<",\"bytes\":"<<data.bytes<<",\"counts\":[";
        for(int k=0;k<4;++k)std::cout<<(k?",":"")<<count[k];std::cout<<"],\"process_counts\":[";
        for(int k=0;k<3;++k)std::cout<<(k?",":"")<<processes[k];std::cout<<"],\"energy_sums\":[";
        for(int k=0;k<4;++k)std::cout<<(k?",":"")<<physical[k];std::cout<<"],\"cosine_sums\":[";
        for(int k=0;k<4;++k)std::cout<<(k?",":"")<<cosine[k];std::cout<<"]}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
