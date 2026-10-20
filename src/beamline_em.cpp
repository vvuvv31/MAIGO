#include "beamline_em.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
namespace carbon {
BeamlineEmPackage BeamlineEmPackage::load(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);char magic[8]{};std::array<std::uint32_t,4> h{};
    in.read(magic,8);in.read(reinterpret_cast<char*>(h.data()),sizeof(h));
    const auto [version,nm,ns,nn]=h;
    const std::uint64_t expected=24ULL+nm*sizeof(BeamlineEmMaterial)+ns*sizeof(BeamlineEmSpecies)+
        std::uint64_t(nm)*ns*sizeof(BeamlineEmRecord)+std::uint64_t(nn)*sizeof(BeamlineEmNode);
    if(!in || std::memcmp(magic,"BLEM0001",8) || version!=1 || nm!=2 || !ns || ns>512 ||
       nn<2 || expected!=std::filesystem::file_size(path) || expected>1024ULL*1024*1024)
        throw std::runtime_error("Invalid full beamline electronic response header");
    BeamlineEmPackage p;p.materials.resize(nm);p.species.resize(ns);p.records.resize(nm*ns);p.nodes.resize(nn);p.ranges.resize(nn);
    auto read=[&](auto&v){in.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(v[0]));};
    read(p.materials);read(p.species);read(p.records);read(p.nodes);
    if(!in)throw std::runtime_error("Truncated full beamline electronic response");
    for(unsigned i=0;i<nm;++i){const auto m=p.materials[i];
        const float rho=i==0?8.96f:.00120479f;
        if(std::abs(m.density-rho)>rho*1e-6f || !(m.excitation_MeV>0&&m.e0_MeV>0) ||
           !std::isfinite(m.density+m.excitation_MeV+m.e0_MeV))
            throw std::runtime_error("Full beamline response requires native G4_Cu/G4_AIR");}
    std::set<std::pair<unsigned,unsigned>> seen;
    for(const auto s:p.species)
        if(!s.z || s.a<s.z || !(s.mass_MeV>0) || !std::isfinite(s.mass_MeV) || !seen.emplace(s.z,s.a).second)
            throw std::runtime_error("Invalid full beamline projectile");
    std::uint64_t offset=0;
    for(const auto r:p.records){
        if(r.offset!=offset || r.count<2 || offset+r.count>nn)
            throw std::runtime_error("Invalid full beamline response bounds");
        double range=0;
        for(unsigned i=0;i<r.count;++i){const auto n=p.nodes[r.offset+i];
            if(!(n.energy>0&&n.stopping>0&&n.ion_variance>=0&&n.universal_variance>=0) ||
               !std::isfinite(n.energy+n.stopping+n.ion_variance+n.universal_variance))
                throw std::runtime_error("Invalid full beamline response node");
            if(!i)range=2.*n.energy/n.stopping; // native low-energy sqrt(E) continuation
            else{const auto l=p.nodes[r.offset+i-1];
                if(!(n.energy>l.energy))throw std::runtime_error("Unordered full beamline energies");
                range+=linear_stopping_path(double(n.energy)-l.energy,double(l.stopping),double(n.stopping));}
            p.ranges[r.offset+i]=range;
            if(!(p.ranges[r.offset+i]>0) || !std::isfinite(p.ranges[r.offset+i]) ||
               (i&&p.ranges[r.offset+i]<=p.ranges[r.offset+i-1]))
                throw std::runtime_error("Non-monotone full electronic CSDA range at record offset="+std::to_string(r.offset)+" node="+std::to_string(i)+" E="+std::to_string(n.energy));
        }
        offset+=r.count;
    }
    if(offset!=nn)throw std::runtime_error("Unclaimed full beamline nodes");
    return p;
}
}
