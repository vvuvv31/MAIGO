#pragma once
#include "carbon/electron_charged_data.hpp"
#include <fstream>
#include <vector>

namespace carbon {
// Host-only pinned angular table loader. It never changes transport physics
// outside the supplied table domain and has no legacy response-bank dependency.
struct ElectronInverseCdfData {
    struct Node {unsigned offset{},count{};};
    std::vector<Node> nodes;
    std::vector<float> energies,log_energies,screening,quantiles;
    unsigned bins{},knots{};
    ElectronInverseCdfData(const std::filesystem::path& index,const std::string& index_sha,
                          const std::string& charged_sha,unsigned expected_nodes) {
        if(compute_file_sha256_hex(index)!=index_sha)throw std::runtime_error("inverse-CDF index SHA mismatch");
        std::ifstream in(index);std::string magic,source,file,sha;unsigned cells;
        in>>magic>>source>>cells>>knots>>bins>>file>>sha;
        if(!in||magic!="MAIEIC01"||source!=charged_sha||cells<expected_nodes*2||cells>100000||
           (bins!=16&&bins!=32&&bins!=64)||knots!=48*bins+1)throw std::runtime_error("inverse-CDF header/source");
        nodes.resize(expected_nodes);energies.resize(cells);log_energies.resize(cells);screening.resize(cells);
        unsigned previous=0;
        for(unsigned i=0;i<cells;++i){unsigned node;double energy,a;in>>node>>energy>>a;
            if(!in||node>=expected_nodes||(i&&node<previous)||(!i&&node!=0)||(i&&node>previous+1))
                throw std::runtime_error("inverse-CDF node order");
            if(!nodes[node].count)nodes[node].offset=i;
            if(!(energy>0)||energy>.651||!(a>0)||!std::isfinite(a))throw std::runtime_error("inverse-CDF parameters");
            energies[i]=float(energy);screening[i]=float(a);log_energies[i]=std::log(energies[i]);
            if(!std::isfinite(energies[i])||!std::isfinite(screening[i])||screening[i]<=0||
               (nodes[node].count&&(energies[i]<=energies[i-1]||log_energies[i]<=log_energies[i-1])))
                throw std::runtime_error("inverse-CDF energy order");
            ++nodes[node].count;previous=node;
        }
        std::string extra;if(in>>extra)throw std::runtime_error("inverse-CDF trailing metadata");
        for(const auto n:nodes)if(n.count<2||energies[n.offset]>.0001001f||energies[n.offset+n.count-1]<.65f)
            throw std::runtime_error("inverse-CDF incomplete node domain");
        const auto path=index.parent_path()/file;
        if(compute_file_sha256_hex(path)!=sha)throw std::runtime_error("inverse-CDF table SHA mismatch");
        const std::size_t size=std::size_t(cells)*knots;
        if(std::filesystem::file_size(path)!=size*sizeof(float))throw std::runtime_error("inverse-CDF table size");
        quantiles.resize(size);std::ifstream raw(path,std::ios::binary);
        raw.read(reinterpret_cast<char*>(quantiles.data()),size*sizeof(float));if(!raw)throw std::runtime_error("inverse-CDF read");
        for(unsigned cell=0;cell<cells;++cell){const auto* q=quantiles.data()+std::size_t(cell)*knots;
            if(q[0]!=0||q[knots-1]!=2)throw std::runtime_error("inverse-CDF support");
            for(unsigned j=0;j<knots;++j)if(!std::isfinite(q[j])||(j&&q[j]<q[j-1]))throw std::runtime_error("inverse-CDF quantiles");
        }
    }
};
} // namespace carbon
