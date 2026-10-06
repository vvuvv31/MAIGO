#include "carbon/electron_rate_bound.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
struct Piece {float low,high;float value[4];};
template<class T>void read(std::ifstream& f,T* p,std::size_t n=1){if(!f.read(reinterpret_cast<char*>(p),n*sizeof(T)))throw std::runtime_error("rate fixture truncated");}
int main(int argc,char** argv){
    if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);
    char magic[8];read(f,magic,8);if(std::string(magic,8)!="ER26RT01")throw std::runtime_error("rate fixture schema");
    unsigned models;read(f,&models);double maximum=0;std::uint64_t pieces=0;bool bad_bound_rejected=false;
    for(unsigned m=0;m<models;++m){
        unsigned controls,intervals;float majorant;read(f,&controls);read(f,&intervals);read(f,&majorant);
        if(!controls||controls>65||!intervals||intervals>100000)throw std::runtime_error("rate fixture dimensions");
        std::vector<float> edges(intervals+1);read(f,edges.data(),edges.size());
        for(unsigned c=0;c<controls;++c){std::vector<Piece> rows(intervals);
            for(unsigned i=0;i<intervals;++i){rows[i].low=edges[i];rows[i].high=edges[i+1];read(f,rows[i].value,4);}
            maximum=std::max(maximum,carbon::electron_rate_bound::certify(rows,edges.front(),edges.back(),majorant));pieces+=intervals;
            if(!bad_bound_rejected){
                float bound=0;for(const auto& r:rows)bound=std::max(bound,carbon::electron_rate_bound::piece_upper(r));
                try{carbon::electron_rate_bound::certify(rows,edges.front(),edges.back(),bound*.5f);}
                catch(const std::runtime_error&){bad_bound_rejected=true;}
            }
        }
    }
    if(f.peek()!=std::char_traits<char>::eof()||!bad_bound_rejected)throw std::runtime_error("rate fixture tail/rejection");
    std::cout<<"{\"models\":"<<models<<",\"control_pieces\":"<<pieces<<",\"maximum_certificate_ratio\":"<<maximum
             <<",\"insufficient_majorant_rejected\":true,\"component_pass\":true}\n";
}
