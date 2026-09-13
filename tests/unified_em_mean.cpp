#include "carbon/unified_em_view.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
    if(argc!=4)throw std::runtime_error("usage: unified_em_mean package sha export-root");
    auto p=carbon::UnifiedEmPackage::load(argv[1],argv[2]);
    std::uint64_t queries=0,failed=0;float maximum=0;int previous=-2,density_index=0;
    for(std::size_t m=0;m<p.materials.size();++m){
        int section=p.materials[m].section;
        if(section!=previous)density_index=0;else ++density_index;previous=section;
        std::ostringstream name;if(section<0)name<<"JEM_W";else name<<"JEM_S"<<std::setw(2)<<std::setfill('0')<<section<<"D"<<density_index;
        for(int s=0;s<18;++s){
            const auto& r=p.records[m*18+s];carbon::UnifiedEmPoint point{&r,p.nodes.data()+r.node_offset,p.segments.data(),1};
            auto path=std::filesystem::path(argv[3])/("ion_"+std::to_string(r.z)+"_"+std::to_string(r.a))/"mean_probe_v1"/name.str()/"mean_probe.csv";
            std::ifstream in(path);if(!in)throw std::runtime_error("Missing probe "+path.string());std::string line;std::getline(in,line);
            while(std::getline(in,line)){
                std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);float e,h,reference,native,range;row>>e>>h>>reference>>native>>range;
                float value=point.mean(e*r.a,h);if(e*r.a-value<=r.lowest_kinetic)value=e*r.a;float error=std::abs(value-reference)/std::max(reference,1e-5f);
                if(error>maximum){maximum=error;if(error>.002)std::cerr<<name.str()<<" Z="<<r.z<<" A="<<r.a<<" E="<<e<<" h="<<h<<" ref="<<reference<<" got="<<value<<" relative="<<error<<"\n";}
                if(error>.002 && std::abs(value-reference)>1e-5)++failed;
                ++queries;
            }
        }
    }
    std::cerr<<"queries="<<queries<<" maximum_relative="<<maximum<<" failures="<<failed<<"\n";
    return failed?1:0;
}
