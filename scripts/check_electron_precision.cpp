#include "carbon/electron_charged_data.hpp"
#include "carbon/electron_elastic_fp32.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: charged-index sha256");
    const std::filesystem::path index=argv[1];
    if(carbon::compute_file_sha256_hex(index)!=argv[2])throw std::runtime_error("Index SHA mismatch");
    std::ifstream in(index);std::string magic;unsigned nodes;in>>magic>>nodes;
    if(magic!="MAIECB01")throw std::runtime_error("Index format");
    std::uint64_t samples=0,invalid=0,selection_changes=0;
    double sum_reference=0,sum_candidate=0,max_norm_error=0,max_rotation_error=0,max_cell_relative=0;
    for(unsigned node=0;node<nodes;++node){
        int section;double density;std::string file,pin;in>>section>>density>>file>>pin;
        carbon::ChargedResponseData data(index.parent_path()/file,pin);const auto view=data.view().elastic;
        for(double energy:{.00010001,.0003,.001,.003,.01,.03,.1,.3,.6499999}){
            double cell_reference=0,cell_candidate=0;
            for(unsigned j=0;j<4096;++j){
                const auto seed=0x982368764abcdULL+node*0x9e3779b97f4a7c15ULL+j*747796405ULL;
                auto a=seed,b=seed;
                auto ua=[&](){a=a*6364136223846793005ULL+1442695040888963407ULL;return (double(a>>12)+.5)*0x1p-52;};
                auto ub=[&](){b=b*6364136223846793005ULL+1442695040888963407ULL;return (double(b>>12)+.5)*0x1p-52;};
                const auto ref=carbon::sample_electron_elastic(view,energy,ua);
                const auto got=carbon::sample_electron_elastic_fp32(view,energy,ub);
                ++samples;if(!ref.valid||!got.valid){++invalid;continue;}
                selection_changes+=(a!=b||ref.energy_index!=got.energy_index);
                cell_reference+=1-ref.cosine;cell_candidate+=got.one_minus_cosine;
                const auto d=electron_joint_research::unit({.123,.456,.789});
                const double spin=ua(),delta=got.one_minus_cosine,cosine=1-delta;
                const auto frame=electron_joint_research::azimuth(d,6.2831853071795864769*spin);
                const auto expected=std::sqrt(delta*(2-delta))*frame.u+cosine*d;
                const auto direction=carbon::electron_rotate_fp32({d.x,d.y,d.z},cosine,delta,spin);
                const electron_joint_research::V v{direction[0],direction[1],direction[2]};
                max_norm_error=std::max(max_norm_error,std::abs(electron_joint_research::dot(v,v)-1));
                const auto error=v-expected;max_rotation_error=std::max(max_rotation_error,std::sqrt(electron_joint_research::dot(error,error)));
            }
            sum_reference+=cell_reference;sum_candidate+=cell_candidate;
            max_cell_relative=std::max(max_cell_relative,std::abs(cell_candidate/cell_reference-1));
        }
    }
    const double relative=sum_candidate/sum_reference-1;
    const bool pass=!invalid && std::abs(relative)<1e-4 && max_cell_relative<.005 &&
        selection_changes<samples/10000 && max_norm_error<1e-12 && max_rotation_error<2e-6;
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"samples\":"<<samples
        <<",\"invalid\":"<<invalid<<",\"rejection_or_energy_selection_changes\":"<<selection_changes
        <<",\"mean_one_minus_cosine_relative_change\":"<<relative<<",\"max_cell_relative_change\":"<<max_cell_relative
        <<",\"maximum_direction_norm_error\":"<<max_norm_error<<",\"maximum_rotation_vector_error\":"<<max_rotation_error<<"}\n";
    return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
