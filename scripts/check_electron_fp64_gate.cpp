#include "carbon/electron_fp32.hpp"
#include "carbon/electron_charged_data.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: charged index");
    const std::filesystem::path path=argv[1];
    std::ifstream input(path);std::string magic;unsigned nodes;input>>magic>>nodes;if(magic!="MAIECB02")throw std::runtime_error("index");
    namespace f=carbon::electron_fp32;
    std::uint64_t samples=0,invalid=0,baseline_gate_mismatches=0,gate_mismatches=0;
    double maximum_baseline_error=0,maximum_gate_error=0;unsigned cases=0;
    for(unsigned node=0;node<nodes;++node){
        int section;double density;std::string file;input>>section>>density>>file;
        carbon::ChargedResponseData data(path.parent_path()/file);const auto native=data.view();
        const auto u=native.urban;const f::ElectronUrbanMaterial p{float(u.mass),float(u.excitation),float(u.e0),float(u.e1),float(u.e2),float(u.f1),float(u.f2)};
        // Reference uses the same represented FP32 material/energy inputs,
        // but exact FP64 product and original double Urban equations. This
        // isolates arithmetic; it is not a native-table precision validation.
        const carbon::ElectronUrbanMaterial ref{p.mass,p.excitation,p.e0,p.e1,p.e2,p.f1,p.f2};
        for(float energy:{.001f,.002f,.01f,.1f,.4f}){
            const float dedx=float(carbon::charged_parameters(native,double(energy)).dedx);
            const float center=1e-5f/dedx;
            const float lengths[]={center,std::nextafter(center,0.f),std::nextafter(center,INFINITY),center*.999f,center*1.001f};
            const float cut=std::min(float(native.electron_cut),energy<=.1f?energy:.5f*energy);
            for(float length:lengths){++cases;const double mean64=double(length)*double(dedx);const float mean=length*dedx;
                const bool expected=mean64<1e-5||double(cut)<=double(p.e0);
                baseline_gate_mismatches+=expected!=(mean<1e-5f||cut<=p.e0);
                double sums[3]{};
                for(unsigned j=0;j<4096;++j){
                    std::uint64_t a=0x82abcdef331ULL+node*0x9e3779b97f4a7c15ULL+j*747796405ULL,b=a,c=a;
                    auto ua=[&](){return float(f::random_draw(a));};auto ub=[&](){return f::random_draw(b);};auto uc=[&](){return f::random_draw(c);};
                    const auto x=carbon::sample_electron_urban_loss(ref,double(energy),mean64,double(cut),ua);
                    const auto y=f::sample_electron_urban_loss(p,energy,mean,cut,ub);
                    const auto z=f::sample_electron_urban_loss_precise_gate(p,energy,mean,cut,uc,mean64);
                    invalid+=!x.valid||!y.valid||!z.valid;gate_mismatches+=expected!=(z.random_draws==0);
                    sums[0]+=x.loss;sums[1]+=y.loss;sums[2]+=z.loss;++samples;
                }
                if(sums[0]>0){maximum_baseline_error=std::max(maximum_baseline_error,std::abs(sums[1]/sums[0]-1));maximum_gate_error=std::max(maximum_gate_error,std::abs(sums[2]/sums[0]-1));}
            }
        }
    }
    const bool pass=!invalid&&!gate_mismatches&&maximum_gate_error<.01;
    std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"nodes\":"<<nodes<<",\"cells\":"<<cases<<",\"samples\":"<<samples<<",\"invalid\":"<<invalid<<",\"baseline_gate_mismatch_cells\":"<<baseline_gate_mismatches<<",\"candidate_gate_mismatch_samples\":"<<gate_mismatches<<",\"max_baseline_cell_mean_relative_error\":"<<maximum_baseline_error<<",\"max_candidate_cell_mean_relative_error\":"<<maximum_gate_error<<"}\n";
    return pass?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
