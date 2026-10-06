#include "carbon/electron_response26.hpp"
#include "carbon/unified_em_view.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    const auto p=carbon::UnifiedEmPackage::load(argv[1]);
    if(p.materials.size()!=26||p.continuous_descriptors.size()!=26*18)
        throw std::runtime_error("require continuous26 ion package");
    carbon::UnifiedEmDevice v;
    v.materials=p.materials.data();v.species=p.species.data();v.records=p.records.data();
    v.nodes=p.nodes.data();v.segments=p.segments.data();v.material_count=p.materials.size();
    v.continuous_descriptors=p.continuous_descriptors.data();v.continuous_scalars=p.continuous_scalars.data();
    v.continuous_ranges=p.continuous_ranges.data();v.continuous_degree=p.continuous_degree;
    v.continuous_scalar_count=p.continuous_scalar_count;v.continuous_lengths=p.continuous_lengths.data();
    unsigned cases=0,zero_yield=0;double error=0.;
    for(unsigned m=0;m<26;++m)for(unsigned a:{1u,12u}){
        const auto species=v.species_index(a==1?1:6,a);
        const auto& descriptor=p.continuous_descriptors[m*18+species];
        for(unsigned d=0;d<16;++d){
            const float x=(float(d)+.5f)/16.f;
            const float rho=std::exp(descriptor.log_min+x*(descriptor.log_max-descriptor.log_min));
            const auto state=v.select(p.materials[m].section,rho,species);
            if(!state.valid)throw std::runtime_error("26 composition selection");
            for(unsigned e=0;e<128;++e){
                const float kinetic=a*std::exp(std::log(1.f)+(float(e)+.5f)/128.f*std::log(300.f));
                const auto prepared=state.prepare(kinetic);
                const float path=std::min(.1f,.01f*prepared.lo.range);
                const float restricted=state.mean(kinetic,path);
                const float delta0=prepared.lo.delta_stopping;
                const float delta=std::max(0.f,delta0-.5f*(restricted+delta0*path)*prepared.lo.delta_slope)*path;
                const float f=carbon::electron_response26::birth_fraction_from_em(restricted,delta);
                if(!std::isfinite(f)||f<0.f||f>1.f)throw std::runtime_error("birth fraction domain");
                const float expected=restricted+delta;
                const auto split=carbon::electron_response26::partition(expected*1.37f,f);
                if(!split.valid)throw std::runtime_error("sampled loss partition");
                const float loss=expected*1.37f;
                if(loss>0)error=std::max(error,std::abs(double(split.local)+split.electron-loss)/loss);
                ++cases;zero_yield+=f==0.f;
            }
        }
    }
    if(error>2e-7)throw std::runtime_error("energy partition");
    std::cout<<"{\"composition_count\":26,\"cases\":"<<cases<<",\"zero_yield_cases\":"<<zero_yield
             <<",\"maximum_partition_relative\":"<<error<<",\"component_pass\":true}\n";
}
