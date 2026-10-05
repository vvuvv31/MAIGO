#include "carbon/unified_em_view.hpp"
#include "carbon/delta_moments_data.hpp"
#include <iostream>
#include <stdexcept>
using namespace carbon;
int main(int argc,char** argv) {
    if(argc!=3)throw std::runtime_error("usage: continuous_em delta");
    auto p=UnifiedEmPackage::load(argv[1]);auto delta=load_delta_moments(argv[2],p.nodes.size());
    if(p.materials.size()!=26 || p.records.size()!=468 || p.continuous_descriptors.size()!=468)throw std::runtime_error("Requires26-composition runtime");
    UnifiedEmDevice full{};full.materials=p.materials.data();full.species=p.species.data();full.records=p.records.data();full.nodes=p.nodes.data();full.segments=p.segments.data();full.material_count=26;full.delta_means=delta.data();
    full.continuous_descriptors=p.continuous_descriptors.data();full.continuous_scalars=p.continuous_scalars.data();full.continuous_ranges=p.continuous_ranges.data();full.continuous_degree=p.continuous_degree;full.continuous_scalar_count=p.continuous_scalar_count;
    auto fast=full;fast.continuous_lengths=p.continuous_lengths.data();
    double logrange=0,rate=0,mean=0;unsigned long long cases=0;
    auto rel=[](float a,float b){return std::abs(double(a)-b)/std::max(std::abs(double(a)),1e-15);};
    for(unsigned m=0;m<26;++m)for(unsigned i=0;i<18;++i) {
        const auto& d=p.continuous_descriptors[m*18+i];const auto& r=p.records[m*18+i];
        for(unsigned q=0;q<17;++q) {
            const float coordinate=float(q)/16;
            const float rho=std::exp(d.log_min+coordinate*(d.log_max-d.log_min));
            auto a=full.select(m-1,rho,i),b=fast.select(m-1,rho,i);
            if(!a.valid || !b.valid)throw std::runtime_error("Invalid continuous density selection");
            for(unsigned j=0;j<97;++j) {
                const float emin=p.nodes[r.node_offset].energy,emax=p.nodes[r.node_offset+r.node_count-1].energy;
                const float t=std::exp(std::log(emin)+float(j)/96*std::log(emax/emin))*r.a;
                if(!a.covers(t) || !b.covers(t))continue;
                const auto x=a.lo(),y=b.lo();const auto px=a.prepare(t),py=b.prepare(t);
                const float e=t*r.ratio;
                logrange=std::max(logrange,std::abs(double(x.range_log_ratio(e))-y.range_log_ratio(e)));
                const float u=.001f+float(j)/96*.999f;const unsigned stride=p.continuous_degree+1;
                const unsigned k=std::min(static_cast<unsigned>(u*(d.rate_count-1)),d.rate_count-1);
                const unsigned row=d.rate_offset+k;
                const float* c=p.continuous_ranges.data()+row*stride+1;
                rate=std::max(rate,std::abs(double(em_chebyshev(c,p.continuous_degree,2*coordinate-1))-em_chebyshev(c,p.continuous_lengths[row],2*coordinate-1)));
                for(float v:{py.lo.range,py.lo.stopping,py.lo.delta_stopping,py.lo.delta_variance,py.lo.dispersion})if(!(std::isfinite(v) && v>=0))throw std::runtime_error("Invalid continuous response");
                for(float cap:{.001f,.1f,.5f}) {
                    const float h=std::min(cap,.5f*std::min(px.lo.range,py.lo.range));
                    const float ma=a.mean(t,h,px),mb=b.mean(t,h,py);
                    if(!(std::isfinite(ma)&&std::isfinite(mb)&&ma>=0&&mb>=0&&ma<=t&&mb<=t))throw std::runtime_error("Invalid finite-step energy accounting");
                    mean=std::max(mean,rel(ma,mb));++cases;
                }
            }
        }
    }
    if(logrange>6e-7 || rate>6e-5 || mean>1e-3)throw std::runtime_error("Truncation exceeded reserved FP32 error bound");
    std::cout<<"{\"passed\":true,\"compositions\":26,\"density_response_states\":0,\"finite_step_cases\":"<<cases<<",\"max_logrange_difference\":"<<logrange<<",\"max_rate_factor_difference\":"<<rate<<",\"max_mean_relative_difference\":"<<mean<<",\"GPU_only_branch_emulated\":"<<(unified_em_continuous_device_only?"true":"false")<<"}\n";
}
