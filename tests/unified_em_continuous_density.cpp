#include "carbon/unified_em_view.hpp"
#include "carbon/delta_moments_data.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace carbon;
struct Error {double mean{},range{},stopping{},delta{},variance{},delta_absolute{},variance_absolute{};unsigned long long cases{};};
int main(int argc,char** argv) {
    if(argc!=5)throw std::runtime_error("usage: reference_em candidate_em reference_delta candidate_delta");
    auto a=UnifiedEmPackage::load(argv[1]),b=UnifiedEmPackage::load(argv[2]);
    auto da=load_delta_moments(argv[3],a.nodes.size()),db=load_delta_moments(argv[4],b.nodes.size());
    if(b.materials.size()!=26 || b.records.size()!=468)throw std::runtime_error("Continuous model must have26 compositions");
    UnifiedEmDevice va{},vb{};
    auto bind=[](auto& v,const auto& p,const auto& d){v.materials=p.materials.data();v.species=p.species.data();v.records=p.records.data();v.nodes=p.nodes.data();v.segments=p.segments.data();v.material_count=p.materials.size();v.delta_means=d.data();v.shared_prepare_grid=p.shared_prepare_grid;v.continuous_descriptors=p.continuous_descriptors.empty()?nullptr:p.continuous_descriptors.data();v.continuous_scalars=p.continuous_scalars.data();v.continuous_ranges=p.continuous_ranges.data();v.continuous_degree=p.continuous_degree;v.continuous_scalar_count=p.continuous_scalar_count;v.continuous_lengths=p.continuous_lengths.data();};
    bind(va,a,da);bind(vb,b,db);
    std::array<Error,4> errors{};constexpr std::array<float,4> minimums={0,.1f,1,10};
    auto rel=[](float x,float y){return std::abs(double(x)-y)/std::max(std::abs(double(x)),1e-20);};
    for(unsigned m=0;m<a.materials.size();++m)for(unsigned ion=0;ion<18;++ion) {

        const auto& r=a.records[m*18+ion];
        for(unsigned j=0;j<257;++j) {
            const float eu=std::exp(std::log(.0001f)+float(j)/256*(std::log(300.f)-std::log(.0001f)));
            const float kinetic=eu*r.a;
            const auto x=va.select(a.materials[m].section,a.materials[m].density,ion),y=vb.select(a.materials[m].section,a.materials[m].density,ion);
            if(!x.valid || !y.valid || !x.covers(kinetic) || !y.covers(kinetic))continue;
            const auto px=x.prepare(kinetic),py=y.prepare(kinetic);
            auto mixx=[&](float lo,float hi){return x.mix(lo,hi);};auto mixy=[&](float lo,float hi){return y.mix(lo,hi);};
            for(unsigned group=0;group<4;++group)if(eu>=minimums[group]) {
                auto& e=errors[group];
                e.range=std::max(e.range,rel(x.range(kinetic),y.range(kinetic)));
                e.stopping=std::max(e.stopping,rel(mixx(px.lo.stopping,px.hi.stopping),mixy(py.lo.stopping,py.hi.stopping)));
                const auto dx=mixx(px.lo.delta_stopping,px.hi.delta_stopping),dy=mixy(py.lo.delta_stopping,py.hi.delta_stopping);
                const auto vx=mixx(px.lo.delta_variance,px.hi.delta_variance),vy=mixy(py.lo.delta_variance,py.hi.delta_variance);
                e.delta_absolute=std::max(e.delta_absolute,std::abs(double(dx)-dy));
                e.variance_absolute=std::max(e.variance_absolute,std::abs(double(vx)-vy));
                if(group==3 && dx>1e-7f && rel(dx,dy)>.01)std::cerr<<"delta outlier m="<<m<<" Z="<<r.z<<" A="<<r.a<<" Eu="<<eu<<" cut="<<y.cut()<<" a="<<dx<<" b="<<dy<<" relative="<<rel(dx,dy)<<'\n';
                e.delta=std::max(e.delta,rel(mixx(px.lo.delta_stopping,px.hi.delta_stopping),mixy(py.lo.delta_stopping,py.hi.delta_stopping)));
                e.variance=std::max(e.variance,rel(mixx(px.lo.delta_variance,px.hi.delta_variance),mixy(py.lo.delta_variance,py.hi.delta_variance)));
                for(float cap:{.001f,.1f,.5f}) {
                    const float h=std::min(cap,.5f*std::min(x.range(kinetic),y.range(kinetic)));
                    if(!(h>0))continue;
                    const float mx=x.mean(kinetic,h,px),my=y.mean(kinetic,h,py);
                    if(!(std::isfinite(mx)&&std::isfinite(my)&&mx>=0&&my>=0))throw std::runtime_error("Invalid finite-step loss");
                    const double change=rel(mx,my);
                    if(group==2 && change>e.mean && change>.01)std::cerr<<"mean outlier material="<<m<<" Z="<<r.z<<" A="<<r.a<<" energy_MeVu="<<eu<<" step="<<h<<" reference="<<mx<<" candidate="<<my<<" relative="<<change<<'\n';
                    e.mean=std::max(e.mean,change);++e.cases;
                }
            }
        }
    }
    std::cout<<"{\"candidate_valid\":true,\"shared_grid\":"<<(b.shared_prepare_grid?"true":"false")<<",\"bands\":[";
    for(unsigned i=0;i<4;++i){auto& e=errors[i];if(i)std::cout<<',';std::cout<<"{\"minimum_MeVu\":"<<minimums[i]<<",\"maximum_MeVu\":300,\"cases\":"<<e.cases<<",\"max_mean_relative\":"<<e.mean<<",\"max_range_relative\":"<<e.range<<",\"max_stopping_relative\":"<<e.stopping<<",\"max_delta_relative\":"<<e.delta<<",\"max_delta_variance_relative\":"<<e.variance<<",\"max_delta_absolute_MeV_mm\":"<<e.delta_absolute<<",\"max_variance_absolute_MeV2_mm\":"<<e.variance_absolute<<'}';}
    std::cout<<"]}\n";
}
