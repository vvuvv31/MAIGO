#include "carbon/unified_em_view.hpp"
#include <fstream>
#include <stdexcept>
using namespace carbon;
struct Probe {unsigned record;float cut,kinetic,first,second;};
static_assert(sizeof(Probe)==20);
int main(int argc,char** argv) {
    if(argc!=3)throw std::runtime_error("usage: continuous_em output");
    auto p=UnifiedEmPackage::load(argv[1]);std::ofstream output(argv[2],std::ios::binary);
    if(!output)throw std::runtime_error("Cannot create moment probes");
    for(unsigned i=0;i<p.records.size();++i) {
        const auto& r=p.records[i];if(r.spin!=0)continue;
        for(float cut:{.00099f,.01f,.06f,.12f})for(unsigned j=0;j<193;++j) {
            const float eu=std::exp(std::log(.01f)+float(j)/192*std::log((6000.f/r.a)/.01f));
            const float t=eu*r.a,u=t/r.mass,re=.51099891f/r.mass;
            const float tm=2*.51099891f*u*(u+2)/(1+2*(u+1)*re+re*re),ff=r.form_factor*tm;
            if(!(ff>1e-6f && ff<=.001f && tm>cut*1.01f))continue;
            UnifiedEmPoint point{};point.record=&r;point.material.cut=cut;
            Probe q{i,cut,t};point.continuous_moments(t,1.f,q.first,q.second);
            if(!(q.first>=0 && q.second>=0 && std::isfinite(q.first) && std::isfinite(q.second)))throw std::runtime_error("Invalid native-spectrum moment");
            output.write(reinterpret_cast<char*>(&q),sizeof q);
        }
    }
}
