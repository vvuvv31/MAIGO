#include "carbon/unified_em_view.hpp"
#include "carbon/delta_moments_data.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <chrono>
using namespace carbon;
int main(int argc,char** argv) {
    if(argc!=5)throw std::runtime_error("usage: reference_em candidate_em reference_delta candidate_delta");
    const auto start=std::chrono::steady_clock::now();
    auto a=UnifiedEmPackage::load(argv[1]);auto b=UnifiedEmPackage::load(argv[2]);
    auto da=load_delta_moments(argv[3],a.nodes.size());auto db=load_delta_moments(argv[4],b.nodes.size());
    if(a.materials.size()!=b.materials.size() || a.records.size()!=b.records.size() || a.nodes.size()!=b.nodes.size() ||
       a.segments.size()!=b.segments.size() || !b.shared_prepare_grid ||
       std::memcmp(a.materials.data(),b.materials.data(),a.materials.size()*sizeof(UnifiedEmMaterial)) ||
       std::memcmp(a.records.data(),b.records.data(),a.records.size()*sizeof(UnifiedEmRecord)) ||
       std::memcmp(a.segments.data(),b.segments.data(),a.segments.size()*sizeof(a.segments[0])))
        throw std::runtime_error("Registry/native spline mismatch");
    double node_error=0,delta_error=0,mean_error=0,range_error=0;std::uint64_t mean_cases=0;
    auto relative=[](float x,float y){return std::abs(double(x)-y)/std::max(std::abs(double(x)),1e-30);};
    for(std::size_t i=0;i<a.nodes.size();++i) {
        const auto* x=reinterpret_cast<const float*>(&a.nodes[i]);const auto* y=reinterpret_cast<const float*>(&b.nodes[i]);
        for(unsigned k=0;k<13;++k) {
            if(k==0 || k>=11 || x[k]==0) {if(x[k]!=y[k])throw std::runtime_error("Energy/discrete/zero support changed");}
            else {
                const auto err=relative(x[k],y[k]);
                if(err>node_error && err>5.1e-5)std::cerr<<"node="<<i<<" field="<<k<<" energy="<<x[0]<<" native="<<x[k]<<" decoded="<<y[k]<<" err="<<err<<"\n";
                node_error=std::max(node_error,err);
            }
        }
    }
    for(std::size_t i=0;i<da.size();++i) {
        if(da[i]==0 && db[i]!=0)throw std::runtime_error("Delta zero support changed");
        delta_error=std::max(delta_error,relative(da[i],db[i]));
    }
    UnifiedEmDevice va{},vb{};
    auto bind=[](UnifiedEmDevice& v,const UnifiedEmPackage& p,const std::vector<float>& d) {
        v.materials=p.materials.data();v.species=p.species.data();v.records=p.records.data();v.nodes=p.nodes.data();
        v.segments=p.segments.data();v.material_count=p.materials.size();v.delta_means=d.data();v.shared_prepare_grid=p.shared_prepare_grid;
    };
    bind(va,a,da);bind(vb,b,db);
    for(unsigned m=0;m<a.materials.size();++m)for(unsigned ion=0;ion<18;++ion) {
        const auto& rec=a.records[m*18+ion];const auto* nodes=a.nodes.data()+rec.node_offset;
        for(unsigned sample=0;sample<129;++sample) {
            // Knot and inter-knot queries span the full stored species energy grid.
            const double index=(rec.node_count-1)*double(sample)/128;
            unsigned lo=std::min(unsigned(index),rec.node_count-2);
            const float energy=float((nodes[lo].energy+(index-lo)*(nodes[lo+1].energy-nodes[lo].energy))*rec.a);
            const auto x=va.select(a.materials[m].section,a.materials[m].density,ion);
            const auto y=vb.select(b.materials[m].section,b.materials[m].density,ion);
            if(!x.valid || !y.valid || !x.covers(energy) || !y.covers(energy))continue;
            const auto px=x.prepare(energy),py=y.prepare(energy);
            range_error=std::max(range_error,relative(x.range(energy),y.range(energy)));
            for(float length:{.001f,.1f,.5f}) {
                const float h=std::min(length,.5f*x.range(energy));if(!(h>0))continue;
                const float mx=x.mean(energy,h,px),my=y.mean(energy,h,py);
                if(!(std::isfinite(mx)&&std::isfinite(my)&&mx>=0&&my>=0))throw std::runtime_error("Invalid finite-step mean");
                mean_error=std::max(mean_error,relative(mx,my));++mean_cases;
            }
        }
    }
    std::cout<<"{\"materials\":"<<a.materials.size()<<",\"composition_bases\":26,\"ions\":18,\"node_values\":"<<a.nodes.size()*13
      <<",\"delta_values\":"<<da.size()<<",\"finite_step_cases\":"<<mean_cases<<",\"max_node_relative_error\":"<<node_error
      <<",\"max_delta_relative_error\":"<<delta_error<<",\"max_mean_relative_error\":"<<mean_error<<",\"max_range_relative_error\":"<<range_error
      <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    if(node_error>5.1e-5 || delta_error>5.1e-5 || mean_error>2e-4 || range_error>1e-4)return 1;
}
