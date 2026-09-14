from pathlib import Path
import json,shutil,difflib,hashlib
R=Path(__file__).resolve().parent;repo=R.parents[2];src=repo/'scratch/delta_moments_current_20260915/source';shutil.copytree(repo/'scratch/delta_mean_current_20260914/source',src)
m=json.loads((R/'moments_table_manifest.json').read_text())
h='''#pragma once
#include <cmath>
#include <algorithm>
#include <limits>
namespace carbon {
// Moment-matched aggregate delta loss. Compound-Poisson cumulants are
// mean=h*lambda*E[accepted epsilon], variance=h*lambda*E[accepted epsilon^2].
// Gamma approximates the distribution, including its zero-event atom/tail.
// Returned draw is uncapped; transport caps against remaining kinetic energy.
template<class Uniform> float delta_moment_draw(float mean,float variance,Uniform& uniform) {
    if(!(mean>=0 && variance>=0 && std::isfinite(mean) && std::isfinite(variance)))return std::numeric_limits<float>::quiet_NaN();
    if(mean==0 || variance==0)return mean;
    float shape=mean*(mean/variance),scale=variance/mean;
    if(!(shape>0 && std::isfinite(shape) && std::isfinite(scale)))return std::numeric_limits<float>::quiet_NaN();
    const bool small=shape<1;
    const float d=(small?shape+1:shape)-1.f/3.f,c=1/std::sqrt(9*d);
    auto u=[&](){float v=uniform();return (v>=0 && v<1)?std::max(v,1e-12f):std::numeric_limits<float>::quiet_NaN();};
    for(int tries=0;tries<256;++tries){
        const float x=std::sqrt(-2*std::log(u()))*std::cos(6.283185307179586f*u());
        float v=1+c*x;if(v<=0)continue;v=v*v*v;
        const float w=u();
        if(w<1-.0331f*x*x*x*x || std::log(w)<.5f*x*x+d*(1-v+std::log(v)))
            return scale*d*v*(small?std::pow(u(),1/shape):1.f);
    }
    return std::numeric_limits<float>::quiet_NaN();
}
}
'''
(src/'include/carbon/delta_moments_candidate.hpp').write_text(h)
p=src/'include/carbon/unified_em_view.hpp';s=p.read_text().replace('#pragma once','#pragma once\n#include "carbon/delta_moments_candidate.hpp"',1).replace('float delta_stopping{};','float delta_stopping{},delta_variance{};').replace('float* delta_mean=nullptr','float* delta_mean=nullptr,float* delta_variance=nullptr').replace('mix(delta_means[lo],delta_means[lo+1])','mix(delta_means[2*lo],delta_means[2*lo+2])')
a='        if(delta_mean)*delta_mean=delta_means?mix(delta_means[2*lo],delta_means[2*lo+2]):0.f;';s=s.replace(a,a+'\n        if(delta_variance)*delta_variance=delta_means?mix(delta_means[2*lo+1],delta_means[2*lo+3]):0.f;')
s=s.replace('float delta=0;const auto n=at(kinetic/r.a,&delta);','float delta=0,variance=0;const auto n=at(kinetic/r.a,&delta,&variance);').replace('n.fluctuation==1,delta*density_scale};','n.fluctuation==1,delta*density_scale,variance*density_scale};').replace('delta_means+r->node_offset','delta_means+2*r->node_offset')
s=s.replace('// Deterministic delta first-moment compensation; no delta RNG or clock.','// Aggregate delta draw; native restricted fluctuation remains independent.')
s=s.replace('    out.delta=std::min(t-loss,delta_mean);loss+=out.delta;','''    const float delta_variance=s.mix(pre.lo.delta_variance,pre.hi.delta_variance)*h;
    const float delta=fluctuations?delta_moment_draw(delta_mean,delta_variance,uniform):delta_mean;
    if(!(delta>=0 && std::isfinite(delta)))return out;
    out.delta=std::min(t-loss,delta);loss+=out.delta;''');p.write_text(s)
p=src/'src/transport_sycl.cpp';s=p.read_text().replace('/scratch/delta_mean_current_20260914/delta_mean.bin','/scratch/delta_moments_current_20260915/delta_moments.bin').replace('1035d2ad5cbae56a247c430abfac2a075ba7add11307d3d4ec5700509861a4ad',m['sha256']).replace('EMDMEAN1','EMDMOMT2').replace('delta_header[0]!=1','delta_header[0]!=2').replace('16+4*package.nodes.size()','16+8*package.nodes.size()').replace('delta_means(package.nodes.size())','delta_means(2*package.nodes.size())').replace('[delta-mean] deterministic first moment, no delta clock','[delta-moments] aggregate gamma mean/variance, no delta clock');p.write_text(s)
p=src/'src/run_quality.cpp';s=p.read_text().replace('delta_mean_only_candidate','delta_moments_candidate').replace('no delta clock/RNG; deterministic first-moment delta compensation','no delta clock; aggregate gamma delta compensation with first and second moments').replace('delta variance omitted','delta spectrum shape and zero-event atom approximated, kinetic cap may alter moments');p.write_text(s)
patch=[];hashes={}
for n in ['include/carbon/unified_em_view.hpp','include/carbon/delta_moments_candidate.hpp','src/transport_sycl.cpp','src/run_quality.cpp']:
 a=(repo/n).read_text() if (repo/n).exists() else '';b=(src/n).read_text();patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+n,tofile='b/'+n));hashes[n]={'base':hashlib.sha256(a.encode()).hexdigest(),'candidate':hashlib.sha256(b.encode()).hexdigest()}
(R/'candidate.patch').write_text(''.join(patch));(R/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
