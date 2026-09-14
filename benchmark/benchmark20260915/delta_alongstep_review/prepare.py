from pathlib import Path
import shutil,difflib,hashlib,json
R=Path(__file__).resolve().parent;repo=R.parents[2];src=repo/'scratch/delta_alongstep_current_20260915/source';shutil.copytree(repo/'scratch/delta_moments_current_20260915/source',src)
p=src/'include/carbon/primary_em_mean_candidate.hpp';s=p.read_text();needle='    Real value(Real x) const {';assert s.count(needle)==1;s=s.replace(needle,'''    Real derivative(Real x) const {
        if(x<lower || x>upper)return 0;
        const Real w=(x-lower)/(upper-lower),a=w-Real(1.0/3),b=w-Real(2.0/3),c=w-1;
        return (-Real(4.5)*y0*(a*b+a*c+b*c)
            +Real(13.5)*ythird*(b*c+w*(b+c))
            -Real(13.5)*ytwothirds*(a*c+w*(a+c))
            +Real(4.5)*y1*(a*b+w*(a+b)))/(upper-lower);
    }
'''+needle);p.write_text(s)
p=src/'include/carbon/unified_em_view.hpp';s=p.read_text().replace('float delta_stopping{},delta_variance{};','float delta_stopping{},delta_variance{},stopping_slope{},delta_slope{};')
s=s.replace('float* delta_variance=nullptr)const','float* delta_variance=nullptr,float* delta_slope=nullptr)const')
a='        if(delta_variance)*delta_variance=delta_means?mix(delta_means[2*lo+1],delta_means[2*lo+3]):0.f;';s=s.replace(a,a+'\n        if(delta_slope)*delta_slope=delta_means?(delta_means[2*lo+2]-delta_means[2*lo])/(b.energy-a.energy)/record->a:0.f;')
s=s.replace('float raw(int kind,float x)const {','float raw(int kind,float x,float* slope=nullptr)const {')
s=s.replace('            if(kind==3)return 0;','            if(kind==3){if(slope)*slope=0;return 0;}').replace('            return v[0].y0*(kind==2?w*w:std::sqrt(w));','''            if(slope)*slope=x>0?v[0].y0*(kind==2?2*w/v[0].lower:1/(2*std::sqrt(w)*v[0].lower)):0.f;
            return v[0].y0*(kind==2?w*w:std::sqrt(w));''')
s=s.replace('        return v[lo].value(x);','        if(slope)*slope=v[lo].derivative(x);\n        return v[lo].value(x);')
s=s.replace('float delta=0,variance=0;const auto n=at(kinetic/r.a,&delta,&variance);','float delta=0,variance=0,delta_slope=0,raw_slope=0;const auto n=at(kinetic/r.a,&delta,&variance,&delta_slope);')
s=s.replace('        return {f,raw(1,kinetic*r.ratio)/(f*r.ratio),f*raw(0,kinetic*r.ratio),','        const float stop=f*raw(0,kinetic*r.ratio,&raw_slope);\n        return {f,raw(1,kinetic*r.ratio)/(f*r.ratio),stop,')
s=s.replace('n.fluctuation==1,delta*density_scale,variance*density_scale};','n.fluctuation==1,delta*density_scale,variance*density_scale,f*r.ratio*raw_slope,delta_slope*density_scale};')
s=s.replace('        return primary_restricted_mean_candidate(kinetic,length,pre.stopping,pre.range,','''        // Correct only the native linear restricted branch. Range inversion
        // already integrates energy dependence and must not be corrected twice.
        float stopping=pre.stopping;
        if(length<pre.range && stopping*length<=r.linear_limit*kinetic) {
            const float corrected=stopping-.5f*length*(stopping+pre.delta_stopping)*pre.stopping_slope;
            if(corrected>0 && corrected*length<=r.linear_limit*kinetic)stopping=corrected;
        }
        return primary_restricted_mean_candidate(kinetic,length,stopping,pre.range,''')
s=s.replace('                const auto m=at(mid/r.a);','                mid=std::max(.5f*kinetic,mid-.5f*pre.delta_stopping*h);\n                const auto m=at(mid/r.a);')
s=s.replace('    const float delta_mean=s.mix(pre.lo.delta_stopping,pre.hi.delta_stopping)*h;','''    const float delta0=s.mix(pre.lo.delta_stopping,pre.hi.delta_stopping);
    const float delta_slope=s.mix(pre.lo.delta_slope,pre.hi.delta_slope);
    const float delta_mean=std::max(0.f,delta0-.5f*(mean+delta0*h)*delta_slope)*h;''');p.write_text(s)
p=src/'src/run_quality.cpp';s=p.read_text().replace('delta_moments_candidate','delta_alongstep_candidate').replace('1% combined mean-loss step cap','1% combined mean-loss step cap; second-order restricted linear-branch and delta mean correction, original delta variance');p.write_text(s)
patch=[];hashes={}
for n in ['include/carbon/primary_em_mean_candidate.hpp','include/carbon/unified_em_view.hpp','src/run_quality.cpp']:
 a=(repo/'scratch/delta_moments_current_20260915/source'/n).read_text();b=(src/n).read_text();patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+n,tofile='b/'+n));hashes[n]=hashlib.sha256(b.encode()).hexdigest()
(R/'candidate_vs_moments.patch').write_text(''.join(patch));(R/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
old=repo/'benchmark/benchmark20260915/delta_moments_review'
for n in ['prepare_benchmarks.py','run_split_shard.py']:
 s=(old/n).read_text().replace('delta_moments_current_20260915','delta_alongstep_current_20260915').replace('delta_moments_current_shard01_split','delta_alongstep_current_shard01_split').replace('delta_moments_merged','delta_alongstep_merged').replace('GPU delta moments','GPU along-step').replace('delta_moments_candidate','delta_alongstep_candidate');(R/n).write_text(s)
