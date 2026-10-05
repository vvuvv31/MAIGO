#pragma once
// Host-only decoder for the research composition-base/density-correction format.
// The selected GPU physics equations and layout are unchanged after decoding.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace carbon::em_density_factorization {
struct Knot { std::uint32_t index; float coefficient; };
struct Run { std::uint32_t index,value; };
static_assert(sizeof(Knot)==8 && sizeof(Run)==8);
template<class T> void read(std::istream& in,T* data,std::size_t n=1) {
    if(n && !in.read(reinterpret_cast<char*>(data),n*sizeof(T)))
        throw std::runtime_error("Truncated factorized EM package");
}
inline std::vector<Run> read_runs(std::istream& in,unsigned count,unsigned maximum) {
    unsigned n{};read(in,&n);
    if(!n || n>count)throw std::runtime_error("Invalid factorized EM run count");
    std::vector<Run> out(n);read(in,out.data(),n);
    unsigned previous=0;
    for(unsigned i=0;i<n;++i) {
        if(out[i].index>=count || (i==0?out[i].index!=0:out[i].index<=previous) || out[i].value>maximum)
            throw std::runtime_error("Invalid factorized EM run");
        previous=out[i].index;
    }
    return out;
}
struct Field {
    std::vector<Knot> knots;
    std::vector<Run> zeros;
    static Field load(std::istream& in,unsigned count) {
        Field out;unsigned n{};read(in,&n);
        if(n<2 || n>count)throw std::runtime_error("Invalid factorized EM knot count");
        out.knots.resize(n);read(in,out.knots.data(),n);
        if(out.knots.front().index!=0 || out.knots.back().index!=count-1)
            throw std::runtime_error("Incomplete factorized EM knot support");
        unsigned previous=0;
        for(unsigned i=0;i<n;++i) {
            const auto& k=out.knots[i];
            if(k.index>=count || (i && k.index<=previous) || !std::isfinite(k.coefficient))
                throw std::runtime_error("Invalid factorized EM knot");
            previous=k.index;
        }
        out.zeros=read_runs(in,count,1);return out;
    }
    void expand(float* dst,unsigned stride,const float* base,unsigned base_stride,
                unsigned count,float scale,bool nonnegative)const {
        unsigned k=0,z=0;
        for(unsigned i=0;i<count;++i) {
            while(k+1<knots.size()-1 && knots[k+1].index<i)++k;
            while(z+1<zeros.size() && zeros[z+1].index<=i)++z;
            const auto& a=knots[k];const auto& b=knots[k+1];
            // Host double interpolation prevents avoidable coefficient rounding;
            // the final expanded table and all GPU scoring remain FP32.
            const double fraction=double(i-a.index)/double(b.index-a.index);
            const double coefficient=i==a.index?double(a.coefficient):i==b.index?double(b.coefficient):
                double(a.coefficient)+fraction*(double(b.coefficient)-a.coefficient);
            const float scaled=base[i*base_stride]*scale;
            float value=zeros[z].value?0.f:float(scaled!=0.f?double(scaled)*coefficient:coefficient);
            if(nonnegative)value=std::max(0.f,value);
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite expanded EM field");
            dst[i*stride]=value;
        }
    }
};
inline void expand_runs(float* dst,unsigned stride,unsigned count,const std::vector<Run>& runs) {
    unsigned r=0;
    for(unsigned i=0;i<count;++i) {
        while(r+1<runs.size() && runs[r+1].index<=i)++r;
        dst[i*stride]=float(runs[r].value);
    }
}
inline void require_end(std::istream& in) {
    if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Factorized EM trailing bytes");
}
} // namespace carbon::em_density_factorization
