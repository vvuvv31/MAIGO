#pragma once
// Host-only interval certificate for the existing FP32 Lagrange evaluator.
// This changes neither coefficients nor the transport's collision majorant.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace carbon::electron_rate_bound {
struct Interval {float low,high;};
inline float down(double x){return std::nextafter(float(x),-std::numeric_limits<float>::infinity());}
inline float up(double x){return std::nextafter(float(x),std::numeric_limits<float>::infinity());}
inline Interval multiply(Interval a,Interval b){
    const double x[]={double(a.low)*b.low,double(a.low)*b.high,double(a.high)*b.low,double(a.high)*b.high};
    return {down(*std::min_element(x,x+4)),up(*std::max_element(x,x+4))};
}
inline Interval add(Interval a,Interval b){return {down(double(a.low)+b.low),up(double(a.high)+b.high)};}
inline Interval subtract(Interval a,float x){return {down(double(a.low)-x),up(double(a.high)-x)};}
inline Interval divide(Interval a,float x){
    const double l=double(a.low)/x,h=double(a.high)/x;
    return {down(std::min(l,h)),up(std::max(l,h))};
}
template<class Piece>inline float piece_upper(const Piece& piece){
    constexpr float x[]={.125f,.375f,.625f,.875f};float upper=0;
    // Dyadic endpoints cover the entire continuum of normalized inputs.
    // Outward rounding encloses both exact intermediate values and their
    // FP32 rounding, including contraction into fused multiply-add.
    for(unsigned bin=0;bin<64;++bin){
        const Interval t{float(bin)/64,float(bin+1)/64};Interval sum{0,0};
        for(unsigned j=0;j<4;++j){Interval term{float(piece.value[j]),float(piece.value[j])};
            for(unsigned k=0;k<4;++k)if(k!=j)term=multiply(term,divide(subtract(t,x[k]),x[j]-x[k]));
            sum=add(sum,term);
        }
        upper=std::max(upper,sum.high);
    }
    return upper;
}
template<class Pieces>inline double certify(const Pieces& rows,float parameter_low,float parameter_high,float majorant){
    if(rows.empty()||!std::isfinite(majorant)||majorant<1e-15f||majorant>1e15f)
        throw std::runtime_error("FP32 hard-rate certificate: unsupported majorant magnitude");
    if(parameter_low<float(rows.front().low)||parameter_high>float(rows.back().high))
        throw std::runtime_error("FP32 hard-rate certificate: incomplete parameter coverage");
    float previous=float(rows.front().low),upper=0;
    for(const auto& r:rows){
        const float lo=float(r.low),hi=float(r.high);
        if(!std::isfinite(lo)||!std::isfinite(hi)||lo!=previous||hi<=lo||hi-lo<std::numeric_limits<float>::min())
            throw std::runtime_error("FP32 hard-rate certificate: invalid converted interval");
        for(double v:r.value)if(!std::isfinite(float(v)))
            throw std::runtime_error("FP32 hard-rate certificate: nonfinite converted coefficient");
        upper=std::max(upper,piece_upper(r));previous=hi;
    }
    // A 1% reserve is much larger than gamma_32 (<2e-6) for the density
    // interpolation and the different evaluation order of the majorant.
    // Node densities are separately restricted to [1e-6,1e3]; these and the
    // rate bounds above exclude overflow and relevant underflow in mixing.
    if(!std::isfinite(upper)||double(upper)>.99*double(majorant))
        throw std::runtime_error("FP32 hard-rate certificate: insufficient rounding reserve");
    return double(upper)/majorant;
}
} // namespace carbon::electron_rate_bound
