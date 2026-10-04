#pragma once
// Research representation of the SAME four-point rate interpolant. Disk data
// and the collision majorant are unchanged. Runtime evaluation is FP32.
#include "carbon/electron_rate_bound.hpp"

namespace carbon::electron_rate_polynomial {
struct Piece {float low{},high{},value[4]{};unsigned polynomial{};};
template<class Piece>inline Piece prepare(Piece r) {
    // Match the already accepted FP32 ordinates, rather than silently using
    // more precise native ordinates. Double is used only at bank loading.
    const double a=float(r.value[0]),b=float(r.value[1]);
    const double c=float(r.value[2]),d=float(r.value[3]);
    Piece poly=r;
    poly.value[0]=float((9*(b+c)-(a+d))/16);
    poly.value[1]=float((a-d)/6+4.5*(c-b));
    poly.value[2]=float(4*((a+d)-(b+c)));
    poly.value[3]=float(32*((b-c)+(d-a)/3));
    // Protect zero crossings and poorly conditioned pieces. The original
    // ordinates occupy the same four slots on these pieces, with an explicit
    // tag, so no new positive-support branch can be introduced near zero.
    using namespace electron_rate_bound;
    float lower=std::numeric_limits<float>::infinity();
    for(unsigned bin=0;bin<64;++bin){
        const Interval z{float(bin)/64-.5f,float(bin+1)/64-.5f};
        Interval value{float(poly.value[3]),float(poly.value[3])};
        for(int i=2;i>=0;--i)value=add(multiply(value,z),{float(poly.value[i]),float(poly.value[i])});
        lower=std::min(lower,value.low);
    }
    const double scale=std::max({std::abs(a),std::abs(b),std::abs(c),std::abs(d)});
    if(scale>0&&lower>.01*scale){poly.polynomial=1;return poly;}
    r.polynomial=0;return r;
}
template<class Piece>inline float evaluate(const Piece& r,float t) {
    if(!r.polynomial){
        constexpr float x[]={.125f,.375f,.625f,.875f};float result=0;
        for(unsigned j=0;j<4;++j){float term=r.value[j];for(unsigned k=0;k<4;++k)if(k!=j)term*=(t-x[k])/(x[j]-x[k]);result+=term;}
        return result;
    }
    const float z=t-.5f;
    return std::fma(std::fma(std::fma(float(r.value[3]),z,float(r.value[2])),
                              z,float(r.value[1])),z,float(r.value[0]));
}
template<class Piece>inline float piece_upper(const Piece& r) {
    if(!r.polynomial)return electron_rate_bound::piece_upper(r);
    using namespace electron_rate_bound;
    float upper=0;
    for(unsigned bin=0;bin<64;++bin){
        const Interval z{float(bin)/64-.5f,float(bin+1)/64-.5f};
        Interval value{float(r.value[3]),float(r.value[3])};
        for(int i=2;i>=0;--i)value=add(multiply(value,z),{float(r.value[i]),float(r.value[i])});
        upper=std::max(upper,value.high);
    }
    return upper;
}
template<class Source>inline Piece convert(const Source& r){
    return prepare(Piece{float(r.low),float(r.high),{float(r.value[0]),float(r.value[1]),float(r.value[2]),float(r.value[3])}});
}
template<class Pieces>inline double certify(const Pieces& rows,float low,float high,float majorant){
    // Retain all original coverage/domain checks and certify the newly stored
    // coefficients separately against the SAME transport majorant.
    electron_rate_bound::certify(rows,low,high,majorant);
    float upper=0;
    for(const auto& r:rows)upper=std::max(upper,piece_upper(convert(r)));
    if(!std::isfinite(upper)||double(upper)>.99*double(majorant))
        throw std::runtime_error("FP32 polynomial rate certificate: insufficient rounding reserve");
    return double(upper)/majorant;
}
} // namespace carbon::electron_rate_polynomial
