#pragma once
#include <cmath>
#include <sycl/sycl.hpp>
namespace carbon {
// Mike Giles, GPU Computing Gems (2011), inverse erf polynomial.
// Positive-product logarithm avoids loss of the finite u=1e-12 endpoint.
template<bool Native=false> inline float fluctuation_normal_quantile(float u) {
 const float q=4*u*(1-u);
 float w=Native?sycl::native::log(q):std::log(q);w=-w;
 float p;
 if(w<5){w-=2.5f;p=2.81022636e-8f;p=3.43273939e-7f+p*w;p=-3.5233877e-6f+p*w;p=-4.39150654e-6f+p*w;p=.00021858087f+p*w;p=-.00125372503f+p*w;p=-.00417768164f+p*w;p=.246640727f+p*w;p=1.50140941f+p*w;}
 else{w=(Native?sycl::native::sqrt(w):std::sqrt(w))-3.f;p=-.000200214257f;p=.000100950558f+p*w;p=.00134934322f+p*w;p=-.00367342844f+p*w;p=.00573950773f+p*w;p=-.0076224613f+p*w;p=.00943887047f+p*w;p=1.00167406f+p*w;p=2.83297682f+p*w;}
 return 1.4142135623730951f*p*(2*u-1);
}
}
