// Standalone arithmetic regression, not a CPU transport implementation.
// g++ -std=c++20 -O2 -I include scripts/check_minibeam_geometry.cpp -o /tmp/check_minibeam_geometry
#include "minibeam_collimator.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>

int main() {
    using namespace carbon;
    unsigned checks=0;
    for (float offset : {0.f,1.5f,-1.5f,4.5f}) {
        for (int slit=-8;slit<=8;++slit) {
            const auto walls=minibeam_slit_interval(slit,3.f,.7f,offset);
            for (int side : {-1,1}) {
                const float wall=side<0?walls.lower_mm:walls.upper_mm;
                int selected=-99;
                if (minibeam_point_in_slit(wall,0,1,0,60,17,.7f,3,50,offset,selected))
                    throw std::runtime_error("represented wall incorrectly owned by air");
                if (minibeam_isotropic_safety_at_point(wall,0,-200,1,0,60,17,.7f,3,50,offset,-250,-150)!=0)
                    throw std::runtime_error("represented wall has nonzero navigation safety");
                const float inward=-side*.003f;
                const float path=minibeam_path_to_surface_interval(wall,0,inward,0,1,0,60,17,.7f,3,50,offset,100.f);
                // Neither opposing wall nor cylinder can be reached in 100 mm.
                // Former slit-zero code proposed 9.93411e-6 mm without moving x.
                if (path!=100.f || wall+path*inward==wall)
                    throw std::runtime_error("ghost wall crossing / nonmoving represented endpoint");
                const float inside=std::nextafter(wall,side<0?walls.upper_mm:walls.lower_mm);
                const float outside=std::nextafter(wall,side<0?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity());
                if (!minibeam_point_in_slit(inside,0,1,0,60,17,.7f,3,50,offset,selected) ||
                    minibeam_point_in_slit(outside,0,1,0,60,17,.7f,3,50,offset,selected))
                    throw std::runtime_error("adjacent represented points have inconsistent material ownership");
                checks+=4;
            }
        }
    }
    // A real crossing remains a crossing; do not fix by suppressing small steps.
    const auto walls=minibeam_slit_interval(0,3.f,.7f,1.5f);
    const float start=walls.lower_mm-.01f;
    const float path=minibeam_path_to_surface_interval(start,0,.003f,0,1,0,60,17,.7f,3,50,1.5f,100.f);
    if (std::fabs(path-(walls.lower_mm-start)/.003f)>1e-6f)
        throw std::runtime_error("real slit entry was suppressed");
    std::cout<<"Passed "<<checks+1<<" slit wall ownership, safety and flight checks\n";
}
