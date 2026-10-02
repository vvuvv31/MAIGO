#include "carbon/electron_ct_birth_position.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
using namespace carbon;
int main() {
    std::vector<float> rho(12*4*607,1.f);
    std::vector<std::uint8_t> section(rho.size(),1);
    ElectronCtGeometry g{{12,4,607},{-60,-42,0},{10,21,.5},rho.data(),section.data()};
    const std::array<double,3> p{-13.254055976867676,3.1604459285736084,303.499755859375};
    const std::array<double,3> d{-0.0096687925979495049,-0.030734028667211533,0.99948078393936157};
    const float step=0.00024426745949313045f;
    const double u=0.99999998718822924;
    const std::array<double,3> old{p[0]+u*step*d[0],p[1]+u*step*d[1],p[2]+u*step*d[2]};
    assert(old[2]>303.5 && !electron_ct_point_material(g,old,d).valid);
    const auto fixed=electron_ct_birth_position(g,p,d,step,u);
    assert(fixed[2]<303.5 && electron_ct_point_material(g,fixed,d).valid);
    const double exact=(303.5-p[2])/d[2];
    for(unsigned k=0;k<3;++k)assert(fixed[k]==p[k]+u*exact*d[k]);
    // A materially excessive step must not be silently clipped.
    const auto bad=electron_ct_birth_position(g,p,d,step*1.001f,.5);
    assert(!electron_ct_point_material(g,bad,d).valid);
    assert(!electron_ct_point_material(g,electron_ct_birth_position(g,old,d,step,.5),d).valid);
    std::mt19937_64 rng(20261001);
    std::size_t checks=0;
    for(unsigned axis=0;axis<3;++axis)for(int sign:{-1,1}) {
        auto start=std::array<double,3>{0,0,100};
        std::array<double,3> dir{};dir[axis]=sign*.99948078393936157;
        const double face=g.origin[axis]+(sign>0?g.bins[axis]*g.spacing[axis]:0);
        start[axis]=face-sign*.000244140625;
        const float len=float((face-start[axis])/dir[axis]);
        for(unsigned i=0;i<100000;++i) {
            const double along=i?std::generate_canonical<double,53>(rng):std::nextafter(1.,0.);
            const auto q=electron_ct_birth_position(g,start,dir,len,along);
            assert(electron_ct_point_material(g,q,dir).valid);++checks;
        }
    }
    // Internal faces keep exact directed material ownership and original
    // arithmetic; no epsilon or inherited primary material is allowed.
    for(unsigned i=0;i<100000;++i) {
        std::array<double,3> a{-5,0,100},v{1,.01,.03};
        const double along=std::generate_canonical<double,53>(rng);
        auto q=electron_ct_birth_position(g,a,v,10.f,along);
        for(unsigned k=0;k<3;++k)assert(q[k]==a[k]+along*10.f*v[k]);
        assert(electron_ct_point_material(g,q,v).valid);++checks;
    }
    const auto face=electron_ct_birth_position(g,{-5,0,100},{1,0,0},10.f,.5);
    assert(face[0]==0);
    // Reproduce normalization rounding with the actual CT x pitch.
    g={{1200,1,1},{-60,0,0},{double(.1f),1,1},rho.data(),section.data()};
    const double hi=-60+1200*double(.1f);
    auto q=electron_ct_birth_position(g,{hi-.000244140625,.5,.5},{1,0,0},.000244140625f,std::nextafter(1.,0.));
    assert(electron_ct_point_material(g,q,{1,0,0}).valid);
    std::cout<<checks<<" birth-segment checks plus failing-seed/outer-face regressions passed\n";
}
