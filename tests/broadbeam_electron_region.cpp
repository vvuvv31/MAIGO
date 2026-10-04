#include "carbon/broadbeam_electron_region.hpp"
#include <cassert>
#include <limits>
int main() {
    using carbon::broadbeam_electron_region;
    const std::array<unsigned,3> dims{7,5,3};
    const std::array<double,3> spacing{.5,2.,.7};
    for(double halo:{0.,.1,.5,1.4,10.})for(unsigned source:{0u,27u,104u}) {
        std::vector<float> rho(105,1.f);rho[source]=.01f;
        const auto mask=broadbeam_electron_region(rho,dims,spacing,.1,halo);
        const int sx=source%7,sy=(source/7)%5,sz=source/35;
        for(int z=0;z<3;++z)for(int y=0;y<5;++y)for(int x=0;x<7;++x) {
            const bool expected=std::abs(x-sx)<=std::ceil(halo/spacing[0])&&
                std::abs(y-sy)<=std::ceil(halo/spacing[1])&&std::abs(z-sz)<=std::ceil(halo/spacing[2]);
            assert(bool(mask[(z*5+y)*7+x])==expected);
        }
    }
    assert(broadbeam_electron_region(std::vector<float>(105,1),dims,spacing,.1,5)==std::vector<std::uint8_t>(105));
    bool rejected=false;try{broadbeam_electron_region(std::vector<float>(105,0),dims,spacing,.1,5);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
}
