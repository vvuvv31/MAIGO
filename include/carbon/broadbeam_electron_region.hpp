#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace carbon {
// Research selection, not a deterministic electron/photon range bound.
// Box dilation conservatively includes voxel boxes within the requested halo.
// Selected packets retain the full shared transport after leaving this mask.
inline std::vector<std::uint8_t> broadbeam_electron_region(
    const std::vector<float>& density, std::array<unsigned,3> dims,
    std::array<double,3> spacing, double threshold, double halo) {
    const std::size_t n=std::size_t(dims[0])*dims[1]*dims[2];
    if(!dims[0]||!dims[1]||!dims[2]||density.size()!=n||
       !std::isfinite(threshold)||threshold<=0||!std::isfinite(halo)||halo<0)
        throw std::invalid_argument("Invalid broadbeam electron region");
    std::vector<std::uint8_t> mask(n),next(n);
    for(std::size_t i=0;i<n;++i) {
        if(!std::isfinite(density[i])||density[i]<=0)
            throw std::invalid_argument("Invalid broadbeam electron density");
        mask[i]=density[i]<=threshold;
    }
    const std::array<std::size_t,3> stride{1,dims[0],std::size_t(dims[0])*dims[1]};
    for(unsigned a=0;a<3;++a) {
        if(!std::isfinite(spacing[a])||spacing[a]<=0)
            throw std::invalid_argument("Invalid broadbeam electron spacing");
        const unsigned radius=static_cast<unsigned>(std::min(double(dims[a]),std::ceil(halo/spacing[a])));
        if(!radius)continue;
        const auto s=stride[a];const unsigned len=dims[a];
        for(std::size_t base=0;base<n;++base) {
            if((base/s)%len)continue;
            unsigned sum=0;
            for(unsigned j=0;j<=std::min(radius,len-1);++j)sum+=mask[base+j*s];
            for(unsigned j=0;j<len;++j) {
                next[base+j*s]=sum>0;
                if(j>=radius)sum-=mask[base+(j-radius)*s];
                if(j+radius+1<len)sum+=mask[base+(j+radius+1)*s];
            }
        }
        mask.swap(next);
    }
    return mask;
}
}
