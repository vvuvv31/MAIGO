#pragma once
#include "carbon/broadbeam_electron_region.hpp"
#include <limits>

namespace carbon {
// Exact Euclidean distance BETWEEN voxel boxes on an anisotropic Cartesian
// grid. Dilating seeds by one index in every axis converts box-to-box gaps
// max(|i-j|-1,0)*spacing into a standard squared distance transform. This is
// a lower bound for any actual birth point, not a CSDA containment proof.
inline std::vector<float> broadbeam_electron_box_distance(
    const std::vector<float>& density,std::array<unsigned,3> dims,
    std::array<double,3> spacing,double threshold) {
    const auto seed=broadbeam_electron_region(density,dims,spacing,threshold,
        *std::min_element(spacing.begin(),spacing.end()));
    const double inf=std::numeric_limits<double>::infinity();
    std::vector<double> distance(seed.size());
    for(std::size_t i=0;i<seed.size();++i)distance[i]=seed[i]?0:inf;
    const std::array<std::size_t,3> stride{1,dims[0],std::size_t(dims[0])*dims[1]};
    const auto maxdim=*std::max_element(dims.begin(),dims.end());
    std::vector<double> f(maxdim),edge(maxdim+1);std::vector<unsigned> site(maxdim);
    for(unsigned a=0;a<3;++a) {
        const auto s=stride[a];const unsigned len=dims[a];const double w=spacing[a]*spacing[a];
        for(std::size_t base=0;base<distance.size();++base) {
            if((base/s)%len)continue;
            for(unsigned q=0;q<len;++q)f[q]=distance[base+q*s];
            int k=-1;
            for(unsigned q=0;q<len;++q) {
                if(!std::isfinite(f[q]))continue;
                double start=-inf;
                while(k>=0) {
                    const double p=site[k];
                    start=((f[q]-f[site[k]])/w+double(q)*q-p*p)/(2*(double(q)-p));
                    if(start>edge[k])break;
                    --k;
                }
                ++k;site[k]=q;edge[k]=k?start:-inf;edge[k+1]=inf;
            }
            if(k<0)continue;
            int at=0;
            for(unsigned q=0;q<len;++q) {
                while(at<k && edge[at+1]<q)++at;
                const double delta=double(q)-site[at];distance[base+q*s]=f[site[at]]+w*delta*delta;
            }
        }
    }
    std::vector<float> result(distance.size());
    for(std::size_t i=0;i<result.size();++i)result[i]=float(std::sqrt(distance[i]));
    return result;
}

// 6-connected boundary air is a geometric proxy, not an anatomical BODY mask.
// Labels: external low density, internal cavity, tissue <1 /1..2.5 /2.5..5 /
// >5 mm. Distances are conservative box-to-box gaps, including zero at faces.
inline std::vector<std::uint8_t> broadbeam_electron_birth_regions(
    const std::vector<float>& density,std::array<unsigned,3> dims,
    double threshold,const std::vector<float>& distance) {
    const auto nx=dims[0],ny=dims[1],nz=dims[2];const std::size_t plane=std::size_t(nx)*ny,n=plane*nz;
    if(!nx||!ny||!nz||density.size()!=n||distance.size()!=n)throw std::invalid_argument("Invalid electron region labels");
    std::vector<std::uint8_t> region(n);std::vector<std::size_t> queue;queue.reserve(n/2);
    for(std::size_t i=0;i<n;++i) {
        region[i]=density[i]<=threshold?1:distance[i]<1?2:distance[i]<2.5?3:distance[i]<=5?4:5;
        if(region[i]==1&&(i%nx==0||i%nx==nx-1||(i/nx)%ny==0||(i/nx)%ny==ny-1||i/plane==0||i/plane==nz-1)) {
            region[i]=0;queue.push_back(i);
        }
    }
    auto visit=[&](std::size_t i){if(region[i]==1){region[i]=0;queue.push_back(i);}};
    for(std::size_t q=0;q<queue.size();++q) {
        const auto i=queue[q];const unsigned x=i%nx,y=(i/nx)%ny,z=i/plane;
        if(x)visit(i-1);if(x+1<nx)visit(i+1);if(y)visit(i-nx);if(y+1<ny)visit(i+nx);if(z)visit(i-plane);if(z+1<nz)visit(i+plane);
    }
    return region;
}
}
