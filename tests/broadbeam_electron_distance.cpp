#include "carbon/broadbeam_electron_distance.hpp"
#include <cassert>
#include <random>
#include <iostream>
int main() {
    std::mt19937 gen(717);
    for(auto dims:{std::array<unsigned,3>{7,5,3},{1,5,1},{4,1,7}}) {
        const unsigned n=dims[0]*dims[1]*dims[2];const std::array<double,3> spacing{.7,2.,.3};
        for(unsigned trial=0;trial<60;++trial) {
            std::vector<float> rho(n,1.f);
            for(unsigned i=0;i<n;++i)if(trial && gen()%9==0)rho[i]=.01f;
            auto distance=carbon::broadbeam_electron_box_distance(rho,dims,spacing,.1);
            for(unsigned i=0;i<n;++i) {
                double expected=INFINITY;
                for(unsigned j=0;j<n;++j)if(rho[j]<=.1) {
                    unsigned ii=i,jj=j;double squared=0;
                    for(unsigned a=0;a<3;++a){const double gap=std::max(0,std::abs(int(ii%dims[a])-int(jj%dims[a]))-1)*spacing[a];squared+=gap*gap;ii/=dims[a];jj/=dims[a];}
                    expected=std::min(expected,std::sqrt(squared));
                }
                assert((std::isinf(expected)&&std::isinf(distance[i])) || std::abs(expected-distance[i])<1e-6);
            }
        }
    }
    const std::array<unsigned,3> dims{7,7,7};std::vector<float> rho(343,1);rho[0]=.01;rho[1]=.01;rho[171]=.01;
    auto dist=carbon::broadbeam_electron_box_distance(rho,dims,{1,2,1},.1);auto labels=carbon::broadbeam_electron_birth_regions(rho,dims,.1,dist);
    assert(labels[0]==0&&labels[1]==0&&labels[171]==1&&labels[170]==2);
    std::cout<<"PASS anisotropic box-distance versus exhaustive search, no seeds, degenerate axes, boundary-air/cavity connectivity\n";
}
