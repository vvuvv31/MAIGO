// Species selection and fine-voxel, true/geometrical-path contracts.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include "carbon/minibeam_collimator.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/water_electron_response.hpp"
namespace carbon { namespace {
#include "detail/sycl_device_math.inc"
#include "detail/sycl_urban_global.inc"
int run(const char* path,const char* sha) {
    auto package=UrbanMcsPackage::load(path,sha);auto view=package.view();
    unsigned total=0,failed=0;
    auto check=[&](bool good,const char* msg){++total;if(!good){++failed;std::printf("FAIL %s\n",msg);}};
    check(minibeam_urban_proton_or_helium(1,1),"proton selected");
    check(minibeam_urban_proton_or_helium(2,3) && minibeam_urban_proton_or_helium(2,4) &&
          minibeam_urban_proton_or_helium(2,6),"helium isotopes selected");
    for(auto za:{std::pair{0,1},std::pair{1,2},std::pair{1,3},std::pair{6,12},std::pair{8,16}})
        check(!minibeam_urban_proton_or_helium(za.first,za.second),"other species untouched");
    const int m=view.material(-1,1.f);check(m>=0,"Water_75eV couple available");
    if(m<0)return 1;
    struct Start {Direction3F pos,dir;};
    const Start starts[]={{{0,0,0},{0,0,1}},{{.025f,0,123.6f},{.3f,.4f,std::sqrt(.75f)}},
                          {{50,0,20},{-.6f,0,.8f}},{{-.1f,0,124},{0,0,-1}}};
    for(unsigned i=0;i<view.species_count;++i) {
        const auto sp=view.species[i];if(!minibeam_urban_proton_or_helium(sp.z,sp.a))continue;
        for(float energy:{.1f,.2f,1.f,10.f,250.f,1000.f})
          for(float ceiling:{.001f,.005f,.25f})for(const auto& start:starts) {
            auto cell=urban_global_water_cell(start.pos,start.dir,{-50,-50,0},{50,50,250},
                                             {.1f,100,.25f},1000,1,1000,50);
            auto p=urban_global_prepare(view,m,i,energy,ceiling,start.pos,start.dir,cell,true,{},2026,41,total);
            check(p.valid,"fine-voxel proposal valid");if(!p.valid)continue;
            for(float fraction:{1.f,.5f,.01f}) {
                auto x=urban_global_finalize(p,p.true_path*fraction,2026,41,total);
                const float norm=x.direction.x*x.direction.x+x.direction.y*x.direction.y+x.direction.z*x.direction.z;
                auto endpoint=urban_voxel_snap_endpoint(cell,x.final_geom_path_mm,
                  {start.pos.x+start.dir.x*x.final_geom_path_mm+x.displacement_mm.x,
                   start.pos.y+start.dir.y*x.final_geom_path_mm+x.displacement_mm.y,
                   start.pos.z+start.dir.z*x.final_geom_path_mm+x.displacement_mm.z});
                check(x.proposal_valid && x.energy_prediction_valid &&
                      x.final_true_path_mm==p.true_path*fraction && x.final_geom_path_mm>0 &&
                      x.final_geom_path_mm<=x.final_true_path_mm && std::fabs(norm-1)<1e-5 &&
                      std::isfinite(endpoint.x+endpoint.y+endpoint.z) &&
                      endpoint.x>=-50 && endpoint.x<=50 && endpoint.y>=-50 && endpoint.y<=50 &&
                      endpoint.z>=0 && endpoint.z<=250,"accepted path/displacement inside water");
            }
        }
    }
    std::printf("MINIBEAM_PROTON_HELIUM_URBAN=%s checks=%u failures=%u\n",failed?"FAIL":"PASS",total,failed);
    return failed?1:0;
}
}}
int main(int argc,char** argv){if(argc!=3)return 2;return carbon::run(argv[1],argv[2]);}
