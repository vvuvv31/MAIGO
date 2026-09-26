// New global-path contracts: true-path collision truncation, charged species,
// and exact navigation between the water mother and its replica ROI.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include "carbon/minibeam_collimator.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/water_electron_response.hpp"
namespace carbon { namespace {
#include "detail/sycl_device_math.inc"
#include "detail/sycl_urban_global.inc"
int run(const char* path,const char* sha) {
    auto package=UrbanMcsPackage::load(path,sha);auto view=package.view();
    unsigned failed=0,total=0;
    auto check=[&](bool good,const char* msg){++total;if(!good){++failed;std::printf("FAIL %s\n",msg);}};
    auto water=[](Direction3F p,Direction3F d){return urban_global_water_cell(p,d,{-80,-80,0},{80,80,350},{1,1,.5},160,160,700,2000);};
    auto outer=water({81,0,10},{-1,0,0});
    check(outer.valid && outer.safety.kind==2 && outer.face_distance==1 && outer.face_mask==1,"exterior water reenters ROI");
    check(urban_v2_safety_at(81,0,10,outer.safety)==1,"exterior safety includes daughter ROI");
    auto point=urban_voxel_snap_endpoint(outer,1,{80.00001f,0,10});
    auto inward=water(point,{-1,0,0}),outward=water(point,{1,0,0});
    check(inward.valid && inward.ix==159 && inward.face_distance==1,"ROI inward exact-face ownership");
    check(outward.valid && outward.safety.kind==2 && outward.face_distance==1920,"ROI outward mother ownership");
    check(urban_global_outside_water({0,0,0},{0,0,-1},2000,350),"upstream exact-face escape");
    check(!urban_global_outside_water({2000,0,10},{-1,0,0},2000,350),"mother inward face remains water");
    check(urban_voxel_axis(-1e-7f,1,-80,80,1,160).index==79,"negative near-zero does not round into next voxel");
    check(urban_voxel_axis(0,-1,-80,80,1,160).index==79,"backward zero face ownership");
    const Direction3F pos{0,0,.5f},dir{.3f,.4f,std::sqrt(.75f)};
    auto cell=urban_voxel_cell(pos,dir,{-1,-1,0},{1,1,1},{2,2,1},1,1,1);
    for(unsigned m=0;m<view.material_count;++m)for(unsigned i=0;i<view.species_count;++i)
      for(float energy:{.1f,1.f,1.13101f,5.24f,5.25f,10.f,100.f,1000.f,5000.f}) {
       const float edge=.05f*view.at(m,i,energy).range(energy);
       for(float ceiling:{.25f,edge,std::nextafter(edge,0.f),std::nextafter(edge,INFINITY)}) {
        auto p=urban_global_prepare(view,m,i,energy,ceiling,pos,dir,cell,true,{},2026,1,1);
        if(!p.valid){++failed;std::printf("FAIL prepare m=%u Z=%u A=%u E=%g\n",m,view.species[i].z,view.species[i].a,energy);continue;}
        for(float fraction:{1.f,.5f,.01f}) {
            const float t=p.true_path*fraction;auto x=urban_global_finalize(p,t,2026,1,1);++total;
            const auto norm=x.direction.x*x.direction.x+x.direction.y*x.direction.y+x.direction.z*x.direction.z;
            const auto dx=x.displacement_mm.x,dy=x.displacement_mm.y,dz=x.displacement_mm.z;
            const auto end=urban_voxel_snap_endpoint(cell,x.final_geom_path_mm,
                {pos.x+dir.x*x.final_geom_path_mm+dx,pos.y+dir.y*x.final_geom_path_mm+dy,pos.z+dir.z*x.final_geom_path_mm+dz});
            bool good=x.proposal_valid && x.energy_prediction_valid &&
                x.final_true_path_mm==t && x.final_geom_path_mm>0 && x.final_geom_path_mm<=t &&
                x.stable_delta_mm>=0 && x.stable_delta_mm<=t && std::isfinite(dx+dy+dz+norm) &&
                std::fabs(norm-1)<1e-5 && end.x>=-1 && end.x<=1 && end.y>=-1 && end.y<=1 && end.z>=0 && end.z<=1;
            if(!good){++failed;if(failed<25)std::printf("FAIL finalize m=%u Z=%u A=%u E=%g f=%g t=%g g=%g predicted=%g valid=%d\n",m,view.species[i].z,view.species[i].a,energy,fraction,t,x.final_geom_path_mm,x.predicted_internal_energy_mev,x.energy_prediction_valid);}
        }
      }
      }
    std::printf("URBAN_GLOBAL_CONTRACT=%s checks=%u failures=%u\n",failed?"FAIL":"PASS",total,failed);
    return failed?1:0;
}
} }
int main(int argc,char** argv){if(argc!=3)return 2;return carbon::run(argv[1],argv[2]);}
