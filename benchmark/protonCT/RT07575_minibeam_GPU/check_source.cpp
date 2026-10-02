// Independent source pose/paraxial sampling check against frozen TOPAS extension.
#include "carbon/topas_spots.hpp"
#include <algorithm>
#include "TsPBSCoordinateModel.hh"
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
int main(int argc,char** argv) {
    assert(argc==3);
    std::map<int,std::array<double,3>> csv;
    std::ifstream in(argv[2]);std::string line;std::getline(in,line);
    while(std::getline(in,line)) {std::stringstream s(line);std::string v;std::array<double,5> f{};int i=0;while(std::getline(s,v,','))f[i++]=std::stod(v);csv[(int)f[0]]={f[1],f[2],f[3]};}
    auto plan=carbon::TopasSpotPlan::from_file(argv[1]);plan.sad_mm=450;
    std::mt19937 rng(107575);std::normal_distribution<float> gauss(0,.006);
    double maxdir=0,maxpos=0;std::size_t tested=0;
    for(const auto& spot:plan.spots) {
        auto xy=csv.at(spot.spot_id);
        auto ray=TsPBSCoordinateModel::Compute(xy[0],xy[1],6227.8,7008.6,450,TsPBSSpotConvention::TPS);
        auto pose=plan.tps_zero_beam_pose_for_spot(spot);
        maxpos=std::max(maxpos,std::abs(pose.origin_x_mm-ray.xSrcMm));
        maxpos=std::max(maxpos,std::abs(pose.origin_y_mm-ray.ySrcMm));
        maxpos=std::max(maxpos,std::abs(pose.origin_z_mm-ray.zSrcMm));
        assert(std::abs(spot.energy_MeV-xy[2])<1e-10);
        float wx=pose.uz_x,wy=pose.uz_z,wz=pose.uz_y;
        for(int j=0;j<4096;++j) {
            float xp=gauss(rng),yp=gauss(rng);
            float tx=std::tan(std::atan2(wx,wz)+xp),ty=std::tan(std::atan2(wy,wz)+yp);
            float norm=std::sqrt(1+tx*tx+ty*ty);
            double x,y,z;
            TsPBSCoordinateModel::DirectionFromAngles(ray,ray.thetaXRad+xp,ray.thetaYRad+yp,x,y,z);
            maxdir=std::max({maxdir,std::abs(tx/norm-x),std::abs(ty/norm-z),std::abs(1/norm-y)});++tested;
        }
    }
    assert(maxdir<3e-7 && maxpos<1e-12);
    std::cout<<"samples="<<tested<<" spots="<<plan.spots.size()<<" histories="<<plan.total_histories()<<" max_position_error_mm="<<maxpos<<" max_direction_error="<<maxdir<<'\n';
}
