#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {
constexpr std::uint32_t kWaterPathNone=0xffffffffU;
struct WaterElectronChannel { double low{},high{},fraction{},unresolved{}; std::uint32_t offset{},count{}; };
struct WaterElectronSample { double cdf{}; std::array<double,3> pre{},post{}; };
struct WaterElectronPathNode { std::uint32_t previous{},reserved{}; std::array<double,3> point{}; };
struct WaterElectronDraw {
    bool valid{}; double fraction{},unresolved{}; std::uint32_t head{kWaterPathNone};
    std::array<double,3> point{};
    std::uint32_t sample_index{kWaterPathNone};
};
inline WaterElectronDraw sample_water_electron_response(double energy,double u,double along,
    const WaterElectronChannel* channels,const WaterElectronSample* samples,const std::uint32_t* heads,
    std::size_t channel_count=60) {
    WaterElectronDraw out;
    if(!channels || !samples || !heads || channel_count==0 || !std::isfinite(energy) ||
       energy<channels[0].low || energy>channels[channel_count-1].high ||
       !std::isfinite(u) || u<0 || u>1 || !std::isfinite(along) || along<0 || along>1)return out;
    // The existing float RNG can round its largest midpoint to exactly 1.
    // The closed upper CDF endpoint selects the last sample, not a miss.
    // Upper bound on channel lows: shared edges select the higher-energy
    // channel, while the final measured high endpoint stays inclusive.
    std::size_t begin=0,end=channel_count;
    while(begin<end) {
        const auto mid=begin+(end-begin)/2;
        if(channels[mid].low<=energy)begin=mid+1;else end=mid;
    }
    if(begin==0)return out;
    const auto bin=begin-1;
    const auto c=channels[bin];
    if(energy<c.low || energy>c.high)return out;
    out.fraction=c.fraction;out.unresolved=c.unresolved;
    if(!c.count) {out.valid=c.fraction==0;return out;}
    std::uint32_t lo=0,hi=c.count-1;
    while(lo<hi) {const auto mid=lo+(hi-lo)/2;
        if(u<samples[c.offset+mid].cdf)hi=mid;else lo=mid+1;}
    const auto index=c.offset+lo;const auto s=samples[index];out.head=heads[index];
    out.sample_index=index;
    for(int a=0;a<3;++a)out.point[a]=s.pre[a]+along*(s.post[a]-s.pre[a]);
    out.valid=true;return out;
}
enum class WaterElectronPathStatus { contained,escaped,invalid };
inline WaterElectronPathStatus water_electron_path_in_slab(
    const WaterElectronDraw& draw,double birth_z,double length,
    const std::array<double,3>& world_z_axis,const WaterElectronPathNode* nodes,std::size_t count,
    const double* prefix_radius=nullptr) {
    if(!draw.valid || !nodes || !std::isfinite(birth_z) || !std::isfinite(length) || length<=0)
        return WaterElectronPathStatus::invalid;
    for(double v:world_z_axis)if(!std::isfinite(v))return WaterElectronPathStatus::invalid;
    auto inside=[&](const std::array<double,3>& p) {
        const double z=birth_z+p[0]*world_z_axis[0]+p[1]*world_z_axis[1]+p[2]*world_z_axis[2];
        return std::isfinite(z) && z>=0 && z<length;
    };
    // Only the physical slab faces are material boundaries. The transverse
    // dose ROI is not a water boundary and must not truncate the trajectory.
    if(!inside(draw.point))return WaterElectronPathStatus::escaped;
    auto head=draw.head;
    if(head!=kWaterPathNone && head>=count)return WaterElectronPathStatus::invalid;
    if(prefix_radius && head!=kWaterPathNone) {
        const double radius=prefix_radius[head]*std::sqrt(world_z_axis[0]*world_z_axis[0]+world_z_axis[1]*world_z_axis[1]+world_z_axis[2]*world_z_axis[2]);
        if(birth_z>radius && birth_z+radius<length)return WaterElectronPathStatus::contained;
    }
    while(head!=kWaterPathNone) {
        if(head>=count)return WaterElectronPathStatus::invalid;
        const auto node=nodes[head];
        if(!inside(node.point))return WaterElectronPathStatus::escaped;
        if(node.previous!=kWaterPathNone && node.previous>=head)return WaterElectronPathStatus::invalid;
        head=node.previous;
    }
    return WaterElectronPathStatus::contained;
}
struct WaterElectronResponseTable {
    int material_section{-1};
    double reference_density_g_cm3{1.0};
    std::string material_name;
    std::vector<WaterElectronChannel> channels;
    std::vector<WaterElectronSample> samples;
    std::vector<std::uint32_t> heads;
    std::vector<WaterElectronPathNode> nodes;
    std::vector<double> prefix_radius;
    static WaterElectronResponseTable load(const std::filesystem::path&,const std::string&,const std::string&,
                                          bool material_response=false);
};
} // namespace carbon
