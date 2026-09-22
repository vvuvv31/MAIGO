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

// Shared Unified-water delta relocation (Phase E). Used by the legacy
// electron diagnostic site AND the primary Unified-EM research path so the
// two cannot drift. Pure functions of explicit inputs (basis vectors
// included): no RNG inside, no scoring, no transport feedback. Host- and
// device-callable; unit-tested with synthetic channels (the physics content
// of a real table is validated separately by its loader pins).
// Two phases preserve the legacy RNG consumption order exactly (phi/birth
// are drawn only when packet > 0):
//   phase 1: draw + packet/unresolved from (u_chan, u_along);
//   phase 2: basis + birth point + path status + ROIs.
// ROI classes on the folded pitch: 0 peak (|u|<=0.6), 1 shoulder (<1.2),
// 2 valley. Path status: 0 contained, 1 escaped/outside, 2 invalid.
struct WaterDeltaPacket {
    bool valid{false};
    double packet_mev{0.0};
    double unresolved_mev{0.0};
    WaterElectronDraw draw{};
};
inline WaterDeltaPacket sample_water_delta_packet(
    double deposited_mev, double energy_mevu, double u_chan, double u_along,
    const WaterElectronChannel* channels, const WaterElectronSample* samples,
    const std::uint32_t* heads, std::size_t channel_count) noexcept {
    WaterDeltaPacket out;
    if (!(deposited_mev > 0.0) || !std::isfinite(deposited_mev)) return out;
    out.draw = sample_water_electron_response(energy_mevu, u_chan, u_along,
                                              channels, samples, heads,
                                              channel_count);
    if (!out.draw.valid) return out;
    out.packet_mev = deposited_mev * out.draw.fraction;
    out.unresolved_mev = deposited_mev * out.draw.unresolved;
    out.valid = true;
    return out;
}
struct WaterDeltaPlacement {
    bool valid{false};
    double point_x{0.0}, point_y{0.0}, point_z{0.0};
    int path_status{2};
    int birth_roi{1}, deposit_roi{1};
};
inline int water_delta_roi(double folded_u_mm) noexcept {
    const double a = folded_u_mm < 0 ? -folded_u_mm : folded_u_mm;
    if (!(a >= 0.0)) return 1;
    if (a <= 0.6) return 0;
    if (a <= 1.2) return 1;
    return 2;
}
inline WaterDeltaPlacement place_water_delta_packet(
    const WaterDeltaPacket& packet, double px, double py, double pz,
    double dx, double dy, double dz, double step_mm, double exx, double exy,
    double exz, double eyx, double eyy, double eyz, double u_birth,
    const WaterElectronPathNode* nodes, std::size_t node_count,
    const double* prefix_radius, double phantom_len_mm,
    double pitch_mm) noexcept {
    WaterDeltaPlacement out;
    if (!packet.valid) return out;
    const double birth = u_birth;
    if (!(birth >= 0.0) || !(birth <= 1.0)) return out;
    const double bx = px + birth * step_mm * dx;
    const double by = py + birth * step_mm * dy;
    const double bz = pz + birth * step_mm * dz;
    const double p0 = packet.draw.point[0], p1 = packet.draw.point[1],
                 p2 = packet.draw.point[2];
    out.point_x = bx + p0 * exx + p1 * eyx + p2 * dx;
    out.point_y = by + p0 * exy + p1 * eyy + p2 * dy;
    out.point_z = bz + p0 * exz + p1 * eyz + p2 * dz;
    const WaterElectronPathStatus status = water_electron_path_in_slab(
        packet.draw, bz, phantom_len_mm, {exz, eyz, dz}, nodes, node_count,
        prefix_radius);
    if (status == WaterElectronPathStatus::invalid) return out;
    out.path_status = (status == WaterElectronPathStatus::contained) ? 0 : 1;
    if (pitch_mm > 0.0) {
        const double bu = bx - pitch_mm * std::floor(bx / pitch_mm + 0.5);
        const double du = out.point_x -
            pitch_mm * std::floor(out.point_x / pitch_mm + 0.5);
        out.birth_roi = water_delta_roi(bu);
        out.deposit_roi = water_delta_roi(du);
    }
    out.valid = true;
    return out;
}
} // namespace carbon
