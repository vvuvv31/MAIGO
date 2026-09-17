#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace carbon {
// Explicit TOPAS V4 little-endian wire layout: 14 int32 + 32 float64 = 312.
// Do not reinterpret_cast an aligned C++ struct onto the packed ntuple.
inline constexpr std::size_t kElectronStateV4RecordBytes=312;
struct ElectronContinuationStep {
    bool valid{};
    std::int32_t run{},event{},track{},parent{},pdg{},step{};
    std::int32_t parent_valid{},parent_step{};
    std::array<double,3> pre_position{},post_position{},pre_direction{},post_direction{};
    std::array<double,3> parent_direction{};
    double pre_energy_MeV{},post_energy_MeV{},deposited_MeV{},length_mm{};
    double birth_energy_MeV{};
    double pre_density_g_cm3{},post_density_g_cm3{};
    std::int32_t pre_material{},post_material{},post_step_status{},track_status{};
};
inline ElectronContinuationStep read_electron_continuation_step(
    const unsigned char* payload,std::size_t bytes,std::uint64_t row) {
    ElectronContinuationStep out;
    if(!payload || bytes%kElectronStateV4RecordBytes || row>=bytes/kElectronStateV4RecordBytes)return out;
    const auto* p=payload+static_cast<std::size_t>(row)*kElectronStateV4RecordBytes;
    const auto integer=[&](std::size_t offset) {
        const std::uint32_t u=static_cast<std::uint32_t>(p[offset]) |
            (static_cast<std::uint32_t>(p[offset+1])<<8) |
            (static_cast<std::uint32_t>(p[offset+2])<<16) |
            (static_cast<std::uint32_t>(p[offset+3])<<24);
        std::int32_t v;std::memcpy(&v,&u,4);return v;
    };
    const auto real=[&](std::size_t offset) {
        std::uint64_t bits=0;
        for(unsigned i=0;i<8;++i)bits|=static_cast<std::uint64_t>(p[offset+i])<<(8*i);
        double value;std::memcpy(&value,&bits,8);return value;
    };
    if(integer(144)!=4)return out;
    out.run=integer(0);out.event=integer(4);out.track=integer(8);
    out.parent=integer(12);out.pdg=integer(16);out.step=integer(20);
    out.parent_valid=integer(148);out.parent_step=integer(196);
    out.birth_energy_MeV=real(48);
    if(out.parent_valid<0 || out.parent_valid>1 || out.parent<0 ||
       (out.parent!=0 && (out.parent_valid!=1 || out.parent_step<=0)))return out;
    for(unsigned a=0;a<3;++a) {
        out.pre_position[a]=real(56+8*a);out.post_position[a]=real(80+8*a);
        out.pre_direction[a]=real(232+8*a);out.post_direction[a]=real(256+8*a);
        out.parent_direction[a]=real(160+8*a);
        if(!std::isfinite(out.pre_position[a]) || !std::isfinite(out.post_position[a]) ||
           !std::isfinite(out.pre_direction[a]) || !std::isfinite(out.post_direction[a]) ||
           !std::isfinite(out.parent_direction[a]))return out;
    }
    out.deposited_MeV=real(104);out.pre_energy_MeV=real(112);out.post_energy_MeV=real(120);
    out.pre_density_g_cm3=real(136);out.length_mm=real(280);out.post_density_g_cm3=real(288);
    out.pre_material=integer(296);out.post_material=integer(300);
    out.post_step_status=integer(304);out.track_status=integer(308);
    for(double v:{out.birth_energy_MeV,out.deposited_MeV,out.pre_energy_MeV,out.post_energy_MeV,out.length_mm,
                  out.pre_density_g_cm3,out.post_density_g_cm3})if(!std::isfinite(v) || v<0)return out;
    if(out.track<=0 || out.step<=0 || out.pre_density_g_cm3<=0 || out.pre_material<0 || out.post_material< -1 ||
       out.post_step_status<0 || out.post_step_status>7 || out.track_status<0 || out.track_status>5)return out;
    out.valid=true;return out;
}
inline bool same_electron_continuation_track(const ElectronContinuationStep& a,
                                            const ElectronContinuationStep& b) {
    return a.valid && b.valid && a.run==b.run && a.event==b.event && a.track==b.track &&
           a.pdg==b.pdg && a.parent==b.parent && static_cast<std::int64_t>(b.step)==static_cast<std::int64_t>(a.step)+1;
}
inline bool generated_by_electron_step(const ElectronContinuationStep& parent,
                                       const ElectronContinuationStep& child) {
    return parent.valid && child.valid && child.step==1 && child.parent_valid==1 &&
           parent.run==child.run && parent.event==child.event &&
           parent.track==child.parent && parent.step==child.parent_step &&
           parent.track!=child.track;
}
} // namespace carbon
