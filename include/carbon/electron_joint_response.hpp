#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {
inline constexpr int kElectronJointEnergyBins=37;
struct ElectronJointSample { double cdf{},longitudinal_mass_g_cm2{},radial_mass_g_cm2{}; };
struct ElectronJointChannel { std::uint32_t offset{},count{}; double fraction{}; };
enum class ElectronJointStatus { hit,unsupported_section,energy_domain,invalid_payload };
struct ElectronJointDraw {
    ElectronJointStatus status{ElectronJointStatus::invalid_payload};
    double fraction{},longitudinal_mass_g_cm2{},radial_mass_g_cm2{};
};
inline ElectronJointDraw sample_electron_joint_device(unsigned section,double energy,double u,
    const ElectronJointChannel* channels,const ElectronJointSample* samples,
    const std::array<double,2>& minimum,const std::array<double,2>& maximum) {
    ElectronJointDraw out;
    if(section!=0 && section!=8) {out.status=ElectronJointStatus::unsupported_section;return out;}
    const int material=section==8;
    if(!std::isfinite(energy) || energy<minimum[material] || energy>maximum[material]) {
        out.status=ElectronJointStatus::energy_domain;return out;
    }
    if(!channels || !samples || !std::isfinite(u) || u<0 || u>1)return out;
    // Explicit histogram intervals. Only the CLOSED 185 upper endpoint belongs
    // to the final interval; no query outside the measured domain is clamped.
    const int eb=energy==185 ? 36 : static_cast<int>(energy/5);
    if(eb<0 || eb>=37)return out;
    const auto ch=channels[material*37+eb];
    if(!ch.count)return out;
    std::uint32_t lo=0,hi=ch.count-1;
    while(lo<hi) {const auto mid=lo+(hi-lo)/2;
        if(u<samples[ch.offset+mid].cdf)hi=mid;else lo=mid+1;}
    const auto v=samples[ch.offset+lo];
    out.status=ElectronJointStatus::hit;out.fraction=ch.fraction;
    out.longitudinal_mass_g_cm2=v.longitudinal_mass_g_cm2;
    out.radial_mass_g_cm2=v.radial_mass_g_cm2;return out;
}
struct ElectronJointResponseTable {
    std::array<ElectronJointChannel,74> channels{};
    std::array<double,2> minimum{},maximum{};
    std::vector<ElectronJointSample> samples;
    static ElectronJointResponseTable from_csv(const std::filesystem::path& path,
        const std::string& expected_data_sha256,const std::string& expected_metadata_sha256);
};
} // namespace carbon
