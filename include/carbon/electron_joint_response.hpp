#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {
inline constexpr int kElectronJointEnergyBins=100;
inline constexpr int kElectronJointSections=25;
inline constexpr int kElectronJointChannels=kElectronJointSections*kElectronJointEnergyBins;
struct ElectronJointSample { double cdf{},longitudinal_mass_g_cm2{},radial_mass_g_cm2{}; };
struct ElectronJointChannel { std::uint32_t offset{},count{}; double fraction{}; };
struct ElectronPathRange { std::uint32_t offset{},count{}; };
enum class ElectronJointStatus { hit,unsupported_section,energy_domain,invalid_payload };
struct ElectronJointDraw {
    ElectronJointStatus status{ElectronJointStatus::invalid_payload};
    double fraction{},longitudinal_mass_g_cm2{},radial_mass_g_cm2{};
    std::uint32_t sample_index{};
};
template<std::size_t N>
inline ElectronJointDraw sample_electron_joint_device(unsigned section,double energy,double u,
    const ElectronJointChannel* channels,const ElectronJointSample* samples,
    const std::array<double,N>& minimum,const std::array<double,N>& maximum,double ceiling=185) {
    static_assert(N==2 || N==kElectronJointSections);
    ElectronJointDraw out;
    if((N==2 && section!=0 && section!=8) || section>=kElectronJointSections) {
        out.status=ElectronJointStatus::unsupported_section;return out;}
    const int material=N==2 ? section==8 : section;
    if(!std::isfinite(minimum[material])) {out.status=ElectronJointStatus::unsupported_section;return out;}
    if(!std::isfinite(energy) || energy<minimum[material] || energy>maximum[material]) {
        out.status=ElectronJointStatus::energy_domain;return out;
    }
    if(!channels || !samples || !std::isfinite(u) || u<0 || u>1)return out;
    // The closed table ceiling belongs to the preceding histogram interval.
    // No out-of-domain query is clamped. Legacy 2-material fixtures stay valid.
    constexpr int stride=N==2 ? 37 : kElectronJointEnergyBins;
    if(energy<0 || energy>ceiling)return out;
    const int eb=energy==ceiling ? static_cast<int>(std::ceil(ceiling/5))-1 : static_cast<int>(energy/5);
    if(eb<0 || eb>=stride)return out;
    const auto ch=channels[material*stride+eb];
    if(!ch.count)return out;
    std::uint32_t lo=0,hi=ch.count-1;
    while(lo<hi) {const auto mid=lo+(hi-lo)/2;
        if(u<samples[ch.offset+mid].cdf)hi=mid;else lo=mid+1;}
    const auto v=samples[ch.offset+lo];
    out.status=ElectronJointStatus::hit;out.fraction=ch.fraction;
    out.sample_index=ch.offset+lo;
    out.longitudinal_mass_g_cm2=v.longitudinal_mass_g_cm2;
    out.radial_mass_g_cm2=v.radial_mass_g_cm2;return out;
}
struct ElectronJointResponseTable {
    std::array<ElectronJointChannel,kElectronJointChannels> channels{};
    std::array<double,kElectronJointSections> minimum{},maximum{};
    double energy_ceiling_MeVu{185};
    bool full_schneider_scope{false};
    std::vector<ElectronJointSample> samples;
    std::vector<ElectronPathRange> path_ranges;
    std::vector<std::array<double,3>> path_vectors;
    std::string path_sha256;
    static ElectronJointResponseTable from_csv(const std::filesystem::path& path,
        const std::string& expected_data_sha256,const std::string& expected_metadata_sha256);
};
} // namespace carbon
