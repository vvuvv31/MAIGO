#pragma once
#include "carbon/electron_ct_boundary.hpp"
#include <array>
#include <cstdint>
#include <limits>

namespace carbon {
inline constexpr auto kElectronContinuationEnd=std::numeric_limits<std::uint64_t>::max();
struct NoExpectedScore {template<class P,class W>void operator()(const P&,W)const{}};
// W is statistical energy, distinct from the sampled particle kinetic energy.
// Exactly one terminal bucket owns W; pending particles retain their full state.
enum class ElectronPacketStatus { active, deposited, escaped, coverage_missing, invalid };
enum class ElectronPacketGap { none, material, electron_energy, photon_continuation };
struct ElectronContinuationCursor {
    bool valid{};std::uint64_t row{kElectronContinuationEnd};
    double energy_MeV{};std::array<double,3> position{},direction{};
    int section{-1};double density_g_cm3{},physical_density_g_cm3{};int pdg{11};
};
struct ElectronEnergyPacket {
    ElectronPacketStatus status{ElectronPacketStatus::invalid};
    ElectronPacketGap gap{ElectronPacketGap::none};
    ElectronContinuationCursor cursor{};double weight_MeV{};
    std::array<double,3> deposit_position{};
    std::uint64_t advances{},electron_children{},photon_children{},source_restarts{},boundary_restarts{};
    std::size_t material_table_index{std::numeric_limits<std::size_t>::max()};
};
struct ElectronMaterialIdentity {int section{-1};double density_g_cm3{};};
} // namespace carbon
