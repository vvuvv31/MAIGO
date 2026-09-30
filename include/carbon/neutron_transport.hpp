#pragma once
#include <array>
#include <cstdint>
namespace carbon {
// Experimental neutron pass: separate physical terminal energy ledger.
enum NeutronEnergy { NInputWater, NInputCopper, NDepositWater, NDepositOutside,
 NEscape, NUnsupported, NBelowGrid, NStepLimit, NQueueOverflow, NKineticGain,
 NPhoton, NElectron, NMissingIon, NGridDeposit, NInput, NEnergyCount };
enum NeutronCount { NBirths, NCollisions, NElastic, NInelastic, NCapture, NFission,
 NScaledEvents, NUnresolvedIntervals, NCharged, NUnsupportedCount, NLimitCount,
 NOverflowCount, NTwoBodyElastic, NRecoilExtensionSteps, NWaterUrbanSteps, NWaterUrbanUncovered, NWaterUrbanFailures, NBoundaryMaterialCorrections, NCountSize };
struct NeutronExperimentalDiagnostics {
 std::array<double,NEnergyCount> energy{};
 std::array<std::uint64_t,NCountSize> counts{};
};
}
#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
namespace carbon {
struct TransportConfig;
struct TransportResult;
void transport_neutrons_experimental(sycl::queue&,const TransportConfig&,TransportResult&);
void prefetch_neutron_inputs(const TransportConfig&);
}
#endif
