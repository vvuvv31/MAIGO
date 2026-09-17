#pragma once
#include "carbon/electron_continuation_replay.hpp"

namespace carbon {
// Buckets are proposal accelerators, NOT kinetic-energy quantization. Every
// candidate must pass an exact energy-interval check before it can be used.
struct ElectronEnergyCrossingView {
    const double* edges{};
    const std::uint64_t* offsets{};
    const std::uint64_t* rows{};
    std::size_t bins{};
    std::uint64_t count{};
    std::size_t bin(double energy) const {
        if(!edges || !offsets || !bins || !std::isfinite(energy) || energy<=edges[0] || energy>edges[bins])return bins;
        std::size_t lo=0,hi=bins;
        while(lo+1<hi) {
            const auto mid=lo+(hi-lo)/2;
            if(energy<edges[mid])hi=mid;else lo=mid;
        }
        return lo;
    }
};
enum class ElectronCrossingStatus { ready, invalid, energy_domain, empty,
                                   rejected, attempts_exhausted, unsupported_particle };
struct ElectronEnergyCrossingDraw {
    ElectronCrossingStatus status{ElectronCrossingStatus::invalid};
    std::size_t source{};
    std::uint64_t row{kElectronContinuationEnd};
    ElectronReplayStep replay{};
    double energy_MeV{},continuous_loss_fraction{},remaining_deposit_MeV{};
    bool exact_bucket_scan{};
};
// Candidate approximation: recorded step deposition is continuous; ALL child
// kinetic energy is discrete at the step end. Never scale child KE or smear it
// through the chord. Photon flights do not obey this energy-crossing law.
inline ElectronEnergyCrossingDraw inspect_electron_energy_crossing(
    const ElectronContinuationView& source,std::uint64_t row,double energy) {
    ElectronEnergyCrossingDraw result;
    result.row=row;result.energy_MeV=energy;
    result.replay=prepare_electron_replay_step(source,row);
    if(result.replay.status!=ElectronReplayStatus::ready || !std::isfinite(energy) || energy<=0)return result;
    const auto& step=result.replay.state;
    if(step.pdg!=11) {result.status=ElectronCrossingStatus::unsupported_particle;return result;}
    const double lower=step.pre_energy_MeV-step.deposited_MeV;
    // Half-open interval makes a monotone electron track appear at most once
    // at a queried energy. Discrete energy jumps are not continuous crossings.
    if(!(step.deposited_MeV>0 && energy>lower && energy<=step.pre_energy_MeV)) {
        result.status=ElectronCrossingStatus::rejected;return result;
    }
    result.continuous_loss_fraction=(step.pre_energy_MeV-energy)/step.deposited_MeV;
    result.remaining_deposit_MeV=energy-lower;
    const double residual=energy-result.remaining_deposit_MeV-step.post_energy_MeV-result.replay.child_energy_MeV;
    if(std::abs(residual)>1e-8+1e-10*energy)return result;
    result.status=ElectronCrossingStatus::ready;
    return result;
}
} // namespace carbon
