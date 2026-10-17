// Scorer for ElectronicLETMoment
#include "ElectronicLETMoment.hh"
#include "TsParameterManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include <cmath>
#include <stdexcept>

ElectronicLETMoment::ElectronicLETMoment(TsParameterManager* pM, TsMaterialManager* mM,
    TsGeometryManager* gM, TsScoringManager* scM, TsExtensionManager* eM,
    G4String name, G4String quantity, G4String output, G4bool sub)
    : TsVBinnedScorer(pM, mM, gM, scM, eM, name, quantity, output, sub),
      moment_(pM->GetIntegerParameter(GetFullParmName("Moment"))),
      minimumEnergy_(pM->GetDoubleParameter(GetFullParmName("MinimumKineticEnergy"), "Energy")) {
    if ((moment_ != 1 && moment_ != 2) || minimumEnergy_ < 0)
        throw std::invalid_argument("ElectronicLETMoment requires Moment=1/2 and a nonnegative energy cutoff");
    // Explicit numeric MeV and mm conversion below; the two moment fields
    // have different dimensions, documented in the campaign/LET metadata.
    SetUnit("");
    stopping_ = ElectronicStoppingTable::Load(
        pM->GetStringParameter(GetFullParmName("StoppingPowerFile")));
}
void ElectronicLETMoment::UpdateForNewRun(G4bool rebuiltSomeComponents) {
    // Geometry updates can replace materials or reuse their addresses. Check
    // their fingerprints again in every run, on each worker as well as master.
    curves_.clear();
    TsVBinnedScorer::UpdateForNewRun(rebuiltSomeComponents);
}
G4bool ElectronicLETMoment::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) { ++fSkippedWhileInactive; return false; }
    const auto* particle = step->GetTrack()->GetDefinition();
    const auto* pre = step->GetPreStepPoint();
    const auto length = step->GetStepLength();
    if (particle->GetAtomicNumber() < 1 || particle->GetAtomicMass() < 1 ||
        pre->GetKineticEnergy() < minimumEnergy_ || !(length > 0)) return false;
    // Never query G4EmCalculator here: it selects and mutates the same
    // models used by transport. The separate native extraction process supplies
    // unrestricted electronic stopping for this exact material and projectile.
    const auto key = std::make_pair(pre->GetMaterial(), particle);
    auto it = curves_.find(key);
    if (it == curves_.end())
        it = curves_.emplace(key, &stopping_->Find(pre->GetMaterial(),
            particle->GetAtomicNumber(), particle->GetAtomicMass())).first;
    const auto stopping = it->second->At(pre->GetKineticEnergy()/MeV);
    if (!std::isfinite(stopping) || stopping < 0)
        throw std::runtime_error("Invalid electronic stopping power in LET scorer");
    const auto value = stopping * (length/mm) * pre->GetWeight();
    AccumulateHit(step, moment_ == 1 ? value : value * stopping);
    return true;
}
