#pragma once
#include "TsVNtupleScorer.hh"
#include <array>
#include <map>
class G4Track;

// Isolated native reference scorer. No transport changes, RNG or GPU scoring.
class ElectronHistoryProfile : public TsVNtupleScorer {
public:
    ElectronHistoryProfile(TsParameterManager*, TsMaterialManager*,
        TsGeometryManager*, TsScoringManager*, TsExtensionManager*,
        G4String, G4String, G4String, G4bool);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;
private:
    G4int run_{-1}, event_{-1}, roots_{0}, steps_{0}, tracks_{0};
    G4double incident_{0}, charged_{0}, sink_{0}, outside_{0}, maxResidual_{0}, maxStep_{0};
    std::array<G4double, 36> profile_{};
    std::map<G4int, bool> excluded_;
    std::map<const G4Track*, bool> pending_;
    void Require(bool, const char*);
};
