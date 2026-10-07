#pragma once
#include "TsVNtupleScorer.hh"
#include <array>
// Records only primary ionization loss and electron birth energy/angle.
// No RNG, track killing, process edits, or secondary transport records.
class IonElectronBirthNtuple : public TsVNtupleScorer {
public:
    IonElectronBirthNtuple(TsParameterManager*,TsMaterialManager*,TsGeometryManager*,
        TsScoringManager*,TsExtensionManager*,G4String,G4String,G4String,G4bool);
    G4bool ProcessHits(G4Step*,G4TouchableHistory*) override;
    void UserHookForEndOfEvent() override;
private:
    G4int run_{-1},event_{-1},kind_{},bin_{},step_{};
    G4double parentE_{},birthE_{},mu_{},loss_{},local_{},children_{},residual_{},maxStep_{},rho_{};
    struct Bin {double loss{},local{},children{},maxResidual{},maxStep{};int steps{};};
    std::array<Bin,90> sums_{};
    double initial_{},last_{};bool root_{};
    void Require(bool,const char*);
};
