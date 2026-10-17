// Scorer for ElectronicLETMoment
#pragma once
#include "TsVBinnedScorer.hh"
#include "ElectronicStoppingTable.hh"

// Independent additive moments for unrestricted, electronic, pre-step LETd:
// numerator sum(S_e^2 dl), denominator sum(S_e dl). Native numeric output
// units are MeV^2/mm and MeV, respectively; their ratio is keV/um.
class ElectronicLETMoment : public TsVBinnedScorer {
public:
    ElectronicLETMoment(TsParameterManager*, TsMaterialManager*, TsGeometryManager*,
        TsScoringManager*, TsExtensionManager*, G4String, G4String, G4String,
        G4bool isSubScorer = false);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UpdateForNewRun(G4bool rebuiltSomeComponents) override;
private:
    std::shared_ptr<const ElectronicStoppingTable> stopping_;
    std::map<std::pair<const G4Material*, const G4ParticleDefinition*>,
             const ElectronicStoppingTable::Curve*> curves_;
    G4int moment_;
    G4double minimumEnergy_;
};
