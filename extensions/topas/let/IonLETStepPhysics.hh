#pragma once
#include "G4VPhysicsConstructor.hh"

class TsParameterManager;

// Optional reference step control, independent of the observational LET scorer.
// Place after the EM constructor. Geant4 copies these controls to its loss
// processes during table preparation, before any histories are transported.
class IonLETStepPhysics : public G4VPhysicsConstructor {
public:
    explicit IonLETStepPhysics(TsParameterManager*);
    void ConstructParticle() override {}
    void ConstructProcess() override;
private:
    G4double fraction_, finalRange_;
    G4bool includeProtons_;
};
