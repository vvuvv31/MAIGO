#pragma once
#include "G4VPhysicsConstructor.hh"
class TsParameterManager;
// Extraction only: isolate ionization/electron response from nuclear recoil.
class PrimaryIonizationResponsePhysics : public G4VPhysicsConstructor {
public:
 explicit PrimaryIonizationResponsePhysics(G4int verbose=0);
 explicit PrimaryIonizationResponsePhysics(TsParameterManager*);
 void ConstructParticle() override {}
 void ConstructProcess() override;
};
