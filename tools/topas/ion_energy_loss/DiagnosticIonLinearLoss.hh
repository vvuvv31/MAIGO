// Physics Module for DiagnosticIonLinearLoss
#pragma once
#include "G4VPhysicsConstructor.hh"
class TsParameterManager;
class DiagnosticIonLinearLoss : public G4VPhysicsConstructor {
public:
 explicit DiagnosticIonLinearLoss(TsParameterManager*);
 void ConstructParticle() override {}
 void ConstructProcess() override;
private:
 G4double limit_;
 G4int bins_;
};
