// Physics Module for EMOnlyNoLossFluctuation

#include "EMOnlyNoLossFluctuation.hh"

#include "G4EmParameters.hh"
#include "G4ios.hh"

EMOnlyNoLossFluctuation::EMOnlyNoLossFluctuation(
    TsParameterManager* /*parameter_manager*/)
    : G4VPhysicsConstructor("EMOnlyNoLossFluctuation") {
    // This is the official Geant4 API behind /process/eLoss/fluct false.
    G4EmParameters::Instance()->SetLossFluctuations(false);
    G4cout << "EMOnlyNoLossFluctuation: G4EmParameters::LossFluctuation = "
           << G4EmParameters::Instance()->LossFluctuation() << G4endl;
}

void EMOnlyNoLossFluctuation::ConstructParticle() {}

void EMOnlyNoLossFluctuation::ConstructProcess() {}
