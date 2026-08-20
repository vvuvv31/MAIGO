// Physics Module for EMOnlyNoLossFluctuation

#ifndef EMOnlyNoLossFluctuation_hh
#define EMOnlyNoLossFluctuation_hh

#include "G4VPhysicsConstructor.hh"

class TsParameterManager;

// Validation-only module that selects the official Geant4 no-loss-fluctuation
// mode before the EM constructors build their energy-loss processes.
class EMOnlyNoLossFluctuation final : public G4VPhysicsConstructor {
public:
    explicit EMOnlyNoLossFluctuation(TsParameterManager* parameter_manager);
    ~EMOnlyNoLossFluctuation() override = default;

    void ConstructParticle() override;
    void ConstructProcess() override;
};

#endif
