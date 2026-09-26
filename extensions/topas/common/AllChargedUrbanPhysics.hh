// Physics Module for AllChargedUrbanPhysics
#ifndef AllChargedUrbanPhysics_hh
#define AllChargedUrbanPhysics_hh
#include "G4VPhysicsConstructor.hh"
class TsParameterManager;
// Place after the EM constructor. Replaces hadron MSC, matching the Urban bank.
class AllChargedUrbanPhysics : public G4VPhysicsConstructor {
public:
 explicit AllChargedUrbanPhysics(G4int verbose = 0);
 explicit AllChargedUrbanPhysics(TsParameterManager*);
 void ConstructParticle() override {}
 void ConstructProcess() override;
};
#endif
