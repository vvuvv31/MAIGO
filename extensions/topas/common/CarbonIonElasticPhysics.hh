// Physics Module for CarbonIonElasticPhysics
#pragma once

#include "G4IonElasticPhysics.hh"

class TsParameterManager;

// Expose Geant4's official ion-elastic constructor to TOPAS module lists.
// Hadron elastic modules alone do not register elastic for GenericIon.
class CarbonIonElasticPhysics : public G4IonElasticPhysics {
public:
    explicit CarbonIonElasticPhysics(G4int verbose = 0);
    explicit CarbonIonElasticPhysics(TsParameterManager*);
    ~CarbonIonElasticPhysics() override;
    void ConstructProcess() override;
};
