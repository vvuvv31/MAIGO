// Physics Module for CarbonIonElasticPhysics

#ifndef CarbonIonElasticPhysics_hh
#define CarbonIonElasticPhysics_hh

#include "G4IonElasticPhysics.hh"

class TsParameterManager;

// TOPAS does not expose G4IonElasticPhysics as a Geant4_Modular module.  This
// named extension keeps the module list explicit while delegating all process
// construction to Geant4's official ion-elastic constructor.
class CarbonIonElasticPhysics : public G4IonElasticPhysics {
public:
    explicit CarbonIonElasticPhysics(G4int verbose = 0);
    explicit CarbonIonElasticPhysics(TsParameterManager*);
    ~CarbonIonElasticPhysics() override;
};

#endif
