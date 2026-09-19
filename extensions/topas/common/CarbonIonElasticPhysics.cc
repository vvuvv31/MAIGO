// Physics Module for CarbonIonElasticPhysics

#include "CarbonIonElasticPhysics.hh"

#include "G4PhysicsConstructorFactory.hh"

G4_DECLARE_PHYSCONSTR_FACTORY(CarbonIonElasticPhysics);

CarbonIonElasticPhysics::CarbonIonElasticPhysics(const G4int verbose)
    : G4IonElasticPhysics(verbose) {}

CarbonIonElasticPhysics::CarbonIonElasticPhysics(TsParameterManager*)
    : CarbonIonElasticPhysics(0) {}

CarbonIonElasticPhysics::~CarbonIonElasticPhysics() = default;
