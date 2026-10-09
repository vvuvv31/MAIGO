// Physics Module for CarbonIonElasticPhysics
#include "CarbonIonElasticPhysics.hh"
#include "G4PhysicsConstructorFactory.hh"
#include "G4GenericIon.hh"
#include "G4HadronElasticProcess.hh"
#include "G4ProcessManager.hh"
#include "G4Exception.hh"
#include "G4ios.hh"

G4_DECLARE_PHYSCONSTR_FACTORY(CarbonIonElasticPhysics);

CarbonIonElasticPhysics::CarbonIonElasticPhysics(const G4int verbose)
    : G4IonElasticPhysics(verbose) {}

CarbonIonElasticPhysics::CarbonIonElasticPhysics(TsParameterManager*)
    : CarbonIonElasticPhysics(0) {}

CarbonIonElasticPhysics::~CarbonIonElasticPhysics() = default;

void CarbonIonElasticPhysics::ConstructProcess() {
    G4IonElasticPhysics::ConstructProcess();
    auto* manager = G4GenericIon::GenericIonDefinition()->GetProcessManager();
    if (manager) {
        auto* processes = manager->GetProcessList();
        for (G4int i = 0; i < manager->GetProcessListLength(); ++i) {
            if (dynamic_cast<G4HadronElasticProcess*>((*processes)[i])) {
                G4cout << "CARBON_REFERENCE_ION_ELASTIC GenericIon process="
                       << (*processes)[i]->GetProcessName()
                       << " registered=true" << G4endl;
                return;
            }
        }
    }
    G4Exception("CarbonIonElasticPhysics::ConstructProcess",
                "MissingGenericIonElastic", FatalException,
                "G4IonElasticPhysics did not register GenericIon elastic.");
}
