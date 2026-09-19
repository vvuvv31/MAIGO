// Physics Module for CarbonInelasticCapturePhysics

#include "CarbonInelasticCapturePhysics.hh"

#include "CarbonInelasticCaptureProcess.hh"

#include "G4BuilderType.hh"
#include "G4Exception.hh"
#include "G4HadronicProcessType.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicsConstructorFactory.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4VProcess.hh"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

G4_DECLARE_PHYSCONSTR_FACTORY(CarbonInelasticCapturePhysics);

namespace {

bool IsAllowedProjectile(const G4ParticleDefinition& particle) {
    // Capture all nuclei and all ordinary hadrons that can be produced by the
    // carbon cascade.  In particular, neutron and meson secondaries must be
    // included: primary-only transport gating is enforced before process
    // competition, so omitting them would leave their inelastic processes
    // active even though charged-fragment inelastic processes are gated.
    // The process filter below still requires exactly one applicable
    // fHadronic/fHadronInelastic process for each allowed projectile.
    const auto& name = particle.GetParticleName();
    if (name == "GenericIon") {
        // TOPAS creates GenericIon( Z, A ) beams through Geant4's generic-ion
        // process manager. The concrete isotope is carried by the dynamic
        // ion state at tracking time, not by the GenericIon definition.
        return true;
    }
    if (particle.GetAtomicNumber() > 0 && particle.GetPDGEncoding() > 0) {
        return true;
    }
    const auto& particle_type = particle.GetParticleType();
    return particle_type == "baryon" || particle_type == "meson";
}

[[noreturn]] void FatalProcessSetup(const G4ParticleDefinition& particle,
                                    const std::string& message) {
    G4ExceptionDescription description;
    description << "CINEL02 process setup for " << particle.GetParticleName() << ": "
                << message;
    G4Exception("CarbonInelasticCapturePhysics", "CINEL02-SETUP", FatalException,
                description);
    throw std::runtime_error(message);
}

}  // namespace

CarbonInelasticCapturePhysics::CarbonInelasticCapturePhysics(G4int verbose)
    : G4VPhysicsConstructor("CarbonInelasticCapturePhysics"), verbose_(verbose) {
    // This module post-processes the hadronic process list by wrapping the
    // process installed by g4ion-inclxx.  It is not itself a hadronic
    // physics builder.  Claiming bHadronInelastic makes Geant4 reject it as
    // a duplicate of the real hadronic builder before ConstructProcess()
    // can install the CINEL02 wrappers.
}

CarbonInelasticCapturePhysics::CarbonInelasticCapturePhysics(TsParameterManager*)
    : CarbonInelasticCapturePhysics(0) {}

void CarbonInelasticCapturePhysics::ConstructParticle() {}

void CarbonInelasticCapturePhysics::ConstructProcess() {
    auto* iterator = GetParticleIterator();
    iterator->reset();
    while ((*iterator)()) {
        auto* particle = iterator->value();
        if (particle == nullptr || !IsAllowedProjectile(*particle)) {
            continue;
        }
        auto* manager = particle->GetProcessManager();
        if (manager == nullptr) {
            FatalProcessSetup(*particle, "particle has no process manager");
        }
        std::vector<G4VProcess*> candidates;
        auto* processes = manager->GetProcessList();
        for (G4int index = 0; index < manager->GetProcessListLength(); ++index) {
            auto* process = (*processes)[index];
            if (process != nullptr && process->GetProcessType() == fHadronic &&
                process->GetProcessSubType() == fHadronInelastic) {
                candidates.push_back(process);
            }
        }
        // Geant4's particle table contains many positive exotic nuclei that
        // are definitions only; INCL++ does not install a hadronic process
        // for them. They are not transport projectiles and must not make
        // module construction fail. For every projectile that does have an
        // applicable inelastic process, ambiguity remains a hard error.
        if (candidates.empty()) {
            continue;
        }
        if (candidates.size() != 1U) {
            FatalProcessSetup(*particle,
                              "expected exactly one fHadronic/fHadronInelastic process, found " +
                                  std::to_string(candidates.size()));
        }
        auto* original = candidates.front();
        if (dynamic_cast<CarbonInelasticCaptureProcess*>(original) != nullptr) {
            continue;
        }
        if (dynamic_cast<G4WrapperProcess*>(original) != nullptr) {
            FatalProcessSetup(*particle, "refusing to wrap an already wrapped process");
        }
        const auto at_rest = manager->GetProcessOrdering(original, idxAtRest);
        const auto along = manager->GetProcessOrdering(original, idxAlongStep);
        const auto post = manager->GetProcessOrdering(original, idxPostStep);
        const auto active = manager->GetProcessActivation(original);
        auto* removed = manager->RemoveProcess(original);
        if (removed != original) {
            FatalProcessSetup(*particle, "process manager removed a different process");
        }
        auto* wrapper = new CarbonInelasticCaptureProcess(
            original, particle->GetParticleName());
        const auto added = manager->AddProcess(wrapper, at_rest, along, post);
        if (added < 0) {
            FatalProcessSetup(*particle, "failed to install capture wrapper");
        }
        manager->SetProcessActivation(wrapper, active);
        if (verbose_ > 0) {
            G4cout << "CINEL02 wrapped " << original->GetProcessName() << " for "
                   << particle->GetParticleName() << G4endl;
        }
    }
}
