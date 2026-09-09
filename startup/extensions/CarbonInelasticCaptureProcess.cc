// Extra Class for CarbonInelasticCaptureProcess

#include "CarbonInelasticCaptureProcess.hh"

#include "CarbonInelasticEventWriter.hh"

#include "G4Exception.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessType.hh"
#include "G4ParticleChange.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace {

bool IsChargedIonProjectile(const G4Track& track) {
    const auto* definition = track.GetDefinition();
    const auto* dynamic = track.GetDynamicParticle();
    return definition != nullptr && dynamic != nullptr &&
           definition->GetAtomicNumber() > 0 &&
           definition->GetAtomicMass() > 0 &&
           dynamic->GetCharge() != 0.0;
}

bool CapturePrimaryOnly() {
    const auto* value = std::getenv("CARBON_CINEL02_PRIMARY_ONLY");
    return value != nullptr &&
           (std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "yes") == 0);
}

bool IsUnchangedNoOp(const G4Track& track, const G4ParticleChange& change) {
    // G4HadronicProcess may be selected at the end of a step when its
    // integral cross-section correction rejects the proposed interaction.
    // In that case PostStepDoIt returns the initialized, unchanged particle
    // change and the process has not sampled a target or a final state.  It
    // must not be serialized as a CINEL02 collision.
    if (change.GetTrackStatus() != fAlive ||
        change.GetNumberOfSecondaries() != 0 ||
        change.GetLocalEnergyDeposit() != 0.0 ||
        change.GetNonIonizingEnergyDeposit() != 0.0) {
        return false;
    }
    const auto* direction = change.GetMomentumDirection();
    const auto* position = change.GetPosition();
    if (direction == nullptr || position == nullptr) {
        return false;
    }
    constexpr G4double kEnergyTolerance = 1.0e-12;
    constexpr G4double kVectorTolerance = 1.0e-12;
    const auto energy_scale = std::max(1.0, std::abs(track.GetKineticEnergy()));
    const auto unchanged_energy =
        std::abs(change.GetEnergy() - track.GetKineticEnergy()) <=
        kEnergyTolerance * energy_scale;
    const auto unchanged_direction =
        (direction->unit() - track.GetMomentumDirection().unit()).mag2() <=
        kVectorTolerance * kVectorTolerance;
    const auto unchanged_position =
        position->x() == track.GetPosition().x() &&
        position->y() == track.GetPosition().y() &&
        position->z() == track.GetPosition().z();
    return unchanged_energy && unchanged_direction && unchanged_position;
}

}  // namespace

G4double CarbonInelasticCaptureProcess::PostStepGetPhysicalInteractionLength(
    const G4Track& track, const G4double previous_step_size,
    G4ForceCondition* condition) {
    // Capture scope alone is insufficient: a secondary must not be allowed to
    // reach the wrapped process and then have its event discarded in
    // PostStepDoIt.  Returning an infinite post-step interaction length gates
    // the process before Geant4's step competition, while EM and elastic
    // processes remain active on the same track.
    if (CapturePrimaryOnly() && track.GetParentID() != 0) {
        if (condition != nullptr) {
            *condition = NotForced;
        }
        return DBL_MAX;
    }
    return G4WrapperProcess::PostStepGetPhysicalInteractionLength(
        track, previous_step_size, condition);
}

CarbonInelasticCaptureProcess::CarbonInelasticCaptureProcess(
    G4VProcess* process, const G4String& projectile_name)
    : G4WrapperProcess("CarbonCINEL02", fHadronic),
      projectile_name_(projectile_name) {
    SetProcessSubType(fHadronInelastic);
    if (process == nullptr || process->GetProcessType() != fHadronic ||
        process->GetProcessSubType() != fHadronInelastic) {
        G4ExceptionDescription description;
        description << "CarbonInelasticCaptureProcess can only wrap one "
                       "fHadronic/fHadronInelastic process for "
                    << projectile_name_;
        G4Exception("CarbonInelasticCaptureProcess", "CINEL02-PROCESS", FatalException,
                    description);
    }
    RegisterProcess(process);
}

G4VParticleChange* CarbonInelasticCaptureProcess::PostStepDoIt(
    const G4Track& track, const G4Step& step) {
    // A process-defined step is not authoritative until this invocation has
    // captured its input and the delegated process has produced a real
    // change.  Clear the previous invocation first so a terminal/no-op step
    // can never reuse the preceding collision snapshot.
    last_input_valid_ = false;
    last_target_valid_ = false;
    // A CINEL02 package for a source projectile must keep each captured
    // event indivisible, but it must not silently turn secondary light-hadron
    // interactions into source-projectile package cells.  Those secondary
    // processes also do not consistently expose a selected target isotope in
    // Geant4's mixed-material process state.  In primary-only campaigns the
    // wrapper remains transparent for all secondaries and captures only the
    // source track that the exposure scorer qualifies.
    if (CapturePrimaryOnly() && track.GetParentID() != 0) {
        return G4WrapperProcess::PostStepDoIt(track, step);
    }
    if (!IsChargedIonProjectile(track)) {
        return G4WrapperProcess::PostStepDoIt(track, step);
    }
    // Geant4 can still ask an ion-inelastic process to define a terminal step
    // after continuous slowing has reduced a primary ion to exactly zero
    // kinetic energy.  That is not a physical C12 inelastic collision and it
    // has no valid CINEL02 collision-energy snapshot.  Keep the delegate
    // transparent and let the exposure scorer treat this as a non-collision
    // terminal step instead of serializing a zero-energy event.
    if (track.GetKineticEnergy() <= 0.0) {
        return G4WrapperProcess::PostStepDoIt(track, step);
    }
    // This snapshot is made before delegation. AlongStepDoIt has already
    // completed, while the wrapped inelastic model has not yet run.
    const auto input = CarbonInelasticEventWriter::CaptureInput(track, step);
    last_input_ = input;
    last_input_valid_ = true;
    auto* change = G4WrapperProcess::PostStepDoIt(track, step);
    if (change == nullptr) {
        G4ExceptionDescription description;
        description << "Wrapped ion-inelastic process returned a null "
                       "G4VParticleChange for "
                    << projectile_name_;
        G4Exception("CarbonInelasticCaptureProcess", "CINEL02-CHANGE", FatalException,
                    description);
    }
    // The hadronic process is expected to return G4ParticleChange. Using a
    // step scorer or a delayed secondary callback would lose the atomic
    // collision contract, so fail closed if this invariant is not met.
    const auto* particle_change = dynamic_cast<const G4ParticleChange*>(change);
    auto* registered = const_cast<G4VProcess*>(GetRegisteredProcess());
    auto* hadronic = dynamic_cast<G4HadronicProcess*>(registered);
    if (particle_change == nullptr || hadronic == nullptr) {
        G4ExceptionDescription description;
        description << "CINEL02 requires a G4ParticleChange from the wrapped "
                       "hadronic process for "
                    << projectile_name_;
        G4Exception("CarbonInelasticCaptureProcess", "CINEL02-CONTRACT", FatalException,
                    description);
    }
    const bool unchanged_noop = IsUnchangedNoOp(track, *particle_change);
    // Opt-in observation only: include rejected candidates that raw capture
    // intentionally omits. No target/model query, random draw or state update.
    const auto* audit = std::getenv("CARBON_HE4_CANDIDATE_AUDIT");
    if (audit != nullptr && std::strcmp(audit, "1") == 0 &&
        track.GetDefinition()->GetAtomicNumber() == 2 &&
        track.GetDefinition()->GetAtomicMass() == 4) {
        std::ostringstream line;
        line << std::setprecision(17) << "HE4_CANDIDATE "
             << input.event_id << ' ' << track.GetTrackID() << ' '
             << input.collision_energy_MeV << ' ' << unchanged_noop;
        G4cout << line.str() << G4endl;
    }
    if (unchanged_noop) {
        // Keep the wrapper transparent and tell the exposure scorer that this
        // process-defined step did not contain an authoritative collision.
        last_input_valid_ = false;
        return change;
    }
    last_target_ = ResolveCarbonCinel02Target(hadronic);
    last_target_valid_ = last_target_.z > 0 &&
                         last_target_.a >= last_target_.z &&
                         !last_target_.name.empty();
    // Append performs only serialization and metadata capture. It never calls
    // a Geant4 process and never draws a random number. The returned object is
    // passed through unchanged to the SteppingManager.
    CarbonInelasticEventWriter::Instance().Append(
        input, *particle_change, *GetRegisteredProcess(), hadronic);
    return change;
}
