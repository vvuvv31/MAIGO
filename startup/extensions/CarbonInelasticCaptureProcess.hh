#ifndef CarbonInelasticCaptureProcess_hh
#define CarbonInelasticCaptureProcess_hh

#include "G4WrapperProcess.hh"

#include "CarbonInelasticEventWriter.hh"
#include "CarbonInelasticTargetIdentity.hh"

class CarbonInelasticCaptureProcess final : public G4WrapperProcess {
public:
    CarbonInelasticCaptureProcess(G4VProcess* process, const G4String& projectile_name);
    ~CarbonInelasticCaptureProcess() override = default;

    G4double PostStepGetPhysicalInteractionLength(
        const G4Track&, G4double, G4ForceCondition*) override;
    G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;

    // The exposure scorer runs after PostStepDoIt, when G4Step's post point
    // may already describe the proposed final state. Expose the exact
    // pre-delegation track snapshot used by the CINEL02 writer so a collision
    // numerator can never condition on that post-reaction energy.
    const CarbonCinel02InputSnapshot* GetLastInputSnapshot() const noexcept {
        return last_input_valid_ ? &last_input_ : nullptr;
    }

    // G4HadronicProcess target state is mutable and may be cleared before the
    // exposure scorer runs.  Keep the value-resolved target from this exact
    // PostStepDoIt invocation so the scorer cannot observe a later state.
    const CarbonCinel02TargetIdentity* GetLastTargetIdentity() const noexcept {
        return last_target_valid_ ? &last_target_ : nullptr;
    }

private:
    G4String projectile_name_;
    CarbonCinel02InputSnapshot last_input_{};
    G4bool last_input_valid_{false};
    CarbonCinel02TargetIdentity last_target_{};
    G4bool last_target_valid_{false};
};

#endif
