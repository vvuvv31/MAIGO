// Scorer for PrimaryCrossingCount

#include "PrimaryCrossingCount.hh"

#include "G4Exception.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4Track.hh"

namespace {

G4int ReadRequiredInteger(
    TsParameterManager* parameter_manager,
    const G4String& parameter_name,
    const char* label) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "PrimaryCrossingCount requires parameter "
                    << parameter_name << " (" << label << ").";
        G4Exception("PrimaryCrossingCount", "MissingProjectileParameter",
                    FatalException, description);
        return 0;
    }
    return parameter_manager->GetIntegerParameter(parameter_name);
}

}  // namespace

PrimaryCrossingCount::PrimaryCrossingCount(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVBinnedScorer(
          parameter_manager, material_manager, geometry_manager,
          scoring_manager, extension_manager, scorer_name, quantity,
          output_file, is_sub_scorer),
      projectile_z_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileZ"), "ProjectileZ")),
      projectile_a_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileA"), "ProjectileA")) {
    if (projectile_z_ <= 0 || projectile_a_ <= 0 ||
        projectile_a_ < projectile_z_) {
        G4ExceptionDescription description;
        description << "ProjectileZ and ProjectileA must satisfy "
                    << "Z > 0 and A >= Z; received Z=" << projectile_z_
                    << ", A=" << projectile_a_ << ".";
        G4Exception("PrimaryCrossingCount", "InvalidProjectileIdentity",
                    FatalException, description);
    }
    SetUnit("");
}

PrimaryCrossingCount::~PrimaryCrossingCount() = default;

G4bool PrimaryCrossingCount::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (step == nullptr || step->GetTrack() == nullptr) {
        return false;
    }

    // Scorer instances are worker-local in TOPAS MT.  The run/event guard is
    // retained in addition to the end-of-event hook so a reused event number
    // from a new run cannot inherit a worker's previous deduplication state.
    const G4int run_id = GetRunID();
    const G4int event_id = GetEventID();
    if (run_id != seen_run_id_ || event_id != seen_event_id_) {
        seen_bin_indices_.clear();
        seen_run_id_ = run_id;
        seen_event_id_ = event_id;
    }

    const G4Track* track = step->GetTrack();
    const G4ParticleDefinition* definition = track->GetDefinition();
    if (track->GetParentID() != 0 || definition == nullptr ||
        definition->GetAtomicNumber() != projectile_z_ ||
        definition->GetAtomicMass() != projectile_a_) {
        return false;
    }

    // TsVGeometryComponent::GetIndex is the same combined component index
    // used by TsVBinnedScorer.  For a 1x1x1400, 0.5 mm phantom this maps to
    // the z-depth division index without a separate coordinate convention.
    const G4int index = fComponent->GetIndex(step);
    if (index < 0 || !seen_bin_indices_.insert(index).second) {
        return false;
    }

    // This is a history crossing count, not a fluence estimate: every event
    // contributes exactly one unit to a bin, independent of track weight.
    AccumulateHit(step, 1.0, index);
    return true;
}

void PrimaryCrossingCount::UserHookForEndOfEvent() {
    seen_bin_indices_.clear();
    seen_run_id_ = -1;
    seen_event_id_ = -1;
}

void PrimaryCrossingCount::Clear() {
    TsVBinnedScorer::Clear();
    seen_bin_indices_.clear();
    seen_run_id_ = -1;
    seen_event_id_ = -1;
}
