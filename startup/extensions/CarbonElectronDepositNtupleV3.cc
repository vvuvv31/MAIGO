// Scorer for CarbonElectronDepositNtupleV3
// Schema v3: exact generating-step association, not last-track-state inference.
// Pre/post are Geant4 step states, NOT an assertion about internal process sampling energy.
#include "CarbonElectronDepositNtupleV3.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

namespace {
constexpr G4int kC12Pdg = 1000060120;
constexpr G4int kElectronPdg = 11;
constexpr G4int kSchemaVersion = 3;
}  // namespace

CarbonElectronDepositNtupleV3::CarbonElectronDepositNtupleV3(
    TsParameterManager* pm, TsMaterialManager* mm, TsGeometryManager* gm,
    TsScoringManager* sm, TsExtensionManager* em, G4String name,
    G4String quantity, G4String output, G4bool sub)
    : TsVNtupleScorer(pm, mm, gm, sm, em, name, quantity, output, sub) {
    fNtuple->RegisterColumnI(&run_, "run");
    fNtuple->RegisterColumnI(&event_, "event");
    fNtuple->RegisterColumnI(&track_, "track");
    fNtuple->RegisterColumnI(&parent_, "parent");
    fNtuple->RegisterColumnI(&pdg_, "pdg");
    fNtuple->RegisterColumnI(&step_, "step");
    fNtuple->RegisterColumnD(&birth_x_, "birth_x_mm", "");
    fNtuple->RegisterColumnD(&birth_y_, "birth_y_mm", "");
    fNtuple->RegisterColumnD(&birth_z_, "birth_z_mm", "");
    fNtuple->RegisterColumnD(&birth_ke_, "birth_ke_MeV", "");
    fNtuple->RegisterColumnD(&pre_x_, "pre_x_mm", "");
    fNtuple->RegisterColumnD(&pre_y_, "pre_y_mm", "");
    fNtuple->RegisterColumnD(&pre_z_, "pre_z_mm", "");
    fNtuple->RegisterColumnD(&post_x_, "post_x_mm", "");
    fNtuple->RegisterColumnD(&post_y_, "post_y_mm", "");
    fNtuple->RegisterColumnD(&post_z_, "post_z_mm", "");
    fNtuple->RegisterColumnD(&edep_, "edep_MeV", "");
    fNtuple->RegisterColumnD(&pre_ke_, "pre_ke_MeV", "");
    fNtuple->RegisterColumnD(&post_ke_, "post_ke_MeV", "");
    fNtuple->RegisterColumnD(&weight_, "weight", "");
    fNtuple->RegisterColumnD(&density_, "density_g_cm3", "");
    // v2 additions. Column order is frozen: old 21-column files must never be
    // silently parsed as v2; the analyzer keys on column names/count.
    fNtuple->RegisterColumnI(&schema_version_, "schema_version");
    fNtuple->RegisterColumnI(&parent_valid_, "parent_valid");
    fNtuple->RegisterColumnD(&parent_ke_, "parent_ke_MeV", "");
    fNtuple->RegisterColumnD(&parent_dir_x_, "parent_dir_x", "");
    fNtuple->RegisterColumnD(&parent_dir_y_, "parent_dir_y", "");
    fNtuple->RegisterColumnD(&parent_dir_z_, "parent_dir_z", "");
    fNtuple->RegisterColumnI(&creator_process_id_, "creator_process_id");
    fNtuple->RegisterColumnD(&birth_density_, "birth_density_g_cm3", "");
    fNtuple->RegisterColumnI(&parent_step_, "parent_step_id");
    fNtuple->RegisterColumnD(&parent_post_ke_, "parent_post_ke_MeV", "");
    fNtuple->RegisterColumnD(&parent_post_dx_, "parent_post_dir_x", "");
    fNtuple->RegisterColumnD(&parent_post_dy_, "parent_post_dir_y", "");
    fNtuple->RegisterColumnD(&parent_post_dz_, "parent_post_dir_z", "");
}

G4int CarbonElectronDepositNtupleV3::CreatorProcessId(const G4String& name, bool is_primary) {
    if (is_primary) return 0;
    if (name == "ionIoni") return 1;
    if (name == "eIoni") return 2;
    if (name == "compt") return 3;
    if (name == "conv") return 4;
    if (name == "phot") return 5;
    if (name == "eBrem") return 6;
    if (name == "CoulombScat") return 7;
    if (name == "msc") return 8;
    return 9;
}

G4bool CarbonElectronDepositNtupleV3::ProcessHits(G4Step* s, G4TouchableHistory*) {
    if (!fIsActive) { ++fSkippedWhileInactive; return false; }
    const auto* t = s->GetTrack();
    // Keep all tracks and zero-deposit steps: ancestor links and kinetic
    // energy leaving the slab must not disappear from the diagnostic.
    run_ = GetRunID(); event_ = GetEventID();
    track_ = t->GetTrackID(); parent_ = t->GetParentID();
    pdg_ = t->GetDefinition()->GetPDGEncoding();
    step_ = t->GetCurrentStepNumber();
    const auto b = t->GetVertexPosition();
    const auto p = s->GetPreStepPoint()->GetPosition();
    const auto q = s->GetPostStepPoint()->GetPosition();
    birth_x_ = b.x()/mm; birth_y_ = b.y()/mm; birth_z_ = b.z()/mm;
    birth_ke_ = t->GetVertexKineticEnergy()/MeV;
    pre_x_ = p.x()/mm; pre_y_ = p.y()/mm; pre_z_ = p.z()/mm;
    post_x_ = q.x()/mm; post_y_ = q.y()/mm; post_z_ = q.z()/mm;
    edep_ = s->GetTotalEnergyDeposit()/MeV;
    pre_ke_ = s->GetPreStepPoint()->GetKineticEnergy()/MeV;
    post_ke_ = s->GetPostStepPoint()->GetKineticEnergy()/MeV;
    weight_ = s->GetPreStepPoint()->GetWeight();
    density_ = s->GetPreStepPoint()->GetMaterial()->GetDensity()/(g/cm3);

    schema_version_ = kSchemaVersion;
    const std::pair<G4int, G4int> event_key(run_, event_);
    if (event_key != active_event_) {
        pending_.clear();
        track_state_.clear();
        birth_density_cache_.clear();
        active_event_ = event_key;
    }
    // Child track IDs are not assigned here. The G4Track pointer is stable
    // while queued; consume it at the child's first step before reuse.
    if (pdg_ == kC12Pdg && parent_ == 0) {
        const auto pre_dir = s->GetPreStepPoint()->GetMomentumDirection();
        const auto post_dir = s->GetPostStepPoint()->GetMomentumDirection();
        const ParentState state{pre_ke_, pre_dir.x(), pre_dir.y(), pre_dir.z(),
            post_ke_, post_dir.x(), post_dir.y(), post_dir.z(), step_};
        const auto* children = s->GetSecondaryInCurrentStep();
        if (children) for (const auto* child : *children) {
            if (child->GetDefinition()->GetPDGEncoding() == kElectronPdg)
                pending_[child] = state;
        }
    }
    if (step_ == 1) {
        birth_density_cache_[track_] = density_;
        const auto it = pending_.find(t);
        if (it != pending_.end()) {
            track_state_[track_] = it->second;
            pending_.erase(it);
        }
    }
    const auto birth_it = birth_density_cache_.find(track_);
    birth_density_ = birth_it == birth_density_cache_.end() ? density_ : birth_it->second;
    const auto* creator = t->GetCreatorProcess();
    creator_process_id_ = CreatorProcessId(
        creator ? creator->GetProcessName() : G4String(""), parent_ == 0);
    parent_valid_ = 0;
    parent_step_ = -1;
    parent_ke_ = parent_post_ke_ = -1;
    parent_dir_x_ = parent_dir_y_ = parent_dir_z_ = 0;
    parent_post_dx_ = parent_post_dy_ = parent_post_dz_ = 0;
    const auto it = track_state_.find(track_);
    if (it != track_state_.end()) {
        const auto& state = it->second;
        parent_valid_ = 1;
        parent_step_ = state.step;
        parent_ke_ = state.ke;
        parent_dir_x_ = state.dx; parent_dir_y_ = state.dy; parent_dir_z_ = state.dz;
        parent_post_ke_ = state.post_ke;
        parent_post_dx_ = state.post_dx;
        parent_post_dy_ = state.post_dy;
        parent_post_dz_ = state.post_dz;
    }
    fNtuple->Fill();
    return true;
}
