// Scorer for CarbonElectronDepositNtuple
// Record schema v2 (see CarbonElectronDepositNtuple.hh): v1-compatible physics,
// 21 v1 columns filled exactly as before plus 8 v2 birth-conditioning columns.
// Only observation/recording changed; no physics process is modified.
#include "CarbonElectronDepositNtuple.hh"
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
constexpr G4int kSchemaVersion = 2;
}  // namespace

CarbonElectronDepositNtuple::CarbonElectronDepositNtuple(
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
}

G4int CarbonElectronDepositNtuple::CreatorProcessId(const G4String& name, bool is_primary) {
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

G4bool CarbonElectronDepositNtuple::ProcessHits(G4Step* s, G4TouchableHistory*) {
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

    // ---- v2 birth conditioning (observation only) ----
    schema_version_ = kSchemaVersion;
    const std::pair<G4int, G4int> event_key(run_, event_);
    const std::tuple<G4int, G4int, G4int> track_key(run_, event_, track_);
    // Cache the C12 pre-step state whenever a C12 crosses this scorer.
    if (pdg_ == kC12Pdg) {
        const G4ThreeVector dir = s->GetPreStepPoint()->GetMomentumDirection();
        parent_cache_[event_key] = ParentState{
            s->GetPreStepPoint()->GetKineticEnergy()/MeV, dir.x(), dir.y(), dir.z()};
    }
    // Birth density: material density at the track's first step in this
    // scorer; reused verbatim on later steps of the same track.
    if (step_ == 1) {
        birth_density_cache_[track_key] = density_;
    }
    const auto birth_density_it = birth_density_cache_.find(track_key);
    birth_density_ = (birth_density_it != birth_density_cache_.end())
        ? birth_density_it->second : density_;
    // Creator process of this track (0 for primaries).
    const G4VProcess* creator = t->GetCreatorProcess();
    creator_process_id_ = CreatorProcessId(
        creator ? creator->GetProcessName() : G4String(""), parent_ == 0);
    // Parent state for electron tracks: snapshot the same-event last-known
    // C12 state at the track's first step, then replay it verbatim on later
    // steps so per-track fields stay consistent. valid==0 rows carry the
    // -1/0 sentinel and must not be used for projection.
    parent_valid_ = 0;
    parent_ke_ = -1.0;
    parent_dir_x_ = parent_dir_y_ = parent_dir_z_ = 0.0;
    if (pdg_ == kElectronPdg) {
        if (step_ == 1) {
            const auto it = parent_cache_.find(event_key);
            if (it != parent_cache_.end()) {
                track_parent_valid_[track_key] = 1;
                track_parent_cache_[track_key] = it->second;
            } else {
                track_parent_valid_[track_key] = 0;
                track_parent_cache_[track_key] = ParentState{-1.0, 0.0, 0.0, 0.0};
            }
        }
        const auto valid_it = track_parent_valid_.find(track_key);
        const auto state_it = track_parent_cache_.find(track_key);
        if (valid_it != track_parent_valid_.end() && state_it != track_parent_cache_.end()) {
            parent_valid_ = valid_it->second;
            parent_ke_ = state_it->second.ke;
            parent_dir_x_ = state_it->second.dir_x;
            parent_dir_y_ = state_it->second.dir_y;
            parent_dir_z_ = state_it->second.dir_z;
        }
    }
    fNtuple->Fill();
    return true;
}
