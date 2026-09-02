// Scorer for CarbonLineageSurvivalNtuple

#include "CarbonLineageSurvivalNtuple.hh"

#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"

#include <utility>

namespace {

G4bool IsBoundary(const G4StepPoint* post) {
    if (post == nullptr) {
        return false;
    }
    const auto status = post->GetStepStatus();
    return status == fGeomBoundary || status == fWorldBoundary;
}

G4bool IsKilled(const G4Track* track) {
    if (track == nullptr) {
        return false;
    }
    const auto status = track->GetTrackStatus();
    return status == fStopAndKill || status == fKillTrackAndSecondaries;
}

}  // namespace

CarbonLineageSurvivalNtuple::CarbonLineageSurvivalNtuple(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVNtupleScorer(parameter_manager, material_manager, geometry_manager,
                      scoring_manager, extension_manager, scorer_name, quantity,
                      output_file, is_sub_scorer) {
    fNtuple->RegisterColumnS(&record_kind_, "Record Kind");
    fNtuple->RegisterColumnI(&run_id_, "Run ID");
    fNtuple->RegisterColumnI(&thread_id_, "Thread ID");
    fNtuple->RegisterColumnI(&event_id_, "Event ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent Track ID");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnI(&generation_, "Transport Generation");
    fNtuple->RegisterColumnI(&episode_index_, "Episode Index");
    fNtuple->RegisterColumnI(&step_count_, "Step Count");
    fNtuple->RegisterColumnI(&reacted_, "Hadronic Interaction");
    fNtuple->RegisterColumnI(&censored_, "Censored At Event End");
    fNtuple->RegisterColumnF(&weight_, "Weight", "");
    fNtuple->RegisterColumnD(&birth_kinetic_energy_mev_,
                             "Birth Kinetic Energy (MeV)", "MeV");
    fNtuple->RegisterColumnD(&terminal_kinetic_energy_mev_,
                             "Terminal Kinetic Energy (MeV)", "MeV");
    fNtuple->RegisterColumnD(&interaction_kinetic_energy_mev_,
                             "Interaction Kinetic Energy (MeV)", "MeV");
    fNtuple->RegisterColumnD(&path_length_mm_, "Path Length (mm)", "mm");
    fNtuple->RegisterColumnD(&birth_x_mm_, "Birth X (mm)", "mm");
    fNtuple->RegisterColumnD(&birth_y_mm_, "Birth Y (mm)", "mm");
    fNtuple->RegisterColumnD(&birth_z_mm_, "Birth Z (mm)", "mm");
    fNtuple->RegisterColumnD(&terminal_x_mm_, "Terminal X (mm)", "mm");
    fNtuple->RegisterColumnD(&terminal_y_mm_, "Terminal Y (mm)", "mm");
    fNtuple->RegisterColumnD(&terminal_z_mm_, "Terminal Z (mm)", "mm");
    fNtuple->RegisterColumnS(&creator_process_, "Creator Process");
    fNtuple->RegisterColumnS(&terminal_reason_, "Terminal Reason");
    fNtuple->RegisterColumnS(&terminal_process_, "Terminal Process");
}

CarbonLineageSurvivalNtuple::~CarbonLineageSurvivalNtuple() = default;

G4bool CarbonLineageSurvivalNtuple::IsTrackedIsotope(
    const G4ParticleDefinition* definition) {
    if (definition == nullptr) {
        return false;
    }
    const auto z = definition->GetAtomicNumber();
    const auto a = definition->GetAtomicMass();
    return (z == 3 && (a == 6 || a == 7)) ||
           (z == 4 && (a == 7 || a == 9 || a == 10));
}

G4bool CarbonLineageSurvivalNtuple::IsHadronicInelastic(
    const G4VProcess* process) {
    return process != nullptr && process->GetProcessType() == fHadronic &&
           process->GetProcessName().find("Inelastic") != G4String::npos;
}

void CarbonLineageSurvivalNtuple::ResetEvent(G4int run_id, G4int event_id) {
    episodes_.clear();
    generations_.clear();
    cached_run_id_ = run_id;
    cached_event_id_ = event_id;
}

void CarbonLineageSurvivalNtuple::EmitEpisode(
    const TrackLineage& episode,
    G4int track_id,
    G4double terminal_kinetic_energy_mev,
    G4double interaction_kinetic_energy_mev,
    G4int reacted,
    G4int censored,
    const G4String& terminal_reason,
    const G4String& terminal_process,
    const G4ThreeVector& terminal_position) {
    record_kind_ = "episode";
    run_id_ = GetRunID();
    thread_id_ = G4Threading::G4GetThreadId();
    event_id_ = GetEventID();
    track_id_ = track_id;
    parent_id_ = episode.parent_id;
    atomic_number_ = episode.atomic_number;
    atomic_mass_ = episode.atomic_mass;
    generation_ = episode.generation;
    episode_index_ = episode.episode_index;
    step_count_ = episode.step_count;
    reacted_ = reacted;
    censored_ = censored;
    weight_ = static_cast<G4float>(episode.weight);
    birth_kinetic_energy_mev_ = episode.birth_kinetic_energy_mev;
    terminal_kinetic_energy_mev_ = terminal_kinetic_energy_mev;
    interaction_kinetic_energy_mev_ = interaction_kinetic_energy_mev;
    path_length_mm_ = episode.path_length_mm;
    birth_x_mm_ = episode.birth_position.x() / mm;
    birth_y_mm_ = episode.birth_position.y() / mm;
    birth_z_mm_ = episode.birth_position.z() / mm;
    terminal_x_mm_ = terminal_position.x() / mm;
    terminal_y_mm_ = terminal_position.y() / mm;
    terminal_z_mm_ = terminal_position.z() / mm;
    creator_process_ = episode.creator_process;
    terminal_reason_ = terminal_reason;
    terminal_process_ = terminal_process;
    fNtuple->Fill();
}

G4bool CarbonLineageSurvivalNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (step == nullptr || step->GetTrack() == nullptr ||
        step->GetPreStepPoint() == nullptr ||
        step->GetPostStepPoint() == nullptr) {
        return false;
    }

    const auto run_id = GetRunID();
    const auto event_id = GetEventID();
    if (run_id != cached_run_id_ || event_id != cached_event_id_) {
        ResetEvent(run_id, event_id);
    }

    const auto* track = step->GetTrack();
    const auto* definition = track->GetDefinition();
    const auto* creator = track->GetCreatorProcess();
    const auto* post_process =
        step->GetPostStepPoint()->GetProcessDefinedStep();
    const auto hadronic_creator = IsHadronicInelastic(creator);

    // Keep a compact ancestry map for all tracks seen in this component.  A
    // primary is generation -1, so its direct hadronic products are G0.
    if (track->GetCurrentStepNumber() == 1 &&
        generations_.find(track->GetTrackID()) == generations_.end()) {
        G4int generation = 0;
        if (track->GetParentID() == 0) {
            generation = -1;
        } else {
            const auto parent = generations_.find(track->GetParentID());
            if (parent != generations_.end()) {
                generation = parent->second + (hadronic_creator ? 1 : 0);
            } else if (hadronic_creator) {
                generation = 0;
            }
        }
        generations_.emplace(track->GetTrackID(), generation);
    }

    if (!IsTrackedIsotope(definition)) {
        return false;
    }

    auto iterator = episodes_.find(track->GetTrackID());
    if (iterator == episodes_.end()) {
        // Only products of a hadronic cascade are lineage births.  This
        // excludes an accidentally configured Li/Be primary from the report.
        if (!hadronic_creator || track->GetCurrentStepNumber() != 1) {
            return false;
        }
        TrackLineage episode;
        const auto generation = generations_.find(track->GetTrackID());
        episode.generation = generation != generations_.end() ? generation->second : 0;
        episode.atomic_number = definition->GetAtomicNumber();
        episode.atomic_mass = definition->GetAtomicMass();
        episode.parent_id = track->GetParentID();
        episode.birth_kinetic_energy_mev = track->GetVertexKineticEnergy() / MeV;
        episode.birth_position = track->GetVertexPosition();
        episode.birth_direction = track->GetVertexMomentumDirection();
        episode.last_position = episode.birth_position;
        episode.last_kinetic_energy_mev = episode.birth_kinetic_energy_mev;
        episode.weight = track->GetWeight();
        episode.creator_process = creator->GetProcessName();
        iterator = episodes_.emplace(track->GetTrackID(), std::move(episode)).first;
    }

    auto& episode = iterator->second;
    episode.path_length_mm += step->GetStepLength() / mm;
    ++episode.step_count;
    episode.last_position = step->GetPostStepPoint()->GetPosition();
    episode.last_kinetic_energy_mev =
        step->GetPostStepPoint()->GetKineticEnergy() / MeV;

    const auto* post = step->GetPostStepPoint();
    const auto process_name = post_process != nullptr
                                  ? post_process->GetProcessName()
                                  : G4String("none");
    if (IsHadronicInelastic(post_process)) {
        EmitEpisode(episode, track->GetTrackID(),
                    post->GetKineticEnergy() / MeV,
                    step->GetPreStepPoint()->GetKineticEnergy() / MeV, 1, 0,
                    "hadronic_interaction", process_name, post->GetPosition());

        // Geant4 can retain the parent track after an ion-inelastic process.
        // Start a new episode at its post-step state so continuation survival
        // is represented without counting the interaction twice.
        if (!IsKilled(track) && post->GetKineticEnergy() > 0.0 &&
            IsTrackedIsotope(definition)) {
            episode.episode_index += 1;
            episode.birth_kinetic_energy_mev = post->GetKineticEnergy() / MeV;
            episode.birth_position = post->GetPosition();
            episode.birth_direction = post->GetMomentumDirection();
            episode.last_position = episode.birth_position;
            episode.last_kinetic_energy_mev = episode.birth_kinetic_energy_mev;
            episode.path_length_mm = 0.0;
            episode.step_count = 0;
            episode.creator_process = "primary_continuation";
            return true;
        }
        episodes_.erase(iterator);
        return true;
    }

    if (post_process != nullptr && post_process->GetProcessType() == fDecay) {
        EmitEpisode(episode, track->GetTrackID(),
                    post->GetKineticEnergy() / MeV, 0.0, 0, 0, "decay",
                    process_name, post->GetPosition());
        episodes_.erase(iterator);
        return true;
    }

    if (IsBoundary(post)) {
        EmitEpisode(episode, track->GetTrackID(),
                    post->GetKineticEnergy() / MeV, 0.0, 0, 0,
                    "boundary_escape", process_name, post->GetPosition());
        episodes_.erase(iterator);
        return true;
    }

    if (IsKilled(track) || post->GetKineticEnergy() <= 0.0) {
        EmitEpisode(episode, track->GetTrackID(),
                    post->GetKineticEnergy() / MeV, 0.0, 0, 0, "stopped",
                    process_name, post->GetPosition());
        episodes_.erase(iterator);
        return true;
    }

    return true;
}

void CarbonLineageSurvivalNtuple::UserHookForEndOfEvent() {
    // Tracks normally terminate before the event hook.  Any remaining
    // episode is a right-censored sample (for example, an event abort or a
    // scorer-volume boundary not observed by this component).
    for (const auto& [track_id, episode] : episodes_) {
        EmitEpisode(episode, track_id, episode.last_kinetic_energy_mev, 0.0, 0,
                    1, "event_end_censored", "none", episode.last_position);
    }
    episodes_.clear();
    generations_.clear();
}
