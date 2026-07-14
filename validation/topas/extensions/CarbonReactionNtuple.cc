// Scorer for CarbonReactionNtuple

#include "CarbonReactionNtuple.hh"

#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

CarbonReactionNtuple::CarbonReactionNtuple(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVNtupleScorer(
          parameter_manager,
          material_manager,
          geometry_manager,
          scoring_manager,
          extension_manager,
          scorer_name,
          quantity,
          output_file,
          is_sub_scorer) {
    fNtuple->RegisterColumnS(&record_kind_, "Record Kind");
    fNtuple->RegisterColumnI(&run_id_, "Run ID");
    fNtuple->RegisterColumnI(&event_id_, "Event ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent Track ID");
    fNtuple->RegisterColumnI(&pdg_id_, "Particle PDG ID");
    fNtuple->RegisterColumnS(&particle_name_, "Particle Name");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnF(&charge_e_, "Charge (e)", "");
    fNtuple->RegisterColumnF(&kinetic_energy_mev_, "Particle Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&incident_energy_mev_, "Incident C12 Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&vertex_x_mm_, "Vertex X (mm)", "");
    fNtuple->RegisterColumnF(&vertex_y_mm_, "Vertex Y (mm)", "");
    fNtuple->RegisterColumnF(&vertex_z_mm_, "Vertex Z (mm)", "");
    fNtuple->RegisterColumnF(&direction_x_, "Initial Direction X", "");
    fNtuple->RegisterColumnF(&direction_y_, "Initial Direction Y", "");
    fNtuple->RegisterColumnF(&direction_z_, "Initial Direction Z", "");
    fNtuple->RegisterColumnF(&weight_, "Weight", "");
    fNtuple->RegisterColumnS(&creator_process_, "Creator Process");
    fNtuple->RegisterColumnI(&process_type_, "Creator Process Type");
    fNtuple->RegisterColumnI(&process_subtype_, "Creator Process Subtype");
    fNtuple->RegisterColumnI(&creator_model_id_, "Creator Model ID");
}

CarbonReactionNtuple::~CarbonReactionNtuple() = default;

G4bool CarbonReactionNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }

    const G4Track* track = step->GetTrack();
    const G4ParticleDefinition* definition = track->GetDefinition();
    const G4VProcess* process = nullptr;
    G4ThreeVector vertex;
    G4ThreeVector direction;

    if (track->GetTrackID() == 1 && track->GetParentID() == 0) {
        process = step->GetPostStepPoint()->GetProcessDefinedStep();
        if (process == nullptr || process->GetProcessType() != fHadronic ||
            process->GetProcessName().find("Inelastic") == G4String::npos) {
            return false;
        }
        record_kind_ = "reaction";
        vertex = step->GetPostStepPoint()->GetPosition();
        direction = step->GetPreStepPoint()->GetMomentumDirection();
        kinetic_energy_mev_ = static_cast<G4float>(step->GetPreStepPoint()->GetKineticEnergy() / MeV);
        incident_energy_mev_ = kinetic_energy_mev_;
        cached_reaction_event_id_ = GetEventID();
        cached_incident_energy_mev_ = incident_energy_mev_;
        creator_model_id_ = -1;
    } else {
        if (track->GetCurrentStepNumber() != 1 || track->GetParentID() != 1) {
            return false;
        }
        process = track->GetCreatorProcess();
        if (process == nullptr || process->GetProcessType() != fHadronic ||
            process->GetProcessName().find("Inelastic") == G4String::npos) {
            return false;
        }
        record_kind_ = "secondary";
        vertex = track->GetVertexPosition();
        direction = track->GetVertexMomentumDirection();
        kinetic_energy_mev_ = static_cast<G4float>(track->GetVertexKineticEnergy() / MeV);
        incident_energy_mev_ = cached_reaction_event_id_ == GetEventID()
                                   ? cached_incident_energy_mev_
                                   : 0.0F;
        creator_model_id_ = track->GetCreatorModelID();
    }

    run_id_ = GetRunID();
    event_id_ = GetEventID();
    track_id_ = track->GetTrackID();
    parent_id_ = track->GetParentID();
    pdg_id_ = definition->GetPDGEncoding();
    particle_name_ = definition->GetParticleName();
    atomic_number_ = definition->GetAtomicNumber();
    atomic_mass_ = definition->GetAtomicMass();
    charge_e_ = static_cast<G4float>(track->GetDynamicParticle()->GetCharge() / eplus);
    vertex_x_mm_ = static_cast<G4float>(vertex.x() / mm);
    vertex_y_mm_ = static_cast<G4float>(vertex.y() / mm);
    vertex_z_mm_ = static_cast<G4float>(vertex.z() / mm);
    direction_x_ = static_cast<G4float>(direction.x());
    direction_y_ = static_cast<G4float>(direction.y());
    direction_z_ = static_cast<G4float>(direction.z());
    weight_ = static_cast<G4float>(track->GetWeight());
    creator_process_ = process->GetProcessName();
    process_type_ = process->GetProcessType();
    process_subtype_ = process->GetProcessSubType();

    fNtuple->Fill();
    return true;
}
