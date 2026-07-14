// Scorer for CarbonCascadeNtuple

#include "CarbonCascadeNtuple.hh"

#include "G4ParticleDefinition.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

namespace {

G4bool IsCascadeProcess(const G4VProcess* process) {
    if (process == nullptr) {
        return false;
    }
    if (process->GetProcessType() == fDecay) {
        return true;
    }
    return process->GetProcessType() == fHadronic &&
           process->GetProcessName().find("Inelastic") != G4String::npos;
}

}  // namespace

CarbonCascadeNtuple::CarbonCascadeNtuple(
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
    fNtuple->RegisterColumnI(&interaction_track_id_, "Interaction Track ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent Track ID");
    fNtuple->RegisterColumnI(&pdg_id_, "Particle PDG ID");
    fNtuple->RegisterColumnS(&particle_name_, "Particle Name");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnF(&charge_e_, "Charge (e)", "");
    fNtuple->RegisterColumnF(&kinetic_energy_mev_, "Particle Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&incident_energy_mev_, "Incident Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(&macroscopic_inelastic_per_mm_,
                             "Macroscopic Inelastic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnI(&projectile_z_, "Projectile Atomic Number Z");
    fNtuple->RegisterColumnI(&projectile_a_, "Projectile Atomic Mass A");
    fNtuple->RegisterColumnF(&vertex_x_mm_, "Vertex X (mm)", "");
    fNtuple->RegisterColumnF(&vertex_y_mm_, "Vertex Y (mm)", "");
    fNtuple->RegisterColumnF(&vertex_z_mm_, "Vertex Z (mm)", "");
    fNtuple->RegisterColumnF(&direction_x_, "Direction X", "");
    fNtuple->RegisterColumnF(&direction_y_, "Direction Y", "");
    fNtuple->RegisterColumnF(&direction_z_, "Direction Z", "");
    fNtuple->RegisterColumnF(&weight_, "Weight", "");
    fNtuple->RegisterColumnS(&process_name_, "Process Name");
    fNtuple->RegisterColumnI(&process_type_, "Process Type");
    fNtuple->RegisterColumnI(&process_subtype_, "Process Subtype");
    fNtuple->RegisterColumnI(&creator_model_id_, "Creator Model ID");
}

CarbonCascadeNtuple::~CarbonCascadeNtuple() = default;

G4bool CarbonCascadeNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }

    const G4int event_id = GetEventID();
    if (event_id != cached_event_id_) {
        interactions_.clear();
        cached_event_id_ = event_id;
    }

    const G4Track* track = step->GetTrack();
    const G4ParticleDefinition* definition = track->GetDefinition();
    const G4VProcess* process = nullptr;
    G4ThreeVector vertex;
    G4ThreeVector direction;
    InteractionContext context;

    const G4VProcess* post_process = step->GetPostStepPoint()->GetProcessDefinedStep();
    const G4bool charged_ion = definition->GetAtomicNumber() > 0 &&
                              definition->GetAtomicMass() > 0;
    if (charged_ion && IsCascadeProcess(post_process)) {
        record_kind_ = "interaction";
        process = post_process;
        interaction_track_id_ = track->GetTrackID();
        vertex = step->GetPostStepPoint()->GetPosition();
        direction = step->GetPreStepPoint()->GetMomentumDirection();
        kinetic_energy_mev_ =
            static_cast<G4float>(step->GetPreStepPoint()->GetKineticEnergy() / MeV);
        context = InteractionContext{kinetic_energy_mev_, definition->GetAtomicNumber(),
                                     definition->GetAtomicMass()};
        interactions_[interaction_track_id_] = context;
        creator_model_id_ = -1;
    } else {
        process = track->GetCreatorProcess();
        if (track->GetCurrentStepNumber() != 1 || !IsCascadeProcess(process)) {
            return false;
        }
        const auto parent = interactions_.find(track->GetParentID());
        if (parent == interactions_.end()) {
            return false;
        }
        record_kind_ = "product";
        interaction_track_id_ = track->GetParentID();
        vertex = track->GetVertexPosition();
        direction = track->GetVertexMomentumDirection();
        kinetic_energy_mev_ =
            static_cast<G4float>(track->GetVertexKineticEnergy() / MeV);
        context = parent->second;
        creator_model_id_ = track->GetCreatorModelID();
    }

    run_id_ = GetRunID();
    thread_id_ = G4Threading::G4GetThreadId();
    event_id_ = event_id;
    track_id_ = track->GetTrackID();
    parent_id_ = track->GetParentID();
    pdg_id_ = definition->GetPDGEncoding();
    particle_name_ = definition->GetParticleName();
    atomic_number_ = definition->GetAtomicNumber();
    atomic_mass_ = definition->GetAtomicMass();
    charge_e_ = static_cast<G4float>(track->GetDynamicParticle()->GetCharge() / eplus);
    incident_energy_mev_ = context.incident_energy_mev;
    macroscopic_inelastic_per_mm_ = 0.0;
    if (definition->GetAtomicNumber() > 0 && definition->GetAtomicMass() > 0 &&
        kinetic_energy_mev_ > 0.0F) {
        const G4Material* material = step->GetPreStepPoint()->GetMaterial();
        const G4double macroscopic =
            G4HadronicProcessStore::Instance()->GetInelasticCrossSectionPerVolume(
                definition, static_cast<G4double>(kinetic_energy_mev_) * MeV, material);
        macroscopic_inelastic_per_mm_ = macroscopic / (1.0 / mm);
    }
    projectile_z_ = context.projectile_z;
    projectile_a_ = context.projectile_a;
    vertex_x_mm_ = static_cast<G4float>(vertex.x() / mm);
    vertex_y_mm_ = static_cast<G4float>(vertex.y() / mm);
    vertex_z_mm_ = static_cast<G4float>(vertex.z() / mm);
    direction_x_ = static_cast<G4float>(direction.x());
    direction_y_ = static_cast<G4float>(direction.y());
    direction_z_ = static_cast<G4float>(direction.z());
    weight_ = static_cast<G4float>(track->GetWeight());
    process_name_ = process->GetProcessName();
    process_type_ = process->GetProcessType();
    process_subtype_ = process->GetProcessSubType();
    fNtuple->Fill();
    return true;
}
