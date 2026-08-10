// Scorer for CarbonNeutralNtuple

#include "CarbonNeutralNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include <limits>

namespace {

G4bool IsSupportedNeutral(const G4ParticleDefinition* definition) {
    const auto pdg = definition->GetPDGEncoding();
    return pdg == 22 || pdg == 2112;
}

G4bool IsNeutralInteraction(const G4ParticleDefinition* definition,
                            const G4VProcess* process) {
    if (process == nullptr || !IsSupportedNeutral(definition)) {
        return false;
    }
    if (definition->GetPDGEncoding() == 2112) {
        return process->GetProcessType() == fHadronic;
    }
    return process->GetProcessType() == fElectromagnetic;
}

G4double MacroscopicTotal(const G4ParticleDefinition* definition,
                          const G4double energy,
                          const G4Material* material) {
    if (definition->GetPDGEncoding() == 2112) {
        auto* store = G4HadronicProcessStore::Instance();
        return store->GetElasticCrossSectionPerVolume(definition, energy, material) +
               store->GetInelasticCrossSectionPerVolume(definition, energy, material) +
               store->GetCaptureCrossSectionPerVolume(definition, energy, material);
    }
    static thread_local G4EmCalculator calculator;
    G4double result = 0.0;
    for (const G4String process : {"phot", "compt", "conv", "Rayl"}) {
        result += calculator.ComputeCrossSectionPerVolume(
            energy, definition->GetParticleName(), process, material->GetName());
    }
    return result;
}

}  // namespace

CarbonNeutralNtuple::CarbonNeutralNtuple(
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
    fNtuple->RegisterColumnI(&interaction_id_, "Interaction Sequence ID");
    fNtuple->RegisterColumnI(&interaction_track_id_, "Interaction Track ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent Track ID");
    fNtuple->RegisterColumnI(&projectile_pdg_id_, "Projectile PDG ID");
    fNtuple->RegisterColumnI(&pdg_id_, "Particle PDG ID");
    fNtuple->RegisterColumnS(&particle_name_, "Particle Name");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnF(&charge_e_, "Charge (e)", "");
    fNtuple->RegisterColumnF(&incident_energy_mev_, "Incident Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&kinetic_energy_mev_, "Particle Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&local_deposit_mev_, "Local Energy Deposit (MeV)", "");
    fNtuple->RegisterColumnD(&macroscopic_total_per_mm_,
                             "Macroscopic Total Cross Section (1/mm)", "");
    fNtuple->RegisterColumnF(&vertex_x_mm_, "Vertex X (mm)", "");
    fNtuple->RegisterColumnF(&vertex_y_mm_, "Vertex Y (mm)", "");
    fNtuple->RegisterColumnF(&vertex_z_mm_, "Vertex Z (mm)", "");
    fNtuple->RegisterColumnF(&incident_direction_x_, "Incident Direction X", "");
    fNtuple->RegisterColumnF(&incident_direction_y_, "Incident Direction Y", "");
    fNtuple->RegisterColumnF(&incident_direction_z_, "Incident Direction Z", "");
    fNtuple->RegisterColumnF(&direction_x_, "Direction X", "");
    fNtuple->RegisterColumnF(&direction_y_, "Direction Y", "");
    fNtuple->RegisterColumnF(&direction_z_, "Direction Z", "");
    fNtuple->RegisterColumnS(&process_name_, "Process Name");
    fNtuple->RegisterColumnI(&process_type_, "Process Type");
    fNtuple->RegisterColumnI(&process_subtype_, "Process Subtype");
    fNtuple->RegisterColumnI(&creator_model_id_, "Creator Model ID");
}

CarbonNeutralNtuple::~CarbonNeutralNtuple() = default;

G4bool CarbonNeutralNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    const auto event_id = GetEventID();
    if (event_id != cached_event_id_) {
        interactions_.clear();
        cached_event_id_ = event_id;
        next_interaction_id_ = 0;
    }

    const auto* track = step->GetTrack();
    const auto* definition = track->GetDefinition();
    const auto* post_process =
        step->GetPostStepPoint()->GetProcessDefinedStep();
    InteractionContext context;
    G4ThreeVector vertex;
    G4ThreeVector direction;

    if (IsNeutralInteraction(definition, post_process)) {
        record_kind_ = "interaction";
        interaction_id_ = next_interaction_id_++;
        interaction_track_id_ = track->GetTrackID();
        vertex = step->GetPostStepPoint()->GetPosition();
        direction = step->GetPostStepPoint()->GetMomentumDirection();
        const auto incident_energy = step->GetPreStepPoint()->GetKineticEnergy();
        const auto* material = step->GetPreStepPoint()->GetMaterial();
        const auto macroscopic = MacroscopicTotal(
            definition, incident_energy, material);
        context = InteractionContext{
            interaction_id_,
            definition->GetPDGEncoding(),
            static_cast<G4float>(incident_energy / MeV),
            static_cast<G4float>(macroscopic / (1.0 / mm)),
            vertex,
            step->GetPreStepPoint()->GetMomentumDirection(),
            post_process->GetProcessName(),
            post_process->GetProcessType(),
            post_process->GetProcessSubType()};
        interactions_[interaction_track_id_].push_back(context);
        kinetic_energy_mev_ = static_cast<G4float>(
            step->GetPostStepPoint()->GetKineticEnergy() / MeV);
        local_deposit_mev_ =
            static_cast<G4float>(step->GetTotalEnergyDeposit() / MeV);
        creator_model_id_ = -1;
    } else {
        const auto* creator = track->GetCreatorProcess();
        if (track->GetCurrentStepNumber() != 1 || creator == nullptr) {
            return false;
        }
        const auto found = interactions_.find(track->GetParentID());
        if (found == interactions_.end()) {
            return false;
        }
        vertex = track->GetVertexPosition();
        auto best_distance = std::numeric_limits<G4double>::max();
        const InteractionContext* best = nullptr;
        for (const auto& candidate : found->second) {
            const auto distance = (candidate.vertex - vertex).mag2();
            if (candidate.process_name == creator->GetProcessName() &&
                distance < best_distance) {
                best_distance = distance;
                best = &candidate;
            }
        }
        if (best == nullptr || best_distance > 1.0e-6 * mm * mm) {
            return false;
        }
        context = *best;
        record_kind_ = "product";
        interaction_id_ = context.interaction_id;
        interaction_track_id_ = track->GetParentID();
        direction = track->GetVertexMomentumDirection();
        kinetic_energy_mev_ =
            static_cast<G4float>(track->GetVertexKineticEnergy() / MeV);
        local_deposit_mev_ = 0.0F;
        creator_model_id_ = track->GetCreatorModelID();
    }

    run_id_ = GetRunID();
    thread_id_ = G4Threading::G4GetThreadId();
    event_id_ = event_id;
    track_id_ = track->GetTrackID();
    parent_id_ = track->GetParentID();
    projectile_pdg_id_ = context.projectile_pdg;
    pdg_id_ = definition->GetPDGEncoding();
    particle_name_ = definition->GetParticleName();
    atomic_number_ = definition->GetAtomicNumber();
    atomic_mass_ = definition->GetAtomicMass();
    charge_e_ =
        static_cast<G4float>(track->GetDynamicParticle()->GetCharge() / eplus);
    incident_energy_mev_ = context.incident_energy_mev;
    macroscopic_total_per_mm_ = context.macroscopic_total_per_mm;
    vertex_x_mm_ = static_cast<G4float>(vertex.x() / mm);
    vertex_y_mm_ = static_cast<G4float>(vertex.y() / mm);
    vertex_z_mm_ = static_cast<G4float>(vertex.z() / mm);
    incident_direction_x_ =
        static_cast<G4float>(context.incident_direction.x());
    incident_direction_y_ =
        static_cast<G4float>(context.incident_direction.y());
    incident_direction_z_ =
        static_cast<G4float>(context.incident_direction.z());
    direction_x_ = static_cast<G4float>(direction.x());
    direction_y_ = static_cast<G4float>(direction.y());
    direction_z_ = static_cast<G4float>(direction.z());
    process_name_ = context.process_name;
    process_type_ = context.process_type;
    process_subtype_ = context.process_subtype;
    fNtuple->Fill();
    return true;
}
