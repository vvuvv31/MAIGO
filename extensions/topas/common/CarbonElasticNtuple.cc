// Scorer for CarbonElasticNtuple

#include "CarbonElasticNtuple.hh"

#include "G4Exception.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include <cmath>
#include <limits>

namespace {

G4int ReadProjectileParameter(TsParameterManager* manager,
                              const G4String& name,
                              const char* label) {
    if (!manager->ParameterExists(name)) {
        G4ExceptionDescription description;
        description << "CarbonElasticNtuple requires " << label << " parameter " << name;
        G4Exception("CarbonElasticNtuple", "MissingProjectileParameter",
                    FatalException, description);
        return 0;
    }
    return manager->GetIntegerParameter(name);
}

G4bool IsElastic(const G4VProcess* process) {
    return process != nullptr && process->GetProcessType() == fHadronic &&
           (process->GetProcessName() == "hadElastic" ||
            process->GetProcessName().find("Elastic") != G4String::npos);
}

G4ThreeVector UnitOrZero(const G4ThreeVector& direction) {
    const auto magnitude = direction.mag();
    return magnitude > 0.0 ? direction / magnitude : G4ThreeVector(0.0, 0.0, 1.0);
}

}  // namespace

CarbonElasticNtuple::CarbonElasticNtuple(
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
                      output_file, is_sub_scorer),
      projectile_z_(ReadProjectileParameter(
          parameter_manager, GetFullParmName("ProjectileZ"), "ProjectileZ")),
      projectile_a_(ReadProjectileParameter(
          parameter_manager, GetFullParmName("ProjectileA"), "ProjectileA")) {
    if (projectile_z_ <= 0 || projectile_a_ < projectile_z_) {
        G4ExceptionDescription description;
        description << "ProjectileZ/A must satisfy Z > 0 and A >= Z; received Z="
                    << projectile_z_ << ", A=" << projectile_a_;
        G4Exception("CarbonElasticNtuple", "InvalidProjectileIdentity",
                    FatalException, description);
    }
    fNtuple->RegisterColumnS(&record_kind_, "Record Kind");
    fNtuple->RegisterColumnI(&run_id_, "Run ID");
    fNtuple->RegisterColumnI(&thread_id_, "Thread ID");
    fNtuple->RegisterColumnI(&event_id_, "Event ID");
    fNtuple->RegisterColumnI(&interaction_id_, "Interaction ID");
    fNtuple->RegisterColumnI(&interaction_track_id_, "Interaction Track ID");
    fNtuple->RegisterColumnI(&track_id_, "Track ID");
    fNtuple->RegisterColumnI(&parent_id_, "Parent Track ID");
    fNtuple->RegisterColumnI(&projectile_pdg_id_, "Projectile PDG ID");
    fNtuple->RegisterColumnI(&pdg_id_, "Particle PDG ID");
    fNtuple->RegisterColumnS(&particle_name_, "Particle Name");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnI(&target_z_, "Target Atomic Number Z");
    fNtuple->RegisterColumnI(&target_a_, "Target Mass Number A");
    fNtuple->RegisterColumnF(&charge_e_, "Charge (e)", "");
    fNtuple->RegisterColumnF(&incident_energy_mev_per_u_, "Incident Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(&outgoing_projectile_energy_mev_per_u_, "Outgoing Projectile Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(&kinetic_energy_mev_, "Particle Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&local_deposit_mev_, "Local Energy Deposit (MeV)", "");
    fNtuple->RegisterColumnD(&macroscopic_elastic_per_mm_, "Macroscopic Elastic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnF(&vertex_x_mm_, "Vertex X (mm)", "");
    fNtuple->RegisterColumnF(&vertex_y_mm_, "Vertex Y (mm)", "");
    fNtuple->RegisterColumnF(&vertex_z_mm_, "Vertex Z (mm)", "");
    fNtuple->RegisterColumnF(&incident_direction_x_, "Incident Direction X", "");
    fNtuple->RegisterColumnF(&incident_direction_y_, "Incident Direction Y", "");
    fNtuple->RegisterColumnF(&incident_direction_z_, "Incident Direction Z", "");
    fNtuple->RegisterColumnF(&direction_x_, "Direction X", "");
    fNtuple->RegisterColumnF(&direction_y_, "Direction Y", "");
    fNtuple->RegisterColumnF(&direction_z_, "Direction Z", "");
    fNtuple->RegisterColumnI(&generation_, "Generation");
    fNtuple->RegisterColumnS(&transport_disposition_, "Transport Disposition");
    fNtuple->RegisterColumnS(&process_name_, "Process Name");
    fNtuple->RegisterColumnI(&process_type_, "Process Type");
    fNtuple->RegisterColumnI(&process_subtype_, "Process Subtype");
    fNtuple->RegisterColumnI(&creator_model_id_, "Creator Model ID");
}

CarbonElasticNtuple::~CarbonElasticNtuple() = default;

G4bool CarbonElasticNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive || step == nullptr || step->GetTrack() == nullptr) {
        if (!fIsActive) {
            ++fSkippedWhileInactive;
        }
        return false;
    }
    const G4int event_id = GetEventID();
    if (event_id != cached_event_id_) {
        interactions_.clear();
        cached_event_id_ = event_id;
        next_interaction_id_ = 0;
    }
    const auto* track = step->GetTrack();
    const auto* definition = track->GetDefinition();
    const auto* post_process = step->GetPostStepPoint()->GetProcessDefinedStep();
    const G4bool is_selected_projectile =
        definition != nullptr && definition->GetAtomicNumber() == projectile_z_ &&
        definition->GetAtomicMass() == projectile_a_;
    InteractionContext context;

    if (track->GetParentID() == 0 && is_selected_projectile && IsElastic(post_process)) {
        const auto* material = step->GetPreStepPoint()->GetMaterial();
        const auto incident_energy = step->GetPreStepPoint()->GetKineticEnergy();
        const auto outgoing_energy = step->GetPostStepPoint()->GetKineticEnergy();
        const auto incident_direction = UnitOrZero(step->GetPreStepPoint()->GetMomentumDirection());
        const auto outgoing_direction = UnitOrZero(step->GetPostStepPoint()->GetMomentumDirection());
        const auto* hadronic_process =
            dynamic_cast<const G4HadronicProcess*>(post_process);
        const auto* target_isotope =
            hadronic_process == nullptr
                ? nullptr
                : const_cast<G4HadronicProcess*>(hadronic_process)->GetTargetIsotope();
        const auto macro = G4HadronicProcessStore::Instance()->GetElasticCrossSectionPerVolume(
            definition, incident_energy, material);
        context.interaction_id = next_interaction_id_++;
        context.interaction_track_id = track->GetTrackID();
        context.projectile_pdg = definition->GetPDGEncoding();
        context.projectile_z = projectile_z_;
        context.projectile_a = projectile_a_;
        context.incident_energy_mev_per_u = static_cast<G4float>(incident_energy / MeV / projectile_a_);
        context.outgoing_energy_mev_per_u = static_cast<G4float>(outgoing_energy / MeV / projectile_a_);
        context.local_deposit_mev = static_cast<G4float>(step->GetTotalEnergyDeposit() / MeV);
        context.macroscopic_elastic_per_mm = macro / (1.0 / mm);
        context.incident_direction = incident_direction;
        context.outgoing_direction = outgoing_direction;
        context.vertex = step->GetPostStepPoint()->GetPosition();
        context.target_z = target_isotope == nullptr ? 0 : target_isotope->GetZ();
        // In Geant4, G4Isotope::GetA() is the atomic molar mass in internal
        // units; G4Isotope::GetN() is the integer nucleon/mass number used by
        // TOPAS isotope definitions (H1 -> 1, O16 -> 16).
        context.target_a = target_isotope == nullptr ? 0 : target_isotope->GetN();
        context.process_name = post_process->GetProcessName();
        context.process_type = post_process->GetProcessType();
        context.process_subtype = post_process->GetProcessSubType();

        record_kind_ = "interaction";
        interactions_[track->GetTrackID()].push_back(context);
        interaction_id_ = context.interaction_id;
        interaction_track_id_ = context.interaction_track_id;
        track_id_ = track->GetTrackID();
        parent_id_ = track->GetParentID();
        projectile_pdg_id_ = context.projectile_pdg;
        pdg_id_ = definition->GetPDGEncoding();
        particle_name_ = definition->GetParticleName();
        atomic_number_ = definition->GetAtomicNumber();
        atomic_mass_ = definition->GetAtomicMass();
        target_z_ = context.target_z;
        target_a_ = context.target_a;
        charge_e_ = static_cast<G4float>(track->GetDynamicParticle()->GetCharge() / eplus);
        incident_energy_mev_per_u_ = context.incident_energy_mev_per_u;
        outgoing_projectile_energy_mev_per_u_ = context.outgoing_energy_mev_per_u;
        kinetic_energy_mev_ = static_cast<G4float>(incident_energy / MeV);
        local_deposit_mev_ = context.local_deposit_mev;
        macroscopic_elastic_per_mm_ = context.macroscopic_elastic_per_mm;
        vertex_x_mm_ = static_cast<G4float>(context.vertex.x() / mm);
        vertex_y_mm_ = static_cast<G4float>(context.vertex.y() / mm);
        vertex_z_mm_ = static_cast<G4float>(context.vertex.z() / mm);
        incident_direction_x_ = static_cast<G4float>(incident_direction.x());
        incident_direction_y_ = static_cast<G4float>(incident_direction.y());
        incident_direction_z_ = static_cast<G4float>(incident_direction.z());
        direction_x_ = static_cast<G4float>(outgoing_direction.x());
        direction_y_ = static_cast<G4float>(outgoing_direction.y());
        direction_z_ = static_cast<G4float>(outgoing_direction.z());
        generation_ = 0;
        transport_disposition_ = outgoing_energy > 0.0 ? "continue" : "kill";
        process_name_ = context.process_name;
        process_type_ = context.process_type;
        process_subtype_ = context.process_subtype;
        creator_model_id_ = -1;
    } else {
        const auto* creator = track->GetCreatorProcess();
        if (track->GetCurrentStepNumber() != 1 || creator == nullptr || !IsElastic(creator)) {
            return false;
        }
        const auto found = interactions_.find(track->GetParentID());
        if (found == interactions_.end()) {
            return false;
        }
        auto best_distance = std::numeric_limits<G4double>::max();
        const InteractionContext* best = nullptr;
        const auto vertex = track->GetVertexPosition();
        for (const auto& candidate : found->second) {
            const auto distance = (candidate.vertex - vertex).mag2();
            if (candidate.process_name == creator->GetProcessName() && distance < best_distance) {
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
        interaction_track_id_ = context.interaction_track_id;
        track_id_ = track->GetTrackID();
        parent_id_ = track->GetParentID();
        projectile_pdg_id_ = context.projectile_pdg;
        pdg_id_ = definition->GetPDGEncoding();
        particle_name_ = definition->GetParticleName();
        atomic_number_ = definition->GetAtomicNumber();
        atomic_mass_ = definition->GetAtomicMass();
        target_z_ = context.target_z;
        target_a_ = context.target_a;
        charge_e_ = static_cast<G4float>(track->GetDynamicParticle()->GetCharge() / eplus);
        incident_energy_mev_per_u_ = context.incident_energy_mev_per_u;
        outgoing_projectile_energy_mev_per_u_ = context.outgoing_energy_mev_per_u;
        kinetic_energy_mev_ = static_cast<G4float>(track->GetVertexKineticEnergy() / MeV);
        local_deposit_mev_ = 0.0F;
        macroscopic_elastic_per_mm_ = 0.0;
        vertex_x_mm_ = static_cast<G4float>(vertex.x() / mm);
        vertex_y_mm_ = static_cast<G4float>(vertex.y() / mm);
        vertex_z_mm_ = static_cast<G4float>(vertex.z() / mm);
        const auto direction = UnitOrZero(track->GetVertexMomentumDirection());
        incident_direction_x_ = static_cast<G4float>(context.incident_direction.x());
        incident_direction_y_ = static_cast<G4float>(context.incident_direction.y());
        incident_direction_z_ = static_cast<G4float>(context.incident_direction.z());
        direction_x_ = static_cast<G4float>(direction.x());
        direction_y_ = static_cast<G4float>(direction.y());
        direction_z_ = static_cast<G4float>(direction.z());
        generation_ = context.generation + 1;
        transport_disposition_ = kinetic_energy_mev_ > 0.0F ? "queue" : "local_deposit";
        process_name_ = creator->GetProcessName();
        process_type_ = creator->GetProcessType();
        process_subtype_ = creator->GetProcessSubType();
        creator_model_id_ = track->GetCreatorModelID();
    }

    run_id_ = GetRunID();
    thread_id_ = G4Threading::G4GetThreadId();
    event_id_ = event_id;
    fNtuple->Fill();
    return true;
}
