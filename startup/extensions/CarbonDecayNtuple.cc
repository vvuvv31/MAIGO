// Scorer for CarbonDecayNtuple

#include "CarbonDecayNtuple.hh"

#include "G4Ions.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include <algorithm>

CarbonDecayNtuple::CarbonDecayNtuple(
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
    fNtuple->RegisterColumnI(&pdg_id_, "Particle PDG ID");
    fNtuple->RegisterColumnS(&particle_name_, "Particle Name");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&atomic_mass_, "Atomic Mass A");
    fNtuple->RegisterColumnF(&charge_e_, "Charge (e)", "");
    fNtuple->RegisterColumnF(&rest_mass_MeV_, "Rest Mass (MeV)", "");
    fNtuple->RegisterColumnF(&excitation_MeV_, "Excitation Energy (MeV)", "");
    fNtuple->RegisterColumnF(&kinetic_energy_MeV_, "Vertex Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&pre_kinetic_energy_MeV_, "Pre-Step Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&post_kinetic_energy_MeV_, "Post-Step Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnF(&step_deposit_MeV_, "Step Energy Deposit (MeV)", "");
    fNtuple->RegisterColumnF(&step_length_mm_, "Step Length (mm)", "");
    fNtuple->RegisterColumnF(&direction_x_, "Vertex Direction X", "");
    fNtuple->RegisterColumnF(&direction_y_, "Vertex Direction Y", "");
    fNtuple->RegisterColumnF(&direction_z_, "Vertex Direction Z", "");
    fNtuple->RegisterColumnF(&vertex_x_mm_, "Vertex X (mm)", "");
    fNtuple->RegisterColumnF(&vertex_y_mm_, "Vertex Y (mm)", "");
    fNtuple->RegisterColumnF(&vertex_z_mm_, "Vertex Z (mm)", "");
    fNtuple->RegisterColumnF(&global_time_ns_, "Global Time (ns)", "");
    fNtuple->RegisterColumnF(&proper_time_ns_, "Proper Time (ns)", "");
    fNtuple->RegisterColumnS(&creator_process_, "Creator Process");
    fNtuple->RegisterColumnS(&post_process_, "Post-Step Process");
    fNtuple->RegisterColumnI(&creator_process_type_, "Creator Process Type");
    fNtuple->RegisterColumnI(&creator_process_subtype_, "Creator Process Subtype");
    fNtuple->RegisterColumnI(&creator_model_id_, "Creator Model ID");
    fNtuple->RegisterColumnI(&track_status_, "Track Status");
}

CarbonDecayNtuple::~CarbonDecayNtuple() = default;

G4bool CarbonDecayNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (step == nullptr || step->GetTrack() == nullptr) return false;
    const auto* track = step->GetTrack();
    const auto* creator = track->GetCreatorProcess();
    const auto* post_process = step->GetPostStepPoint() != nullptr
                                   ? step->GetPostStepPoint()->GetProcessDefinedStep()
                                   : nullptr;
    const auto* definition = track->GetDefinition();
    const auto* dynamic = track->GetDynamicParticle();
    if (definition == nullptr || dynamic == nullptr) return false;
    const bool is_decay_daughter = creator != nullptr &&
                                   creator->GetProcessType() == fDecay &&
                                   track->GetCurrentStepNumber() == 1;
    const bool is_be6_parent_step = definition->GetAtomicNumber() == 4 &&
                                    (definition->GetAtomicMass() == 6 ||
                                     definition->GetBaryonNumber() == 6) &&
                                    post_process != nullptr &&
                                    post_process->GetProcessType() == fDecay;
    if (!is_decay_daughter && !is_be6_parent_step) return false;

    record_kind_ = is_decay_daughter ? "decay_daughter" : "decay_parent";
    run_id_ = GetRunID();
    thread_id_ = G4Threading::G4GetThreadId();
    event_id_ = GetEventID();
    track_id_ = track->GetTrackID();
    parent_id_ = track->GetParentID();
    pdg_id_ = definition->GetPDGEncoding();
    particle_name_ = definition->GetParticleName();
    atomic_number_ = definition->GetAtomicNumber();
    atomic_mass_ = definition->GetAtomicMass() > 0
                       ? definition->GetAtomicMass()
                       : definition->GetBaryonNumber();
    charge_e_ = static_cast<G4float>(dynamic->GetCharge() / eplus);
    rest_mass_MeV_ = static_cast<G4float>(dynamic->GetMass() / MeV);
    excitation_MeV_ = 0.0F;
    if (const auto* ion = dynamic_cast<const G4Ions*>(definition); ion != nullptr) {
        excitation_MeV_ = static_cast<G4float>(ion->GetExcitationEnergy() / MeV);
    }
    kinetic_energy_MeV_ = static_cast<G4float>(track->GetVertexKineticEnergy() / MeV);
    pre_kinetic_energy_MeV_ = static_cast<G4float>(step->GetPreStepPoint()->GetKineticEnergy() / MeV);
    post_kinetic_energy_MeV_ = static_cast<G4float>(step->GetPostStepPoint()->GetKineticEnergy() / MeV);
    step_deposit_MeV_ = static_cast<G4float>(step->GetTotalEnergyDeposit() / MeV);
    step_length_mm_ = static_cast<G4float>(step->GetStepLength() / mm);
    const auto direction = track->GetVertexMomentumDirection();
    direction_x_ = static_cast<G4float>(direction.x());
    direction_y_ = static_cast<G4float>(direction.y());
    direction_z_ = static_cast<G4float>(direction.z());
    const auto vertex = track->GetVertexPosition();
    vertex_x_mm_ = static_cast<G4float>(vertex.x() / mm);
    vertex_y_mm_ = static_cast<G4float>(vertex.y() / mm);
    vertex_z_mm_ = static_cast<G4float>(vertex.z() / mm);
    global_time_ns_ = static_cast<G4float>(track->GetGlobalTime() / ns);
    proper_time_ns_ = static_cast<G4float>(track->GetProperTime() / ns);
    creator_process_ = creator != nullptr ? creator->GetProcessName() : "none";
    creator_process_type_ = creator != nullptr ? creator->GetProcessType() : 0;
    creator_process_subtype_ = creator != nullptr ? creator->GetProcessSubType() : 0;
    creator_model_id_ = track->GetCreatorModelID();
    post_process_ = post_process != nullptr ? post_process->GetProcessName() : "none";
    track_status_ = static_cast<G4int>(track->GetTrackStatus());
    fNtuple->Fill();
    return true;
}
