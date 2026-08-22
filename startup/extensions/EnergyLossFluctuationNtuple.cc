// Scorer for EnergyLossFluctuationNtuple

#include "EnergyLossFluctuationNtuple.hh"

#include "G4Exception.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4StepStatus.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"

namespace {

G4int ReadRequiredInteger(
    TsParameterManager* parameter_manager,
    const G4String& parameter_name,
    const char* label) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "EnergyLossFluctuationNtuple requires parameter "
                    << parameter_name << " (" << label << ").";
        G4Exception("EnergyLossFluctuationNtuple",
                    "MissingProjectileParameter", FatalException,
                    description);
        return 0;
    }
    return parameter_manager->GetIntegerParameter(parameter_name);
}

}  // namespace

EnergyLossFluctuationNtuple::EnergyLossFluctuationNtuple(
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
          parameter_manager, material_manager, geometry_manager,
          scoring_manager, extension_manager, scorer_name, quantity,
          output_file, is_sub_scorer),
      configured_z_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileZ"), "ProjectileZ")),
      configured_a_(ReadRequiredInteger(
          parameter_manager, GetFullParmName("ProjectileA"), "ProjectileA")) {
    if (configured_z_ <= 0 || configured_a_ < configured_z_) {
        G4ExceptionDescription description;
        description << "ProjectileZ/A must satisfy Z > 0 and A >= Z; received Z="
                    << configured_z_ << ", A=" << configured_a_ << ".";
        G4Exception("EnergyLossFluctuationNtuple",
                    "InvalidProjectileIdentity", FatalException,
                    description);
    }

    fNtuple->RegisterColumnI(&run_id_, "Run ID");
    fNtuple->RegisterColumnI(&event_id_, "Event ID");
    fNtuple->RegisterColumnI(&thread_id_, "Thread ID");
    fNtuple->RegisterColumnI(&primary_track_id_, "Primary Track ID");
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&mass_number_, "Mass Number A");
    fNtuple->RegisterColumnS(&material_name_, "Material Name");
    fNtuple->RegisterColumnD(&entry_energy_mev_, "Entry Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(&exit_energy_mev_, "Exit Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &kinetic_energy_loss_mev_, "Primary Kinetic Energy Loss (MeV)", "");
    fNtuple->RegisterColumnD(
        &primary_local_deposit_mev_, "Primary Local Deposit (MeV)", "");
    fNtuple->RegisterColumnD(&path_length_mm_, "Primary Path Length (mm)", "");
    fNtuple->RegisterColumnI(&step_count_, "Primary Step Count");
    fNtuple->RegisterColumnB(&material_consistent_, "Material Consistent");
    fNtuple->RegisterColumnB(&completed_, "Completed");
    fNtuple->RegisterColumnS(&completion_status_, "Completion Status");
}

EnergyLossFluctuationNtuple::~EnergyLossFluctuationNtuple() = default;

G4bool EnergyLossFluctuationNtuple::ProcessHits(
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

    const auto* track = step->GetTrack();
    const auto* definition = track->GetDefinition();
    if (track->GetParentID() != 0 || definition == nullptr ||
        definition->GetAtomicNumber() != configured_z_ ||
        definition->GetAtomicMass() != configured_a_) {
        return false;
    }

    // A fluctuation package requires exactly one selected primary per event.
    // Ignore any additional source primaries so they cannot be silently folded
    // into the same sample; the exporter template emits one primary/history.
    if (saw_primary_ && track->GetTrackID() != primary_track_id_) {
        return false;
    }

    const auto* material = step->GetPreStepPoint()->GetMaterial();
    const G4String step_material =
        material != nullptr ? material->GetName() : G4String("<null>");
    if (!saw_primary_) {
        saw_primary_ = true;
        run_id_ = GetRunID();
        event_id_ = GetEventID();
        thread_id_ = G4Threading::G4GetThreadId();
        primary_track_id_ = track->GetTrackID();
        atomic_number_ = definition->GetAtomicNumber();
        mass_number_ = definition->GetAtomicMass();
        material_name_ = step_material;
        entry_energy_mev_ = step->GetPreStepPoint()->GetKineticEnergy() / MeV;
    } else if (step_material != material_name_) {
        material_consistent_ = false;
    }

    exit_energy_mev_ = step->GetPostStepPoint()->GetKineticEnergy() / MeV;
    primary_local_deposit_mev_ += step->GetTotalEnergyDeposit() / MeV;
    path_length_mm_ += step->GetStepLength() / mm;
    ++step_count_;

    const auto step_status = step->GetPostStepPoint()->GetStepStatus();
    if (step_status == fGeomBoundary || step_status == fWorldBoundary) {
        completed_ = true;
        completion_status_ = "exited";
    } else if (track->GetTrackStatus() == fStopAndKill ||
               track->GetTrackStatus() == fKillTrackAndSecondaries) {
        completed_ = true;
        completion_status_ = "stopped";
    } else {
        completed_ = false;
        completion_status_ = "incomplete";
    }
    return true;
}

void EnergyLossFluctuationNtuple::UserHookForEndOfEvent() {
    if (saw_primary_) {
        kinetic_energy_loss_mev_ = entry_energy_mev_ - exit_energy_mev_;
        fNtuple->Fill();
    }
    ResetEvent();
}

void EnergyLossFluctuationNtuple::ResetEvent() {
    run_id_ = -1;
    event_id_ = -1;
    thread_id_ = -1;
    primary_track_id_ = -1;
    atomic_number_ = 0;
    mass_number_ = 0;
    material_name_.clear();
    entry_energy_mev_ = 0.0;
    exit_energy_mev_ = 0.0;
    kinetic_energy_loss_mev_ = 0.0;
    primary_local_deposit_mev_ = 0.0;
    path_length_mm_ = 0.0;
    step_count_ = 0;
    material_consistent_ = true;
    completed_ = false;
    completion_status_.clear();
    saw_primary_ = false;
}
