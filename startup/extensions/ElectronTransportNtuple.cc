// Scorer for ElectronTransportNtuple

#include "ElectronTransportNtuple.hh"

#include "G4Electron.hh"
#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4Positron.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <cmath>

ElectronTransportNtuple::ElectronTransportNtuple(
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
          parameter_manager, material_manager, geometry_manager, scoring_manager,
          extension_manager, scorer_name, quantity, output_file, is_sub_scorer) {
    fNtuple->RegisterColumnD(&kinetic_energy_MeV_, "Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &electron_collisional_MeV_per_mm_,
        "Electron Collisional dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &electron_radiative_MeV_per_mm_,
        "Electron Radiative dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &electron_total_MeV_per_mm_, "Electron Total dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &positron_collisional_MeV_per_mm_,
        "Positron Collisional dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &positron_radiative_MeV_per_mm_,
        "Positron Radiative dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &positron_total_MeV_per_mm_, "Positron Total dE/dx (MeV/mm)", "");
}

ElectronTransportNtuple::~ElectronTransportNtuple() = default;

G4bool ElectronTransportNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) return false;

    const auto* material = step->GetPreStepPoint()->GetMaterial();
    const auto* electron = G4Electron::ElectronDefinition();
    const auto* positron = G4Positron::PositronDefinition();
    G4EmCalculator calculator;
    constexpr G4int grid_intervals = 1000;
    constexpr G4double minimum_energy_MeV = 0.001;
    constexpr G4double maximum_energy_MeV = 500.0;
    const auto log_ratio = std::log(maximum_energy_MeV / minimum_energy_MeV);
    for (G4int index = 0; index <= grid_intervals; ++index) {
        const auto fraction = static_cast<G4double>(index) / grid_intervals;
        const auto energy = minimum_energy_MeV * std::exp(fraction * log_ratio) * MeV;
        const auto electron_collision = calculator.ComputeDEDX(
            energy, electron, "eIoni", material);
        const auto electron_radiation = calculator.ComputeDEDX(
            energy, electron, "eBrem", material);
        const auto positron_collision = calculator.ComputeDEDX(
            energy, positron, "eIoni", material);
        const auto positron_radiation = calculator.ComputeDEDX(
            energy, positron, "eBrem", material);

        kinetic_energy_MeV_ = energy / MeV;
        electron_collisional_MeV_per_mm_ = electron_collision / (MeV / mm);
        electron_radiative_MeV_per_mm_ = electron_radiation / (MeV / mm);
        electron_total_MeV_per_mm_ =
            electron_collisional_MeV_per_mm_ + electron_radiative_MeV_per_mm_;
        positron_collisional_MeV_per_mm_ = positron_collision / (MeV / mm);
        positron_radiative_MeV_per_mm_ = positron_radiation / (MeV / mm);
        positron_total_MeV_per_mm_ =
            positron_collisional_MeV_per_mm_ + positron_radiative_MeV_per_mm_;
        fNtuple->Fill();
    }
    filled_ = true;
    return true;
}
