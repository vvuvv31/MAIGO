// Scorer for CarbonStoppingPowerNtuple

#include "CarbonStoppingPowerNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

CarbonStoppingPowerNtuple::CarbonStoppingPowerNtuple(
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
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "Energy (MeV/u)", "");
    fNtuple->RegisterColumnF(&total_energy_mev_, "Total Kinetic Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &electronic_dedx_mev_per_mm_, "Electronic dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(&total_dedx_mev_per_mm_, "Total dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(&csda_range_mm_, "CSDA Range (mm)", "");
}

CarbonStoppingPowerNtuple::~CarbonStoppingPowerNtuple() = default;

G4bool CarbonStoppingPowerNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }

    const G4ParticleDefinition* carbon = step->GetTrack()->GetDefinition();
    if (carbon->GetAtomicNumber() != 6 || carbon->GetAtomicMass() != 12) {
        return false;
    }

    const G4Material* material = step->GetPreStepPoint()->GetMaterial();
    G4EmCalculator em_calculator;

    // Match the production GPU water table exactly. The low-energy offset
    // avoids querying zero while the 0.1 MeV/u interval resolves the steep
    // stopping-power rise near the end of range.
    for (G4int energy_index = 0; energy_index <= 4000; ++energy_index) {
        const G4double energy_per_u =
            (0.01 + 0.1 * static_cast<G4double>(energy_index)) * MeV;
        const G4double total_energy = 12.0 * energy_per_u;
        const G4double electronic = em_calculator.ComputeElectronicDEDX(
            total_energy, carbon, material);
        const G4double total = em_calculator.ComputeTotalDEDX(
            total_energy, carbon, material);
        // CSDA tables are disabled by the reference physics list. Querying
        // them emits one warning per grid point and does not affect the GPU
        // transport, which integrates dE/dx directly.
        const G4double range = 0.0;

        energy_mev_per_u_ = static_cast<G4float>(energy_per_u / MeV);
        total_energy_mev_ = static_cast<G4float>(total_energy / MeV);
        electronic_dedx_mev_per_mm_ = electronic / (MeV / mm);
        total_dedx_mev_per_mm_ = total / (MeV / mm);
        csda_range_mm_ = range / mm;
        fNtuple->Fill();
    }

    filled_ = true;
    return true;
}
