// Scorer for CarbonLossRangeNtuple

// Dumps the ionIoni loss tables
// (range / inverse-range / restricted dE/dx) for the scoring volume's
// ACTUAL material-cuts couple. This is the reference behind
// G4VMscModel::GetRange / GetEnergy / GetDEDX as used by
// G4UrbanMscModel::ComputeTruePathLengthLimit and SampleScattering.

// Scorer for CarbonLossRangeNtuple

#include "CarbonLossRangeNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4IonisParamMat.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleDefinition.hh"
#include "G4Region.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"

CarbonLossRangeNtuple::CarbonLossRangeNtuple(
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
    fNtuple->RegisterColumnD(&loss_range_mm_, "Loss Range (mm)", "");
    fNtuple->RegisterColumnD(&csda_range_mm_, "CSDA Range (mm)", "");
    fNtuple->RegisterColumnD(
        &restricted_dedx_mev_per_mm_, "Restricted dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &inverse_residual_mev_, "Inverse Roundtrip Residual (MeV)", "");
}

CarbonLossRangeNtuple::~CarbonLossRangeNtuple() = default;

G4bool CarbonLossRangeNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }

    const G4ParticleDefinition* projectile = step->GetTrack()->GetDefinition();
    const G4int mass_number = projectile->GetAtomicMass();
    if (projectile->GetAtomicNumber() <= 0 || mass_number <= 0) {
        return false;
    }

    const G4Material* material = step->GetPreStepPoint()->GetMaterial();
    // Material identity for the metadata record (Zeffenters the Urban msc
    // cache; radlen enters theta0). Printed once per run.
    if (!filled_) {
        G4cout << "CarbonLossRangeNtuple material=" << material->GetName()
               << " density_g_per_cm3=" << material->GetDensity() / (g / cm3)
               << " Zeff=" << material->GetIonisation()->GetZeffective()
               << " radlen_mm=" << material->GetRadlen() / mm
               << G4endl;
    }
    // nullptr region = default-region couple = the production 0.05 mm cuts
    // couple for the Cu box (same couple the GPU loss table must reproduce).
    const G4Region* region = nullptr;
    G4EmCalculator em_calculator;

    // Same grid convention as CarbonStoppingPowerNtuple (0.01 + 0.1*i MeV/u,
    // 4001 nodes to 400.01 MeV/u) so the loss table and the stopping CSV
    // share nodes. Covers the required 20-300 MeV/u plus low-energy tail
    // for GetEnergy(range-t) near stopping.
    for (G4int energy_index = 0; energy_index <= 4000; ++energy_index) {
        const G4double energy_per_u =
            (0.01 + 0.1 * static_cast<G4double>(energy_index)) * MeV;
        const G4double total_energy = static_cast<G4double>(mass_number) * energy_per_u;
        const G4double loss_range = em_calculator.GetRangeFromRestricteDEDX(
            total_energy, projectile, material, region);
        const G4double csda_range = em_calculator.GetCSDARange(
            total_energy, projectile, material, region);
        const G4double restricted =
            em_calculator.GetDEDX(total_energy, projectile, material, region);
        const G4double back_energy = em_calculator.GetKinEnergy(
            loss_range, projectile, material, region);

        energy_mev_per_u_ = static_cast<G4float>(energy_per_u / MeV);
        total_energy_mev_ = static_cast<G4float>(total_energy / MeV);
        loss_range_mm_ = loss_range / mm;
        csda_range_mm_ = csda_range / mm;
        restricted_dedx_mev_per_mm_ = restricted / (MeV / mm);
        inverse_residual_mev_ = (back_energy - total_energy) / MeV;
        fNtuple->Fill();
    }

    filled_ = true;
    return true;
}
