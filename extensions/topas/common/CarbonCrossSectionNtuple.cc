// Scorer for CarbonCrossSectionNtuple

#include "CarbonCrossSectionNtuple.hh"

#include "G4Element.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <cmath>

CarbonCrossSectionNtuple::CarbonCrossSectionNtuple(
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
    fNtuple->RegisterColumnD(&sigma_h_barn_, "Projectile-H Inelastic Cross Section (barn)", "");
    fNtuple->RegisterColumnD(&sigma_o_barn_, "Projectile-O Inelastic Cross Section (barn)", "");
    fNtuple->RegisterColumnD(&macro_h_per_mm_, "Hydrogen Macroscopic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnD(&macro_o_per_mm_, "Oxygen Macroscopic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnD(&macro_water_per_mm_, "Water Macroscopic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnD(&mean_free_path_mm_, "Water Mean Free Path (mm)", "");
}

CarbonCrossSectionNtuple::~CarbonCrossSectionNtuple() = default;

G4bool CarbonCrossSectionNtuple::ProcessHits(G4Step* step, G4TouchableHistory*) {
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
    const G4ElementVector* elements = material->GetElementVector();
    const G4double* atom_densities = material->GetVecNbOfAtomsPerVolume();
    G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();

    for (G4int energy_index = -8; energy_index <= 400; ++energy_index) {
        const G4double energy_per_u = (energy_index <= 0 ? 0.1 * (energy_index + 9) : static_cast<G4double>(energy_index)) * MeV;
        const G4double total_energy = static_cast<G4double>(mass_number) * energy_per_u;
        G4double sigma_h = 0.0;
        G4double sigma_o = 0.0;
        G4double macro_h = 0.0;
        G4double macro_o = 0.0;

        for (std::size_t element_index = 0; element_index < material->GetNumberOfElements(); ++element_index) {
            const G4Element* element = (*elements)[element_index];
            const G4double microscopic = store->GetInelasticCrossSectionPerAtom(
                projectile, total_energy, element, material);
            const G4int atomic_number = static_cast<G4int>(std::lround(element->GetZ()));
            if (atomic_number == 1) {
                sigma_h = microscopic;
                macro_h = microscopic * atom_densities[element_index];
            } else if (atomic_number == 8) {
                sigma_o = microscopic;
                macro_o = microscopic * atom_densities[element_index];
            }
        }

        const G4double macro_water = store->GetInelasticCrossSectionPerVolume(
            projectile, total_energy, material);
        energy_mev_per_u_ = static_cast<G4float>(energy_per_u / MeV);
        total_energy_mev_ = static_cast<G4float>(total_energy / MeV);
        sigma_h_barn_ = sigma_h / barn;
        sigma_o_barn_ = sigma_o / barn;
        macro_h_per_mm_ = macro_h / (1.0 / mm);
        macro_o_per_mm_ = macro_o / (1.0 / mm);
        macro_water_per_mm_ = macro_water / (1.0 / mm);
        mean_free_path_mm_ = macro_water > 0.0 ? (1.0 / macro_water) / mm : 0.0;
        fNtuple->Fill();
    }

    filled_ = true;
    return true;
}
