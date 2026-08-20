// Scorer for ElasticCrossSectionQueryNtuple

#include "ElasticCrossSectionQueryNtuple.hh"

#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <array>

ElasticCrossSectionQueryNtuple::ElasticCrossSectionQueryNtuple(
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
                      scoring_manager, extension_manager, scorer_name,
                      quantity, output_file, is_sub_scorer) {
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "Energy (MeV/u)", "");
    fNtuple->RegisterColumnD(&macroscopic_elastic_per_mm_,
                             "Macroscopic Elastic Cross Section (1/mm)", "");
}

ElasticCrossSectionQueryNtuple::~ElasticCrossSectionQueryNtuple() = default;

G4bool ElasticCrossSectionQueryNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step == nullptr || step->GetTrack() == nullptr ||
        step->GetTrack()->GetParentID() != 0) {
        return false;
    }
    const auto* projectile = step->GetTrack()->GetDefinition();
    const auto mass_number = projectile->GetAtomicMass();
    const auto* material = step->GetPreStepPoint()->GetMaterial();
    if (projectile->GetAtomicNumber() <= 0 || mass_number <= 0 || material == nullptr) {
        return false;
    }
    constexpr std::array<G4double, 10> energies_mev_per_u{
        0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.1, 1.25, 1.5};
    auto* store = G4HadronicProcessStore::Instance();
    for (const auto energy : energies_mev_per_u) {
        const auto macro = store->GetElasticCrossSectionPerVolume(
            projectile, energy * mass_number * MeV, material);
        energy_mev_per_u_ = static_cast<G4float>(energy);
        macroscopic_elastic_per_mm_ = macro / (1.0 / mm);
        fNtuple->Fill();
    }
    filled_ = true;
    return true;
}
