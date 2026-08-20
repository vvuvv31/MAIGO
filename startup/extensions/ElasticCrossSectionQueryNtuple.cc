// Scorer for ElasticCrossSectionQueryNtuple

#include "ElasticCrossSectionQueryNtuple.hh"

#include "G4Exception.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <cmath>
#include <cstdint>

namespace {

G4double ReadRequiredEnergyParameter(TsParameterManager* parameter_manager,
                                     const G4String& parameter_name,
                                     const char* label) {
    if (!parameter_manager->ParameterExists(parameter_name)) {
        G4ExceptionDescription description;
        description << "ElasticCrossSectionQueryNtuple requires " << label
                    << " parameter " << parameter_name;
        G4Exception("ElasticCrossSectionQueryNtuple", "MissingGridParameter",
                    FatalException, description);
        return 0.0;
    }
    return parameter_manager->GetDoubleParameter(parameter_name, "Energy") / MeV;
}

}  // namespace

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
    energy_min_mev_ = ReadRequiredEnergyParameter(
        parameter_manager, GetFullParmName("EnergyMin"), "EnergyMin");
    energy_max_mev_ = ReadRequiredEnergyParameter(
        parameter_manager, GetFullParmName("EnergyMax"), "EnergyMax");
    energy_step_mev_ = ReadRequiredEnergyParameter(
        parameter_manager, GetFullParmName("EnergyStep"), "EnergyStep");
    const auto span = energy_max_mev_ - energy_min_mev_;
    const auto count = span / energy_step_mev_;
    if (!std::isfinite(energy_min_mev_) || !std::isfinite(energy_max_mev_) ||
        !std::isfinite(energy_step_mev_) || energy_min_mev_ < 0.0 ||
        energy_max_mev_ < energy_min_mev_ || energy_step_mev_ <= 0.0 ||
        !std::isfinite(count) || count < 0.0 ||
        std::abs(count - std::round(count)) > 1.0e-8 || count > 1000000.0) {
        G4ExceptionDescription description;
        description << "ElasticCrossSectionQueryNtuple requires a finite, non-negative "
                    << "uniform EnergyMin/EnergyMax grid with positive EnergyStep; got "
                    << energy_min_mev_ << ", " << energy_max_mev_ << ", "
                    << energy_step_mev_;
        G4Exception("ElasticCrossSectionQueryNtuple", "InvalidGridParameter",
                    FatalException, description);
        return;
    }
    energy_count_ = static_cast<G4int>(std::llround(count)) + 1;
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
    auto* store = G4HadronicProcessStore::Instance();
    for (G4int index = 0; index < energy_count_; ++index) {
        const auto energy = index + 1 == energy_count_
                                ? energy_max_mev_
                                : energy_min_mev_ + index * energy_step_mev_;
        const auto macro = store->GetElasticCrossSectionPerVolume(
            projectile, energy * mass_number * MeV, material);
        energy_mev_per_u_ = static_cast<G4float>(energy);
        macroscopic_elastic_per_mm_ = macro / (1.0 / mm);
        fNtuple->Fill();
    }
    filled_ = true;
    return true;
}
