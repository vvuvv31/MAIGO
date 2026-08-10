// Scorer for NeutralCrossSectionNtuple

#include "NeutralCrossSectionNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <array>
#include <cmath>

namespace {

G4double MacroscopicTotal(
    const G4ParticleDefinition* definition,
    const G4double energy,
    const G4Material* material) {
    if (definition->GetPDGEncoding() == 2112) {
        auto* store = G4HadronicProcessStore::Instance();
        return store->GetElasticCrossSectionPerVolume(
                   definition, energy, material) +
               store->GetInelasticCrossSectionPerVolume(
                   definition, energy, material) +
               store->GetCaptureCrossSectionPerVolume(
                   definition, energy, material);
    }
    G4EmCalculator calculator;
    G4double result = 0.0;
    for (const G4String process : {"phot", "compt", "conv", "Rayl"}) {
        result += calculator.ComputeCrossSectionPerVolume(
            energy, definition->GetParticleName(), process,
            material->GetName());
    }
    return result;
}

}  // namespace

NeutralCrossSectionNtuple::NeutralCrossSectionNtuple(
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
          output_file, is_sub_scorer) {
    fNtuple->RegisterColumnI(&pdg_id_, "PDG ID");
    fNtuple->RegisterColumnF(&energy_mev_, "Energy (MeV)", "");
    fNtuple->RegisterColumnD(
        &macroscopic_total_per_mm_,
        "Macroscopic Total Cross Section (1/mm)", "");
    fNtuple->RegisterColumnD(
        &mean_free_path_mm_, "Mean Free Path (mm)", "");
}

NeutralCrossSectionNtuple::~NeutralCrossSectionNtuple() = default;

G4bool NeutralCrossSectionNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }

    constexpr G4int grid_size = 601;
    constexpr G4double minimum_energy_mev = 1.0e-4;
    constexpr G4double maximum_energy_mev = 1000.0;
    const auto log_minimum = std::log(minimum_energy_mev);
    const auto log_step =
        (std::log(maximum_energy_mev) - log_minimum) /
        static_cast<G4double>(grid_size - 1);
    const auto* material = step->GetPreStepPoint()->GetMaterial();
    auto* particle_table = G4ParticleTable::GetParticleTable();
    const std::array<G4int, 2> pdg_ids{{22, 2112}};
    for (const auto pdg_id : pdg_ids) {
        const auto* definition = particle_table->FindParticle(pdg_id);
        if (definition == nullptr) {
            continue;
        }
        for (G4int index = 0; index < grid_size; ++index) {
            const auto energy =
                std::exp(log_minimum +
                         static_cast<G4double>(index) * log_step) *
                MeV;
            const auto macroscopic =
                MacroscopicTotal(definition, energy, material);
            pdg_id_ = pdg_id;
            energy_mev_ = static_cast<G4float>(energy / MeV);
            macroscopic_total_per_mm_ =
                macroscopic / (1.0 / mm);
            mean_free_path_mm_ =
                macroscopic > 0.0 ? (1.0 / macroscopic) / mm : 0.0;
            fNtuple->Fill();
        }
    }
    filled_ = true;
    return true;
}
