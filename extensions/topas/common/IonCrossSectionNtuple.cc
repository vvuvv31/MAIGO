// Scorer for IonCrossSectionNtuple

#include "IonCrossSectionNtuple.hh"

#include "G4HadronicProcessStore.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <array>
#include <utility>

IonCrossSectionNtuple::IonCrossSectionNtuple(
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
    fNtuple->RegisterColumnI(&atomic_number_, "Atomic Number Z");
    fNtuple->RegisterColumnI(&mass_number_, "Mass Number A");
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "Energy (MeV/u)", "");
    fNtuple->RegisterColumnD(
        &macroscopic_inelastic_per_mm_,
        "Macroscopic Inelastic Cross Section (1/mm)", "");
    fNtuple->RegisterColumnD(
        &mean_free_path_mm_, "Inelastic Mean Free Path (mm)", "");
}

IonCrossSectionNtuple::~IonCrossSectionNtuple() = default;

G4bool IonCrossSectionNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }

    constexpr std::array<std::pair<G4int, G4int>, 32> species{{
        {1, 1}, {1, 2}, {1, 3},
        {2, 3}, {2, 4}, {2, 6}, {2, 8},
        {3, 6}, {3, 7}, {3, 8}, {3, 9},
        {4, 4}, {4, 6}, {4, 7}, {4, 9}, {4, 10},
        {5, 8}, {5, 10}, {5, 11}, {5, 12},
        {6, 9}, {6, 10}, {6, 11}, {6, 12}, {6, 13}, {6, 14},
        {7, 12}, {7, 14}, {7, 15},
        {8, 15}, {8, 16}, {9, 19},
    }};

    const auto* material =
        step->GetPreStepPoint()->GetMaterial();
    auto* particle_table =
        G4ParticleTable::GetParticleTable();
    auto* ion_table = particle_table->GetIonTable();
    auto* store = G4HadronicProcessStore::Instance();

    for (const auto& [z, a] : species) {
        G4ParticleDefinition* ion = nullptr;
        if (z == 1 && a == 1) {
            ion = particle_table->FindParticle("proton");
        } else if (z == 1 && a == 2) {
            ion = particle_table->FindParticle("deuteron");
        } else if (z == 1 && a == 3) {
            ion = particle_table->FindParticle("triton");
        } else if (z == 2 && a == 3) {
            ion = particle_table->FindParticle("He3");
        } else if (z == 2 && a == 4) {
            ion = particle_table->FindParticle("alpha");
        } else {
            ion = ion_table->GetIon(z, a, 0.0);
        }
        if (ion == nullptr) {
            continue;
        }
        for (G4int energy_index = 0; energy_index <= 400; ++energy_index) {
            const auto energy_per_u =
                (0.01 + static_cast<G4double>(energy_index)) * MeV;
            const auto total_energy =
                static_cast<G4double>(a) * energy_per_u;
            const auto macroscopic =
                store->GetInelasticCrossSectionPerVolume(
                    ion, total_energy, material);
            atomic_number_ = z;
            mass_number_ = a;
            energy_mev_per_u_ =
                static_cast<G4float>(energy_per_u / MeV);
            macroscopic_inelastic_per_mm_ =
                macroscopic / (1.0 / mm);
            mean_free_path_mm_ =
                macroscopic > 0.0
                    ? (1.0 / macroscopic) / mm
                    : 0.0;
            fNtuple->Fill();
        }
    }
    filled_ = true;
    return true;
}
