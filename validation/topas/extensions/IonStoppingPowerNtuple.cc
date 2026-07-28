// Scorer for IonStoppingPowerNtuple

#include "IonStoppingPowerNtuple.hh"

#include "G4EmCalculator.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"

#include <algorithm>
#include <array>
#include <utility>

IonStoppingPowerNtuple::IonStoppingPowerNtuple(
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
        &unrestricted_dedx_mev_per_mm_, "Unrestricted Electronic dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &restricted_dedx_mev_per_mm_, "Restricted Electronic dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &transport_table_dedx_mev_per_mm_,
        "Transport Table dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &nuclear_dedx_mev_per_mm_, "Nuclear dE/dx (MeV/mm)", "");
    fNtuple->RegisterColumnD(
        &delta_electron_fraction_, "Delta Electron Fraction", "");
}

IonStoppingPowerNtuple::~IonStoppingPowerNtuple() = default;

G4bool IonStoppingPowerNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step->GetTrack()->GetParentID() != 0) {
        return false;
    }

    // Exact species present in the Geant4 11.3.2 400 MeV/u cascade package.
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

    const auto* material = step->GetPreStepPoint()->GetMaterial();
    auto* ion_table = G4ParticleTable::GetParticleTable()->GetIonTable();
    G4EmCalculator calculator;
    constexpr G4double production_range_cut = 0.05 * mm;

    for (const auto& [z, a] : species) {
        // Geant4 transports the five light stable ions with dedicated particle
        // definitions. Query those exact definitions so the lookup table uses
        // the same EM model path as myHadronLET on real tracks.
        G4ParticleDefinition* ion = nullptr;
        if (z == 1 && a == 1) {
            ion = G4ParticleTable::GetParticleTable()->FindParticle("proton");
        } else if (z == 1 && a == 2) {
            ion = G4ParticleTable::GetParticleTable()->FindParticle("deuteron");
        } else if (z == 1 && a == 3) {
            ion = G4ParticleTable::GetParticleTable()->FindParticle("triton");
        } else if (z == 2 && a == 3) {
            ion = G4ParticleTable::GetParticleTable()->FindParticle("He3");
        } else if (z == 2 && a == 4) {
            ion = G4ParticleTable::GetParticleTable()->FindParticle("alpha");
        } else {
            ion = ion_table->GetIon(z, a, 0.0);
        }
        if (ion == nullptr) {
            continue;
        }
        // Uniform 0.1 MeV/u grid starting at 0.01 MeV/u. The former 1 MeV/u
        // lower bound clamped the Bragg-tail stopping power of p/d/t/He ions
        // and systematically biased fragment LET low.
        for (G4int energy_index = 0; energy_index <= 4000; ++energy_index) {
            const auto energy_per_u =
                (0.01 + 0.1 * static_cast<G4double>(energy_index)) * MeV;
            const auto total_energy = static_cast<G4double>(a) * energy_per_u;
            const auto unrestricted = calculator.ComputeElectronicDEDX(
                total_energy, ion, material);
            const auto restricted = calculator.ComputeDEDXForCutInRange(
                total_energy, ion, material, production_range_cut);
            const auto transport_table =
                calculator.GetDEDX(total_energy, ion, material);
            const auto nuclear =
                calculator.ComputeNuclearDEDX(total_energy, ion, material);

            atomic_number_ = z;
            mass_number_ = a;
            energy_mev_per_u_ =
                static_cast<G4float>(energy_per_u / MeV);
            unrestricted_dedx_mev_per_mm_ =
                unrestricted / (MeV / mm);
            restricted_dedx_mev_per_mm_ =
                restricted / (MeV / mm);
            transport_table_dedx_mev_per_mm_ =
                transport_table / (MeV / mm);
            nuclear_dedx_mev_per_mm_ = nuclear / (MeV / mm);
            delta_electron_fraction_ =
                unrestricted > 0.0
                    ? std::max(0.0, 1.0 - restricted / unrestricted)
                    : 0.0;
            fNtuple->Fill();
        }
    }
    filled_ = true;
    return true;
}
