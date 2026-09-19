// Scorer for Cinel02MaterialRateNtuple

#include "Cinel02MaterialRateNtuple.hh"

#include "G4Element.hh"
#include "G4Exception.hh"
#include "G4HadronicProcessStore.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"

#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace {

constexpr std::array<std::pair<G4int, G4int>, 17> kProjectiles{{
    {1, 1}, {1, 2}, {1, 3}, {2, 3}, {2, 4}, {2, 6},
    {3, 6}, {3, 7}, {4, 7}, {4, 9}, {4, 10},
    {5, 8}, {5, 10}, {5, 11}, {6, 10}, {6, 11}, {6, 12},
}};

G4int ReadOptionalInteger(
    TsParameterManager* parameter_manager, const G4String& name,
    const G4int fallback) {
    return parameter_manager->ParameterExists(name)
               ? parameter_manager->GetIntegerParameter(name)
               : fallback;
}

G4double ReadOptionalEnergy(
    TsParameterManager* parameter_manager, const G4String& name,
    const G4double fallback) {
    return parameter_manager->ParameterExists(name)
               ? parameter_manager->GetDoubleParameter(name, "Energy") / MeV
               : fallback;
}

G4String ProjectileName(const G4int z, const G4int a) {
    if (z == 1 && a == 1) return "proton";
    if (z == 1 && a == 2) return "deuteron";
    if (z == 1 && a == 3) return "triton";
    if (z == 2 && a == 3) return "He3";
    if (z == 2 && a == 4) return "alpha";
    return "";
}

G4ParticleDefinition* FindProjectile(
    G4ParticleTable* particle_table, const G4int z, const G4int a) {
    if (particle_table == nullptr) return nullptr;
    const auto name = ProjectileName(z, a);
    if (!name.empty()) return particle_table->FindParticle(name);
    auto* ion_table = particle_table->GetIonTable();
    return ion_table == nullptr ? nullptr : ion_table->GetIon(z, a, 0.0);
}

void Fatal(const G4String& message) {
    G4ExceptionDescription description;
    description << message;
    G4Exception("Cinel02MaterialRateNtuple", "InvalidParameter",
                FatalException, description);
}

}  // namespace

Cinel02MaterialRateNtuple::Cinel02MaterialRateNtuple(
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
      material_section_(ReadOptionalInteger(
          parameter_manager, GetFullParmName("MaterialSection"), -1)),
      energy_min_mev_per_u_(ReadOptionalEnergy(
          parameter_manager, GetFullParmName("EnergyMin"), 0.0)),
      energy_width_mev_per_u_(ReadOptionalEnergy(
          parameter_manager, GetFullParmName("EnergyWidth"), 1.0)),
      energy_count_(ReadOptionalInteger(
          parameter_manager, GetFullParmName("EnergyCount"), 401)) {
    if (material_section_ < 0 || material_section_ > 24 ||
        !std::isfinite(energy_min_mev_per_u_) || energy_min_mev_per_u_ < 0.0 ||
        !std::isfinite(energy_width_mev_per_u_) || energy_width_mev_per_u_ <= 0.0 ||
        energy_count_ <= 0) {
        Fatal("MaterialSection must be in [0,24] and the energy grid must be finite");
    }

    fNtuple->RegisterColumnI(&material_section_, "material_section");
    fNtuple->RegisterColumnS(&material_name_, "material_name");
    fNtuple->RegisterColumnD(
        &material_density_g_per_cm3_, "material_density_g_per_cm3", "");
    fNtuple->RegisterColumnI(&projectile_z_, "projectile_z");
    fNtuple->RegisterColumnI(&projectile_a_, "projectile_a");
    fNtuple->RegisterColumnI(&target_z_, "target_z");
    fNtuple->RegisterColumnI(&target_a_, "target_a");
    fNtuple->RegisterColumnD(&target_mass_fraction_, "target_mass_fraction", "");
    fNtuple->RegisterColumnD(
        &target_number_density_per_mm3_, "target_number_density_per_mm3", "");
    fNtuple->RegisterColumnF(&energy_mev_per_u_, "energy_MeV_per_u", "");
    fNtuple->RegisterColumnD(
        &microscopic_cross_section_barn_, "microscopic_cross_section_barn", "");
    fNtuple->RegisterColumnD(
        &macroscopic_cross_section_per_mm_,
        "macroscopic_cross_section_per_mm", "");
}

Cinel02MaterialRateNtuple::~Cinel02MaterialRateNtuple() = default;

G4bool Cinel02MaterialRateNtuple::ProcessHits(
    G4Step* step, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_ || step == nullptr || step->GetTrack() == nullptr ||
        step->GetTrack()->GetParentID() != 0 ||
        step->GetPreStepPoint() == nullptr) {
        return false;
    }

    const auto* material = step->GetPreStepPoint()->GetMaterial();
    if (material == nullptr || material->GetNumberOfElements() <= 0) {
        Fatal("the extraction component has no material composition");
        return false;
    }
    const auto* elements = material->GetElementVector();
    const auto* atom_densities = material->GetVecNbOfAtomsPerVolume();
    const auto* mass_fractions = material->GetFractionVector();
    const G4Element* hydrogen = nullptr;
    const G4Element* oxygen = nullptr;
    G4double hydrogen_fraction = 0.0;
    G4double oxygen_fraction = 0.0;
    G4double hydrogen_density = 0.0;
    G4double oxygen_density = 0.0;
    for (G4int index = 0; index < material->GetNumberOfElements(); ++index) {
        const auto z = static_cast<G4int>(std::lround((*elements)[index]->GetZ()));
        if (z == 1) {
            hydrogen = (*elements)[index];
            hydrogen_fraction = mass_fractions[index];
            hydrogen_density = atom_densities[index];
        } else if (z == 8) {
            oxygen = (*elements)[index];
            oxygen_fraction = mass_fractions[index];
            oxygen_density = atom_densities[index];
        }
    }

    auto* particle_table = G4ParticleTable::GetParticleTable();
    auto* store = G4HadronicProcessStore::Instance();
    for (const auto& [z, a] : kProjectiles) {
        auto* projectile = FindProjectile(particle_table, z, a);
        if (projectile == nullptr) continue;
        for (const auto target : std::array<std::tuple<G4int, G4int, const G4Element*, G4double, G4double>, 2>{{
                 {1, 1, hydrogen, hydrogen_fraction, hydrogen_density},
                 {8, 16, oxygen, oxygen_fraction, oxygen_density}}}) {
            const auto [target_z, target_a, element, fraction, number_density] = target;
            for (G4int energy_index = 0; energy_index < energy_count_; ++energy_index) {
                const auto energy_per_u =
                    (energy_min_mev_per_u_ + energy_index * energy_width_mev_per_u_) * MeV;
                const auto total_energy = static_cast<G4double>(a) * energy_per_u;
                G4double microscopic = 0.0;
                if (element != nullptr && number_density > 0.0) {
                    microscopic = store->GetInelasticCrossSectionPerAtom(
                        projectile, total_energy, element, material);
                }
                const auto macroscopic = microscopic * number_density;
                material_name_ = material->GetName();
                material_density_g_per_cm3_ = material->GetDensity() / (g / cm3);
                this->projectile_z_ = z;
                this->projectile_a_ = a;
                this->target_z_ = target_z;
                this->target_a_ = target_a;
                target_mass_fraction_ = fraction;
                target_number_density_per_mm3_ = number_density / (1.0 / mm3);
                energy_mev_per_u_ = static_cast<G4float>(energy_per_u / MeV);
                microscopic_cross_section_barn_ = microscopic / barn;
                macroscopic_cross_section_per_mm_ = macroscopic / (1.0 / mm);
                fNtuple->Fill();
            }
        }
    }
    filled_ = true;
    return true;
}
