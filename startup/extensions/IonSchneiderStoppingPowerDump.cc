// Scorer for IonSchneiderStoppingPowerDump: per-ion per-Schneider-section
// UNRESTRICTED electronic stopping powers for MAIGO scheme-2 secondary
// stopping (SCHNIOSP v1). Mirrors CarbonSchneiderStoppingPowerDump material
// resolution; species/energy conventions mirror IonStoppingPowerNtuple
// (dedicated light-ion definitions, 0.01 MeV/u lower bound).

#include "IonSchneiderStoppingPowerDump.hh"

#include "G4EmCalculator.hh"
#include "G4Version.hh"
#include "G4Exception.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4UIcommand.hh"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace {

struct SectionProbe {
    int section_id;
    int hu_min;
    int hu_max;
    int rep_hu;
};

constexpr std::array<SectionProbe, 25> kSectionProbes = {{
    {0,  -1000, -950, -975},
    {1,  -950,  -120, -535},
    {2,  -120,  -83,  -102},
    {3,  -83,   -53,  -68},
    {4,  -53,   -23,  -38},
    {5,  -23,   7,    -8},
    {6,  7,     18,   12},
    {7,  18,    80,   49},
    {8,  80,    120,  100},
    {9,  120,  200,  160},
    {10, 200,   300,  250},
    {11, 300,   400,  350},
    {12, 400,   500,  450},
    {13, 500,   600,  550},
    {14, 600,   700,  650},
    {15, 700,   800,  750},
    {16, 800,   900,  850},
    {17, 900,   1000, 950},
    {18, 1000,  1100, 1050},
    {19, 1100,  1200, 1150},
    {20, 1200,  1300, 1250},
    {21, 1300,  1400, 1350},
    {22, 1400,  1500, 1450},
    {23, 1500,  2995, 2247},
    {24, 2995,  2996, 2995}
}};

G4String MaterialNameFromHU(int hu) {
    if (hu < 0) {
        return "PatientTissueFromHUNegative" + G4UIcommand::ConvertToString(-hu);
    }
    return "PatientTissueFromHU" + G4UIcommand::ConvertToString(hu);
}

// GPU charged-species slots 0..17 (must match get_charged_species_idx).
constexpr std::array<std::pair<G4int, G4int>, 18> kSpecies{{
    {1, 1}, {1, 2}, {1, 3},
    {2, 3}, {2, 4}, {2, 6},
    {3, 6}, {3, 7},
    {4, 7}, {4, 9}, {4, 10},
    {5, 8}, {5, 10}, {5, 11},
    {6, 10}, {6, 11}, {6, 12},
    {4, 6},
}};

G4ParticleDefinition* IonDefinition(G4int z, G4int a) {
    if (z == 1 && a == 1) return G4ParticleTable::GetParticleTable()->FindParticle("proton");
    if (z == 1 && a == 2) return G4ParticleTable::GetParticleTable()->FindParticle("deuteron");
    if (z == 1 && a == 3) return G4ParticleTable::GetParticleTable()->FindParticle("triton");
    if (z == 2 && a == 3) return G4ParticleTable::GetParticleTable()->FindParticle("He3");
    if (z == 2 && a == 4) return G4ParticleTable::GetParticleTable()->FindParticle("alpha");
    return G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(z, a, 0.0);
}

}  // namespace

IonSchneiderStoppingPowerDump::IonSchneiderStoppingPowerDump(
    TsParameterManager* parameter_manager,
    TsMaterialManager* material_manager,
    TsGeometryManager* geometry_manager,
    TsScoringManager* scoring_manager,
    TsExtensionManager* extension_manager,
    G4String scorer_name,
    G4String quantity,
    G4String output_file,
    G4bool is_sub_scorer)
    : TsVScorer(parameter_manager, material_manager, geometry_manager,
                scoring_manager, extension_manager, scorer_name, quantity,
                output_file, is_sub_scorer) {
    if (fPm->ParameterExists(GetFullParmName("MinEnergyMeVu"))) {
        min_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MinEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("MaxEnergyMeVu"))) {
        max_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MaxEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("EnergyStepMeVu"))) {
        energy_step_mevu_ = fPm->GetDoubleParameter(GetFullParmName("EnergyStepMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("OutputJsonFile"))) {
        output_json_file_ = fPm->GetStringParameter(GetFullParmName("OutputJsonFile"));
    } else {
        output_json_file_ = output_file;
    }
    if (fPm->ParameterExists(GetFullParmName("OutputCsvFile"))) {
        output_csv_file_ = fPm->GetStringParameter(GetFullParmName("OutputCsvFile"));
    }
}

IonSchneiderStoppingPowerDump::~IonSchneiderStoppingPowerDump() = default;

G4bool IonSchneiderStoppingPowerDump::ProcessHits(G4Step*, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_) {
        return false;
    }

    std::cout << "[IonSchneiderStoppingPowerDump] Starting ion stopping-power extraction..."
              << std::endl;

    std::vector<double> energies;
    for (double e = min_energy_mevu_; e <= max_energy_mevu_ + 1e-6; e += energy_step_mevu_) {
        energies.push_back(e);
    }
    const std::size_t num_energies = energies.size();
    std::cout << "[IonSchneiderStoppingPowerDump] Energy grid size: " << num_energies
              << ", species: " << kSpecies.size() << std::endl;

    std::vector<const G4Material*> materials(25, nullptr);
    std::vector<std::string> mat_names(25);
    std::vector<double> densities(25);
    for (std::size_t s = 0; s < 25; ++s) {
        mat_names[s] = MaterialNameFromHU(kSectionProbes[s].rep_hu);
        materials[s] = G4Material::GetMaterial(mat_names[s], false);
        if (!materials[s]) {
            const G4String msg = "Missing Schneider material: " + mat_names[s];
            G4Exception("IonSchneiderStoppingPowerDump", "MissingMaterial", FatalException, msg.c_str());
        }
        densities[s] = materials[s]->GetDensity() / (g / cm3);
    }

    G4EmCalculator em_calc;
    struct Cell { int z, a; std::vector<double> linear; };
    // [section][species][energy]
    std::vector<std::vector<std::vector<double>>> table(
        25, std::vector<std::vector<double>>(kSpecies.size(), std::vector<double>(num_energies, 0.0)));

    for (std::size_t s = 0; s < 25; ++s) {
        const G4Material* mat = materials[s];
        for (std::size_t k = 0; k < kSpecies.size(); ++k) {
            const auto [z, a] = kSpecies[k];
            G4ParticleDefinition* ion = IonDefinition(z, a);
            if (!ion) {
                G4Exception("IonSchneiderStoppingPowerDump", "MissingIon", FatalException,
                            ("No ion definition for Z=" + std::to_string(z) +
                             " A=" + std::to_string(a)).c_str());
            }
            for (std::size_t i = 0; i < num_energies; ++i) {
                const G4double total_energy = static_cast<G4double>(a) * energies[i] * MeV;
                const G4double dedx = em_calc.ComputeElectronicDEDX(total_energy, ion, mat);
                table[s][k][i] = dedx / (MeV / mm);
                if (!std::isfinite(table[s][k][i]) || table[s][k][i] <= 0)
                    G4Exception("IonSchneiderStoppingPowerDump", "InvalidStopping", FatalException,
                                "Nonpositive or nonfinite electronic stopping");
            }
        }
        std::cout << "[IonSchneiderStoppingPowerDump] section " << s << " done" << std::endl;
    }

    if (!output_csv_file_.empty()) {
        std::filesystem::path csv_path(output_csv_file_.c_str());
        std::filesystem::create_directories(csv_path.parent_path());
        std::ofstream csv(csv_path);
        csv << std::setprecision(10);
        csv << "# Geant4/TOPAS ion Unrestricted Electronic Stopping Power for 25 Schneider Sections\n";
        csv << "energy_mevu,section_id,species_z,species_a,material_name,density_g_cm3,linear_stopping_power_mev_per_mm\n";
        for (std::size_t i = 0; i < num_energies; ++i) {
            for (std::size_t s = 0; s < 25; ++s) {
                for (std::size_t k = 0; k < kSpecies.size(); ++k) {
                    csv << energies[i] << "," << s << ","
                        << kSpecies[k].first << "," << kSpecies[k].second << ","
                        << mat_names[s] << "," << densities[s] << ","
                        << table[s][k][i] << "\n";
                }
            }
        }
        std::cout << "[IonSchneiderStoppingPowerDump] Wrote CSV: " << output_csv_file_ << std::endl;
    }

    if (!output_json_file_.empty()) {
        std::filesystem::path json_path(output_json_file_.c_str());
        std::filesystem::create_directories(json_path.parent_path());
        std::ofstream jout(json_path);
        jout << std::setprecision(10);
        jout << "{\n";
        jout << "  \"geant4_version\": \"" << G4Version << "\",\n";
        jout << "  \"schema_version\": 1,\n";
        jout << "  \"task\": \"Ion Schneider Stopping Power Extraction\",\n";
        jout << "  \"extractor\": \"IonSchneiderStoppingPowerDump\",\n";
        jout << "  \"quantity\": \"unrestricted electronic dE/dx\",\n";
        jout << "  \"num_sections\": 25,\n";
        jout << "  \"num_species\": " << kSpecies.size() << ",\n";
        jout << "  \"species_za\": [";
        for (std::size_t k = 0; k < kSpecies.size(); ++k) {
            jout << "[" << kSpecies[k].first << "," << kSpecies[k].second << "]"
                 << (k + 1 < kSpecies.size() ? "," : "");
        }
        jout << "],\n";
        jout << "  \"num_energies\": " << num_energies << ",\n";
        jout << "  \"energy_min_mevu\": " << energies.front() << ",\n";
        jout << "  \"energy_max_mevu\": " << energies.back() << ",\n";
        jout << "  \"energy_step_mevu\": " << energy_step_mevu_ << ",\n";
        jout << "  \"sections\": [\n";
        for (std::size_t s = 0; s < 25; ++s) {
            const double probe = table[s][16][1000];
            jout << "    {\"section_id\": " << s
                 << ", \"material_name\": \"" << mat_names[s] << "\""
                 << ", \"density_g_cm3\": " << densities[s]
                 << ", \"c12_linear_at_100_01_mevu\": " << probe << "}"
                 << (s + 1 < 25 ? "," : "") << "\n";
        }
        jout << "  ]\n}\n";
        std::cout << "[IonSchneiderStoppingPowerDump] Wrote JSON: " << output_json_file_ << std::endl;
    }

    filled_ = true;
    return true;
}

void IonSchneiderStoppingPowerDump::Output() {}
