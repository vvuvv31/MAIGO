// Scorer for CarbonSchneiderStoppingPowerDump
// Extracts Geant4/TOPAS C12 mass and linear stopping powers across all 25 Schneider sections
// on the 4001-node transport energy grid (0.01 to 400.01 MeV/u).

#include "CarbonSchneiderStoppingPowerDump.hh"

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4EmCalculator.hh"
#include "G4Exception.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4UIcommand.hh"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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
    {9,  120,   200,  160},
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

}  // namespace

CarbonSchneiderStoppingPowerDump::CarbonSchneiderStoppingPowerDump(
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
    if (fPm->ParameterExists(GetFullParmName("ProjectileZ")))
        projectile_z_ = fPm->GetIntegerParameter(GetFullParmName("ProjectileZ"));
    if (fPm->ParameterExists(GetFullParmName("ProjectileA")))
        projectile_a_ = fPm->GetIntegerParameter(GetFullParmName("ProjectileA"));
    if (projectile_z_ <= 0 || projectile_a_ < projectile_z_)
        G4Exception("CarbonSchneiderStoppingPowerDump", "InvalidProjectile", FatalException,
                    "Require positive Z and A >= Z");
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

CarbonSchneiderStoppingPowerDump::~CarbonSchneiderStoppingPowerDump() = default;

G4bool CarbonSchneiderStoppingPowerDump::ProcessHits(G4Step*, G4TouchableHistory*) {
    if (!fIsActive) {
        ++fSkippedWhileInactive;
        return false;
    }
    if (filled_) {
        return false;
    }

    std::cout << "[CarbonSchneiderStoppingPowerDump] Starting stopping-power extraction..." << std::endl;

    // 1. Resolve the requested projectile
    G4ParticleDefinition* projectile = (projectile_z_ == 1 && projectile_a_ == 1)
        ? G4ParticleTable::GetParticleTable()->FindParticle("proton")
        : G4IonTable::GetIonTable()->GetIon(projectile_z_, projectile_a_, 0.0);
    if (!projectile) {
        G4Exception("CarbonSchneiderStoppingPowerDump", "MissingProjectile", FatalException,
                    "Failed to find requested projectile definition!");
    }

    // 2. Build energy grid matching MAIGO transport grid
    std::vector<double> energies;
    for (double e = min_energy_mevu_; e <= max_energy_mevu_ + 1e-6; e += energy_step_mevu_) {
        energies.push_back(e);
    }
    const std::size_t num_energies = energies.size();
    std::cout << "[CarbonSchneiderStoppingPowerDump] Energy grid size: " << num_energies
              << " (" << energies.front() << " to " << energies.back() << " MeV/u)" << std::endl;

    // 3. Fetch materials for all 25 sections
    std::vector<const G4Material*> materials(25, nullptr);
    std::vector<G4String> mat_names(25);
    std::vector<double> densities(25);
    std::vector<double> mean_excitations(25);

    for (std::size_t s = 0; s < 25; ++s) {
        mat_names[s] = MaterialNameFromHU(kSectionProbes[s].rep_hu);
        materials[s] = G4Material::GetMaterial(mat_names[s], false);
        if (!materials[s]) {
            const G4String msg = "Missing Schneider material: " + mat_names[s];
            G4Exception("CarbonSchneiderStoppingPowerDump", "MissingMaterial", FatalException, msg.c_str());
        }
        densities[s] = materials[s]->GetDensity() / (g / cm3);
        mean_excitations[s] = materials[s]->GetIonisation()->GetMeanExcitationEnergy() / eV;
    }

    // 4. Compute stopping powers using G4EmCalculator
    G4EmCalculator em_calc;

    struct SectionData {
        int section_id;
        int rep_hu;
        std::string material_name;
        double density_g_cm3;
        double I_eV;
        std::vector<double> mass_sp_MeV_mm_per_g_cm3;
        std::vector<double> linear_sp_MeV_per_mm;
        std::vector<double> csda_range_mm;
    };

    std::vector<SectionData> section_results(25);

    for (std::size_t s = 0; s < 25; ++s) {
        SectionData& sd = section_results[s];
        sd.section_id = kSectionProbes[s].section_id;
        sd.rep_hu = kSectionProbes[s].rep_hu;
        sd.material_name = mat_names[s];
        sd.density_g_cm3 = densities[s];
        sd.I_eV = mean_excitations[s];
        sd.mass_sp_MeV_mm_per_g_cm3.resize(num_energies, 0.0);
        sd.linear_sp_MeV_per_mm.resize(num_energies, 0.0);
        sd.csda_range_mm.resize(num_energies, 0.0);

        const G4Material* mat = materials[s];
        const double rho = densities[s];

        for (std::size_t i = 0; i < num_energies; ++i) {
            const double e_mevu = energies[i];
            const G4double total_energy = static_cast<double>(projectile_a_) * e_mevu * MeV;

            const G4double total_dedx = em_calc.ComputeTotalDEDX(total_energy, projectile, mat);
            const double linear_sp = total_dedx / (MeV / mm);
            const double mass_sp = linear_sp / rho;

            sd.linear_sp_MeV_per_mm[i] = linear_sp;
            sd.mass_sp_MeV_mm_per_g_cm3[i] = mass_sp;
        }

        // Integrate CSDA range in mm:
        // R(E) = \int_0^E static_cast<double>(projectile_a_) * dE' / S_linear(E')
        // Initial bin: linear extrapolation to zero energy
        sd.csda_range_mm[0] = (static_cast<double>(projectile_a_) * energies[0]) / sd.linear_sp_MeV_per_mm[0];
        for (std::size_t i = 1; i < num_energies; ++i) {
            const double e_prev = energies[i - 1];
            const double e_curr = energies[i];
            const double s_prev = sd.linear_sp_MeV_per_mm[i - 1];
            const double s_curr = sd.linear_sp_MeV_per_mm[i];

            double delta_r = 0.0;
            if (std::abs(s_curr - s_prev) > 1e-9) {
                delta_r = static_cast<double>(projectile_a_) * ((e_curr - e_prev) / (s_curr - s_prev)) * std::log(s_curr / s_prev);
            } else {
                delta_r = static_cast<double>(projectile_a_) * (e_curr - e_prev) / s_prev;
            }
            sd.csda_range_mm[i] = sd.csda_range_mm[i - 1] + delta_r;
        }
    }

    
    // 4b. Density scaling audit across ALL 25 Schneider sections at lower bound HU, rep HU, and upper bound HU
    std::vector<std::string> density_audit_json;
    const double audit_energies_mevu[] = {0.5, 5.0, 20.0, 100.0, 200.0, 300.0, 430.0};
    constexpr std::size_t num_audit_energies = 7;

    std::cout << "[CarbonSchneiderStoppingPowerDump] Auditing section-internal density scaling for all 25 sections across 7 energies..." << std::endl;
    for (std::size_t s = 0; s < 25; ++s) {
        const auto& probe = kSectionProbes[s];
        const int hu_low = probe.hu_min;
        const int hu_rep = probe.rep_hu;
        const int hu_high = (s == 24) ? 2995 : (probe.hu_max - 1);

        const G4String name_low = MaterialNameFromHU(hu_low);
        const G4String name_rep = MaterialNameFromHU(hu_rep);
        const G4String name_high = MaterialNameFromHU(hu_high);

        const G4Material* mat_low = G4Material::GetMaterial(name_low, false);
        const G4Material* mat_rep = materials[s];
        const G4Material* mat_high = G4Material::GetMaterial(name_high, false);

        if (!mat_low) {
            const G4String msg = "Missing lower boundary Schneider material: " + name_low;
            G4Exception("CarbonSchneiderStoppingPowerDump", "MissingBoundaryMaterial", FatalException, msg.c_str());
        }
        if (!mat_high) {
            const G4String msg = "Missing upper boundary Schneider material: " + name_high;
            G4Exception("CarbonSchneiderStoppingPowerDump", "MissingBoundaryMaterial", FatalException, msg.c_str());
        }

        const double rho_low = mat_low->GetDensity() / (g / cm3);
        const double rho_rep = densities[s];
        const double rho_high = mat_high->GetDensity() / (g / cm3);

        std::stringstream ss;
        ss << "    {\n"
           << "      \"section_id\": " << s << ",\n"
           << "      \"label\": \"" << mat_names[s] << "\",\n"
           << "      \"hu_bounds\": [" << hu_low << ", " << hu_rep << ", " << hu_high << "],\n"
           << "      \"densities_g_cm3\": [" << rho_low << ", " << rho_rep << ", " << rho_high << "],\n"
           << "      \"nominal_density_g_cm3\": " << rho_rep << ",\n"
           << "      \"tests\": [\n";

        for (size_t ei = 0; ei < num_audit_energies; ++ei) {
            const double e_mevu = audit_energies_mevu[ei];
            const G4double total_energy = static_cast<double>(projectile_a_) * e_mevu * MeV;

            const double s_nom_mass = em_calc.ComputeTotalDEDX(total_energy, projectile, mat_rep) / (MeV/mm) / rho_rep;
            const double s_low_mass = em_calc.ComputeTotalDEDX(total_energy, projectile, mat_low) / (MeV/mm) / rho_low;
            const double s_high_mass = em_calc.ComputeTotalDEDX(total_energy, projectile, mat_high) / (MeV/mm) / rho_high;

            const double diff_low = std::abs(s_low_mass - s_nom_mass) / s_nom_mass;
            const double diff_high = std::abs(s_high_mass - s_nom_mass) / s_nom_mass;
            const double max_rel_err = std::max(diff_low, diff_high);

            ss << "        {\"energy_mevu\": " << e_mevu
               << ", \"s_mass_nom\": " << s_nom_mass
               << ", \"s_mass_low\": " << s_low_mass
               << ", \"s_mass_high\": " << s_high_mass
               << ", \"max_rel_err\": " << max_rel_err << "}"
               << (ei + 1 < num_audit_energies ? ",\n" : "\n");
        }
        ss << "      ]\n    }";
        density_audit_json.push_back(ss.str());
    }

    std::cout << "[CarbonSchneiderStoppingPowerDump] Successfully calculated stopping powers for 25 sections." << std::endl;

    // 5. Write CSV file if requested
    if (!output_csv_file_.empty()) {
        std::filesystem::path csv_path(output_csv_file_.c_str());
        std::filesystem::create_directories(csv_path.parent_path());
        std::ofstream csv(csv_path);
        csv << std::setprecision(10);
        csv << "# Geant4/TOPAS Stopping Power Table for 25 Schneider Sections\n";
        csv << "energy_mevu,section_id,material_name,density_g_cm3,mass_stopping_power_mev_mm_per_g_cm3,linear_stopping_power_mev_per_mm,csda_range_mm\n";
        for (std::size_t i = 0; i < num_energies; ++i) {
            for (std::size_t s = 0; s < 25; ++s) {
                const auto& sd = section_results[s];
                csv << energies[i] << ","
                    << sd.section_id << ","
                    << sd.material_name << ","
                    << sd.density_g_cm3 << ","
                    << sd.mass_sp_MeV_mm_per_g_cm3[i] << ","
                    << sd.linear_sp_MeV_per_mm[i] << ","
                    << sd.csda_range_mm[i] << "\n";
            }
        }
        std::cout << "[CarbonSchneiderStoppingPowerDump] Wrote CSV: " << output_csv_file_ << std::endl;
    }

    // 6. Write JSON metadata and complete table
    if (!output_json_file_.empty()) {
        std::filesystem::path json_path(output_json_file_.c_str());
        std::filesystem::create_directories(json_path.parent_path());
        std::ofstream jout(json_path);
        jout << std::setprecision(10);
        jout << "{\n";
        jout << "  \"schema_version\": 1,\n";
        jout << "  \"task\": \"Step 14 Schneider Stopping Power Extraction\",\n";
        jout << "  \"extractor\": \"CarbonSchneiderStoppingPowerDump\",\n";
        jout << "  \"projectile\": {\"name\": \"" << projectile->GetParticleName()
             << "\", \"z\": " << projectile_z_ << ", \"a\": " << projectile_a_ << "},\n";
        jout << "  \"num_sections\": 25,\n";
        jout << "  \"num_energies\": " << num_energies << ",\n";
        jout << "  \"energy_min_mevu\": " << energies.front() << ",\n";
        jout << "  \"energy_max_mevu\": " << energies.back() << ",\n";
        jout << "  \"energy_step_mevu\": " << energy_step_mevu_ << ",\n";
        jout << "  \"density_scaling_audit\": [\n";
        for (size_t i = 0; i < density_audit_json.size(); ++i) {
            jout << density_audit_json[i] << (i + 1 < density_audit_json.size() ? ",\n" : "\n");
        }
        jout << "  ],\n";
        jout << "  \"sections\": [\n";
        for (std::size_t s = 0; s < 25; ++s) {
            const auto& sd = section_results[s];
            jout << "    {\n";
            jout << "      \"section_id\": " << sd.section_id << ",\n";
            jout << "      \"rep_hu\": " << sd.rep_hu << ",\n";
            jout << "      \"material_name\": \"" << sd.material_name << "\",\n";
            jout << "      \"density_g_cm3\": " << sd.density_g_cm3 << ",\n";
            jout << "      \"mean_excitation_energy_eV\": " << sd.I_eV << ",\n";
            jout << "      \"mass_stopping_power_at_100_mevu\": " << sd.mass_sp_MeV_mm_per_g_cm3[999] << ",\n";
            jout << "      \"csda_range_mm_at_100_mevu\": " << sd.csda_range_mm[999] << ",\n";
            jout << "      \"csda_range_mm_at_200_mevu\": " << sd.csda_range_mm[1999] << ",\n";
            jout << "      \"csda_range_mm_at_300_mevu\": " << sd.csda_range_mm[2999] << "\n";
            jout << "    }" << (s + 1 < 25 ? "," : "") << "\n";
        }
        jout << "  ]\n";
        jout << "}\n";
        std::cout << "[CarbonSchneiderStoppingPowerDump] Wrote JSON: " << output_json_file_ << std::endl;
    }

    filled_ = true;
    return true;
}

void CarbonSchneiderStoppingPowerDump::Output() {}
