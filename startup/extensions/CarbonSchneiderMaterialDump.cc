// Scorer for CarbonSchneiderMaterialDump
// Extracts Geant4/TOPAS Schneider material truth for all 25 sections

#include "CarbonSchneiderMaterialDump.hh"

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"
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

constexpr std::array<const char*, 13> kCanonicalElements = {
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen",
    "Magnesium", "Phosphorus", "Sulfur", "Chlorine",
    "Argon", "Calcium", "Sodium", "Potassium", "Titanium"
};

constexpr std::array<int, 13> kCanonicalZ = {
    1, 6, 7, 8, 12, 15, 16, 17, 18, 20, 11, 19, 22
};

struct SectionProbe {
    int section_id;
    int hu_min;
    int hu_max;
    int rep_hu;
};

// 25 Schneider sections with representative HUs:
// Section 0: [-1000, -950) -> -975
// Section 1: [-950, -120)  -> -535
// Section 2: [-120, -83)   -> -102
// Section 3: [-83, -53)    -> -68
// Section 4: [-53, -23)    -> -38
// Section 5: [-23, 7)      -> -8
// Section 6: [7, 18)       -> 12
// Section 7: [18, 80)      -> 49
// Section 8: [80, 120)     -> 100
// Section 9: [120, 200)    -> 160
// Section 10: [200, 300)   -> 250
// Section 11: [300, 400)   -> 350
// Section 12: [400, 500)   -> 450
// Section 13: [500, 600)   -> 550
// Section 14: [600, 700)   -> 650
// Section 15: [700, 800)   -> 750
// Section 16: [800, 900)   -> 850
// Section 17: [900, 1000)  -> 950
// Section 18: [1000, 1100) -> 1050
// Section 19: [1100, 1200) -> 1150
// Section 20: [1200, 1300) -> 1250
// Section 21: [1300, 1400) -> 1350
// Section 22: [1400, 1500) -> 1450
// Section 23: [1500, 2995) -> 2247
// Section 24: [2995, 2996) -> 2995
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

constexpr std::array<int, 26> kSchneiderEdges = {{
    -1000, -950, -120, -83, -53, -23, 7, 18, 80, 120,
    200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 2995, 2996
}};

G4String MaterialNameFromHU(int hu) {
    if (hu < 0) {
        return "PatientTissueFromHUNegative" + G4UIcommand::ConvertToString(-hu);
    }
    return "PatientTissueFromHU" + G4UIcommand::ConvertToString(hu);
}

}  // namespace

CarbonSchneiderMaterialDump::CarbonSchneiderMaterialDump(
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
    output_path_ = output_file;
    if (output_path_.empty()) {
        output_path_ = "topas-schneider-materials.json";
    }
}

CarbonSchneiderMaterialDump::~CarbonSchneiderMaterialDump() = default;

G4bool CarbonSchneiderMaterialDump::ProcessHits(G4Step*, G4TouchableHistory*) {
    if (!dumped_) {
        DumpSchneiderMaterials();
        dumped_ = true;
    }
    return true;
}

void CarbonSchneiderMaterialDump::Output() {
    if (!dumped_) {
        DumpSchneiderMaterials();
        dumped_ = true;
    }
}

void CarbonSchneiderMaterialDump::DumpSchneiderMaterials() {
    if (output_path_.empty()) return;

    std::filesystem::path out_path(std::string(output_path_.c_str()));
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(out_path);
    if (!out) {
        G4cerr << "CarbonSchneiderMaterialDump: Cannot open output file " << output_path_ << G4endl;
        return;
    }

    out << std::setprecision(14);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"extractor\": \"CarbonSchneiderMaterialDump\",\n";
    out << "  \"section_count\": 25,\n";
    out << "  \"sections\": [\n";

    for (std::size_t s = 0; s < kSectionProbes.size(); ++s) {
        const auto& probe = kSectionProbes[s];
        const G4String mat_name = MaterialNameFromHU(probe.rep_hu);
        const G4Material* material = G4Material::GetMaterial(mat_name, false);

        if (!material) {
            G4cerr << "CarbonSchneiderMaterialDump: Material not found: " << mat_name << G4endl;
            continue;
        }

        const G4double density_g_cm3 = material->GetDensity() / (g / cm3);
        const G4double radlen_mm = material->GetRadlen() / mm;
        const G4double mean_ex_eV = material->GetIonisation()->GetMeanExcitationEnergy() / eV;
        const std::size_t num_elements = material->GetNumberOfElements();

        const G4ElementVector* elem_vec = material->GetElementVector();
        const G4double* frac_vec = material->GetFractionVector();
        const G4double* atom_dens_vec = material->GetVecNbOfAtomsPerVolume();

        out << "    {\n";
        out << "      \"section_id\": " << probe.section_id << ",\n";
        out << "      \"hu_min_inclusive\": " << probe.hu_min << ",\n";
        out << "      \"hu_max_exclusive\": " << probe.hu_max << ",\n";
        out << "      \"representative_HU\": " << probe.rep_hu << ",\n";
        out << "      \"material_name\": \"" << material->GetName() << "\",\n";
        out << "      \"density_g_cm3\": " << density_g_cm3 << ",\n";
        out << "      \"radiation_length_mm\": " << radlen_mm << ",\n";
        out << "      \"mean_excitation_energy_eV\": " << mean_ex_eV << ",\n";
        out << "      \"number_of_elements\": " << num_elements << ",\n";
        out << "      \"elements\": [\n";

        for (std::size_t el = 0; el < kCanonicalElements.size(); ++el) {
            const G4String canon_name = kCanonicalElements[el];
            const int canon_z = kCanonicalZ[el];

            double mass_fraction = 0.0;
            double atom_density_per_mm3 = 0.0;
            double a_g_mol = 0.0;

            for (std::size_t i = 0; i < num_elements; ++i) {
                const G4Element* elem = (*elem_vec)[i];
                if (elem->GetZ() == canon_z || elem->GetName() == canon_name) {
                    mass_fraction = frac_vec[i];
                    atom_density_per_mm3 = atom_dens_vec[i] / (1.0 / mm3);
                    a_g_mol = elem->GetA() / (g / mole);
                    break;
                }
            }

            out << "        {\n";
            out << "          \"canonical_index\": " << el << ",\n";
            out << "          \"name\": \"" << canon_name << "\",\n";
            out << "          \"z\": " << canon_z << ",\n";
            out << "          \"a_g_mol\": " << a_g_mol << ",\n";
            out << "          \"mass_fraction\": " << mass_fraction << ",\n";
            out << "          \"atom_density_per_mm3\": " << atom_density_per_mm3 << "\n";
            out << "        }" << (el + 1 < kCanonicalElements.size() ? "," : "") << "\n";
        }

        out << "      ]\n";
        out << "    }" << (s + 1 < kSectionProbes.size() ? "," : "") << "\n";
    }

    out << "  ],\n";
    out << "  \"boundaries\": [\n";

    std::size_t boundary_count = 0;
    for (std::size_t e = 0; e < kSchneiderEdges.size(); ++e) {
        const int edge = kSchneiderEdges[e];
        for (const int delta : {-1, 0, 1}) {
            const int hu = edge + delta;
            if (hu < -1000 || hu >= 2996) continue;

            const G4String mat_name = MaterialNameFromHU(hu);
            const G4Material* material = G4Material::GetMaterial(mat_name, false);
            if (!material) continue;

            if (boundary_count > 0) out << ",\n";
            out << "    {\n";
            out << "      \"edge_hu\": " << edge << ",\n";
            out << "      \"delta\": " << delta << ",\n";
            out << "      \"tested_hu\": " << hu << ",\n";
            out << "      \"material_name\": \"" << material->GetName() << "\",\n";
            out << "      \"density_g_cm3\": " << (material->GetDensity() / (g / cm3)) << "\n";
            out << "    }";
            boundary_count++;
        }
    }

    out << "\n  ]\n";
    out << "}\n";
    out.close();

    G4cout << "CarbonSchneiderMaterialDump: Successfully dumped " << kSectionProbes.size()
           << " Schneider materials to " << output_path_ << G4endl;
}
