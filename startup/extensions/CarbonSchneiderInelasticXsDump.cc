// Scorer for CarbonSchneiderInelasticXsDump
// Extracts Geant4/TOPAS C12 inelastic cross sections across all 25 Schneider sections
// with strict process attachment verification and complete provenance recording.

#include "CarbonSchneiderInelasticXsDump.hh"

#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4Exception.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4UIcommand.hh"
#include "G4VProcess.hh"

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

CarbonSchneiderInelasticXsDump::CarbonSchneiderInelasticXsDump(
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
    output_json_path_ = output_file;
    if (output_json_path_.empty()) {
        output_json_path_ = "topas-c12-schneider-inelastic-xs.json";
    }

    if (fPm->ParameterExists(GetFullParmName("OutputCsvFile"))) {
        output_csv_path_ = fPm->GetStringParameter(GetFullParmName("OutputCsvFile"));
    } else {
        output_csv_path_ = "c12_schneider_inelastic_cross_sections_geant4_11_3_2.csv";
    }

    if (fPm->ParameterExists(GetFullParmName("MinEnergyMeVu"))) {
        min_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MinEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("MaxEnergyMeVu"))) {
        max_energy_mevu_ = fPm->GetDoubleParameter(GetFullParmName("MaxEnergyMeVu"), "Energy") / MeV;
    }
    if (fPm->ParameterExists(GetFullParmName("EnergyStepMeVu"))) {
        energy_step_mevu_ = fPm->GetDoubleParameter(GetFullParmName("EnergyStepMeVu"), "Energy") / MeV;
    }

    if (min_energy_mevu_ <= 0.0 || max_energy_mevu_ < min_energy_mevu_ || energy_step_mevu_ <= 0.0) {
        G4Exception("CarbonSchneiderInelasticXsDump", "InvalidEnergyGridParameters", FatalException,
                    "Invalid energy grid parameters: Min <= 0 or Max < Min or Step <= 0!");
    }
}

CarbonSchneiderInelasticXsDump::~CarbonSchneiderInelasticXsDump() = default;

G4bool CarbonSchneiderInelasticXsDump::ProcessHits(G4Step*, G4TouchableHistory*) {
    if (!dumped_) {
        DumpCrossSections();
        dumped_ = true;
    }
    return true;
}

void CarbonSchneiderInelasticXsDump::Output() {
    if (!dumped_) {
        DumpCrossSections();
        dumped_ = true;
    }
}

void CarbonSchneiderInelasticXsDump::DumpCrossSections() {
    // 1. Locate C12 ion definition
    G4ParticleTable* particle_table = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* projectile = particle_table->GetIonTable()->GetIon(6, 12, 0.0);
    if (!projectile) {
        G4Exception("CarbonSchneiderInelasticXsDump", "MissingProjectile", FatalException,
                    "Failed to find C12 ion definition in G4ParticleTable!");
    }

    // 2. Strict Process Verification: C12 must have exactly 1 attached hadronic inelastic process
    G4ProcessManager* pman = projectile->GetProcessManager();
    if (!pman) {
        G4Exception("CarbonSchneiderInelasticXsDump", "MissingProcessManager", FatalException,
                    "C12 projectile has null G4ProcessManager!");
    }

    G4ProcessVector* pvec = pman->GetProcessList();
    G4HadronicProcess* inelastic_process = nullptr;
    int inelastic_count = 0;
    for (std::size_t i = 0; i < pvec->size(); ++i) {
        G4VProcess* proc = (*pvec)[i];
        if (proc->GetProcessType() == fHadronic) {
            auto* hp = dynamic_cast<G4HadronicProcess*>(proc);
            if (hp && hp->GetProcessSubType() == fHadronInelastic) {
                inelastic_process = hp;
                inelastic_count++;
            }
        }
    }

    if (inelastic_count != 1 || !inelastic_process) {
        const G4String msg = "Expected exactly 1 attached hadronic inelastic process for C12, found " +
                             std::to_string(inelastic_count);
        G4Exception("CarbonSchneiderInelasticXsDump", "AmbiguousInelasticProcess", FatalException,
                    msg.c_str());
    }

    const G4String process_name = inelastic_process->GetProcessName();
    const G4int process_type = inelastic_process->GetProcessType();
    const G4int process_sub_type = inelastic_process->GetProcessSubType();

    G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
    if (!store) {
        G4Exception("CarbonSchneiderInelasticXsDump", "MissingProcessStore", FatalException,
                    "HadronicProcessStore unavailable!");
    }

    // 3. Build energy grid
    std::vector<double> energies;
    for (double e = min_energy_mevu_; e <= max_energy_mevu_ + 1e-9; e += energy_step_mevu_) {
        energies.push_back(e);
    }
    if (energies.empty()) {
        G4Exception("CarbonSchneiderInelasticXsDump", "EmptyEnergyGrid", FatalException,
                    "Constructed energy grid is empty!");
    }

    // 4. Fetch materials for all 25 sections (fail-fast on missing material)
    std::vector<const G4Material*> materials(25, nullptr);
    for (std::size_t s = 0; s < 25; ++s) {
        const G4String mat_name = MaterialNameFromHU(kSectionProbes[s].rep_hu);
        materials[s] = G4Material::GetMaterial(mat_name, false);
        if (!materials[s]) {
            const G4String msg = "Missing Schneider material: " + mat_name;
            G4Exception("CarbonSchneiderInelasticXsDump", "MissingMaterial", FatalException, msg.c_str());
        }
    }

    // 5. Structure for JSON recording
    struct ElementRateDetail {
        std::string name;
        int z;
        double a_g_mol;
        double mass_fraction;
        double atom_density_per_mm3;
        double atom_density_per_cm3;
        double microscopic_sigma_barn;
        double partial_macro_per_mm;
        double mass_partial_per_mm_at_1g_cm3;
    };

    struct EnergyPointData {
        double energy_mevu;
        double total_energy_mev;
        double total_macro_per_mm;
        double mass_total_per_mm_at_1g_cm3;
        double partial_sum_macro_per_mm;
        double partial_sum_discrepancy;
        std::array<ElementRateDetail, 13> elements;
    };

    std::vector<std::vector<EnergyPointData>> grid_data(25);
    for (std::size_t s = 0; s < 25; ++s) {
        grid_data[s].reserve(energies.size());
        const G4Material* mat = materials[s];
        const G4double density = mat->GetDensity() / (g / cm3);
        const std::size_t n_elem = mat->GetNumberOfElements();
        const G4ElementVector* elem_vec = mat->GetElementVector();
        const G4double* atom_dens_vec = mat->GetVecNbOfAtomsPerVolume();
        const G4double* frac_vec = mat->GetFractionVector();

        for (const double e_mevu : energies) {
            const G4double total_kinetic_energy = 12.0 * e_mevu * MeV;
            const G4double total_macro = store->GetInelasticCrossSectionPerVolume(
                projectile, total_kinetic_energy, mat) / (1.0 / mm);
            const G4double mass_total = total_macro / density;

            EnergyPointData pt{};
            pt.energy_mevu = e_mevu;
            pt.total_energy_mev = 12.0 * e_mevu;
            pt.total_macro_per_mm = total_macro;
            pt.mass_total_per_mm_at_1g_cm3 = mass_total;

            double sum_partial = 0.0;
            for (std::size_t c = 0; c < kCanonicalElements.size(); ++c) {
                const int canon_z = kCanonicalZ[c];
                const char* canon_name = kCanonicalElements[c];

                ElementRateDetail el_det{};
                el_det.name = canon_name;
                el_det.z = canon_z;
                el_det.a_g_mol = 0.0;
                el_det.mass_fraction = 0.0;
                el_det.atom_density_per_mm3 = 0.0;
                el_det.atom_density_per_cm3 = 0.0;
                el_det.microscopic_sigma_barn = 0.0;
                el_det.partial_macro_per_mm = 0.0;
                el_det.mass_partial_per_mm_at_1g_cm3 = 0.0;

                for (std::size_t i = 0; i < n_elem; ++i) {
                    const G4Element* elem = (*elem_vec)[i];
                    if (elem->GetZ() == canon_z || elem->GetName() == canon_name) {
                        const G4double micro = store->GetInelasticCrossSectionPerAtom(
                            projectile, total_kinetic_energy, elem, mat) / barn;
                        const G4double atom_density_per_mm3 = atom_dens_vec[i] / (1.0 / mm3);
                        const G4double partial_macro = atom_dens_vec[i] * (micro * barn) / (1.0 / mm);

                        el_det.a_g_mol = elem->GetA() / (g / mole);
                        el_det.mass_fraction = frac_vec[i];
                        el_det.atom_density_per_mm3 = atom_density_per_mm3;
                        el_det.atom_density_per_cm3 = atom_density_per_mm3 * 1000.0;
                        el_det.microscopic_sigma_barn = micro;
                        el_det.partial_macro_per_mm = partial_macro;
                        el_det.mass_partial_per_mm_at_1g_cm3 = partial_macro / density;

                        sum_partial += partial_macro;
                        break;
                    }
                }
                pt.elements[c] = el_det;
            }

            pt.partial_sum_macro_per_mm = sum_partial;
            pt.partial_sum_discrepancy = std::abs(total_macro - sum_partial);

            // Fail-fast if partial sum diverges from total material rate
            if (pt.partial_sum_discrepancy > 1.0e-5 * std::max(total_macro, 1.0e-6)) {
                G4Exception("CarbonSchneiderInelasticXsDump", "PartialSumDiscrepancyExceeded", FatalException,
                            "Partial element cross sections sum diverges from total material rate!");
            }

            grid_data[s].push_back(pt);
        }
    }

    // 6. Write CSV file compatible with CrossSectionTable::from_schneider_csv
    if (!output_csv_path_.empty()) {
        std::filesystem::path csv_p(std::string(output_csv_path_.c_str()));
        if (!csv_p.parent_path().empty()) {
            std::filesystem::create_directories(csv_p.parent_path());
        }
        std::ofstream csv(csv_p);
        if (!csv) {
            const G4String msg = "Cannot open CSV output file: " + output_csv_path_;
            G4Exception("CarbonSchneiderInelasticXsDump", "CannotOpenCsv", FatalException, msg.c_str());
        }

        csv << std::setprecision(10);
        csv << "energy_MeV_per_u";
        for (std::size_t s = 0; s < 25; ++s) {
            csv << ",section_" << (s < 10 ? "0" : "") << s << "_mass_xs_per_mm_at_1g_cm3";
        }
        csv << "\n";

        for (std::size_t e_idx = 0; e_idx < energies.size(); ++e_idx) {
            csv << energies[e_idx];
            for (std::size_t s = 0; s < 25; ++s) {
                csv << "," << grid_data[s][e_idx].mass_total_per_mm_at_1g_cm3;
            }
            csv << "\n";
        }
        csv.close();
        G4cout << "CarbonSchneiderInelasticXsDump: Successfully wrote CSV to " << output_csv_path_ << G4endl;
    }

    // 7. Write Detailed JSON artifact
    if (!output_json_path_.empty()) {
        std::filesystem::path json_p(std::string(output_json_path_.c_str()));
        if (!json_p.parent_path().empty()) {
            std::filesystem::create_directories(json_p.parent_path());
        }
        std::ofstream jout(json_p);
        if (!jout) {
            const G4String msg = "Cannot open JSON output file: " + output_json_path_;
            G4Exception("CarbonSchneiderInelasticXsDump", "CannotOpenJson", FatalException, msg.c_str());
        }

        jout << std::setprecision(12);
        jout << "{\n";
        jout << "  \"schema_version\": 1,\n";
        jout << "  \"extractor\": \"CarbonSchneiderInelasticXsDump\",\n";
        jout << "  \"process_provenance\": {\n";
        jout << "    \"process_name\": \"" << process_name << "\",\n";
        jout << "    \"process_type\": " << process_type << ",\n";
        jout << "    \"process_sub_type\": " << process_sub_type << ",\n";
        jout << "    \"process_type_name\": \"fHadronic\",\n";
        jout << "    \"process_sub_type_name\": \"fHadronInelastic\",\n";
        jout << "    \"dataset_model_family\": \"Glauber-Gribov / G4HadronInelasticDataSet\"\n";
        jout << "  },\n";
        jout << "  \"projectile\": \"C12\",\n";
        jout << "  \"projectile_z\": 6,\n";
        jout << "  \"projectile_a\": 12,\n";
        jout << "  \"min_energy_mevu\": " << min_energy_mevu_ << ",\n";
        jout << "  \"max_energy_mevu\": " << max_energy_mevu_ << ",\n";
        jout << "  \"energy_points_count\": " << energies.size() << ",\n";
        jout << "  \"canonical_elements\": [\n";
        for (std::size_t c = 0; c < kCanonicalElements.size(); ++c) {
            jout << "    {\"canonical_index\": " << c
                 << ", \"name\": \"" << kCanonicalElements[c]
                 << "\", \"z\": " << kCanonicalZ[c] << "}"
                 << (c + 1 < kCanonicalElements.size() ? "," : "") << "\n";
        }
        jout << "  ],\n";
        jout << "  \"sections\": [\n";

        for (std::size_t s = 0; s < 25; ++s) {
            const G4Material* mat = materials[s];
            jout << "    {\n";
            jout << "      \"section_id\": " << s << ",\n";
            jout << "      \"representative_HU\": " << kSectionProbes[s].rep_hu << ",\n";
            jout << "      \"material_name\": \"" << mat->GetName() << "\",\n";
            jout << "      \"density_g_cm3\": " << (mat->GetDensity() / (g / cm3)) << ",\n";
            jout << "      \"grid\": [\n";

            for (std::size_t e_idx = 0; e_idx < energies.size(); ++e_idx) {
                const auto& pt = grid_data[s][e_idx];
                jout << "        {\n";
                jout << "          \"energy_mevu\": " << pt.energy_mevu << ",\n";
                jout << "          \"total_energy_mev\": " << pt.total_energy_mev << ",\n";
                jout << "          \"direct_material_macro_per_mm\": " << pt.total_macro_per_mm << ",\n";
                jout << "          \"summed_macro_per_mm\": " << pt.partial_sum_macro_per_mm << ",\n";
                jout << "          \"partial_sum_discrepancy\": " << pt.partial_sum_discrepancy << ",\n";
                jout << "          \"mass_total_per_mm_at_1g_cm3\": " << pt.mass_total_per_mm_at_1g_cm3 << ",\n";
                jout << "          \"elements\": [\n";

                for (std::size_t c = 0; c < 13; ++c) {
                    const auto& el = pt.elements[c];
                    jout << "            {\n";
                    jout << "              \"canonical_index\": " << c << ",\n";
                    jout << "              \"target_name\": \"" << el.name << "\",\n";
                    jout << "              \"target_z\": " << el.z << ",\n";
                    jout << "              \"target_atomic_mass_g_mol\": " << el.a_g_mol << ",\n";
                    jout << "              \"mass_fraction\": " << el.mass_fraction << ",\n";
                    jout << "              \"atom_density_per_mm3\": " << el.atom_density_per_mm3 << ",\n";
                    jout << "              \"atom_density_per_cm3\": " << el.atom_density_per_cm3 << ",\n";
                    jout << "              \"microscopic_sigma_barn\": " << el.microscopic_sigma_barn << ",\n";
                    jout << "              \"partial_macro_per_mm\": " << el.partial_macro_per_mm << ",\n";
                    jout << "              \"mass_partial_per_mm_at_1g_cm3\": " << el.mass_partial_per_mm_at_1g_cm3 << "\n";
                    jout << "            }" << (c + 1 < 13 ? "," : "") << "\n";
                }

                jout << "          ]\n";
                jout << "        }" << (e_idx + 1 < energies.size() ? "," : "") << "\n";
            }

            jout << "      ]\n";
            jout << "    }" << (s + 1 < 25 ? "," : "") << "\n";
        }

        jout << "  ]\n";
        jout << "}\n";
        jout.close();
        G4cout << "CarbonSchneiderInelasticXsDump: Successfully wrote JSON to " << output_json_path_ << G4endl;
    }
}
