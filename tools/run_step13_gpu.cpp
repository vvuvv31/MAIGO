#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/sha256.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TestCaseConfig {
    std::string id;
    std::string category;
    int section_id{0};
    double energy_mevu{100.0};
    double thickness_mm{10.0};
    int depth_bins{5};
    uint64_t histories{100000};
    double beam_angle_deg{0.0};
    std::string cctg_file;
    std::string gpu_json_output;
    uint32_t nx{20};
    uint32_t ny{20};
    uint32_t nz{5};
    double spacing_x_mm{2.0};
    double spacing_y_mm{2.0};
    double spacing_z_mm{2.0};
    double origin_x_mm{-20.0};
    double origin_y_mm{-20.0};
    double origin_z_mm{0.0};
    bool is_bragg_check{false};
};

std::vector<TestCaseConfig> load_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + manifest_path.string());
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();

    std::vector<TestCaseConfig> cases;
    size_t pos = 0;
    while ((pos = content.find("\"id\": \"", pos)) != std::string::npos) {
        pos += 7;
        const size_t end_id = content.find("\"", pos);
        const std::string id = content.substr(pos, end_id - pos);

        auto extract_str = [&](const std::string& key) -> std::string {
            const std::string pattern = "\"" + key + "\": \"";
            const size_t kpos = content.find(pattern, end_id);
            if (kpos == std::string::npos || kpos > content.find("\"id\": \"", end_id)) return "";
            const size_t val_start = kpos + pattern.length();
            const size_t val_end = content.find("\"", val_start);
            return content.substr(val_start, val_end - val_start);
        };

        auto extract_num = [&](const std::string& key) -> double {
            const std::string pattern = "\"" + key + "\": ";
            const size_t kpos = content.find(pattern, end_id);
            if (kpos == std::string::npos || kpos > content.find("\"id\": \"", end_id)) return 0.0;
            const size_t val_start = kpos + pattern.length();
            char* end_ptr = nullptr;
            return std::strtod(content.c_str() + val_start, &end_ptr);
        };

        auto extract_bool = [&](const std::string& key) -> bool {
            const std::string pattern = "\"" + key + "\": ";
            const size_t kpos = content.find(pattern, end_id);
            if (kpos == std::string::npos || kpos > content.find("\"id\": \"", end_id)) return false;
            return content.substr(kpos + pattern.length(), 4) == "true";
        };

        TestCaseConfig tc;
        tc.id = id;
        tc.category = extract_str("category");
        tc.section_id = static_cast<int>(extract_num("section_id"));
        tc.energy_mevu = extract_num("energy_mevu");
        tc.thickness_mm = extract_num("thickness_mm");
        tc.depth_bins = static_cast<int>(extract_num("depth_bins"));
        tc.histories = static_cast<uint64_t>(extract_num("histories"));
        tc.beam_angle_deg = extract_num("beam_angle_deg");
        tc.cctg_file = extract_str("cctg_file");
        tc.gpu_json_output = extract_str("gpu_json_output");
        tc.nx = static_cast<uint32_t>(extract_num("nx"));
        tc.ny = static_cast<uint32_t>(extract_num("ny"));
        tc.nz = static_cast<uint32_t>(extract_num("nz"));
        tc.spacing_x_mm = extract_num("spacing_x_mm");
        tc.spacing_y_mm = extract_num("spacing_y_mm");
        tc.spacing_z_mm = extract_num("spacing_z_mm");
        tc.origin_x_mm = extract_num("origin_x_mm");
        tc.origin_y_mm = extract_num("origin_y_mm");
        tc.origin_z_mm = extract_num("origin_z_mm");
        tc.is_bragg_check = extract_bool("is_bragg_check");

        cases.push_back(tc);
        pos = end_id;
    }
    return cases;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path manifest_path = (argc > 1) 
            ? argv[1] 
            : "/mnt/sda/wuwei/step13_primary_ct/manifest.json";

        const std::string target_id = (argc > 2) ? argv[2] : "";

        std::cout << "Loading manifest: " << manifest_path << std::endl;
        const auto cases = load_manifest(manifest_path);
        std::cout << "Found " << cases.size() << " test cases in manifest." << std::endl;

        const auto repo_dir = std::filesystem::path("/mnt/sdb/wuwei/MAIGO");
        const auto water_sp_path = repo_dir / "data/stopping_power_water_geant4_11_3_2.csv";
        const auto xs_path = repo_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";

        std::cout << "Loading StoppingPowerTable from " << water_sp_path << std::endl;
        const auto water_sp = carbon::StoppingPowerTable::from_csv(water_sp_path);
        const carbon::CrossSectionTable zero_xs{{0.0, 1000.0}, {0.0, 0.0}};

        for (const auto& tc : cases) {
            if (!target_id.empty() && tc.id != target_id) {
                continue;
            }
            std::cout << "\n=======================================================\n"
                      << "Running GPU simulation for: " << tc.id << " (" << tc.category << ")\n"
                      << "  Energy: " << tc.energy_mevu << " MeV/u, Thickness: " << tc.thickness_mm
                      << " mm, Histories: " << tc.histories << ", Grid: " << tc.nx << "x" << tc.ny << "x" << tc.nz
                      << "\n=======================================================" << std::endl;

            carbon::TransportConfig cfg;
            cfg.primary_atomic_number = 6;
            cfg.primary_mass_number = 12;
            cfg.initial_energy_MeVu = tc.energy_mevu;
            cfg.phantom_length_mm = tc.thickness_mm;
            cfg.depth_bin_width_mm = tc.thickness_mm / static_cast<double>(tc.depth_bins);
            cfg.number_of_histories = tc.histories;

            cfg.enable_ct_grid = true;
            cfg.ct_grid_file = tc.cctg_file;
            cfg.ct_schneider_cross_section_file = xs_path.string();
            cfg.ct_schneider_stopping_power_file = (repo_dir / "data/schneider/schneider_stopping_v1.bin").string();
            cfg.ct_validation_mode = "primary-attenuation-only";
            cfg.nuclear_model = "geant4";
            cfg.enable_inelastic = true;
            cfg.enable_nuclear_elastic = false;
            cfg.enable_secondary_transport = false;
            cfg.enable_energy_straggling = false;
            cfg.beam_energy_spread = 0.0;
            cfg.ct_use_density_mass_spr = true;
            cfg.ct_stopping_power_scale = 1.0;
            cfg.energy_cutoff_MeV = 6.0; // 0.5 MeV/u
            cfg.maximum_step_mm = 1.0;
            cfg.maximum_relative_energy_loss = 0.005;

            // 3D Voxel Scoring
            cfg.enable_voxel_scoring = true;
            cfg.voxel_bins_x = tc.nx;
            cfg.voxel_bins_y = tc.ny;
            cfg.voxel_bins_z = tc.nz;
            cfg.voxel_size_x_mm = tc.spacing_x_mm;
            cfg.voxel_size_y_mm = tc.spacing_y_mm;
            cfg.voxel_size_z_mm = tc.spacing_z_mm;
            cfg.voxel_origin_x_mm = tc.origin_x_mm;
            cfg.voxel_origin_y_mm = tc.origin_y_mm;
            cfg.voxel_origin_z_mm = tc.origin_z_mm;

            // Oblique beam setup
            if (tc.beam_angle_deg != 0.0) {
                const double rad = tc.beam_angle_deg * M_PI / 180.0;
                cfg.beam_uz_x = std::sin(rad);
                cfg.beam_uz_y = 0.0;
                cfg.beam_uz_z = std::cos(rad);
                cfg.source_origin_x_mm = 0.0;
                cfg.source_origin_y_mm = 0.0;
                cfg.source_origin_z_mm = 0.0;
            }

            cfg.validate();

            const auto result = carbon::transport_sycl(cfg, water_sp, zero_xs, "default");

            // Calculate 1D depth-dose IDD from 3D voxel dose
            const auto idd_dose = carbon::compute_idd_from_3d_voxel_dose(
                result.voxel_deposited_energy_MeV, tc.nx, tc.ny, tc.nz);
            const auto bp_metrics = carbon::compute_bragg_peak_metrics(
                idd_dose, tc.spacing_z_mm, tc.origin_z_mm);

            // Audit terminal conservation
            const uint64_t n_inelastic = result.nuclear_interactions;
            const uint64_t n_survived = (tc.histories > n_inelastic) ? (tc.histories - n_inelastic) : 0;
            const uint64_t stopped = result.primary_stopped_count;
            const uint64_t inelastic_term = result.primary_inelastic_terminated_count;
            const uint64_t other_term = result.primary_other_terminal_count;
            const uint64_t escaped = result.primary_escaped_ct_count;
            const uint64_t sum_terminal = stopped + inelastic_term + other_term + escaped;
            const bool exact_conservation = (sum_terminal == tc.histories);

            // Check section mapping audit on every first interaction
            uint64_t section_mapping_mismatch = 0;
            std::vector<uint64_t> first_interactions_by_expected_section(25, 0);
            std::vector<uint64_t> first_interactions_by_recorded_section(25, 0);

            for (const auto& rec : result.primary_first_interactions) {
                uint32_t expected_sec = 0;
                if (tc.category == "case3_staircase_25sec") {
                    expected_sec = static_cast<uint32_t>(std::clamp(
                        static_cast<int>(std::floor((rec.depth_mm - tc.origin_z_mm) / tc.spacing_z_mm)),
                        0, 24));
                } else {
                    expected_sec = static_cast<uint32_t>(tc.section_id);
                }

                if (expected_sec < 25) {
                    first_interactions_by_expected_section[expected_sec]++;
                }
                if (rec.section_id < 25) {
                    first_interactions_by_recorded_section[rec.section_id]++;
                }
                if (rec.section_id != expected_sec) {
                    section_mapping_mismatch++;
                }
            }

            // Real transport diagnostics from result
            const uint64_t gen_secondaries = result.generated_direct_secondaries;
            const uint64_t queued_sec = result.queued_secondaries;
            const uint64_t trans_sec = result.transported_secondaries;
            const uint64_t sec_overflow = result.secondary_queue_overflow;
            const uint64_t elastic_overflow = result.elastic_queue_overflow;
            const uint64_t neutral_overflow = result.neutral_queue_overflow;
            const uint64_t electron_overflow = result.electron_queue_overflow;
            const uint64_t fred_resample_failed = result.fred_resample_failed_events;
            const uint64_t fred_cap_overflow = result.fred_product_capacity_overflow_events;
            uint64_t replay_valid_sum = 0;
            for (const auto v : result.cinel02_replay_valid_counts) {
                replay_valid_sum += v;
            }

            std::cout << "  Simulated " << tc.histories << " histories.\n"
                      << "  Inelastic interactions: " << n_inelastic << " ("
                      << (100.0 * n_inelastic / tc.histories) << "%)\n"
                      << "  Survived: " << n_survived << "\n"
                      << "  Terminal counts: stopped=" << stopped << ", inelastic=" << inelastic_term
                      << ", other=" << other_term << ", escaped=" << escaped
                      << " (sum=" << sum_terminal << ", exact=" << (exact_conservation ? "YES" : "NO") << ")\n"
                      << "  Energy balance rel error: " << result.relative_energy_balance_error() << "\n"
                      << "  Section mapping mismatch: " << section_mapping_mismatch << "\n";
            if (bp_metrics.found_r80) {
                std::cout << "  Bragg Peak: depth=" << bp_metrics.peak_depth_mm << " mm, dose="
                          << bp_metrics.peak_dose_MeV << " MeV, R80=" << bp_metrics.r80_distal_mm
                          << " mm, R50=" << bp_metrics.r50_distal_mm << " mm\n";
            }

            // Write JSON result
            std::filesystem::path out_path = tc.gpu_json_output;
            std::filesystem::create_directories(out_path.parent_path());
            std::ofstream out(out_path);
            const std::string stopping_file_sha = carbon::compute_file_sha256_hex(cfg.ct_schneider_stopping_power_file);
            const auto meta_sidecar = repo_dir / "data/schneider/schneider_stopping_v1.metadata.json";
            const std::string stopping_meta_sha = carbon::compute_file_sha256_hex(meta_sidecar);

            out << std::setprecision(10);
            out << "{\n";
            out << "  \"schema_version\": 1,\n";
            out << "  \"id\": \"" << tc.id << "\",\n";
            out << "  \"category\": \"" << tc.category << "\",\n";
            out << "  \"section_id\": " << tc.section_id << ",\n";
            out << "  \"energy_mevu\": " << tc.energy_mevu << ",\n";
            out << "  \"thickness_mm\": " << tc.thickness_mm << ",\n";
            out << "  \"depth_bins\": " << tc.depth_bins << ",\n";
            out << "  \"histories\": " << tc.histories << ",\n";
            out << "  \"entering_primaries\": " << tc.histories << ",\n";
            out << "  \"verified_stopping_power_file\": \"" << cfg.ct_schneider_stopping_power_file.string() << "\",\n";
            out << "  \"verified_stopping_power_sha256\": \"" << stopping_file_sha << "\",\n";
            out << "  \"verified_stopping_metadata_sha256\": \"" << stopping_meta_sha << "\",\n";
            out << "  \"total_first_inelastic_count\": " << n_inelastic << ",\n";
            out << "  \"survived_count\": " << n_survived << ",\n";
            out << "  \"survival_fraction\": " << (static_cast<double>(n_survived) / static_cast<double>(tc.histories)) << ",\n";
            out << "  \"terminal_counts\": {\n";
            out << "    \"stopped\": " << stopped << ",\n";
            out << "    \"inelastic\": " << inelastic_term << ",\n";
            out << "    \"other\": " << other_term << ",\n";
            out << "    \"escaped\": " << escaped << "\n";
            out << "  },\n";
            out << "  \"exact_terminal_conservation\": " << (exact_conservation ? "true" : "false") << ",\n";
            out << "  \"section_mapping_mismatch\": " << section_mapping_mismatch << ",\n";
            out << "  \"relative_energy_balance_error\": " << result.relative_energy_balance_error() << ",\n";
            out << "  \"diagnostics\": {\n";
            out << "    \"generated_direct_secondaries\": " << gen_secondaries << ",\n";
            out << "    \"queued_secondaries\": " << queued_sec << ",\n";
            out << "    \"transported_secondaries\": " << trans_sec << ",\n";
            out << "    \"secondary_queue_overflow\": " << sec_overflow << ",\n";
            out << "    \"elastic_queue_overflow\": " << elastic_overflow << ",\n";
            out << "    \"neutral_queue_overflow\": " << neutral_overflow << ",\n";
            out << "    \"electron_queue_overflow\": " << electron_overflow << ",\n";
            out << "    \"primary_other_terminal_count\": " << other_term << ",\n";
            out << "    \"fred_resample_failed_events\": " << fred_resample_failed << ",\n";
            out << "    \"fred_product_capacity_overflow_events\": " << fred_cap_overflow << ",\n";
            out << "    \"cinel02_replay_valid_sum\": " << replay_valid_sum << "\n";
            out << "  },\n";
            out << "  \"first_interactions_by_expected_section\": [";
            for (size_t s = 0; s < 25; ++s) {
                out << first_interactions_by_expected_section[s] << (s + 1 < 25 ? ", " : "");
            }
            out << "],\n";
            out << "  \"first_interactions_by_recorded_section\": [";
            for (size_t s = 0; s < 25; ++s) {
                out << first_interactions_by_recorded_section[s] << (s + 1 < 25 ? ", " : "");
            }
            out << "],\n";

            out << "  \"bragg_peak_metrics\": {\n";
            out << "    \"found_r80\": " << (bp_metrics.found_r80 ? "true" : "false") << ",\n";
            out << "    \"found_r50\": " << (bp_metrics.found_r50 ? "true" : "false") << ",\n";
            out << "    \"peak_depth_mm\": " << bp_metrics.peak_depth_mm << ",\n";
            out << "    \"peak_dose_MeV\": " << bp_metrics.peak_dose_MeV << ",\n";
            out << "    \"r80_distal_mm\": " << (std::isnan(bp_metrics.r80_distal_mm) ? 0.0 : bp_metrics.r80_distal_mm) << ",\n";
            out << "    \"r50_distal_mm\": " << (std::isnan(bp_metrics.r50_distal_mm) ? 0.0 : bp_metrics.r50_distal_mm) << "\n";
            out << "  },\n";

            // Depth checkpoints (survival curve at (k+1)*dz)
            out << "  \"depth_checkpoints\": [\n";
            for (int k = 0; k < tc.depth_bins; ++k) {
                const double depth = (k + 1) * (tc.thickness_mm / tc.depth_bins);
                uint64_t surv = 0;
                if (k + 1 < tc.depth_bins) {
                    surv = (k + 1 < static_cast<int>(result.primary_survival_counts.size()))
                        ? result.primary_survival_counts[k + 1] : 0;
                } else {
                    surv = n_survived;
                }
                const double s_frac = static_cast<double>(surv) / static_cast<double>(tc.histories);
                out << "    {\n";
                out << "      \"checkpoint_index\": " << k << ",\n";
                out << "      \"depth_mm\": " << depth << ",\n";
                out << "      \"unreacted_count\": " << surv << ",\n";
                out << "      \"survival_fraction\": " << s_frac << "\n";
                out << "    }" << (k + 1 < tc.depth_bins ? ",\n" : "\n");
            }
            out << "  ],\n";

            // 1D IDD dose
            out << "  \"idd_dose_MeV\": [\n";
            for (size_t i = 0; i < idd_dose.size(); ++i) {
                out << "    " << idd_dose[i] << (i + 1 < idd_dose.size() ? ",\n" : "\n");
            }
            out << "  ],\n";

            // First interactions sample
            const size_t max_sample = std::min(size_t(5000), result.primary_first_interactions.size());
            out << "  \"first_interactions_count\": " << result.primary_first_interactions.size() << ",\n";
            out << "  \"first_interactions_sample\": [\n";
            for (size_t i = 0; i < max_sample; ++i) {
                const auto& rec = result.primary_first_interactions[i];
                out << "    {\n";
                out << "      \"x_mm\": " << rec.x_mm << ",\n";
                out << "      \"y_mm\": " << rec.y_mm << ",\n";
                out << "      \"depth_mm\": " << rec.depth_mm << ",\n";
                out << "      \"energy_MeVu\": " << rec.energy_MeVu << ",\n";
                out << "      \"section_id\": " << rec.section_id << ",\n";
                out << "      \"density_g_per_cm3\": " << rec.density_g_per_cm3 << "\n";
                out << "    }" << (i + 1 < max_sample ? ",\n" : "\n");
            }
            out << "  ]\n";
            out << "}\n";
            std::cout << "  Saved GPU result to: " << out_path << std::endl;
        }

        std::cout << "\nAll GPU simulations completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
