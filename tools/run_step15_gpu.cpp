#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/sha256.hpp"
#include "carbon/transport.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/multiple_scattering.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace {

struct TestCaseConfig {
    std::string id;
    std::string type;
    int section_id{0};
    double energy_mevu{200.0};
    double thickness_mm{40.0};
    uint64_t histories{100000};
    std::string cctg_file;
    std::string gpu_json_output;
    uint32_t nx{40};
    uint32_t ny{40};
    uint32_t nz{40};
    double spacing_x_mm{1.0};
    double spacing_y_mm{1.0};
    double spacing_z_mm{1.0};
    double origin_x_mm{-20.0};
    double origin_y_mm{-20.0};
    double origin_z_mm{0.0};
};

std::vector<TestCaseConfig> load_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + manifest_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<TestCaseConfig> cases;

    size_t pos = 0;
    while ((pos = content.find("\"id\": \"", pos)) != std::string::npos) {
        const size_t start_id = pos + 7;
        const size_t end_id = content.find("\"", start_id);
        const std::string id = content.substr(start_id, end_id - start_id);

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

        TestCaseConfig tc;
        tc.id = id;
        tc.type = extract_str("type");
        tc.section_id = static_cast<int>(extract_num("section_id"));
        tc.energy_mevu = extract_num("energy_mevu");
        tc.thickness_mm = extract_num("thickness_mm");
        tc.histories = static_cast<uint64_t>(extract_num("histories"));
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
            : "/mnt/sda/wuwei/step15_schneider_mcs/manifest.json";

        const auto cases = load_manifest(manifest_path);
        std::cout << "Loaded manifest with " << cases.size() << " test cases." << std::endl;

        const std::filesystem::path repo_dir = "/mnt/sdb/wuwei/MAIGO";
        const auto water_sp_path = repo_dir / "data/stopping_power_water_geant4_11_3_2.csv";
        const auto xs_path = repo_dir / "data/schneider/c12_schneider_inelastic_mass_xs.csv";
        const auto sp_bin_path = repo_dir / "data/schneider/schneider_stopping_v1.bin";
        const auto rad_len_json = repo_dir / "data/schneider/schneider_radiation_lengths.json";

        const auto water_sp = carbon::StoppingPowerTable::from_csv(water_sp_path);
        const carbon::CrossSectionTable zero_xs{{0.0, 1000.0}, {0.0, 0.0}};

        const std::string stopping_bin_sha = carbon::compute_file_sha256_hex(sp_bin_path);
        const std::string rad_len_sha = carbon::compute_file_sha256_hex(rad_len_json);

        for (const auto& tc : cases) {
            std::cout << "\n=======================================================\n"
                      << "Running GPU MCS simulation for: " << tc.id << " (" << tc.type << ")\n"
                      << "  Energy: " << tc.energy_mevu << " MeV/u, Thickness: " << tc.thickness_mm
                      << " mm, Histories: " << tc.histories << ", Grid: " << tc.nx << "x" << tc.ny << "x" << tc.nz
                      << "\n=======================================================" << std::endl;

            carbon::TransportConfig cfg;
            cfg.primary_atomic_number = 6;
            cfg.primary_mass_number = 12;
            cfg.initial_energy_MeVu = tc.energy_mevu;
            cfg.phantom_length_mm = tc.thickness_mm;
            cfg.depth_bin_width_mm = tc.spacing_z_mm;
            cfg.number_of_histories = tc.histories;

            cfg.enable_ct_grid = true;
            cfg.ct_grid_file = tc.cctg_file;
            cfg.ct_schneider_cross_section_file = xs_path.string();
            cfg.ct_schneider_stopping_power_file = sp_bin_path.string();
            cfg.ct_validation_mode = "primary-attenuation-only";
            cfg.nuclear_model = "geant4";
            cfg.enable_inelastic = true;
            cfg.enable_nuclear_elastic = false;
            cfg.enable_secondary_transport = false;
            cfg.enable_energy_straggling = false;
            cfg.beam_energy_spread = 0.0;
            cfg.ct_use_density_mass_spr = true;
            cfg.ct_stopping_power_scale = 1.0;
            cfg.energy_cutoff_MeV = 6.0;
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

            // MCS Configuration
            cfg.enable_multiple_scattering = true;
            cfg.enable_ct_material_mcs = true;
            cfg.multiple_scattering_model = "fred_2gr";
            cfg.fred_2gr_mcs_file = repo_dir / "data/packages/fred_3_76_mcs_2gr.bin";
            cfg.multiple_scattering_scale = 1.40;

            cfg.validate();

            const auto result = carbon::transport_sycl(cfg, water_sp, zero_xs, "default");

            // Process 3D dose to compute transverse spread metrics per slice
            const auto& voxel_dose = result.voxel_deposited_energy_MeV;
            const uint32_t nx = tc.nx;
            const uint32_t ny = tc.ny;
            const uint32_t nz = tc.nz;
            const double dx = tc.spacing_x_mm;
            const double dy = tc.spacing_y_mm;
            const double dz = tc.spacing_z_mm;

            std::vector<double> sigma_x_by_depth(nz, 0.0);
            std::vector<double> sigma_y_by_depth(nz, 0.0);
            std::vector<double> sigma_r_by_depth(nz, 0.0);
            std::vector<double> core_halo_ratio_by_depth(nz, 0.0);
            std::vector<double> slice_integrated_dose(nz, 0.0);

            for (uint32_t z = 0; z < nz; ++z) {
                double total_slice_dose = 0.0;
                double sum_x = 0.0, sum_x2 = 0.0;
                double sum_y = 0.0, sum_y2 = 0.0;

                std::vector<std::pair<double, double>> radial_voxels; // (radius, dose)

                for (uint32_t y = 0; y < ny; ++y) {
                    const double pos_y = (static_cast<double>(y) + 0.5 - 0.5 * ny) * dy;
                    for (uint32_t x = 0; x < nx; ++x) {
                        const double pos_x = (static_cast<double>(x) + 0.5 - 0.5 * nx) * dx;
                        const size_t v_idx = (z * ny + y) * nx + x;
                        const double d_val = voxel_dose[v_idx];

                        const double r = std::sqrt(pos_x * pos_x + pos_y * pos_y);
                        if (r <= 6.0) {
                            total_slice_dose += d_val;
                            sum_x += pos_x * d_val;
                            sum_x2 += pos_x * pos_x * d_val;
                            sum_y += pos_y * d_val;
                            sum_y2 += pos_y * pos_y * d_val;
                            radial_voxels.push_back({r, d_val});
                        }
                    }
                }

                slice_integrated_dose[z] = total_slice_dose;

                if (total_slice_dose > 0.0) {
                    const double mean_x = sum_x / total_slice_dose;
                    const double var_x = std::max(0.0, (sum_x2 / total_slice_dose) - mean_x * mean_x);
                    const double sx = std::sqrt(var_x);

                    const double mean_y = sum_y / total_slice_dose;
                    const double var_y = std::max(0.0, (sum_y2 / total_slice_dose) - mean_y * mean_y);
                    const double sy = std::sqrt(var_y);

                    const double sr = std::sqrt(0.5 * (var_x + var_y));

                    sigma_x_by_depth[z] = sx;
                    sigma_y_by_depth[z] = sy;
                    sigma_r_by_depth[z] = sr;

                    // Compute R50 and R80
                    std::sort(radial_voxels.begin(), radial_voxels.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });

                    double cum = 0.0;
                    double r50 = 0.0, r80 = 0.0;
                    for (const auto& rv : radial_voxels) {
                        cum += rv.second;
                        if (r50 == 0.0 && cum >= 0.50 * total_slice_dose) {
                            r50 = rv.first;
                        }
                        if (r80 == 0.0 && cum >= 0.80 * total_slice_dose) {
                            r80 = rv.first;
                            break;
                        }
                    }
                    if (r50 > 0.0) {
                        core_halo_ratio_by_depth[z] = r80 / r50;
                    }
                }
            }

            // Endpoint lateral profile along X (summed over Y at slice nz - 1)
            std::vector<double> endpoint_profile_x(nx, 0.0);
            const uint32_t end_z = nz - 1;
            for (uint32_t x = 0; x < nx; ++x) {
                double col_dose = 0.0;
                for (uint32_t y = 0; y < ny; ++y) {
                    col_dose += voxel_dose[(end_z * ny + y) * nx + x];
                }
                endpoint_profile_x[x] = col_dose;
            }

            // Write GPU JSON output
            std::filesystem::path out_path = tc.gpu_json_output;
            std::filesystem::create_directories(out_path.parent_path());
            std::ofstream out(out_path);

            out << std::setprecision(10);
            out << "{\n";
            out << "  \"schema_version\": 1,\n";
            out << "  \"id\": \"" << tc.id << "\",\n";
            out << "  \"type\": \"" << tc.type << "\",\n";
            out << "  \"energy_mevu\": " << tc.energy_mevu << ",\n";
            out << "  \"thickness_mm\": " << tc.thickness_mm << ",\n";
            out << "  \"nx\": " << nx << ",\n";
            out << "  \"ny\": " << ny << ",\n";
            out << "  \"nz\": " << nz << ",\n";
            out << "  \"spacing_x_mm\": " << dx << ",\n";
            out << "  \"spacing_y_mm\": " << dy << ",\n";
            out << "  \"spacing_z_mm\": " << dz << ",\n";
            out << "  \"histories\": " << tc.histories << ",\n";
            out << "  \"verified_stopping_power_sha256\": \"" << stopping_bin_sha << "\",\n";
            out << "  \"verified_radiation_length_sha256\": \"" << rad_len_sha << "\",\n";
            out << "  \"endpoint_sigma_r_mm\": " << sigma_r_by_depth[end_z] << ",\n";
            out << "  \"endpoint_core_halo_ratio\": " << core_halo_ratio_by_depth[end_z] << ",\n";

            out << "  \"sigma_r_by_depth_mm\": [";
            for (size_t i = 0; i < nz; ++i) {
                out << sigma_r_by_depth[i] << (i + 1 < nz ? ", " : "");
            }
            out << "],\n";

            out << "  \"core_halo_ratio_by_depth\": [";
            for (size_t i = 0; i < nz; ++i) {
                out << core_halo_ratio_by_depth[i] << (i + 1 < nz ? ", " : "");
            }
            out << "],\n";

            out << "  \"slice_integrated_dose_MeV\": [";
            for (size_t i = 0; i < nz; ++i) {
                out << slice_integrated_dose[i] << (i + 1 < nz ? ", " : "");
            }
            out << "],\n";

            out << "  \"endpoint_profile_x\": [";
            for (size_t i = 0; i < nx; ++i) {
                out << endpoint_profile_x[i] << (i + 1 < nx ? ", " : "");
            }
            out << "]\n";
            out << "}\n";

            std::cout << "  Endpoint sigma_r: " << sigma_r_by_depth[end_z] << " mm, R80/R50: "
                      << core_halo_ratio_by_depth[end_z] << std::endl;
            std::cout << "  Saved GPU result to: " << out_path << std::endl;
        }

        std::cout << "\nAll GPU MCS simulations completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in run_step15_gpu: " << e.what() << std::endl;
        return 1;
    }
}
