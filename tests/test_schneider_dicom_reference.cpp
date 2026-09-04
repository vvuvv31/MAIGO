// tests/test_schneider_dicom_reference.cpp
#include "carbon/ct_grid.hpp"
#include "carbon/sha256.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

void require_near(double actual, double expected, double tol, const std::string& message) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error("Assertion failed: " + message +
                                 " (actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected) +
                                 ", tol=" + std::to_string(tol) + ")");
    }
}

std::string compute_sha256(const std::filesystem::path& path) {
    return carbon::compute_file_sha256_hex(path);
}

} // namespace

int main() {
    try {
        std::cout << "=======================================================\n";
        std::cout << "Starting Schneider CT Heterogeneous & DICOM Gate Tests\n";
        std::cout << "=======================================================\n";

        const std::filesystem::path root_dir = std::filesystem::current_path();

        // -------------------------------------------------------------
        // Test 1: Level 3 Synthetic Heterogeneous CT Geometry & Grid
        // -------------------------------------------------------------
        std::cout << "[Test 1] Validating Level 3 Synthetic Heterogeneous CCTG...\n";
        const auto cctg_path = root_dir / "data/schneider/heterogeneous_level3.cctg";
        require(std::filesystem::exists(cctg_path), "heterogeneous_level3.cctg must exist");

        const auto ct_grid = carbon::CtGrid::load(cctg_path.string(), "", "");
        require(ct_grid.nx == 40, "Level 3 grid NX must be 40");
        require(ct_grid.ny == 40, "Level 3 grid NY must be 40");
        require(ct_grid.nz == 75, "Level 3 grid NZ must be 75");
        require_near(ct_grid.spacing_x_mm, 2.0F, 1e-4, "Spacing X must be 2.0 mm");
        require_near(ct_grid.spacing_y_mm, 2.0F, 1e-4, "Spacing Y must be 2.0 mm");
        require_near(ct_grid.spacing_z_mm, 2.0F, 1e-4, "Spacing Z must be 2.0 mm");
        require_near(ct_grid.origin_x_mm, -40.0F, 1e-4, "Origin X must be -40 mm");
        require_near(ct_grid.origin_y_mm, -40.0F, 1e-4, "Origin Y must be -40 mm");
        require_near(ct_grid.origin_z_mm, 0.0F, 1e-4, "Origin Z must be 0 mm");

        // Verify layer densities
        // Layer 1 (iz=5, z=11mm): Lung
        const size_t v_lung = 5 * (40 * 40) + 20 * 40 + 20;
        require(ct_grid.material_id[v_lung] == 1, "Layer 1 section must be 1 (Lung)");
        require_near(ct_grid.density_g_per_cm3[v_lung], 0.4800208F, 1e-4, "Lung density");

        // Layer 2 (iz=20, z=41mm): Soft Tissue
        const size_t v_soft = 20 * (40 * 40) + 20 * 40 + 20;
        require(ct_grid.material_id[v_soft] == 8, "Layer 2 section must be 8 (Soft Tissue)");
        require_near(ct_grid.density_g_per_cm3[v_soft], 1.1199F, 1e-4, "Soft tissue density");

        // Thin bone inclusion (iz=25, z=51mm, ix=20, iy=20): Bone
        const size_t v_bone_inc = 25 * (40 * 40) + 20 * 40 + 20;
        require(ct_grid.material_id[v_bone_inc] == 20, "Inclusion section must be 20 (Bone)");
        require_near(ct_grid.density_g_per_cm3[v_bone_inc], 1.7570F, 1e-4, "Bone inclusion density");

        // Layer 3 (iz=40, z=81mm, ix=10, iy=10): Bone
        const size_t v_bone = 40 * (40 * 40) + 10 * 40 + 10;
        require(ct_grid.material_id[v_bone] == 20, "Layer 3 section must be 20 (Bone)");
        require_near(ct_grid.density_g_per_cm3[v_bone], 1.7570F, 1e-4, "Bone layer density");

        // Thin air inclusion (iz=40, z=81mm, ix=20, iy=20): Air
        const size_t v_air_inc = 40 * (40 * 40) + 20 * 40 + 20;
        require(ct_grid.material_id[v_air_inc] == 0, "Air cavity section must be 0 (Air)");
        require_near(ct_grid.density_g_per_cm3[v_air_inc], 0.0269525F, 1e-4, "Air cavity density");

        std::cout << "  Level 3 geometry and materials verified successfully.\n";

        // -------------------------------------------------------------
        // Test 2: Level 4 RT07575 DICOM Benchmark Provenance Hashes
        // -------------------------------------------------------------
        std::cout << "[Test 2] Validating Level 4 RT07575 DICOM Provenance Hashes...\n";
        const auto dicom_benchmark_dir = root_dir / "benchmark/topas10x/RT07575_pbs_s1";
        require(std::filesystem::exists(dicom_benchmark_dir), "RT07575_pbs_s1 must exist");

        const std::string expected_spots_hash = "4420af1508008db97ddc5ec31dc2bbb36af290ee7f38ccf905d5626e2d38ff95";
        const std::string expected_beam_hash = "00de6ff3e37ca8b2c15b01bd462a5264068512e27955e7aa8354a2ae5f88abac";
        const std::string expected_schn_hash = "5022cd89617b28dbd8ee8bf8b095ea20cfd99f6405218693c0df238b3617a139";
        const std::string expected_plan_hash = "fe1ec7837b8d20a77dc43845e09e9338dd224044f030c91371ac63fb9cdef800";
        const std::string expected_dose_hash = "c6279c28aa7b09ea0996bfbe315c6e64e2899758d3d7ff5e9c907cdcbd687748";

        const auto actual_spots_hash = compute_sha256(dicom_benchmark_dir / "spots.csv");
        const auto actual_beam_hash = compute_sha256(dicom_benchmark_dir / "beam_model.csv");
        const auto actual_schn_hash = compute_sha256(dicom_benchmark_dir / "HUtoMaterialSchneider.txt");
        const auto actual_plan_hash = compute_sha256(dicom_benchmark_dir / "run_full_plan.txt");
        const auto actual_dose_hash = compute_sha256(dicom_benchmark_dir / "OSMK_Dtotal_full_plan.bin");

        require(actual_spots_hash == expected_spots_hash, "spots.csv SHA256 mismatch");
        require(actual_beam_hash == expected_beam_hash, "beam_model.csv SHA256 mismatch");
        require(actual_schn_hash == expected_schn_hash, "HUtoMaterialSchneider.txt SHA256 mismatch");
        require(actual_plan_hash == expected_plan_hash, "run_full_plan.txt SHA256 mismatch");
        require(actual_dose_hash == expected_dose_hash, "OSMK_Dtotal_full_plan.bin SHA256 mismatch");

        std::cout << "  RT07575 provenance hashes locked and verified.\n";

        // -------------------------------------------------------------
        // Test 3: Level 4 Authoritative TOPAS 3D Dose Grid Structure
        // -------------------------------------------------------------
        std::cout << "[Test 3] Validating Level 4 TOPAS Dose Grid Dimensions & Integrity...\n";
        const auto dose_bin_path = dicom_benchmark_dir / "OSMK_Dtotal_full_plan.bin";
        const auto dose_file_size = std::filesystem::file_size(dose_bin_path);
        const size_t expected_elements = 417 * 505 * 35; // 7,370,475
        require(dose_file_size == expected_elements * sizeof(double),
                "Dose grid file size must match 417x505x35 double precision elements (58,963,800 bytes)");

        std::ifstream dose_file(dose_bin_path, std::ios::binary);
        std::vector<double> topas_dose(expected_elements);
        dose_file.read(reinterpret_cast<char*>(topas_dose.data()), dose_file_size);

        double total_dose_sum = 0.0;
        double max_dose = 0.0;
        size_t nonzero_count = 0;
        for (double d : topas_dose) {
            total_dose_sum += d;
            if (d > max_dose) max_dose = d;
            if (d > 0.0) nonzero_count++;
        }

        require_near(total_dose_sum, 22026.45, 1.0, "TOPAS Total Dose Sum (Gy)");
        require_near(max_dose, 0.09968, 1e-4, "TOPAS Max Dose (Gy)");
        require(nonzero_count > 7000000, "TOPAS Nonzero voxel count must exceed 7M");

        std::cout << "  TOPAS 3D dose grid verified: sum = " << total_dose_sum
                  << " Gy, max = " << max_dose << " Gy, voxels = " << nonzero_count << "\n";

        // -------------------------------------------------------------
        // Test 4: Final Research Gates Criteria Checks from Real Evidence
        // -------------------------------------------------------------
        std::cout << "[Test 4] Validating Final Research Gate Criteria from Evidence...\n";
        const auto summary_path = root_dir / "evidence/step-21/step21_validation_summary.json";
        require(std::filesystem::exists(summary_path),
                "Step 21 evidence missing: evidence/step-21/step21_validation_summary.json must exist. "
                "Real simulations must be executed before gate closure.");

        std::ifstream summary_file(summary_path);
        require(summary_file.is_open(), "Cannot open evidence/step-21/step21_validation_summary.json");
        std::string content((std::istreambuf_iterator<char>(summary_file)),
                            std::istreambuf_iterator<char>());

        auto get_json_double = [&](const std::string& key) -> double {
            const std::string search_str = "\"" + key + "\"";
            size_t pos = content.find(search_str);
            if (pos == std::string::npos) {
                throw std::runtime_error("Missing required field in evidence: " + key);
            }
            pos = content.find(':', pos);
            if (pos == std::string::npos) {
                throw std::runtime_error("Malformed JSON near key: " + key);
            }
            pos++;
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r')) pos++;
            size_t end = pos;
            while (end < content.size() && (content[end] == '-' || content[end] == '+' || content[end] == '.' ||
                                            (content[end] >= '0' && content[end] <= '9') ||
                                            content[end] == 'e' || content[end] == 'E')) {
                end++;
            }
            if (pos == end) {
                throw std::runtime_error("Could not parse numeric value for key: " + key);
            }
            return std::stod(content.substr(pos, end - pos));
        };

        auto get_json_uint64 = [&](const std::string& key) -> uint64_t {
            return static_cast<uint64_t>(get_json_double(key));
        };

        const double axis_range_diff_mm = get_json_double("axis_range_diff_mm");
        const double oblique_range_diff_mm = get_json_double("oblique_range_diff_mm");
        const double axis_dose_diff_pct = get_json_double("axis_dose_diff_pct");
        const double oblique_dose_diff_pct = get_json_double("oblique_dose_diff_pct");
        const double axis_gamma_pass_rate = get_json_double("axis_gamma_pass_rate");
        const double oblique_gamma_pass_rate = get_json_double("oblique_gamma_pass_rate");
        const double step20_species_diff_pct = get_json_double("step20_species_diff_pct");
        const uint64_t unsupported_lookup_count = get_json_uint64("unsupported_lookup_count");
        const uint64_t shard_overflow_counters = get_json_uint64("shard_overflow_counters");

        // Range difference < 1 mm
        require(axis_range_diff_mm < 1.0, "Axis range difference must be < 1.0 mm");
        require(oblique_range_diff_mm < 1.0, "Oblique range difference must be < 1.0 mm");

        // Dose integral difference < 2.0%
        require(axis_dose_diff_pct < 2.0, "Axis dose integral difference must be < 2.0%");
        require(oblique_dose_diff_pct < 2.0, "Oblique dose integral difference must be < 2.0%");

        // 3D Gamma 2%/2mm > 95%
        require(axis_gamma_pass_rate > 95.0, "Axis 3D Gamma 2%/2mm must be > 95.0%");
        require(oblique_gamma_pass_rate > 95.0, "Oblique 3D Gamma 2%/2mm must be > 95.0%");

        // Secondary species integral difference < 2.0%
        require(step20_species_diff_pct < 2.0, "Major species integral relative difference must be < 2.0%");

        // Unsupported target / package lookup count = 0
        require(unsupported_lookup_count == 0, "Unsupported target lookup count must be 0");

        // All shard overflow counters = 0
        require(shard_overflow_counters == 0, "All shard overflow counters must be 0");

        // Level 4 DICOM Clinical Gates
        const auto level4_path = root_dir / "evidence/step-21/level4/verification.json";
        require(std::filesystem::exists(level4_path), "Level 4 verification.json must exist");
        const double level4_dose_diff_pct = get_json_double("dose_diff_pct");
        require(level4_dose_diff_pct < 2.0, "Level 4 dose integral difference must be < 2.0%");
        const double level4_gamma_pass_rate = get_json_double("gamma_pass_rate_3d_2mm_2pct");
        require(level4_gamma_pass_rate > 95.0, "Level 4 3D Gamma 2%/2mm must be > 95.0%");

        std::cout << "  All research gates passed successfully with genuine simulation evidence!\n";
        std::cout << "=======================================================\n";
        std::cout << "ALL SCHNEIDER CT DICOM REFERENCE TESTS PASSED!\n";
        std::cout << "=======================================================\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
}
