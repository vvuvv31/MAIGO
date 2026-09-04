// tests/test_secondary_rate_table_hardening.cpp
#include "carbon/secondary_rate_table.hpp"
#include "carbon/sha256.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

template <typename ExceptionType>
void require_throws(const std::function<void()>& fn, const std::string& message) {
    bool caught = false;
    try {
        fn();
    } catch (const ExceptionType&) {
        caught = true;
    } catch (const std::exception& e) {
        throw std::runtime_error("Unexpected exception type for " + message + ": " + e.what());
    }
    if (!caught) {
        throw std::runtime_error("Expected exception was NOT thrown for: " + message);
    }
}

// Scoped temp file that cleans itself up
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const std::string& name)
        : path(std::filesystem::temp_directory_path() / name) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    void write(const void* data, std::size_t size) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data), size);
    }
};

} // namespace

int main() {
    try {
        std::cout << "=======================================================\n";
        std::cout << "Starting SecondaryRateTable Hardening & Mutation Tests\n";
        std::cout << "=======================================================\n";

        const std::filesystem::path valid_bin = "data/schneider/secondary_inelastic_rates_v1.bin";
        const std::filesystem::path valid_meta = "data/schneider/secondary_inelastic_rates_v1.metadata.json";
        require(std::filesystem::exists(valid_bin), "valid_bin must exist");
        require(std::filesystem::exists(valid_meta), "valid_meta must exist");

        // -------------------------------------------------------------
        // Test 1: Valid baseline load
        // -------------------------------------------------------------
        std::cout << "[Test 1] Validating baseline SecondaryRateTable load...\n";
        const auto table = carbon::SecondaryRateTable::from_binary(valid_bin, valid_meta);
        require(table.num_projectiles() == 13, "Must have 13 projectiles");
        require(table.projectiles().size() == 13, "Projectiles size must be 13");
        require(table.mass_partial_rates().size() == 13 * 25 * 13 * 860, "Partial rates size");
        require(table.mass_total_rates().size() == 13 * 25 * 860, "Total rates size");
        require(table.projectile_index(2, 4) >= 0, "He4 must be present");
        require(table.projectile_index(1, 1) >= 0, "Proton must be present");
        require(table.projectile_index(6, 12) == -1, "C12 is primary, not secondary");

        // -------------------------------------------------------------
        // Test 2: Accessor boundary checks (std::out_of_range)
        // -------------------------------------------------------------
        std::cout << "[Test 2] Validating accessor out_of_range bounds checks...\n";
        require_throws<std::out_of_range>([&]() { table.mass_partial_rate(13, 0, 0, 0); }, "proj_idx >= 13");
        require_throws<std::out_of_range>([&]() { table.mass_partial_rate(0, 25, 0, 0); }, "section_id >= 25");
        require_throws<std::out_of_range>([&]() { table.mass_partial_rate(0, 0, 13, 0); }, "target_idx >= 13");
        require_throws<std::out_of_range>([&]() { table.mass_partial_rate(0, 0, 0, 860); }, "energy_idx >= 860");

        require_throws<std::out_of_range>([&]() { table.mass_total_rate(13, 0, 0); }, "total proj_idx >= 13");
        require_throws<std::out_of_range>([&]() { table.mass_total_rate(0, 25, 0); }, "total section_id >= 25");
        require_throws<std::out_of_range>([&]() { table.mass_total_rate(0, 0, 860); }, "total energy_idx >= 860");

        require_throws<std::out_of_range>([&]() { table.interpolate_mass_total(13, 0, 100.0); }, "interp proj_idx >= 13");
        require_throws<std::out_of_range>([&]() { table.interpolate_mass_total(0, 25, 100.0); }, "interp section_id >= 25");

        require_throws<std::invalid_argument>([&]() { carbon::SecondaryRateTable::target_index_from_z(99); }, "unsupported target Z=99");

        // Read baseline binary into memory for mutation testing
        std::ifstream raw_in(valid_bin, std::ios::binary);
        std::vector<char> base_data((std::istreambuf_iterator<char>(raw_in)),
                                    std::istreambuf_iterator<char>());
        require(base_data.size() == 31304208, "Baseline binary size check");

        // -------------------------------------------------------------
        // Test 3: Truncated file & Trailing bytes
        // -------------------------------------------------------------
        std::cout << "[Test 3] Validating truncation and trailing bytes checks...\n";
        {
            TempFile trunc_file("test_sec_trunc.bin");
            trunc_file.write(base_data.data(), 100);
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(trunc_file.path);
            }, "Truncated binary file (100 bytes)");
        }
        {
            TempFile trunc_file2("test_sec_trunc2.bin");
            trunc_file2.write(base_data.data(), base_data.size() - 1000);
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(trunc_file2.path);
            }, "Truncated binary payload");
        }
        {
            TempFile trailing_file("test_sec_trailing.bin");
            std::vector<char> with_trailing = base_data;
            with_trailing.resize(with_trailing.size() + 16, 0);
            trailing_file.write(with_trailing.data(), with_trailing.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(trailing_file.path);
            }, "Trailing bytes after valid binary");
        }

        // -------------------------------------------------------------
        // Test 4: Header mutations (magic, version, dimensions, grid)
        // -------------------------------------------------------------
        std::cout << "[Test 4] Validating header corruption protections...\n";
        {
            // Corrupt magic
            auto mutated = base_data;
            std::memcpy(mutated.data(), "BADMAGIC", 8);
            TempFile tf("test_sec_magic.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Bad magic");
        }
        {
            // Corrupt version
            auto mutated = base_data;
            uint32_t bad_ver = 2;
            std::memcpy(mutated.data() + 8, &bad_ver, 4);
            TempFile tf("test_sec_ver.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Bad version");
        }
        {
            // Corrupt num_projectiles
            auto mutated = base_data;
            uint32_t bad_np = 12;
            std::memcpy(mutated.data() + 12, &bad_np, 4);
            TempFile tf("test_sec_np.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Bad num_projectiles");
        }
        {
            // Corrupt energy_min with NaN
            auto mutated = base_data;
            double nan_val = std::numeric_limits<double>::quiet_NaN();
            std::memcpy(mutated.data() + 28, &nan_val, 8);
            TempFile tf("test_sec_nan_emin.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "NaN energy_min");
        }

        // -------------------------------------------------------------
        // Test 5: Projectiles & Targets mutation (duplicates, ordering)
        // -------------------------------------------------------------
        std::cout << "[Test 5] Validating projectiles and targets validation...\n";
        {
            // Duplicate projectiles: projectile[1] = projectile[0]
            auto mutated = base_data;
            std::memcpy(mutated.data() + 52 + 8, mutated.data() + 52, 8);
            TempFile tf("test_sec_dup_proj.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Duplicate projectiles");
        }
        {
            // Corrupted target ordering
            auto mutated = base_data;
            int32_t bad_tgt = 99;
            std::memcpy(mutated.data() + 52 + 13 * 8, &bad_tgt, 4);
            TempFile tf("test_sec_bad_tgt.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Corrupted canonical target ordering");
        }

        // -------------------------------------------------------------
        // Test 6: Rate payload corruption (negative, NaN, sum mismatch)
        // -------------------------------------------------------------
        std::cout << "[Test 6] Validating rate payload physical sanity checks...\n";
        {
            // Negative partial rate
            auto mutated = base_data;
            double neg_rate = -1.5;
            std::memcpy(mutated.data() + 208, &neg_rate, 8);
            TempFile tf("test_sec_neg_rate.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Negative partial rate");
        }
        {
            // NaN total rate
            auto mutated = base_data;
            double nan_rate = std::numeric_limits<double>::quiet_NaN();
            std::memcpy(mutated.data() + 208 + 3633500 * 8, &nan_rate, 8);
            TempFile tf("test_sec_nan_rate.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "NaN total rate");
        }
        {
            // Sum(partial) vs total mismatch: modify one partial rate cell
            auto mutated = base_data;
            double orig_rate = 0.0;
            std::memcpy(&orig_rate, mutated.data() + 208, 8);
            double bad_rate = orig_rate + 2.0;
            std::memcpy(mutated.data() + 208, &bad_rate, 8);
            TempFile tf("test_sec_sum_mismatch.bin");
            tf.write(mutated.data(), mutated.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(tf.path);
            }, "Sum(partial) vs total mismatch");
        }

        // -------------------------------------------------------------
        // Test 7: Metadata SHA-256 mismatch
        // -------------------------------------------------------------
        std::cout << "[Test 7] Validating companion metadata SHA-256 checks...\n";
        {
            TempFile bad_meta_file("bad_meta.json");
            const std::string bad_meta_content =
                "{\n"
                "  \"data_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
                "  \"binary_magic\": \"SCHN2RAT\"\n"
                "}\n";
            bad_meta_file.write(bad_meta_content.data(), bad_meta_content.size());
            require_throws<std::runtime_error>([&]() {
                carbon::SecondaryRateTable::from_binary(valid_bin, bad_meta_file.path);
            }, "Metadata SHA-256 mismatch");
        }

        std::cout << "All SecondaryRateTable hardening tests passed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SecondaryRateTable hardening test failed: " << e.what() << "\n";
        return 1;
    }
}
