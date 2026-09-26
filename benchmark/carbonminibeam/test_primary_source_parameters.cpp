#include "carbon/source_parameters.hpp"
#include "carbon/transport_config.hpp"
#include "carbon/all_ion_elastic.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/sha256.hpp"
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#include "carbon/minibeam_collimator.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/rng.hpp"
#include "carbon/water_electron_response.hpp"
namespace carbon { namespace {
#include "detail/sycl_device_math.inc"
} }
#endif

namespace {
int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f, const char* fragment) {
    try { f(); } catch (const std::exception& e) {
        require(std::string(e.what()).find(fragment) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error(std::string("Expected rejection: ") + fragment);
}
struct Temp {
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("maigo-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { std::filesystem::create_directories(dir); }
    ~Temp() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
    auto write(const char* name, const std::string& text) const {
        auto p = dir / name; std::ofstream(p) << text; return p;
    }
};
void replace_once(std::string& s, const std::string& from, const std::string& to) {
    const auto pos = s.find(from);
    require(pos != std::string::npos, "fixture token missing");
    s.replace(pos, from.size(), to);
}
}

int main(int argc, char** argv) {
    try {
        using namespace carbon;
        Temp tmp;
        const auto proton_yaml = tmp.write("proton.yaml",
            "primary_particle: proton\ninitial_energy_MeV: 150\nbeam_energy_spread_percent: 0.5\n");
        const auto proton = load_primary_source_parameters(proton_yaml);
        require(proton.ion.atomic_number == 1 && proton.ion.mass_number == 1, "proton identity");
        require(proton.initial_total_energy_MeV() == 150 && proton.initial_energy_MeVu == 150, "proton energy");
        require(std::abs(proton.ion.rest_mass_MeV - 938.27208816) < 1e-9, "proton nuclear mass");
        require(proton.beam_energy_spread == 0.005, "percent to relative RMS");
        rejects([&] { load_config(proton_yaml); }, "Non-C12 primary requires");
        const auto c12 = load_primary_source_parameters(tmp.write("c12.yaml",
            "primary_particle: C12\ninitial_energy_MeV: 3000\nbeam_energy_spread: 0.005\n"));
        require(c12.initial_energy_MeVu == 250 && c12.initial_total_energy_MeV() == 3000, "C12 total energy conversion");
        const auto numeric = load_primary_source_parameters(tmp.write("numeric.yaml",
            "primary_atomic_number: 6\nprimary_mass_number: 12\ninitial_energy_MeVu: 250\n"));
        require(numeric.ion.rest_mass_MeV == 12 * ion_nucleon_rest_mass_MeV, "legacy C12 mass preserved");
        require(numeric.initial_total_energy_MeV() == c12.initial_total_energy_MeV(), "legacy energy compatibility");
        const auto custom = load_primary_source_parameters(tmp.write("ion.yaml",
            "primary_particle: ion\nprimary_atomic_number: 8\nprimary_mass_number: 16\n"
            "primary_rest_mass_MeV: 14895\ninitial_energy_MeV: 1600\n"));
        require(custom.ion.mass_number == 16 && custom.initial_energy_MeVu == 100, "custom ion and mass");
        for (const auto& [text, error] : {
            std::pair{"primary_particle: proton\nprimary_atomic_number: 6\nprimary_mass_number: 12\n", "conflicts"},
            {"primary_atomic_number: 1\n", "Specify both"},
            {"primary_particle: ion\n", "requires explicit"},
            {"primary_particle: positron\n", "Unknown primary_particle"},
            {"initial_energy_MeV: 150\ninitial_energy_MeVu: 150\n", "Specify only one"},
            {"beam_energy_spread: 0.01\nbeam_energy_spread_percent: 1\n", "Specify only one"},
            {"initial_energy_MeV: nan\n", "finite and positive"},
            {"initial_energy_MeV: inf\n", "finite and positive"},
            {"beam_energy_spread_percent: nan\n", "finite"},
            {"primary_rest_mass_MeV: nan\n", "Invalid primary"},
            {"primary_rest_mass_MeV: -1\n", "Invalid primary"},
            {"primary_atomic_number: 8\nprimary_mass_number: 16\n", "explicit primary_rest_mass"},
            {"primary_atomic_number: 1\nprimary_mass_number: 1.5\n", "Invalid integer"},
            {"primary_particle: proton\nprimary_particle: c12\n", "Duplicate configuration key"}}) {
            rejects([&] { load_primary_source_parameters(tmp.write("invalid.yaml", text)); }, error);
        }
        TransportConfig config;
        require(config.primary_mcs_mass_override_MeV() == 0, "legacy MCS default unchanged");
        config.primary_atomic_number = 1; config.primary_mass_number = 1;
        require(config.primary_mcs_mass_override_MeV() == proton.ion.rest_mass_MeV, "proton MCS mass");
        config.primary_rest_mass_MeV = 940;
        require(config.primary_ion().rest_mass_MeV == config.primary_mcs_mass_override_MeV(), "explicit mass propagates");
        config.require_primary_reference_mass(940.0, "synthetic");
        rejects([&] { config.require_primary_reference_mass(931.494, "synthetic"); }, "disagrees");
        config.primary_urban_reference_particle = "C12_Z6_A12_charge6";
        rejects([&] { config.validate_primary_urban_reference_identity(); }, "conflicts with source");
        config.primary_urban_reference_particle.clear();
        config.primary_urban_ionisation_process = "ionIoni";
        rejects([&] { config.validate_primary_urban_reference_identity(); }, "conflicts with source");
        config.primary_urban_ionisation_process.clear();
        require(config.resolved_primary_urban_reference_particle() == "proton" &&
                config.resolved_primary_urban_ionisation_process() == "hIoni", "proton reference identity");
        require(primary_component_label(1, 1) == "primary_proton" &&
                primary_component_label(6, 12) == "primary_c12", "component identity / compatibility");
        require(primary_energy_band(150, 1) == 1 && primary_energy_band(150, 12) == 0, "ROI uses actual A");
        // New EM attachments must accept their own pins and reject stale parent/data pins.
        const auto moments_file = tmp.dir / "synthetic-moments.bin";
        {
            std::ofstream out(moments_file, std::ios::binary);
            const std::uint32_t header[]{2, 2};
            const float values[]{1, 2, 3, 4};
            out.write("EMDMOMT2", 8);
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        const auto moments_hash = compute_file_sha256_hex(moments_file);
        const std::string synthetic_parent(64, 'a');
        require(load_delta_moments(moments_file, synthetic_parent, 2, moments_hash,
                                  synthetic_parent).at(3) == 4, "new delta moments accepted");
        rejects([&] { load_delta_moments(moments_file, synthetic_parent, 2, moments_hash,
                                       std::string(64, 'b')); }, "source-package SHA mismatch");
        rejects([&] { load_delta_moments(moments_file, synthetic_parent, 2, std::string(64, 'b'),
                                       synthetic_parent); }, "SHA256 mismatch");
        AllIonElasticView elastic;
        require(elastic.projectile(1, 1) == 0 && elastic.projectile(6, 12) == 16 &&
                elastic.projectile(8, 16) == -1, "elastic lookup by source identity");
#ifdef CARBON_HAS_SYCL
        // Metamorphic contract: at fixed total kinetic energy, Z and rest mass,
        // changing A must not change Highland kinematics or Urban MFP.
        // These are parameter-plumbing tests, not validation of proton MCS.
        const float mass = static_cast<float>(proton.ion.rest_mass_MeV);
        for (float energy : {50.0F, 150.0F, 230.0F}) {
            const float h1 = highland_projected_rms_angle_device(energy, 1, 1, 1.0F, 1.0F, 36.08F, mass);
            const float h12 = highland_projected_rms_angle_device(energy, 1, 12, 1.0F, 1.0F, 36.08F, mass);
            require(h1 > 0 && std::abs(h1-h12) < 2e-6F*h1, "Highland honors explicit mass independently of A");
            const float expected = highland_projected_rms_angle_with_mass_device(energy, 1, mass, 1.0F, 1.0F, 36.08F);
            require(std::abs(h1-expected) < 2e-6F*h1, "Highland agrees with explicit total-mass interface");
            for (int kind : {0,1}) {
                UrbanV2Material mat;
                mat.projectile_z = 1; mat.projectile_a = 1; mat.mass_mev = mass; mat.mfp_kind = kind;
                const float m1 = urban_v2_transport_mfp(mat, energy);
                mat.projectile_a = 12;
                const float m12 = urban_v2_transport_mfp(mat, energy);
                require(m1 > 0 && m1 == m12, "Urban MFP propagates actual primary mass");
            }
        }
#endif
        // Synthetic metadata contract, not a physical calibration/reference.
        UrbanLossRangeTable reference({1,2}, {1,2}, {1,1}, 0, 1, 1, 1,
            "VALID_ACTIVE_STEP_CONTEXT", "G4EmStandardPhysics_option4", "proton", "Water_75eV",
            "active_UrbanMsc_GetRange_bound_hIoni_GetRange",
            "active_UrbanMsc_GetEnergy_and_hIoni_GetKineticEnergy",
            "active_UrbanMsc_GetDEDX_bound_hIoni_GetDEDX_restricted", 0.01);
        require(reference.is_active_reference("proton", "hIoni", "Water_75eV", 0.01), "generic reference match");
        require(!reference.is_active_c12_reference("Water_75eV", 0.01), "no C12 substitution");
        require(!reference.is_active_reference("proton", "ionIoni", "Water_75eV", 0.01), "process mismatch rejected");
        require(!reference.is_active_reference("proton", "hIoni", "Water_75eV", 0.05), "cut mismatch rejected");
        require(!reference.is_active_reference("proton", "hIoni", "G4_Cu", 0.01), "material mismatch rejected");
        if (argc > 1) {
            // Parse existing YAML only; never load its old physical tables.
            const auto old_path = std::filesystem::path(argv[1]) / "config/unified_water_highland_smoke.yaml";
            const auto old = load_config(old_path);
            std::ifstream in(old_path);
            std::string updated((std::istreambuf_iterator<char>(in)), {});
            replace_once(updated, "primary_atomic_number: 6\nprimary_mass_number: 12", "primary_particle: c12");
            replace_once(updated, "initial_energy_MeVu: 200", "initial_energy_MeV: 2400");
            replace_once(updated, "beam_energy_spread: 0.01", "beam_energy_spread_percent: 1");
            replace_once(updated, "ct_schneider_c12_cinel03_file:", "ct_schneider_primary_cinel03_file:");
            const auto now = load_config(tmp.write("updated.yaml", updated));
            require(old.initial_total_energy_MeV() == now.initial_total_energy_MeV() &&
                    old.beam_energy_spread == now.beam_energy_spread &&
                    old.primary_ion().rest_mass_MeV == now.primary_ion().rest_mass_MeV,
                    "old/new YAML source equivalence");
            require(old.ct_schneider_primary_cinel03_file == now.ct_schneider_primary_cinel03_file,
                    "legacy primary package alias");
            const auto roundtrip = load_config(tmp.write("canonical.yaml", now.canonical_config_text));
            require(roundtrip.initial_total_energy_MeV() == 2400, "canonical config roundtrip");
            auto proton_config_text = updated;
            replace_once(proton_config_text, "primary_particle: c12", "primary_particle: proton");
            replace_once(proton_config_text, "initial_energy_MeV: 2400", "initial_energy_MeV: 150");
            const auto proton_config = load_config(tmp.write("proton-full.yaml", proton_config_text));
            require(proton_config.primary_atomic_number == 1 && proton_config.primary_mass_number == 1 &&
                    proton_config.initial_total_energy_MeV() == 150, "non-minibeam proton configuration accepted");
            if (std::filesystem::exists(proton_config.ct_schneider_primary_rate_file))
                rejects([&] { validate_schneider_ct_startup(proton_config); }, "Primary rate projectile does not match");
            auto new_package = proton_config_text;
            replace_once(new_package, delta_moments_source_sha256, std::string(64, 'a'));
            rejects([&] { load_config(tmp.write("unbound-em.yaml", new_package)); }, "Delta moments require");
            new_package += "\nem_delta_moments_sha256: " + std::string(64, 'b') +
                           "\nem_delta_moments_source_sha256: " + std::string(64, 'a') + "\n";
            const auto new_config = load_config(tmp.write("new-packages.yaml", new_package));
            require(new_config.em_package_sha256 == std::string(64, 'a') &&
                    new_config.em_delta_moments_sha256 == std::string(64, 'b'), "new package pins parsed");
            rejects([&] { load_config(tmp.write("alias-conflict.yaml", updated +
                "ct_schneider_c12_cinel03_file: unused.bin\n")); }, "Do not specify both");
        }
        std::cout << "PASS " << checks << " source/identity/data-contract checks; no MC or old data calibration\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
