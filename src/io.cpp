#include "carbon/io.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/particle.hpp"
#include "carbon/charged_species.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace carbon {
namespace {
constexpr double kMeV_to_joule=1.602176634e-13;
double voxel_mass_kg(const TransportConfig& config) {
    return config.voxel_size_x_mm * config.voxel_size_y_mm *
           config.scorer_spacing_z_mm() *
           config.water_density_g_per_cm3 * 1.0e-6;
}
struct ScorerMasses {
    double uniform_mass_kg{};
    std::shared_ptr<const std::vector<double>> medium;
    double operator[](std::size_t i)const {return medium?(*medium)[i]:uniform_mass_kg;}
};
ScorerMasses voxel_masses_kg(const TransportConfig& config) {
    if (!config.enable_ct_grid)return {voxel_mass_kg(config),{}};
    const auto count = config.number_of_voxels();
    const auto volume_mm3 = config.voxel_size_x_mm * config.voxel_size_y_mm *
        config.scorer_spacing_z_mm();
    const auto grid_owner = CtGrid::load(config.ct_grid_file);
    const auto& grid = *grid_owner;
    if (grid.nx != config.voxel_bins_x || grid.ny != config.voxel_bins_y ||
        grid.nz != config.number_of_bins() ||
        std::abs(static_cast<double>(grid.spacing_x_mm) - config.voxel_size_x_mm) > 1.0e-6 ||
        std::abs(static_cast<double>(grid.spacing_y_mm) - config.voxel_size_y_mm) > 1.0e-6 ||
        std::abs(static_cast<double>(grid.spacing_z_mm) - config.depth_bin_width_mm) > 1.0e-6) {
        throw std::invalid_argument(
            "CT dose-to-medium requires the voxel scorer grid to match the CT grid");
    }
    struct Cache {
        std::shared_ptr<const CtGrid> grid;
        double volume_mm3{};
        std::shared_ptr<const std::vector<double>> masses;
    };
    thread_local Cache cached;
    if(cached.grid!=grid_owner || cached.volume_mm3!=volume_mm3 || !cached.masses) {
        auto masses=std::make_shared<std::vector<double>>(count);
        for (std::size_t i = 0; i < count; ++i) {
            // Preserve the dose-to-medium density floor and FP64 evaluation.
            const auto density = std::max(1.0e-6, static_cast<double>(grid.density_g_per_cm3[i]));
            (*masses)[i] = volume_mm3 * density * 1.0e-6;
        }
        cached={grid_owner,volume_mm3,std::move(masses)};
    }
    return {0.,cached.masses};
}
double energy_MeV_to_dose_Gy(double energy_MeV, double mass_kg) {
    if (mass_kg <= 0.0) {
        throw std::invalid_argument("scorer mass must be positive for dose conversion");
    }
    return energy_MeV * kMeV_to_joule / mass_kg;
}
double scored_dose_Gy(const TransportConfig& config, double energy_MeV, double mass_kg) {
    return energy_MeV_to_dose_Gy(energy_MeV * config.dose_output_scale, mass_kg);
}
void ensure_parent_directory(const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}
std::string schneider_rate_provenance_json(const std::filesystem::path& path,
                                         int primary_z, int primary_a) {
    std::ostringstream out;
    out << std::setprecision(12);
    if (path.empty() || !std::filesystem::exists(path)) {
        return "{\"present\": false}";
    }
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    std::uint32_t version{0};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    const bool is_sec = (std::memcmp(magic, "SCHN2RAT", 8) == 0);
    const bool is_pri = (std::strncmp(magic, "SCHNRATE", 8) == 0);
    if (!in || (!is_sec && !is_pri)) {
        return "{\"present\": true, \"readable\": false}";
    }
    std::uint32_t np = 1, ns = 0, nt = 0, ne = 0;
    if (is_sec) {
        in.read(reinterpret_cast<char*>(&np), sizeof(np));
        in.read(reinterpret_cast<char*>(&ns), sizeof(ns));
        in.read(reinterpret_cast<char*>(&nt), sizeof(nt));
        in.read(reinterpret_cast<char*>(&ne), sizeof(ne));
    } else {
        in.read(reinterpret_cast<char*>(&ns), sizeof(ns));
        in.read(reinterpret_cast<char*>(&nt), sizeof(nt));
        in.read(reinterpret_cast<char*>(&ne), sizeof(ne));
    }
    double emin = 0.0, emax = 0.0, estep = 0.0;
    in.read(reinterpret_cast<char*>(&emin), sizeof(emin));
    in.read(reinterpret_cast<char*>(&emax), sizeof(emax));
    in.read(reinterpret_cast<char*>(&estep), sizeof(estep));
    std::vector<std::int32_t> keys;
    if (is_sec && np <= 64) {
        keys.resize(static_cast<std::size_t>(np) * 2);
        in.read(reinterpret_cast<char*>(keys.data()),
                static_cast<std::streamsize>(keys.size() * sizeof(std::int32_t)));
    }
    if (!in) {
        return "{\"present\": true, \"readable\": false}";
    }
    out << "{\"present\": true, \"path\": \"" << path.string() << "\", \"binary_magic\": \"" << std::string(magic, 8)
        << "\", \"binary_version\": " << version << ", \"num_projectiles\": " << np
        << ", \"projectiles\": [";
    if (is_sec) {
        for (std::uint32_t i = 0; i < np && 2 * i + 1 < keys.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << "{\"z\": " << keys[2 * i] << ", \"a\": " << keys[2 * i + 1] << "}";
        }
    } else {
        out << "{\"z\": " << primary_z << ", \"a\": " << primary_a
            << ", \"note\": \"single primary, identity checked against source\"}";
    }
    out << "], \"energy_grid\": {\"emin\": " << emin << ", \"emax\": " << emax
        << ", \"step\": " << estep << ", \"count\": " << ne
        << "}}";
    return out.str();
}
std::string schneider_package_provenance_json(const std::filesystem::path& path) {
    std::ostringstream out;
    if (path.empty() || !std::filesystem::exists(path)) {
        return "{\"present\": false}";
    }
    out << "{\"present\": true, \"path\": \"" << path.string() << "\"";
    const std::filesystem::path sidecar =
        path.parent_path() / (path.stem().string() + ".channels.json");
    if (std::filesystem::exists(sidecar)) {
        out << ", \"channels_file\": \"" << sidecar.string() << "\"";
    }
    out << "}";
    return out.str();
}
void write_schneider_physics_provenance(std::ofstream& output, const TransportConfig& config) {
    const std::filesystem::path primary_rate =
        config.ct_schneider_primary_rate_file;
    const std::filesystem::path secondary_rate =
        config.ct_schneider_secondary_rate_file;
    const std::filesystem::path primary_pkg =
        config.ct_schneider_primary_cinel03_file;
    const std::filesystem::path secondary_pkg =
        config.ct_schneider_secondary_cinel03_file;
    output << "  \"schneider_physics_provenance\": {\n"
           << "    \"primary_rate\": " << schneider_rate_provenance_json(primary_rate, config.primary_atomic_number, config.primary_mass_number) << ",\n"
           << "    \"secondary_rate\": " << schneider_rate_provenance_json(secondary_rate, config.primary_atomic_number, config.primary_mass_number) << ",\n"
           << "    \"primary_package\": " << schneider_package_provenance_json(primary_pkg) << ",\n"
           << "    \"secondary_package\": " << schneider_package_provenance_json(secondary_pkg)
           << "\n  },\n";
}
void write_validation_scope(std::ofstream& output, const TransportConfig& config) {
    const std::string geometry = config.unified_water_nuclear_transport ? "native_G4_WATER" : config.ct_grid_file.empty()
                                     ? "none"
                                     : config.ct_grid_file.filename().string();
    std::string registry = "unknown";
    const std::filesystem::path sec_rate =
        config.ct_schneider_secondary_rate_file;
    if (std::filesystem::exists(sec_rate)) {
        std::ifstream in(sec_rate, std::ios::binary);
        char magic[8]{};
        std::uint32_t version{0};
        in.read(magic, 8);
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (in && std::memcmp(magic, "SCHN2RAT", 8) == 0) {
            registry = (version == 3) ? "v2.1-14p" : "unknown";
        }
    }
    output << "  \"secondary_out_of_scope_nuclear_policy\": \""
           << config.secondary_out_of_scope_nuclear_policy << "\",\n";
    output << "  \"validation_scope\": {\n"
           << "    \"geometry\": \"" << geometry << "\",\n"
           << "    \"primary\": \"Z=" << config.primary_atomic_number
           << " A=" << config.primary_mass_number << "\",\n"
           << "    \"projectile_registry\": \"" << registry << "\",\n"
           << "    \"out_of_scope_projectiles\": [\"He6\", \"B8\", \"C10\"],\n"
           << "    \"secondary_out_of_scope_nuclear_policy\": \""
           << config.secondary_out_of_scope_nuclear_policy << "\",\n"
           << "    \"purpose\": \"" << (config.unified_water_nuclear_transport ? "UNIFIED_WATER_FRAMEWORK_MATCH_PENDING" : "GPU_TOPAS_CT_DOSE_MATCH_RESEARCH") << "\",\n"
           << "    \"production_generalization\": false\n"
           << "  },\n";
}
void write_out_of_scope_isotope_summary(std::ofstream& output, const TransportResult& result) {
    struct Iso {
        const char* name;
        int z, a;
        std::size_t species;
    };
    constexpr Iso kIsos[] = {{"He6", 2, 6, 5}, {"B8", 5, 8, 11}, {"C10", 6, 10, 14}};
    const double beam = result.initial_energy_MeV;
    output << "  \"out_of_scope_isotope_summary\": [\n";
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& iso = kIsos[i];
        std::uint64_t tracks = 0;
        double birth_mev = 0.0;
        for (const auto& track : result.schneider_unsupported_tracks) {
            if (track.projectile_z == iso.z && track.projectile_a == iso.a) {
                ++tracks;
                birth_mev += track.birth_energy_MeV;
            }
        }
        const auto at = [&](std::size_t metric) {
            return result.ion_species_transport_ledger_MeV[
                iso.species * IonSpeciesLedgerSchema::metric_count + metric];
        };
        const bool recorded = (at(0) != 0.0 || at(1) != 0.0 || at(3) != 0.0 ||
                               at(5) != 0.0 || at(7) != 0.0 || at(9) != 0.0);
        output << "    {\"isotope\": \"" << iso.name << "\", \"tracks\": " << tracks
               << ", \"birth_energy_MeV\": " << birth_mev
               << ", \"birth_fraction_of_beam\": "
               << (beam > 0.0 ? birth_mev / beam : 0.0)
               << ", \"species_ledger_recorded\": " << (recorded ? "true" : "false");
        if (recorded) {
            output << ", \"ledger_birth_energy_MeV\": " << at(0)
                   << ", \"deposited_energy_MeV\": " << (at(1) + at(3) + at(5))
                   << ", \"escaped_energy_MeV\": " << (at(7) + at(9));
        } else {
            output << ", \"ledger_birth_energy_MeV\": null"
                   << ", \"deposited_energy_MeV\": null"
                   << ", \"escaped_energy_MeV\": null";
        }
        output << "}" << (i + 1 < 3 ? ",\n" : "\n");
    }
    output << "  ],\n";
}
const char* schneider_miss_status_name(const std::uint8_t status) noexcept {
    switch (status) {
    case 0: return "BelowEnergyDomain";
    case 1: return "AboveEnergyDomain";
    case 2: return "MissingProjectile";
    case 3: return "MissingTarget";
    case 4: return "EnergyGapTooLarge";
    case 5: return "EmptyNode";
    default: return "Unknown";
    }
}
}
void write_dense_voxel_dose_mhd(const std::filesystem::path& mhd_path,
                                const TransportConfig& config,
                                const TransportResult& result) {
    if (!config.enable_voxel_scoring) {
        throw std::invalid_argument(
            "Dense MHD dose output requested while voxel scoring is disabled");
    }
    if (result.voxel_deposited_energy_MeV.size() != config.number_of_voxels()) {
        throw std::invalid_argument("Transport result voxel count does not match configuration");
    }
    if (mhd_path.empty()) {
        throw std::invalid_argument("Dense MHD output path is empty");
    }

    ensure_parent_directory(mhd_path);
    const auto masses_kg = voxel_masses_kg(config);
    const auto nx = config.voxel_bins_x;
    const auto ny = config.voxel_bins_y;
    const auto nz = config.number_of_bins();
    const auto plane_size = nx * ny;
    const auto count = config.number_of_voxels();

    // MHD RAW layout matches sparse_dose_to_mhd: z major, then y, x fastest.
    // linear_zyx = iz * nx * ny + iy * nx + ix
    std::vector<float> raw(count, 0.0F);
    for (std::size_t iz = 0; iz < nz; ++iz) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
            for (std::size_t ix = 0; ix < nx; ++ix) {
                const auto index_xyz = iz * plane_size + iy * nx + ix;
                const auto energy = result.voxel_deposited_energy_MeV[index_xyz];
                if (energy == 0.0) {
                    continue;
                }
                const auto linear_zyx = iz * plane_size + iy * nx + ix;
                // Same index layout as GPU scorer (z-major plane of xy).
                raw[linear_zyx] =
                    static_cast<float>(scored_dose_Gy(config, energy, masses_kg[index_xyz]));
            }
        }
    }

    std::filesystem::path mhd = mhd_path;
    if (mhd.extension() != ".mhd") {
        mhd += ".mhd";
    }
    const auto raw_path = mhd.parent_path() / (mhd.stem().string() + ".raw");
    {
        std::ofstream raw_out(raw_path, std::ios::binary);
        if (!raw_out) {
            throw std::runtime_error("Cannot create RAW file: " + raw_path.string());
        }
        // MET_FLOAT little-endian (BinaryDataByteOrderMSB = False). Host is
        // little-endian on supported platforms (x86_64 / Arc GPU workstations).
        raw_out.write(reinterpret_cast<const char*>(raw.data()),
                      static_cast<std::streamsize>(raw.size() * sizeof(float)));
    }

    const auto x_extent_mm = static_cast<double>(nx) * config.voxel_size_x_mm;
    const auto y_extent_mm = static_cast<double>(ny) * config.voxel_size_y_mm;
    // Physical coordinate of the first voxel center (same as sparse CSV).
    // Default: 0-centered scorer. When a matching CT grid is enabled, use the
    // CT low-edge origin + half-voxel so MHD axes match transport sampling.
    double origin_x = 0.5 * config.voxel_size_x_mm - 0.5 * x_extent_mm;
    double origin_y = 0.5 * config.voxel_size_y_mm - 0.5 * y_extent_mm;
    double origin_z = 0.5 * config.scorer_spacing_z_mm();
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid_owner = CtGrid::load(config.ct_grid_file);
        const auto& grid = *grid_owner;
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz &&
            std::abs(static_cast<double>(grid.spacing_x_mm) - config.voxel_size_x_mm) <
                1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_y_mm) - config.voxel_size_y_mm) <
                1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_z_mm) -
                     config.scorer_spacing_z_mm()) <
                1.0e-6) {
            origin_x = static_cast<double>(grid.origin_x_mm) + 0.5 * config.voxel_size_x_mm;
            origin_y = static_cast<double>(grid.origin_y_mm) + 0.5 * config.voxel_size_y_mm;
            origin_z = static_cast<double>(grid.origin_z_mm) +
                       0.5 * config.scorer_spacing_z_mm();
        }
    }

    std::ofstream header(mhd, std::ios::binary);
    if (!header) {
        throw std::runtime_error("Cannot create MHD file: " + mhd.string());
    }
    header << std::setprecision(12);
    header << "ObjectType = Image\n"
           << "NDims = 3\n"
           << "BinaryData = True\n"
           << "BinaryDataByteOrderMSB = False\n"
           << "CompressedData = False\n"
           << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
           << "Offset = " << origin_x << ' ' << origin_y << ' ' << origin_z << '\n'
           << "CenterOfRotation = 0 0 0\n"
           << "ElementSpacing = " << config.voxel_size_x_mm << ' '
           << config.voxel_size_y_mm << ' ' << config.scorer_spacing_z_mm() << '\n'
           << "DimSize = " << nx << ' ' << ny << ' ' << nz << '\n'
           << "ElementType = MET_FLOAT\n"
           << "DoseUnits = Gy\n"
           << "ElementDataFile = " << raw_path.filename().string() << '\n';
}
void write_energy_ledger_json(const std::filesystem::path& path,
                              const TransportConfig& config,
                              const TransportResult& result) {
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create energy ledger: " + path.string());
    }
    const auto sum_depth = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0);
    };
    output << std::setprecision(12);
    output << "{\n"
           << "  \"histories\": " << config.number_of_histories << ",\n"
           << "  \"E_in_MeV\": " << result.initial_energy_MeV << ",\n"
           << "  \"E_dep_MeV\": " << result.total_deposited_energy_MeV << ",\n"
           << "  \"E_dep_in_grid_MeV\": " << result.in_grid_deposited_energy_MeV << ",\n"
           << "  \"E_dep_outside_grid_MeV\": "
           << result.outside_grid_deposited_energy_MeV << ",\n"
           << "  \"E_dep_depth_MeV\": " << sum_depth(result.deposited_energy_MeV)
           << ",\n"
           << "  \"E_primary_depth_MeV\": "
           << sum_depth(result.primary_deposited_energy_MeV) << ",\n"
           << "  \"E_secondary_transport_MeV\": "
           << result.secondary_deposited_energy_MeV << ",\n"
           << "  \"E_esc_MeV\": " << result.escaped_energy_MeV << ",\n"
           << "  \"E_secondary_esc_MeV\": " << result.secondary_escaped_energy_MeV
           << ",\n"
           << "  \"E_beamline_MeV\": " << result.beamline_removed_energy_MeV
           << ",\n"
           << "  \"E_untracked_MeV\": " << result.untracked_nuclear_energy_MeV << ",\n"
           << "  \"E_queue_lost_MeV\": "
           << result.secondary_queue_overflow_energy_MeV
           << ",\n"
           << "  \"E_queue_lost_MeV_included_in_untracked\": "
           << result.secondary_queue_overflow_energy_MeV
           << ",\n"
           << "  \"E_secondary_overflow_MeV\": "
           << result.secondary_queue_overflow_energy_MeV << ",\n"
           << "  \"nuclear_interactions\": " << result.nuclear_interactions << ",\n"
           << "  \"material_physics_mode\": \"" << material_physics_mode_name(config.material_physics_mode) << "\",\n"
           << "  \"ct_secondary_exact_faces\": " << (config.ct_secondary_exact_faces ? "true" : "false") << ",\n"
           << "  \"ct_schneider_primary_rate_file\": \"" << config.ct_schneider_primary_rate_file.string() << "\",\n"
           << "  \"ct_schneider_primary_cinel03_file\": \"" << config.ct_schneider_primary_cinel03_file.string() << "\",\n"
           << "  \"ct_schneider_secondary_rate_file\": \"" << config.ct_schneider_secondary_rate_file.string() << "\",\n"
           << "  \"ct_schneider_secondary_cinel03_file\": \"" << config.ct_schneider_secondary_cinel03_file.string() << "\",\n"
           << "  \"ct_schneider_physics_bundle_file\": \"" << config.ct_schneider_physics_bundle_file.string() << "\",\n";
    output << "  \"unified_water_nuclear_transport\": " << (config.unified_water_nuclear_transport ? "true" : "false") << ",\n";
    output << "  \"all_ion_primary_elastic_interactions\": " << result.primary_elastic_interactions << ",\n"
           << "  \"all_ion_secondary_elastic_interactions\": " << result.secondary_elastic_interactions << ",\n"
           << "  \"elastic_post_em_null_collisions\": " << result.elastic_post_em_null_collisions << ",\n";
    output << "  \"all_ion_elastic_file\": \"" << config.all_ion_elastic_file.string() << "\",\n"
           << "  \"elastic_recoil_stopping_file\": \"" << config.elastic_recoil_stopping_file.string() << "\",\n";
    if (config.unified_water_nuclear_transport) {
        output << "  \"nuclear_recoil_reference_grid_file\": \"" << config.primary_stopping_power_file.string() << "\",\n"
               << "  \"unified_water_material_id_note\": \"255 is a diagnostic sentinel, not a Schneider section; rate uses a separate water row\",\n";
    }
    output << "  \"transport_profile\": \"" << config.transport_profile << "\",\n"
           << "  \"fermi_eyges_gaussian_only\": " << (config.fermi_eyges_gaussian_only ? "true" : "false") << ",\n";

    write_schneider_physics_provenance(output, config);
    write_validation_scope(output, config);
    output << "  \"schneider_diagnostics\": {\n"
           << "    \"primary_rate_queries\": " << result.schneider_diagnostics.primary_rate_queries << ",\n"
           << "    \"primary_hazards\": " << result.schneider_diagnostics.primary_hazards << ",\n"
           << "    \"primary_exact_target_hits\": " << result.schneider_diagnostics.primary_exact_target_hits << ",\n"
           << "    \"primary_missing_projectile\": " << result.schneider_diagnostics.primary_missing_projectile << ",\n"
           << "    \"primary_missing_target\": " << result.schneider_diagnostics.primary_missing_target << ",\n"
           << "    \"primary_below_domain\": " << result.schneider_diagnostics.primary_below_domain << ",\n"
           << "    \"primary_above_domain\": " << result.schneider_diagnostics.primary_above_domain << ",\n"
           << "    \"primary_energy_gap_misses\": " << result.schneider_diagnostics.primary_energy_gap_misses << ",\n"
           << "    \"primary_empty_nodes\": " << result.schneider_diagnostics.primary_empty_nodes << ",\n"
           << "    \"primary_events_replayed\": " << result.schneider_diagnostics.primary_events_replayed << ",\n"
           << "    \"primary_charged_products_born\": " << result.schneider_diagnostics.primary_charged_products_born << ",\n"
           << "    \"primary_charged_products_queued\": " << result.schneider_diagnostics.primary_charged_products_queued << ",\n"
           << "    \"primary_charged_cutoff_kills\": " << result.schneider_diagnostics.primary_charged_cutoff_kills << ",\n"
           << "    \"primary_be6_kills\": " << result.schneider_diagnostics.primary_be6_kills << ",\n"
           << "    \"primary_queue_overflows\": " << result.schneider_diagnostics.primary_queue_overflows << ",\n"
           << "    \"secondary_tracks_started\": " << result.schneider_diagnostics.secondary_tracks_started << ",\n"
           << "    \"secondary_steps\": " << result.schneider_diagnostics.secondary_steps << ",\n"
           << "    \"secondary_fe_c12_steps\": " << result.schneider_diagnostics.secondary_fe_c12_steps << ",\n"
           << "    \"secondary_fe_he4_steps\": " << result.schneider_diagnostics.secondary_fe_he4_steps << ",\n"
           << "    \"secondary_fe_pdt_steps\": " << result.schneider_diagnostics.secondary_fe_pdt_steps << ",\n"
           << "    \"secondary_fe_other_charged_steps\": " << result.schneider_diagnostics.secondary_fe_other_charged_steps << ",\n"
           << "    \"secondary_highland_steps\": " << result.schneider_diagnostics.secondary_highland_steps << ",\n"
           << "    \"secondary_rate_queries\": " << result.schneider_diagnostics.secondary_rate_queries << ",\n"
           << "    \"secondary_hazards\": " << result.schneider_diagnostics.secondary_hazards << ",\n"
           << "    \"secondary_exact_target_hits\": " << result.schneider_diagnostics.secondary_exact_target_hits << ",\n"
           << "    \"secondary_missing_projectile\": " << result.schneider_diagnostics.secondary_missing_projectile << ",\n"
           << "    \"secondary_missing_target\": " << result.schneider_diagnostics.secondary_missing_target << ",\n"
           << "    \"secondary_below_domain\": " << result.schneider_diagnostics.secondary_below_domain << ",\n"
           << "    \"secondary_above_domain\": " << result.schneider_diagnostics.secondary_above_domain << ",\n"
           << "    \"secondary_energy_gap_misses\": " << result.schneider_diagnostics.secondary_energy_gap_misses << ",\n"
           << "    \"secondary_empty_nodes\": " << result.schneider_diagnostics.secondary_empty_nodes << ",\n"
           << "    \"secondary_events_replayed\": " << result.schneider_diagnostics.secondary_events_replayed << ",\n"
           << "    \"secondary_charged_products_born\": " << result.schneider_diagnostics.secondary_charged_products_born << ",\n"
           << "    \"secondary_charged_products_queued\": " << result.schneider_diagnostics.secondary_charged_products_queued << ",\n"
           << "    \"secondary_charged_cutoff_kills\": " << result.schneider_diagnostics.secondary_charged_cutoff_kills << ",\n"
           << "    \"secondary_be6_kills\": " << result.schneider_diagnostics.secondary_be6_kills << ",\n"
           << "    \"secondary_queue_overflows\": " << result.schneider_diagnostics.secondary_queue_overflows << ",\n"
           << "    \"secondary_stopped_before_replay\": " << result.schneider_diagnostics.secondary_stopped_before_replay << ",\n"
           << "    \"secondary_post_em_null_collisions\": " << result.schneider_diagnostics.secondary_post_em_null_collisions << ",\n"
           << "    \"secondary_post_em_null_energy_MeV\": " << result.schneider_diagnostics.secondary_post_em_null_energy_MeV << ",\n"
           << "    \"primary_post_em_null_collisions\": " << result.schneider_diagnostics.primary_post_em_null_collisions << ",\n"
           << "    \"be6_topas_compat_kills\": " << result.schneider_diagnostics.be6_topas_compat_kills << ",\n"
           << "    \"unsupported_projectile_steps\": " << result.schneider_diagnostics.unsupported_projectile_steps << ",\n"
           << "    \"unsupported_projectile_tracks\": " << result.schneider_diagnostics.unsupported_projectile_tracks << ",\n"
           << "    \"unsupported_be6_tracks\": " << result.schneider_diagnostics.unsupported_be6_tracks << ",\n"
           << "    \"unsupported_projectile_birth_energy_MeV\": " << result.schneider_diagnostics.unsupported_projectile_birth_energy_MeV << ",\n"
           << "    \"miss_log_dropped\": " << result.schneider_diagnostics.miss_log_dropped << ",\n"
           << "    \"unsupported_log_dropped\": " << result.schneider_diagnostics.unsupported_log_dropped << ",\n"
           << "    \"unsupported_targets\": " << result.schneider_diagnostics.unsupported_targets << ",\n"
           << "    \"queue_overflows\": " << result.schneider_diagnostics.queue_overflows << ",\n"
           << "    \"primary_hit_rate_events_over_hazards\": " << result.schneider_diagnostics.primary_hit_rate() << ",\n"
           << "    \"secondary_hit_rate_events_over_hazards\": " << result.schneider_diagnostics.secondary_hit_rate() << ",\n"
           << "    \"primary_selected_energy_mismatch_sum\": " << result.schneider_diagnostics.primary_selected_energy_mismatch_sum << ",\n"
           << "    \"primary_selected_energy_mismatch_max\": " << result.schneider_diagnostics.primary_selected_energy_mismatch_max << ",\n"
           << "    \"secondary_selected_energy_mismatch_sum\": " << result.schneider_diagnostics.secondary_selected_energy_mismatch_sum << ",\n"
           << "    \"secondary_selected_energy_mismatch_max\": " << result.schneider_diagnostics.secondary_selected_energy_mismatch_max << ",\n"
           << "    \"lookup_failure_energy_MeV\": " << result.schneider_diagnostics.lookup_failure_energy_MeV << ",\n"
           << "    \"be6_kill_energy_MeV\": " << result.schneider_diagnostics.be6_kill_energy_MeV << ",\n"
           << "    \"neutral_product_kinetic_MeV\": " << result.schneider_diagnostics.neutral_product_kinetic_MeV << ",\n"
           << "    \"reaction_q_residual_MeV\": " << result.schneider_diagnostics.reaction_q_residual_MeV << ",\n"
           << "    \"E_transported_secondaries_birth_MeV\": " << result.energy_ledger.E_transported_secondaries << ",\n"
           << "    \"E_charged_birth_MeV\": " << result.energy_ledger.E_charged_birth << ",\n"
           << "    \"E_be6_kill_MeV\": " << result.energy_ledger.E_be6_kill << ",\n"
           << "    \"E_lookup_failure_MeV\": " << result.energy_ledger.E_lookup_failure << ",\n"
           << "    \"E_neutral_product_kinetic_MeV\": " << result.energy_ledger.E_neutral_product_kinetic << ",\n"
           << "    \"E_reaction_q_residual_MeV\": " << result.energy_ledger.E_reaction_q_residual << ",\n"
           << "    \"E_unsupported_charged_MeV\": " << result.energy_ledger.E_unsupported_charged << ",\n"
           << "    \"E_out_of_domain_MeV\": " << result.energy_ledger.E_out_of_domain << "\n"
           << "  },\n"
           << "  \"ion_species_transport_ledger_layout\": "
              "{\"shape\":[" << IonSpeciesLedgerSchema::species_count << ","
           << IonSpeciesLedgerSchema::metric_count
           << "],\"order\":[\"species\",\"metric\"],"
              "\"species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\","
              "\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\","
              "\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"],"
              "\"metric\":[\"queued_birth_kinetic\",\"continuous_deposit_all\","
              "\"continuous_deposit_fov\",\"nuclear_local_deposit_all\","
              "\"nuclear_local_deposit_fov\",\"terminal_deposit_all\","
              "\"terminal_deposit_fov\",\"boundary_escape_kinetic\",\"reaction_export_kinetic\",\"step_limit_escape_kinetic\",\"reaction_import_kinetic\","
              "\"cinel03_replay_input_fov\",\"cinel03_replay_step_dE_fov\"]},\n"
           << "  \"ion_species_transport_ledger_MeV\": [";
    for (std::size_t i = 0;
         i < result.ion_species_transport_ledger_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.ion_species_transport_ledger_MeV[i];
    }
    output << "],\n";
    write_out_of_scope_isotope_summary(output, result);
    output
           << "  \"ion_species_terminal_reason_layout\": "
              "{\"shape\":[18,6],\"order\":[\"species\",\"reason\"],"
              "\"reason\":[\"initial_below_cutoff\",\"reaction_killed\","
              "\"terminal_deposit\",\"boundary_escape\",\"step_limit\","
              "\"continuous_stop\"]},\n"
           << "  \"ion_species_terminal_reason_counts\": [";
    for (std::size_t i = 0;
         i < result.ion_species_terminal_reason_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.ion_species_terminal_reason_counts[i];
    }
    output << "],\n"
           << "  \"primary_elastic_interactions\": "
           << result.primary_elastic_interactions << ",\n"
           << "  \"elastic_local_deposited_MeV\": "
           << result.elastic_local_deposited_energy_MeV << ",\n"
           << "  \"elastic_queued_charged_MeV\": "
           << result.elastic_queued_charged_energy_MeV << ",\n"
           << "  \"elastic_queue_overflow\": "
           << result.elastic_queue_overflow << ",\n"
           << "  \"elastic_queue_overflow_MeV\": "
           << result.elastic_queue_overflow_energy_MeV << ",\n"
           << "  \"cascade_interactions\": " << result.cascade_interactions << ",\n"
           << "  \"secondary_queue_overflow\": " << result.secondary_queue_overflow
           << ",\n"
           << "  \"cascade_queue_overflow\": " << result.cascade_queue_overflow
           << ",\n"
           << "  \"physical_energy_balance_error\": "
           << result.physical_relative_energy_balance_error() << "\n"
           << "}\n";
}
void write_schneider_miss_log_json(const std::filesystem::path& path,
                                   const TransportResult& result) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create Schneider miss log: " + path.string());
    }
    // Bucket summary keyed by (is_primary, status, projectile, target).
    struct BucketKey {
        std::uint8_t is_primary, status;
        std::int16_t pz, pa, tz;
        bool operator<(const BucketKey& o) const {
            return std::tie(is_primary, status, pz, pa, tz) <
                   std::tie(o.is_primary, o.status, o.pz, o.pa, o.tz);
        }
    };
    struct Bucket {
        std::uint64_t count{0};
        double query_sum{0.0}, incident_sum{0.0}, rate_sum{0.0}, step_sum{0.0};
    };
    std::map<BucketKey, Bucket> buckets;
    for (const auto& r : result.schneider_miss_log) {
        auto& b = buckets[{r.is_primary, r.status, r.projectile_z,
                           r.projectile_a, r.target_z}];
        ++b.count;
        b.query_sum += r.query_energy_MeV_per_u;
        b.incident_sum += r.incident_energy_MeV;
        b.rate_sum += r.total_rate_per_mm;
        b.step_sum += r.hazard_step_mm;
    }
    output << std::setprecision(12)
           << "{\n  \"schema_version\": 1,\n  \"record_count\": "
           << result.schneider_miss_log.size()
           << ",\n  \"dropped\": " << result.schneider_diagnostics.miss_log_dropped
           << ",\n  \"buckets\": [";
    bool first_bucket = true;
    for (const auto& [key, bucket] : buckets) {
        output << (first_bucket ? "\n" : ",\n")
               << "    {\"is_primary\": " << static_cast<int>(key.is_primary)
               << ", \"status\": \"" << schneider_miss_status_name(key.status)
               << "\", \"projectile_z\": " << key.pz
               << ", \"projectile_a\": " << key.pa
               << ", \"target_z\": " << key.tz
               << ", \"count\": " << bucket.count
               << ", \"query_sum_MeV_per_u\": " << bucket.query_sum
               << ", \"incident_sum_MeV\": " << bucket.incident_sum
               << ", \"total_rate_sum_per_mm\": " << bucket.rate_sum
               << ", \"hazard_step_sum_mm\": " << bucket.step_sum << "}";
        first_bucket = false;
    }
    output << (buckets.empty() ? "]" : "\n  ]");
    output << ",\n  \"records\": [";
    for (std::size_t i = 0; i < result.schneider_miss_log.size(); ++i) {
        const auto& r = result.schneider_miss_log[i];
        output << (i == 0 ? "\n" : ",\n")
               << "    {\"projectile_z\": " << r.projectile_z
               << ", \"projectile_a\": " << r.projectile_a
               << ", \"target_z\": " << r.target_z
               << ", \"status\": \"" << schneider_miss_status_name(r.status)
               << "\", \"section_id\": " << static_cast<int>(r.section_id)
               << ", \"is_primary\": " << static_cast<int>(r.is_primary)
               << ", \"generation\": " << static_cast<int>(r.generation)
               << ", \"query_energy_MeV_per_u\": " << r.query_energy_MeV_per_u
               << ", \"step_dE_MeV\": " << r.step_dE_MeV
               << ", \"total_rate_per_mm\": " << r.total_rate_per_mm
               << ", \"hazard_step_mm\": " << r.hazard_step_mm
               << ", \"incident_energy_MeV\": " << r.incident_energy_MeV
               << ", \"birth_energy_MeV\": " << r.birth_energy_MeV << "}";
    }
    output << (result.schneider_miss_log.empty() ? "]" : "\n  ]") << "\n}\n";
    if (!output) {
        throw std::runtime_error("Failed while writing Schneider miss log: " +
                                 path.string());
    }
}
void write_schneider_unsupported_tracks_json(const std::filesystem::path& path,
                                             const TransportResult& result) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create Schneider unsupported-track log: " + path.string());
    }
    struct IsoKey {
        std::int16_t z, a;
        bool operator<(const IsoKey& o) const {
            return std::tie(z, a) < std::tie(o.z, o.a);
        }
    };
    struct IsoAgg {
        std::uint64_t tracks{0};
        double birth_sum{0.0}, birth_max{0.0};
        std::uint64_t gen0{0}, gen1{0}, gen2plus{0};
    };
    std::map<IsoKey, IsoAgg> agg;
    for (const auto& t : result.schneider_unsupported_tracks) {
        auto& g = agg[{t.projectile_z, t.projectile_a}];
        ++g.tracks;
        g.birth_sum += t.birth_energy_MeV;
        g.birth_max = std::max(g.birth_max,
                               static_cast<double>(t.birth_energy_MeV));
        if (t.generation == 0) {
            ++g.gen0;
        } else if (t.generation == 1) {
            ++g.gen1;
        } else {
            ++g.gen2plus;
        }
    }
    output << std::setprecision(12)
           << "{\n  \"schema_version\": 1,\n  \"track_count\": "
           << result.schneider_unsupported_tracks.size()
           << ",\n  \"dropped\": "
           << result.schneider_diagnostics.unsupported_log_dropped
           << ",\n  \"by_isotope\": [";
    bool first_iso = true;
    for (const auto& [key, g] : agg) {
        output << (first_iso ? "\n" : ",\n")
               << "    {\"z\": " << key.z << ", \"a\": " << key.a
               << ", \"tracks\": " << g.tracks
               << ", \"birth_energy_sum_MeV\": " << g.birth_sum
               << ", \"birth_energy_max_MeV\": " << g.birth_max
               << ", \"gen0\": " << g.gen0 << ", \"gen1\": " << g.gen1
               << ", \"gen2plus\": " << g.gen2plus << "}";
        first_iso = false;
    }
    output << (agg.empty() ? "]" : "\n  ]");
    output << ",\n  \"records\": [";
    for (std::size_t i = 0; i < result.schneider_unsupported_tracks.size(); ++i) {
        const auto& t = result.schneider_unsupported_tracks[i];
        output << (i == 0 ? "\n" : ",\n")
               << "    {\"z\": " << t.projectile_z << ", \"a\": " << t.projectile_a
               << ", \"generation\": " << t.generation
               << ", \"birth_energy_MeV\": " << t.birth_energy_MeV
               << ", \"birth_x_mm\": " << t.birth_x_mm
               << ", \"birth_y_mm\": " << t.birth_y_mm
               << ", \"birth_z_mm\": " << t.birth_z_mm << "}";
    }
    output << (result.schneider_unsupported_tracks.empty() ? "]" : "\n  ]")
           << "\n}\n";
    if (!output) {
        throw std::runtime_error(
            "Failed while writing Schneider unsupported-track log: " +
            path.string());
    }
}
}
