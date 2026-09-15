#include "carbon/io.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/particle.hpp"
#include "carbon/sha256.hpp"
#include "carbon/charged_species.hpp"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace carbon {
namespace {

// CODATA MeV → J. Dose = E_J / mass_kg.
constexpr double kMeV_to_joule = 1.602176634e-13;

// mass [kg] = volume [mm^3] × density [g/cm^3] × 1e-6
// (1 mm^3 × 1 g/cm^3 = 1e-3 g = 1e-6 kg).
double idd_bin_mass_kg(const TransportConfig& config) {
    return config.scorer_area_mm2 * config.depth_bin_width_mm *
           config.water_density_g_per_cm3 * 1.0e-6;
}

double voxel_mass_kg(const TransportConfig& config) {
    return config.voxel_size_x_mm * config.voxel_size_y_mm *
           config.scorer_spacing_z_mm() *
           config.water_density_g_per_cm3 * 1.0e-6;
}

std::vector<double> voxel_masses_kg(const TransportConfig& config) {
    const auto count = config.number_of_voxels();
    std::vector<double> masses(count, voxel_mass_kg(config));
    const auto nx = config.voxel_bins_x;
    const auto ny = config.voxel_bins_y;
    const auto nz = config.number_of_bins();
    const auto plane_size = nx * ny;
    const auto dx = config.voxel_size_x_mm;
    const auto dy = config.voxel_size_y_mm;
    const auto dz = config.scorer_spacing_z_mm();
    const auto volume_mm3 = dx * dy * dz;

    if (config.enable_layered_phantom) {
        auto layer_start_mm = 0.0;
        for (std::size_t iz = 0; iz < nz; ++iz) {
            const auto voxel_start_mm = static_cast<double>(iz) * dz;
            const auto voxel_end_mm = voxel_start_mm + dz;
            auto density_integral = 0.0;
            layer_start_mm = 0.0;
            for (const auto& layer : config.slab_layers) {
                const auto overlap_mm = std::max(
                    0.0,
                    std::min(voxel_end_mm, layer.z_end_mm) -
                        std::max(voxel_start_mm, layer_start_mm));
                density_integral += overlap_mm * layer.density_g_per_cm3;
                layer_start_mm = layer.z_end_mm;
            }
            const auto average_density = density_integral / dz;
            std::fill_n(
                masses.begin() + static_cast<std::ptrdiff_t>(iz * plane_size),
                plane_size,
                volume_mm3 * average_density * 1.0e-6);
        }
        return masses;
    }
    if (config.enable_hetero_insert) {
        const auto scorer_min_x = -0.5 * static_cast<double>(nx) * dx;
        const auto scorer_min_y = -0.5 * static_cast<double>(ny) * dy;
        const auto overlap_length = [](const double first_min,
                                       const double first_max,
                                       const double second_min,
                                       const double second_max) {
            return std::max(
                0.0, std::min(first_max, second_max) -
                         std::max(first_min, second_min));
        };
        for (std::size_t iz = 0; iz < nz; ++iz) {
            const auto z0 = static_cast<double>(iz) * dz;
            const auto z_overlap = overlap_length(
                z0, z0 + dz, config.hetero_insert.z_min_mm,
                config.hetero_insert.z_max_mm);
            for (std::size_t iy = 0; iy < ny; ++iy) {
                const auto y0 = scorer_min_y + static_cast<double>(iy) * dy;
                const auto y_overlap = overlap_length(
                    y0, y0 + dy, config.hetero_insert.y_min_mm,
                    config.hetero_insert.y_max_mm);
                for (std::size_t ix = 0; ix < nx; ++ix) {
                    const auto x0 = scorer_min_x + static_cast<double>(ix) * dx;
                    const auto x_overlap = overlap_length(
                        x0, x0 + dx, config.hetero_insert.x_min_mm,
                        config.hetero_insert.x_max_mm);
                    const auto insert_volume_mm3 =
                        x_overlap * y_overlap * z_overlap;
                    const auto water_volume_mm3 =
                        volume_mm3 - insert_volume_mm3;
                    const auto index = iz * plane_size + iy * nx + ix;
                    masses[index] =
                        (water_volume_mm3 * config.water_density_g_per_cm3 +
                         insert_volume_mm3 *
                             config.hetero_insert.density_g_per_cm3) *
                        1.0e-6;
                }
            }
        }
        return masses;
    }
    if (!config.enable_ct_grid) {
        return masses;
    }
    const auto grid = CtGrid::from_config(config);
    if (grid.nx != config.voxel_bins_x || grid.ny != config.voxel_bins_y ||
        grid.nz != config.number_of_bins() ||
        std::abs(static_cast<double>(grid.spacing_x_mm) - config.voxel_size_x_mm) > 1.0e-6 ||
        std::abs(static_cast<double>(grid.spacing_y_mm) - config.voxel_size_y_mm) > 1.0e-6 ||
        std::abs(static_cast<double>(grid.spacing_z_mm) - config.depth_bin_width_mm) > 1.0e-6) {
        throw std::invalid_argument(
            "CT dose-to-medium requires the voxel scorer grid to match the CT grid");
    }
    for (std::size_t i = 0; i < count; ++i) {
        // Keep very-low-density CT voxels finite while retaining their actual
        // dose-to-medium mass rather than assuming unit-density water.
        const auto density = std::max(1.0e-6, static_cast<double>(grid.density_g_per_cm3[i]));
        masses[i] = volume_mm3 * density * 1.0e-6;
    }
    return masses;
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

}  // namespace

void write_depth_dose_csv(const std::filesystem::path& path,
                          const TransportConfig& config,
                          const TransportResult& result) {
    if (result.deposited_energy_MeV.size() != config.number_of_bins()) {
        throw std::invalid_argument("Transport result bin count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create output file: " + path.string());
    }

    const auto maximum = *std::max_element(result.deposited_energy_MeV.begin(),
                                           result.deposited_energy_MeV.end());
    const auto bin_mass_kg = idd_bin_mass_kg(config);

    // Total tallies over all sampled primaries (not per-primary).
    output << "depth_mm,energy_deposition_MeV,dose_Gy,relative_dose\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < result.deposited_energy_MeV.size(); ++bin) {
        const auto energy_total = result.deposited_energy_MeV[bin];
        const auto dose_total = scored_dose_Gy(config, energy_total, bin_mass_kg);
        const auto relative_dose = maximum > 0.0 ? energy_total / maximum : 0.0;
        const auto depth_center_mm = (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm << ',' << energy_total << ',' << dose_total << ','
               << relative_dose << '\n';
    }
}

void write_depth_dose_Gy_csv(const std::filesystem::path& path,
                             const TransportConfig& config,
                             const TransportResult& result) {
    if (result.deposited_energy_MeV.size() != config.number_of_bins()) {
        throw std::invalid_argument("Transport result bin count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create dose output file: " + path.string());
    }

    const auto bin_mass_kg = idd_bin_mass_kg(config);
    double maximum_dose = 0.0;
    for (const auto energy_MeV : result.deposited_energy_MeV) {
        maximum_dose = std::max(maximum_dose, scored_dose_Gy(config, energy_MeV, bin_mass_kg));
    }

    // Total dose over all sampled primaries (Gy, not Gy/primary).
    output << "depth_mm,dose_Gy,relative_dose\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < result.deposited_energy_MeV.size(); ++bin) {
        const auto dose_total =
            scored_dose_Gy(config, result.deposited_energy_MeV[bin], bin_mass_kg);
        const auto relative_dose =
            maximum_dose > 0.0 ? dose_total / maximum_dose : 0.0;
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm << ',' << dose_total << ',' << relative_dose << '\n';
    }
}

void write_fragment_species_csv(const std::filesystem::path& path,
                                const TransportConfig& config,
                                const TransportResult& result) {
    const auto bins = config.number_of_bins();
    const std::vector<const std::vector<double>*> columns{
        &result.deposited_energy_MeV,
        &result.primary_deposited_energy_MeV,
        &result.secondary_carbon_deposited_energy_MeV,
        &result.secondary_boron_deposited_energy_MeV,
        &result.secondary_beryllium_deposited_energy_MeV,
        &result.secondary_lithium_deposited_energy_MeV,
        &result.secondary_helium_deposited_energy_MeV,
        &result.secondary_proton_deposited_energy_MeV,
        &result.secondary_other_charged_deposited_energy_MeV,
    };
    if (std::any_of(columns.begin(), columns.end(),
                    [bins](const auto* column) { return column->size() != bins; })) {
        throw std::invalid_argument("Fragment species result bin count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create fragment species output file: " + path.string());
    }
    const auto maximum = *std::max_element(result.deposited_energy_MeV.begin(),
                                           result.deposited_energy_MeV.end());
    output << "depth_mm,total_MeV,primary_MeV,"
              "secondary_carbon_MeV,secondary_boron_MeV,"
              "secondary_beryllium_MeV,secondary_lithium_MeV,"
              "secondary_helium_MeV,secondary_proton_MeV,"
              "secondary_other_MeV,relative_total\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm;
        for (const auto* column : columns) {
            output << ',' << (*column)[bin];
        }
        const auto relative_total =
            maximum > 0.0 ? result.deposited_energy_MeV[bin] / maximum : 0.0;
        output << ',' << relative_total << '\n';
    }
}

void write_letd_csv(const std::filesystem::path& path,
                    const TransportConfig& config,
                    const TransportResult& result) {
    const auto bins = config.number_of_bins();
    const std::array<const std::vector<double>*, 4> moments{
        &result.primary_letd_numerator,
        &result.primary_letd_denominator,
        &result.all_hadron_letd_numerator,
        &result.all_hadron_letd_denominator,
    };
    if (std::any_of(moments.begin(), moments.end(),
                    [bins](const auto* values) { return values->size() != bins; })) {
        throw std::invalid_argument("LET result bin count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create LET output file: " + path.string());
    }
    output << "depth_mm,primary_letd_MeV_per_mm_per_g_cm3,"
              "all_hadron_letd_MeV_per_mm_per_g_cm3,"
              "primary_numerator,primary_denominator_MeV,"
              "all_hadron_numerator,all_hadron_denominator_MeV\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const auto primary_denominator = result.primary_letd_denominator[bin];
        const auto all_denominator = result.all_hadron_letd_denominator[bin];
        const auto primary_letd =
            primary_denominator > 0.0
                ? result.primary_letd_numerator[bin] / primary_denominator
                : 0.0;
        const auto all_letd =
            all_denominator > 0.0
                ? result.all_hadron_letd_numerator[bin] / all_denominator
                : 0.0;
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm << ',' << primary_letd << ',' << all_letd << ','
               << result.primary_letd_numerator[bin] << ','
               << primary_denominator << ','
               << result.all_hadron_letd_numerator[bin] << ','
               << all_denominator << '\n';
    }
}

void write_dense_voxel_letd_mhd(const std::filesystem::path& mhd_path,
                                const TransportConfig& config,
                                const TransportResult& result) {
    if (!config.enable_let_scoring || !config.enable_voxel_scoring) {
        throw std::invalid_argument(
            "Dense voxel LET output requires LET and voxel scoring");
    }
    const auto count = config.number_of_voxels();
    const std::array<const std::vector<double>*, 4> moments{
        &result.primary_voxel_letd_numerator,
        &result.primary_voxel_letd_denominator,
        &result.all_hadron_voxel_letd_numerator,
        &result.all_hadron_voxel_letd_denominator,
    };
    if (std::any_of(moments.begin(), moments.end(),
                    [count](const auto* values) {
                        return values->size() != count;
                    })) {
        throw std::invalid_argument(
            "Voxel LET result size does not match configuration");
    }
    if (mhd_path.empty()) {
        throw std::invalid_argument("Dense voxel LET MHD output path is empty");
    }

    std::filesystem::path base = mhd_path;
    if (base.extension() == ".mhd") {
        base.replace_extension();
    }
    ensure_parent_directory(base);

    const auto nx = config.voxel_bins_x;
    const auto ny = config.voxel_bins_y;
    const auto nz = config.number_of_bins();
    const auto x_extent_mm = static_cast<double>(nx) * config.voxel_size_x_mm;
    const auto y_extent_mm = static_cast<double>(ny) * config.voxel_size_y_mm;
    double origin_x = 0.5 * config.voxel_size_x_mm - 0.5 * x_extent_mm;
    double origin_y = 0.5 * config.voxel_size_y_mm - 0.5 * y_extent_mm;
    double origin_z = 0.5 * config.scorer_spacing_z_mm();
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid = CtGrid::from_config(config);
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz) {
            origin_x =
                static_cast<double>(grid.origin_x_mm) +
                0.5 * config.voxel_size_x_mm;
            origin_y =
                static_cast<double>(grid.origin_y_mm) +
                0.5 * config.voxel_size_y_mm;
            origin_z =
                static_cast<double>(grid.origin_z_mm) +
                0.5 * config.scorer_spacing_z_mm();
        }
    }

    const auto write_map = [&](const std::string& suffix,
                               const std::vector<double>& numerator,
                               const std::vector<double>& denominator) {
        auto header_path =
            base.parent_path() / (base.filename().string() + suffix + ".mhd");
        const auto raw_path =
            header_path.parent_path() / (header_path.stem().string() + ".raw");
        std::vector<float> raw(count, 0.0F);
        for (std::size_t index = 0; index < count; ++index) {
            if (denominator[index] > 0.0) {
                raw[index] = static_cast<float>(
                    numerator[index] / denominator[index]);
            }
        }
        {
            std::ofstream raw_out(raw_path, std::ios::binary);
            if (!raw_out) {
                throw std::runtime_error(
                    "Cannot create voxel LET RAW file: " + raw_path.string());
            }
            raw_out.write(
                reinterpret_cast<const char*>(raw.data()),
                static_cast<std::streamsize>(raw.size() * sizeof(float)));
        }
        std::ofstream header(header_path, std::ios::binary);
        if (!header) {
            throw std::runtime_error(
                "Cannot create voxel LET MHD file: " + header_path.string());
        }
        header << std::setprecision(12)
               << "ObjectType = Image\n"
               << "NDims = 3\n"
               << "BinaryData = True\n"
               << "BinaryDataByteOrderMSB = False\n"
               << "CompressedData = False\n"
               << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
               << "Offset = " << origin_x << ' ' << origin_y << ' '
               << origin_z << '\n'
               << "CenterOfRotation = 0 0 0\n"
               << "ElementSpacing = " << config.voxel_size_x_mm << ' '
               << config.voxel_size_y_mm << ' '
               << config.scorer_spacing_z_mm() << '\n'
               << "DimSize = " << nx << ' ' << ny << ' ' << nz << '\n'
               << "ElementType = MET_FLOAT\n"
               << "LETUnits = MeV/mm/(g/cm3)\n"
               << "ElementDataFile = " << raw_path.filename().string() << '\n';
    };

    write_map("_primary",
              result.primary_voxel_letd_numerator,
              result.primary_voxel_letd_denominator);
    write_map("_all_hadron",
              result.all_hadron_voxel_letd_numerator,
              result.all_hadron_voxel_letd_denominator);
}

void write_fragment_species_letd_csv(const std::filesystem::path& path,
                                     const TransportConfig& config,
                                     const TransportResult& result) {
    constexpr std::size_t categories = charged_origin_category_count;
    const auto bins = config.number_of_bins();
    const auto expected = categories * bins;
    if (result.charged_origin_letd_numerator.size() != expected ||
        result.charged_origin_letd_denominator.size() != expected) {
        throw std::invalid_argument(
            "Fragment-species LET result size does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create fragment-species LET output file: " + path.string());
    }
    constexpr std::array<const char*, categories> names{
        "primary", "secondary_carbon", "secondary_boron",
        "secondary_beryllium", "secondary_lithium", "secondary_helium",
        "secondary_hydrogen", "secondary_other_charged"};
    output << "depth_mm";
    for (const auto* name : names) {
        output << ',' << name << "_letd_MeV_per_mm_per_g_cm3";
    }
    for (const auto* name : names) {
        output << ',' << name << "_denominator_MeV";
    }
    output << '\n' << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        output << (static_cast<double>(bin) + 0.5) *
                      config.depth_bin_width_mm;
        for (std::size_t category = 0; category < categories; ++category) {
            const auto index = category * bins + bin;
            const auto denominator =
                result.charged_origin_letd_denominator[index];
            const auto value =
                denominator > 0.0
                    ? result.charged_origin_letd_numerator[index] / denominator
                    : 0.0;
            output << ',' << value;
        }
        for (std::size_t category = 0; category < categories; ++category) {
            output << ','
                   << result.charged_origin_letd_denominator[
                          category * bins + bin];
        }
        output << '\n';
    }
}

void write_light_isotope_letd_csv(const std::filesystem::path& path,
                                  const TransportConfig& config,
                                  const TransportResult& result) {
    constexpr std::size_t categories = light_isotope_category_count;
    const auto bins = config.number_of_bins();
    const auto expected = categories * bins;
    if (result.light_isotope_letd_numerator.size() != expected ||
        result.light_isotope_letd_denominator.size() != expected) {
        throw std::invalid_argument(
            "Light-isotope LET result size does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create light-isotope LET output file: " + path.string());
    }
    constexpr std::array<const char*, categories> names{
        "proton", "deuteron", "triton", "he3", "he4",
        "nitrogen", "oxygen", "fluorine"};
    output << "depth_mm";
    for (const auto* name : names) {
        output << ',' << name << "_letd_MeV_per_mm_per_g_cm3";
    }
    for (const auto* name : names) {
        output << ',' << name << "_denominator_MeV";
    }
    output << '\n' << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        output << (static_cast<double>(bin) + 0.5) *
                      config.depth_bin_width_mm;
        for (std::size_t category = 0; category < categories; ++category) {
            const auto index = category * bins + bin;
            const auto denominator =
                result.light_isotope_letd_denominator[index];
            output << ','
                   << (denominator > 0.0
                           ? result.light_isotope_letd_numerator[index] /
                                 denominator
                           : 0.0);
        }
        for (std::size_t category = 0; category < categories; ++category) {
            output << ','
                   << result.light_isotope_letd_denominator[
                          category * bins + bin];
        }
        output << '\n';
    }
}

void write_fragment_birth_spectrum_csv(const std::filesystem::path& prefix,
                                       const TransportConfig& config,
                                       const TransportResult& result) {
    constexpr std::size_t categories = light_isotope_category_count;
    constexpr std::array<const char*, categories> names{
        "proton", "deuteron", "triton", "he3", "he4",
        "nitrogen", "oxygen", "fluorine"};
    const auto depth_bins = config.number_of_bins();
    const auto gen_size = categories * birth_generation_bin_count;
    const auto mevu_size = birth_hist_plane_size(birth_mevu_bin_count);
    const auto depth_size = birth_hist_plane_size(depth_bins);
    const auto cos_size = birth_hist_plane_size(birth_cos_bin_count);
    const auto parent_mevu_size =
        birth_hist_plane_size(birth_parent_mevu_bin_count);
    const auto parent_z_size = birth_hist_plane_size(birth_parent_z_bin_count);
    const auto joint_size = birth_joint_plane_size();

    if (result.birth_counts_by_generation.size() != gen_size ||
        result.birth_ke_sum_MeV_by_generation.size() != gen_size ||
        result.birth_mevu_hist.size() != mevu_size ||
        result.birth_depth_hist.size() != depth_size ||
        result.birth_cos_hist.size() != cos_size ||
        result.birth_parent_mevu_hist.size() != parent_mevu_size ||
        result.birth_parent_z_hist.size() != parent_z_size ||
        result.birth_parent_product_mevu_hist.size() != joint_size) {
        throw std::invalid_argument(
            "Fragment birth-spectrum result size does not match configuration");
    }

    const auto write_open = [&](const std::string& suffix) {
        const auto path = std::filesystem::path(prefix.string() + suffix);
        ensure_parent_directory(path);
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                "Cannot create birth-spectrum output file: " + path.string());
        }
        return output;
    };

    {
        auto output = write_open("_helium_joint.csv");
        output << "source_history,generation,Z,A,kinetic_energy_MeV,x_mm,y_mm,z_mm,direction_x,direction_y,direction_z,weight\n"
               << std::setprecision(17);
        for (const auto& birth : result.helium_birth_records) {
            output << birth.history << ',' << birth.generation << ",2," << birth.mass_number
                   << ',' << birth.kinetic_energy_MeV << ',' << birth.x_mm << ',' << birth.y_mm
                   << ',' << birth.z_mm << ',' << birth.direction_x << ',' << birth.direction_y
                   << ',' << birth.direction_z << ',' << birth.weight << '\n';
        }
        if (!output) throw std::runtime_error("Failed to write helium joint birth spectrum");
    }
    {
        auto output = write_open("_he4_hazard.csv");
        output << "tau_start,tau_simpson,energy_tau_simpson_MeV,candidates,candidate_energy_MeV,path_mm\n"
               << std::setprecision(17);
        for (std::size_t i=0; i<result.helium4_hazard_audit.size(); ++i)
            output << (i ? "," : "") << result.helium4_hazard_audit[i];
        output << '\n';
        if (!output) throw std::runtime_error("Failed to write He4 hazard audit");
    }
    {
        auto output = write_open("_summary.csv");
        output << "species,generation,count,mean_kinetic_energy_MeV,"
                  "yield_per_primary\n"
               << std::setprecision(12);
        const auto histories =
            static_cast<double>(std::max<std::uint64_t>(1, config.number_of_histories));
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                const auto index = cat * birth_generation_bin_count + gen;
                const auto count = result.birth_counts_by_generation[index];
                const auto ke_sum = result.birth_ke_sum_MeV_by_generation[index];
                output << names[cat] << ',' << gen << ',' << count << ','
                       << (count > 0 ? ke_sum / static_cast<double>(count) : 0.0)
                       << ',' << static_cast<double>(count) / histories << '\n';
            }
        }
    }
    {
        auto output = write_open("_mevu.csv");
        output << "species,generation,mevu_bin_low,mevu_bin_high,count\n"
               << std::setprecision(12);
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t bin = 0; bin < birth_mevu_bin_count; ++bin) {
                    const auto count = result.birth_mevu_hist[birth_hist_index(
                        cat, gen, bin, birth_mevu_bin_count)];
                    if (count == 0) {
                        continue;
                    }
                    const auto low =
                        static_cast<double>(bin) * birth_mevu_bin_width;
                    output << names[cat] << ',' << gen << ',' << low << ','
                           << (low + birth_mevu_bin_width) << ',' << count
                           << '\n';
                }
            }
        }
    }
    {
        auto output = write_open("_depth.csv");
        output << "species,generation,depth_mm,count\n" << std::setprecision(12);
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t bin = 0; bin < depth_bins; ++bin) {
                    const auto count = result.birth_depth_hist[birth_hist_index(
                        cat, gen, bin, depth_bins)];
                    if (count == 0) {
                        continue;
                    }
                    output << names[cat] << ',' << gen << ','
                           << (static_cast<double>(bin) + 0.5) *
                                  config.depth_bin_width_mm
                           << ',' << count << '\n';
                }
            }
        }
    }
    {
        auto output = write_open("_costheta.csv");
        output << "species,generation,cos_bin_low,cos_bin_high,count\n"
               << std::setprecision(12);
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t bin = 0; bin < birth_cos_bin_count; ++bin) {
                    const auto count = result.birth_cos_hist[birth_hist_index(
                        cat, gen, bin, birth_cos_bin_count)];
                    if (count == 0) {
                        continue;
                    }
                    const auto low =
                        -1.0 + 2.0 * static_cast<double>(bin) / birth_cos_bin_count;
                    const auto high = -1.0 + 2.0 * static_cast<double>(bin + 1) /
                                                 birth_cos_bin_count;
                    output << names[cat] << ',' << gen << ',' << low << ','
                           << high << ',' << count << '\n';
                }
            }
        }
    }
    {
        auto output = write_open("_parent_mevu.csv");
        output << "species,generation,parent_mevu_bin_low,parent_mevu_bin_high,"
                  "count\n"
               << std::setprecision(12);
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t bin = 0; bin < birth_parent_mevu_bin_count;
                     ++bin) {
                    const auto count =
                        result.birth_parent_mevu_hist[birth_hist_index(
                            cat, gen, bin, birth_parent_mevu_bin_count)];
                    if (count == 0) {
                        continue;
                    }
                    const auto low =
                        static_cast<double>(bin) * birth_parent_mevu_bin_width;
                    output << names[cat] << ',' << gen << ',' << low << ','
                           << (low + birth_parent_mevu_bin_width) << ',' << count
                           << '\n';
                }
            }
        }
    }
    {
        auto output = write_open("_parent_z.csv");
        output << "species,generation,parent_Z,count\n";
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t bin = 0; bin < birth_parent_z_bin_count;
                     ++bin) {
                    const auto count =
                        result.birth_parent_z_hist[birth_hist_index(
                            cat, gen, bin, birth_parent_z_bin_count)];
                    if (count == 0) {
                        continue;
                    }
                    output << names[cat] << ',' << gen << ',' << bin << ','
                           << count << '\n';
                }
            }
        }
    }
    {
        // Sparse joint parent×product MeV/u histogram for conditioned spectrum
        // comparisons (generation-aware).
        auto output = write_open("_parent_product_mevu.csv");
        output << "species,generation,parent_mevu_bin_low,parent_mevu_bin_high,"
                  "product_mevu_bin_low,product_mevu_bin_high,count\n"
               << std::setprecision(12);
        for (std::size_t cat = 0; cat < categories; ++cat) {
            for (std::size_t gen = 0; gen < birth_generation_bin_count; ++gen) {
                for (std::size_t pbin = 0; pbin < birth_parent_mevu_bin_count;
                     ++pbin) {
                    for (std::size_t ebin = 0; ebin < birth_mevu_bin_count;
                         ++ebin) {
                        const auto count =
                            result.birth_parent_product_mevu_hist[birth_joint_index(
                                cat, gen, pbin, ebin)];
                        if (count == 0) {
                            continue;
                        }
                        const auto plow = static_cast<double>(pbin) *
                                          birth_parent_mevu_bin_width;
                        const auto elow =
                            static_cast<double>(ebin) * birth_mevu_bin_width;
                        output << names[cat] << ',' << gen << ',' << plow << ','
                               << (plow + birth_parent_mevu_bin_width) << ','
                               << elow << ',' << (elow + birth_mevu_bin_width)
                               << ',' << count << '\n';
                    }
                }
            }
        }
    }
}

void write_fragment_species_dose_Gy_csv(const std::filesystem::path& path,
                                        const TransportConfig& config,
                                        const TransportResult& result) {
    const auto bins = config.number_of_bins();
    const std::vector<const std::vector<double>*> columns{
        &result.deposited_energy_MeV,
        &result.primary_deposited_energy_MeV,
        &result.secondary_carbon_deposited_energy_MeV,
        &result.secondary_boron_deposited_energy_MeV,
        &result.secondary_beryllium_deposited_energy_MeV,
        &result.secondary_lithium_deposited_energy_MeV,
        &result.secondary_helium_deposited_energy_MeV,
        &result.secondary_proton_deposited_energy_MeV,
        &result.secondary_other_charged_deposited_energy_MeV,
    };
    if (std::any_of(columns.begin(), columns.end(),
                    [bins](const auto* column) { return column->size() != bins; })) {
        throw std::invalid_argument(
            "Fragment species result bin count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create fragment species dose output file: " + path.string());
    }
    const auto bin_mass_kg = idd_bin_mass_kg(config);
    double maximum_dose = 0.0;
    for (const auto energy_MeV : result.deposited_energy_MeV) {
        maximum_dose = std::max(maximum_dose, scored_dose_Gy(config, energy_MeV, bin_mass_kg));
    }

    output << "depth_mm,total_Gy,primary_Gy,"
              "secondary_carbon_Gy,secondary_boron_Gy,"
              "secondary_beryllium_Gy,secondary_lithium_Gy,"
              "secondary_helium_Gy,secondary_proton_Gy,"
              "secondary_other_Gy,relative_total\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm;
        for (const auto* column : columns) {
            output << ',' << scored_dose_Gy(config, (*column)[bin], bin_mass_kg);
        }
        const auto total_dose =
            scored_dose_Gy(config, result.deposited_energy_MeV[bin], bin_mass_kg);
        const auto relative_total =
            maximum_dose > 0.0 ? total_dose / maximum_dose : 0.0;
        output << ',' << relative_total << '\n';
    }
}

void write_sparse_voxel_dose_csv(const std::filesystem::path& path,
                                 const TransportConfig& config,
                                 const TransportResult& result) {
    if (!config.enable_voxel_scoring) {
        throw std::invalid_argument("Voxel dose output requested while voxel scoring is disabled");
    }
    if (result.voxel_deposited_energy_MeV.size() != config.number_of_voxels()) {
        throw std::invalid_argument("Transport result voxel count does not match configuration");
    }
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto histories = static_cast<double>(config.number_of_histories);
    for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
        const auto begin = result.voxel_deposited_energy_MeV.begin() +
                           static_cast<std::ptrdiff_t>(z * plane_size);
        const auto reconstructed =
            std::accumulate(begin, begin + static_cast<std::ptrdiff_t>(plane_size), 0.0);
        // Compare plane-sum of voxels against in-FOV depth tally (same spatial domain).
        // Full plane depth dose contains lateral leakage outside the finite voxel grid.
        const auto depth = (!result.in_fov_deposited_energy_MeV.empty())
                               ? result.in_fov_deposited_energy_MeV[z]
                               : result.deposited_energy_MeV[z];
        const auto abs_diff = std::abs(reconstructed - depth);
        const auto scale = std::max({std::abs(reconstructed), std::abs(depth), 1.0});
#if defined(CARBON_DOSE_FP32)
        const auto tol = std::max(1.0e-5 * histories, 1.0e-3 * scale);
        if (abs_diff > tol) {
            std::cerr << "warning: voxel vs in-FOV depth-dose closure weak at z bin " << z
                      << " (diff=" << abs_diff << " MeV, tol=" << tol
                      << " MeV, rel=" << (abs_diff / scale) << ")\n";
        }
#else
        const auto tol = std::max(1.0e-9 * histories, 5.0e-5 * scale);
        if (abs_diff > tol) {
            throw std::runtime_error(
                "Voxel dose does not close to the in-FOV depth-dose tally at z bin " +
                std::to_string(z) + " (diff=" + std::to_string(abs_diff) +
                " MeV, tol=" + std::to_string(tol) + " MeV)");
        }
#endif
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create voxel dose output file: " + path.string());
    }
    const auto masses_kg = voxel_masses_kg(config);
    const auto x_extent_mm =
        static_cast<double>(config.voxel_bins_x) * config.voxel_size_x_mm;
    const auto y_extent_mm =
        static_cast<double>(config.voxel_bins_y) * config.voxel_size_y_mm;
    output << "ix,iy,iz,x_mm,y_mm,z_mm,energy_deposition_MeV,dose_Gy\n";
    output << std::setprecision(12);
    for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
        for (std::size_t y = 0; y < config.voxel_bins_y; ++y) {
            for (std::size_t x = 0; x < config.voxel_bins_x; ++x) {
                const auto index = z * plane_size + y * config.voxel_bins_x + x;
                const auto energy = result.voxel_deposited_energy_MeV[index];
                if (energy == 0.0) {
                    continue;
                }
                const auto dose_total = scored_dose_Gy(config, energy, masses_kg[index]);
                const auto x_mm =
                    (static_cast<double>(x) + 0.5) * config.voxel_size_x_mm -
                    0.5 * x_extent_mm;
                const auto y_mm =
                    (static_cast<double>(y) + 0.5) * config.voxel_size_y_mm -
                    0.5 * y_extent_mm;
                const auto z_mm =
                    (static_cast<double>(z) + 0.5) *
                    config.scorer_spacing_z_mm();
                output << x << ',' << y << ',' << z << ',' << x_mm << ',' << y_mm << ','
                       << z_mm << ',' << energy << ',' << dose_total << '\n';
            }
        }
    }
}

void write_sparse_voxel_dose_Gy_csv(const std::filesystem::path& path,
                                    const TransportConfig& config,
                                    const TransportResult& result) {
    if (!config.enable_voxel_scoring) {
        throw std::invalid_argument(
            "Voxel dose Gy output requested while voxel scoring is disabled");
    }
    if (result.voxel_deposited_energy_MeV.size() != config.number_of_voxels()) {
        throw std::invalid_argument("Transport result voxel count does not match configuration");
    }
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create voxel dose Gy output file: " + path.string());
    }
    const auto masses_kg = voxel_masses_kg(config);
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto x_extent_mm =
        static_cast<double>(config.voxel_bins_x) * config.voxel_size_x_mm;
    const auto y_extent_mm =
        static_cast<double>(config.voxel_bins_y) * config.voxel_size_y_mm;
    output << "ix,iy,iz,x_mm,y_mm,z_mm,dose_Gy\n";
    output << std::setprecision(12);
    for (std::size_t z = 0; z < config.number_of_bins(); ++z) {
        for (std::size_t y = 0; y < config.voxel_bins_y; ++y) {
            for (std::size_t x = 0; x < config.voxel_bins_x; ++x) {
                const auto index = z * plane_size + y * config.voxel_bins_x + x;
                const auto energy = result.voxel_deposited_energy_MeV[index];
                if (energy == 0.0) {
                    continue;
                }
                const auto dose_total = scored_dose_Gy(config, energy, masses_kg[index]);
                const auto x_mm =
                    (static_cast<double>(x) + 0.5) * config.voxel_size_x_mm -
                    0.5 * x_extent_mm;
                const auto y_mm =
                    (static_cast<double>(y) + 0.5) * config.voxel_size_y_mm -
                    0.5 * y_extent_mm;
                const auto z_mm =
                    (static_cast<double>(z) + 0.5) *
                    config.scorer_spacing_z_mm();
                output << x << ',' << y << ',' << z << ',' << x_mm << ',' << y_mm << ','
                       << z_mm << ',' << dose_total << '\n';
            }
        }
    }
}
void write_sparse_charged_origin_voxel_dose_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result) {
    if (!config.enable_charged_origin_voxel_scoring) {
        throw std::invalid_argument(
            "Charged-origin voxel output requested while its scoring is disabled");
    }
    const auto voxel_count = config.number_of_voxels();
    const auto expected_values = charged_origin_category_count * voxel_count;
    if (result.voxel_deposited_energy_MeV.size() != voxel_count ||
        result.charged_origin_voxel_deposited_energy_MeV.size() != expected_values) {
        throw std::invalid_argument(
            "Charged-origin voxel result size does not match configuration");
    }

    const std::array<const std::vector<double>*, charged_origin_category_count>
        depth_categories{
            &result.primary_deposited_energy_MeV,
            &result.secondary_carbon_deposited_energy_MeV,
            &result.secondary_boron_deposited_energy_MeV,
            &result.secondary_beryllium_deposited_energy_MeV,
            &result.secondary_lithium_deposited_energy_MeV,
            &result.secondary_helium_deposited_energy_MeV,
            &result.secondary_proton_deposited_energy_MeV,
            &result.secondary_other_charged_deposited_energy_MeV,
        };
    const auto bins = config.number_of_bins();
    if (std::any_of(depth_categories.begin(), depth_categories.end(),
                    [bins](const auto* category) {
                        return category->size() != bins;
                    })) {
        throw std::invalid_argument(
            "Charged-origin depth category size does not match configuration");
    }

    const auto histories = static_cast<double>(config.number_of_histories);
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
#if !defined(CARBON_DOSE_FP32)
    const auto closure_tol = [histories](double a, double b) {
        const auto scale = std::max({std::abs(a), std::abs(b), 1.0});
        return std::max(1.0e-9 * histories, 5.0e-5 * scale);
    };
#endif
    for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
        double reconstructed = 0.0;
        for (std::size_t category = 0; category < charged_origin_category_count;
             ++category) {
            reconstructed += result.charged_origin_voxel_deposited_energy_MeV[
                category * voxel_count + voxel];
        }
        const auto total = result.voxel_deposited_energy_MeV[voxel];
#if defined(CARBON_DOSE_FP32)
        // FP32 atomics are accumulated in a different order for the total and
        // origin-category arrays.  Preserve the audit, but do not discard an
        // otherwise valid formal run solely because these two reductions are
        // not bitwise identical.
        const auto scale = std::max({std::abs(reconstructed), std::abs(total), 1.0});
        const auto tol = std::max(1.0e-6 * histories, 5.0e-2 * scale);
        if (std::abs(reconstructed - total) > tol) {
            std::cerr << "warning: charged-origin voxel categories do not close to total at "
                      << "voxel " << voxel << " (diff="
                      << std::abs(reconstructed - total) << " MeV, tol=" << tol
                      << " MeV, rel=" << std::abs(reconstructed - total) / scale << ")\n";
        }
#else
        if (std::abs(reconstructed - total) > closure_tol(reconstructed, total)) {
            throw std::runtime_error(
                "Charged-origin categories do not close to total at voxel " +
                std::to_string(voxel));
        }
#endif
    }
    for (std::size_t category = 0; category < charged_origin_category_count;
         ++category) {
        const auto category_offset = category * voxel_count;
        for (std::size_t z = 0; z < bins; ++z) {
            const auto begin =
                result.charged_origin_voxel_deposited_energy_MeV.begin() +
                static_cast<std::ptrdiff_t>(category_offset + z * plane_size);
            const auto reconstructed = std::accumulate(
                begin, begin + static_cast<std::ptrdiff_t>(plane_size), 0.0);
            const auto depth = (*depth_categories[category])[z];
#if defined(CARBON_DOSE_FP32)
            // FP32 dose atomics are non-associative.  The voxel scorer also
            // reports this condition as a warning; do the same here so the
            // optional charged-origin audit cannot turn a valid formal run
            // into a hard failure solely because of accumulation order.
            const auto scale = std::max({std::abs(reconstructed), std::abs(depth), 1.0});
            const auto tol = std::max(1.0e-6 * histories, 5.0e-2 * scale);
            if (std::abs(reconstructed - depth) > tol) {
                std::cerr << "warning: charged-origin voxel category does not close to depth "
                          << "tally for category " << category << " at z bin " << z
                          << " (diff=" << std::abs(reconstructed - depth)
                          << " MeV, tol=" << tol << " MeV, rel="
                          << std::abs(reconstructed - depth) / scale << ")\n";
            }
#else
            if (std::abs(reconstructed - depth) > closure_tol(reconstructed, depth)) {
                throw std::runtime_error(
                    "Charged-origin voxel category does not close to depth tally "
                    "for category " +
                    std::to_string(category) + " at z bin " + std::to_string(z));
            }
#endif
        }
    }

    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create charged-origin voxel dose output file: " +
            path.string());
    }
    const auto x_extent_mm =
        static_cast<double>(config.voxel_bins_x) * config.voxel_size_x_mm;
    const auto y_extent_mm =
        static_cast<double>(config.voxel_bins_y) * config.voxel_size_y_mm;
    output << "ix,iy,iz,x_mm,y_mm,z_mm,total_MeV,"
              "primary_MeV,secondary_carbon_MeV,"
              "secondary_boron_MeV,secondary_beryllium_MeV,"
              "secondary_lithium_MeV,secondary_helium_MeV,"
              "secondary_proton_MeV,secondary_other_charged_MeV\n";
    output << std::setprecision(12);
    for (std::size_t z = 0; z < bins; ++z) {
        for (std::size_t y = 0; y < config.voxel_bins_y; ++y) {
            for (std::size_t x = 0; x < config.voxel_bins_x; ++x) {
                const auto voxel = z * plane_size + y * config.voxel_bins_x + x;
                const auto total = result.voxel_deposited_energy_MeV[voxel];
                if (total == 0.0) {
                    continue;
                }
                const auto x_mm =
                    (static_cast<double>(x) + 0.5) * config.voxel_size_x_mm -
                    0.5 * x_extent_mm;
                const auto y_mm =
                    (static_cast<double>(y) + 0.5) * config.voxel_size_y_mm -
                    0.5 * y_extent_mm;
                const auto z_mm =
                    (static_cast<double>(z) + 0.5) *
                    config.scorer_spacing_z_mm();
                output << x << ',' << y << ',' << z << ',' << x_mm << ','
                       << y_mm << ',' << z_mm << ',' << total;
                for (std::size_t category = 0;
                     category < charged_origin_category_count; ++category) {
                    output << ','
                           << result.charged_origin_voxel_deposited_energy_MeV[
                                  category * voxel_count + voxel];
                }
                output << '\n';
            }
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
        const auto grid = CtGrid::from_config(config);
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

void write_dense_primary_voxel_fluence_mhd(
    const std::filesystem::path& mhd_path,
    const TransportConfig& config,
    const TransportResult& result) {
    if (!config.enable_voxel_scoring) {
        throw std::invalid_argument(
            "Primary voxel fluence output requires voxel scoring");
    }
    const auto nx = config.voxel_bins_x;
    const auto ny = config.voxel_bins_y;
    const auto nz = config.number_of_bins();
    const auto count = config.number_of_voxels();
    if (result.primary_voxel_track_length_mm.size() != count) {
        throw std::invalid_argument(
            "Primary voxel track-length buffer size does not match scorer grid");
    }
    const auto voxel_volume_mm3 = config.voxel_size_x_mm *
                                  config.voxel_size_y_mm *
                                  config.scorer_spacing_z_mm();
    if (!(voxel_volume_mm3 > 0.0)) {
        throw std::invalid_argument("Primary voxel fluence requires positive volume");
    }

    std::vector<float> raw(count, 0.0F);
    for (std::size_t index = 0; index < count; ++index) {
        raw[index] = static_cast<float>(
            result.primary_voxel_track_length_mm[index] / voxel_volume_mm3);
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
        raw_out.write(reinterpret_cast<const char*>(raw.data()),
                      static_cast<std::streamsize>(raw.size() * sizeof(float)));
    }

    const auto x_extent_mm = static_cast<double>(nx) * config.voxel_size_x_mm;
    const auto y_extent_mm = static_cast<double>(ny) * config.voxel_size_y_mm;
    double origin_x = 0.5 * config.voxel_size_x_mm - 0.5 * x_extent_mm;
    double origin_y = 0.5 * config.voxel_size_y_mm - 0.5 * y_extent_mm;
    double origin_z = 0.5 * config.scorer_spacing_z_mm();
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid = CtGrid::from_config(config);
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz &&
            std::abs(static_cast<double>(grid.spacing_x_mm) -
                     config.voxel_size_x_mm) < 1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_y_mm) -
                     config.voxel_size_y_mm) < 1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_z_mm) -
                     config.scorer_spacing_z_mm()) < 1.0e-6) {
            origin_x = static_cast<double>(grid.origin_x_mm) +
                       0.5 * config.voxel_size_x_mm;
            origin_y = static_cast<double>(grid.origin_y_mm) +
                       0.5 * config.voxel_size_y_mm;
            origin_z = static_cast<double>(grid.origin_z_mm) +
                       0.5 * config.scorer_spacing_z_mm();
        }
    }

    std::ofstream header(mhd, std::ios::binary);
    if (!header) {
        throw std::runtime_error("Cannot create MHD file: " + mhd.string());
    }
    header << std::setprecision(12)
           << "ObjectType = Image\n"
           << "NDims = 3\n"
           << "BinaryData = True\n"
           << "BinaryDataByteOrderMSB = False\n"
           << "CompressedData = False\n"
           << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
           << "Offset = " << origin_x << " " << origin_y << " " << origin_z << "\n"
           << "CenterOfRotation = 0 0 0\n"
           << "ElementSpacing = " << config.voxel_size_x_mm << " "
           << config.voxel_size_y_mm << " " << config.scorer_spacing_z_mm() << "\n"
           << "DimSize = " << nx << " " << ny << " " << nz << "\n"
           << "ElementType = MET_FLOAT\n"
           << "FluenceUnits = 1/mm2\n"
           << "ElementDataFile = " << raw_path.filename().string() << "\n";
}

void write_sparse_charged_origin_voxel_dose_Gy_csv(
    const std::filesystem::path& path,
    const TransportConfig& config,
    const TransportResult& result) {
    if (!config.enable_charged_origin_voxel_scoring) {
        throw std::invalid_argument(
            "Charged-origin voxel Gy output requested while its scoring is disabled");
    }
    const auto voxel_count = config.number_of_voxels();
    const auto expected_values = charged_origin_category_count * voxel_count;
    if (result.voxel_deposited_energy_MeV.size() != voxel_count ||
        result.charged_origin_voxel_deposited_energy_MeV.size() != expected_values) {
        throw std::invalid_argument(
            "Charged-origin voxel result size does not match configuration");
    }

    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create charged-origin voxel dose Gy output file: " +
            path.string());
    }
    const auto mass_kg = voxel_mass_kg(config);
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto bins = config.number_of_bins();
    const auto x_extent_mm =
        static_cast<double>(config.voxel_bins_x) * config.voxel_size_x_mm;
    const auto y_extent_mm =
        static_cast<double>(config.voxel_bins_y) * config.voxel_size_y_mm;

    output << "ix,iy,iz,x_mm,y_mm,z_mm,total_Gy,"
              "primary_Gy,secondary_carbon_Gy,"
              "secondary_boron_Gy,secondary_beryllium_Gy,"
              "secondary_lithium_Gy,secondary_helium_Gy,"
              "secondary_proton_Gy,secondary_other_charged_Gy\n";
    output << std::setprecision(12);
    for (std::size_t z = 0; z < bins; ++z) {
        for (std::size_t y = 0; y < config.voxel_bins_y; ++y) {
            for (std::size_t x = 0; x < config.voxel_bins_x; ++x) {
                const auto voxel = z * plane_size + y * config.voxel_bins_x + x;
                const auto total = result.voxel_deposited_energy_MeV[voxel];
                if (total == 0.0) {
                    continue;
                }
                const auto x_mm =
                    (static_cast<double>(x) + 0.5) * config.voxel_size_x_mm -
                    0.5 * x_extent_mm;
                const auto y_mm =
                    (static_cast<double>(y) + 0.5) * config.voxel_size_y_mm -
                    0.5 * y_extent_mm;
                const auto z_mm =
                    (static_cast<double>(z) + 0.5) *
                    config.scorer_spacing_z_mm();
                output << x << ',' << y << ',' << z << ',' << x_mm << ','
                       << y_mm << ',' << z_mm << ','
                       << scored_dose_Gy(config, total, mass_kg);
                for (std::size_t category = 0;
                     category < charged_origin_category_count; ++category) {
                    output << ','
                           << scored_dose_Gy(
                                  config,
                                  result.charged_origin_voxel_deposited_energy_MeV[
                                      category * voxel_count + voxel],
                                  mass_kg);
                }
                output << '\n';
            }
        }
    }
}

void write_dense_charged_origin_voxel_dose_mhd(
    const std::filesystem::path& prefix,
    const TransportConfig& config,
    const TransportResult& result) {
    if (!config.enable_charged_origin_voxel_scoring ||
        !config.enable_voxel_scoring) {
        throw std::invalid_argument(
            "Dense charged-origin MHD output requires origin and voxel scoring");
    }
    const auto voxel_count = config.number_of_voxels();
    if (result.charged_origin_voxel_deposited_energy_MeV.size() !=
        charged_origin_category_count * voxel_count) {
        throw std::invalid_argument(
            "Charged-origin voxel result size does not match configuration");
    }
    if (prefix.empty()) {
        throw std::invalid_argument("Charged-origin MHD prefix is empty");
    }

    constexpr std::array<const char*, charged_origin_category_count> labels{
        "primary", "secondary_carbon", "secondary_boron",
        "secondary_beryllium", "secondary_lithium", "secondary_helium",
        "secondary_proton", "secondary_other_charged"};
    const auto masses_kg = voxel_masses_kg(config);
    const auto nx = config.voxel_bins_x;
    const auto ny = config.voxel_bins_y;
    const auto nz = config.number_of_bins();

    double origin_x = 0.5 * config.voxel_size_x_mm -
                      0.5 * static_cast<double>(nx) * config.voxel_size_x_mm;
    double origin_y = 0.5 * config.voxel_size_y_mm -
                      0.5 * static_cast<double>(ny) * config.voxel_size_y_mm;
    double origin_z = 0.5 * config.scorer_spacing_z_mm();
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid = CtGrid::from_config(config);
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz) {
            origin_x = static_cast<double>(grid.origin_x_mm) +
                       0.5 * config.voxel_size_x_mm;
            origin_y = static_cast<double>(grid.origin_y_mm) +
                       0.5 * config.voxel_size_y_mm;
            origin_z = static_cast<double>(grid.origin_z_mm) +
                       0.5 * config.scorer_spacing_z_mm();
        }
    }

    auto base = prefix;
    if (base.extension() == ".mhd") {
        base.replace_extension();
    }
    ensure_parent_directory(base);
    for (std::size_t category = 0; category < labels.size(); ++category) {
        const auto header_path = base.parent_path() /
            (base.filename().string() + "_" + labels[category] + ".mhd");
        const auto raw_path = header_path.parent_path() /
            (header_path.stem().string() + ".raw");
        std::vector<float> raw(voxel_count, 0.0F);
        const auto offset = category * voxel_count;
        for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
            raw[voxel] = static_cast<float>(scored_dose_Gy(
                config,
                result.charged_origin_voxel_deposited_energy_MeV[offset + voxel],
                masses_kg[voxel]));
        }
        {
            std::ofstream output(raw_path, std::ios::binary);
            if (!output) {
                throw std::runtime_error(
                    "Cannot create charged-origin RAW file: " + raw_path.string());
            }
            output.write(reinterpret_cast<const char*>(raw.data()),
                         static_cast<std::streamsize>(raw.size() * sizeof(float)));
        }
        std::ofstream header(header_path, std::ios::binary);
        if (!header) {
            throw std::runtime_error(
                "Cannot create charged-origin MHD file: " + header_path.string());
        }
        header << std::setprecision(12)
               << "ObjectType = Image\n"
               << "NDims = 3\n"
               << "BinaryData = True\n"
               << "BinaryDataByteOrderMSB = False\n"
               << "CompressedData = False\n"
               << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
               << "Offset = " << origin_x << ' ' << origin_y << ' '
               << origin_z << '\n'
               << "CenterOfRotation = 0 0 0\n"
               << "ElementSpacing = " << config.voxel_size_x_mm << ' '
               << config.voxel_size_y_mm << ' '
               << config.scorer_spacing_z_mm() << '\n'
               << "DimSize = " << nx << ' ' << ny << ' ' << nz << '\n'
               << "ElementType = MET_FLOAT\n"
               << "DoseUnits = Gy\n"
               << "DoseOriginCategory = " << labels[category] << '\n'
               << "ElementDataFile = " << raw_path.filename().string() << '\n';
    }
    if (!result.be_isotope_origin_voxel_deposited_energy_MeV.empty()) {
    if (result.be_isotope_origin_voxel_deposited_energy_MeV.size() !=
        be_isotope_origin_category_count * voxel_count) {
        throw std::invalid_argument("Be-isotope origin voxel result size mismatch");
    }
    constexpr std::array<const char*, be_isotope_origin_category_count>
        be_labels{"be6", "be7", "be9", "be10"};
    for (std::size_t category = 0; category < be_labels.size(); ++category) {
        const auto raw_path = base.parent_path() /
            (base.filename().string() + "_" + be_labels[category] + ".raw");
        std::vector<float> raw(voxel_count, 0.0F);
        const auto offset = category * voxel_count;
        for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
            raw[voxel] = static_cast<float>(scored_dose_Gy(
                config, result.be_isotope_origin_voxel_deposited_energy_MeV[
                            offset + voxel], masses_kg[voxel]));
        }
        std::ofstream output(raw_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("Cannot create Be-isotope RAW file: " +
                                     raw_path.string());
        }
        output.write(reinterpret_cast<const char*>(raw.data()),
                     static_cast<std::streamsize>(raw.size() * sizeof(float)));
    }
    }
    if (result.he_isotope_origin_voxel_deposited_energy_MeV.empty()) return;
    if (result.he_isotope_origin_voxel_deposited_energy_MeV.size() !=
        he_isotope_origin_category_count * voxel_count) {
        throw std::invalid_argument("He-isotope origin voxel result size mismatch");
    }
    constexpr std::array<const char*, he_isotope_origin_category_count>
        he_labels{"he3", "he4", "he_other"};
    for (std::size_t category = 0; category < he_labels.size(); ++category) {
        const auto raw_path = base.parent_path() /
            (base.filename().string() + "_" + he_labels[category] + ".raw");
        std::vector<float> raw(voxel_count, 0.0F);
        const auto offset = category * voxel_count;
        for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
            raw[voxel] = static_cast<float>(scored_dose_Gy(
                config, result.he_isotope_origin_voxel_deposited_energy_MeV[
                            offset + voxel], masses_kg[voxel]));
        }
        std::ofstream output(raw_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("Cannot create He-isotope RAW file: " +
                                     raw_path.string());
        }
        output.write(reinterpret_cast<const char*>(raw.data()),
                     static_cast<std::streamsize>(raw.size() * sizeof(float)));
    }
}

namespace {

// JSON provenance object for one Schneider rate binary: path, SHA, magic,
// version, projectile registry, grid, and channel-domain-block SHA. Returns
// {"present": false} when the file does not exist.
std::string schneider_rate_provenance_json(const std::filesystem::path& path) {
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
    std::string dom_sha = "absent";
    if (version == 3 && nt > 0 && nt <= 64) {
        const std::size_t dom_bytes =
            static_cast<std::size_t>(np) * static_cast<std::size_t>(nt) * 24;
        const auto fsz = std::filesystem::file_size(path);
        if (fsz >= dom_bytes + 8) {
            in.seekg(static_cast<std::streamoff>(fsz - dom_bytes));
            std::vector<char> dom(dom_bytes);
            in.read(dom.data(), static_cast<std::streamsize>(dom_bytes));
            if (in) {
                dom_sha = compute_sha256_hex(dom.data(), dom.size());
            }
        }
    }
    out << "{\"present\": true, \"path\": \"" << path.string() << "\", \"sha256\": \""
        << compute_file_sha256_hex(path) << "\", \"binary_magic\": \"" << std::string(magic, 8)
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
        out << "{\"z\": 6, \"a\": 12, \"note\": \"C12-only, no projectile axis\"}";
    }
    out << "], \"energy_grid\": {\"emin\": " << emin << ", \"emax\": " << emax
        << ", \"step\": " << estep << ", \"count\": " << ne
        << "}, \"channel_domain_block_sha256\": \"" << dom_sha << "\"}";
    return out.str();
}

// JSON provenance object for one CINEL03 event package binary (+ channels
// sidecar SHA when the <stem>.channels.json sibling is present).
std::string schneider_package_provenance_json(const std::filesystem::path& path) {
    std::ostringstream out;
    if (path.empty() || !std::filesystem::exists(path)) {
        return "{\"present\": false}";
    }
    out << "{\"present\": true, \"path\": \"" << path.string() << "\", \"sha256\": \""
        << compute_file_sha256_hex(path) << "\"";
    const std::filesystem::path sidecar =
        path.parent_path() / (path.stem().string() + ".channels.json");
    if (std::filesystem::exists(sidecar)) {
        out << ", \"channels_file\": \"" << sidecar.string() << "\", \"channels_sha256\": \""
            << compute_file_sha256_hex(sidecar) << "\"";
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
        config.ct_schneider_c12_cinel03_file;
    const std::filesystem::path secondary_pkg =
        config.ct_schneider_secondary_cinel03_file;
    output << "  \"schneider_physics_provenance\": {\n"
           << "    \"primary_rate\": " << schneider_rate_provenance_json(primary_rate) << ",\n"
           << "    \"secondary_rate\": " << schneider_rate_provenance_json(secondary_rate) << ",\n"
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
            registry = (version == 3) ? "v2.1-14p" : (version == 1) ? "v1-13" : "unknown";
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


}  // namespace

// Declared research approximation: out-of-scope projectiles (no event data
// on disk) keep charged EM transport with secondary nuclear reactions
// disabled. Tracks/birth are measured per isotope from the unsupported-track
// log. Deposited/escaped are isotope-resolved only when the species ledger
// recorded them (null when the species ledger recorded nothing): then the global
// energy closure bounds them (birth = deposited + escaped, no loss proven
// by the balance error). Residual kinetic energy is never dumped locally.
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
            return result.cinel02_species_transport_ledger_MeV[
                iso.species * Cinel02SpeciesLedgerSchema::metric_count + metric];
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
    const auto be6_species = get_charged_species_idx(4, 6);
    std::uint64_t generated_be6_count = 0;
    std::uint64_t queued_be6_count = 0;
    double generated_be6_kinetic_MeV = 0.0;
    double queued_be6_kinetic_MeV = 0.0;
    if (be6_species >= 0 &&
        be6_species < static_cast<int>(Cinel02ReplayLedgerSchema::species_count)) {
        for (std::size_t parent = 0;
             parent < Cinel02ReplayLedgerSchema::species_count; ++parent) {
            const auto slot = parent * Cinel02ReplayLedgerSchema::species_count +
                              static_cast<std::size_t>(be6_species);
            generated_be6_count += result.cinel02_generated_transition_counts[slot];
            queued_be6_count += result.cinel02_queued_transition_counts[slot];
            generated_be6_kinetic_MeV +=
                result.cinel02_generated_transition_kinetic_MeV[slot];
            queued_be6_kinetic_MeV +=
                result.cinel02_queued_transition_kinetic_MeV[slot];
        }
    }
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
           << "  \"E_material_electron_untracked_MeV\": " << result.material_electron_untracked_MeV << ",\n"
           << "  \"E_material_photon_untracked_MeV\": " << result.material_electron_photon_untracked_MeV << ",\n"
           << "  \"E_topas_compat_discarded_kinetic_MeV\": " << result.topas_compat_discarded_kinetic_total_MeV() << ",\n"
           << "  \"E_queue_lost_MeV\": "
           << result.secondary_queue_overflow_energy_MeV +
                  result.neutral_queue_overflow_energy_MeV
           << ",\n"
           << "  \"E_queue_lost_MeV_included_in_untracked\": "
           << result.secondary_queue_overflow_energy_MeV
           << ",\n"
           << "  \"E_secondary_overflow_MeV\": "
           << result.secondary_queue_overflow_energy_MeV << ",\n"
           << "  \"E_neutral_overflow_MeV\": "
           << result.neutral_queue_overflow_energy_MeV << ",\n"
           << "  \"E_neutral_unsupported_product_MeV\": "
           << result.neutral_unsupported_product_energy_MeV << ",\n"
           << "  \"E_neutral_package_closure_residual_MeV\": "
           << result.neutral_package_closure_residual_MeV << ",\n"
           << "  \"nuclear_interactions\": " << result.nuclear_interactions << ",\n"
           << "  \"material_physics_mode\": \"" << material_physics_mode_name(config.material_physics_mode) << "\",\n"
           << "  \"upstream_air_mcs_covariance_enabled\": " << (config.spots_enable_upstream_air_mcs ? "true" : "false") << ",\n"
           << "  \"ct_primary_midpoint_stopping\": " << (config.ct_primary_midpoint_stopping ? "true" : "false") << ",\n"
           << "  \"ct_secondary_exact_faces\": " << (config.ct_secondary_exact_faces ? "true" : "false") << ",\n"
           << "  \"ct_secondary_ion_section_stopping_file\": \"" << config.ct_secondary_ion_section_stopping_file.string() << "\",\n"
           << "  \"ct_secondary_ion_section_stopping_sha256\": \"" << config.ct_secondary_ion_section_stopping_sha256 << "\",\n"
           << "  \"ct_secondary_ion_section_stopping_metadata_sha256\": \"" << config.ct_secondary_ion_section_stopping_metadata_sha256 << "\",\n"
           << "  \"ct_primary_midpoint_stopping_diagnostic\": " << (config.ct_primary_midpoint_stopping_diagnostic ? "true" : "false") << ",\n"
           << "  \"ct_secondary_exact_faces_diagnostic\": " << (config.ct_secondary_exact_faces_diagnostic ? "true" : "false") << ",\n"
           << "  \"ct_secondary_mcs_off_diagnostic\": " << (config.ct_secondary_mcs_off_diagnostic ? "true" : "false") << ",\n"
           << "  \"ct_secondary_schneider_sp_diagnostic\": " << (config.ct_secondary_schneider_sp_diagnostic ? "true" : "false") << ",\n"
           << "  \"upstream_air_mcs_file\": \"" << config.spots_upstream_air_mcs_file.string() << "\",\n"
           << "  \"upstream_air_mcs_sha256\": \"" << config.spots_upstream_air_mcs_sha256 << "\",\n"
           << "  \"ct_schneider_primary_rate_file\": \"" << config.ct_schneider_primary_rate_file.string() << "\",\n"
           << "  \"ct_schneider_c12_cinel03_file\": \"" << config.ct_schneider_c12_cinel03_file.string() << "\",\n"
           << "  \"ct_schneider_secondary_rate_file\": \"" << config.ct_schneider_secondary_rate_file.string() << "\",\n"
           << "  \"ct_schneider_secondary_cinel03_file\": \"" << config.ct_schneider_secondary_cinel03_file.string() << "\",\n"
           << "  \"ct_schneider_stopping_power_file\": \"" << config.ct_schneider_stopping_power_file.string() << "\",\n"
           << "  \"ct_schneider_cross_section_file\": \"" << config.ct_schneider_cross_section_file.string() << "\",\n"
           << "  \"ct_schneider_delta_tail_file\": \"" << config.ct_schneider_delta_tail_file.string() << "\",\n"
           << "  \"ct_schneider_delta_longitudinal_file\": \"" << config.ct_schneider_delta_longitudinal_file.string() << "\",\n"
           << "  \"E_schneider_primary_delta_tail_moved_MeV\": "
           << result.schneider_primary_delta_tail_moved_MeV << ",\n"
           << "  \"E_schneider_primary_delta_tail_fallback_MeV\": "
           << result.schneider_primary_delta_tail_fallback_MeV << ",\n"
           << "  \"E_schneider_primary_delta_tail_escaped_scorer_MeV\": "
           << result.schneider_primary_delta_tail_escaped_scorer_MeV << ",\n"
           << "  \"E_schneider_primary_delta_longitudinal_moved_MeV\": "
           << result.schneider_primary_delta_longitudinal_moved_MeV << ",\n"
           << "  \"E_schneider_primary_delta_longitudinal_fallback_MeV\": "
           << result.schneider_primary_delta_longitudinal_fallback_MeV << ",\n"
           << "  \"E_schneider_primary_delta_longitudinal_escaped_scorer_MeV\": "
           << result.schneider_primary_delta_longitudinal_escaped_scorer_MeV << ",\n"
           << "  \"ct_schneider_physics_bundle_file\": \"" << config.ct_schneider_physics_bundle_file.string() << "\",\n";
    output << "  \"ct_electron_segment_transport\": "
           << (!config.ct_electron_joint_response_diagnostic_file.empty() ? "true" : "false") << ",\n";
    output << "  \"unified_water_nuclear_transport\": " << (config.unified_water_nuclear_transport ? "true" : "false") << ",\n";
    output << "  \"all_ion_primary_elastic_interactions\": " << result.primary_elastic_interactions << ",\n"
           << "  \"all_ion_secondary_elastic_interactions\": " << result.secondary_elastic_interactions << ",\n"
           << "  \"elastic_post_em_null_collisions\": " << result.elastic_post_em_null_collisions << ",\n";
    output << "  \"all_ion_elastic_file\": \"" << config.all_ion_elastic_file.string() << "\",\n"
           << "  \"all_ion_elastic_sha256\": \"" << config.all_ion_elastic_sha256 << "\",\n"
           << "  \"elastic_recoil_stopping_sha256\": \"" << config.elastic_recoil_stopping_sha256 << "\",\n";
    if (config.unified_water_nuclear_transport) {
        output << "  \"unified_water_material_sha256\": \"" << config.unified_water_material_sha256 << "\",\n"
               << "  \"unified_water_primary_stopping_file\": \"" << config.primary_stopping_power_file.string() << "\",\n"
               << "  \"unified_water_material_id_note\": \"255 is a diagnostic sentinel, not a Schneider section; rate uses a separate water row\",\n";
    }
    if(!config.material_electron_response_index_file.empty()) {
        const auto& d=result.electron_joint_diagnostics;
        output<<"  \"material_electron_response\": {\"status\": \"unvalidated_main_kernel_integration\", "
              <<"\"density_sampling\": \""<<(config.enable_ct_grid?"candidate_same_section_density_mixture_birth_energy_packets":"deposited_fraction_weighted_brackets_no_coordinate_scaling")<<"\", "
              <<"\"index_sha256\": \""<<config.material_electron_response_index_sha256<<"\", "
              <<"\"memory_mode\": \""<<config.material_electron_response_memory_mode<<"\", "
              <<"\"density_g_cm3\": "<<(config.enable_ct_grid?"null":std::to_string(config.water_density_g_per_cm3))<<", "
              <<"\"heterogeneous_continuation\": "<<(config.enable_ct_grid?"true":"false")
              <<", \"unresolved_photon_policy\": \""<<(config.enable_ct_grid?"separate_untracked_energy_not_patient_escape":"explicit_escape_bound_not_validated")<<"\", "
              <<"\"queries\": "<<d.queries<<", \"domain_misses\": "<<d.domain_misses
              <<", \"invalid_paths\": "<<d.invalid_marches<<", \"path_replays\": "<<d.ordered_path_replays
              <<", \"voxel_redistributed_MeV\": "<<d.redistributed_MeV
              <<", \"outside_packet_MeV\": "<<d.escaped_MeV<<"},\n";
    }
    if(!config.water_electron_response_diagnostic_file.empty()) {
        const auto& d=result.electron_joint_diagnostics;
        output<<"  \"water_electron_response\": {\"status\": \""
              <<(config.water_electron_nuclear_diagnostic ? "unvalidated_water_nuclear_diagnostic" : "unvalidated_water_em_diagnostic")<<"\", "
              <<"\"nuclear_diagnostic\": "<<(config.water_electron_nuclear_diagnostic ? "true" : "false")<<", "
              <<"\"data_sha256\": \""<<config.water_electron_response_sha256<<"\", "
              <<"\"metadata_sha256\": \""<<config.water_electron_response_metadata_sha256<<"\", "
              <<"\"unresolved_photon_policy\": \"explicit_escape_bound_not_validated\", "
              <<"\"queries\": "<<d.queries<<", \"domain_misses\": "<<d.domain_misses
              <<", \"invalid_paths\": "<<d.invalid_marches<<", \"path_replays\": "<<d.ordered_path_replays
              <<", \"voxel_redistributed_MeV\": "<<d.redistributed_MeV
              <<", \"physical_or_scorer_outside_packet_MeV\": "<<d.escaped_MeV
              <<", \"lookup_retained_MeV\": "<<d.domain_retained_MeV<<"},\n";
    }
    if(!config.ct_electron_joint_response_diagnostic_file.empty()) {
        const auto& d=result.electron_joint_diagnostics;
        output<<"  \"electron_joint_response\": {\"status\": \"unvalidated_interface_diagnostic\", "
              <<"\"patient_experiment\": "<<(config.ct_electron_joint_patient_experiment ? "true" : "false")<<", "
              <<"\"data_sha256\": \""<<config.ct_electron_joint_response_sha256<<"\", "
              <<"\"metadata_sha256\": \""<<config.ct_electron_joint_response_metadata_sha256<<"\", "
              <<"\"ordered_path_sha256\": \""<<result.electron_ordered_path_sha256<<"\", "
              <<"\"ordered_path_replays\": "<<d.ordered_path_replays<<", "
              <<"\"replaces_transverse_tail\": true, \"queries\": "<<d.queries
              <<", \"domain_misses\": "<<d.domain_misses<<", \"invalid_marches\": "<<d.invalid_marches
              <<", \"redistributed_MeV\": "<<d.redistributed_MeV<<", \"escaped_MeV\": "<<d.escaped_MeV
              <<", \"domain_retained_MeV\": "<<d.domain_retained_MeV<<"},\n";
    }
    if (!config.ct_schneider_delta_longitudinal_file.empty()) {
        const auto& file = config.ct_schneider_delta_longitudinal_file;
        const auto metadata = file.parent_path() / (file.stem().string() + ".metadata.json");
        const auto manifest = file.parent_path() / (file.stem().string() + ".candidate.json");
        output << "  \"longitudinal_candidate\": {\"status\": \"unvalidated_diagnostic\", "
               << "\"scale\": " << config.ct_schneider_delta_longitudinal_scale
               << ", \"homogeneous_density_diagnostic\": "
               << (config.ct_longitudinal_homogeneous_density_diagnostic ? "true" : "false")
               << ", \"interface_mass_diagnostic\": "
               << (config.ct_longitudinal_interface_mass_diagnostic ? "true" : "false")
               << ", \"runtime_kernel\": \""
               << (config.ct_longitudinal_interface_mass_diagnostic
                       ? "interface_mass_column_uniform_birth_diagnostic_v2" : config.ct_longitudinal_homogeneous_density_diagnostic
                       ? "homogeneous_mass_thickness_diagnostic_v1"
                       : "homogeneous_reference_density_v1")
               << "\", \"kernel_density_g_cm3\": "
               << result.longitudinal_diagnostic_density_g_cm3
               << ", \"data_sha256\": \"" << compute_file_sha256_hex(file)
               << "\", \"metadata_sha256\": \"" << compute_file_sha256_hex(metadata)
               << "\", \"manifest_sha256\": \"" << compute_file_sha256_hex(manifest)
               << "\", \"out_of_domain_queries\": "
               << result.schneider_primary_delta_longitudinal_domain_queries
               << ", \"out_of_domain_local_energy_MeV\": "
               << result.schneider_primary_delta_longitudinal_domain_energy_MeV
               << ", \"invalid_marches\": "
               << result.schneider_primary_delta_longitudinal_invalid_marches
               << "},\n";
        if (config.ct_longitudinal_homogeneous_density_diagnostic || config.ct_longitudinal_interface_mass_diagnostic) {
            output << "  \"longitudinal_domain_log\": {\"capacity\": "
                   << kLongitudinalDomainLogCap << ", \"truncated\": "
                   << (result.schneider_primary_delta_longitudinal_domain_queries >
                       result.longitudinal_domain_log.size() ? "true" : "false")
                   << ", \"records\": [";
            for (std::size_t i=0; i<result.longitudinal_domain_log.size(); ++i) {
                const auto& r = result.longitudinal_domain_log[i];
                if (i) output << ',';
                output << "{\"history\":" << r.history << ",\"step\":" << r.step
                       << ",\"initial_energy_MeVu\":" << r.initial_energy_MeVu
                       << ",\"query_energy_MeVu\":" << r.query_energy_MeVu
                       << ",\"step_start_z_mm\":" << r.step_start_z_mm
                       << ",\"step_length_mm\":" << r.step_length_mm
                       << ",\"retained_MeV\":" << r.retained_MeV << '}';
            }
            output << "]},\n";
        }
    }
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
           << "  \"cinel02_energy_ledger_layout\": "
              "[\"collision_input_kinetic\", \"pre_collision_em_loss\", "
              "\"process_local_deposit\", \"niel_diagnostic_only\", "
              "\"surviving_parent_kinetic\", \"charged_product_kinetic\", "
              "\"neutral_product_kinetic\", \"unsupported_product_kinetic\"],\n"
           << "  \"cinel02_energy_ledger_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_energy_ledger_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_energy_ledger_MeV[i];
    }
    const auto cinel02_kinetic_q_bucket =
        result.cinel02_energy_ledger_MeV[0] -
        result.cinel02_energy_ledger_MeV[2] -
        result.cinel02_energy_ledger_MeV[4] -
        result.cinel02_energy_ledger_MeV[5] -
        result.cinel02_energy_ledger_MeV[6] -
        result.cinel02_energy_ledger_MeV[7];
    output << "],\n"
           << "  \"cinel02_species_transport_ledger_layout\": "
              "{\"shape\":[" << Cinel02SpeciesLedgerSchema::species_count << ","
           << Cinel02SpeciesLedgerSchema::metric_count
           << "],\"order\":[\"species\",\"metric\"],"
              "\"species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\","
              "\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\","
              "\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"],"
              "\"metric\":[\"queued_birth_kinetic\",\"continuous_deposit_all\","
              "\"continuous_deposit_fov\",\"nuclear_local_deposit_all\","
              "\"nuclear_local_deposit_fov\",\"terminal_deposit_all\","
              "\"terminal_deposit_fov\",\"boundary_escape_kinetic\",\"reaction_export_kinetic\",\"step_limit_escape_kinetic\",\"reaction_import_kinetic\","
              "\"cinel03_replay_input_fov\",\"cinel03_replay_step_dE_fov\"]},\n"
           << "  \"cinel02_species_transport_ledger_MeV\": [";
    for (std::size_t i = 0;
         i < result.cinel02_species_transport_ledger_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_species_transport_ledger_MeV[i];
    }
    output << "],\n";
    write_out_of_scope_isotope_summary(output, result);
    output << "  \"cinel02_reference_compatibility\": {\n"
           << "    \"mode\": \""
           << (config.cinel02_topas_compatibility_mode
                   ? "geant4_11_3_2_topas_4_2_p3"
                   : "native_transport")
           << "\",\n"
           << "    \"unsupported_prompt_ion_policy\": \""
           << (config.cinel02_topas_compatibility_mode
                   ? "topas_compat_kill"
                   : "stable_for_transport")
           << "\",\n"
           << "    \"policy_species\": [\"6Be\"],\n"
           << "    \"produced_be6_count\": " << generated_be6_count << ",\n"
           << "    \"produced_be6_kinetic_MeV\": " << generated_be6_kinetic_MeV << ",\n"
           << "    \"queued_be6_count\": " << queued_be6_count << ",\n"
           << "    \"queued_be6_kinetic_MeV\": " << queued_be6_kinetic_MeV << ",\n"
           << "    \"topas_compat_discarded_counts\": [";
    for (std::size_t i = 0;
         i < result.cinel02_topas_compat_discarded_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_topas_compat_discarded_counts[i];
    }
    const auto discarded_kinetic_total = std::accumulate(
        result.cinel02_topas_compat_discarded_kinetic_MeV.begin(),
        result.cinel02_topas_compat_discarded_kinetic_MeV.end(), 0.0);
    const auto discarded_count_total = std::accumulate(
        result.cinel02_topas_compat_discarded_counts.begin(),
        result.cinel02_topas_compat_discarded_counts.end(), std::uint64_t{0});
    output << "],\n"
           << "    \"topas_compat_discarded_kinetic_MeV\": [";
    for (std::size_t i = 0;
         i < result.cinel02_topas_compat_discarded_kinetic_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_topas_compat_discarded_kinetic_MeV[i];
    }
    output << "],\n"
           << "    \"topas_compat_discarded_count_total\": "
           << discarded_count_total << ",\n"
           << "    \"topas_compat_discarded_kinetic_total_MeV\": "
           << discarded_kinetic_total << "\n"
           << "  },\n"
           << "  \"cinel02_replay_handoff_layout\": "
              "{\"species\": [\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\","
              "\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\","
              "\"10B\",\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"],"
              "\"delta_unit\": \"MeV_per_u\","
              "\"delta_sign\": \"package_incident_minus_runtime_incident\"},\n"
           << "  \"cinel02_replay_delta_MeV_per_u\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_delta_MeV_per_u.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_delta_MeV_per_u[i];
    }
    output << "],\n  \"cinel02_replay_abs_delta_MeV_per_u\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_abs_delta_MeV_per_u.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_abs_delta_MeV_per_u[i];
    }
    output << "],\n  \"cinel02_replay_delta_positive_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_delta_positive_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_delta_positive_counts[i];
    }
    output << "],\n  \"cinel02_replay_delta_negative_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_delta_negative_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_delta_negative_counts[i];
    }
    output << "],\n  \"cinel02_replay_valid_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_valid_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_valid_counts[i];
    }
    output << "],\n"
           << "  \"cinel02_replay_status_layout\": "
              "{\"shape\":[18,2,3,40,5],\"order\":[\"projectile_species\",\"target\",\"reaction_generation\",\"energy_bin\",\"status\"],"
              "\"status\":[\"collision_candidate\",\"replay_valid\",\"replay_lookup_miss\",\"replay_invalid_event\",\"post_em_below_cutoff\"],"
              "\"projectile_species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\",\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\",\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"],"
              "\"targets\":[\"H\",\"O\"],\"reaction_generations\":[0,1,2],"
              "\"energy_bin_edges_MeV_per_u\":[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,\"infinity\"]},\n"
           << "  \"cinel02_replay_status_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_replay_status_counts[i];
    }
    output << "],\n  \"cinel02_replay_status_rate_query_energy_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_rate_query_energy_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_replay_status_rate_query_energy_MeV[i];
    }
    output << "],\n  \"cinel02_replay_status_replay_query_energy_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_replay_query_energy_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_replay_status_replay_query_energy_MeV[i];
    }
    output << "],\n  \"cinel02_replay_status_continuous_loss_to_collision_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_continuous_loss_to_collision_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_replay_status_continuous_loss_to_collision_MeV[i];
    }
    output << "],\n  \"cinel02_replay_status_delta_MeV_per_u\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_delta_MeV_per_u.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_replay_status_delta_MeV_per_u[i];
    }
    output << "],\n  \"cinel02_replay_status_abs_delta_MeV_per_u\": [";
    for (std::size_t i = 0; i < result.cinel02_replay_status_abs_delta_MeV_per_u.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_replay_status_abs_delta_MeV_per_u[i];
    }
    output << "],\n"
           << "  \"cinel02_secondary_exposure_layout\": "
              "{\"shape\":[18,3,40],\"order\":[\"projectile_species\",\"transport_generation\",\"energy_bin\"],"
              "\"sum_metrics\":[\"path_mm_total\",\"path_mm_generation_eligible\",\"path_mm_generation_blocked\",\"path_mm_rate_covered\",\"path_mm_rate_uncovered\",\"path_mm_h_uncovered\",\"path_mm_o_uncovered\",\"hazard_h\",\"hazard_o\",\"hazard_total\",\"hazard_blocked_h\",\"hazard_blocked_o\",\"hazard_blocked_total\",\"path_mm_continuous_rate_covered\",\"path_mm_continuous_rate_uncovered\",\"hazard_continuous\",\"hazard_blocked_continuous\",\"stopping_loss_MeV\"],"
              "\"count_metrics\":[\"collision_candidates\",\"replay_valid\",\"parent_killed\",\"parent_continued\"],"
              "\"energy_bin_width_MeV_per_u\":10,\"transport_generations\":[0,1,2],"
              "\"projectile_species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\",\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\",\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"]},\n"
           << "  \"cinel02_secondary_exposure_sums\": [";
    for (std::size_t i = 0;
         i < result.cinel02_secondary_exposure_sums.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_secondary_exposure_sums[i];
    }
    output << "],\n  \"cinel02_secondary_exposure_counts\": [";
    for (std::size_t i = 0;
         i < result.cinel02_secondary_exposure_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_secondary_exposure_counts[i];
    }
    output << "],\n"
           << "  \"cinel02_parent_outcome_layout\": "
              "{\"shape\":[18,2,3,2],\"order\":[\"projectile_species\",\"target\",\"reaction_generation\",\"outcome\"],"
              "\"outcome\":[\"continued\",\"killed\"],\"targets\":[\"H\",\"O\"],\"reaction_generations\":[0,1,2]},\n"
           << "  \"cinel02_parent_outcome_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_parent_outcome_counts[i];
    }
    output << "],\n  \"cinel02_parent_outcome_incident_energy_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_incident_energy_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_parent_outcome_incident_energy_MeV[i];
    }
    output << "],\n  \"cinel02_parent_outcome_after_energy_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_after_energy_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_parent_outcome_after_energy_MeV[i];
    }
    output << "],\n  \"cinel02_parent_outcome_local_deposit_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_local_deposit_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_parent_outcome_local_deposit_MeV[i];
    }
    output << "],\n  \"cinel02_parent_outcome_export_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_export_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_parent_outcome_export_MeV[i];
    }
    output << "],\n  \"cinel02_parent_outcome_import_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_parent_outcome_import_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_parent_outcome_import_MeV[i];
    }
    output << "],\n"
           << "  \"cinel02_transition_layout\": "
              "{\"shape\":[18,18],\"order\":[\"parent_species\",\"child_species\"],"
              "\"species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\",\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\",\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"]},\n"
           << "  \"cinel02_generated_transition_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_generated_transition_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_generated_transition_counts[i];
    }
    output << "],\n  \"cinel02_generated_transition_kinetic_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_generated_transition_kinetic_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_generated_transition_kinetic_MeV[i];
    }
    output << "],\n  \"cinel02_queued_transition_counts\": [";
    for (std::size_t i = 0; i < result.cinel02_queued_transition_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_queued_transition_counts[i];
    }
    output << "],\n  \"cinel02_queued_transition_kinetic_MeV\": [";
    for (std::size_t i = 0; i < result.cinel02_queued_transition_kinetic_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_queued_transition_kinetic_MeV[i];
    }
    output << "],\n"
           << "  \"cinel02_species_terminal_reason_layout\": "
              "{\"shape\":[18,6],\"order\":[\"species\",\"reason\"],"
              "\"reason\":[\"initial_below_cutoff\",\"reaction_killed\","
              "\"terminal_deposit\",\"boundary_escape\",\"step_limit\","
              "\"continuous_stop\"]},\n"
           << "  \"cinel02_species_terminal_reason_counts\": [";
    for (std::size_t i = 0;
         i < result.cinel02_species_terminal_reason_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_species_terminal_reason_counts[i];
    }
    output << "],\n"
           << "  \"cinel02_kinetic_q_excitation_bucket_MeV\": "
           << cinel02_kinetic_q_bucket << ",\n"
           << "  \"cinel02_diagnostic_layout\": "
              "[\"primary_rate_queries\", \"primary_rate_covered\", \"primary_hazards\", "
              "\"primary_event_hits\", \"primary_event_misses\", \"primary_invalid_events\", "
              "\"primary_h_hazards\", \"primary_o_hazards\", \"primary_parent_continue\", "
              "\"primary_parent_killed\", \"primary_role0\", \"primary_role1\", "
              "\"primary_role2\", \"primary_role_other\", \"secondary_rate_queries\", "
              "\"secondary_rate_covered\", \"secondary_hazards\", \"secondary_event_hits\", "
              "\"secondary_event_misses\", \"secondary_invalid_events\", "
              "\"secondary_h_hazards\", \"secondary_o_hazards\", \"born_z1\", \"born_z2\", "
              "\"born_z3\", \"born_z4\", \"born_z5\", \"born_z6\", \"queued_z1\", "
              "\"queued_z2\", \"queued_z3\", \"queued_z4\", \"queued_z5\", \"queued_z6\", \"unsupported_charged_isotope\"],\n"
              "  \"cinel02_hazard_histogram_layout\": {\"offset\": 64, "
              "\"shape\": [18, 2, 3, 8], \"order\": "
              "[\"projectile_species\", \"target\", \"generation\", \"energy_bin\"], "
              "\"projectile_species\": [\"1H\", \"2H\", \"3H\", \"3He\", \"4He\", "
              "\"6He\", \"6Li\", \"7Li\", \"7Be\", \"9Be\", \"10Be\", \"8B\", "
              "\"10B\", \"11B\", \"10C\", \"11C\", \"12C\", \"reserved\"], "
              "\"targets\": [\"H\", \"O\"], \"generations\": [0, 1, 2], "
              "\"energy_bin_edges_MeV_per_u\": [0, 50, 100, 150, 200, 250, 300, 350, "
              "\"infinity\"]},\n"
              "  \"cinel02_secondary_transition_layout\": {\"offset\": 928, "
              "\"shape\": [2, 6, 7], \"order\": [\"generation\", \"parent_z\", \"child_z_bucket\"], "
              "\"generations\": [1, 2], \"parent_z\": [1, 2, 3, 4, 5, 6], "
              "\"child_z_bucket\": [0, 1, 2, 3, 4, 5, 6], "
              "\"child_bucket_0\": \"role0 product outside Z=1..6\"},\n"
              "  \"cinel02_secondary_parent_outcome_layout\": {\"offset\": 1012, "
              "\"shape\": [2, 6, 2], \"order\": [\"generation\", \"parent_z\", \"outcome\"], "
              "\"generations\": [1, 2], \"parent_z\": [1, 2, 3, 4, 5, 6], "
              "\"outcome\": [\"continue\", \"killed\"]},\n"
              "  \"cinel02_secondary_transition_energy_layout\": {\"offset\": 1036, "
              "\"shape\": [2, 6, 7], \"unit\": \"keV\", "
              "\"order\": [\"generation\", \"parent_z\", \"child_z_bucket\"]},\n"
              "  \"cinel02_secondary_parent_incident_energy_layout\": {\"offset\": 1120, "
              "\"shape\": [2, 6], \"unit\": \"keV\", "
              "\"generations\": [1, 2], \"parent_z\": [1, 2, 3, 4, 5, 6]},\n"
              "  \"cinel02_be_channel_count_layout\": {\"offset\": 1132, "
              "\"shape\": [2, 2, 8, 3], \"order\": [\"generation\", \"target\", \"energy_bin\", \"parent_channel\"], "
              "\"generations\": [1, 2], \"targets\": [\"H\", \"O\"], "
              "\"parent_channels\": [\"Be_to_Be\", \"B_to_Be\", \"C_to_Be\"]},\n"
              "  \"cinel02_be_channel_energy_layout\": {\"offset\": 1228, "
              "\"shape\": [2, 2, 8, 3], \"unit\": \"keV\"},\n"
              "  \"cinel02_be_incident_count_layout\": {\"offset\": 1324, "
              "\"shape\": [2, 2, 8]},\n"
              "  \"cinel02_be_incident_energy_layout\": {\"offset\": 1356, "
              "\"shape\": [2, 2, 8], \"unit\": \"keV\", "
              "\"energy_bin_edges_MeV_per_u\": [0, 50, 100, 150, 200, 250, 300, 350, \"infinity\"]},\n"
              "  \"cinel02_be_isotope_birth_count_layout\": {\"offset\": 1388, "
              "\"shape\": [2, 2, 3, 4], \"order\": [\"generation\", \"target\", \"parent_channel\", \"be_isotope\"], "
              "\"generations\": [0, 1], \"targets\": [\"H\", \"O\"], "
              "\"parent_channels\": [\"Be\", \"B\", \"C\"], "
              "\"be_isotopes\": [6, 7, 9, 10]},\n"
              "  \"cinel02_be_isotope_birth_energy_layout\": {\"offset\": 1436, "
              "\"shape\": [2, 2, 3, 4], \"unit\": \"keV\"},\n"
              "  \"cinel02_be_isotope_birth_depth_layout\": {\"offset\": 1484, "
              "\"shape\": [2, 2, 3, 4], \"unit\": \"um\"},\n"
              "  \"cinel02_be_isotope_birth_cosine_layout\": {\"offset\": 1532, "
              "\"shape\": [2, 2, 3, 4], \"encoding\": \"sum((cosine+1)*1e6)\", "
              "\"frame\": \"projectile-local incident direction\"},\n"
              "  \"cinel02_c12_o16_be10_count_by_parent_energy_layout\": {\"offset\": 1580, "
              "\"shape\": [8], \"energy_bin_edges_MeV_per_u\": [0, 50, 100, 150, 200, 250, 300, 350, \"infinity\"]},\n"
              "  \"cinel02_c12_o16_be10_energy_by_parent_energy_layout\": {\"offset\": 1588, "
              "\"shape\": [8], \"unit\": \"keV\"},\n"
              "  \"cinel02_c12_o16_reaction_parent_energy_layout\": {\"offset\": 1596, "
              "\"shape\": [8], \"unit\": \"keV_per_u\"},\n"              "  \"cinel02_c12_o16_be_isotope_count_by_parent_energy_layout\": {\"offset\": 1604, "
              "\"shape\": [4, 8], \"order\": [\"be_isotope\", \"parent_energy_bin\"], "
              "\"be_isotopes\": [6, 7, 9, 10]},\n"
              "  \"cinel02_c12_o16_be_isotope_energy_by_parent_energy_layout\": {\"offset\": 1636, "
              "\"shape\": [4, 8], \"unit\": \"keV\"},\n"
              "  \"cinel02_diagnostics\": [";
    for (std::size_t i = 0; i < result.cinel02_diagnostics.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.cinel02_diagnostics[i];
    }
    output << "],\n"
           << "  \"primary_elastic_interactions\": "
           << result.primary_elastic_interactions << ",\n"
           << "  \"elastic_local_deposited_MeV\": "
           << result.elastic_local_deposited_energy_MeV << ",\n"
           << "  \"elastic_queued_charged_MeV\": "
           << result.elastic_queued_charged_energy_MeV << ",\n"
           << "  \"elastic_queued_neutral_MeV\": "
           << result.elastic_queued_neutral_energy_MeV << ",\n"
           << "  \"elastic_queue_overflow\": "
           << result.elastic_queue_overflow << ",\n"
           << "  \"elastic_queue_overflow_MeV\": "
           << result.elastic_queue_overflow_energy_MeV << ",\n"
           << "  \"cascade_interactions\": " << result.cascade_interactions << ",\n"
           << "  \"secondary_queue_overflow\": " << result.secondary_queue_overflow
           << ",\n"
           << "  \"cascade_queue_overflow\": " << result.cascade_queue_overflow
           << ",\n"
           << "  \"neutral_queue_overflow\": " << result.neutral_queue_overflow
           << ",\n"
           << "  \"physical_energy_balance_error\": "
           << result.physical_relative_energy_balance_error() << ",\n"
           << "  \"energy_balance_error\": "
           << result.relative_energy_balance_error() << "\n"
           << "}\n";
}

void write_validation_scorer_csvs(const std::filesystem::path& directory,
                                  const TransportConfig& config,
                                  const TransportResult& result) {
    const auto bins = config.number_of_bins();
    const auto write_double = [&](const std::filesystem::path& name,
                                  const char* header,
                                  const std::vector<double>& values) {
        if (values.size() != bins) {
            return;
        }
        const auto path = directory / name;
        ensure_parent_directory(path);
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("Cannot create " + path.string());
        }
        output << "depth_mm," << header << '\n';
        output << std::setprecision(12);
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const auto depth =
                (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
            output << depth << ',' << values[bin] << '\n';
        }
    };
    const auto write_count = [&](const std::filesystem::path& name,
                                 const char* header,
                                 const std::vector<std::uint64_t>& values) {
        if (values.size() != bins) {
            return;
        }
        const auto path = directory / name;
        ensure_parent_directory(path);
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("Cannot create " + path.string());
        }
        output << "depth_mm," << header << '\n';
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const auto depth =
                (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
            output << depth << ',' << values[bin] << '\n';
        }
    };
    const auto bin_mass = idd_bin_mass_kg(config);
    if (result.primary_deposited_energy_MeV.size() == bins) {
        std::vector<double> primary_gy(bins);
        std::vector<double> secondary_gy(bins);
        for (std::size_t bin = 0; bin < bins; ++bin) {
            primary_gy[bin] = scored_dose_Gy(
                config, result.primary_deposited_energy_MeV[bin], bin_mass);
            const auto secondary_MeV =
                result.deposited_energy_MeV.size() == bins
                    ? result.deposited_energy_MeV[bin] -
                          result.primary_deposited_energy_MeV[bin]
                    : 0.0;
            secondary_gy[bin] = scored_dose_Gy(config, secondary_MeV, bin_mass);
        }
        write_double("dose_primary.csv", "dose_Gy", primary_gy);
        write_double("dose_secondary.csv", "dose_Gy", secondary_gy);
    }
    const auto area = config.scorer_area_mm2;
    const auto to_fluence = [area](const std::vector<double>& track_mm) {
        std::vector<double> phi(track_mm.size(), 0.0);
        if (!(area > 0.0)) {
            return phi;
        }
        for (std::size_t i = 0; i < track_mm.size(); ++i) {
            phi[i] = track_mm[i] / area;
        }
        return phi;
    };
    write_double("fluence_primary.csv", "fluence_per_mm2",
                 to_fluence(result.primary_fluence_mm));
    write_double("fluence_C.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_carbon_fluence_mm));
    write_double("fluence_B.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_boron_fluence_mm));
    write_double("fluence_Be.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_beryllium_fluence_mm));
    write_double("fluence_Li.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_lithium_fluence_mm));
    write_double("fluence_He.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_helium_fluence_mm));
    write_double("fluence_p.csv", "fluence_per_mm2",
                 to_fluence(result.secondary_proton_fluence_mm));
    write_count("primary_survival.csv", "count", result.primary_survival_counts);
    write_count("inelastic_reactions.csv", "count",
                result.inelastic_reaction_counts);
}

namespace {

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

}  // namespace

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

}  // namespace carbon
