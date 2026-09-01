#include "carbon/io.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/particle.hpp"
#include "carbon/detail/fred_fragmentation_data.hpp"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
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
    if (result.be_isotope_origin_voxel_deposited_energy_MeV.empty()) return;
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
           << "  \"fred_inelastic_events\": " << result.fred_inelastic_events << ",\n"
           << "  \"fred_mean_retries\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_retry_sum) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_energy_scaled_events\": " << result.fred_energy_scaled_events
           << ",\n"
           << "  \"fred_energy_scaled_fraction\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_energy_scaled_events) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_projectile_az_open_events\": "
           << result.fred_projectile_az_open_events << ",\n"
           << "  \"fred_mean_leftover_target_a\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_leftover_target_a_sum) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_mean_leftover_target_z\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_leftover_target_z_sum) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_mean_leftover_projectile_a\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_leftover_projectile_a_sum) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_mean_leftover_projectile_z\": "
           << (result.fred_inelastic_events == 0
                   ? 0.0
                   : static_cast<double>(result.fred_leftover_projectile_z_sum) /
                         static_cast<double>(result.fred_inelastic_events))
           << ",\n"
           << "  \"fred_model_residual_MeV\": " << result.fred_model_residual_MeV << ",\n"
           << "  \"fred_q_MeV\": " << result.fred_q_MeV << ",\n"
           << "  \"fred_neutron_ke_MeV\": " << result.fred_neutron_ke_MeV << ",\n"
           << "  \"fred_remnant_local_MeV\": " << result.fred_remnant_local_MeV << ",\n"
           << "  \"fred_resample_failed_events\": " << result.fred_resample_failed_events
           << ",\n"
           << "  \"fred_resample_failed_energy_MeV\": "
           << result.fred_resample_failed_energy_MeV << ",\n"
           << "  \"fred_product_capacity_overflow_events\": "
           << result.fred_product_capacity_overflow_events << ",\n"
           << "  \"fred_product_capacity_overflow_energy_MeV\": "
           << result.fred_product_capacity_overflow_energy_MeV << ",\n"
           << "  \"fred_invert_error_proj_h\": " << result.fred_invert_error_proj_h << ",\n"
           << "  \"fred_invert_error_proj_o\": " << result.fred_invert_error_proj_o << ",\n"
           << "  \"fred_invert_error_tgt_h\": " << result.fred_invert_error_tgt_h << ",\n"
           << "  \"fred_invert_error_tgt_o\": " << result.fred_invert_error_tgt_o << ",\n"
           << "  \"fred_isotope_counts\": [";
    for (std::size_t i = 0; i < result.fred_isotope_counts.size(); ++i) {
        output << (i == 0 ? "" : ", ") << result.fred_isotope_counts[i];
    }
    output << "],\n"
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
              "{\"shape\":[18,11],\"order\":[\"species\",\"metric\"],"
              "\"species\":[\"1H\",\"2H\",\"3H\",\"3He\",\"4He\",\"6He\","
              "\"6Li\",\"7Li\",\"7Be\",\"9Be\",\"10Be\",\"8B\",\"10B\","
              "\"11B\",\"10C\",\"11C\",\"12C\",\"6Be\"],"
              "\"metric\":[\"queued_birth_kinetic\",\"continuous_deposit_all\","
              "\"continuous_deposit_fov\",\"nuclear_local_deposit_all\","
              "\"nuclear_local_deposit_fov\",\"terminal_deposit_all\","
              "\"terminal_deposit_fov\",\"boundary_escape_kinetic\",\"reaction_export_kinetic\",\"step_limit_escape_kinetic\",\"reaction_import_kinetic\"]},\n"
           << "  \"cinel02_species_transport_ledger_MeV\": [";
    for (std::size_t i = 0;
         i < result.cinel02_species_transport_ledger_MeV.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << result.cinel02_species_transport_ledger_MeV[i];
    }
    output << "],\n"
           << "  \"cinel02_reference_compatibility\": {\n"
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
              "\"sum_metrics\":[\"path_mm_total\",\"path_mm_generation_eligible\",\"path_mm_generation_blocked\",\"path_mm_rate_covered\",\"path_mm_rate_uncovered\",\"path_mm_h_uncovered\",\"path_mm_o_uncovered\",\"hazard_h\",\"hazard_o\",\"hazard_total\"],"
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

}  // namespace carbon
