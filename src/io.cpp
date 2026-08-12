#include "carbon/io.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/particle.hpp"


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
    return config.voxel_size_x_mm * config.voxel_size_y_mm * config.depth_bin_width_mm *
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
    const auto dz = config.depth_bin_width_mm;
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
    const auto grid = CtGrid::from_binary(config.ct_grid_file);
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
        &result.primary_c12_deposited_energy_MeV,
        &result.secondary_carbon_deposited_energy_MeV,
        &result.boron_deposited_energy_MeV,
        &result.beryllium_deposited_energy_MeV,
        &result.lithium_deposited_energy_MeV,
        &result.helium_deposited_energy_MeV,
        &result.proton_deposited_energy_MeV,
        &result.other_charged_deposited_energy_MeV,
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
    output << "depth_mm,total_MeV,primary_c12_MeV,"
              "secondary_carbon_MeV,boron_MeV,"
              "beryllium_MeV,lithium_MeV,"
              "helium_MeV,proton_MeV,"
              "other_MeV,relative_total\n";
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
        &result.primary_c12_letd_numerator,
        &result.primary_c12_letd_denominator,
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
    output << "depth_mm,primary_c12_letd_MeV_per_mm_per_g_cm3,"
              "all_hadron_letd_MeV_per_mm_per_g_cm3,"
              "primary_c12_numerator,primary_c12_denominator_MeV,"
              "all_hadron_numerator,all_hadron_denominator_MeV\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const auto primary_denominator = result.primary_c12_letd_denominator[bin];
        const auto all_denominator = result.all_hadron_letd_denominator[bin];
        const auto primary_letd =
            primary_denominator > 0.0
                ? result.primary_c12_letd_numerator[bin] / primary_denominator
                : 0.0;
        const auto all_letd =
            all_denominator > 0.0
                ? result.all_hadron_letd_numerator[bin] / all_denominator
                : 0.0;
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm << ',' << primary_letd << ',' << all_letd << ','
               << result.primary_c12_letd_numerator[bin] << ','
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
        &result.primary_c12_voxel_letd_numerator,
        &result.primary_c12_voxel_letd_denominator,
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
    double origin_z = 0.5 * config.depth_bin_width_mm;
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid = CtGrid::from_binary(config.ct_grid_file);
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz) {
            origin_x =
                static_cast<double>(grid.origin_x_mm) +
                0.5 * config.voxel_size_x_mm;
            origin_y =
                static_cast<double>(grid.origin_y_mm) +
                0.5 * config.voxel_size_y_mm;
            origin_z =
                static_cast<double>(grid.origin_z_mm) +
                0.5 * config.depth_bin_width_mm;
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
               << config.depth_bin_width_mm << '\n'
               << "DimSize = " << nx << ' ' << ny << ' ' << nz << '\n'
               << "ElementType = MET_FLOAT\n"
               << "LETUnits = MeV/mm/(g/cm3)\n"
               << "ElementDataFile = " << raw_path.filename().string() << '\n';
    };

    write_map("_primary_c12",
              result.primary_c12_voxel_letd_numerator,
              result.primary_c12_voxel_letd_denominator);
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
        "primary_c12", "secondary_carbon", "boron", "beryllium",
        "lithium", "helium", "hydrogen", "other_charged"};
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
        &result.primary_c12_deposited_energy_MeV,
        &result.secondary_carbon_deposited_energy_MeV,
        &result.boron_deposited_energy_MeV,
        &result.beryllium_deposited_energy_MeV,
        &result.lithium_deposited_energy_MeV,
        &result.helium_deposited_energy_MeV,
        &result.proton_deposited_energy_MeV,
        &result.other_charged_deposited_energy_MeV,
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

    output << "depth_mm,total_Gy,primary_c12_Gy,"
              "secondary_carbon_Gy,boron_Gy,"
              "beryllium_Gy,lithium_Gy,"
              "helium_Gy,proton_Gy,"
              "other_Gy,relative_total\n";
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
        // Depth vs plane-sum of voxels: exact for FP64 ordered adds; FP32 device
        // atomics are non-associative and lose precision when totals are huge (10M
        // histories). Use relative tolerance; on FP32 builds only warn so outputs
        // still write for benchmark comparison.
        const auto depth = result.deposited_energy_MeV[z];
        const auto abs_diff = std::abs(reconstructed - depth);
        const auto scale = std::max({std::abs(reconstructed), std::abs(depth), 1.0});
#if defined(CARBON_DOSE_FP32)
        const auto tol = std::max(1.0e-6 * histories, 5.0e-2 * scale);  // 5% rel
        if (abs_diff > tol) {
            std::cerr << "warning: voxel vs depth-dose closure weak at z bin " << z
                      << " (diff=" << abs_diff << " MeV, tol=" << tol
                      << " MeV, rel=" << (abs_diff / scale) << ")\n";
        }
#else
        const auto tol = std::max(1.0e-9 * histories, 5.0e-5 * scale);
        if (abs_diff > tol) {
            throw std::runtime_error(
                "Voxel dose does not close to the depth-dose tally at z bin " +
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
                    (static_cast<double>(z) + 0.5) * config.depth_bin_width_mm;
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
                    (static_cast<double>(z) + 0.5) * config.depth_bin_width_mm;
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
            &result.primary_c12_deposited_energy_MeV,
            &result.secondary_carbon_deposited_energy_MeV,
            &result.boron_deposited_energy_MeV,
            &result.beryllium_deposited_energy_MeV,
            &result.lithium_deposited_energy_MeV,
            &result.helium_deposited_energy_MeV,
            &result.proton_deposited_energy_MeV,
            &result.other_charged_deposited_energy_MeV,
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
              "primary_c12_MeV,secondary_carbon_MeV,"
              "boron_MeV,beryllium_MeV,"
              "lithium_MeV,helium_MeV,"
              "proton_MeV,other_charged_MeV\n";
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
                    (static_cast<double>(z) + 0.5) * config.depth_bin_width_mm;
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
    double origin_z = 0.5 * config.depth_bin_width_mm;
    if (config.enable_ct_grid && !config.ct_grid_file.empty()) {
        const auto grid = CtGrid::from_binary(config.ct_grid_file);
        if (grid.nx == nx && grid.ny == ny && grid.nz == nz &&
            std::abs(static_cast<double>(grid.spacing_x_mm) - config.voxel_size_x_mm) <
                1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_y_mm) - config.voxel_size_y_mm) <
                1.0e-6 &&
            std::abs(static_cast<double>(grid.spacing_z_mm) - config.depth_bin_width_mm) <
                1.0e-6) {
            origin_x = static_cast<double>(grid.origin_x_mm) + 0.5 * config.voxel_size_x_mm;
            origin_y = static_cast<double>(grid.origin_y_mm) + 0.5 * config.voxel_size_y_mm;
            origin_z = static_cast<double>(grid.origin_z_mm) + 0.5 * config.depth_bin_width_mm;
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
           << config.voxel_size_y_mm << ' ' << config.depth_bin_width_mm << '\n'
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
              "primary_c12_Gy,secondary_carbon_Gy,"
              "boron_Gy,beryllium_Gy,"
              "lithium_Gy,helium_Gy,"
              "proton_Gy,other_charged_Gy\n";
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
                    (static_cast<double>(z) + 0.5) * config.depth_bin_width_mm;
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

}  // namespace carbon
