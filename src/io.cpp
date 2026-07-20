#include "carbon/io.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/particle.hpp"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
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
    const auto volume_mm3 = config.voxel_size_x_mm * config.voxel_size_y_mm *
                            config.depth_bin_width_mm;
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
        const auto dose_total = energy_MeV_to_dose_Gy(energy_total, bin_mass_kg);
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
        maximum_dose = std::max(maximum_dose, energy_MeV_to_dose_Gy(energy_MeV, bin_mass_kg));
    }

    // Total dose over all sampled primaries (Gy, not Gy/primary).
    output << "depth_mm,dose_Gy,relative_dose\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < result.deposited_energy_MeV.size(); ++bin) {
        const auto dose_total =
            energy_MeV_to_dose_Gy(result.deposited_energy_MeV[bin], bin_mass_kg);
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
        maximum_dose = std::max(maximum_dose, energy_MeV_to_dose_Gy(energy_MeV, bin_mass_kg));
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
            output << ',' << energy_MeV_to_dose_Gy((*column)[bin], bin_mass_kg);
        }
        const auto total_dose =
            energy_MeV_to_dose_Gy(result.deposited_energy_MeV[bin], bin_mass_kg);
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
        // Absolute MeV closure tolerance scaled by history count.
        if (std::abs(reconstructed - result.deposited_energy_MeV[z]) > 1.0e-9 * histories) {
            throw std::runtime_error(
                "Voxel dose does not close to the depth-dose tally at z bin " +
                std::to_string(z));
        }
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
                const auto dose_total = energy_MeV_to_dose_Gy(energy, masses_kg[index]);
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
                const auto dose_total = energy_MeV_to_dose_Gy(energy, masses_kg[index]);
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

    constexpr double closure_tolerance_MeV_per_primary = 1.0e-9;
    const auto histories = static_cast<double>(config.number_of_histories);
    const auto plane_size = config.voxel_bins_x * config.voxel_bins_y;
    for (std::size_t voxel = 0; voxel < voxel_count; ++voxel) {
        double reconstructed = 0.0;
        for (std::size_t category = 0; category < charged_origin_category_count;
             ++category) {
            reconstructed += result.charged_origin_voxel_deposited_energy_MeV[
                category * voxel_count + voxel];
        }
        if (std::abs(reconstructed - result.voxel_deposited_energy_MeV[voxel]) >
            closure_tolerance_MeV_per_primary * histories) {
            throw std::runtime_error(
                "Charged-origin categories do not close to total at voxel " +
                std::to_string(voxel));
        }
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
            if (std::abs(reconstructed - (*depth_categories[category])[z]) >
                closure_tolerance_MeV_per_primary * histories) {
                throw std::runtime_error(
                    "Charged-origin voxel category does not close to depth tally "
                    "for category " +
                    std::to_string(category) + " at z bin " + std::to_string(z));
            }
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
                    static_cast<float>(energy_MeV_to_dose_Gy(energy, masses_kg[index_xyz]));
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
    const auto origin_x = 0.5 * config.voxel_size_x_mm - 0.5 * x_extent_mm;
    const auto origin_y = 0.5 * config.voxel_size_y_mm - 0.5 * y_extent_mm;
    const auto origin_z = 0.5 * config.depth_bin_width_mm;

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
                       << energy_MeV_to_dose_Gy(total, mass_kg);
                for (std::size_t category = 0;
                     category < charged_origin_category_count; ++category) {
                    output << ','
                           << energy_MeV_to_dose_Gy(
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
