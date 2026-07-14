#include "carbon/io.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace carbon {

void write_depth_dose_csv(const std::filesystem::path& path,
                          const TransportConfig& config,
                          const TransportResult& result) {
    if (result.deposited_energy_MeV.size() != config.number_of_bins()) {
        throw std::invalid_argument("Transport result bin count does not match configuration");
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create output file: " + path.string());
    }

    const auto histories = static_cast<double>(config.number_of_histories);
    const auto maximum = *std::max_element(result.deposited_energy_MeV.begin(),
                                           result.deposited_energy_MeV.end());
    constexpr double MeV_to_joule = 1.602176634e-13;
    const auto bin_mass_kg = config.scorer_area_mm2 * config.depth_bin_width_mm *
                             config.water_density_g_per_cm3 * 1.0e-6;

    output << "depth_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary,relative_dose\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < result.deposited_energy_MeV.size(); ++bin) {
        const auto energy_per_primary = result.deposited_energy_MeV[bin] / histories;
        const auto dose_per_primary = energy_per_primary * MeV_to_joule / bin_mass_kg;
        const auto relative_dose = maximum > 0.0 ? result.deposited_energy_MeV[bin] / maximum : 0.0;
        const auto depth_center_mm = (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm << ',' << energy_per_primary << ',' << dose_per_primary << ','
               << relative_dose << '\n';
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
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create fragment species output file: " + path.string());
    }
    const auto histories = static_cast<double>(config.number_of_histories);
    const auto maximum = *std::max_element(result.deposited_energy_MeV.begin(),
                                           result.deposited_energy_MeV.end());
    output << "depth_mm,total_MeV_per_primary,primary_c12_MeV_per_primary,"
              "secondary_carbon_MeV_per_primary,boron_MeV_per_primary,"
              "beryllium_MeV_per_primary,lithium_MeV_per_primary,"
              "helium_MeV_per_primary,proton_MeV_per_primary,"
              "other_MeV_per_primary,relative_total\n";
    output << std::setprecision(12);
    for (std::size_t bin = 0; bin < bins; ++bin) {
        const auto depth_center_mm =
            (static_cast<double>(bin) + 0.5) * config.depth_bin_width_mm;
        output << depth_center_mm;
        for (const auto* column : columns) {
            output << ',' << (*column)[bin] / histories;
        }
        const auto relative_total =
            maximum > 0.0 ? result.deposited_energy_MeV[bin] / maximum : 0.0;
        output << ',' << relative_total << '\n';
    }
}

}  // namespace carbon
