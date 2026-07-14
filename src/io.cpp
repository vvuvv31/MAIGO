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
    std::ofstream output(path);
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

}  // namespace carbon

