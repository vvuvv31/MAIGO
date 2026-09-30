#pragma once
#include "carbon/urban_mcs_package.hpp"

#include <filesystem>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace carbon {

[[nodiscard]] double ion_effective_charge(int atomic_number, double energy_MeVu);
[[nodiscard]] double stopping_power_scale_from_reference_ion(
    int atomic_number, int reference_atomic_number, double energy_MeVu);

class StoppingPowerTable;

[[nodiscard]] std::vector<float> load_ion_species_stopping_power_lut(
    const std::filesystem::path& path,
    std::size_t table_size,
    float scale = 1.0F);

class StoppingPowerTable {
public:
    StoppingPowerTable(std::vector<double> energies_MeVu,
                       std::vector<double> stopping_powers_MeV_per_mm);

    static StoppingPowerTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    // CSDA range in mm from zero energy to energy_MeVu. The table abscissa is
    // kinetic energy per nucleon (MeV/u); values are total-ion stopping power
    // (MeV/mm), so mass_number is required for the A*dE_u/S(E_u) integral.
    [[nodiscard]] double csda_range_mm(double energy_MeVu, int mass_number) const;
    // Invert the CSDA range after a physical path length in mm. Energy is
    // returned in MeV/u and is clamped to zero when the path exhausts range.
    [[nodiscard]] double csda_energy_after_distance_MeVu(
        double initial_energy_MeVu, double distance_mm, int mass_number) const;
    [[nodiscard]] const std::vector<double>& cumulative_ranges_mm() const noexcept;
    [[nodiscard]] double minimum_energy_MeVu() const noexcept;
    [[nodiscard]] double maximum_energy_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> stopping_powers_MeV_per_mm_;
    std::vector<double> cumulative_ranges_mm_;
};

// Restricted loss-range table for one ion/material/cuts couple
// (theRangeTableForLoss semantics): total kinetic energy -> restricted-loss
// range -> restricted dE/dx. Extracted from Geant4 via CarbonLossRangeNtuple.
// Columns: energy_MeVu, energy_total_MeV, loss_range_mm, csda_range_mm,
// restricted_dedx_MeV_per_mm, inverse_residual_MeV. Device lookup uses
// E_total and R (both strictly increasing); CSDA + residual are metadata.
class UrbanLossRangeTable {
public:
    UrbanLossRangeTable(std::vector<double> e_total_mev,
                        std::vector<double> range_mm,
                        std::vector<double> dedx_mev_per_mm,
                        double max_inverse_residual_mev,
                        double zeff = std::numeric_limits<double>::quiet_NaN(),
                        double radlen_mm = std::numeric_limits<double>::quiet_NaN(),
                        double density_g_per_cm3 =
                            std::numeric_limits<double>::quiet_NaN(),
                        std::string oracle_status = {},
                        std::string physics_list = {},
                        std::string particle = {},
                        std::string material = {},
                        std::string range_source = {},
                        std::string inverse_source = {},
                        std::string dedx_source = {},
                        double production_cut_mm =
                            std::numeric_limits<double>::quiet_NaN());

    static UrbanLossRangeTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<double>& energies_total_mev() const noexcept;
    [[nodiscard]] const std::vector<double>& ranges_mm() const noexcept;
    [[nodiscard]] const std::vector<double>& dedx_values() const noexcept;
    [[nodiscard]] double max_inverse_residual_mev() const noexcept;
    // Optional `# key value` header metadata (NaN when absent, e.g. Cu CSV).
    [[nodiscard]] double zeff() const noexcept;
    [[nodiscard]] double radlen_mm() const noexcept;
    [[nodiscard]] double density_g_per_cm3() const noexcept;
    [[nodiscard]] const std::string& oracle_status() const noexcept;
    [[nodiscard]] const std::string& physics_list() const noexcept;
    [[nodiscard]] const std::string& particle() const noexcept;
    [[nodiscard]] const std::string& material() const noexcept;
    [[nodiscard]] const std::string& range_source() const noexcept;
    [[nodiscard]] const std::string& inverse_source() const noexcept;
    [[nodiscard]] const std::string& dedx_source() const noexcept;
    [[nodiscard]] double production_cut_mm() const noexcept;
    [[nodiscard]] bool is_active_reference(const std::string& expected_particle,
        const std::string& expected_loss_process, const std::string& expected_material,
        double expected_cut_mm) const noexcept;
    [[nodiscard]] bool is_active_c12_reference(const std::string& expected_material,
                                               double expected_cut_mm) const noexcept;

private:
    std::vector<double> e_total_mev_;
    std::vector<double> range_mm_;
    std::vector<double> dedx_mev_per_mm_;
    double max_inverse_residual_mev_{0.0};
    double zeff_{std::numeric_limits<double>::quiet_NaN()};
    double radlen_mm_{std::numeric_limits<double>::quiet_NaN()};
    double density_g_per_cm3_{std::numeric_limits<double>::quiet_NaN()};
    std::string oracle_status_;
    std::string physics_list_;
    std::string particle_;
    std::string material_;
    std::string range_source_;
    std::string inverse_source_;
    std::string dedx_source_;
    double production_cut_mm_{std::numeric_limits<double>::quiet_NaN()};
};

// Flat-array CSDA helpers for future SYCL kernels. Arrays use the same contract
// as StoppingPowerTable: energy is MeV/u, stopping power is total-ion MeV/mm,
// cumulative range is A=1 mm, and mass_number supplies the overall A factor.
class IonStoppingPowerTables {
public:
    static constexpr std::size_t mass_stride = 32;
    static constexpr std::size_t atomic_number_slots = 10;
    static constexpr std::size_t species_slots =
        mass_stride * atomic_number_slots;

    static IonStoppingPowerTables from_csv(
        const std::filesystem::path& path,
        const StoppingPowerTable& carbon_stopping_power,
        bool normalize_to_file_carbon = false);

    [[nodiscard]] const std::vector<float>& ratios_to_carbon() const noexcept;
    [[nodiscard]] const std::vector<float>& delta_electron_fractions() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& species_present() const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;

private:
    std::size_t energy_grid_size_{};
    std::vector<float> ratios_to_carbon_;
    std::vector<float> delta_electron_fractions_;
    std::vector<std::uint8_t> species_present_;
};

}  // namespace carbon
