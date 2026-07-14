#pragma once

#include <filesystem>
#include <vector>

namespace carbon {

class StoppingPowerTable {
public:
    StoppingPowerTable(std::vector<double> energies_MeVu,
                       std::vector<double> stopping_powers_MeV_per_mm);

    static StoppingPowerTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] double minimum_energy_MeVu() const noexcept;
    [[nodiscard]] double maximum_energy_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> stopping_powers_MeV_per_mm_;
};

}  // namespace carbon

