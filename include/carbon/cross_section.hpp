#pragma once

#include <filesystem>
#include <vector>

namespace carbon {

class CrossSectionTable {
public:
    CrossSectionTable(std::vector<double> energies_MeVu,
                      std::vector<double> macroscopic_cross_sections_per_mm);

    static CrossSectionTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] double interpolate(double energy_MeVu) const noexcept;
    [[nodiscard]] const std::vector<double>& energies() const noexcept;
    [[nodiscard]] const std::vector<double>& values() const noexcept;

private:
    std::vector<double> energies_MeVu_;
    std::vector<double> macroscopic_cross_sections_per_mm_;
};

}  // namespace carbon
