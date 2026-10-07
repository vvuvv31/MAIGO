#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace carbon {

class CrossSectionTable {
public:
    CrossSectionTable() = default;
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

// Consumes optical depth over a segment of length step_mm in a medium with macro_xs (mm^-1).
// If collision occurs within step_mm: returns true, clamps step_mm to collision distance, resets tau_active.
// Otherwise: returns false, subtracts macro_xs * step_mm from tau_remaining, keeps tau_active true.
inline bool consume_schneider_optical_depth_segment(
    float& tau_remaining,
    bool& tau_active,
    float& step_mm,
    float macro_xs,
    float rng_u) noexcept {
    if (!tau_active) {
        tau_remaining = -std::log(std::max(rng_u, 1.0e-12F));
        tau_active = true;
    }
    const float delta_tau = macro_xs * step_mm;
    if (macro_xs > 0.0F && delta_tau >= tau_remaining) {
        const float collision_s = tau_remaining / macro_xs;
        step_mm = std::min(step_mm, collision_s);
        tau_remaining = 0.0F;
        tau_active = false;
        return true;
    }
    tau_remaining -= delta_tau;
    return false;
}

class IonCrossSectionTables {
public:
    static constexpr std::size_t mass_stride = 32;
    static constexpr std::size_t atomic_number_slots = 10;
    static constexpr std::size_t species_slots =
        mass_stride * atomic_number_slots;

    static IonCrossSectionTables from_csv(
        const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& values() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& species_present()
        const noexcept;
    [[nodiscard]] std::size_t energy_grid_size() const noexcept;
    [[nodiscard]] float minimum_energy_MeVu() const noexcept;
    [[nodiscard]] float energy_step_MeVu() const noexcept;

private:
    std::size_t energy_grid_size_{};
    float minimum_energy_MeVu_{};
    float energy_step_MeVu_{};
    std::vector<float> values_;
    std::vector<std::uint8_t> species_present_;
};

}  // namespace carbon
