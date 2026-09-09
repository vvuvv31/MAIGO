#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace carbon {

template <typename Scalar>
inline Scalar scale_energy_loss_ratio_preserving_mean(
    const Scalar ratio, const Scalar scale) noexcept {
    return std::max(Scalar{0}, Scalar{1} + scale * (ratio - Scalar{1}));
}

template <typename Scalar>
inline Scalar sample_energy_loss_ratio_from_grid(
    const Scalar* energies_MeVu,
    const std::size_t energy_count,
    const Scalar* areal_densities_g_per_cm2,
    const std::size_t density_count,
    const Scalar* probabilities,
    const std::size_t probability_count,
    const Scalar* loss_ratio_quantiles,
    const Scalar energy_MeVu,
    const Scalar areal_density_g_per_cm2,
    Scalar uniform) noexcept {
    const auto bracket = [](const Scalar* grid, const std::size_t count,
                            const Scalar value, std::size_t& lower,
                            Scalar& fraction) {
        if (value <= grid[0]) {
            lower = 0;
            fraction = Scalar{0};
            return;
        }
        if (value >= grid[count - 1]) {
            lower = count - 2;
            fraction = Scalar{1};
            return;
        }
        lower = 0;
        while (lower + 1 < count && value > grid[lower + 1]) ++lower;
        fraction = (value - grid[lower]) /
                   (grid[lower + 1] - grid[lower]);
    };

    std::size_t energy_index = 0;
    std::size_t density_index = 0;
    Scalar energy_fraction = 0;
    Scalar density_fraction = 0;
    bracket(energies_MeVu, energy_count, energy_MeVu,
            energy_index, energy_fraction);
    bracket(areal_densities_g_per_cm2, density_count,
            areal_density_g_per_cm2, density_index, density_fraction);
    uniform = uniform < Scalar{0} ? Scalar{0}
                                 : (uniform > Scalar{1} ? Scalar{1} : uniform);
    std::size_t probability_upper = 0;
    while (probability_upper + 1 < probability_count &&
           uniform > probabilities[probability_upper]) {
        ++probability_upper;
    }
    const auto probability_lower =
        probability_upper == 0 ? std::size_t{0} : probability_upper - 1;
    const auto probability_fraction =
        probability_lower == probability_upper
            ? Scalar{0}
            : (uniform - probabilities[probability_lower]) /
                  (probabilities[probability_upper] -
                   probabilities[probability_lower]);
    const auto quantile = [&](const std::size_t energy,
                              const std::size_t density) {
        const auto base =
            (energy * density_count + density) * probability_count;
        return loss_ratio_quantiles[base + probability_lower] +
               probability_fraction *
                   (loss_ratio_quantiles[base + probability_upper] -
                    loss_ratio_quantiles[base + probability_lower]);
    };
    const auto lower_energy = quantile(energy_index, density_index) +
        density_fraction *
            (quantile(energy_index, density_index + 1) -
             quantile(energy_index, density_index));
    const auto upper_energy = quantile(energy_index + 1, density_index) +
        density_fraction *
            (quantile(energy_index + 1, density_index + 1) -
             quantile(energy_index + 1, density_index));
    return lower_energy + energy_fraction * (upper_energy - lower_energy);
}

// Empirical inverse-CDFs for a specific ion and material. Each distribution
// describes sampled energy loss divided by its mean at one energy/thickness
// grid point, so transport can retain the stopping-power table as the source
// of the continuous-loss mean.
class EnergyLossFluctuationTable {
public:
    static EnergyLossFluctuationTable from_csv(
        const std::filesystem::path& path, bool mean_loss_fraction_axis = false);

    [[nodiscard]] const std::vector<double>& second_axis_values() const noexcept {
        return areal_densities_g_per_cm2_;
    }

    [[nodiscard]] double sample_loss_ratio(
        double energy_MeVu,
        double areal_density_g_per_cm2,
        double uniform) const noexcept;

    [[nodiscard]] int projectile_atomic_number() const noexcept;
    [[nodiscard]] int projectile_mass_number() const noexcept;
    [[nodiscard]] const std::string& material_name() const noexcept;
    [[nodiscard]] const std::vector<double>& energies_MeVu() const noexcept;
    [[nodiscard]] const std::vector<double>& areal_densities_g_per_cm2()
        const noexcept;
    [[nodiscard]] const std::vector<double>& probabilities() const noexcept;
    [[nodiscard]] const std::vector<double>& loss_ratio_quantiles()
        const noexcept;

private:
    int projectile_atomic_number_{};
    int projectile_mass_number_{};
    std::string material_name_;
    std::vector<double> energies_MeVu_;
    std::vector<double> areal_densities_g_per_cm2_;
    std::vector<double> probabilities_;
    // energy-major, then areal-density, then quantile.
    std::vector<double> loss_ratio_quantiles_;
};

}  // namespace carbon
