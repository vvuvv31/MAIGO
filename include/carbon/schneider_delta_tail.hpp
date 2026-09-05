#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace carbon {

// TOPAS/Geant4-derived transverse delta-electron tail for primary C-12 in
// Schneider section 0.  The table is intentionally narrow in scope: it is not
// an electron transport model for water, other sections, or fragments.
class SchneiderDeltaTailTable {
public:
    static SchneiderDeltaTailTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& energies_MeV_per_u() const noexcept {
        return energies_MeV_per_u_;
    }
    [[nodiscard]] const std::vector<float>& moved_fractions() const noexcept {
        return moved_fractions_;
    }
    [[nodiscard]] const std::vector<float>& quantiles() const noexcept {
        return quantiles_;
    }
    [[nodiscard]] const std::vector<float>& radii_mm() const noexcept {
        return radii_mm_;
    }
    [[nodiscard]] std::size_t energy_count() const noexcept {
        return energies_MeV_per_u_.size();
    }
    [[nodiscard]] std::size_t quantile_count() const noexcept {
        return quantiles_.size();
    }

private:
    std::vector<float> energies_MeV_per_u_;
    std::vector<float> moved_fractions_;
    std::vector<float> quantiles_;
    // Row-major [energy][quantile].
    std::vector<float> radii_mm_;
};

inline void schneider_delta_tail_lookup_device(
    const float energy_MeV_per_u,
    const float u,
    const float* energies,
    const float* moved_fractions,
    const float* radii_mm,
    const std::size_t energy_count,
    const std::size_t quantile_count,
    float& moved_fraction,
    float& radius_mm) noexcept {
    moved_fraction = 0.0F;
    radius_mm = 0.0F;
    if (energies == nullptr || moved_fractions == nullptr || radii_mm == nullptr ||
        energy_count == 0 || quantile_count < 2) {
        return;
    }
    if (energy_MeV_per_u < energies[0] ||
        energy_MeV_per_u > energies[energy_count - 1]) {
        return;
    }
    std::size_t e0 = 0;
    float ef = 0.0F;
    if (energy_count > 1) {
        if (energy_MeV_per_u == energies[energy_count - 1]) {
            e0 = energy_count - 2;
            ef = 1.0F;
        } else if (energy_MeV_per_u > energies[0]) {
            while (e0 + 1 < energy_count - 1 &&
                   energy_MeV_per_u > energies[e0 + 1]) {
                ++e0;
            }
            const auto de = energies[e0 + 1] - energies[e0];
            ef = de > 0.0F ? (energy_MeV_per_u - energies[e0]) / de : 0.0F;
        }
    }
    const auto e1 = energy_count > 1 ? e0 + 1 : e0;
    moved_fraction = moved_fractions[e0] +
                     ef * (moved_fractions[e1] - moved_fractions[e0]);

    auto uq = u;
    if (uq < 0.0F) uq = 0.0F;
    if (uq > 1.0F) uq = 1.0F;
    const auto qf = uq * static_cast<float>(quantile_count - 1);
    auto q0 = static_cast<std::size_t>(qf);
    if (q0 >= quantile_count - 1) q0 = quantile_count - 2;
    const auto qfrac = qf - static_cast<float>(q0);
    const auto r00 = radii_mm[e0 * quantile_count + q0];
    const auto r01 = radii_mm[e0 * quantile_count + q0 + 1];
    const auto r10 = radii_mm[e1 * quantile_count + q0];
    const auto r11 = radii_mm[e1 * quantile_count + q0 + 1];
    const auto r0 = r00 + qfrac * (r01 - r00);
    const auto r1 = r10 + qfrac * (r11 - r10);
    radius_mm = r0 + ef * (r1 - r0);
}

// TOPAS/Geant4-derived forward delta-electron kernel for primary C-12 in
// Schneider section 0. Supplement to the transverse tail above: a fraction
// forward_fraction of the local deposit is carried forward along the beam
// with an exponential range of mean lambda_mm. Same narrow scope.
class SchneiderLongitudinalTable {
public:
    static SchneiderLongitudinalTable from_csv(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<float>& energies_MeV_per_u() const noexcept {
        return energies_MeV_per_u_;
    }
    [[nodiscard]] const std::vector<float>& forward_fractions() const noexcept {
        return forward_fractions_;
    }
    [[nodiscard]] const std::vector<float>& lambdas_mm() const noexcept {
        return lambdas_mm_;
    }
    [[nodiscard]] std::size_t energy_count() const noexcept {
        return energies_MeV_per_u_.size();
    }

private:
    std::vector<float> energies_MeV_per_u_;
    std::vector<float> forward_fractions_;
    std::vector<float> lambdas_mm_;
};

inline void schneider_longitudinal_lookup_device(
    const float energy_MeV_per_u,
    const float* energies,
    const float* forward_fractions,
    const float* lambdas_mm,
    const std::size_t energy_count,
    float& forward_fraction,
    float& lambda_mm) noexcept {
    forward_fraction = 0.0F;
    lambda_mm = 0.0F;
    if (energies == nullptr || forward_fractions == nullptr ||
        lambdas_mm == nullptr || energy_count == 0) {
        return;
    }
    // Nearest-edge clamp outside the calibrated [150, 225] MeV/u interval
    // (unlike the transverse table, which returns zero out of range). Slab
    // fits show a weak energy dependence (A: 6.7% at 150, 7.4% at 200, 7.5%
    // at 225), so edge values err by <10% on out-of-range spots, while
    // returning zero would drop up to 40% of a clinical beam (all-or-nothing
    // error). Beams routinely span 110-240 MeV/u.
    std::size_t e0 = 0;
    float ef = 0.0F;
    if (energy_count > 1) {
        if (energy_MeV_per_u <= energies[0]) {
            e0 = 0;
            ef = 0.0F;
        } else if (energy_MeV_per_u >= energies[energy_count - 1]) {
            e0 = energy_count - 2;
            ef = 1.0F;
        } else {
            while (e0 + 1 < energy_count - 1 &&
                   energy_MeV_per_u > energies[e0 + 1]) {
                ++e0;
            }
            const auto de = energies[e0 + 1] - energies[e0];
            ef = de > 0.0F ? (energy_MeV_per_u - energies[e0]) / de : 0.0F;
        }
    }
    const auto e1 = energy_count > 1 ? e0 + 1 : e0;
    forward_fraction = forward_fractions[e0] +
                       ef * (forward_fractions[e1] - forward_fractions[e0]);
    lambda_mm = lambdas_mm[e0] + ef * (lambdas_mm[e1] - lambdas_mm[e0]);
}

}  // namespace carbon
