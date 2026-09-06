#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace carbon {

// Actual float density in the frozen HU=-1000 extraction CCTG, not rounded 0.01132.
inline constexpr double kLongitudinalReferenceDensityGPerCm3 = 0.01131606474518776;

// Host preflight: every voxel must be the SAME frozen probe density/section.
// No tolerance permitting hidden density steps; never authorize patient CT.
inline double longitudinal_probe_density(const std::vector<float>& density,
                                         const std::vector<std::uint8_t>& sections) {
    if (density.empty() || density.size() != sections.size())
        throw std::invalid_argument("Longitudinal homogeneous diagnostic: invalid grid");
    const float rho = density.front();
    if (rho != static_cast<float>(kLongitudinalReferenceDensityGPerCm3) &&
        rho != 0.03932345286011696F && rho != 0.06621015816926956F)
        throw std::invalid_argument("Longitudinal homogeneous diagnostic: unsupported density");
    for (std::size_t i=0; i<density.size(); ++i)
        if (sections[i] != 0 || density[i] != rho)
            throw std::invalid_argument("Longitudinal homogeneous diagnostic: heterogeneous grid");
    return rho;
}

// Narrow, explicit interface experiment; rejects all patient material grids.
inline void reject_unvalidated_longitudinal_heterogeneity(const std::vector<float>& density,
                                                        const std::vector<std::uint8_t>& sections) {
    if(density.empty() || density.size()!=sections.size())
        throw std::invalid_argument("Longitudinal candidate: invalid grid");
    for(std::size_t i=1;i<density.size();++i)
        if(density[i]!=density[0] || sections[i]!=sections[0])
            throw std::invalid_argument("Unvalidated longitudinal candidate rejects heterogeneous grids; explicit isolated interface diagnostic required");
}

inline void validate_longitudinal_interface_grid(const std::vector<float>& density,
                                                const std::vector<std::uint8_t>& sections) {
    if (density.empty() || density.size()!=sections.size())
        throw std::invalid_argument("Longitudinal interface diagnostic: invalid grid");
    for(std::size_t i=0;i<density.size();++i) {
        const bool air=sections[i]==0 && density[i]==static_cast<float>(kLongitudinalReferenceDensityGPerCm3);
        const bool tissue=sections[i]==8 && density[i]==1.0787997245788574F;
        if(!air && !tissue)
            throw std::invalid_argument("Longitudinal interface diagnostic: only HU -1000/100 probes allowed");
    }
}

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

// Historical slab-fitted candidate, NOT a validated electron transport model.
// Only pinned input data may be loaded; diagnostic scope is enforced by config.
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

inline bool schneider_longitudinal_lookup_device(
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
        return false;
    }
    // No extrapolation, including NaN/Inf. A rejected query retains local
    // energy and is counted by transport; it does not lose beam energy.
    if (!(energy_MeV_per_u >= energies[0] &&
          energy_MeV_per_u <= energies[energy_count - 1])) return false;
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
    return true;
}

}  // namespace carbon
