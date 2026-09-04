#pragma once

#include "carbon/schneider_rate_table.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include <array>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

struct SchneiderTargetSample {
    int target_z{0};
    std::size_t target_index{0};
    float target_probability{0.0F};
};

struct SchneiderTargetSamplerDeviceTable {
    const float* cdf_table{nullptr};
    const float* total_mass_rates{nullptr};
    // v3 only: raw partials [section][target][energy] + per-target domain.
    // The v3 device path never uses cdf/total_mass_rates (mask applies at
    // query energy, so node-interpolated totals would leak across domain
    // boundaries); it computes masked partials and their sum on the fly.
    const float* partial_rates{nullptr};
    const float* domain_emin{nullptr};
    const float* domain_emax{nullptr};
    const unsigned char* domain_has{nullptr};
    std::uint32_t rate_version{1};
    std::size_t num_sections{25};
    std::size_t num_energies{860};
    std::size_t num_targets{13};
    float energy_min_MeV_per_u{0.5F};
    float energy_step_MeV_per_u{0.5F};
    float inverse_energy_step{2.0F};
};

struct SchneiderTargetDiagnostics {
    std::uint64_t interactions_by_section[25]{};
    std::uint64_t interactions_by_target_z[32]{}; // indexed by Z <= 31
    std::uint64_t interactions_by_energy_bin[450]{}; // 1 MeV/u bins up to 450
    std::uint64_t lookup_success_count{0};
    std::uint64_t lookup_failure_count{0};
    std::uint64_t replayed_events_count{0};
    std::uint64_t unsupported_target_count{0};
};

class SchneiderTargetSampler {
public:
    SchneiderTargetSampler() = default;
    explicit SchneiderTargetSampler(const SchneiderRateTable& rate_table);

    [[nodiscard]] SchneiderTargetSample sample_target(
        std::size_t section_id, float energy_MeV_per_u, float u01) const;

    [[nodiscard]] std::array<float, 13> target_probabilities(
        std::size_t section_id, float energy_MeV_per_u) const;

    [[nodiscard]] float total_mass_rate(std::size_t section_id, float energy_MeV_per_u) const;

    [[nodiscard]] const std::vector<float>& cdf_table() const noexcept { return cdf_table_; }
    [[nodiscard]] const std::vector<float>& total_mass_rates() const noexcept { return total_mass_rates_; }
    [[nodiscard]] const std::vector<float>& partial_rates() const noexcept { return partial_rates_; }
    [[nodiscard]] std::uint32_t rate_version() const noexcept { return rate_version_; }
    [[nodiscard]] const std::vector<float>& domain_emin() const noexcept { return domain_emin_; }
    [[nodiscard]] const std::vector<float>& domain_emax() const noexcept { return domain_emax_; }
    [[nodiscard]] const std::vector<unsigned char>& domain_has() const noexcept { return domain_has_; }

    [[nodiscard]] SchneiderTargetSamplerDeviceTable device_table() const noexcept {
        SchneiderTargetSamplerDeviceTable t;
        t.cdf_table = cdf_table_.data();
        t.total_mass_rates = total_mass_rates_.data();
        t.partial_rates = partial_rates_.data();
        t.domain_emin = domain_emin_.data();
        t.domain_emax = domain_emax_.data();
        t.domain_has = domain_has_.data();
        t.rate_version = rate_version_;
        t.num_sections = num_sections_;
        t.num_energies = num_energies_;
        t.num_targets = num_targets_;
        t.energy_min_MeV_per_u = static_cast<float>(energy_min_MeV_per_u_);
        t.energy_step_MeV_per_u = static_cast<float>(energy_step_MeV_per_u_);
        t.inverse_energy_step = static_cast<float>(1.0 / energy_step_MeV_per_u_);
        return t;
    }

private:
    std::uint32_t rate_version_{1};
    std::size_t num_sections_{25};
    std::size_t num_energies_{860};
    std::size_t num_targets_{13};
    double energy_min_MeV_per_u_{0.5};
    double energy_step_MeV_per_u_{0.5};
    std::vector<float> cdf_table_;
    std::vector<float> total_mass_rates_;
    // v3: raw partials [section][target][energy] + per-target domain.
    std::vector<float> partial_rates_;
    std::vector<float> domain_emin_;
    std::vector<float> domain_emax_;
    std::vector<unsigned char> domain_has_;
};

// Out-of-grid policy is data-driven (same contract as the secondary rate
// device functions): the v1 grid ([0.5,430]/860) keeps its legacy endpoint
// clamp for bitwise reproducibility of frozen results; any other grid is
// strict (out-of-grid -> invalid target 0, no clamp, no extrapolation).
inline bool schneider_grid_is_legacy_v1(float energy_min_MeV_per_u,
                                        std::size_t num_energies) noexcept {
    const float d = energy_min_MeV_per_u - 0.5F;
    return num_energies == 860 && (d < 1.0e-6F && d > -1.0e-6F);
}

// Device-safe target sampling helper
inline int sample_schneider_target_device(
    const SchneiderTargetSamplerDeviceTable& table,
    std::size_t section_id,
    float energy_MeV_per_u,
    float u01) noexcept
{
    if (section_id >= table.num_sections || table.cdf_table == nullptr ||
        table.num_energies < 2) {
        return 0;
    }
    const float e_max = table.energy_min_MeV_per_u +
                        static_cast<float>(table.num_energies - 1) / table.inverse_energy_step;
    float e = energy_MeV_per_u;
    if (schneider_grid_is_legacy_v1(table.energy_min_MeV_per_u, table.num_energies)) {
        e = e < table.energy_min_MeV_per_u ? table.energy_min_MeV_per_u
                                           : (e > e_max ? e_max : e);
    } else if (!(e >= table.energy_min_MeV_per_u) || !(e <= e_max)) {
        return 0;
    }
    const float node_flt = (e - table.energy_min_MeV_per_u) * table.inverse_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(table.num_energies) - 1) {
        node_idx = static_cast<int>(table.num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);

    const std::size_t base0 = section_id * (table.num_energies * table.num_targets) +
                              static_cast<std::size_t>(node_idx) * table.num_targets;
    const std::size_t base1 = base0 + table.num_targets;
    const float u = u01 < 0.0F ? 0.0F : (u01 >= 1.0F ? 0.9999999F : u01);

    // All-zero CDF row (every partial zeroed at this energy): no valid
    // target exists; return invalid instead of falling through to the last
    // element with zero weight.
    const float cdf_last =
        table.cdf_table[base0 + table.num_targets - 1] +
        frac * (table.cdf_table[base1 + table.num_targets - 1] -
                table.cdf_table[base0 + table.num_targets - 1]);
    if (!(cdf_last > 0.0F)) {
        return 0;
    }
    for (std::size_t k = 0; k < table.num_targets; ++k) {
        const float cdf_k = table.cdf_table[base0 + k] + frac * (table.cdf_table[base1 + k] - table.cdf_table[base0 + k]);
        if (u < cdf_k) {
            return kSchneiderCanonicalZ[k];
        }
    }
    return kSchneiderCanonicalZ[table.num_targets - 1];
}

// Device-safe total mass rate lookup with linear interpolation between energy nodes.
// Same out-of-grid contract as the sampler above: legacy v1 grid clamps,
// other grids return exactly 0 outside [emin, emax].
inline float schneider_total_mass_rate_device(
    const SchneiderTargetSamplerDeviceTable& table,
    std::size_t section_id,
    float energy_MeV_per_u) noexcept
{
    if (section_id >= table.num_sections || table.total_mass_rates == nullptr ||
        table.num_energies < 2) {
        return 0.0F;
    }
    const float e_max = table.energy_min_MeV_per_u +
                        static_cast<float>(table.num_energies - 1) / table.inverse_energy_step;
    float e = energy_MeV_per_u;
    if (schneider_grid_is_legacy_v1(table.energy_min_MeV_per_u, table.num_energies)) {
        e = e < table.energy_min_MeV_per_u ? table.energy_min_MeV_per_u
                                           : (e > e_max ? e_max : e);
    } else if (!(e >= table.energy_min_MeV_per_u) || !(e <= e_max)) {
        return 0.0F;
    }
    const float node_flt = (e - table.energy_min_MeV_per_u) * table.inverse_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(table.num_energies) - 1) {
        node_idx = static_cast<int>(table.num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);
    const std::size_t base = section_id * table.num_energies + static_cast<std::size_t>(node_idx);
    return table.total_mass_rates[base] + frac * (table.total_mass_rates[base + 1] - table.total_mass_rates[base]);
}

// ---- v3 masked primary path (rate_version 3 only; v1 never calls these) ----

struct SchneiderMaskedRates {
    float partials[13];
    float total{0.0F};
};

// v3 masked partial computation for the C12 primary rate. Per-target domain
// applied BEFORE interpolation: unsupported channels, or queries outside a
// channel's [emin, emax], contribute EXACTLY 0. total is the masked sum, so
// the hazard can never fire where no channel has support. Strict zero
// outside the global grid; no clamp, no extrapolation.
inline SchneiderMaskedRates schneider_masked_rates_device(
    const SchneiderTargetSamplerDeviceTable& table,
    std::size_t section_id,
    float energy_MeV_per_u) noexcept {
    SchneiderMaskedRates out;
    for (std::size_t k = 0; k < 13; ++k) {
        out.partials[k] = 0.0F;
    }
    if (section_id >= table.num_sections || table.partial_rates == nullptr ||
        table.domain_emin == nullptr || table.domain_emax == nullptr ||
        table.domain_has == nullptr || table.num_energies < 2) {
        return out;
    }
    const float e_max = table.energy_min_MeV_per_u +
                        static_cast<float>(table.num_energies - 1) / table.inverse_energy_step;
    const float e = energy_MeV_per_u;
    if (!(e >= table.energy_min_MeV_per_u) || !(e <= e_max)) {
        return out;
    }
    float node_flt = (e - table.energy_min_MeV_per_u) * table.inverse_energy_step;
    auto node_idx = static_cast<int>(node_flt);
    if (node_idx < 0) node_idx = 0;
    if (node_idx >= static_cast<int>(table.num_energies) - 1) {
        node_idx = static_cast<int>(table.num_energies) - 2;
    }
    const float frac = node_flt - static_cast<float>(node_idx);
    float sum = 0.0F;
    for (std::size_t k = 0; k < 13; ++k) {
        if (table.domain_has[k] == 0 || e < table.domain_emin[k] ||
            e > table.domain_emax[k]) {
            out.partials[k] = 0.0F;
            continue;
        }
        const std::size_t base =
            section_id * (13 * table.num_energies) + k * table.num_energies +
            static_cast<std::size_t>(node_idx);
        float v = table.partial_rates[base] +
                  frac * (table.partial_rates[base + 1] - table.partial_rates[base]);
        if (v < 0.0F) v = 0.0F;
        out.partials[k] = v;
        sum += v;
    }
    // Same hazard/sampler unity as the secondary path (see above).
    if (!(sum > 1.0e-12F)) {
        for (std::size_t k = 0; k < 13; ++k) {
            out.partials[k] = 0.0F;
        }
        sum = 0.0F;
    }
    out.total = sum;
    return out;
}

// Categorical draw from v3 masked primary partials. Zero-weight channels are
// never selected; an all-zero row returns invalid target 0.
inline int sample_masked_schneider_target_device(const float (&partials)[13],
                                                 float u01) noexcept {
    float sum = 0.0F;
    for (std::size_t k = 0; k < 13; ++k) {
        sum += partials[k];
    }
    if (sum <= 1.0e-12F) {
        return 0;
    }
    const float u = (u01 < 0.0F ? 0.0F : (u01 >= 1.0F ? 0.9999999F : u01)) * sum;
    float cum = 0.0F;
    for (std::size_t k = 0; k < 13; ++k) {
        cum += partials[k];
        if (u < cum || k == 12) {
            return kSchneiderCanonicalZ[k];
        }
    }
    return kSchneiderCanonicalZ[12];
}

} // namespace carbon
