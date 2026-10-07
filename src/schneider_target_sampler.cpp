#include "carbon/schneider_target_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace carbon {

SchneiderTargetSampler::SchneiderTargetSampler(const SchneiderRateTable& rate_table)
    : num_sections_(kSchneiderNumSections),
      num_energies_(rate_table.num_energies()),
      num_targets_(kSchneiderNumTargets),
      energy_min_MeV_per_u_(rate_table.energy_min_mevu()),
      energy_step_MeV_per_u_(rate_table.energy_step_mevu())
{
    if (rate_table.binary_version() != 3 || !rate_table.has_channel_domains()) {
        throw std::invalid_argument("Schneider target sampler requires masked v3 rates");
    }
    partial_rates_.resize(num_sections_ * num_targets_ * num_energies_, 0.0F);
    domain_emin_.resize(num_targets_);
    domain_emax_.resize(num_targets_);
    domain_has_.resize(num_targets_);
    for (std::size_t k = 0; k < num_targets_; ++k) {
        const auto& d = rate_table.channel_domain(k);
        domain_emin_[k] = static_cast<float>(d.energy_min_mevu);
        domain_emax_[k] = static_cast<float>(d.energy_max_mevu);
        domain_has_[k] = d.has_support;
    }
    // Keep raw partials: masks apply at query energy, before interpolation.
    for (std::size_t s = 0; s < num_sections_; ++s) {
        for (std::size_t k = 0; k < num_targets_; ++k) {
            for (std::size_t e = 0; e < num_energies_; ++e) {
                const double raw = rate_table.mass_partial_rate(s, k, e);
                partial_rates_[s * (num_targets_ * num_energies_) + k * num_energies_ + e] =
                    static_cast<float>((std::isfinite(raw) && raw > 0.0) ? raw : 0.0);
            }
        }
    }
}

SchneiderTargetSample SchneiderTargetSampler::sample_target(
    std::size_t section_id, float energy_MeV_per_u, float u01) const
{
    if (section_id >= num_sections_) {
        throw std::out_of_range("Schneider section ID out of range: " + std::to_string(section_id));
    }

    const auto probs = target_probabilities(section_id, energy_MeV_per_u);
    const float u = std::clamp(u01, 0.0F, 0.9999999F);

    float cumsum = 0.0F;
    for (std::size_t k = 0; k < num_targets_; ++k) {
        cumsum += probs[k];
        if (u < cumsum || k == num_targets_ - 1) {
            return SchneiderTargetSample{
                kSchneiderCanonicalZ[k],
                k,
                probs[k]
            };
        }
    }
    return SchneiderTargetSample{kSchneiderCanonicalZ.back(), num_targets_ - 1, probs.back()};
}

std::array<float, 13> SchneiderTargetSampler::target_probabilities(
    std::size_t section_id, float energy_MeV_per_u) const
{
    if (section_id >= num_sections_) {
        throw std::out_of_range("Schneider section ID out of range: " + std::to_string(section_id));
    }

    // v3: host mirror of the device mask. Strict zero outside the global
    // grid; per-target domain mask before interpolation; no clamp.
    std::array<float, 13> probs{};
    const double e = energy_MeV_per_u;
    const double e_max = energy_min_MeV_per_u_ + (num_energies_ - 1) * energy_step_MeV_per_u_;
    if (!(e >= energy_min_MeV_per_u_) || !(e <= e_max)) {
        return probs;
    }
    const double node_flt = (e - energy_min_MeV_per_u_) / energy_step_MeV_per_u_;
    std::size_t i0 = static_cast<std::size_t>(std::floor(node_flt));
    if (i0 >= num_energies_ - 1) {
        i0 = num_energies_ - 2;
    }
    const std::size_t i1 = i0 + 1;
    const float alpha = static_cast<float>(node_flt - static_cast<double>(i0));
    float sum_p = 0.0F;
    for (std::size_t k = 0; k < num_targets_; ++k) {
        if (domain_has_[k] == 0 || e < domain_emin_[k] || e > domain_emax_[k]) {
            probs[k] = 0.0F;
            continue;
        }
        const std::size_t base =
            section_id * (num_targets_ * num_energies_) + k * num_energies_;
        float v = (1.0F - alpha) * partial_rates_[base + i0] + alpha * partial_rates_[base + i1];
        if (v < 0.0F) {
            v = 0.0F;
        }
        probs[k] = v;
        sum_p += v;
    }
    if (sum_p > 1.0e-7F) {
        for (std::size_t k = 0; k < num_targets_; ++k) {
            probs[k] /= sum_p;
        }
    }
    return probs;

}

float SchneiderTargetSampler::total_mass_rate(std::size_t section_id, float energy_MeV_per_u) const {
    if (section_id >= num_sections_) {
        throw std::out_of_range("Schneider section ID out of range: " + std::to_string(section_id));
    }

    // v3: total = sum of masked interpolated partials (host mirror of the
    // device single-pass computation). Interpolating node totals would leak
    // across domain boundaries.
    const double e = energy_MeV_per_u;
    const double e_max = energy_min_MeV_per_u_ + (num_energies_ - 1) * energy_step_MeV_per_u_;
    if (!(e >= energy_min_MeV_per_u_) || !(e <= e_max)) {
        return 0.0F;
    }
    const double node_flt = (e - energy_min_MeV_per_u_) / energy_step_MeV_per_u_;
    std::size_t i0 = static_cast<std::size_t>(std::floor(node_flt));
    if (i0 >= num_energies_ - 1) {
        i0 = num_energies_ - 2;
    }
    const std::size_t i1 = i0 + 1;
    const float alpha = static_cast<float>(node_flt - static_cast<double>(i0));
    float sum = 0.0F;
    for (std::size_t k = 0; k < num_targets_; ++k) {
        if (domain_has_[k] == 0 || e < domain_emin_[k] || e > domain_emax_[k]) {
            continue;
        }
        const std::size_t base =
            section_id * (num_targets_ * num_energies_) + k * num_energies_;
        float v = (1.0F - alpha) * partial_rates_[base + i0] + alpha * partial_rates_[base + i1];
        if (v < 0.0F) {
            v = 0.0F;
        }
        sum += v;
    }
    // Device unity: the masked total below the sampler cutoff is exactly
    // zero (float-sliver guard, same 1e-12 threshold as device).
    if (!(sum > 1.0e-12F)) {
        return 0.0F;
    }
    return sum;

}

} // namespace carbon
