#include "carbon/schneider_target_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace carbon {

SchneiderTargetSampler::SchneiderTargetSampler(const SchneiderRateTable& rate_table)
    : num_sections_(kSchneiderNumSections),
      num_energies_(kSchneiderNumEnergies),
      num_targets_(kSchneiderNumTargets),
      energy_min_MeV_per_u_(rate_table.energy_min_mevu()),
      energy_step_MeV_per_u_(rate_table.energy_step_mevu())
{
    cdf_table_.resize(num_sections_ * num_energies_ * num_targets_, 0.0F);
    total_mass_rates_.resize(num_sections_ * num_energies_, 0.0F);

    for (std::size_t s = 0; s < num_sections_; ++s) {
        for (std::size_t e = 0; e < num_energies_; ++e) {
            double sum_partials = 0.0;
            std::array<double, kSchneiderNumTargets> partials{};

            for (std::size_t k = 0; k < num_targets_; ++k) {
                const double r = rate_table.mass_partial_rate(s, k, e);
                partials[k] = (std::isfinite(r) && r > 0.0) ? r : 0.0;
                sum_partials += partials[k];
            }

            const std::size_t total_idx = s * num_energies_ + e;
            total_mass_rates_[total_idx] = static_cast<float>(sum_partials);

            const std::size_t base = s * (num_energies_ * num_targets_) + e * num_targets_;

            if (sum_partials > 1.0e-12) {
                double cumsum = 0.0;
                for (std::size_t k = 0; k < num_targets_; ++k) {
                    cumsum += partials[k];
                    cdf_table_[base + k] = static_cast<float>(std::min(1.0, cumsum / sum_partials));
                }
                cdf_table_[base + num_targets_ - 1] = 1.0F; // Guarantee exact closure to 1.0
            } else {
                for (std::size_t k = 0; k < num_targets_; ++k) {
                    cdf_table_[base + k] = 0.0F;
                }
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

    const auto e_clamped = std::clamp(
        static_cast<double>(energy_MeV_per_u), energy_min_MeV_per_u_,
        energy_min_MeV_per_u_ + (num_energies_ - 1) * energy_step_MeV_per_u_);

    const double node_flt = (e_clamped - energy_min_MeV_per_u_) / energy_step_MeV_per_u_;
    const auto i0 = static_cast<std::size_t>(std::clamp(static_cast<long>(std::floor(node_flt)), 0L, static_cast<long>(num_energies_ - 1)));
    const auto i1 = std::min(i0 + 1, num_energies_ - 1);
    const float alpha = static_cast<float>(node_flt - static_cast<double>(i0));

    const std::size_t base0 = section_id * (num_energies_ * num_targets_) + i0 * num_targets_;
    const std::size_t base1 = section_id * (num_energies_ * num_targets_) + i1 * num_targets_;

    std::array<float, 13> probs{};
    float prev_c0 = 0.0F;
    float prev_c1 = 0.0F;
    float sum_p = 0.0F;

    for (std::size_t k = 0; k < num_targets_; ++k) {
        const float c0 = cdf_table_[base0 + k];
        const float c1 = cdf_table_[base1 + k];
        const float p0 = std::max(0.0F, c0 - prev_c0);
        const float p1 = std::max(0.0F, c1 - prev_c1);
        probs[k] = (1.0F - alpha) * p0 + alpha * p1;
        sum_p += probs[k];
        prev_c0 = c0;
        prev_c1 = c1;
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

    const auto e_clamped = std::clamp(
        static_cast<double>(energy_MeV_per_u), energy_min_MeV_per_u_,
        energy_min_MeV_per_u_ + (num_energies_ - 1) * energy_step_MeV_per_u_);

    const double node_flt = (e_clamped - energy_min_MeV_per_u_) / energy_step_MeV_per_u_;
    const auto i0 = static_cast<std::size_t>(std::clamp(static_cast<long>(std::floor(node_flt)), 0L, static_cast<long>(num_energies_ - 1)));
    const auto i1 = std::min(i0 + 1, num_energies_ - 1);
    const float alpha = static_cast<float>(node_flt - static_cast<double>(i0));

    const float r0 = total_mass_rates_[section_id * num_energies_ + i0];
    const float r1 = total_mass_rates_[section_id * num_energies_ + i1];
    return (1.0F - alpha) * r0 + alpha * r1;
}

} // namespace carbon
