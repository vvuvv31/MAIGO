#include "carbon/schneider_target_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace carbon {

SchneiderTargetSampler::SchneiderTargetSampler(const SchneiderRateTable& rate_table)
    : rate_version_(rate_table.binary_version()),
      num_sections_(kSchneiderNumSections),
      num_energies_(rate_table.num_energies()),
      num_targets_(kSchneiderNumTargets),
      energy_min_MeV_per_u_(rate_table.energy_min_mevu()),
      energy_step_MeV_per_u_(rate_table.energy_step_mevu())
{
    cdf_table_.resize(num_sections_ * num_energies_ * num_targets_, 0.0F);
    total_mass_rates_.resize(num_sections_ * num_energies_, 0.0F);
    partial_rates_.resize(num_sections_ * num_targets_ * num_energies_, 0.0F);
    domain_emin_.resize(num_targets_, static_cast<float>(energy_min_MeV_per_u_));
    domain_emax_.resize(num_targets_,
                        static_cast<float>(energy_min_MeV_per_u_ +
                                           (num_energies_ - 1) * energy_step_MeV_per_u_));
    domain_has_.resize(num_targets_, 1);
    if (rate_table.has_channel_domains()) {
        for (std::size_t k = 0; k < num_targets_; ++k) {
            const auto& d = rate_table.channel_domain(k);
            domain_emin_[k] = static_cast<float>(d.energy_min_mevu);
            domain_emax_[k] = static_cast<float>(d.energy_max_mevu);
            domain_has_[k] = d.has_support;
        }
    }

    for (std::size_t s = 0; s < num_sections_; ++s) {
        for (std::size_t e = 0; e < num_energies_; ++e) {
            double sum_partials = 0.0;
            std::array<double, kSchneiderNumTargets> partials{};

            for (std::size_t k = 0; k < num_targets_; ++k) {
                // partial_rates_ always stores RAW node partials: the v3
                // device path applies the domain mask at QUERY energy, so
                // masking here would reintroduce the boundary leak. v3 node
                // totals are masked sums (mask at node energy); v1 keeps the
                // legacy unmasked sums EXACTLY.
                const double raw = rate_table.mass_partial_rate(s, k, e);
                const double clean = (std::isfinite(raw) && raw > 0.0) ? raw : 0.0;
                partial_rates_[s * (num_targets_ * num_energies_) + k * num_energies_ + e] =
                    static_cast<float>(clean);
                double node_val = clean;
                if (rate_version_ == 3) {
                    node_val = rate_table.masked_partial_at_node(s, k, e);
                    if (!std::isfinite(node_val) || node_val < 0.0) {
                        node_val = 0.0;
                    }
                }
                partials[k] = node_val;
                sum_partials += node_val;
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

    // v3: host mirror of the device mask. Strict zero outside the global
    // grid; per-target domain mask before interpolation; no clamp.
    if (rate_version_ == 3) {
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

    // v3: total = sum of masked interpolated partials (host mirror of the
    // device single-pass computation). Interpolating node totals would leak
    // across domain boundaries.
    if (rate_version_ == 3) {
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
