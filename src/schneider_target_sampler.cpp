#include "schneider_target_sampler.hpp"
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

} // namespace carbon
