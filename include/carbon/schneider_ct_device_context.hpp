#pragma once

#include "carbon/transport_config.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/material_nuclear_rates.hpp"

#include <cstdint>

#if defined(CARBON_HAS_SYCL) || defined(SYCL_LANGUAGE_VERSION)
namespace sycl {
inline namespace _V1 {
class queue;
}
}
namespace carbon::detail {
struct DeviceMemoryTracker;
}
#endif

namespace carbon {

struct SchneiderCtDeviceContext {
    // Active physics mode
    MaterialPhysicsMode mode{MaterialPhysicsMode::Water};
    bool unified_water{false};
    MaterialNuclearRateView water_primary_rates{}, water_secondary_rates{};
    double water_radiation_length_g_cm2{};

    [[nodiscard]] bool uses_cinel03() const noexcept {
        return is_schneider_ct() || unified_water;
    }
    [[nodiscard]] SchneiderMaskedRates primary_rates(std::size_t section,float energy) const noexcept {
        if (!unified_water) return schneider_masked_rates_device(primary_sampler,section,energy);
        const auto rates=water_primary_rates.evaluate(0,energy);
        SchneiderMaskedRates out{};
        for(std::size_t t=0;t<13;++t) { out.partials[t]=static_cast<float>(rates.partials[t]); out.total+=out.partials[t]; }
        return out;
    }
    [[nodiscard]] SecondaryMaskedRates secondary_rates(int projectile,std::size_t section,float energy) const noexcept {
        if (!unified_water) return secondary_masked_rates_device(sec_partial_rates,sec_domain_emin,
            sec_domain_emax,sec_domain_has,sec_num_projectiles,projectile,section,energy,
            sec_energy_min_MeV_per_u,sec_inv_energy_step,sec_num_energies);
        const auto rates=water_secondary_rates.evaluate(static_cast<std::size_t>(projectile),energy);
        SecondaryMaskedRates out{};
        for(std::size_t t=0;t<13;++t) { out.partials[t]=static_cast<float>(rates.partials[t]); out.total+=out.partials[t]; }
        return out;
    }

    // Primary Target Sampler
    SchneiderTargetSamplerDeviceTable primary_sampler{};

    // Primary C12 CINEL03 Package
    const Cinel03EnergyNode* c12_energy_nodes{nullptr};
    const std::uint32_t* c12_event_offsets{nullptr};
    const std::uint32_t* c12_event_indices{nullptr};
    const Cinel03DeviceInteraction* c12_interactions{nullptr};
    const Cinel03DeviceProduct* c12_products{nullptr};
    std::uint32_t c12_node_count{0};
    std::uint32_t c12_total_events{0};
    std::uint32_t c12_total_products{0};

    // Secondary Rates
    const float* sec_total_rates{nullptr};
    const float* sec_partial_rates{nullptr};
    // v3 only: bundle-ordered projectile registry keys [z0,a0,z1,a1,...]
    // and per-(projectile,target) valid-domain arrays ([proj][target]).
    // The v3 kernel path never uses sec_total_rates (mask applies at query
    // energy); it sums masked partials on the fly. v1 ignores the v3
    // pointers and keeps the legacy behavior EXACTLY.
    const std::int32_t* sec_proj_keys{nullptr};
    const float* sec_domain_emin{nullptr};
    const float* sec_domain_emax{nullptr};
    const unsigned char* sec_domain_has{nullptr};
    std::uint32_t sec_num_projectiles{13};
    std::uint32_t sec_num_sections{25};
    std::uint32_t sec_num_targets{13};
    std::uint32_t sec_num_energies{860};
    float sec_energy_min_MeV_per_u{0.5F};
    float sec_energy_step_MeV_per_u{0.5F};
    float sec_inv_energy_step{2.0F};

    // Secondary CINEL03 Package
    const Cinel03EnergyNode* sec_energy_nodes{nullptr};
    const std::uint32_t* sec_event_offsets{nullptr};
    const std::uint32_t* sec_event_indices{nullptr};
    const Cinel03DeviceInteraction* sec_interactions{nullptr};
    const Cinel03DeviceProduct* sec_products{nullptr};
    std::uint32_t sec_node_count{0};
    std::uint32_t sec_total_events{0};
    std::uint32_t sec_total_products{0};

    // Stopping power
    const float* stopping_power_device{nullptr};
    std::uint32_t sp_sections{25};
    std::uint32_t sp_energies{4302};
    float sp_e_min{0.01F};
    float sp_e_max{430.11F};
    float sp_inv_dE{10.0F};

    [[nodiscard]] bool is_schneider_ct() const noexcept {
        return mode == MaterialPhysicsMode::SchneiderCt;
    }
};

#if defined(CARBON_HAS_SYCL) || defined(SYCL_LANGUAGE_VERSION)
SchneiderCtDeviceContext upload_schneider_ct_device_context(
    sycl::queue& queue,
    carbon::detail::DeviceMemoryTracker& mem_tracker,
    const TransportConfig& config,
    const float* schneider_stopping_device = nullptr,
    std::uint32_t schneider_sp_sections = 25,
    std::uint32_t schneider_sp_energies = 4302,
    float schneider_sp_e_min = 0.01F,
    float schneider_sp_e_max = 430.11F,
    float schneider_sp_inv_dE = 10.0F);
#endif

} // namespace carbon
