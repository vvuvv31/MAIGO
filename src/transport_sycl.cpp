#include "carbon/device.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace carbon {
namespace {

constexpr std::size_t fragment_species_count = 7;

bool is_uniform_grid(const std::vector<double>& energies) {
    const auto expected_step = energies[1] - energies[0];
    for (std::size_t index = 2; index < energies.size(); ++index) {
        const auto actual_step = energies[index] - energies[index - 1];
        if (std::abs(actual_step - expected_step) > 1.0e-6 * expected_step) {
            return false;
        }
    }
    return true;
}

}  // namespace

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages,
                               const CascadePackageTable* cascade_packages) {
    config.validate();
    if (!is_uniform_grid(stopping_power.energies())) {
        throw std::invalid_argument("The current SYCL backend requires a uniform stopping-power grid");
    }
    if (!is_uniform_grid(cross_section.energies())) {
        throw std::invalid_argument("The current SYCL backend requires a uniform cross-section grid");
    }
    if (config.enable_secondary_generation && !config.enable_primary_attenuation) {
        throw std::invalid_argument(
            "Secondary generation requires enable_primary_attenuation=true");
    }
    if (config.enable_secondary_generation && reaction_packages == nullptr) {
        throw std::invalid_argument("Secondary generation requires a reaction package table");
    }
    if (config.enable_fragment_cascade && cascade_packages == nullptr) {
        throw std::invalid_argument("Fragment cascade requires a cascade package table");
    }

    auto queue = make_sycl_queue(device_name);
    const auto start = std::chrono::steady_clock::now();
    const auto table_size = stopping_power.values().size();
    const auto cross_section_table_size = cross_section.values().size();
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_histories = config.number_of_histories;
    const auto enable_secondary_generation = config.enable_secondary_generation;
    const auto enable_secondary_transport = config.enable_secondary_transport;
    const auto enable_fragment_cascade = config.enable_fragment_cascade;
    const auto automatic_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() / 16
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * 16;
    const auto secondary_queue_capacity =
        config.secondary_queue_capacity == 0 ? automatic_queue_capacity
                                             : config.secondary_queue_capacity;
    if (enable_secondary_generation &&
        secondary_queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Secondary queue capacity exceeds the uint32 runtime limit");
    }

    const auto& device = queue.get_device();
    if (!device.has(sycl::aspect::fp64) || !device.has(sycl::aspect::atomic64)) {
        throw std::runtime_error(
            "The current accurate SYCL scorer requires fp64 and atomic64 device aspects");
    }

    auto* table_device = sycl::malloc_device<float>(table_size, queue);
    auto* cross_section_device =
        sycl::malloc_device<float>(cross_section_table_size, queue);
    auto* dose_device = sycl::malloc_device<double>(number_of_bins, queue);
    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* nuclear_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);
    ReactionEnergyBin* reaction_bins_device = nullptr;
    ReactionPackage* reactions_device = nullptr;
    ReactionSecondary* reaction_secondaries_device = nullptr;
    SecondaryParticle1D* secondary_queue_device = nullptr;
    std::uint64_t* secondary_queue_counter_device = nullptr;
    std::uint64_t* secondary_queue_filled_device = nullptr;
    SecondaryGenerationSummary* secondary_summaries_device = nullptr;
    double* fragment_dose_device = nullptr;
    float* secondary_deposited_device = nullptr;
    float* secondary_escaped_device = nullptr;
    std::uint32_t* secondary_steps_device = nullptr;
    CascadeProjectile* cascade_projectiles_device = nullptr;
    CascadeCrossSectionSample* cascade_cross_sections_device = nullptr;
    CascadeInteraction* cascade_interactions_device = nullptr;
    ReactionSecondary* cascade_products_device = nullptr;
    CascadeTransportSummary* cascade_summaries_device = nullptr;
    if (enable_secondary_generation) {
        reaction_bins_device = sycl::malloc_device<ReactionEnergyBin>(
            reaction_packages->energy_bins().size(), queue);
        reactions_device = sycl::malloc_device<ReactionPackage>(
            reaction_packages->reactions().size(), queue);
        reaction_secondaries_device = sycl::malloc_device<ReactionSecondary>(
            reaction_packages->secondaries().size(), queue);
        secondary_queue_device =
            sycl::malloc_device<SecondaryParticle1D>(secondary_queue_capacity, queue);
        secondary_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_summaries_device =
            sycl::malloc_device<SecondaryGenerationSummary>(number_of_histories, queue);
        if (enable_secondary_transport) {
            fragment_dose_device = sycl::malloc_device<double>(
                fragment_species_count * number_of_bins, queue);
            secondary_deposited_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_escaped_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_steps_device =
                sycl::malloc_device<std::uint32_t>(secondary_queue_capacity, queue);
        }
        if (enable_fragment_cascade) {
            cascade_projectiles_device = sycl::malloc_device<CascadeProjectile>(
                cascade_packages->projectiles().size(), queue);
            cascade_cross_sections_device = sycl::malloc_device<CascadeCrossSectionSample>(
                cascade_packages->cross_sections().size(), queue);
            cascade_interactions_device = sycl::malloc_device<CascadeInteraction>(
                cascade_packages->interactions().size(), queue);
            cascade_products_device = sycl::malloc_device<ReactionSecondary>(
                cascade_packages->products().size(), queue);
            cascade_summaries_device = sycl::malloc_device<CascadeTransportSummary>(
                secondary_queue_capacity, queue);
        }
    }
    const auto free_device = [&queue](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };
    const auto secondary_allocation_failed =
        enable_secondary_generation &&
        (reaction_bins_device == nullptr || reactions_device == nullptr ||
         reaction_secondaries_device == nullptr || secondary_queue_device == nullptr ||
         secondary_queue_counter_device == nullptr || secondary_queue_filled_device == nullptr ||
         secondary_summaries_device == nullptr ||
         (enable_secondary_transport &&
           (fragment_dose_device == nullptr || secondary_deposited_device == nullptr ||
           secondary_escaped_device == nullptr || secondary_steps_device == nullptr)) ||
         (enable_fragment_cascade &&
          (cascade_projectiles_device == nullptr || cascade_cross_sections_device == nullptr ||
           cascade_interactions_device == nullptr || cascade_products_device == nullptr ||
           cascade_summaries_device == nullptr)));
    if (table_device == nullptr || cross_section_device == nullptr || dose_device == nullptr ||
        deposited_device == nullptr ||
        escaped_device == nullptr || nuclear_device == nullptr || steps_device == nullptr ||
        secondary_allocation_failed) {
        free_device(table_device);
        free_device(cross_section_device);
        free_device(dose_device);
        free_device(deposited_device);
        free_device(escaped_device);
        free_device(nuclear_device);
        free_device(steps_device);
        free_device(reaction_bins_device);
        free_device(reactions_device);
        free_device(reaction_secondaries_device);
        free_device(secondary_queue_device);
        free_device(secondary_queue_counter_device);
        free_device(secondary_queue_filled_device);
        free_device(secondary_summaries_device);
        free_device(fragment_dose_device);
        free_device(secondary_deposited_device);
        free_device(secondary_escaped_device);
        free_device(secondary_steps_device);
        free_device(cascade_projectiles_device);
        free_device(cascade_cross_sections_device);
        free_device(cascade_interactions_device);
        free_device(cascade_products_device);
        free_device(cascade_summaries_device);
        throw std::runtime_error("SYCL USM device allocation failed");
    }

    std::vector<float> table_host(table_size);
    std::transform(stopping_power.values().begin(), stopping_power.values().end(), table_host.begin(),
                   [](double value) { return static_cast<float>(value); });
    queue.copy(table_host.data(), table_device, table_size);
    std::vector<float> cross_section_host(cross_section_table_size);
    std::transform(cross_section.values().begin(), cross_section.values().end(),
                   cross_section_host.begin(),
                   [](double value) { return static_cast<float>(value); });
    queue.copy(cross_section_host.data(), cross_section_device, cross_section_table_size);
    queue.memset(dose_device, 0, number_of_bins * sizeof(double));
    if (enable_secondary_generation) {
        queue.copy(reaction_packages->energy_bins().data(), reaction_bins_device,
                   reaction_packages->energy_bins().size());
        queue.copy(reaction_packages->reactions().data(), reactions_device,
                   reaction_packages->reactions().size());
        queue.copy(reaction_packages->secondaries().data(), reaction_secondaries_device,
                   reaction_packages->secondaries().size());
        queue.memset(secondary_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(secondary_queue_filled_device, 0, sizeof(std::uint64_t));
        if (enable_secondary_transport) {
            queue.memset(fragment_dose_device, 0,
                         fragment_species_count * number_of_bins * sizeof(double));
        }
        if (enable_fragment_cascade) {
            queue.copy(cascade_packages->projectiles().data(), cascade_projectiles_device,
                       cascade_packages->projectiles().size());
            queue.copy(cascade_packages->cross_sections().data(), cascade_cross_sections_device,
                       cascade_packages->cross_sections().size());
            queue.copy(cascade_packages->interactions().data(), cascade_interactions_device,
                       cascade_packages->interactions().size());
            queue.copy(cascade_packages->products().data(), cascade_products_device,
                       cascade_packages->products().size());
            queue.memset(cascade_summaries_device, 0,
                         secondary_queue_capacity * sizeof(CascadeTransportSummary));
        }
    }

    constexpr std::size_t local_size = 128;
    const auto global_size =
        ((number_of_histories + local_size - 1) / local_size) * local_size;
    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);
    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto random_seed = config.random_seed;
    const auto enable_primary_attenuation = config.enable_primary_attenuation;
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.mass_number);
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);
    const auto minimum_cross_section_energy =
        static_cast<float>(cross_section.energies().front());
    const auto inverse_cross_section_step =
        1.0f / static_cast<float>(cross_section.energies()[1] - cross_section.energies()[0]);
    const auto reaction_energy_bin_count =
        enable_secondary_generation
            ? static_cast<std::uint32_t>(reaction_packages->energy_bins().size())
            : 0U;
    const auto minimum_reaction_energy =
        enable_secondary_generation ? reaction_packages->minimum_energy_MeV_per_u() : 0.0F;
    const auto inverse_reaction_energy_bin_width =
        enable_secondary_generation
            ? 1.0F / reaction_packages->energy_bin_width_MeV_per_u()
            : 0.0F;
    const auto secondary_queue_capacity_u32 =
        static_cast<std::uint32_t>(secondary_queue_capacity);
    const auto cascade_projectile_count = enable_fragment_cascade
                                              ? cascade_packages->projectiles().size()
                                              : std::size_t{0};
    const auto maximum_cascade_generations = config.maximum_cascade_generations;

    auto kernel_event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{global_size}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) {
            const auto history = item.get_global_linear_id();
            if (history >= number_of_histories) {
                return;
            }

            auto energy_MeV = initial_energy_MeV;
            auto position_mm = 0.0f;
            auto history_deposited_MeV = 0.0f;
            auto history_nuclear_MeV = 0.0f;
            SecondaryGenerationSummary secondary_summary{};
            std::uint32_t steps = 0;
            while (energy_MeV > energy_cutoff_MeV && position_mm < phantom_length_mm) {
                const auto energy_MeVu = energy_MeV * inverse_mass_number;
                auto floating_index = (energy_MeVu - minimum_table_energy) * inverse_table_step;
                auto index = static_cast<int>(sycl::floor(floating_index));
                index = sycl::max(0, sycl::min(index, static_cast<int>(table_size) - 2));
                const auto fraction = sycl::clamp(floating_index - static_cast<float>(index),
                                                  0.0f, 1.0f);
                const auto stopping_power_MeV_per_mm =
                    table_device[index] +
                    fraction * (table_device[index + 1] - table_device[index]);

                auto step_mm = sycl::fmin(
                    maximum_step_mm,
                    maximum_relative_energy_loss * energy_MeV / stopping_power_MeV_per_mm);
                const auto bin = sycl::min(
                    static_cast<int>(position_mm / depth_bin_width_mm),
                    static_cast<int>(number_of_bins) - 1);
                const auto next_bin_boundary_mm =
                    static_cast<float>(bin + 1) * depth_bin_width_mm;
                step_mm = sycl::fmin(step_mm, next_bin_boundary_mm - position_mm);
                step_mm = sycl::fmin(step_mm, phantom_length_mm - position_mm);

                const auto mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);
                if (enable_energy_straggling) {
                    const auto uniform1 = sycl::fmax(
                        rng::uniform01(random_seed, history, steps, 0), 1.0e-12f);
                    const auto uniform2 = rng::uniform01(random_seed, history, steps, 1);
                    constexpr float two_pi = 6.2831853071795864769f;
                    const auto gaussian = sycl::sqrt(-2.0f * sycl::log(uniform1)) *
                                          sycl::cos(two_pi * uniform2);

                    constexpr float nucleon_mass_MeV = 931.49410242f;
                    constexpr float carbon_atomic_number = 6.0f;
                    constexpr float carbon_charge_power = 0.30285343214f;
                    const auto gamma = 1.0f + energy_MeVu / nucleon_mass_MeV;
                    const auto beta_squared =
                        sycl::fmax(0.0f, 1.0f - 1.0f / (gamma * gamma));
                    const auto beta = sycl::sqrt(beta_squared);
                    const auto effective_charge =
                        carbon_atomic_number *
                        (1.0f - sycl::exp(-125.0f * beta * carbon_charge_power));
                    constexpr float bethe_K_MeV_cm2_per_g = 0.307075f;
                    constexpr float electron_mass_MeV = 0.51099895f;
                    constexpr float water_Z_over_A = 0.55509f;
                    const auto variance_MeV2 =
                        bethe_K_MeV_cm2_per_g * electron_mass_MeV * effective_charge *
                        effective_charge * water_Z_over_A * water_density_g_per_cm3 *
                        (step_mm / 10.0f);
                    const auto sigma_MeV =
                        straggling_scale * sycl::sqrt(sycl::fmax(0.0f, variance_MeV2));
                    deposited_MeV = sycl::clamp(
                        mean_loss_MeV + sigma_MeV * gaussian, 0.0f, energy_MeV);
                }
                sycl::atomic_ref<double,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[bin]);
                atomic_dose.fetch_add(static_cast<double>(deposited_MeV));
                history_deposited_MeV += deposited_MeV;
                energy_MeV -= deposited_MeV;
                position_mm += step_mm;
                if (enable_primary_attenuation && energy_MeV > energy_cutoff_MeV) {
                    const auto post_step_energy_MeVu = energy_MeV * inverse_mass_number;
                    auto cross_section_floating_index =
                        (post_step_energy_MeVu - minimum_cross_section_energy) *
                        inverse_cross_section_step;
                    auto cross_section_index =
                        static_cast<int>(sycl::floor(cross_section_floating_index));
                    cross_section_index = sycl::max(
                        0, sycl::min(cross_section_index,
                                     static_cast<int>(cross_section_table_size) - 2));
                    const auto cross_section_fraction = sycl::clamp(
                        cross_section_floating_index - static_cast<float>(cross_section_index),
                        0.0f, 1.0f);
                    const auto macroscopic_cross_section_per_mm =
                        cross_section_device[cross_section_index] +
                        cross_section_fraction *
                            (cross_section_device[cross_section_index + 1] -
                             cross_section_device[cross_section_index]);
                    const auto probability = 1.0f - sycl::exp(
                        -macroscopic_cross_section_per_mm * step_mm);
                    const auto uniform = rng::uniform01(random_seed, history, steps, 2);
                    if (uniform < probability) {
                        history_nuclear_MeV = energy_MeV;
                        if (enable_secondary_generation) {
                            auto reaction_bin_index = static_cast<int>(sycl::floor(
                                (post_step_energy_MeVu - minimum_reaction_energy) *
                                inverse_reaction_energy_bin_width));
                            reaction_bin_index = sycl::max(
                                0, sycl::min(reaction_bin_index,
                                             static_cast<int>(reaction_energy_bin_count) - 1));
                            const auto reaction_bin = reaction_bins_device[reaction_bin_index];
                            const auto package_uniform =
                                rng::uniform01(random_seed, history, steps, 3);
                            const auto package_in_bin = sycl::min(
                                static_cast<std::uint32_t>(
                                    package_uniform * reaction_bin.reaction_count),
                                reaction_bin.reaction_count - 1U);
                            const auto reaction =
                                reactions_device[reaction_bin.reaction_offset + package_in_bin];
                            secondary_summary.direct_count = reaction.secondary_count;

                            std::uint32_t queueable_count = 0;
                            auto queueable_energy_MeV = 0.0F;
                            for (std::uint32_t secondary_index = 0;
                                 secondary_index < reaction.secondary_count;
                                 ++secondary_index) {
                                const auto secondary = reaction_secondaries_device[
                                    reaction.secondary_offset + secondary_index];
                                const auto is_neutral = secondary.pdg_id == 22 ||
                                                        secondary.pdg_id == 2112;
                                const auto is_supported = secondary.atomic_number > 0 &&
                                                          secondary.mass_number > 0;
                                if (is_neutral) {
                                    secondary_summary.neutral_energy_MeV +=
                                        secondary.kinetic_energy_MeV;
                                } else if (is_supported) {
                                    ++queueable_count;
                                    queueable_energy_MeV += secondary.kinetic_energy_MeV;
                                } else {
                                    secondary_summary.unsupported_charged_energy_MeV +=
                                        secondary.kinetic_energy_MeV;
                                }
                            }

                            if (queueable_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    queue_counter(*secondary_queue_counter_device);
                                const auto queue_offset =
                                    queue_counter.fetch_add(queueable_count);
                                const auto package_fits =
                                    queue_offset <= secondary_queue_capacity_u32 &&
                                    queueable_count <=
                                        secondary_queue_capacity_u32 - queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index < reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary = reaction_secondaries_device[
                                            reaction.secondary_offset + secondary_index];
                                        const auto is_supported =
                                            secondary.atomic_number > 0 &&
                                            secondary.mass_number > 0;
                                        if (is_supported) {
                                            const auto origin_category =
                                                charged_dose_category(
                                                    secondary.atomic_number,
                                                    secondary.mass_number);
                                            secondary_queue_device[output_index++] =
                                                SecondaryParticle1D{
                                                    position_mm,
                                                    secondary.kinetic_energy_MeV,
                                                    secondary.direction_z,
                                                    secondary.pdg_id,
                                                    secondary.atomic_number,
                                                    secondary.mass_number,
                                                    origin_category,
                                                    0,
                                                    0,
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled_counter(*secondary_queue_filled_device);
                                    filled_counter.fetch_add(queueable_count);
                                    secondary_summary.queued_count = queueable_count;
                                    secondary_summary.queued_energy_MeV =
                                        queueable_energy_MeV;
                                } else {
                                    secondary_summary.overflow_count = queueable_count;
                                    secondary_summary.overflow_energy_MeV =
                                        queueable_energy_MeV;
                                }
                            }
                        }
                        energy_MeV = 0.0f;
                    }
                }
                ++steps;
            }

            if (energy_MeV > 0.0f && position_mm < phantom_length_mm) {
                const auto bin = sycl::min(
                    static_cast<int>(position_mm / depth_bin_width_mm),
                    static_cast<int>(number_of_bins) - 1);
                sycl::atomic_ref<double,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[bin]);
                atomic_dose.fetch_add(static_cast<double>(energy_MeV));
                history_deposited_MeV += energy_MeV;
                energy_MeV = 0.0f;
            }
            deposited_device[history] = history_deposited_MeV;
            escaped_device[history] = energy_MeV;
            nuclear_device[history] = history_nuclear_MeV;
            steps_device[history] = steps;
            if (enable_secondary_generation) {
                secondary_summaries_device[history] = secondary_summary;
            }
        });
    kernel_event.wait_and_throw();

    std::uint64_t transported_queue_count = 0;
    if (enable_secondary_transport) {
        std::uint64_t generation_begin = 0;
        std::uint64_t generation_end = 0;
        queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
        while (generation_begin < generation_end) {
            const auto generation_size = generation_end - generation_begin;
            const auto secondary_global_size =
                ((generation_size + local_size - 1) / local_size) * local_size;
            auto secondary_kernel_event = queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{secondary_global_size},
                              sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                const auto generation_index = item.get_global_linear_id();
                if (generation_index >= generation_size) {
                    return;
                }
                const auto particle_index = generation_begin + generation_index;
                auto deposited_MeV = 0.0F;
                auto escaped_MeV = 0.0F;
                std::uint32_t steps = 0;
                CascadeTransportSummary cascade_summary{};
                if (particle_index < generation_end) {
                    const auto particle = secondary_queue_device[particle_index];
                    auto energy_MeV = particle.kinetic_energy_MeV;
                    auto position_mm = particle.position_mm;
                    const auto direction_z = sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                    const auto absolute_direction_z = sycl::fabs(direction_z);
                    const auto atomic_number = static_cast<int>(particle.atomic_number);
                    const auto mass_number = static_cast<int>(particle.mass_number);
                    const auto species_index =
                        sycl::min(static_cast<std::size_t>(particle.origin_category),
                                  fragment_species_count - 1);

                    while (energy_MeV > energy_cutoff_MeV) {
                        const auto escaped_forward =
                            direction_z >= 0.0F && position_mm >= phantom_length_mm;
                        const auto escaped_backward =
                            direction_z < 0.0F && position_mm <= 0.0F;
                        if (escaped_forward || escaped_backward) {
                            break;
                        }

                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));

                        if (absolute_direction_z < 1.0e-6F) {
                            sycl::atomic_ref<
                                double,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                atomic_fragment_dose(
                                    fragment_dose_device[species_index * number_of_bins + bin]);
                            atomic_fragment_dose.fetch_add(static_cast<double>(energy_MeV));
                            deposited_MeV += energy_MeV;
                            energy_MeV = 0.0F;
                            break;
                        }

                        const auto energy_MeVu =
                            energy_MeV / static_cast<float>(mass_number);
                        auto floating_index =
                            (energy_MeVu - minimum_table_energy) * inverse_table_step;
                        auto index = static_cast<int>(sycl::floor(floating_index));
                        index = sycl::max(
                            0, sycl::min(index, static_cast<int>(table_size) - 2));
                        const auto fraction = sycl::clamp(
                            floating_index - static_cast<float>(index), 0.0F, 1.0F);
                        const auto carbon_stopping_power_MeV_per_mm =
                            table_device[index] +
                            fraction * (table_device[index + 1] - table_device[index]);

                        constexpr float nucleon_mass_MeV = 931.49410242F;
                        const auto gamma = 1.0F + energy_MeVu / nucleon_mass_MeV;
                        const auto beta_squared =
                            sycl::fmax(0.0F, 1.0F - 1.0F / (gamma * gamma));
                        const auto beta = sycl::sqrt(beta_squared);
                        const auto charge = static_cast<float>(atomic_number);
                        const auto effective_charge =
                            charge *
                            (1.0F - sycl::exp(
                                         -125.0F * beta * sycl::pow(charge, -2.0F / 3.0F)));
                        constexpr float carbon_charge = 6.0F;
                        const auto carbon_effective_charge =
                            carbon_charge *
                            (1.0F - sycl::exp(
                                         -125.0F * beta *
                                         sycl::pow(carbon_charge, -2.0F / 3.0F)));
                        const auto charge_ratio =
                            effective_charge / carbon_effective_charge;
                        const auto stopping_power_MeV_per_mm =
                            carbon_stopping_power_MeV_per_mm * charge_ratio * charge_ratio;
                        auto path_step_mm = sycl::fmin(
                            maximum_step_mm,
                            maximum_relative_energy_loss * energy_MeV /
                                stopping_power_MeV_per_mm);
                        const auto boundary_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto distance_to_boundary_mm =
                            direction_z < 0.0F ? position_mm - boundary_mm
                                               : boundary_mm - position_mm;
                        path_step_mm = sycl::fmin(
                            path_step_mm, distance_to_boundary_mm / absolute_direction_z);
                        const auto step_deposited_MeV = sycl::fmin(
                            stopping_power_MeV_per_mm * path_step_mm, energy_MeV);
                        sycl::atomic_ref<
                            double,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_fragment_dose(
                                fragment_dose_device[species_index * number_of_bins + bin]);
                        atomic_fragment_dose.fetch_add(
                            static_cast<double>(step_deposited_MeV));
                        deposited_MeV += step_deposited_MeV;
                        energy_MeV -= step_deposited_MeV;
                        position_mm += direction_z * path_step_mm;

                        if (enable_fragment_cascade &&
                            particle.generation < maximum_cascade_generations &&
                            energy_MeV > energy_cutoff_MeV) {
                            int projectile_index = -1;
                            for (std::size_t candidate = 0;
                                 candidate < cascade_projectile_count; ++candidate) {
                                const auto projectile = cascade_projectiles_device[candidate];
                                if (projectile.atomic_number == atomic_number &&
                                    projectile.mass_number == mass_number) {
                                    projectile_index = static_cast<int>(candidate);
                                    break;
                                }
                            }
                            if (projectile_index >= 0) {
                                const auto projectile =
                                    cascade_projectiles_device[projectile_index];
                                const auto current_energy_MeVu =
                                    energy_MeV / static_cast<float>(mass_number);
                                std::uint32_t upper = 0;
                                while (upper < projectile.cross_section_count &&
                                       cascade_cross_sections_device[
                                           projectile.cross_section_offset + upper]
                                               .energy_MeV_per_u < current_energy_MeVu) {
                                    ++upper;
                                }
                                float macroscopic_cross_section_per_mm = 0.0F;
                                if (upper == 0) {
                                    macroscopic_cross_section_per_mm =
                                        cascade_cross_sections_device[
                                            projectile.cross_section_offset]
                                            .macroscopic_cross_section_per_mm;
                                } else if (upper >= projectile.cross_section_count) {
                                    macroscopic_cross_section_per_mm =
                                        cascade_cross_sections_device[
                                            projectile.cross_section_offset +
                                            projectile.cross_section_count - 1]
                                            .macroscopic_cross_section_per_mm;
                                } else {
                                    const auto lower_sample = cascade_cross_sections_device[
                                        projectile.cross_section_offset + upper - 1];
                                    const auto upper_sample = cascade_cross_sections_device[
                                        projectile.cross_section_offset + upper];
                                    const auto interval = upper_sample.energy_MeV_per_u -
                                                          lower_sample.energy_MeV_per_u;
                                    const auto xs_fraction = interval > 0.0F
                                                                 ? sycl::clamp(
                                                                       (current_energy_MeVu -
                                                                        lower_sample.energy_MeV_per_u) /
                                                                           interval,
                                                                       0.0F, 1.0F)
                                                                 : 0.0F;
                                    macroscopic_cross_section_per_mm =
                                        lower_sample.macroscopic_cross_section_per_mm +
                                        xs_fraction *
                                            (upper_sample.macroscopic_cross_section_per_mm -
                                             lower_sample.macroscopic_cross_section_per_mm);
                                }
                                const auto interaction_probability =
                                    1.0F - sycl::exp(-macroscopic_cross_section_per_mm *
                                                     path_step_mm);
                                if (rng::uniform01(random_seed, particle_index, steps, 8) <
                                    interaction_probability) {
                                    std::uint32_t nearest = 0;
                                    while (nearest + 1 < projectile.interaction_count &&
                                           cascade_interactions_device[
                                               projectile.interaction_offset + nearest + 1]
                                                   .incident_energy_MeV_per_u <
                                               current_energy_MeVu) {
                                        ++nearest;
                                    }
                                    if (nearest + 1 < projectile.interaction_count) {
                                        const auto lower_delta = sycl::fabs(
                                            cascade_interactions_device[
                                                projectile.interaction_offset + nearest]
                                                    .incident_energy_MeV_per_u -
                                            current_energy_MeVu);
                                        const auto upper_delta = sycl::fabs(
                                            cascade_interactions_device[
                                                projectile.interaction_offset + nearest + 1]
                                                    .incident_energy_MeV_per_u -
                                            current_energy_MeVu);
                                        nearest += upper_delta < lower_delta ? 1U : 0U;
                                    }
                                    constexpr std::uint32_t sampling_window = 8;
                                    const auto window_begin =
                                        nearest > sampling_window / 2
                                            ? nearest - sampling_window / 2
                                            : 0U;
                                    const auto window_count = sycl::min(
                                        sampling_window,
                                        projectile.interaction_count - window_begin);
                                    const auto package_uniform =
                                        rng::uniform01(random_seed, particle_index, steps, 9);
                                    const auto selected_in_window = sycl::min(
                                        static_cast<std::uint32_t>(package_uniform * window_count),
                                        window_count - 1U);
                                    const auto interaction = cascade_interactions_device[
                                        projectile.interaction_offset + window_begin +
                                        selected_in_window];
                                    const auto energy_scale =
                                        interaction.incident_energy_MeV_per_u > 0.0F
                                            ? current_energy_MeVu /
                                                  interaction.incident_energy_MeV_per_u
                                            : 1.0F;
                                    cascade_summary.interaction_count = 1;
                                    cascade_summary.direct_count = interaction.product_count;
                                    cascade_summary.incident_energy_MeV = energy_MeV;
                                    std::uint32_t queueable_count = 0;
                                    auto queueable_energy = 0.0F;
                                    for (std::uint32_t product_index = 0;
                                         product_index < interaction.product_count;
                                         ++product_index) {
                                        const auto product = cascade_products_device[
                                            interaction.product_offset + product_index];
                                        const auto scaled_energy =
                                            product.kinetic_energy_MeV * energy_scale;
                                        const auto is_neutral =
                                            product.pdg_id == 22 || product.pdg_id == 2112;
                                        const auto supported = product.atomic_number > 0 &&
                                                               product.mass_number > 0;
                                        if (is_neutral) {
                                            cascade_summary.neutral_energy_MeV += scaled_energy;
                                        } else if (supported) {
                                            ++queueable_count;
                                            queueable_energy += scaled_energy;
                                        } else {
                                            cascade_summary.unsupported_charged_energy_MeV +=
                                                scaled_energy;
                                        }
                                    }
                                    if (queueable_count > 0) {
                                        sycl::atomic_ref<
                                            std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            queue_counter(*secondary_queue_counter_device);
                                        const auto queue_offset =
                                            queue_counter.fetch_add(queueable_count);
                                        const auto package_fits =
                                            queue_offset <= secondary_queue_capacity_u32 &&
                                            queueable_count <=
                                                secondary_queue_capacity_u32 - queue_offset;
                                        if (package_fits) {
                                            auto output_index = queue_offset;
                                            for (std::uint32_t product_index = 0;
                                                 product_index < interaction.product_count;
                                                 ++product_index) {
                                                const auto product = cascade_products_device[
                                                    interaction.product_offset + product_index];
                                                if (product.atomic_number > 0 &&
                                                    product.mass_number > 0) {
                                                    const auto child_direction = sycl::clamp(
                                                        product.direction_z *
                                                            (direction_z < 0.0F ? -1.0F : 1.0F),
                                                        -1.0F, 1.0F);
                                                    secondary_queue_device[output_index++] =
                                                        SecondaryParticle1D{
                                                            position_mm,
                                                            product.kinetic_energy_MeV *
                                                                energy_scale,
                                                            child_direction,
                                                            product.pdg_id,
                                                            product.atomic_number,
                                                            product.mass_number,
                                                            charged_dose_category(
                                                                product.atomic_number,
                                                                product.mass_number),
                                                            static_cast<std::uint8_t>(
                                                                particle.generation + 1),
                                                            0,
                                                        };
                                                }
                                            }
                                            sycl::atomic_ref<
                                                std::uint64_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                filled_counter(*secondary_queue_filled_device);
                                            filled_counter.fetch_add(queueable_count);
                                            cascade_summary.queued_count = queueable_count;
                                            cascade_summary.queued_energy_MeV = queueable_energy;
                                        } else {
                                            cascade_summary.overflow_count = queueable_count;
                                            cascade_summary.overflow_energy_MeV = queueable_energy;
                                        }
                                    }
                                    energy_MeV = 0.0F;
                                }
                            }
                        }
                        ++steps;
                    }

                    const auto stopped_inside =
                        energy_MeV > 0.0F && position_mm >= 0.0F &&
                        position_mm < phantom_length_mm &&
                        !((direction_z < 0.0F && position_mm <= 0.0F) ||
                          (direction_z >= 0.0F && position_mm >= phantom_length_mm));
                    if (stopped_inside) {
                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                        sycl::atomic_ref<
                            double,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_fragment_dose(
                                fragment_dose_device[species_index * number_of_bins + bin]);
                        atomic_fragment_dose.fetch_add(static_cast<double>(energy_MeV));
                        deposited_MeV += energy_MeV;
                        energy_MeV = 0.0F;
                    }
                    escaped_MeV = energy_MeV;
                }
                secondary_deposited_device[particle_index] = deposited_MeV;
                secondary_escaped_device[particle_index] = escaped_MeV;
                secondary_steps_device[particle_index] = steps;
                if (enable_fragment_cascade) {
                    cascade_summaries_device[particle_index] = cascade_summary;
                }
                });
            secondary_kernel_event.wait_and_throw();
            transported_queue_count = generation_end;
            if (!enable_fragment_cascade) {
                break;
            }
            generation_begin = generation_end;
            queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
            generation_end = std::min<std::uint64_t>(generation_end,
                                                     secondary_queue_capacity);
        }
    }

    std::vector<double> dose_host(number_of_bins);
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<float> nuclear_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<SecondaryGenerationSummary> secondary_summaries_host;
    std::vector<double> fragment_dose_host;
    std::vector<float> secondary_deposited_host;
    std::vector<float> secondary_escaped_host;
    std::vector<std::uint32_t> secondary_steps_host;
    std::vector<CascadeTransportSummary> cascade_summaries_host;
    queue.copy(dose_device, dose_host.data(), number_of_bins);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(nuclear_device, nuclear_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories).wait_and_throw();
    if (enable_secondary_generation) {
        secondary_summaries_host.resize(number_of_histories);
        queue.copy(secondary_summaries_device, secondary_summaries_host.data(),
                   number_of_histories)
            .wait_and_throw();
    }
    if (enable_secondary_transport) {
        fragment_dose_host.resize(fragment_species_count * number_of_bins);
        secondary_deposited_host.resize(secondary_queue_capacity);
        secondary_escaped_host.resize(secondary_queue_capacity);
        secondary_steps_host.resize(secondary_queue_capacity);
        queue.copy(fragment_dose_device, fragment_dose_host.data(),
                   fragment_dose_host.size());
        queue.copy(secondary_deposited_device, secondary_deposited_host.data(),
                   secondary_queue_capacity);
        queue.copy(secondary_escaped_device, secondary_escaped_host.data(),
                   secondary_queue_capacity);
        queue.copy(secondary_steps_device, secondary_steps_host.data(),
                   secondary_queue_capacity)
            .wait_and_throw();
        if (enable_fragment_cascade) {
            cascade_summaries_host.resize(secondary_queue_capacity);
            queue.copy(cascade_summaries_device, cascade_summaries_host.data(),
                       secondary_queue_capacity)
                .wait_and_throw();
        }
    }

    free_device(table_device);
    free_device(cross_section_device);
    free_device(dose_device);
    free_device(deposited_device);
    free_device(escaped_device);
    free_device(nuclear_device);
    free_device(steps_device);
    free_device(reaction_bins_device);
    free_device(reactions_device);
    free_device(reaction_secondaries_device);
    free_device(secondary_queue_device);
    free_device(secondary_queue_counter_device);
    free_device(secondary_queue_filled_device);
    free_device(secondary_summaries_device);
    free_device(fragment_dose_device);
    free_device(secondary_deposited_device);
    free_device(secondary_escaped_device);
    free_device(secondary_steps_device);
    free_device(cascade_projectiles_device);
    free_device(cascade_cross_sections_device);
    free_device(cascade_interactions_device);
    free_device(cascade_products_device);
    free_device(cascade_summaries_device);

    TransportResult result;
    result.backend = "sycl-" + device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    if (config.enable_primary_attenuation) {
        result.backend += "+attenuation";
    }
    if (enable_secondary_generation) {
        result.backend += "+secondary-generation";
    }
    if (enable_secondary_transport) {
        result.backend += "+secondary-transport";
    }
    if (enable_fragment_cascade) {
        result.backend += "+fragment-cascade";
    }
    result.primary_c12_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.initial_energy_MeV =
        config.initial_total_energy_MeV() * static_cast<double>(number_of_histories);
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);
    result.untracked_nuclear_energy_MeV =
        std::accumulate(nuclear_host.begin(), nuclear_host.end(), 0.0);
    result.nuclear_interactions = static_cast<std::uint64_t>(std::count_if(
        nuclear_host.begin(), nuclear_host.end(), [](float energy) { return energy > 0.0f; }));
    if (enable_secondary_generation) {
        result.sampled_reaction_packages = result.nuclear_interactions;
        for (const auto& summary : secondary_summaries_host) {
            result.generated_direct_secondaries += summary.direct_count;
            result.queued_secondaries += summary.queued_count;
            result.secondary_queue_overflow += summary.overflow_count;
            result.queued_secondary_energy_MeV += summary.queued_energy_MeV;
            result.secondary_queue_overflow_energy_MeV += summary.overflow_energy_MeV;
            result.untransported_neutral_energy_MeV += summary.neutral_energy_MeV;
            result.untransported_unsupported_charged_energy_MeV +=
                summary.unsupported_charged_energy_MeV;
        }
        result.generated_direct_secondary_energy_MeV =
            result.queued_secondary_energy_MeV +
            result.secondary_queue_overflow_energy_MeV +
            result.untransported_neutral_energy_MeV +
            result.untransported_unsupported_charged_energy_MeV;
        result.nuclear_energy_not_in_direct_secondaries_MeV =
            result.untracked_nuclear_energy_MeV -
            result.generated_direct_secondary_energy_MeV;
    }
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});
    if (enable_secondary_transport) {
        const auto extract_species = [&](std::size_t species_index) {
            const auto begin = fragment_dose_host.begin() +
                               static_cast<std::ptrdiff_t>(species_index * number_of_bins);
            return std::vector<double>(begin,
                                       begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.secondary_carbon_deposited_energy_MeV = extract_species(0);
        result.boron_deposited_energy_MeV = extract_species(1);
        result.beryllium_deposited_energy_MeV = extract_species(2);
        result.lithium_deposited_energy_MeV = extract_species(3);
        result.helium_deposited_energy_MeV = extract_species(4);
        result.proton_deposited_energy_MeV = extract_species(5);
        result.other_charged_deposited_energy_MeV = extract_species(6);
        const std::vector<const std::vector<double>*> fragment_species{
            &result.secondary_carbon_deposited_energy_MeV,
            &result.boron_deposited_energy_MeV,
            &result.beryllium_deposited_energy_MeV,
            &result.lithium_deposited_energy_MeV,
            &result.helium_deposited_energy_MeV,
            &result.proton_deposited_energy_MeV,
            &result.other_charged_deposited_energy_MeV,
        };
        for (const auto* species : fragment_species) {
            for (std::size_t bin = 0; bin < number_of_bins; ++bin) {
                result.deposited_energy_MeV[bin] += (*species)[bin];
            }
        }
        result.transported_secondaries = transported_queue_count;
        result.secondary_deposited_energy_MeV =
            std::accumulate(secondary_deposited_host.begin(),
                            secondary_deposited_host.end(), 0.0);
        result.secondary_escaped_energy_MeV =
            std::accumulate(secondary_escaped_host.begin(),
                            secondary_escaped_host.end(), 0.0);
        result.secondary_transport_steps =
            std::accumulate(secondary_steps_host.begin(), secondary_steps_host.end(),
                            std::uint64_t{0});
        result.total_deposited_energy_MeV += result.secondary_deposited_energy_MeV;
        result.escaped_energy_MeV += result.secondary_escaped_energy_MeV;
        result.untracked_nuclear_energy_MeV -= result.queued_secondary_energy_MeV;
        if (enable_fragment_cascade) {
            for (std::size_t index = 0;
                 index < static_cast<std::size_t>(transported_queue_count); ++index) {
                const auto& summary = cascade_summaries_host[index];
                result.cascade_interactions += summary.interaction_count;
                result.generated_cascade_products += summary.direct_count;
                result.queued_cascade_secondaries += summary.queued_count;
                result.cascade_queue_overflow += summary.overflow_count;
                result.queued_cascade_energy_MeV += summary.queued_energy_MeV;
                result.cascade_nuclear_energy_MeV += summary.incident_energy_MeV;
            }
            result.untracked_nuclear_energy_MeV +=
                result.cascade_nuclear_energy_MeV - result.queued_cascade_energy_MeV;
        }
        result.total_steps += result.secondary_transport_steps;
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

}  // namespace carbon

#endif
