#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace carbon {

// Global dose scorer scalar for device atomics / USM dose buffers.
// CARBON_DOSE_FP32=1 (CMake) uses float atomics — much faster on consumer NVIDIA
// GPUs (Turing FP64 is ~1/32 rate). Host TransportResult remains double.
#if defined(CARBON_DOSE_FP32)
using DoseAtomicT = float;
inline constexpr bool k_dose_atomic_fp32 = true;
#else
using DoseAtomicT = double;
inline constexpr bool k_dose_atomic_fp32 = false;
#endif

// Device-side profile increment. Compiles to nothing unless
// CARBON_TRANSPORT_PROFILE is defined at build time.
inline void profile_add(std::uint64_t* counters,
                        const TransportProfileSlot slot,
                        const std::uint64_t n = 1) {
#ifdef CARBON_TRANSPORT_PROFILE
    if (counters == nullptr || n == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic_counter(counters[static_cast<std::size_t>(slot)]);
    atomic_counter.fetch_add(n);
#else
    (void)counters;
    (void)slot;
    (void)n;
#endif
}

inline void profile_face(std::uint64_t* counters,
                         const bool is_primary,
                         const CtClampPath path) {
#ifdef CARBON_TRANSPORT_PROFILE
    profile_add(counters, is_primary ? TransportProfileSlot::primary_face_clamp_calls
                                     : TransportProfileSlot::secondary_face_clamp_calls);
    profile_add(counters, is_primary ? primary_face_slot(path) : secondary_face_slot(path));
#else
    (void)counters;
    (void)is_primary;
    (void)path;
#endif
}

struct SyclTransportContext::Impl {
    explicit Impl(const std::string& requested_device_name)
        : device_name(requested_device_name), queue(make_sycl_queue(requested_device_name)) {}

    ~Impl() { clear(); }

    void clear() noexcept {
        const auto release = [this](auto*& pointer) {
            if (pointer != nullptr) {
                sycl::free(pointer, queue);
                pointer = nullptr;
            }
        };
        release(table_device);
        release(cross_section_device);
        release(reaction_bins_device);
        release(reactions_device);
        release(reaction_secondaries_device);
        release(cascade_projectiles_device);
        release(cascade_cross_sections_device);
        release(cascade_interactions_device);
        release(cascade_products_device);
        release(neutral_projectiles_device);
        release(neutral_cross_sections_device);
        release(neutral_interactions_device);
        release(neutral_products_device);
        initialized = false;
    }

    void ensure_initialized(const StoppingPowerTable& stopping_power,
                            const CrossSectionTable& cross_section,
                            const ReactionPackageTable* reaction_packages,
                            const CascadePackageTable* cascade_packages,
                            const NeutralPackageTable* neutral_packages) {
        if (initialized) {
            if (stopping_power_host != &stopping_power ||
                cross_section_host != &cross_section || reaction_packages_host != reaction_packages ||
                cascade_packages_host != cascade_packages ||
                neutral_packages_host != neutral_packages) {
                throw std::invalid_argument(
                    "SyclTransportContext cannot be reused with different physics tables");
            }
            return;
        }

        stopping_power_host = &stopping_power;
        cross_section_host = &cross_section;
        reaction_packages_host = reaction_packages;
        cascade_packages_host = cascade_packages;
        neutral_packages_host = neutral_packages;

        try {
            table_device = sycl::malloc_device<float>(stopping_power.values().size(), queue);
            cross_section_device =
                sycl::malloc_device<float>(cross_section.values().size(), queue);
            if (reaction_packages != nullptr) {
                reaction_bins_device = sycl::malloc_device<ReactionEnergyBin>(
                    reaction_packages->energy_bins().size(), queue);
                reactions_device = sycl::malloc_device<ReactionPackage>(
                    reaction_packages->reactions().size(), queue);
                reaction_secondaries_device = sycl::malloc_device<ReactionSecondary>(
                    reaction_packages->secondaries().size(), queue);
            }
            if (cascade_packages != nullptr) {
                cascade_projectiles_device = sycl::malloc_device<CascadeProjectile>(
                    cascade_packages->projectiles().size(), queue);
                cascade_cross_sections_device = sycl::malloc_device<CascadeCrossSectionSample>(
                    cascade_packages->cross_sections().size(), queue);
                cascade_interactions_device = sycl::malloc_device<CascadeInteraction>(
                    cascade_packages->interactions().size(), queue);
                cascade_products_device = sycl::malloc_device<ReactionSecondary>(
                    cascade_packages->products().size(), queue);
            }
            if (neutral_packages != nullptr) {
                neutral_projectiles_device = sycl::malloc_device<NeutralProjectile>(
                    neutral_packages->projectiles().size(), queue);
                neutral_cross_sections_device = sycl::malloc_device<NeutralCrossSectionSample>(
                    neutral_packages->cross_sections().size(), queue);
                neutral_interactions_device = sycl::malloc_device<NeutralInteraction>(
                    neutral_packages->interactions().size(), queue);
                neutral_products_device = sycl::malloc_device<ReactionSecondary>(
                    neutral_packages->products().size(), queue);
            }

            const auto allocation_failed =
                table_device == nullptr || cross_section_device == nullptr ||
                (reaction_packages != nullptr &&
                 (reaction_bins_device == nullptr || reactions_device == nullptr ||
                  reaction_secondaries_device == nullptr)) ||
                (cascade_packages != nullptr &&
                 (cascade_projectiles_device == nullptr ||
                  cascade_cross_sections_device == nullptr ||
                  cascade_interactions_device == nullptr || cascade_products_device == nullptr)) ||
                (neutral_packages != nullptr &&
                 (neutral_projectiles_device == nullptr ||
                  neutral_cross_sections_device == nullptr ||
                  neutral_interactions_device == nullptr || neutral_products_device == nullptr));
            if (allocation_failed) {
                throw std::bad_alloc();
            }

            std::vector<float> table_host(stopping_power.values().size());
            std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                           table_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            std::vector<float> xs_host(cross_section.values().size());
            std::transform(cross_section.values().begin(), cross_section.values().end(),
                           xs_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(table_host.data(), table_device, table_host.size());
            queue.copy(xs_host.data(), cross_section_device, xs_host.size());
            if (reaction_packages != nullptr) {
                queue.copy(reaction_packages->energy_bins().data(), reaction_bins_device,
                           reaction_packages->energy_bins().size());
                queue.copy(reaction_packages->reactions().data(), reactions_device,
                           reaction_packages->reactions().size());
                queue.copy(reaction_packages->secondaries().data(), reaction_secondaries_device,
                           reaction_packages->secondaries().size());
            }
            if (cascade_packages != nullptr) {
                queue.copy(cascade_packages->projectiles().data(), cascade_projectiles_device,
                           cascade_packages->projectiles().size());
                queue.copy(cascade_packages->cross_sections().data(),
                           cascade_cross_sections_device,
                           cascade_packages->cross_sections().size());
                queue.copy(cascade_packages->interactions().data(), cascade_interactions_device,
                           cascade_packages->interactions().size());
                queue.copy(cascade_packages->products().data(), cascade_products_device,
                           cascade_packages->products().size());
            }
            if (neutral_packages != nullptr) {
                queue.copy(neutral_packages->projectiles().data(), neutral_projectiles_device,
                           neutral_packages->projectiles().size());
                queue.copy(neutral_packages->cross_sections().data(),
                           neutral_cross_sections_device,
                           neutral_packages->cross_sections().size());
                queue.copy(neutral_packages->interactions().data(), neutral_interactions_device,
                           neutral_packages->interactions().size());
                queue.copy(neutral_packages->products().data(), neutral_products_device,
                           neutral_packages->products().size());
            }
            queue.wait_and_throw();
            initialized = true;
        } catch (...) {
            clear();
            throw;
        }
    }

    std::string device_name;
    sycl::queue queue;
    bool initialized{false};
    const StoppingPowerTable* stopping_power_host{nullptr};
    const CrossSectionTable* cross_section_host{nullptr};
    const ReactionPackageTable* reaction_packages_host{nullptr};
    const CascadePackageTable* cascade_packages_host{nullptr};
    const NeutralPackageTable* neutral_packages_host{nullptr};
    float* table_device{nullptr};
    float* cross_section_device{nullptr};
    ReactionEnergyBin* reaction_bins_device{nullptr};
    ReactionPackage* reactions_device{nullptr};
    ReactionSecondary* reaction_secondaries_device{nullptr};
    CascadeProjectile* cascade_projectiles_device{nullptr};
    CascadeCrossSectionSample* cascade_cross_sections_device{nullptr};
    CascadeInteraction* cascade_interactions_device{nullptr};
    ReactionSecondary* cascade_products_device{nullptr};
    NeutralProjectile* neutral_projectiles_device{nullptr};
    NeutralCrossSectionSample* neutral_cross_sections_device{nullptr};
    NeutralInteraction* neutral_interactions_device{nullptr};
    ReactionSecondary* neutral_products_device{nullptr};
};

SyclTransportContext::SyclTransportContext(const std::string& device_name)
    : impl_(std::make_unique<Impl>(device_name)) {}

SyclTransportContext::~SyclTransportContext() = default;
SyclTransportContext::SyclTransportContext(SyclTransportContext&&) noexcept = default;
SyclTransportContext& SyclTransportContext::operator=(SyclTransportContext&&) noexcept = default;

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

double event_duration_seconds(const sycl::event& event) {
    const auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(end - start) * 1.0e-9;
}

struct Direction3F {
    float x;
    float y;
    float z;
};

Direction3F rotate_local_direction(const float local_x,
                                   const float local_y,
                                   const float local_z,
                                   const Direction3F parent_direction) noexcept {
    if (!sycl::isfinite(local_x) || !sycl::isfinite(local_y)) {
        return Direction3F{0.0F, 0.0F,
                           sycl::clamp(local_z, -1.0F, 1.0F) *
                               (parent_direction.z < 0.0F ? -1.0F : 1.0F)};
    }

    const auto parent_norm = sycl::sqrt(parent_direction.x * parent_direction.x +
                                        parent_direction.y * parent_direction.y +
                                        parent_direction.z * parent_direction.z);
    const auto inverse_parent_norm = parent_norm > 0.0F ? 1.0F / parent_norm : 1.0F;
    const Direction3F w{parent_direction.x * inverse_parent_norm,
                        parent_direction.y * inverse_parent_norm,
                        parent_norm > 0.0F ? parent_direction.z * inverse_parent_norm : 1.0F};
    const Direction3F reference =
        sycl::fabs(w.x) < 0.9F ? Direction3F{1.0F, 0.0F, 0.0F}
                               : Direction3F{0.0F, 1.0F, 0.0F};
    const auto projection = reference.x * w.x + reference.y * w.y + reference.z * w.z;
    Direction3F u{reference.x - projection * w.x,
                  reference.y - projection * w.y,
                  reference.z - projection * w.z};
    const auto inverse_u_norm =
        1.0F / sycl::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = Direction3F{u.x * inverse_u_norm, u.y * inverse_u_norm, u.z * inverse_u_norm};
    const Direction3F v{w.y * u.z - w.z * u.y,
                        w.z * u.x - w.x * u.z,
                        w.x * u.y - w.y * u.x};
    Direction3F output{local_x * u.x + local_y * v.x + local_z * w.x,
                       local_x * u.y + local_y * v.y + local_z * w.y,
                       local_x * u.z + local_y * v.z + local_z * w.z};
    const auto inverse_output_norm =
        1.0F / sycl::sqrt(output.x * output.x + output.y * output.y + output.z * output.z);
    output = Direction3F{output.x * inverse_output_norm,
                         output.y * inverse_output_norm,
                         output.z * inverse_output_norm};
    return output;
}

float highland_projected_rms_angle_device(const float kinetic_energy_MeV,
                                          const int atomic_number,
                                          const int mass_number,
                                          const float path_length_mm,
                                          const float density_g_per_cm3) noexcept {
    if (kinetic_energy_MeV <= 0.0F || atomic_number <= 0 || mass_number <= 0 ||
        path_length_mm <= 0.0F || density_g_per_cm3 <= 0.0F) {
        return 0.0F;
    }
    const auto energy_MeV_per_u = kinetic_energy_MeV / static_cast<float>(mass_number);
    const auto total_energy_MeV_per_u =
        energy_MeV_per_u + static_cast<float>(nucleon_rest_mass_MeV);
    const auto momentum_MeV_per_c_per_u = sycl::sqrt(
        energy_MeV_per_u *
        (energy_MeV_per_u + 2.0F * static_cast<float>(nucleon_rest_mass_MeV)));
    const auto beta = momentum_MeV_per_c_per_u / total_energy_MeV_per_u;
    const auto momentum_MeV_per_c =
        static_cast<float>(mass_number) * momentum_MeV_per_c_per_u;
    const auto radiation_lengths =
        density_g_per_cm3 * (path_length_mm / 10.0F) /
        static_cast<float>(water_radiation_length_g_per_cm2);
    if (beta <= 0.0F || momentum_MeV_per_c <= 0.0F || radiation_lengths <= 0.0F) {
        return 0.0F;
    }
    const auto charge = static_cast<float>(atomic_number);
    const auto logarithm_argument =
        radiation_lengths * charge * charge / (beta * beta);
    const auto correction =
        sycl::fmax(0.0F, 1.0F + 0.038F * sycl::log(logarithm_argument));
    return static_cast<float>(highland_energy_constant_MeV) * charge /
           (beta * momentum_MeV_per_c) * sycl::sqrt(radiation_lengths) * correction;
}

Direction3F scatter_direction(const Direction3F direction,
                              const float projected_rms_angle_rad,
                              const std::uint64_t seed,
                              const std::uint64_t history_id,
                              const std::uint64_t step_index,
                              const std::uint32_t random_dimension) noexcept {
    if (projected_rms_angle_rad <= 0.0F) {
        return direction;
    }
    const auto uniform1 = sycl::fmax(
        rng::uniform01(seed, history_id, step_index, random_dimension), 1.0e-12F);
    const auto uniform2 =
        rng::uniform01(seed, history_id, step_index, random_dimension + 1U);
    constexpr float two_pi = 6.2831853071795864769F;
    const auto radius = sycl::sqrt(-2.0F * sycl::log(uniform1));
    const auto local_x = projected_rms_angle_rad * radius * sycl::cos(two_pi * uniform2);
    const auto local_y = projected_rms_angle_rad * radius * sycl::sin(two_pi * uniform2);
    return rotate_local_direction(local_x, local_y, 1.0F, direction);
}

// Deposit energy along +z with exponential attenuation from z0.
// lambda_mm <= 0 → local dump in the production bin.
// renormalize=true → scale so full amount is deposited in remaining phantom
//   (used for interim neutral kerma; avoids high-E tail escape underdose).
// renormalize=false → omit energy past phantom (electronic delta near exit).
inline void score_exponential_depth(
    const float amount_MeV,
    const float z_mm,
    const float bin_width_mm,
    const float phantom_length_mm,
    const std::uint32_t number_of_bins,
    const float lambda_mm,
    DoseAtomicT* dose_row,
    const bool renormalize = false) noexcept {
    if (amount_MeV <= 0.0F || dose_row == nullptr || number_of_bins == 0 ||
        bin_width_mm <= 0.0F) {
        return;
    }
    auto z0 = z_mm;
    if (z0 < 0.0F) {
        z0 = 0.0F;
    }
    if (z0 >= phantom_length_mm) {
        return;
    }
    if (lambda_mm <= 1.0e-3F) {
        auto bin = static_cast<int>(z0 / bin_width_mm);
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= static_cast<int>(number_of_bins)) {
            bin = static_cast<int>(number_of_bins) - 1;
        }
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(dose_row[static_cast<std::size_t>(bin)]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV));
        return;
    }
    const auto inv_lambda = 1.0F / lambda_mm;
    const auto remaining = phantom_length_mm - z0;
    auto norm = 1.0F;
    if (renormalize) {
        const auto contained = 1.0F - sycl::exp(-remaining * inv_lambda);
        if (contained > 1.0e-6F) {
            norm = 1.0F / contained;
        }
    }
    const auto start_bin = static_cast<std::uint32_t>(z0 / bin_width_mm);
    for (std::uint32_t bin = start_bin; bin < number_of_bins; ++bin) {
        const auto z_lo =
            sycl::fmax(z0, static_cast<float>(bin) * bin_width_mm);
        const auto z_hi = sycl::fmin(phantom_length_mm,
                                     static_cast<float>(bin + 1U) * bin_width_mm);
        if (z_hi <= z_lo) {
            continue;
        }
        const auto s_lo = z_lo - z0;
        const auto s_hi = z_hi - z0;
        // ∫_{s_lo}^{s_hi} (1/λ) exp(-s/λ) ds
        const auto frac =
            (sycl::exp(-s_lo * inv_lambda) - sycl::exp(-s_hi * inv_lambda)) * norm;
        if (frac <= 0.0F) {
            continue;
        }
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(dose_row[bin]);
        atomic_dose.fetch_add(static_cast<DoseAtomicT>(amount_MeV * frac));
        if (sycl::exp(-s_hi * inv_lambda) < 1.0e-5F) {
            break;
        }
    }
}

// Voxel analogue of score_exponential_depth. The proxy keeps the production
// voxel's transverse (x,y) index and distributes energy only along +z.
inline void score_exponential_voxel_depth(
    const float amount_MeV,
    const float z_mm,
    const float bin_width_mm,
    const float phantom_length_mm,
    const std::uint32_t number_of_bins,
    const float lambda_mm,
    const std::size_t voxel_plane_size,
    const std::size_t production_voxel_index,
    double* voxel_dose,
    const bool renormalize = false) noexcept {
    if (amount_MeV <= 0.0F || voxel_dose == nullptr || number_of_bins == 0 ||
        voxel_plane_size == 0 || bin_width_mm <= 0.0F) {
        return;
    }
    auto z0 = z_mm;
    if (z0 < 0.0F) {
        z0 = 0.0F;
    }
    if (z0 >= phantom_length_mm) {
        return;
    }
    const auto voxel_in_plane = production_voxel_index % voxel_plane_size;
    if (lambda_mm <= 1.0e-3F) {
        auto bin = static_cast<int>(z0 / bin_width_mm);
        bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(voxel_dose[static_cast<std::size_t>(bin) * voxel_plane_size +
                                   voxel_in_plane]);
        atomic_dose.fetch_add(static_cast<double>(amount_MeV));
        return;
    }
    const auto inv_lambda = 1.0F / lambda_mm;
    const auto remaining = phantom_length_mm - z0;
    auto norm = 1.0F;
    if (renormalize) {
        const auto contained = 1.0F - sycl::exp(-remaining * inv_lambda);
        if (contained > 1.0e-6F) {
            norm = 1.0F / contained;
        }
    }
    const auto start_bin = static_cast<std::uint32_t>(z0 / bin_width_mm);
    for (std::uint32_t bin = start_bin; bin < number_of_bins; ++bin) {
        const auto z_lo = sycl::fmax(z0, static_cast<float>(bin) * bin_width_mm);
        const auto z_hi = sycl::fmin(phantom_length_mm,
                                     static_cast<float>(bin + 1U) * bin_width_mm);
        if (z_hi <= z_lo) {
            continue;
        }
        const auto s_lo = z_lo - z0;
        const auto s_hi = z_hi - z0;
        const auto frac =
            (sycl::exp(-s_lo * inv_lambda) - sycl::exp(-s_hi * inv_lambda)) * norm;
        if (frac <= 0.0F) {
            continue;
        }
        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_dose(voxel_dose[static_cast<std::size_t>(bin) * voxel_plane_size +
                                   voxel_in_plane]);
        atomic_dose.fetch_add(static_cast<double>(amount_MeV * frac));
        if (sycl::exp(-s_hi * inv_lambda) < 1.0e-5F) {
            break;
        }
    }
}

// Energy-dependent electronic delta fraction (0 at low E → full scale at 400 MeV/u).
inline float electronic_buildup_fraction_at_energy(
    const float energy_MeVu, const float fraction_at_400) noexcept {
    if (fraction_at_400 <= 0.0F) {
        return 0.0F;
    }
    const auto scale = energy_MeVu > 0.0F ? energy_MeVu / 400.0F : 0.0F;
    const auto clamped = scale < 1.0F ? scale : 1.0F;
    return fraction_at_400 * clamped;
}

// Kerma fraction: f0 for E<=200 MeV/u; linear ramp to f0*high_scale at E>=400.
inline float neutral_kerma_fraction_at_energy(const float energy_MeVu,
                                               const float f0,
                                               const float high_scale) noexcept {
    if (f0 <= 0.0F) {
        return 0.0F;
    }
    if (high_scale <= 1.0F || energy_MeVu <= 200.0F) {
        return f0;
    }
    auto t = (energy_MeVu - 200.0F) / 200.0F;
    if (t > 1.0F) {
        t = 1.0F;
    }
    return f0 * (1.0F + (high_scale - 1.0F) * t);
}

inline std::uint32_t cascade_cross_section_lower_bound(
    const CascadeCrossSectionSample* samples,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (samples[offset + middle].energy_MeV_per_u < energy_MeV_per_u) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

// Host: interpolate each cascade projectile's XS samples onto the water SP energy
// grid so the device kernel only does clamp/index/lerp (no binary search per step).
std::vector<float> build_cascade_xs_energy_lut(
    const CascadePackageTable& packages,
    const std::vector<double>& energies_MeVu) {
    const auto projectile_count = packages.projectiles().size();
    const auto energy_count = energies_MeVu.size();
    std::vector<float> lut(projectile_count * energy_count, 0.0F);
    const auto& samples = packages.cross_sections();
    for (std::size_t projectile_index = 0; projectile_index < projectile_count;
         ++projectile_index) {
        const auto& projectile = packages.projectiles()[projectile_index];
        const auto offset = projectile.cross_section_offset;
        const auto count = projectile.cross_section_count;
        if (count == 0) {
            continue;
        }
        for (std::size_t energy_index = 0; energy_index < energy_count; ++energy_index) {
            const auto energy = static_cast<float>(energies_MeVu[energy_index]);
            std::uint32_t lower = 0;
            std::uint32_t upper = count;
            while (lower < upper) {
                const auto middle = lower + (upper - lower) / 2U;
                if (samples[offset + middle].energy_MeV_per_u < energy) {
                    lower = middle + 1U;
                } else {
                    upper = middle;
                }
            }
            float value = 0.0F;
            if (lower == 0U) {
                value = samples[offset].macroscopic_cross_section_per_mm;
            } else if (lower >= count) {
                value = samples[offset + count - 1U].macroscopic_cross_section_per_mm;
            } else {
                const auto& lo = samples[offset + lower - 1U];
                const auto& hi = samples[offset + lower];
                const auto interval = hi.energy_MeV_per_u - lo.energy_MeV_per_u;
                const auto frac =
                    interval > 0.0F
                        ? std::clamp((energy - lo.energy_MeV_per_u) / interval, 0.0F, 1.0F)
                        : 0.0F;
                value = lo.macroscopic_cross_section_per_mm +
                        frac * (hi.macroscopic_cross_section_per_mm -
                                lo.macroscopic_cross_section_per_mm);
            }
            lut[projectile_index * energy_count + energy_index] = value;
        }
    }
    return lut;
}

inline int cascade_projectile_index(const CascadeProjectile* projectiles,
                                    const std::size_t count,
                                    const int atomic_number,
                                    const int mass_number) noexcept {
    std::size_t lower = 0;
    std::size_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        const auto projectile = projectiles[middle];
        const auto precedes_target =
            projectile.atomic_number < atomic_number ||
            (projectile.atomic_number == atomic_number &&
             projectile.mass_number < mass_number);
        if (precedes_target) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    if (lower < count && projectiles[lower].atomic_number == atomic_number &&
        projectiles[lower].mass_number == mass_number) {
        return static_cast<int>(lower);
    }
    return -1;
}

inline std::uint32_t cascade_interaction_lower_bound(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (interactions[offset + middle].incident_energy_MeV_per_u < energy_MeV_per_u) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t nearest_cascade_interaction(
    const CascadeInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV_per_u) noexcept {
    const auto lower =
        cascade_interaction_lower_bound(interactions, offset, count, energy_MeV_per_u);
    if (lower == 0U) {
        return 0U;
    }
    if (lower >= count) {
        return count - 1U;
    }
    const auto lower_delta = sycl::fabs(
        interactions[offset + lower - 1U].incident_energy_MeV_per_u - energy_MeV_per_u);
    const auto upper_delta = sycl::fabs(
        interactions[offset + lower].incident_energy_MeV_per_u - energy_MeV_per_u);
    return upper_delta < lower_delta ? lower : lower - 1U;
}

inline std::uint32_t neutral_cross_section_lower_bound(
    const NeutralCrossSectionSample* samples,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (samples[offset + middle].energy_MeV < energy_MeV) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t neutral_interaction_lower_bound(
    const NeutralInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        if (interactions[offset + middle].incident_energy_MeV < energy_MeV) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

inline std::uint32_t nearest_neutral_interaction(
    const NeutralInteraction* interactions,
    const std::uint32_t offset,
    const std::uint32_t count,
    const float energy_MeV) noexcept {
    const auto lower =
        neutral_interaction_lower_bound(interactions, offset, count, energy_MeV);
    if (lower == 0U) {
        return 0U;
    }
    if (lower >= count) {
        return count - 1U;
    }
    const auto lower_delta = sycl::fabs(
        interactions[offset + lower - 1U].incident_energy_MeV - energy_MeV);
    const auto upper_delta =
        sycl::fabs(interactions[offset + lower].incident_energy_MeV - energy_MeV);
    return upper_delta < lower_delta ? lower : lower - 1U;
}

void score_secondary_dose_device(
    const DoseAtomicT amount_MeV,
    const bool is_neutral_lineage,
    const std::size_t species_index,
    const std::size_t neutral_origin,
    const int bin,
    const std::size_t number_of_bins,
    const bool enable_voxel_scoring,
    const bool enable_charged_origin_voxel_scoring,
    const std::size_t voxel_index,
    const std::size_t charged_origin_voxel_offset,
    const std::size_t neutral_origin_voxel_offset,
    DoseAtomicT* fragment_dose_device,
    DoseAtomicT* voxel_dose_device,
    DoseAtomicT* charged_origin_voxel_dose_device,
    DoseAtomicT* neutral_origin_dose_device,
    DoseAtomicT* neutral_origin_voxel_dose_device) noexcept {
    if (amount_MeV <= DoseAtomicT{0}) {
        return;
    }
    if (is_neutral_lineage) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_origin(neutral_origin_dose_device[neutral_origin * number_of_bins +
                                                     static_cast<std::size_t>(bin)]);
        atomic_origin.fetch_add(amount_MeV);
        if (enable_voxel_scoring && neutral_origin_voxel_dose_device != nullptr) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_voxel(
                    neutral_origin_voxel_dose_device[neutral_origin_voxel_offset +
                                                     voxel_index]);
            atomic_voxel.fetch_add(amount_MeV);
        }
        // Neutral-origin dose is a category split, not a replacement for the
        // aggregate voxel scorer used to write the total 3D dose.
        if (enable_voxel_scoring && voxel_dose_device != nullptr) {
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_total_voxel(voxel_dose_device[voxel_index]);
            atomic_total_voxel.fetch_add(amount_MeV);
        }
        return;
    }
    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic_fragment(
            fragment_dose_device[species_index * number_of_bins +
                                 static_cast<std::size_t>(bin)]);
    atomic_fragment.fetch_add(amount_MeV);
    if (enable_voxel_scoring) {
        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_voxel(voxel_dose_device[voxel_index]);
        atomic_voxel.fetch_add(amount_MeV);
        if (enable_charged_origin_voxel_scoring) {
            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_category(
                    charged_origin_voxel_dose_device[charged_origin_voxel_offset +
                                                     voxel_index]);
            atomic_category.fetch_add(amount_MeV);
        }
    }
}
}  // namespace

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               const ReactionPackageTable* reaction_packages,
                               const CascadePackageTable* cascade_packages,
                               const NeutralPackageTable* neutral_packages,
                               SyclTransportContext* context) {
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
    if (config.enable_neutral_transport && neutral_packages == nullptr) {
        throw std::invalid_argument("Neutral transport requires a neutral package table");
    }

    if (context != nullptr && context->impl_->device_name != device_name) {
        throw std::invalid_argument(
            "SyclTransportContext device does not match transport device");
    }
    auto queue = context != nullptr ? context->impl_->queue : make_sycl_queue(device_name);
    const auto reuse_immutable_buffers = context != nullptr;
    const auto start = std::chrono::steady_clock::now();
    auto primary_kernel_seconds = 0.0;
    auto secondary_kernel_seconds = 0.0;
    auto neutral_kernel_seconds = 0.0;
    auto charged_after_neutral_kernel_seconds = 0.0;
    const auto table_size = stopping_power.values().size();
    const auto cross_section_table_size = cross_section.values().size();
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_histories = config.number_of_histories;
    const auto primary_spot_count = config.primary_spot_batch.size();
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_charged_origin_voxel_scoring =
        config.enable_charged_origin_voxel_scoring;
    const auto number_of_voxels =
        enable_voxel_scoring ? config.number_of_voxels() : std::size_t{0};
    const auto voxel_plane_size = config.voxel_bins_x * config.voxel_bins_y;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    // Default: origin-centered scorer (homogeneous / water phantoms).
    // Extent is always [min, min + n * size); do not use max = -min (wrong for
    // odd bin counts when later rebased onto a CT origin).
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    const auto enable_secondary_generation = config.enable_secondary_generation;
    const auto enable_secondary_transport = config.enable_secondary_transport;
    const auto enable_fragment_cascade = config.enable_fragment_cascade;
    const auto enable_neutral_transport = config.enable_neutral_transport;
    const auto neutral_allow_continuation =
        enable_neutral_transport && config.neutral_transport_mode == "full";
    const auto automatic_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() / 16
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * 16;
    auto secondary_queue_capacity =
        config.secondary_queue_capacity == 0 ? automatic_queue_capacity
                                             : config.secondary_queue_capacity;
    // Production SOBP measurements use about 1.8 neutral slots/history for
    // first-interaction mode and 2.4 for two generations. Preserve explicit
    // headroom without allocating and clearing 32 slots/history for every spot.
    const std::size_t automatic_neutral_slots_per_history =
        neutral_allow_continuation ? 8U : 4U;
    const auto automatic_neutral_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() /
                                  automatic_neutral_slots_per_history
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * automatic_neutral_slots_per_history;
    auto neutral_queue_capacity =
        config.neutral_queue_capacity == 0 ? automatic_neutral_queue_capacity
                                           : config.neutral_queue_capacity;
    if (enable_secondary_generation &&
        secondary_queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Secondary queue capacity exceeds the uint32 runtime limit");
    }
    if (enable_neutral_transport &&
        neutral_queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Neutral queue capacity exceeds the uint32 runtime limit");
    }

    const auto& device = queue.get_device();
    if constexpr (!k_dose_atomic_fp32) {
        if (!device.has(sycl::aspect::fp64) || !device.has(sycl::aspect::atomic64)) {
            throw std::runtime_error(
                "The FP64 SYCL dose scorer requires fp64 and atomic64 device aspects "
                "(rebuild with -DCARBON_DOSE_FP32=ON for float atomics)");
        }
    }

    // CUDA (especially under WSL2) cannot host multi-minute single kernels: the
    // WDDM/TDR path and shared GPU stack freeze the whole distro, and USM often
    // reserves a huge host VA. Clamp queues early so estimates/allocations stay
    // modest. Intel Level Zero keeps the original larger defaults.
    const bool is_cuda_backend =
        device.get_backend() == sycl::backend::ext_oneapi_cuda;
    if (is_cuda_backend) {
        // Hard ceiling only for pathological sizes; real limit is the device
        // memory budget below. SOBP 10M + cascade needs ~3–4 slots/history
        // (~30–40M) when launched as one batched plan.
        constexpr std::size_t kCudaMaxSecondarySlots = 64ULL * 1024ULL * 1024ULL;
        constexpr std::size_t kCudaMaxNeutralSlots = 16ULL * 1024ULL * 1024ULL;
        const auto cuda_auto_secondary =
            std::max(number_of_histories * 4U, std::size_t{8192});
        if (config.secondary_queue_capacity == 0) {
            secondary_queue_capacity =
                std::min(secondary_queue_capacity, cuda_auto_secondary);
        }
        secondary_queue_capacity =
            std::min(secondary_queue_capacity, kCudaMaxSecondarySlots);
        if (config.neutral_queue_capacity == 0) {
            neutral_queue_capacity =
                std::min(neutral_queue_capacity,
                         std::max(number_of_histories * 2U, std::size_t{4096}));
        }
        neutral_queue_capacity =
            std::min(neutral_queue_capacity, kCudaMaxNeutralSlots);
        std::cout << "CUDA backend detected: WSL-safe queue caps "
                  << "secondary_queue_capacity=" << secondary_queue_capacity
                  << " neutral_queue_capacity=" << neutral_queue_capacity << '\n'
                  << std::flush;
    }

    // Soft device-memory budget: keep estimated USM under max_device_memory_fraction
    // of global memory (default 80%). Shrink secondary/neutral queues first.
    const auto device_global_bytes =
        device.get_info<sycl::info::device::global_mem_size>();
    if (device_global_bytes == 0) {
        throw std::runtime_error(
            "SYCL device reported global_mem_size=0; refusing to allocate");
    }
    // Soft clamp only for extreme YAML values; large SOBP needs >35% for queues.
    auto memory_fraction = config.max_device_memory_fraction;
    if (is_cuda_backend && memory_fraction > 0.70) {
        memory_fraction = 0.70;
        std::cout << "CUDA backend: clamping max_device_memory_fraction to 0.70\n"
                  << std::flush;
    }
    const auto memory_budget_bytes = static_cast<std::size_t>(
        static_cast<double>(device_global_bytes) *
        std::min(1.0, std::max(0.05, memory_fraction)));

    const auto estimate_device_bytes = [&](std::size_t sec_cap,
                                          std::size_t neu_cap) -> std::size_t {
        std::size_t bytes = 0;
        bytes += table_size * sizeof(float);
        bytes += cross_section_table_size * sizeof(float);
        bytes += number_of_bins * sizeof(DoseAtomicT);
        bytes += number_of_histories * (3 * sizeof(float) + sizeof(std::uint32_t));
        bytes += primary_spot_count * sizeof(PrimarySpotBatchEntry);
        if (enable_voxel_scoring) {
            bytes += number_of_voxels * sizeof(DoseAtomicT);
        }
        if (enable_charged_origin_voxel_scoring) {
            bytes += charged_origin_category_count * number_of_voxels * sizeof(DoseAtomicT);
        }
        if (enable_secondary_generation && reaction_packages != nullptr) {
            bytes += reaction_packages->energy_bins().size() * sizeof(ReactionEnergyBin);
            bytes += reaction_packages->reactions().size() * sizeof(ReactionPackage);
            bytes += reaction_packages->secondaries().size() * sizeof(ReactionSecondary);
            bytes += sec_cap * sizeof(SecondaryParticle3D);
            bytes += 2 * sizeof(std::uint64_t);
            bytes += number_of_histories * sizeof(SecondaryGenerationSummary);
            if (enable_secondary_transport) {
                bytes += fragment_species_count * number_of_bins * sizeof(DoseAtomicT);
                bytes += sec_cap * (2 * sizeof(float) + sizeof(std::uint32_t));
            }
            if (enable_fragment_cascade && cascade_packages != nullptr) {
                bytes += cascade_packages->projectiles().size() * sizeof(CascadeProjectile);
                bytes += cascade_packages->cross_sections().size() *
                         sizeof(CascadeCrossSectionSample);
                bytes += cascade_packages->interactions().size() * sizeof(CascadeInteraction);
                bytes += cascade_packages->products().size() * sizeof(ReactionSecondary);
                bytes += sec_cap * sizeof(CascadeTransportSummary);
            }
        }
        if (enable_neutral_transport && neutral_packages != nullptr) {
            bytes += neutral_packages->projectiles().size() * sizeof(NeutralProjectile);
            bytes += neutral_packages->cross_sections().size() *
                     sizeof(NeutralCrossSectionSample);
            bytes += neutral_packages->interactions().size() * sizeof(NeutralInteraction);
            bytes += neutral_packages->products().size() * sizeof(ReactionSecondary);
            bytes += neu_cap * sizeof(NeutralParticle3D);
            bytes += 2 * sizeof(std::uint64_t);
            bytes += neu_cap * sizeof(NeutralTransportSummary);
            bytes += neutral_origin_category_count * number_of_bins * sizeof(DoseAtomicT);
            if (enable_voxel_scoring) {
                bytes += neutral_origin_category_count * number_of_voxels * sizeof(DoseAtomicT);
            }
        }
        if (config.enable_ct_grid) {
            // Conservative upper bound until grid is loaded below.
            bytes += 64ULL * 1024ULL * 1024ULL;
        }
        // Driver / allocator overhead headroom.
        bytes = bytes + bytes / 8;
        return bytes;
    };

    {
        auto estimated = estimate_device_bytes(secondary_queue_capacity, neutral_queue_capacity);
        if (estimated > memory_budget_bytes) {
            const auto fixed = estimate_device_bytes(0, 0);
            if (fixed >= memory_budget_bytes) {
                throw std::runtime_error(
                    "Device memory budget exceeded by fixed buffers alone (tables/packages/"
                    "histories). Reduce histories, disable voxels, or raise "
                    "max_device_memory_fraction (device=" +
                    std::to_string(device_global_bytes / (1024ULL * 1024ULL)) +
                    " MiB, budget=" +
                    std::to_string(memory_budget_bytes / (1024ULL * 1024ULL)) +
                    " MiB, fixed~" + std::to_string(fixed / (1024ULL * 1024ULL)) + " MiB)");
            }
            const auto variable_budget = memory_budget_bytes - fixed;
            // Per secondary slot: particle + dep/esc/steps + cascade summary.
            std::size_t bytes_per_sec = sizeof(SecondaryParticle3D);
            if (enable_secondary_transport) {
                bytes_per_sec += 2 * sizeof(float) + sizeof(std::uint32_t);
            }
            if (enable_fragment_cascade) {
                bytes_per_sec += sizeof(CascadeTransportSummary);
            }
            std::size_t bytes_per_neu = 0;
            if (enable_neutral_transport) {
                bytes_per_neu =
                    sizeof(NeutralParticle3D) + sizeof(NeutralTransportSummary);
            }
            // Prefer keeping secondary capacity; scale both proportionally.
            const auto total_var =
                secondary_queue_capacity * bytes_per_sec +
                neutral_queue_capacity * bytes_per_neu;
            if (total_var > 0) {
                const auto scale = static_cast<double>(variable_budget) /
                                   static_cast<double>(total_var);
                if (scale < 1.0) {
                    secondary_queue_capacity = static_cast<std::size_t>(
                        std::floor(static_cast<double>(secondary_queue_capacity) * scale));
                    neutral_queue_capacity = static_cast<std::size_t>(
                        std::floor(static_cast<double>(neutral_queue_capacity) * scale));
                }
            }
            // Enforce minimum usable queues.
            if (enable_secondary_generation) {
                secondary_queue_capacity =
                    std::max<std::size_t>(secondary_queue_capacity, number_of_histories);
            }
            if (enable_neutral_transport) {
                neutral_queue_capacity =
                    std::max<std::size_t>(neutral_queue_capacity, number_of_histories);
            }
            estimated = estimate_device_bytes(secondary_queue_capacity, neutral_queue_capacity);
            if (estimated > memory_budget_bytes) {
                throw std::runtime_error(
                    "Unable to fit device buffers under max_device_memory_fraction=" +
                    std::to_string(config.max_device_memory_fraction) + " (estimate " +
                    std::to_string(estimated / (1024ULL * 1024ULL)) + " MiB > budget " +
                    std::to_string(memory_budget_bytes / (1024ULL * 1024ULL)) + " MiB)");
            }
            std::cout << "Device memory budget: scaled queues to fit "
                      << static_cast<int>(config.max_device_memory_fraction * 100.0)
                      << "% of "
                      << (device_global_bytes / (1024ULL * 1024ULL)) << " MiB"
                      << " (estimate " << (estimated / (1024ULL * 1024ULL)) << " MiB;"
                      << " secondary_queue_capacity=" << secondary_queue_capacity
                      << "; neutral_queue_capacity=" << neutral_queue_capacity << ")\n";
        } else {
            std::cout << "Device memory estimate: "
                      << (estimated / (1024ULL * 1024ULL)) << " MiB / budget "
                      << (memory_budget_bytes / (1024ULL * 1024ULL)) << " MiB ("
                      << static_cast<int>(config.max_device_memory_fraction * 100.0)
                      << "% of "
                      << (device_global_bytes / (1024ULL * 1024ULL)) << " MiB); "
                      << "secondary_queue_capacity=" << secondary_queue_capacity
                      << "\n";
        }
    }

    if (reuse_immutable_buffers) {
        context->impl_->ensure_initialized(stopping_power, cross_section, reaction_packages,
                                           cascade_packages, neutral_packages);
    }
    auto* table_device = reuse_immutable_buffers
                             ? context->impl_->table_device
                             : sycl::malloc_device<float>(table_size, queue);
    auto* cross_section_device =
        reuse_immutable_buffers
            ? context->impl_->cross_section_device
            : sycl::malloc_device<float>(cross_section_table_size, queue);
    auto* dose_device = sycl::malloc_device<DoseAtomicT>(number_of_bins, queue);
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? sycl::malloc_device<DoseAtomicT>(number_of_voxels, queue)
                                  : nullptr;
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? sycl::malloc_device<DoseAtomicT>(
                  charged_origin_category_count * number_of_voxels, queue)
            : nullptr;
    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* nuclear_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);
#ifdef CARBON_TRANSPORT_PROFILE
    auto* profile_counters_device =
        sycl::malloc_device<std::uint64_t>(transport_profile_slot_count(), queue);
#else
    std::uint64_t* profile_counters_device = nullptr;
#endif
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            sycl::malloc_device<PrimarySpotBatchEntry>(primary_spot_count, queue);
        if (primary_spots_device != nullptr) {
            queue.copy(config.primary_spot_batch.data(), primary_spots_device,
                       primary_spot_count)
                .wait_and_throw();
        }
    }
    ReactionEnergyBin* reaction_bins_device = nullptr;
    ReactionPackage* reactions_device = nullptr;
    ReactionSecondary* reaction_secondaries_device = nullptr;
    SecondaryParticle3D* secondary_queue_device = nullptr;
    std::uint64_t* secondary_queue_counter_device = nullptr;
    std::uint64_t* secondary_queue_filled_device = nullptr;
    std::uint64_t* secondary_work_counter_device = nullptr;
    SecondaryGenerationSummary* secondary_summaries_device = nullptr;
    DoseAtomicT* fragment_dose_device = nullptr;
    float* secondary_deposited_device = nullptr;
    float* secondary_escaped_device = nullptr;
    std::uint32_t* secondary_steps_device = nullptr;
    CascadeProjectile* cascade_projectiles_device = nullptr;
    CascadeCrossSectionSample* cascade_cross_sections_device = nullptr;
    CascadeInteraction* cascade_interactions_device = nullptr;
    ReactionSecondary* cascade_products_device = nullptr;
    CascadeTransportSummary* cascade_summaries_device = nullptr;
    // Projectile-major XS on water SP energy grid: size n_proj * table_size.
    float* cascade_xs_lut_device = nullptr;
    std::size_t cascade_xs_lut_size = 0;
    NeutralProjectile* neutral_projectiles_device = nullptr;
    NeutralCrossSectionSample* neutral_cross_sections_device = nullptr;
    NeutralInteraction* neutral_interactions_device = nullptr;
    ReactionSecondary* neutral_products_device = nullptr;
    NeutralParticle3D* neutral_queue_device = nullptr;
    std::uint64_t* neutral_queue_counter_device = nullptr;
    std::uint64_t* neutral_queue_filled_device = nullptr;
    NeutralTransportSummary* neutral_summaries_device = nullptr;
    DoseAtomicT* neutral_origin_dose_device = nullptr;
    DoseAtomicT* neutral_origin_voxel_dose_device = nullptr;
    if (enable_secondary_generation) {
        reaction_bins_device =
            reuse_immutable_buffers
                ? context->impl_->reaction_bins_device
                : sycl::malloc_device<ReactionEnergyBin>(
                      reaction_packages->energy_bins().size(), queue);
        reactions_device = reuse_immutable_buffers
                               ? context->impl_->reactions_device
                               : sycl::malloc_device<ReactionPackage>(
                                     reaction_packages->reactions().size(), queue);
        reaction_secondaries_device =
            reuse_immutable_buffers
                ? context->impl_->reaction_secondaries_device
                : sycl::malloc_device<ReactionSecondary>(
                      reaction_packages->secondaries().size(), queue);
        secondary_queue_device =
            sycl::malloc_device<SecondaryParticle3D>(secondary_queue_capacity, queue);
        secondary_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        secondary_summaries_device =
            sycl::malloc_device<SecondaryGenerationSummary>(number_of_histories, queue);
        if (enable_secondary_transport) {
            secondary_work_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
            fragment_dose_device = sycl::malloc_device<DoseAtomicT>(
                fragment_species_count * number_of_bins, queue);
            secondary_deposited_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_escaped_device =
                sycl::malloc_device<float>(secondary_queue_capacity, queue);
            secondary_steps_device =
                sycl::malloc_device<std::uint32_t>(secondary_queue_capacity, queue);
        }
        if (enable_fragment_cascade) {
            cascade_projectiles_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_projectiles_device
                    : sycl::malloc_device<CascadeProjectile>(
                          cascade_packages->projectiles().size(), queue);
            cascade_cross_sections_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_cross_sections_device
                    : sycl::malloc_device<CascadeCrossSectionSample>(
                          cascade_packages->cross_sections().size(), queue);
            cascade_interactions_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_interactions_device
                    : sycl::malloc_device<CascadeInteraction>(
                          cascade_packages->interactions().size(), queue);
            cascade_products_device =
                reuse_immutable_buffers
                    ? context->impl_->cascade_products_device
                    : sycl::malloc_device<ReactionSecondary>(
                          cascade_packages->products().size(), queue);
            cascade_summaries_device = sycl::malloc_device<CascadeTransportSummary>(
                secondary_queue_capacity, queue);
            cascade_xs_lut_size =
                cascade_packages->projectiles().size() * table_size;
            cascade_xs_lut_device =
                sycl::malloc_device<float>(cascade_xs_lut_size, queue);
        }
    }
    if (enable_neutral_transport) {
        neutral_projectiles_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_projectiles_device
                : sycl::malloc_device<NeutralProjectile>(
                      neutral_packages->projectiles().size(), queue);
        neutral_cross_sections_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_cross_sections_device
                : sycl::malloc_device<NeutralCrossSectionSample>(
                      neutral_packages->cross_sections().size(), queue);
        neutral_interactions_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_interactions_device
                : sycl::malloc_device<NeutralInteraction>(
                      neutral_packages->interactions().size(), queue);
        neutral_products_device =
            reuse_immutable_buffers
                ? context->impl_->neutral_products_device
                : sycl::malloc_device<ReactionSecondary>(
                      neutral_packages->products().size(), queue);
        neutral_queue_device =
            sycl::malloc_device<NeutralParticle3D>(neutral_queue_capacity, queue);
        neutral_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_summaries_device =
            sycl::malloc_device<NeutralTransportSummary>(neutral_queue_capacity, queue);
        neutral_origin_dose_device = sycl::malloc_device<DoseAtomicT>(
            neutral_origin_category_count * number_of_bins, queue);
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_device = sycl::malloc_device<DoseAtomicT>(
                neutral_origin_category_count * number_of_voxels, queue);
        }
    }
    const auto free_device = [&queue](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };
    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers) {
            free_device(pointer);
        }
    };

    const auto enable_layered_phantom = config.enable_layered_phantom;
    const auto slab_layer_count =
        enable_layered_phantom
            ? static_cast<std::uint32_t>(config.slab_layers.size())
            : 0U;
    float* slab_z_ends_device = nullptr;
    float* slab_densities_device = nullptr;
    if (slab_layer_count > 0) {
        slab_z_ends_device = sycl::malloc_device<float>(slab_layer_count, queue);
        slab_densities_device = sycl::malloc_device<float>(slab_layer_count, queue);
        if (slab_z_ends_device == nullptr || slab_densities_device == nullptr) {
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            free_immutable_device(table_device);
            free_immutable_device(cross_section_device);
            free_device(dose_device);
            free_device(voxel_dose_device);
            free_device(charged_origin_voxel_dose_device);
            free_device(deposited_device);
            free_device(escaped_device);
            free_device(nuclear_device);
            free_device(steps_device);
            free_device(profile_counters_device);
            throw std::bad_alloc();
        }
        std::vector<float> slab_z_host(slab_layer_count);
        std::vector<float> slab_rho_host(slab_layer_count);
        for (std::uint32_t index = 0; index < slab_layer_count; ++index) {
            slab_z_host[index] =
                static_cast<float>(config.slab_layers[index].z_end_mm);
            slab_rho_host[index] =
                static_cast<float>(config.slab_layers[index].density_g_per_cm3);
        }
        queue.memcpy(slab_z_ends_device, slab_z_host.data(),
                     sizeof(float) * slab_layer_count)
            .wait_and_throw();
        queue.memcpy(slab_densities_device, slab_rho_host.data(),
                     sizeof(float) * slab_layer_count)
            .wait_and_throw();
    }

    // Optional absolute per-layer material tables (real bone/lung, not density-scaled water).
    const auto use_material_tables =
        enable_layered_phantom && !config.slab_stopping_power_files.empty();
    const auto material_table_count =
        use_material_tables ? static_cast<std::uint32_t>(config.slab_stopping_power_files.size())
                            : 0U;
    float* material_sp_device = nullptr;
    float* material_xs_device = nullptr;
    if (use_material_tables) {
        if (material_table_count != slab_layer_count) {
            throw std::invalid_argument("Material table count must match slab layer count");
        }
        std::vector<float> material_sp_host(static_cast<std::size_t>(material_table_count) *
                                            table_size);
        std::vector<float> material_xs_host(
            static_cast<std::size_t>(material_table_count) * cross_section_table_size);
        for (std::uint32_t mat = 0; mat < material_table_count; ++mat) {
            const auto sp_table =
                StoppingPowerTable::from_csv(config.slab_stopping_power_files[mat]);
            const auto xs_table =
                CrossSectionTable::from_csv(config.slab_cross_section_files[mat]);
            if (sp_table.values().size() != table_size ||
                !is_uniform_grid(sp_table.energies()) ||
                std::abs(sp_table.energies().front() - stopping_power.energies().front()) >
                    1.0e-9 ||
                std::abs(sp_table.energies()[1] - sp_table.energies()[0] -
                         (stopping_power.energies()[1] - stopping_power.energies()[0])) >
                    1.0e-9) {
                throw std::invalid_argument(
                    "Slab material SP table must match the primary water energy grid");
            }
            if (xs_table.values().size() != cross_section_table_size ||
                !is_uniform_grid(xs_table.energies()) ||
                std::abs(xs_table.energies().front() - cross_section.energies().front()) >
                    1.0e-9) {
                throw std::invalid_argument(
                    "Slab material XS table must match the primary water energy grid");
            }
            for (std::size_t i = 0; i < table_size; ++i) {
                material_sp_host[static_cast<std::size_t>(mat) * table_size + i] =
                    static_cast<float>(sp_table.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                material_xs_host[static_cast<std::size_t>(mat) * cross_section_table_size + i] =
                    static_cast<float>(xs_table.values()[i]);
            }
        }
        material_sp_device = sycl::malloc_device<float>(material_sp_host.size(), queue);
        material_xs_device = sycl::malloc_device<float>(material_xs_host.size(), queue);
        if (material_sp_device == nullptr || material_xs_device == nullptr) {
            free_device(material_sp_device);
            free_device(material_xs_device);
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            throw std::bad_alloc();
        }
        queue.memcpy(material_sp_device, material_sp_host.data(),
                     sizeof(float) * material_sp_host.size())
            .wait_and_throw();
        queue.memcpy(material_xs_device, material_xs_host.data(),
                     sizeof(float) * material_xs_host.size())
            .wait_and_throw();
    }

    // 7b: single AABB insert (bone strip / cavity) in water background.
    const auto enable_hetero_insert = config.enable_hetero_insert;
    const auto insert_x_min = static_cast<float>(config.hetero_insert.x_min_mm);
    const auto insert_x_max = static_cast<float>(config.hetero_insert.x_max_mm);
    const auto insert_y_min = static_cast<float>(config.hetero_insert.y_min_mm);
    const auto insert_y_max = static_cast<float>(config.hetero_insert.y_max_mm);
    const auto insert_z_min = static_cast<float>(config.hetero_insert.z_min_mm);
    const auto insert_z_max = static_cast<float>(config.hetero_insert.z_max_mm);
    const auto insert_density_g_per_cm3 =
        static_cast<float>(config.hetero_insert.density_g_per_cm3);
    const auto use_insert_material_tables =
        enable_hetero_insert && !config.insert_stopping_power_file.empty();
    float* insert_sp_device = nullptr;
    float* insert_xs_device = nullptr;
    if (use_insert_material_tables) {
        const auto insert_sp =
            StoppingPowerTable::from_csv(config.insert_stopping_power_file);
        const auto insert_xs =
            CrossSectionTable::from_csv(config.insert_cross_section_file);
        if (insert_sp.values().size() != table_size ||
            !is_uniform_grid(insert_sp.energies()) ||
            std::abs(insert_sp.energies().front() - stopping_power.energies().front()) >
                1.0e-9) {
            throw std::invalid_argument(
                "Insert SP table must match the primary water energy grid");
        }
        if (insert_xs.values().size() != cross_section_table_size ||
            !is_uniform_grid(insert_xs.energies()) ||
            std::abs(insert_xs.energies().front() - cross_section.energies().front()) >
                1.0e-9) {
            throw std::invalid_argument(
                "Insert XS table must match the primary water energy grid");
        }
        std::vector<float> insert_sp_host(table_size);
        std::vector<float> insert_xs_host(cross_section_table_size);
        for (std::size_t i = 0; i < table_size; ++i) {
            insert_sp_host[i] = static_cast<float>(insert_sp.values()[i]);
        }
        for (std::size_t i = 0; i < cross_section_table_size; ++i) {
            insert_xs_host[i] = static_cast<float>(insert_xs.values()[i]);
        }
        insert_sp_device = sycl::malloc_device<float>(table_size, queue);
        insert_xs_device = sycl::malloc_device<float>(cross_section_table_size, queue);
        if (insert_sp_device == nullptr || insert_xs_device == nullptr) {
            free_device(insert_sp_device);
            free_device(insert_xs_device);
            free_device(material_sp_device);
            free_device(material_xs_device);
            free_device(slab_z_ends_device);
            free_device(slab_densities_device);
            throw std::bad_alloc();
        }
        queue.memcpy(insert_sp_device, insert_sp_host.data(), sizeof(float) * table_size)
            .wait_and_throw();
        queue
            .memcpy(insert_xs_device, insert_xs_host.data(),
                    sizeof(float) * cross_section_table_size)
            .wait_and_throw();
    }

    // 7c: CT voxel density + material-id field.
    const auto enable_ct_grid = config.enable_ct_grid;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;
    CtGrid ct_grid_host{};
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    // Schneider section mass-SP: (Z/A)_rel and Bragg I [eV] (CCTG v2/v3).
    // Precomputed mass-SP factor LUT: section * table_size + energy_index.
    // Replaces per-step Bethe log evaluation (P2).
    float* ct_mass_sp_factor_lut_device = nullptr;
    float* ct_sp_device = nullptr;  // optional absolute 4-class tables
    float* ct_xs_device = nullptr;
    float* ct_ref_density_device = nullptr;
    std::uint32_t ct_nx = 0;
    std::uint32_t ct_ny = 0;
    std::uint32_t ct_nz = 0;
    std::uint32_t ct_n_mass_factors = 0;
    float ct_origin_x = 0.0F;
    float ct_origin_y = 0.0F;
    float ct_origin_z = 0.0F;
    float ct_spacing_x = 1.0F;
    float ct_spacing_y = 1.0F;
    float ct_spacing_z = 1.0F;
    // Prefer Schneider mass-SP scaling when the grid carries factors (CCTG v2/v3).
    // Material SP and nuclear XS are independent: a v3 mass-SP grid can still
    // use composition-specific 4-class nuclear cross sections.
    auto use_ct_mass_sp = false;
    auto use_ct_material_sp = false;
    auto use_ct_material_xs = false;
    auto ct_material_ids_are_schneider_sections = false;
    if (enable_ct_grid) {
        ct_grid_host = CtGrid::from_binary(config.ct_grid_file);
        ct_nx = ct_grid_host.nx;
        ct_ny = ct_grid_host.ny;
        ct_nz = ct_grid_host.nz;
        ct_origin_x = ct_grid_host.origin_x_mm;
        ct_origin_y = ct_grid_host.origin_y_mm;
        ct_origin_z = ct_grid_host.origin_z_mm;
        ct_spacing_x = ct_grid_host.spacing_x_mm;
        ct_spacing_y = ct_grid_host.spacing_y_mm;
        ct_spacing_z = ct_grid_host.spacing_z_mm;
        // Align dose scorer xy bins with the CT sample grid so that:
        //   floor((x - origin) / spacing) matches for density, mass and tally.
        // Previously the scorer was forced to a 0-centered box (edge -n*s/2),
        // which is 0.25 mm (X) / 1 mm (Y) off the TPS-90 CT origins and breaks
        // dose-to-medium index pairing used by voxel_masses_kg.
        if (enable_voxel_scoring && ct_nx == voxel_bins_x && ct_ny == voxel_bins_y &&
            std::fabs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
            std::fabs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
            voxel_min_x_mm = ct_origin_x;
            voxel_min_y_mm = ct_origin_y;
            voxel_max_x_mm =
                ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
            voxel_max_y_mm =
                ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
        }
        use_ct_mass_sp = ct_grid_host.has_mass_sp_factors();
        ct_material_ids_are_schneider_sections =
            ct_grid_host.file_version != CtGrid::version_legacy;
        use_ct_material_sp =
            !use_ct_mass_sp &&
            (!config.ct_air_stopping_power_file.empty() ||
             !config.ct_lung_stopping_power_file.empty() ||
             !config.ct_water_stopping_power_file.empty() ||
             !config.ct_bone_stopping_power_file.empty());
        use_ct_material_xs =
            !config.ct_air_cross_section_file.empty() ||
            !config.ct_lung_cross_section_file.empty() ||
            !config.ct_water_cross_section_file.empty() ||
            !config.ct_bone_cross_section_file.empty();
        const auto ct_count = ct_grid_host.number_of_voxels();
        ct_density_device = sycl::malloc_device<float>(ct_count, queue);
        ct_material_device = sycl::malloc_device<std::uint8_t>(ct_count, queue);
        if (ct_density_device == nullptr || ct_material_device == nullptr) {
            free_device(ct_density_device);
            free_device(ct_material_device);
            throw std::bad_alloc();
        }
        queue
            .memcpy(ct_density_device, ct_grid_host.density_g_per_cm3.data(),
                    sizeof(float) * ct_count)
            .wait_and_throw();
        queue
            .memcpy(ct_material_device, ct_grid_host.material_id.data(),
                    sizeof(std::uint8_t) * ct_count)
            .wait_and_throw();

        if (use_ct_mass_sp) {
            const auto& za_host = !ct_grid_host.mass_sp_za_rel.empty()
                                      ? ct_grid_host.mass_sp_za_rel
                                      : ct_grid_host.mass_sp_factor;
            ct_n_mass_factors = static_cast<std::uint32_t>(za_host.size());
            std::vector<float> I_host = ct_grid_host.mass_sp_I_eV;
            if (I_host.size() != za_host.size()) {
                I_host.assign(za_host.size(), 78.0F);
            }
            // P2: precompute mass-SP factor on the water energy grid so the
            // kernel only does clamp/index/lerp (no log/Bethe on every step).
            // Layout: section-major, size n_sections * table_size.
            // Bake ct_stopping_power_scale into the LUT (avoids a per-step mul).
            const auto sp_scale =
                static_cast<float>(config.ct_stopping_power_scale);
            const auto lut_count =
                static_cast<std::size_t>(ct_n_mass_factors) * table_size;
            std::vector<float> mass_factor_lut(lut_count);
            for (std::uint32_t sec = 0; sec < ct_n_mass_factors; ++sec) {
                const auto za = za_host[sec];
                const auto I_eV = I_host[sec];
                const auto base = static_cast<std::size_t>(sec) * table_size;
                for (std::size_t i = 0; i < table_size; ++i) {
                    mass_factor_lut[base + i] =
                        sp_scale * ct_mass_sp_energy_factor(
                                       za, I_eV,
                                       static_cast<float>(
                                           stopping_power.energies()[i]));
                }
            }
            ct_mass_sp_factor_lut_device = sycl::malloc_device<float>(lut_count, queue);
            if (ct_mass_sp_factor_lut_device == nullptr) {
                free_device(ct_density_device);
                free_device(ct_material_device);
                throw std::bad_alloc();
            }
            queue
                .memcpy(ct_mass_sp_factor_lut_device, mass_factor_lut.data(),
                        sizeof(float) * lut_count)
                .wait_and_throw();
        }

        // Optional absolute 4-class tables (legacy path when no mass-SP LUT).
        std::vector<float> ct_sp_host(4 * table_size);
        std::vector<float> ct_xs_host(4 * cross_section_table_size);
        std::vector<float> ct_ref_host = {1.0F, 1.0F, 1.0F, 1.85F};
        for (std::uint32_t mat = 0; mat < 4; ++mat) {
            for (std::size_t i = 0; i < table_size; ++i) {
                ct_sp_host[mat * table_size + i] =
                    static_cast<float>(stopping_power.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                ct_xs_host[mat * cross_section_table_size + i] =
                    static_cast<float>(cross_section.values()[i]);
            }
        }
        if (use_ct_material_sp || use_ct_material_xs) {
            const std::array<std::filesystem::path, 4> sp_paths = {
                config.ct_air_stopping_power_file.empty() ? config.stopping_power_file
                                                          : config.ct_air_stopping_power_file,
                config.ct_lung_stopping_power_file.empty()
                    ? config.stopping_power_file
                    : config.ct_lung_stopping_power_file,
                config.ct_water_stopping_power_file.empty()
                    ? config.stopping_power_file
                    : config.ct_water_stopping_power_file,
                config.ct_bone_stopping_power_file.empty()
                    ? config.stopping_power_file
                    : config.ct_bone_stopping_power_file,
            };
            const std::array<std::filesystem::path, 4> xs_paths = {
                config.ct_air_cross_section_file.empty() ? config.nuclear_cross_section_file
                                                         : config.ct_air_cross_section_file,
                config.ct_lung_cross_section_file.empty()
                    ? config.nuclear_cross_section_file
                    : config.ct_lung_cross_section_file,
                config.ct_water_cross_section_file.empty()
                    ? config.nuclear_cross_section_file
                    : config.ct_water_cross_section_file,
                config.ct_bone_cross_section_file.empty()
                    ? config.nuclear_cross_section_file
                    : config.ct_bone_cross_section_file,
            };
            if (!config.ct_bone_stopping_power_file.empty()) {
                ct_ref_host[3] = 1.85F;
            }
            for (std::uint32_t mat = 0; mat < 4; ++mat) {
                if (use_ct_material_sp) {
                    const auto sp_table = StoppingPowerTable::from_csv(sp_paths[mat]);
                    if (sp_table.values().size() != table_size) {
                        throw std::invalid_argument(
                            "CT material stopping-power tables must match water grid size");
                    }
                    for (std::size_t i = 0; i < table_size; ++i) {
                        ct_sp_host[mat * table_size + i] =
                            static_cast<float>(sp_table.values()[i]);
                    }
                }
                if (use_ct_material_xs) {
                    const auto xs_table = CrossSectionTable::from_csv(xs_paths[mat]);
                    if (xs_table.values().size() != cross_section_table_size) {
                        throw std::invalid_argument(
                            "CT material cross-section tables must match water grid size");
                    }
                    for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                        ct_xs_host[mat * cross_section_table_size + i] =
                            static_cast<float>(xs_table.values()[i]);
                    }
                }
            }
        }
        ct_sp_device = sycl::malloc_device<float>(ct_sp_host.size(), queue);
        ct_xs_device = sycl::malloc_device<float>(ct_xs_host.size(), queue);
        ct_ref_density_device = sycl::malloc_device<float>(4, queue);
        if (ct_sp_device == nullptr || ct_xs_device == nullptr ||
            ct_ref_density_device == nullptr) {
            free_device(ct_sp_device);
            free_device(ct_xs_device);
            free_device(ct_ref_density_device);
            free_device(ct_mass_sp_factor_lut_device);
            free_device(ct_density_device);
            free_device(ct_material_device);
            throw std::bad_alloc();
        }
        queue.memcpy(ct_sp_device, ct_sp_host.data(), sizeof(float) * ct_sp_host.size())
            .wait_and_throw();
        queue.memcpy(ct_xs_device, ct_xs_host.data(), sizeof(float) * ct_xs_host.size())
            .wait_and_throw();
        queue.memcpy(ct_ref_density_device, ct_ref_host.data(), sizeof(float) * 4)
            .wait_and_throw();
    }
    const auto secondary_allocation_failed =
        enable_secondary_generation &&
        (reaction_bins_device == nullptr || reactions_device == nullptr ||
         reaction_secondaries_device == nullptr || secondary_queue_device == nullptr ||
         secondary_queue_counter_device == nullptr || secondary_queue_filled_device == nullptr ||
         secondary_summaries_device == nullptr ||
         (enable_secondary_transport &&
           (secondary_work_counter_device == nullptr || fragment_dose_device == nullptr ||
           secondary_deposited_device == nullptr ||
           secondary_escaped_device == nullptr || secondary_steps_device == nullptr)) ||
         (enable_fragment_cascade &&
          (cascade_projectiles_device == nullptr || cascade_cross_sections_device == nullptr ||
           cascade_interactions_device == nullptr || cascade_products_device == nullptr ||
           cascade_summaries_device == nullptr || cascade_xs_lut_device == nullptr)));
    const auto neutral_allocation_failed =
        enable_neutral_transport &&
        (neutral_projectiles_device == nullptr || neutral_cross_sections_device == nullptr ||
         neutral_interactions_device == nullptr || neutral_products_device == nullptr ||
         neutral_queue_device == nullptr || neutral_queue_counter_device == nullptr ||
         neutral_queue_filled_device == nullptr || neutral_summaries_device == nullptr ||
         neutral_origin_dose_device == nullptr ||
         (enable_voxel_scoring && neutral_origin_voxel_dose_device == nullptr));
    if (table_device == nullptr || cross_section_device == nullptr || dose_device == nullptr ||
        (enable_voxel_scoring && voxel_dose_device == nullptr) ||
        (enable_charged_origin_voxel_scoring &&
         charged_origin_voxel_dose_device == nullptr) ||
        deposited_device == nullptr ||
        escaped_device == nullptr || nuclear_device == nullptr || steps_device == nullptr ||
#ifdef CARBON_TRANSPORT_PROFILE
        profile_counters_device == nullptr ||
#endif
        (primary_spot_count > 0 && primary_spots_device == nullptr) ||
        secondary_allocation_failed || neutral_allocation_failed) {
        free_immutable_device(table_device);
        free_immutable_device(cross_section_device);
        free_device(dose_device);
        free_device(voxel_dose_device);
        free_device(charged_origin_voxel_dose_device);
        free_device(deposited_device);
        free_device(escaped_device);
        free_device(nuclear_device);
        free_device(steps_device);
        free_device(profile_counters_device);
        free_device(primary_spots_device);
        free_device(slab_z_ends_device);
        free_device(slab_densities_device);
        free_device(material_sp_device);
        free_device(material_xs_device);
        free_device(insert_sp_device);
        free_device(insert_xs_device);
        free_device(ct_density_device);
        free_device(ct_material_device);
        free_device(ct_mass_sp_factor_lut_device);
        free_device(ct_sp_device);
        free_device(ct_xs_device);
        free_device(ct_ref_density_device);
        free_immutable_device(reaction_bins_device);
        free_immutable_device(reactions_device);
        free_immutable_device(reaction_secondaries_device);
        free_device(secondary_queue_device);
        free_device(secondary_queue_counter_device);
        free_device(secondary_queue_filled_device);
        free_device(secondary_work_counter_device);
        free_device(secondary_summaries_device);
        free_device(fragment_dose_device);
        free_device(secondary_deposited_device);
        free_device(secondary_escaped_device);
        free_device(secondary_steps_device);
        free_immutable_device(cascade_projectiles_device);
        free_immutable_device(cascade_cross_sections_device);
        free_immutable_device(cascade_interactions_device);
        free_immutable_device(cascade_products_device);
        free_device(cascade_summaries_device);
        free_device(cascade_xs_lut_device);
        free_immutable_device(neutral_projectiles_device);
        free_immutable_device(neutral_cross_sections_device);
        free_immutable_device(neutral_interactions_device);
        free_immutable_device(neutral_products_device);
        free_device(neutral_queue_device);
        free_device(neutral_queue_counter_device);
        free_device(neutral_queue_filled_device);
        free_device(neutral_summaries_device);
        free_device(neutral_origin_dose_device);
        free_device(neutral_origin_voxel_dose_device);
        throw std::runtime_error("SYCL USM device allocation failed");
    }

    if (!reuse_immutable_buffers) {
        std::vector<float> table_host(table_size);
        std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                       table_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(table_host.data(), table_device, table_size);
        std::vector<float> cross_section_host(cross_section_table_size);
        std::transform(cross_section.values().begin(), cross_section.values().end(),
                       cross_section_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(cross_section_host.data(), cross_section_device, cross_section_table_size);
        queue.wait_and_throw();
    }
    if (enable_fragment_cascade && cascade_xs_lut_device != nullptr &&
        cascade_packages != nullptr) {
        auto cascade_xs_lut_host =
            build_cascade_xs_energy_lut(*cascade_packages, stopping_power.energies());
        if (cascade_xs_lut_host.size() != cascade_xs_lut_size) {
            free_device(cascade_xs_lut_device);
            throw std::runtime_error("Cascade XS LUT size mismatch");
        }
        queue.copy(cascade_xs_lut_host.data(), cascade_xs_lut_device, cascade_xs_lut_size)
            .wait_and_throw();
        std::cout << "Cascade XS LUT: " << cascade_packages->projectiles().size()
                  << " projectiles x " << table_size << " energies on water SP grid\n"
                  << std::flush;
    }
    queue.memset(dose_device, 0, number_of_bins * sizeof(DoseAtomicT));
#ifdef CARBON_TRANSPORT_PROFILE
    if (profile_counters_device != nullptr) {
        queue.memset(profile_counters_device, 0,
                     transport_profile_slot_count() * sizeof(std::uint64_t));
    }
#endif
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_secondary_generation) {
        if (!reuse_immutable_buffers) {
            queue.copy(reaction_packages->energy_bins().data(), reaction_bins_device,
                       reaction_packages->energy_bins().size());
            queue.copy(reaction_packages->reactions().data(), reactions_device,
                       reaction_packages->reactions().size());
            queue.copy(reaction_packages->secondaries().data(), reaction_secondaries_device,
                       reaction_packages->secondaries().size());
        }
        queue.memset(secondary_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(secondary_queue_filled_device, 0, sizeof(std::uint64_t));
        if (enable_secondary_transport) {
            queue.memset(fragment_dose_device, 0,
                         fragment_species_count * number_of_bins * sizeof(DoseAtomicT));
        }
        if (enable_fragment_cascade) {
            if (!reuse_immutable_buffers) {
                queue.copy(cascade_packages->projectiles().data(), cascade_projectiles_device,
                           cascade_packages->projectiles().size());
                queue.copy(cascade_packages->cross_sections().data(),
                           cascade_cross_sections_device,
                           cascade_packages->cross_sections().size());
                queue.copy(cascade_packages->interactions().data(), cascade_interactions_device,
                           cascade_packages->interactions().size());
                queue.copy(cascade_packages->products().data(), cascade_products_device,
                           cascade_packages->products().size());
            }
            queue.memset(cascade_summaries_device, 0,
                         secondary_queue_capacity * sizeof(CascadeTransportSummary));
        }
    }
    if (enable_neutral_transport) {
        if (!reuse_immutable_buffers) {
            queue.copy(neutral_packages->projectiles().data(), neutral_projectiles_device,
                       neutral_packages->projectiles().size());
            queue.copy(neutral_packages->cross_sections().data(),
                       neutral_cross_sections_device,
                       neutral_packages->cross_sections().size());
            queue.copy(neutral_packages->interactions().data(), neutral_interactions_device,
                       neutral_packages->interactions().size());
            queue.copy(neutral_packages->products().data(), neutral_products_device,
                       neutral_packages->products().size());
        }
        queue.memset(neutral_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_queue_filled_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_summaries_device, 0,
                     neutral_queue_capacity * sizeof(NeutralTransportSummary));
        queue.memset(neutral_origin_dose_device, 0,
                     neutral_origin_category_count * number_of_bins * sizeof(DoseAtomicT));
        if (enable_voxel_scoring) {
            queue.memset(neutral_origin_voxel_dose_device, 0,
                         neutral_origin_category_count * number_of_voxels * sizeof(DoseAtomicT));
        }
    }

    // GPU backends (CUDA / Level Zero / OpenCL GPU / --device gpu|cuda|...) use larger
    // work-groups; SYCL CPU and other selectors stay at 128. CUDA stays at 128:
    // the transport kernels are register-heavy (FP64 atomics + cascade) and 256
    // often hurts occupancy while making each submit longer under WSL.
    const bool prefer_large_workgroup =
        !is_cuda_backend &&
        (device.is_gpu() || device_name == "gpu" || device_name == "cuda" ||
         device_name == "nvidia" || device_name == "level_zero" || device_name == "intel" ||
         device_name == "arc" || device_name == "opencl");
    const std::size_t local_size = prefer_large_workgroup ? 256U : 128U;
    const auto compute_units = static_cast<std::size_t>(
        std::max(1U, device.get_info<sycl::info::device::max_compute_units>()));
    // Secondary: one particle per work-item. Large batches fill the GPU; max
    // step caps prevent WSL freezes from thrashing tracks.
    const std::size_t worker_multiplier = is_cuda_backend ? 4U : 8U;
    auto persistent_secondary_workers =
        compute_units * local_size * worker_multiplier;
    if (is_cuda_backend) {
        persistent_secondary_workers =
            std::min(persistent_secondary_workers, std::size_t{32768});
    }
    persistent_secondary_workers =
        std::max(persistent_secondary_workers, local_size);

    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        if (is_cuda_backend) {
            // Prefer large primary chunks for 100k–10M plans.
            history_chunk = number_of_histories > 1'000'000 ? 16384 : 4096;
        } else if (device.is_gpu()) {
            history_chunk = 8192;
        } else {
            history_chunk = number_of_histories;
        }
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);
    if (is_cuda_backend) {
        history_chunk = std::min(history_chunk, std::size_t{65536});
    }
    std::size_t secondary_batch = config.secondary_batch_size;
    if (secondary_batch == 0) {
        // CUDA: large batches for throughput (max_secondary_steps prevents hangs).
        secondary_batch = is_cuda_backend
                              ? (number_of_histories > 1'000'000 ? 16384 : 8192)
                              : std::numeric_limits<std::size_t>::max() / 4;
    }
    if (is_cuda_backend) {
        secondary_batch = std::min(secondary_batch, std::size_t{65536});
    }
    secondary_batch = std::max<std::size_t>(local_size, secondary_batch);

    // Throttle progress I/O: 10M plans previously printed ~100k+ lines/sec of
    // secondary batch logs, which dominated wall time on WSL.
    const auto progress_log_every_histories =
        std::max<std::size_t>(history_chunk, number_of_histories / 50);
    const auto progress_log_every_secondary =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(secondary_batch),
                                250'000ULL);

    std::cout << "Launch plan: histories=" << number_of_histories
              << " history_chunk=" << history_chunk
              << " secondary_batch=" << secondary_batch
              << " local_size=" << local_size
              << " secondary_workers=" << persistent_secondary_workers
              << (is_cuda_backend ? " [CUDA]" : "")
              << (k_dose_atomic_fp32 ? " dose_atomic=fp32" : " dose_atomic=fp64")
              << '\n'
              << std::flush;
    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);
    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto random_seed = config.random_seed;
    const auto enable_primary_attenuation = config.enable_primary_attenuation;
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime =
        static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime =
        static_cast<float>(config.emittance_sigma_y_prime);
    const auto source_origin_x_mm = static_cast<float>(config.source_origin_x_mm);
    const auto source_origin_y_mm = static_cast<float>(config.source_origin_y_mm);
    const auto source_origin_z_mm = static_cast<float>(config.source_origin_z_mm);
    const auto beam_ux_x = static_cast<float>(config.beam_ux_x);
    const auto beam_ux_y = static_cast<float>(config.beam_ux_y);
    const auto beam_ux_z = static_cast<float>(config.beam_ux_z);
    const auto beam_uy_x = static_cast<float>(config.beam_uy_x);
    const auto beam_uy_y = static_cast<float>(config.beam_uy_y);
    const auto beam_uy_z = static_cast<float>(config.beam_uy_z);
    const auto beam_uz_x = static_cast<float>(config.beam_uz_x);
    const auto beam_uz_y = static_cast<float>(config.beam_uz_y);
    const auto beam_uz_z = static_cast<float>(config.beam_uz_z);
    const auto emittance_correlation_x =
        static_cast<float>(config.emittance_correlation_x);
    const auto emittance_correlation_y =
        static_cast<float>(config.emittance_correlation_y);
    const auto neutral_local_kerma_fraction =
        static_cast<float>(config.neutral_local_kerma_fraction);
    const auto neutral_kerma_high_energy_scale =
        static_cast<float>(config.neutral_kerma_high_energy_scale);
    const auto neutral_kerma_mean_free_path_mm =
        static_cast<float>(config.neutral_kerma_mean_free_path_mm);
    const auto electronic_buildup_fraction =
        static_cast<float>(config.electronic_buildup_fraction);
    const auto electronic_buildup_mfp_mm =
        static_cast<float>(config.electronic_buildup_mfp_mm);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.mass_number);
    const auto primary_mass_number = config.mass_number;
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
    const auto neutral_queue_capacity_u32 =
        static_cast<std::uint32_t>(neutral_queue_capacity);
    const auto cascade_projectile_count = enable_fragment_cascade
                                              ? cascade_packages->projectiles().size()
                                              : std::size_t{0};
    const auto neutral_projectile_count = enable_neutral_transport
                                              ? neutral_packages->projectiles().size()
                                              : std::size_t{0};
    const auto maximum_cascade_generations = config.maximum_cascade_generations;
    const auto maximum_neutral_generations = config.maximum_neutral_generations;

    for (std::size_t hist_offset = 0; hist_offset < number_of_histories;
         hist_offset += history_chunk) {
        const auto chunk_count =
            std::min(history_chunk, number_of_histories - hist_offset);
        const auto chunk_global =
            ((chunk_count + local_size - 1) / local_size) * local_size;
        auto kernel_event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{chunk_global}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) {
            const auto lane = item.get_global_linear_id();
            if (lane >= chunk_count) {
                return;
            }
            const auto global_history = hist_offset + lane;

            const PrimarySpotBatchEntry* spot = nullptr;
            std::uint64_t rng_history = global_history;
            if (primary_spot_count > 0) {
                std::size_t lower = 0;
                std::size_t upper = primary_spot_count;
                while (lower + 1 < upper) {
                    const auto middle = lower + (upper - lower) / 2;
                    if (global_history < primary_spots_device[middle].history_begin) {
                        upper = middle;
                    } else {
                        lower = middle;
                    }
                }
                spot = primary_spots_device + lower;
                rng_history = global_history - spot->history_begin;
            }
            const auto spot_seed = spot != nullptr ? spot->random_seed : random_seed;
            const auto spot_initial_energy_MeV =
                spot != nullptr ? spot->floats[0] : initial_energy_MeV;
            const auto spot_energy_spread =
                spot != nullptr ? spot->floats[1] : beam_energy_spread;

            auto energy_MeV = spot_initial_energy_MeV;
            if (spot_energy_spread > 0.0F) {
                // TOPAS BeamEnergySpread: Gaussian RMS = spread * mean total KE.
                const auto u0 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 40), 1.0e-12F);
                const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 41);
                constexpr float two_pi = 6.2831853071795864769F;
                const auto gauss =
                    sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                energy_MeV =
                    spot_initial_energy_MeV * (1.0F + spot_energy_spread * gauss);
                if (energy_MeV < energy_cutoff_MeV) {
                    energy_MeV = energy_cutoff_MeV;
                }
            }
            // Local beam frame (defaults: origin 0, +z beam). Emittance is local x/y.
            auto local_x_mm = 0.0F;
            auto local_y_mm = 0.0F;
            auto local_dx = 0.0F;
            auto local_dy = 0.0F;
            auto local_dz = 1.0F;
            if (enable_flat_source) {
                local_x_mm = flat_source_half_width_x_mm *
                             (2.0F * rng::uniform01(spot_seed, rng_history, 0, 34) - 1.0F);
                local_y_mm = flat_source_half_width_y_mm *
                             (2.0F * rng::uniform01(spot_seed, rng_history, 0, 35) - 1.0F);
            } else if (enable_emittance_source) {
                // TOPAS BiGaussian: sample (x,x') and (y,y') from bivariate normals.
                // x' = dx/dz (unitless, rad-like). Independent axes with correlations.
                const auto u0 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 30), 1.0e-12F);
                const auto u1 = rng::uniform01(spot_seed, rng_history, 0, 31);
                const auto u2 = sycl::fmax(
                    rng::uniform01(spot_seed, rng_history, 0, 32), 1.0e-12F);
                const auto u3 = rng::uniform01(spot_seed, rng_history, 0, 33);
                constexpr float two_pi = 6.2831853071795864769F;
                const auto g0 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                const auto g1 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::sin(two_pi * u1);
                const auto g2 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                const auto g3 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::sin(two_pi * u3);
                const auto sigma_x = spot != nullptr ? spot->floats[2]
                                                     : emittance_sigma_x_mm;
                const auto sigma_y = spot != nullptr ? spot->floats[3]
                                                     : emittance_sigma_y_mm;
                const auto sigma_x_prime =
                    spot != nullptr ? spot->floats[4]
                                    : emittance_sigma_x_prime;
                const auto sigma_y_prime =
                    spot != nullptr ? spot->floats[5]
                                    : emittance_sigma_y_prime;
                local_x_mm = sigma_x * g0;
                local_y_mm = sigma_y * g2;
                const auto rho_x = sycl::clamp(
                    spot != nullptr ? spot->floats[6]
                                    : emittance_correlation_x,
                    -0.9999F, 0.9999F);
                const auto rho_y = sycl::clamp(
                    spot != nullptr ? spot->floats[7]
                                    : emittance_correlation_y,
                    -0.9999F, 0.9999F);
                const auto x_prime =
                    sigma_x_prime *
                    (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                const auto y_prime =
                    sigma_y_prime *
                    (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                // Paraxial unit direction from slopes (dx/dz, dy/dz).
                const auto inv_norm =
                    sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                local_dx = x_prime * inv_norm;
                local_dy = y_prime * inv_norm;
                local_dz = inv_norm;
            }
            // World = origin + ux*x + uy*y ; dir = ux*dx + uy*dy + uz*dz
            const auto origin_x =
                spot != nullptr ? spot->floats[8] : source_origin_x_mm;
            const auto origin_y =
                spot != nullptr ? spot->floats[9] : source_origin_y_mm;
            const auto origin_z =
                spot != nullptr ? spot->floats[10] : source_origin_z_mm;
            const auto ux_x = spot != nullptr ? spot->floats[11] : beam_ux_x;
            const auto ux_y = spot != nullptr ? spot->floats[12] : beam_ux_y;
            const auto ux_z = spot != nullptr ? spot->floats[13] : beam_ux_z;
            const auto uy_x = spot != nullptr ? spot->floats[14] : beam_uy_x;
            const auto uy_y = spot != nullptr ? spot->floats[15] : beam_uy_y;
            const auto uy_z = spot != nullptr ? spot->floats[16] : beam_uy_z;
            const auto uz_x = spot != nullptr ? spot->floats[17] : beam_uz_x;
            const auto uz_y = spot != nullptr ? spot->floats[18] : beam_uz_y;
            const auto uz_z = spot != nullptr ? spot->floats[19] : beam_uz_z;
            auto position_x_mm =
                origin_x + ux_x * local_x_mm + uy_x * local_y_mm;
            auto position_y_mm =
                origin_y + ux_y * local_x_mm + uy_y * local_y_mm;
            auto position_z_mm =
                origin_z + ux_z * local_x_mm + uy_z * local_y_mm;
            auto direction_x =
                ux_x * local_dx + uy_x * local_dy + uz_x * local_dz;
            auto direction_y =
                ux_y * local_dx + uy_y * local_dy + uz_y * local_dz;
            auto direction_z =
                ux_z * local_dx + uy_z * local_dy + uz_z * local_dz;
            {
                const auto inv_n = sycl::rsqrt(sycl::fmax(
                    1.0e-20F,
                    direction_x * direction_x + direction_y * direction_y +
                        direction_z * direction_z));
                direction_x *= inv_n;
                direction_y *= inv_n;
                direction_z *= inv_n;
            }
            // TPS-90 entrance sampling: origin lies on the CT face (z=0) but
            // ux/uy for a slightly tilted beam still have small z components, so
            // origin + x*ux + y*uy leaves the z=0 plane.  Project back along the
            // particle direction onto z=0 so births are not biased to one side
            // of the aperture (which shifted patient-Z COM by ~3–4 mm).
            if (sycl::fabs(direction_z) > 1.0e-8F &&
                sycl::fabs(position_z_mm) > 1.0e-6F) {
                const auto t_plane = -position_z_mm / direction_z;
                position_x_mm += t_plane * direction_x;
                position_y_mm += t_plane * direction_y;
                position_z_mm = 0.0F;
            }
            auto history_deposited_MeV = 0.0f;
            auto history_nuclear_MeV = 0.0f;
            SecondaryGenerationSummary secondary_summary{};
            std::uint32_t steps = 0;
            double pending_primary_depth_MeV = 0.0;
            int pending_primary_bin = 0;
            double pending_primary_voxel_MeV = 0.0;
            std::size_t pending_primary_voxel = 0;
            constexpr std::uint32_t max_primary_steps = 2'000'000U;
            while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                const auto escaped_z =
                    position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                const auto escaped_xy =
                    enable_voxel_scoring &&
                    (position_x_mm < voxel_min_x_mm || position_x_mm >= voxel_max_x_mm ||
                     position_y_mm < voxel_min_y_mm || position_y_mm >= voxel_max_y_mm);
                if (escaped_z || escaped_xy) {
                    break;
                }

                const auto absolute_direction_x = sycl::fabs(direction_x);
                const auto absolute_direction_y = sycl::fabs(direction_y);
                const auto absolute_direction_z = sycl::fabs(direction_z);
                auto bin = direction_z < 0.0F
                               ? static_cast<int>(
                                     sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                               : static_cast<int>(
                                     sycl::floor(position_z_mm / depth_bin_width_mm));
                bin = sycl::max(
                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                if (enable_voxel_scoring) {
                    const auto x_coordinate =
                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                    const auto y_coordinate =
                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                    voxel_x = direction_x < 0.0F
                                  ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                  : static_cast<int>(sycl::floor(x_coordinate));
                    voxel_y = direction_y < 0.0F
                                  ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                  : static_cast<int>(sycl::floor(y_coordinate));
                    voxel_x = sycl::max(
                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                    voxel_y = sycl::max(
                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                }
                const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                         static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                         static_cast<std::size_t>(voxel_x);
                const auto energy_MeVu = energy_MeV * inverse_mass_number;
                auto floating_index = (energy_MeVu - minimum_table_energy) * inverse_table_step;
                auto index = static_cast<int>(sycl::floor(floating_index));
                index = sycl::max(0, sycl::min(index, static_cast<int>(table_size) - 2));
                const auto fraction = sycl::clamp(floating_index - static_cast<float>(index),
                                                  0.0f, 1.0f);
                const auto in_insert =
                    enable_hetero_insert &&
                    inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                         insert_x_min, insert_x_max, insert_y_min,
                                         insert_y_max, insert_z_min, insert_z_max);
                auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                    position_z_mm, slab_z_ends_device, slab_densities_device,
                    slab_layer_count, water_density_g_per_cm3);
                if (in_insert) {
                    local_density_g_per_cm3 = insert_density_g_per_cm3;
                }
                std::uint8_t ct_material = 2;
                auto in_ct = false;
                if (enable_ct_grid) {
                    float ct_rho = water_density_g_per_cm3;
                    in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                      ct_origin_x, ct_origin_y, ct_origin_z, ct_spacing_x,
                                      ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz,
                                      ct_density_device, ct_material_device, ct_rho,
                                      ct_material);
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_ct_samples);
                    if (in_ct) {
                        local_density_g_per_cm3 = ct_rho;
                    }
                }
                const auto layer_for_material =
                    slab_layer_count > 0
                        ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                           slab_layer_count)
                        : 0U;
                float stopping_power_MeV_per_mm = 0.0F;
                if (enable_ct_grid) {
                    const auto water_sp =
                        table_device[index] +
                        fraction * (table_device[index + 1] - table_device[index]);
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_sp_table_lookups);
                    if (use_ct_mass_sp && in_ct && ct_mass_sp_factor_lut_device != nullptr &&
                        ct_n_mass_factors > 0) {
                        // Precomputed Schneider mass-SP factor LUT × density (P2).
                        const auto sec = static_cast<std::uint32_t>(ct_material);
                        const auto fi =
                            sec < ct_n_mass_factors ? sec : (ct_n_mass_factors - 1U);
                        const auto base = static_cast<std::size_t>(fi) * table_size +
                                          static_cast<std::size_t>(index);
                        const auto mass_factor =
                            ct_mass_sp_factor_lut_device[base] +
                            fraction * (ct_mass_sp_factor_lut_device[base + 1U] -
                                        ct_mass_sp_factor_lut_device[base]);
                        stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                            water_sp, local_density_g_per_cm3, mass_factor);
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::primary_mass_sp_lookups);
                    } else if (use_ct_material_sp && in_ct && ct_sp_device != nullptr &&
                               ct_ref_density_device != nullptr) {
                        // Legacy absolute material SP × (local ρ / ρ_ref).
                        const auto mat = static_cast<std::uint32_t>(ct_material_class(
                            ct_material, ct_material_ids_are_schneider_sections));
                        const auto base = mat * table_size;
                        const auto sp_abs =
                            ct_sp_device[base + static_cast<std::size_t>(index)] +
                            fraction *
                                (ct_sp_device[base + static_cast<std::size_t>(index) + 1] -
                                 ct_sp_device[base + static_cast<std::size_t>(index)]);
                        const auto ref_rho =
                            sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                        stopping_power_MeV_per_mm =
                            sp_abs *
                            (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) / ref_rho);
                    } else {
                        // Water-equivalent: water SP × local density.
                        stopping_power_MeV_per_mm =
                            water_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                    }
                } else if (in_insert && use_insert_material_tables) {
                    stopping_power_MeV_per_mm =
                        insert_sp_device[static_cast<std::size_t>(index)] +
                        fraction *
                            (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                             insert_sp_device[static_cast<std::size_t>(index)]);
                } else if (material_table_count > 0) {
                    const auto base = static_cast<std::size_t>(layer_for_material) * table_size;
                    stopping_power_MeV_per_mm =
                        material_sp_device[base + static_cast<std::size_t>(index)] +
                        fraction *
                            (material_sp_device[base + static_cast<std::size_t>(index) + 1] -
                             material_sp_device[base + static_cast<std::size_t>(index)]);
                } else {
                    const auto table_stopping_power_MeV_per_mm =
                        table_device[index] +
                        fraction * (table_device[index + 1] - table_device[index]);
                    // Density-only hetero: scale water SP by local ρ (insert or slab).
                    const auto scale_density =
                        (slab_layer_count > 0 || in_insert) ? local_density_g_per_cm3
                                                            : 1.0F;
                    stopping_power_MeV_per_mm =
                        (slab_layer_count > 0 || in_insert)
                            ? table_stopping_power_MeV_per_mm * scale_density
                            : table_stopping_power_MeV_per_mm;
                }

                auto step_mm = sycl::fmin(
                    maximum_step_mm,
                    maximum_relative_energy_loss * energy_MeV /
                        sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                if (absolute_direction_z >= 1.0e-6F) {
                    const auto boundary_z_mm =
                        direction_z < 0.0F
                            ? static_cast<float>(bin) * depth_bin_width_mm
                            : static_cast<float>(bin + 1) * depth_bin_width_mm;
                    const auto dz_step = (boundary_z_mm - position_z_mm) / direction_z;
                    if (dz_step > 0.0F) {
                        step_mm = sycl::fmin(step_mm, dz_step);
                    }
                }
                if (slab_layer_count > 0) {
                    const auto slab_step = distance_to_slab_interface_mm(
                        position_z_mm, direction_z, slab_z_ends_device, slab_layer_count,
                        phantom_length_mm);
                    step_mm = sycl::fmin(step_mm, slab_step);
                }
                if (enable_hetero_insert) {
                    const auto insert_step = distance_to_insert_interface_mm(
                        position_x_mm, position_y_mm, position_z_mm, direction_x,
                        direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                        insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                    step_mm = sycl::fmin(step_mm, insert_step);
                }
                if (enable_ct_grid && in_ct) {
                    // Face clamp only when density/material changes along the step
                    // (unless ct_skip_homogeneous_face_clamp is false).
                    CtClampPath clamp_path = CtClampPath::three_axis;
                    step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                        step_mm, position_x_mm, position_y_mm, position_z_mm,
                        direction_x, direction_y, direction_z, ct_origin_x,
                        ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                        ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                        ct_material_device, local_density_g_per_cm3, ct_material,
                        ct_skip_homogeneous_face_clamp,
#ifdef CARBON_TRANSPORT_PROFILE
                        &clamp_path
#else
                        nullptr
#endif
                    );
                    profile_face(profile_counters_device, true, clamp_path);
                }
                if (enable_voxel_scoring && absolute_direction_x >= 1.0e-6F) {
                    const auto boundary_x_mm =
                        voxel_min_x_mm +
                        static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                            voxel_size_x_mm;
                    step_mm = sycl::fmin(
                        step_mm, (boundary_x_mm - position_x_mm) / direction_x);
                }
                if (enable_voxel_scoring && absolute_direction_y >= 1.0e-6F) {
                    const auto boundary_y_mm =
                        voxel_min_y_mm +
                        static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                            voxel_size_y_mm;
                    step_mm = sycl::fmin(
                        step_mm, (boundary_y_mm - position_y_mm) / direction_y);
                }
                // Accept CT micro-steps (DDA already avoids zero-length face clamps).
                // Only snap/nudge when the step is non-positive or non-CT thrash.
                const auto min_step_accept =
                    (enable_ct_grid && in_ct) ? 1.0e-8F : 1.0e-6F;
                if (step_mm <= min_step_accept) {
                    constexpr auto infinity =
                        std::numeric_limits<float>::infinity();
                    auto snapped_to_boundary = false;
                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if ((boundary_z_mm - position_z_mm) / direction_z <= 1.0e-6F) {
                            position_z_mm = sycl::nextafter(
                                boundary_z_mm, direction_z > 0.0F ? infinity : -infinity);
                            snapped_to_boundary = true;
                        }
                    }
                    if (slab_layer_count > 0 && absolute_direction_z >= 1.0e-6F) {
                        const auto layer = slab_layer_index(
                            position_z_mm, slab_z_ends_device, slab_layer_count);
                        const auto interface_z =
                            direction_z > 0.0F
                                ? slab_z_ends_device[layer]
                                : (layer == 0U ? 0.0F : slab_z_ends_device[layer - 1U]);
                        if (sycl::fabs((interface_z - position_z_mm) / direction_z) <=
                            1.0e-6F) {
                            position_z_mm = sycl::nextafter(
                                interface_z, direction_z > 0.0F ? infinity : -infinity);
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max,
                            insert_y_min, insert_y_max, insert_z_min, insert_z_max,
                            phantom_length_mm);
                        if (insert_step <= 1.0e-6F) {
                            // Nudge along the ray to leave/enter the insert AABB.
                            constexpr float nudge = 1.0e-4F;
                            position_x_mm += direction_x * nudge;
                            position_y_mm += direction_y * nudge;
                            position_z_mm += direction_z * nudge;
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_ct_grid && in_ct) {
                        // Particle sitting on a CT face with vanishing step: nudge past.
                        constexpr float nudge = 1.0e-4F;
                        position_x_mm += direction_x * nudge;
                        position_y_mm += direction_y * nudge;
                        position_z_mm += direction_z * nudge;
                        snapped_to_boundary = true;
                    }
                    if (enable_voxel_scoring && absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(
                                voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        if ((boundary_x_mm - position_x_mm) / direction_x <= 1.0e-6F) {
                            position_x_mm = sycl::nextafter(
                                boundary_x_mm,
                                direction_x > 0.0F ? infinity : -infinity);
                            snapped_to_boundary = true;
                        }
                    }
                    if (enable_voxel_scoring && absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(
                                voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        if ((boundary_y_mm - position_y_mm) / direction_y <= 1.0e-6F) {
                            position_y_mm = sycl::nextafter(
                                boundary_y_mm,
                                direction_y > 0.0F ? infinity : -infinity);
                            snapped_to_boundary = true;
                        }
                    }
                    if (snapped_to_boundary) {
                        ++steps;
                        profile_add(
                            profile_counters_device,
                            TransportProfileSlot::primary_boundary_nudge_continues);
                        continue;
                    }
                    break;
                }

                const auto mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);
                if (enable_energy_straggling) {
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_straggling);
                    const auto uniform1 = sycl::fmax(
                        rng::uniform01(spot_seed, rng_history, steps, 0), 1.0e-12f);
                    const auto uniform2 = rng::uniform01(spot_seed, rng_history, steps, 1);
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
                        effective_charge * water_Z_over_A * local_density_g_per_cm3 *
                        (step_mm / 10.0f);
                    const auto sigma_MeV =
                        straggling_scale * sycl::sqrt(sycl::fmax(0.0f, variance_MeV2));
                    deposited_MeV = sycl::clamp(
                        mean_loss_MeV + sigma_MeV * gaussian, 0.0f, energy_MeV);
                }
                // Electronic build-up: local (1-f)*dE + short-range delta f*dE along +z.
                // Equilibrium dose still ≈ full unrestricted SP; surface suppressed.
                const auto e_frac = electronic_buildup_fraction_at_energy(
                    energy_MeVu, electronic_buildup_fraction);
                const auto delayed_MeV = deposited_MeV * e_frac;
                const auto local_MeV = deposited_MeV - delayed_MeV;
                if (pending_primary_depth_MeV > 0.0 && pending_primary_bin != bin) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                    profile_add(profile_counters_device,
                                TransportProfileSlot::primary_dose_depth_atomics);
                    pending_primary_depth_MeV = 0.0;
                }
                if (pending_primary_depth_MeV == 0.0) {
                    pending_primary_bin = bin;
                }
                pending_primary_depth_MeV += static_cast<double>(local_MeV);
                if (delayed_MeV > 0.0F) {
                    const auto z_mid = position_z_mm + 0.5F * direction_z * step_mm;
                    score_exponential_depth(
                        delayed_MeV, z_mid, depth_bin_width_mm, phantom_length_mm,
                        static_cast<std::uint32_t>(number_of_bins),
                        electronic_buildup_mfp_mm, dose_device);
                }
                if (enable_voxel_scoring) {
                    // Aggregate consecutive deposits in one voxel before the
                    // expensive global FP64 atomic update.
                    if (pending_primary_voxel_MeV > 0.0 &&
                        pending_primary_voxel != voxel_index) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_voxel_dose(
                                voxel_dose_device[pending_primary_voxel]);
                        atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::primary_dose_voxel_atomics);
                        if (enable_charged_origin_voxel_scoring) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_category_dose(
                                    charged_origin_voxel_dose_device[
                                        pending_primary_voxel]);
                            atomic_category_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        }
                        pending_primary_voxel_MeV = 0.0;
                    }
                    if (pending_primary_voxel_MeV == 0.0) {
                        pending_primary_voxel = voxel_index;
                    }
                    pending_primary_voxel_MeV += static_cast<double>(deposited_MeV);
                }
                history_deposited_MeV += deposited_MeV;
                const auto scattering_energy_MeV =
                    energy_MeV - 0.5F * deposited_MeV;
                energy_MeV -= deposited_MeV;
                position_x_mm += direction_x * step_mm;
                position_y_mm += direction_y * step_mm;
                position_z_mm += direction_z * step_mm;
                if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
                    profile_add(profile_counters_device, TransportProfileSlot::primary_mcs);
                    const auto projected_rms_angle_rad =
                        highland_projected_rms_angle_device(
                            scattering_energy_MeV, 6, primary_mass_number, step_mm,
                            local_density_g_per_cm3);
                    const auto scattered = scatter_direction(
                        Direction3F{direction_x, direction_y, direction_z},
                        projected_rms_angle_rad, spot_seed, rng_history, steps, 4);
                    direction_x = scattered.x;
                    direction_y = scattered.y;
                    direction_z = scattered.z;
                }
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
                    float macroscopic_cross_section_per_mm = 0.0F;
                    if (enable_ct_grid) {
                        if (use_ct_material_xs && in_ct && ct_xs_device != nullptr &&
                            ct_ref_density_device != nullptr) {
                            const auto mat = static_cast<std::uint32_t>(ct_material_class(
                                ct_material, ct_material_ids_are_schneider_sections));
                            const auto base = mat * cross_section_table_size;
                            const auto xs_abs =
                                ct_xs_device[base +
                                             static_cast<std::size_t>(cross_section_index)] +
                                cross_section_fraction *
                                    (ct_xs_device[base + static_cast<std::size_t>(
                                                              cross_section_index) +
                                                  1] -
                                     ct_xs_device[base + static_cast<std::size_t>(
                                                              cross_section_index)]);
                            const auto ref_rho =
                                sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                            macroscopic_cross_section_per_mm =
                                xs_abs *
                                (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) / ref_rho);
                        } else {
                            const auto water_xs =
                                cross_section_device[cross_section_index] +
                                cross_section_fraction *
                                    (cross_section_device[cross_section_index + 1] -
                                     cross_section_device[cross_section_index]);
                            macroscopic_cross_section_per_mm =
                                water_xs * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        }
                    } else if (in_insert && use_insert_material_tables) {
                        macroscopic_cross_section_per_mm =
                            insert_xs_device[static_cast<std::size_t>(cross_section_index)] +
                            cross_section_fraction *
                                (insert_xs_device[static_cast<std::size_t>(
                                                      cross_section_index) +
                                                  1] -
                                 insert_xs_device[static_cast<std::size_t>(
                                     cross_section_index)]);
                    } else if (material_table_count > 0) {
                        const auto base = static_cast<std::size_t>(layer_for_material) *
                                          cross_section_table_size;
                        macroscopic_cross_section_per_mm =
                            material_xs_device[base + static_cast<std::size_t>(
                                                          cross_section_index)] +
                            cross_section_fraction *
                                (material_xs_device[base + static_cast<std::size_t>(
                                                                cross_section_index) +
                                                    1] -
                                 material_xs_device[base + static_cast<std::size_t>(
                                                                cross_section_index)]);
                    } else {
                        macroscopic_cross_section_per_mm =
                            cross_section_device[cross_section_index] +
                            cross_section_fraction *
                                (cross_section_device[cross_section_index + 1] -
                                 cross_section_device[cross_section_index]);
                        if (slab_layer_count > 0 || in_insert) {
                            macroscopic_cross_section_per_mm *= local_density_g_per_cm3;
                        }
                    }
                    const auto probability = 1.0f - sycl::exp(
                        -macroscopic_cross_section_per_mm * step_mm);
                    const auto uniform = rng::uniform01(spot_seed, rng_history, steps, 2);
                    if (uniform < probability) {
                        history_nuclear_MeV = energy_MeV;
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::primary_nuclear);
                        if (enable_secondary_generation) {
                            auto reaction_bin_index = static_cast<int>(sycl::floor(
                                (post_step_energy_MeVu - minimum_reaction_energy) *
                                inverse_reaction_energy_bin_width));
                            reaction_bin_index = sycl::max(
                                0, sycl::min(reaction_bin_index,
                                             static_cast<int>(reaction_energy_bin_count) - 1));
                            const auto reaction_bin = reaction_bins_device[reaction_bin_index];
                            const auto package_uniform =
                                rng::uniform01(spot_seed, rng_history, steps, 3);
                            const auto package_in_bin = sycl::min(
                                static_cast<std::uint32_t>(
                                    package_uniform * reaction_bin.reaction_count),
                                reaction_bin.reaction_count - 1U);
                            const auto reaction =
                                reactions_device[reaction_bin.reaction_offset + package_in_bin];
                            secondary_summary.direct_count = reaction.secondary_count;

                            std::uint32_t queueable_count = 0;
                            auto queueable_energy_MeV = 0.0F;
                            std::uint32_t neutral_queueable_count = 0;
                            auto neutral_queueable_energy_MeV = 0.0F;
                            auto package_secondary_ke_MeV = 0.0F;
                            for (std::uint32_t secondary_index = 0;
                                 secondary_index < reaction.secondary_count;
                                 ++secondary_index) {
                                const auto secondary = reaction_secondaries_device[
                                    reaction.secondary_offset + secondary_index];
                                package_secondary_ke_MeV += secondary.kinetic_energy_MeV;
                                const auto is_neutral = secondary.pdg_id == 22 ||
                                                        secondary.pdg_id == 2112;
                                const auto is_supported = secondary.atomic_number > 0 &&
                                                          secondary.mass_number > 0;
                                if (is_neutral) {
                                    if (enable_neutral_transport) {
                                        ++neutral_queueable_count;
                                        neutral_queueable_energy_MeV +=
                                            secondary.kinetic_energy_MeV;
                                    } else {
                                        const auto kerma_frac = neutral_kerma_fraction_at_energy(
                                            energy_MeVu, neutral_local_kerma_fraction,
                                            neutral_kerma_high_energy_scale);
                                        const auto kerma_MeV =
                                            secondary.kinetic_energy_MeV * kerma_frac;
                                        const auto residual_MeV =
                                            secondary.kinetic_energy_MeV - kerma_MeV;
                                        secondary_summary.neutral_energy_MeV += residual_MeV;
                                        if (kerma_MeV > 0.0F &&
                                            fragment_dose_device != nullptr) {
                                            // Interim n/γ kerma into "other" fragment channel,
                                            // distributed along +z; renormalize into phantom.
                                            constexpr std::size_t other_species = 6;
                                            score_exponential_depth(
                                                kerma_MeV, position_z_mm, depth_bin_width_mm,
                                                phantom_length_mm,
                                                static_cast<std::uint32_t>(number_of_bins),
                                                neutral_kerma_mean_free_path_mm,
                                                fragment_dose_device +
                                                    other_species * number_of_bins,
                                                true);
                                            if (enable_voxel_scoring) {
                                                score_exponential_voxel_depth(
                                                    kerma_MeV, position_z_mm,
                                                    depth_bin_width_mm, phantom_length_mm,
                                                    static_cast<std::uint32_t>(number_of_bins),
                                                    neutral_kerma_mean_free_path_mm,
                                                    voxel_plane_size, voxel_index,
                                                    voxel_dose_device, true);
                                            }
                                        }
                                    }
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
                                            const auto child_direction = rotate_local_direction(
                                                secondary.direction_x,
                                                secondary.direction_y,
                                                secondary.direction_z,
                                                Direction3F{direction_x, direction_y, direction_z});
                                            secondary_queue_device[output_index++] =
                                                SecondaryParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    secondary.kinetic_energy_MeV,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    secondary.pdg_id,
                                                    secondary.atomic_number,
                                                    secondary.mass_number,
                                                    origin_category,
                                                    0,
                                                    charged_lineage,
                                                    rng::child_stream(
                                                        global_history,
                                                        rng::branch_tag(
                                                            rng::branch_role_primary_charged,
                                                            secondary_index)),
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
                            if (neutral_queueable_count > 0) {
                                sycl::atomic_ref<
                                    std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    neutral_counter(*neutral_queue_counter_device);
                                const auto neutral_offset =
                                    neutral_counter.fetch_add(neutral_queueable_count);
                                const auto neutral_fits =
                                    neutral_offset <= neutral_queue_capacity_u32 &&
                                    neutral_queueable_count <=
                                        neutral_queue_capacity_u32 - neutral_offset;
                                if (neutral_fits) {
                                    auto output_index = neutral_offset;
                                    for (std::uint32_t secondary_index = 0;
                                         secondary_index < reaction.secondary_count;
                                         ++secondary_index) {
                                        const auto secondary = reaction_secondaries_device[
                                            reaction.secondary_offset + secondary_index];
                                        if (secondary.pdg_id == 22 ||
                                            secondary.pdg_id == 2112) {
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    secondary.direction_x,
                                                    secondary.direction_y,
                                                    secondary.direction_z,
                                                    Direction3F{direction_x, direction_y,
                                                                direction_z});
                                            const auto lineage =
                                                neutral_lineage_from_pdg(secondary.pdg_id);
                                            neutral_queue_device[output_index++] =
                                                NeutralParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    secondary.kinetic_energy_MeV,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    secondary.pdg_id,
                                                    static_cast<std::uint8_t>(
                                                        neutral_origin_category_from_lineage(
                                                            lineage)),
                                                    0,
                                                    0,
                                                    rng::child_stream(
                                                        global_history,
                                                        rng::branch_tag(
                                                            rng::branch_role_primary_neutral,
                                                            secondary_index)),
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<
                                        std::uint64_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        filled_counter(*neutral_queue_filled_device);
                                    filled_counter.fetch_add(neutral_queueable_count);
                                    secondary_summary.queued_neutral_count =
                                        neutral_queueable_count;
                                    secondary_summary.queued_neutral_energy_MeV =
                                        neutral_queueable_energy_MeV;
                                } else {
                                    secondary_summary.neutral_queue_overflow_count =
                                        neutral_queueable_count;
                                    secondary_summary.neutral_queue_overflow_energy_MeV =
                                        neutral_queueable_energy_MeV;
                                    secondary_summary.neutral_energy_MeV +=
                                        neutral_queueable_energy_MeV;
                                }
                            }
                            // Residual nucleus / reaction Q not carried by listed
                            // secondaries: deposit locally (G4 heavy residual heat).
                            const auto residual_nuclear_MeV = sycl::fmax(
                                0.0F, energy_MeV - package_secondary_ke_MeV);
                            if (residual_nuclear_MeV > 0.0F) {
                                auto rbin = direction_z < 0.0F
                                                ? static_cast<int>(sycl::ceil(
                                                      position_z_mm /
                                                      depth_bin_width_mm)) -
                                                      1
                                                : static_cast<int>(sycl::floor(
                                                      position_z_mm /
                                                      depth_bin_width_mm));
                                rbin = sycl::max(
                                    0, sycl::min(rbin,
                                                 static_cast<int>(number_of_bins) - 1));
                                sycl::atomic_ref<
                                    double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_r(dose_device[rbin]);
                                atomic_r.fetch_add(
                                    static_cast<double>(residual_nuclear_MeV));
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) /
                                        voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) /
                                        voxel_size_y_mm;
                                    auto voxel_x =
                                        direction_x < 0.0F
                                            ? static_cast<int>(
                                                  sycl::ceil(x_coordinate)) -
                                                  1
                                            : static_cast<int>(
                                                  sycl::floor(x_coordinate));
                                    auto voxel_y =
                                        direction_y < 0.0F
                                            ? static_cast<int>(
                                                  sycl::ceil(y_coordinate)) -
                                                  1
                                            : static_cast<int>(
                                                  sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0, sycl::min(voxel_x,
                                                     static_cast<int>(voxel_bins_x) -
                                                         1));
                                    voxel_y = sycl::max(
                                        0, sycl::min(voxel_y,
                                                     static_cast<int>(voxel_bins_y) -
                                                         1));
                                    const auto vidx =
                                        static_cast<std::size_t>(rbin) *
                                            voxel_plane_size +
                                        static_cast<std::size_t>(voxel_y) *
                                            voxel_bins_x +
                                        static_cast<std::size_t>(voxel_x);
                                    sycl::atomic_ref<
                                        double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        atomic_v(voxel_dose_device[vidx]);
                                    atomic_v.fetch_add(
                                        static_cast<double>(residual_nuclear_MeV));
                                    if (enable_charged_origin_voxel_scoring) {
                                        // Count residual heat with primary C-12
                                        // origin so charged-origin categories close.
                                        sycl::atomic_ref<
                                            double, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            atomic_cat(
                                                charged_origin_voxel_dose_device[vidx]);
                                        atomic_cat.fetch_add(static_cast<double>(
                                            residual_nuclear_MeV));
                                    }
                                }
                            }
                        }
                        energy_MeV = 0.0f;
                    }
                }
                ++steps;
                profile_add(profile_counters_device, TransportProfileSlot::primary_steps);
            }

            const auto stopped_inside =
                energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                position_z_mm < phantom_length_mm &&
                (!enable_voxel_scoring ||
                 (position_x_mm >= voxel_min_x_mm && position_x_mm < voxel_max_x_mm &&
                  position_y_mm >= voxel_min_y_mm && position_y_mm < voxel_max_y_mm));
            if (stopped_inside) {
                auto bin = direction_z < 0.0F
                               ? static_cast<int>(
                                     sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                               : static_cast<int>(
                                     sycl::floor(position_z_mm / depth_bin_width_mm));
                bin = sycl::max(
                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                sycl::atomic_ref<DoseAtomicT,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[bin]);
                atomic_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                if (enable_voxel_scoring) {
                    const auto x_coordinate =
                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                    const auto y_coordinate =
                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                    auto voxel_x = direction_x < 0.0F
                                       ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                       : static_cast<int>(sycl::floor(x_coordinate));
                    auto voxel_y = direction_y < 0.0F
                                       ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                       : static_cast<int>(sycl::floor(y_coordinate));
                    voxel_x = sycl::max(
                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                    voxel_y = sycl::max(
                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                    const auto voxel_index =
                        static_cast<std::size_t>(bin) * voxel_plane_size +
                        static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                        static_cast<std::size_t>(voxel_x);
                    sycl::atomic_ref<DoseAtomicT,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_voxel_dose(voxel_dose_device[voxel_index]);
                    atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                    if (enable_charged_origin_voxel_scoring) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_category_dose(
                                charged_origin_voxel_dose_device[voxel_index]);
                        atomic_category_dose.fetch_add(static_cast<DoseAtomicT>(energy_MeV));
                    }
                }
                history_deposited_MeV += energy_MeV;
                energy_MeV = 0.0F;
            }
            if (pending_primary_depth_MeV > 0.0) {
                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[pending_primary_bin]);
                atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                profile_add(profile_counters_device,
                            TransportProfileSlot::primary_dose_depth_atomics);
            }
            if (enable_voxel_scoring && pending_primary_voxel_MeV > 0.0) {
                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                profile_add(profile_counters_device,
                            TransportProfileSlot::primary_dose_voxel_atomics);
                if (enable_charged_origin_voxel_scoring) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_category_dose(
                            charged_origin_voxel_dose_device[pending_primary_voxel]);
                    atomic_category_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                }
            }
            deposited_device[global_history] = history_deposited_MeV;
            escaped_device[global_history] = energy_MeV;
            nuclear_device[global_history] = history_nuclear_MeV;
            steps_device[global_history] = steps;
            if (enable_secondary_generation) {
                secondary_summaries_device[global_history] = secondary_summary;
            }
        });
        kernel_event.wait_and_throw();
        primary_kernel_seconds += event_duration_seconds(kernel_event);
        const auto primary_done = hist_offset + chunk_count;
        if (primary_done == number_of_histories ||
            hist_offset == 0 ||
            (progress_log_every_histories > 0 &&
             primary_done / progress_log_every_histories !=
                 hist_offset / progress_log_every_histories)) {
            std::cout << "  primary: " << primary_done << '/' << number_of_histories
                      << " histories\n"
                      << std::flush;
        }
    }

    std::uint64_t transported_queue_count = 0;
    if (enable_secondary_transport) {
        std::uint64_t generation_begin = 0;
        std::uint64_t generation_end = 0;
        queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
        std::uint32_t secondary_generation_index = 0;
        while (generation_begin < generation_end) {
            // Capture the generation limit before cascade products extend the queue.
            const auto generation_limit = generation_end;
            std::uint64_t batch_begin = generation_begin;
            std::uint64_t last_logged_secondary = generation_begin;
            std::cout << "  secondary gen " << secondary_generation_index << ": "
                      << generation_limit - generation_begin << " particles ["
                      << generation_begin << ',' << generation_limit << ")\n"
                      << std::flush;
            while (batch_begin < generation_limit) {
            const auto batch_end = std::min<std::uint64_t>(
                batch_begin + static_cast<std::uint64_t>(secondary_batch),
                generation_limit);
            const auto batch_size = batch_end - batch_begin;
            // One secondary particle per work-item; large batches fill CUDA SMs.
            const auto secondary_global_size =
                ((batch_size + local_size - 1) / local_size) * local_size;
            auto secondary_kernel_event = queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{secondary_global_size},
                              sycl::range<1>{local_size}},
            [=](sycl::nd_item<1> item) {
                const auto lane = item.get_global_linear_id();
                if (lane >= batch_size) {
                    return;
                }
                const auto particle_index = batch_begin + lane;
                auto deposited_MeV = 0.0F;
                auto escaped_MeV = 0.0F;
                std::uint32_t steps = 0;
                CascadeTransportSummary cascade_summary{};
                {
                    const auto particle = secondary_queue_device[particle_index];
                    // Philox stream is particle identity, not atomic queue slot.
                    const auto rng_stream = particle.rng_stream;
                    auto energy_MeV = particle.kinetic_energy_MeV;
                    auto position_x_mm = particle.position_x_mm;
                    auto position_y_mm = particle.position_y_mm;
                    auto position_z_mm = particle.position_z_mm;
                    auto direction_x = particle.direction_x;
                    auto direction_y = particle.direction_y;
                    auto direction_z = sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                    const auto atomic_number = static_cast<int>(particle.atomic_number);
                    const auto mass_number = static_cast<int>(particle.mass_number);
                    const auto inverse_mass_number_for_particle =
                        1.0F / static_cast<float>(mass_number);
                    const auto cascade_projectile_index_for_particle =
                        enable_fragment_cascade &&
                                particle.generation < maximum_cascade_generations
                            ? cascade_projectile_index(
                                  cascade_projectiles_device, cascade_projectile_count,
                                  atomic_number, mass_number)
                            : -1;
                    CascadeProjectile cascade_projectile{};
                    if (cascade_projectile_index_for_particle >= 0) {
                        cascade_projectile = cascade_projectiles_device[
                            cascade_projectile_index_for_particle];
                    }
                    const auto charge = static_cast<float>(atomic_number);
                    const auto charge_power = sycl::pow(charge, -2.0F / 3.0F);
                    constexpr float carbon_charge = 6.0F;
                    const auto carbon_charge_power =
                        sycl::pow(carbon_charge, -2.0F / 3.0F);
                    const auto is_neutral_lineage =
                        particle.reserved == neutron_lineage ||
                        particle.reserved == gamma_lineage;
                    const auto neutral_origin =
                        neutral_origin_category_from_lineage(particle.reserved);
                    const auto species_index =
                        sycl::min(static_cast<std::size_t>(particle.origin_category),
                                  fragment_species_count - 1);
                    const auto charged_origin_voxel_offset =
                        (species_index + 1) * number_of_voxels;
                    const auto neutral_origin_voxel_offset =
                        neutral_origin * number_of_voxels;
                    auto pending_dose_MeV = DoseAtomicT{0};
                    auto pending_bin = 0;
                    std::size_t pending_voxel_index = 0;
                    // Primary already caps steps; secondary previously did not, so a
                    // voxel-boundary nudge thrash could run for minutes on CUDA.
                    constexpr std::uint32_t max_secondary_steps = 500'000U;

                    while (energy_MeV > energy_cutoff_MeV && steps < max_secondary_steps) {
                        const auto escaped_z =
                            position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                        const auto escaped_xy =
                            enable_voxel_scoring &&
                            (position_x_mm < voxel_min_x_mm || position_x_mm >= voxel_max_x_mm ||
                             position_y_mm < voxel_min_y_mm || position_y_mm >= voxel_max_y_mm);
                        if (escaped_z || escaped_xy) {
                            break;
                        }
                        const auto absolute_direction_x = sycl::fabs(direction_x);
                        const auto absolute_direction_y = sycl::fabs(direction_y);
                        const auto absolute_direction_z = sycl::fabs(direction_z);

                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_z_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                        auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                        auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                        if (enable_voxel_scoring) {
                            const auto x_coordinate =
                                (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                            const auto y_coordinate =
                                (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                            voxel_x = direction_x < 0.0F
                                          ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(x_coordinate));
                            voxel_y = direction_y < 0.0F
                                          ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(y_coordinate));
                            voxel_x = sycl::max(
                                0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                            voxel_y = sycl::max(
                                0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                        }
                        const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                                 static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                                 static_cast<std::size_t>(voxel_x);

                        if (absolute_direction_x < 1.0e-6F &&
                            absolute_direction_y < 1.0e-6F && absolute_direction_z < 1.0e-6F) {
                            score_secondary_dose_device(
                                pending_dose_MeV, is_neutral_lineage, species_index,
                                neutral_origin, pending_bin, number_of_bins,
                                enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring,
                                pending_voxel_index, charged_origin_voxel_offset,
                                neutral_origin_voxel_offset, fragment_dose_device,
                                voxel_dose_device, charged_origin_voxel_dose_device,
                                neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            pending_dose_MeV = DoseAtomicT{0};
                            score_secondary_dose_device(
                                energy_MeV, is_neutral_lineage, species_index, neutral_origin,
                                bin, number_of_bins, enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring, voxel_index,
                                charged_origin_voxel_offset, neutral_origin_voxel_offset,
                                fragment_dose_device, voxel_dose_device,
                                charged_origin_voxel_dose_device, neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            deposited_MeV += energy_MeV;
                            energy_MeV = 0.0F;
                            break;
                        }

                        const auto energy_MeVu =
                            energy_MeV * inverse_mass_number_for_particle;
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
                        const auto effective_charge =
                            charge *
                            (1.0F - sycl::exp(-125.0F * beta * charge_power));
                        const auto carbon_effective_charge =
                            carbon_charge *
                            (1.0F - sycl::exp(-125.0F * beta * carbon_charge_power));
                        const auto charge_ratio =
                            effective_charge / carbon_effective_charge;
                        const auto in_insert =
                            enable_hetero_insert &&
                            inside_hetero_insert(position_x_mm, position_y_mm, position_z_mm,
                                                 insert_x_min, insert_x_max, insert_y_min,
                                                 insert_y_max, insert_z_min, insert_z_max);
                        auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                            position_z_mm, slab_z_ends_device, slab_densities_device,
                            slab_layer_count, water_density_g_per_cm3);
                        if (in_insert) {
                            local_density_g_per_cm3 = insert_density_g_per_cm3;
                        }
                        std::uint8_t ct_material = 2;
                        auto in_ct = false;
                        if (enable_ct_grid) {
                            float ct_rho = water_density_g_per_cm3;
                            in_ct = ct_sample(position_x_mm, position_y_mm, position_z_mm,
                                              ct_origin_x, ct_origin_y, ct_origin_z,
                                              ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                              ct_nx, ct_ny, ct_nz, ct_density_device,
                                              ct_material_device, ct_rho, ct_material);
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_ct_samples);
                            if (in_ct) {
                                local_density_g_per_cm3 = ct_rho;
                            }
                        }
                        const auto layer_for_material =
                            slab_layer_count > 0
                                ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                                   slab_layer_count)
                                : 0U;
                        float carbon_sp_local = carbon_stopping_power_MeV_per_mm;
                        if (enable_ct_grid) {
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_sp_table_lookups);
                            if (use_ct_mass_sp && in_ct &&
                                ct_mass_sp_factor_lut_device != nullptr &&
                                ct_n_mass_factors > 0) {
                                const auto sec = static_cast<std::uint32_t>(ct_material);
                                const auto fi = sec < ct_n_mass_factors
                                                    ? sec
                                                    : (ct_n_mass_factors - 1U);
                                const auto base = static_cast<std::size_t>(fi) * table_size +
                                                  static_cast<std::size_t>(index);
                                const auto mass_factor =
                                    ct_mass_sp_factor_lut_device[base] +
                                    fraction * (ct_mass_sp_factor_lut_device[base + 1U] -
                                                ct_mass_sp_factor_lut_device[base]);
                                carbon_sp_local = ct_mass_scaled_stopping_power(
                                    carbon_stopping_power_MeV_per_mm,
                                    local_density_g_per_cm3, mass_factor);
                                profile_add(profile_counters_device,
                                            TransportProfileSlot::secondary_mass_sp_lookups);
                            } else if (use_ct_material_sp && in_ct &&
                                       ct_sp_device != nullptr &&
                                       ct_ref_density_device != nullptr) {
                                const auto mat = static_cast<std::uint32_t>(ct_material_class(
                                    ct_material, ct_material_ids_are_schneider_sections));
                                const auto base = mat * table_size;
                                const auto sp_abs =
                                    ct_sp_device[base + static_cast<std::size_t>(index)] +
                                    fraction *
                                        (ct_sp_device[base +
                                                     static_cast<std::size_t>(index) + 1] -
                                         ct_sp_device[base +
                                                     static_cast<std::size_t>(index)]);
                                const auto ref_rho =
                                    sycl::fmax(ct_ref_density_device[mat], 1.0e-6F);
                                carbon_sp_local =
                                    sp_abs *
                                    (sycl::fmax(local_density_g_per_cm3, 1.0e-6F) /
                                     ref_rho);
                            } else {
                                carbon_sp_local = carbon_stopping_power_MeV_per_mm *
                                                  sycl::fmax(local_density_g_per_cm3,
                                                             1.0e-6F);
                            }
                        } else if (in_insert && use_insert_material_tables) {
                            carbon_sp_local =
                                insert_sp_device[static_cast<std::size_t>(index)] +
                                fraction *
                                    (insert_sp_device[static_cast<std::size_t>(index) + 1] -
                                     insert_sp_device[static_cast<std::size_t>(index)]);
                        } else if (material_table_count > 0) {
                            const auto base =
                                static_cast<std::size_t>(layer_for_material) * table_size;
                            carbon_sp_local =
                                material_sp_device[base + static_cast<std::size_t>(index)] +
                                fraction *
                                    (material_sp_device[base +
                                                       static_cast<std::size_t>(index) + 1] -
                                     material_sp_device[base +
                                                       static_cast<std::size_t>(index)]);
                        }
                        auto stopping_power_MeV_per_mm =
                            carbon_sp_local * charge_ratio * charge_ratio;
                        // Density scale only for density-only slab/insert (not absolute
                        // material tables, not CT which already scaled carbon_sp_local).
                        if ((slab_layer_count > 0 || in_insert) && !enable_ct_grid &&
                            !(in_insert && use_insert_material_tables) &&
                            material_table_count == 0) {
                            stopping_power_MeV_per_mm *= local_density_g_per_cm3;
                        }
                        auto path_step_mm = sycl::fmin(
                            maximum_step_mm,
                            maximum_relative_energy_loss * energy_MeV /
                                sycl::fmax(stopping_power_MeV_per_mm, 1.0e-6F));
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        const auto distance_to_boundary_mm =
                            direction_z < 0.0F ? position_z_mm - boundary_z_mm
                                               : boundary_z_mm - position_z_mm;
                        path_step_mm = sycl::fmin(
                            path_step_mm, distance_to_boundary_mm / absolute_direction_z);
                        if (slab_layer_count > 0) {
                            path_step_mm = sycl::fmin(
                                path_step_mm,
                                distance_to_slab_interface_mm(
                                    position_z_mm, direction_z, slab_z_ends_device,
                                    slab_layer_count, phantom_length_mm));
                        }
                        if (enable_hetero_insert) {
                            path_step_mm = sycl::fmin(
                                path_step_mm,
                                distance_to_insert_interface_mm(
                                    position_x_mm, position_y_mm, position_z_mm, direction_x,
                                    direction_y, direction_z, insert_x_min, insert_x_max,
                                    insert_y_min, insert_y_max, insert_z_min, insert_z_max,
                                    phantom_length_mm));
                        }
                        if (enable_ct_grid && in_ct) {
                            CtClampPath clamp_path = CtClampPath::three_axis;
                            path_step_mm = clamp_step_to_ct_faces_if_needed(
                                path_step_mm, position_x_mm, position_y_mm, position_z_mm,
                                direction_x, direction_y, direction_z, ct_origin_x,
                                ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                                ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                                ct_material_device, local_density_g_per_cm3, ct_material,
                                ct_skip_homogeneous_face_clamp,
#ifdef CARBON_TRANSPORT_PROFILE
                                &clamp_path
#else
                                nullptr
#endif
                            );
                            profile_face(profile_counters_device, false, clamp_path);
                        }
                        if (enable_voxel_scoring && absolute_direction_x >= 1.0e-6F) {
                            const auto boundary_x_mm =
                                voxel_min_x_mm +
                                static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                    voxel_size_x_mm;
                            const auto distance_to_boundary_x_mm =
                                (boundary_x_mm - position_x_mm) / direction_x;
                            path_step_mm = sycl::fmin(path_step_mm, distance_to_boundary_x_mm);
                        }
                        if (enable_voxel_scoring && absolute_direction_y >= 1.0e-6F) {
                            const auto boundary_y_mm =
                                voxel_min_y_mm +
                                static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                    voxel_size_y_mm;
                            const auto distance_to_boundary_y_mm =
                                (boundary_y_mm - position_y_mm) / direction_y;
                            path_step_mm = sycl::fmin(path_step_mm, distance_to_boundary_y_mm);
                        }
                        const auto min_step_accept =
                            (enable_ct_grid && in_ct) ? 1.0e-8F : 1.0e-6F;
                        if (path_step_mm <= min_step_accept) {
                            constexpr auto infinity =
                                std::numeric_limits<float>::infinity();
                            auto snapped_to_boundary = false;
                            if (absolute_direction_z >= 1.0e-6F &&
                                distance_to_boundary_mm / absolute_direction_z <= 1.0e-6F) {
                                position_z_mm = sycl::nextafter(
                                    boundary_z_mm, direction_z > 0.0F ? infinity : -infinity);
                                snapped_to_boundary = true;
                            }
                            if (slab_layer_count > 0 && absolute_direction_z >= 1.0e-6F) {
                                const auto layer = slab_layer_index(
                                    position_z_mm, slab_z_ends_device, slab_layer_count);
                                const auto interface_z =
                                    direction_z > 0.0F
                                        ? slab_z_ends_device[layer]
                                        : (layer == 0U ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                if (sycl::fabs((interface_z - position_z_mm) / direction_z) <=
                                    1.0e-6F) {
                                    position_z_mm = sycl::nextafter(
                                        interface_z,
                                        direction_z > 0.0F ? infinity : -infinity);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_voxel_scoring && absolute_direction_x >= 1.0e-6F) {
                                const auto boundary_x_mm =
                                    voxel_min_x_mm +
                                    static_cast<float>(
                                        voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                        voxel_size_x_mm;
                                if ((boundary_x_mm - position_x_mm) / direction_x <= 1.0e-6F) {
                                    position_x_mm = sycl::nextafter(
                                        boundary_x_mm,
                                        direction_x > 0.0F ? infinity : -infinity);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_voxel_scoring && absolute_direction_y >= 1.0e-6F) {
                                const auto boundary_y_mm =
                                    voxel_min_y_mm +
                                    static_cast<float>(
                                        voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                        voxel_size_y_mm;
                                if ((boundary_y_mm - position_y_mm) / direction_y <= 1.0e-6F) {
                                    position_y_mm = sycl::nextafter(
                                        boundary_y_mm,
                                        direction_y > 0.0F ? infinity : -infinity);
                                    snapped_to_boundary = true;
                                }
                            }
                            if (enable_ct_grid && in_ct) {
                                constexpr float nudge = 1.0e-4F;
                                position_x_mm += direction_x * nudge;
                                position_y_mm += direction_y * nudge;
                                position_z_mm += direction_z * nudge;
                                snapped_to_boundary = true;
                            }
                            if (snapped_to_boundary) {
                                ++steps;
                                continue;
                            }
                        }
                        const auto step_deposited_MeV = sycl::fmin(
                            stopping_power_MeV_per_mm * path_step_mm, energy_MeV);
                        const auto sec_e_frac = electronic_buildup_fraction_at_energy(
                            energy_MeVu, electronic_buildup_fraction);
                        const auto sec_delayed = step_deposited_MeV * sec_e_frac;
                        const auto sec_local = step_deposited_MeV - sec_delayed;
                        if (pending_dose_MeV > 0.0 &&
                            (pending_bin != bin ||
                             pending_voxel_index != voxel_index)) {
                            score_secondary_dose_device(
                                pending_dose_MeV, is_neutral_lineage, species_index,
                                neutral_origin, pending_bin, number_of_bins,
                                enable_voxel_scoring,
                                enable_charged_origin_voxel_scoring,
                                pending_voxel_index, charged_origin_voxel_offset,
                                neutral_origin_voxel_offset, fragment_dose_device,
                                voxel_dose_device, charged_origin_voxel_dose_device,
                                neutral_origin_dose_device,
                                neutral_origin_voxel_dose_device);
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_dose_atomics);
                            pending_dose_MeV = DoseAtomicT{0};
                        }
                        if (pending_dose_MeV == 0.0) {
                            pending_bin = bin;
                            pending_voxel_index = voxel_index;
                        }
                        pending_dose_MeV += static_cast<DoseAtomicT>(sec_local);
                        if (sec_delayed > 0.0F && !is_neutral_lineage &&
                            fragment_dose_device != nullptr) {
                            const auto z_mid =
                                position_z_mm + 0.5F * direction_z * path_step_mm;
                            score_exponential_depth(
                                sec_delayed, z_mid, depth_bin_width_mm, phantom_length_mm,
                                static_cast<std::uint32_t>(number_of_bins),
                                electronic_buildup_mfp_mm,
                                fragment_dose_device + species_index * number_of_bins);
                        }
                        deposited_MeV += step_deposited_MeV;
                        const auto scattering_energy_MeV =
                            energy_MeV - 0.5F * step_deposited_MeV;
                        energy_MeV -= step_deposited_MeV;
                        position_x_mm += direction_x * path_step_mm;
                        position_y_mm += direction_y * path_step_mm;
                        position_z_mm += direction_z * path_step_mm;
                        if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
                            profile_add(profile_counters_device,
                                        TransportProfileSlot::secondary_mcs);
                            const auto projected_rms_angle_rad =
                                highland_projected_rms_angle_device(
                                    scattering_energy_MeV, atomic_number, mass_number,
                                    path_step_mm, local_density_g_per_cm3);
                            const auto scattered = scatter_direction(
                                Direction3F{direction_x, direction_y, direction_z},
                                projected_rms_angle_rad, random_seed, rng_stream,
                                steps, 10);
                            direction_x = scattered.x;
                            direction_y = scattered.y;
                            direction_z = scattered.z;
                        }

                        if (cascade_projectile_index_for_particle >= 0 &&
                            energy_MeV > energy_cutoff_MeV &&
                            cascade_xs_lut_device != nullptr) {
                                const auto projectile = cascade_projectile;
                                const auto current_energy_MeVu =
                                    energy_MeV * inverse_mass_number_for_particle;
                                // O(1) LUT on water SP energy grid (same index/fraction
                                // as the stopping-power table).
                                const auto lut_base =
                                    static_cast<std::size_t>(
                                        cascade_projectile_index_for_particle) *
                                        table_size +
                                    static_cast<std::size_t>(index);
                                float macroscopic_cross_section_per_mm =
                                    cascade_xs_lut_device[lut_base] +
                                    fraction * (cascade_xs_lut_device[lut_base + 1U] -
                                                cascade_xs_lut_device[lut_base]);
                                // Density scale for layered phantom or CT (master).
                                if (slab_layer_count > 0 || enable_ct_grid) {
                                    macroscopic_cross_section_per_mm *=
                                        local_density_g_per_cm3;
                                }
                                const auto interaction_probability =
                                    1.0F - sycl::exp(-macroscopic_cross_section_per_mm *
                                                     path_step_mm);
                                if (rng::uniform01(random_seed, rng_stream, steps, 8) <
                                    interaction_probability) {
                                    const auto nearest = nearest_cascade_interaction(
                                        cascade_interactions_device,
                                        projectile.interaction_offset,
                                        projectile.interaction_count,
                                        current_energy_MeVu);
                                    constexpr std::uint32_t sampling_window = 8;
                                    const auto window_begin =
                                        nearest > sampling_window / 2
                                            ? nearest - sampling_window / 2
                                            : 0U;
                                    const auto window_count = sycl::min(
                                        sampling_window,
                                        projectile.interaction_count - window_begin);
                                    const auto package_uniform =
                                        rng::uniform01(random_seed, rng_stream, steps, 9);
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
                                    profile_add(profile_counters_device,
                                                TransportProfileSlot::secondary_cascade);
                                    cascade_summary.direct_count = interaction.product_count;
                                    cascade_summary.incident_energy_MeV = energy_MeV;
                                    std::uint32_t queueable_count = 0;
                                    auto queueable_energy = 0.0F;
                                    std::uint32_t neutral_queueable_count = 0;
                                    auto neutral_queueable_energy = 0.0F;
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
                                            if (enable_neutral_transport) {
                                                ++neutral_queueable_count;
                                                neutral_queueable_energy += scaled_energy;
                                            } else {
                                                const auto kerma_frac =
                                                    neutral_kerma_fraction_at_energy(
                                                        energy_MeVu,
                                                        neutral_local_kerma_fraction,
                                                        neutral_kerma_high_energy_scale);
                                                const auto kerma_MeV =
                                                    scaled_energy * kerma_frac;
                                                const auto residual_MeV =
                                                    scaled_energy - kerma_MeV;
                                                cascade_summary.neutral_energy_MeV +=
                                                    residual_MeV;
                                                if (kerma_MeV > 0.0F &&
                                                    fragment_dose_device != nullptr) {
                                                    constexpr std::size_t other_species = 6;
                                                    score_exponential_depth(
                                                        kerma_MeV, position_z_mm,
                                                        depth_bin_width_mm, phantom_length_mm,
                                                        static_cast<std::uint32_t>(number_of_bins),
                                                        neutral_kerma_mean_free_path_mm,
                                                        fragment_dose_device +
                                                            other_species * number_of_bins,
                                                        true);
                                                    if (enable_voxel_scoring) {
                                                        score_exponential_voxel_depth(
                                                            kerma_MeV, position_z_mm,
                                                            depth_bin_width_mm,
                                                            phantom_length_mm,
                                                            static_cast<std::uint32_t>(
                                                                number_of_bins),
                                                            neutral_kerma_mean_free_path_mm,
                                                            voxel_plane_size, voxel_index,
                                                            voxel_dose_device, true);
                                                    }
                                                }
                                            }
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
                                                    const auto child_direction =
                                                        rotate_local_direction(
                                                            product.direction_x,
                                                            product.direction_y,
                                                            product.direction_z,
                                                            Direction3F{direction_x, direction_y,
                                                                        direction_z});
                                                    secondary_queue_device[output_index++] =
                                                        SecondaryParticle3D{
                                                            position_x_mm,
                                                            position_y_mm,
                                                            position_z_mm,
                                                            product.kinetic_energy_MeV *
                                                                energy_scale,
                                                            child_direction.x,
                                                            child_direction.y,
                                                            child_direction.z,
                                                            product.pdg_id,
                                                            product.atomic_number,
                                                            product.mass_number,
                                                            charged_dose_category(
                                                                product.atomic_number,
                                                                product.mass_number),
                                                            static_cast<std::uint8_t>(
                                                                particle.generation + 1),
                                                            charged_lineage,
                                                            rng::child_stream(
                                                                rng_stream,
                                                                rng::branch_tag(
                                                                    rng::branch_role_cascade_charged,
                                                                    product_index)),
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
                                    if (neutral_queueable_count > 0) {
                                        sycl::atomic_ref<
                                            std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            neutral_counter(*neutral_queue_counter_device);
                                        const auto neutral_offset =
                                            neutral_counter.fetch_add(neutral_queueable_count);
                                        const auto neutral_fits =
                                            neutral_offset <= neutral_queue_capacity_u32 &&
                                            neutral_queueable_count <=
                                                neutral_queue_capacity_u32 - neutral_offset;
                                        if (neutral_fits) {
                                            auto output_index = neutral_offset;
                                            for (std::uint32_t product_index = 0;
                                                 product_index < interaction.product_count;
                                                 ++product_index) {
                                                const auto product = cascade_products_device[
                                                    interaction.product_offset + product_index];
                                                if (product.pdg_id == 22 ||
                                                    product.pdg_id == 2112) {
                                                    const auto child_direction =
                                                        rotate_local_direction(
                                                            product.direction_x,
                                                            product.direction_y,
                                                            product.direction_z,
                                                            Direction3F{direction_x, direction_y,
                                                                        direction_z});
                                                    const auto lineage =
                                                        neutral_lineage_from_pdg(product.pdg_id);
                                                    neutral_queue_device[output_index++] =
                                                        NeutralParticle3D{
                                                            position_x_mm,
                                                            position_y_mm,
                                                            position_z_mm,
                                                            product.kinetic_energy_MeV *
                                                                energy_scale,
                                                            child_direction.x,
                                                            child_direction.y,
                                                            child_direction.z,
                                                            product.pdg_id,
                                                            static_cast<std::uint8_t>(
                                                                neutral_origin_category_from_lineage(
                                                                    lineage)),
                                                            static_cast<std::uint8_t>(
                                                                particle.generation + 1),
                                                            0,
                                                            rng::child_stream(
                                                                rng_stream,
                                                                rng::branch_tag(
                                                                    rng::branch_role_cascade_neutral,
                                                                    product_index)),
                                                        };
                                                }
                                            }
                                            sycl::atomic_ref<
                                                std::uint64_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                filled_counter(*neutral_queue_filled_device);
                                            filled_counter.fetch_add(neutral_queueable_count);
                                            cascade_summary.queued_neutral_count =
                                                neutral_queueable_count;
                                            cascade_summary.queued_neutral_energy_MeV =
                                                neutral_queueable_energy;
                                        } else {
                                            cascade_summary.neutral_queue_overflow_count =
                                                neutral_queueable_count;
                                            cascade_summary.neutral_queue_overflow_energy_MeV =
                                                neutral_queueable_energy;
                                            cascade_summary.neutral_energy_MeV +=
                                                neutral_queueable_energy;
                                        }
                                    }
                                    energy_MeV = 0.0F;
                                }
                        }
                        ++steps;
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::secondary_steps);
                    }

                    score_secondary_dose_device(
                        pending_dose_MeV, is_neutral_lineage, species_index,
                        neutral_origin, pending_bin, number_of_bins,
                        enable_voxel_scoring,
                        enable_charged_origin_voxel_scoring,
                        pending_voxel_index, charged_origin_voxel_offset,
                        neutral_origin_voxel_offset, fragment_dose_device,
                        voxel_dose_device, charged_origin_voxel_dose_device,
                        neutral_origin_dose_device,
                        neutral_origin_voxel_dose_device);
                    if (pending_dose_MeV > 0.0) {
                        profile_add(profile_counters_device,
                                    TransportProfileSlot::secondary_dose_atomics);
                    }
                    {
                        const auto hist_base = static_cast<std::size_t>(
                            TransportProfileSlot::secondary_track_hist_base);
                        const auto bucket = secondary_track_hist_bucket(steps);
                        profile_add(profile_counters_device,
                                    static_cast<TransportProfileSlot>(hist_base + bucket));
                    }
                    const auto stopped_inside =
                        energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                        position_z_mm < phantom_length_mm &&
                        (!enable_voxel_scoring ||
                         (position_x_mm >= voxel_min_x_mm && position_x_mm < voxel_max_x_mm &&
                          position_y_mm >= voxel_min_y_mm && position_y_mm < voxel_max_y_mm)) &&
                        !((direction_z < 0.0F && position_z_mm <= 0.0F) ||
                          (direction_z >= 0.0F && position_z_mm >= phantom_length_mm));
                    if (stopped_inside) {
                        auto bin = direction_z < 0.0F
                                       ? static_cast<int>(
                                             sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                       : static_cast<int>(
                                             sycl::floor(position_z_mm / depth_bin_width_mm));
                        bin = sycl::max(
                            0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                        auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                        auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                        if (enable_voxel_scoring) {
                            const auto x_coordinate =
                                (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                            const auto y_coordinate =
                                (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                            voxel_x = direction_x < 0.0F
                                          ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(x_coordinate));
                            voxel_y = direction_y < 0.0F
                                          ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                          : static_cast<int>(sycl::floor(y_coordinate));
                            voxel_x = sycl::max(
                                0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                            voxel_y = sycl::max(
                                0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                        }
                        const auto voxel_index = static_cast<std::size_t>(bin) * voxel_plane_size +
                                                 static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                                 static_cast<std::size_t>(voxel_x);
                        score_secondary_dose_device(
                            energy_MeV, is_neutral_lineage, species_index, neutral_origin,
                            bin, number_of_bins, enable_voxel_scoring,
                            enable_charged_origin_voxel_scoring, voxel_index,
                            charged_origin_voxel_offset, neutral_origin_voxel_offset,
                            fragment_dose_device, voxel_dose_device,
                            charged_origin_voxel_dose_device, neutral_origin_dose_device,
                            neutral_origin_voxel_dose_device);
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
            const auto batch_seconds = event_duration_seconds(secondary_kernel_event);
            secondary_kernel_seconds += batch_seconds;
            if (batch_end == generation_limit ||
                batch_end - last_logged_secondary >= progress_log_every_secondary) {
                const auto pct =
                    generation_limit > generation_begin
                        ? 100.0 * static_cast<double>(batch_end - generation_begin) /
                              static_cast<double>(generation_limit - generation_begin)
                        : 100.0;
                std::cout << "  secondary: " << batch_end << '/' << generation_limit
                          << " (" << pct << "% gen" << secondary_generation_index
                          << ", last batch " << batch_seconds << " s, cum "
                          << secondary_kernel_seconds << " s)\n"
                          << std::flush;
                last_logged_secondary = batch_end;
            }
            batch_begin = batch_end;
            }  // secondary batch within generation
            transported_queue_count = generation_limit;
            if (!enable_fragment_cascade) {
                break;
            }
            generation_begin = generation_limit;
            queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
            generation_end = std::min<std::uint64_t>(generation_end,
                                                     secondary_queue_capacity);
            ++secondary_generation_index;
        }
    }

    std::uint64_t transported_neutral_count = 0;
    std::uint64_t charged_after_neutral_begin = transported_queue_count;
    if (enable_neutral_transport) {
        std::uint64_t neutral_generation_begin = 0;
        std::uint64_t neutral_generation_end = 0;
        queue.copy(neutral_queue_filled_device, &neutral_generation_end, 1).wait_and_throw();
        std::uint32_t neutral_generation = 0;
        while (neutral_generation_begin < neutral_generation_end &&
               neutral_generation < maximum_neutral_generations) {
            const auto generation_size = neutral_generation_end - neutral_generation_begin;
            const auto neutral_global_size =
                ((generation_size + local_size - 1) / local_size) * local_size;
            auto neutral_kernel_event = queue.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{neutral_global_size},
                                  sycl::range<1>{local_size}},
                [=](sycl::nd_item<1> item) {
                    const auto generation_index = item.get_global_linear_id();
                    if (generation_index >= generation_size) {
                        return;
                    }
                    const auto particle_index = neutral_generation_begin + generation_index;
                    NeutralTransportSummary summary{};
                    std::uint32_t steps = 0;
                    if (particle_index < neutral_generation_end) {
                        const auto particle = neutral_queue_device[particle_index];
                        const auto rng_stream = particle.rng_stream;
                        auto energy_MeV = particle.kinetic_energy_MeV;
                        auto position_x_mm = particle.position_x_mm;
                        auto position_y_mm = particle.position_y_mm;
                        auto position_z_mm = particle.position_z_mm;
                        auto direction_x = particle.direction_x;
                        auto direction_y = particle.direction_y;
                        auto direction_z = sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                        const auto origin_category = static_cast<std::size_t>(
                            sycl::min(static_cast<std::uint32_t>(particle.origin_category),
                                      static_cast<std::uint32_t>(
                                          neutral_origin_category_count - 1)));
                        const auto lineage = particle.pdg_id == 22 ? gamma_lineage
                                                                   : neutron_lineage;
                        constexpr std::uint32_t max_neutral_steps = 500'000U;

                        while (energy_MeV > energy_cutoff_MeV && steps < max_neutral_steps) {
                            const auto escaped_z = position_z_mm < 0.0F ||
                                                   position_z_mm >= phantom_length_mm;
                            const auto escaped_xy =
                                enable_voxel_scoring &&
                                (position_x_mm < voxel_min_x_mm ||
                                 position_x_mm >= voxel_max_x_mm ||
                                 position_y_mm < voxel_min_y_mm ||
                                 position_y_mm >= voxel_max_y_mm);
                            if (escaped_z || escaped_xy) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }

                            int projectile_index = -1;
                            for (std::size_t candidate = 0;
                                 candidate < neutral_projectile_count; ++candidate) {
                                if (neutral_projectiles_device[candidate].pdg_id ==
                                    particle.pdg_id) {
                                    projectile_index = static_cast<int>(candidate);
                                    break;
                                }
                            }
                            if (projectile_index < 0) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }
                            const auto projectile =
                                neutral_projectiles_device[projectile_index];
                            const auto upper = neutral_cross_section_lower_bound(
                                neutral_cross_sections_device,
                                projectile.cross_section_offset,
                                projectile.cross_section_count,
                                energy_MeV);
                            float macroscopic_total_per_mm = 0.0F;
                            if (upper == 0) {
                                macroscopic_total_per_mm =
                                    neutral_cross_sections_device
                                        [projectile.cross_section_offset]
                                            .macroscopic_total_per_mm;
                            } else if (upper >= projectile.cross_section_count) {
                                macroscopic_total_per_mm =
                                    neutral_cross_sections_device
                                        [projectile.cross_section_offset +
                                         projectile.cross_section_count - 1]
                                            .macroscopic_total_per_mm;
                            } else {
                                const auto lower_sample = neutral_cross_sections_device
                                    [projectile.cross_section_offset + upper - 1];
                                const auto upper_sample = neutral_cross_sections_device
                                    [projectile.cross_section_offset + upper];
                                const auto interval =
                                    upper_sample.energy_MeV - lower_sample.energy_MeV;
                                const auto xs_fraction =
                                    interval > 0.0F
                                        ? sycl::clamp((energy_MeV - lower_sample.energy_MeV) /
                                                          interval,
                                                      0.0F, 1.0F)
                                        : 0.0F;
                                macroscopic_total_per_mm =
                                    lower_sample.macroscopic_total_per_mm +
                                    xs_fraction *
                                        (upper_sample.macroscopic_total_per_mm -
                                         lower_sample.macroscopic_total_per_mm);
                            }
                            if (macroscopic_total_per_mm <= 0.0F) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }
                            const auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                                position_z_mm, slab_z_ends_device, slab_densities_device,
                                slab_layer_count, water_density_g_per_cm3);
                            if (slab_layer_count > 0) {
                                macroscopic_total_per_mm *= local_density_g_per_cm3;
                            }

                            const auto free_path_mm =
                                -sycl::log(sycl::fmax(
                                    rng::uniform01(random_seed, rng_stream, steps, 20),
                                    1.0e-12F)) /
                                macroscopic_total_per_mm;
                            if (slab_layer_count > 0) {
                                const auto to_interface = distance_to_slab_interface_mm(
                                    position_z_mm, direction_z, slab_z_ends_device,
                                    slab_layer_count, phantom_length_mm);
                                if (free_path_mm > to_interface && to_interface > 0.0F) {
                                    // Cross interface without interaction; re-sample in new layer.
                                    constexpr auto infinity =
                                        std::numeric_limits<float>::infinity();
                                    position_x_mm += direction_x * to_interface;
                                    position_y_mm += direction_y * to_interface;
                                    position_z_mm += direction_z * to_interface;
                                    if (sycl::fabs(direction_z) >= 1.0e-6F) {
                                        const auto layer = slab_layer_index(
                                            position_z_mm - direction_z * 1.0e-5F,
                                            slab_z_ends_device, slab_layer_count);
                                        const auto interface_z =
                                            direction_z > 0.0F
                                                ? slab_z_ends_device[layer]
                                                : (layer == 0U
                                                       ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                        position_z_mm = sycl::nextafter(
                                            interface_z,
                                            direction_z > 0.0F ? infinity : -infinity);
                                    }
                                    ++steps;
                                    continue;
                                }
                            }
                            position_x_mm += direction_x * free_path_mm;
                            position_y_mm += direction_y * free_path_mm;
                            position_z_mm += direction_z * free_path_mm;
                            ++steps;

                            const auto left_z = position_z_mm < 0.0F ||
                                                position_z_mm >= phantom_length_mm;
                            const auto left_xy =
                                enable_voxel_scoring &&
                                (position_x_mm < voxel_min_x_mm ||
                                 position_x_mm >= voxel_max_x_mm ||
                                 position_y_mm < voxel_min_y_mm ||
                                 position_y_mm >= voxel_max_y_mm);
                            if (left_z || left_xy) {
                                summary.escaped_energy_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                                break;
                            }

                            const auto nearest = nearest_neutral_interaction(
                                neutral_interactions_device,
                                projectile.interaction_offset,
                                projectile.interaction_count,
                                energy_MeV);
                            constexpr std::uint32_t sampling_window = 8;
                            const auto window_begin =
                                nearest > sampling_window / 2
                                    ? nearest - sampling_window / 2
                                    : 0U;
                            const auto window_count = sycl::min(
                                sampling_window,
                                projectile.interaction_count - window_begin);
                            const auto selected_in_window = sycl::min(
                                static_cast<std::uint32_t>(
                                    rng::uniform01(random_seed, rng_stream, steps, 21) *
                                    window_count),
                                window_count - 1U);
                            const auto interaction = neutral_interactions_device
                                [projectile.interaction_offset + window_begin +
                                 selected_in_window];
                            const auto energy_scale =
                                interaction.incident_energy_MeV > 0.0F
                                    ? energy_MeV / interaction.incident_energy_MeV
                                    : 1.0F;
                            summary.interaction_count += 1;
                            const auto local_deposit =
                                interaction.local_deposit_MeV * energy_scale;
                            if (local_deposit > 0.0F) {
                                auto bin = direction_z < 0.0F
                                               ? static_cast<int>(sycl::ceil(
                                                     position_z_mm / depth_bin_width_mm)) -
                                                     1
                                               : static_cast<int>(sycl::floor(
                                                     position_z_mm / depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x = direction_x < 0.0F
                                                  ? static_cast<int>(sycl::ceil(x_coordinate)) -
                                                        1
                                                  : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y = direction_y < 0.0F
                                                  ? static_cast<int>(sycl::ceil(y_coordinate)) -
                                                        1
                                                  : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0, sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0, sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                score_secondary_dose_device(
                                    local_deposit, true, 0, origin_category, bin,
                                    number_of_bins, enable_voxel_scoring, false, voxel_index,
                                    0, origin_category * number_of_voxels, fragment_dose_device,
                                    voxel_dose_device, charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                summary.local_deposit_MeV += local_deposit;
                            }

                            std::uint32_t charged_count = 0;
                            auto charged_energy = 0.0F;
                            for (std::uint32_t product_index = 0;
                                 product_index < interaction.product_count; ++product_index) {
                                const auto product = neutral_products_device
                                    [interaction.product_offset + product_index];
                                const auto scaled =
                                    product.kinetic_energy_MeV * energy_scale;
                                if (product.atomic_number > 0 && product.mass_number > 0 &&
                                    scaled > 0.0F) {
                                    ++charged_count;
                                    charged_energy += scaled;
                                }
                            }
                            if (charged_count > 0) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    queue_counter(*secondary_queue_counter_device);
                                const auto queue_offset = queue_counter.fetch_add(charged_count);
                                const auto package_fits =
                                    queue_offset <= secondary_queue_capacity_u32 &&
                                    charged_count <=
                                        secondary_queue_capacity_u32 - queue_offset;
                                if (package_fits) {
                                    auto output_index = queue_offset;
                                    for (std::uint32_t product_index = 0;
                                         product_index < interaction.product_count;
                                         ++product_index) {
                                        const auto product = neutral_products_device
                                            [interaction.product_offset + product_index];
                                        const auto scaled =
                                            product.kinetic_energy_MeV * energy_scale;
                                        if (product.atomic_number > 0 &&
                                            product.mass_number > 0 && scaled > 0.0F) {
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    product.direction_x, product.direction_y,
                                                    product.direction_z,
                                                    Direction3F{direction_x, direction_y,
                                                                direction_z});
                                            secondary_queue_device[output_index++] =
                                                SecondaryParticle3D{
                                                    position_x_mm,
                                                    position_y_mm,
                                                    position_z_mm,
                                                    scaled,
                                                    child_direction.x,
                                                    child_direction.y,
                                                    child_direction.z,
                                                    product.pdg_id,
                                                    product.atomic_number,
                                                    product.mass_number,
                                                    charged_dose_category(
                                                        product.atomic_number,
                                                        product.mass_number),
                                                    particle.generation,
                                                    lineage,
                                                    rng::child_stream(
                                                        rng_stream,
                                                        rng::branch_tag(
                                                            rng::branch_role_neutral_charged,
                                                            product_index)),
                                                };
                                        }
                                    }
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        filled_counter(*secondary_queue_filled_device);
                                    filled_counter.fetch_add(charged_count);
                                    summary.queued_charged_count += charged_count;
                                    summary.queued_charged_energy_MeV += charged_energy;
                                } else {
                                    summary.charged_overflow_count += charged_count;
                                    summary.charged_overflow_energy_MeV += charged_energy;
                                }
                            }

                            // Nested neutral products stay residual (not re-queued in mode D).
                            for (std::uint32_t product_index = 0;
                                 product_index < interaction.product_count; ++product_index) {
                                const auto product = neutral_products_device
                                    [interaction.product_offset + product_index];
                                if (product.pdg_id == 22 || product.pdg_id == 2112) {
                                    summary.residual_energy_MeV +=
                                        product.kinetic_energy_MeV * energy_scale;
                                }
                            }

                            const auto continuation =
                                interaction.continuation_energy_MeV * energy_scale;
                            // Mode D (first_interaction): free path + one package only.
                            // Continuation kinetic energy becomes residual, not re-queued.
                            if (continuation > 0.0F && neutral_allow_continuation &&
                                continuation > energy_cutoff_MeV &&
                                particle.generation + 1 < maximum_neutral_generations) {
                                const auto child_direction = rotate_local_direction(
                                    interaction.continuation_direction_x,
                                    interaction.continuation_direction_y,
                                    interaction.continuation_direction_z,
                                    Direction3F{direction_x, direction_y, direction_z});
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    neutral_counter(*neutral_queue_counter_device);
                                const auto neutral_offset = neutral_counter.fetch_add(1);
                                if (neutral_offset < neutral_queue_capacity_u32) {
                                    neutral_queue_device[neutral_offset] = NeutralParticle3D{
                                        position_x_mm,
                                        position_y_mm,
                                        position_z_mm,
                                        continuation,
                                        child_direction.x,
                                        child_direction.y,
                                        child_direction.z,
                                        particle.pdg_id,
                                        particle.origin_category,
                                        static_cast<std::uint8_t>(particle.generation + 1),
                                        0,
                                        rng::child_stream(
                                            rng_stream,
                                            rng::branch_tag(
                                                rng::branch_role_neutral_continuation, 0U)),
                                    };
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        filled_counter(*neutral_queue_filled_device);
                                    filled_counter.fetch_add(1);
                                    summary.continuation_count += 1;
                                    summary.continuation_energy_MeV += continuation;
                                } else {
                                    summary.neutral_overflow_count += 1;
                                    summary.neutral_overflow_energy_MeV += continuation;
                                }
                            } else if (continuation > 0.0F) {
                                summary.residual_energy_MeV += continuation;
                            }
                            energy_MeV = 0.0F;
                        }
                    }
                    neutral_summaries_device[particle_index] = summary;
                    (void)steps;
                });
            neutral_kernel_event.wait_and_throw();
            neutral_kernel_seconds += event_duration_seconds(neutral_kernel_event);
            transported_neutral_count = neutral_generation_end;
            neutral_generation_begin = neutral_generation_end;
            queue.copy(neutral_queue_filled_device, &neutral_generation_end, 1)
                .wait_and_throw();
            neutral_generation_end =
                std::min<std::uint64_t>(neutral_generation_end, neutral_queue_capacity);
            ++neutral_generation;
        }

        // Transport charged products created by neutral interactions.
        if (enable_secondary_transport) {
            std::uint64_t generation_end = 0;
            queue.copy(secondary_queue_filled_device, &generation_end, 1).wait_and_throw();
            generation_end =
                std::min<std::uint64_t>(generation_end, secondary_queue_capacity);
            if (generation_end > charged_after_neutral_begin) {
                const auto generation_begin = charged_after_neutral_begin;
                for (std::uint64_t batch_begin = generation_begin; batch_begin < generation_end;
                     batch_begin += static_cast<std::uint64_t>(secondary_batch)) {
                const auto batch_end = std::min<std::uint64_t>(
                    batch_begin + static_cast<std::uint64_t>(secondary_batch),
                    generation_end);
                const auto batch_size = batch_end - batch_begin;
                const auto secondary_global_size =
                    ((batch_size + local_size - 1) / local_size) * local_size;
                auto secondary_kernel_event = queue.parallel_for(
                    sycl::nd_range<1>{sycl::range<1>{secondary_global_size},
                                      sycl::range<1>{local_size}},
                    [=](sycl::nd_item<1> item) {
                        const auto generation_index = item.get_global_linear_id();
                        if (generation_index >= batch_size) {
                            return;
                        }
                        const auto particle_index = batch_begin + generation_index;
                        auto deposited_MeV = 0.0F;
                        auto escaped_MeV = 0.0F;
                        std::uint32_t steps = 0;
                        if (particle_index < generation_end) {
                            const auto particle = secondary_queue_device[particle_index];
                            auto energy_MeV = particle.kinetic_energy_MeV;
                            auto position_x_mm = particle.position_x_mm;
                            auto position_y_mm = particle.position_y_mm;
                            auto position_z_mm = particle.position_z_mm;
                            auto direction_x = particle.direction_x;
                            auto direction_y = particle.direction_y;
                            auto direction_z =
                                sycl::clamp(particle.direction_z, -1.0F, 1.0F);
                            const auto is_neutral_lineage =
                                particle.reserved == neutron_lineage ||
                                particle.reserved == gamma_lineage;
                            const auto neutral_origin =
                                neutral_origin_category_from_lineage(particle.reserved);
                            const auto species_index = sycl::min(
                                static_cast<std::size_t>(particle.origin_category),
                                fragment_species_count - 1);
                            const auto charged_origin_voxel_offset =
                                (species_index + 1) * number_of_voxels;
                            const auto neutral_origin_voxel_offset =
                                neutral_origin * number_of_voxels;
                            const auto atomic_number =
                                static_cast<int>(particle.atomic_number);
                            const auto mass_number = static_cast<int>(particle.mass_number);
                            const auto inverse_mass_number_for_particle =
                                1.0F / static_cast<float>(mass_number);
                            const auto charge = static_cast<float>(atomic_number);
                            const auto charge_power =
                                sycl::pow(charge, -2.0F / 3.0F);
                            constexpr float carbon_charge = 6.0F;
                            const auto carbon_charge_power =
                                sycl::pow(carbon_charge, -2.0F / 3.0F);

                            constexpr std::uint32_t max_secondary_steps = 500'000U;
                            while (energy_MeV > energy_cutoff_MeV &&
                                   steps < max_secondary_steps) {
                                const auto escaped_z =
                                    position_z_mm < 0.0F ||
                                    position_z_mm >= phantom_length_mm;
                                const auto escaped_xy =
                                    enable_voxel_scoring &&
                                    (position_x_mm < voxel_min_x_mm ||
                                     position_x_mm >= voxel_max_x_mm ||
                                     position_y_mm < voxel_min_y_mm ||
                                     position_y_mm >= voxel_max_y_mm);
                                if (escaped_z || escaped_xy) {
                                    break;
                                }
                                auto bin =
                                    direction_z < 0.0F
                                        ? static_cast<int>(sycl::ceil(position_z_mm /
                                                                      depth_bin_width_mm)) -
                                              1
                                        : static_cast<int>(sycl::floor(position_z_mm /
                                                                       depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x =
                                        direction_x < 0.0F
                                            ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y =
                                        direction_y < 0.0F
                                            ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0,
                                        sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0,
                                        sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                const auto energy_MeVu =
                                    energy_MeV * inverse_mass_number_for_particle;
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
                                const auto effective_charge =
                                    charge *
                                    (1.0F - sycl::exp(-125.0F * beta * charge_power));
                                const auto carbon_effective_charge =
                                    carbon_charge *
                                    (1.0F - sycl::exp(
                                                -125.0F * beta * carbon_charge_power));
                                const auto charge_ratio =
                                    effective_charge / carbon_effective_charge;
                                const auto in_insert =
                                    enable_hetero_insert &&
                                    inside_hetero_insert(
                                        position_x_mm, position_y_mm, position_z_mm,
                                        insert_x_min, insert_x_max, insert_y_min,
                                        insert_y_max, insert_z_min, insert_z_max);
                                auto local_density_g_per_cm3 = slab_density_g_per_cm3(
                                    position_z_mm, slab_z_ends_device, slab_densities_device,
                                    slab_layer_count, water_density_g_per_cm3);
                                if (in_insert) {
                                    local_density_g_per_cm3 = insert_density_g_per_cm3;
                                }
                                const auto layer_for_material =
                                    slab_layer_count > 0
                                        ? slab_layer_index(position_z_mm, slab_z_ends_device,
                                                           slab_layer_count)
                                        : 0U;
                                float carbon_sp_local = carbon_stopping_power_MeV_per_mm;
                                if (in_insert && use_insert_material_tables) {
                                    carbon_sp_local =
                                        insert_sp_device[static_cast<std::size_t>(index)] +
                                        fraction *
                                            (insert_sp_device[static_cast<std::size_t>(
                                                                  index) +
                                                              1] -
                                             insert_sp_device[static_cast<std::size_t>(
                                                 index)]);
                                } else if (material_table_count > 0) {
                                    const auto base =
                                        static_cast<std::size_t>(layer_for_material) *
                                        table_size;
                                    carbon_sp_local =
                                        material_sp_device[base +
                                                          static_cast<std::size_t>(index)] +
                                        fraction *
                                            (material_sp_device[base +
                                                               static_cast<std::size_t>(
                                                                   index) +
                                                               1] -
                                             material_sp_device[base +
                                                               static_cast<std::size_t>(
                                                                   index)]);
                                }
                                auto stopping_power_MeV_per_mm =
                                    carbon_sp_local * charge_ratio * charge_ratio;
                                if ((slab_layer_count > 0 || in_insert) &&
                                    !(in_insert && use_insert_material_tables) &&
                                    material_table_count == 0) {
                                    stopping_power_MeV_per_mm *= local_density_g_per_cm3;
                                }
                                auto path_step_mm = sycl::fmin(
                                    maximum_step_mm,
                                    maximum_relative_energy_loss * energy_MeV /
                                        stopping_power_MeV_per_mm);
                                const auto absolute_direction_z = sycl::fabs(direction_z);
                                const auto boundary_z_mm =
                                    direction_z < 0.0F
                                        ? static_cast<float>(bin) * depth_bin_width_mm
                                        : static_cast<float>(bin + 1) * depth_bin_width_mm;
                                const auto distance_to_boundary_mm =
                                    direction_z < 0.0F ? position_z_mm - boundary_z_mm
                                                       : boundary_z_mm - position_z_mm;
                                if (absolute_direction_z >= 1.0e-6F) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_boundary_mm / absolute_direction_z);
                                }
                                if (slab_layer_count > 0) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_slab_interface_mm(
                                            position_z_mm, direction_z, slab_z_ends_device,
                                            slab_layer_count, phantom_length_mm));
                                }
                                if (enable_hetero_insert) {
                                    path_step_mm = sycl::fmin(
                                        path_step_mm,
                                        distance_to_insert_interface_mm(
                                            position_x_mm, position_y_mm, position_z_mm,
                                            direction_x, direction_y, direction_z,
                                            insert_x_min, insert_x_max, insert_y_min,
                                            insert_y_max, insert_z_min, insert_z_max,
                                            phantom_length_mm));
                                }
                                if (path_step_mm <= 1.0e-6F) {
                                    constexpr auto infinity =
                                        std::numeric_limits<float>::infinity();
                                    if (absolute_direction_z >= 1.0e-6F) {
                                        position_z_mm = sycl::nextafter(
                                            boundary_z_mm,
                                            direction_z > 0.0F ? infinity : -infinity);
                                    }
                                    if (slab_layer_count > 0 &&
                                        absolute_direction_z >= 1.0e-6F) {
                                        const auto layer = slab_layer_index(
                                            position_z_mm, slab_z_ends_device,
                                            slab_layer_count);
                                        const auto interface_z =
                                            direction_z > 0.0F
                                                ? slab_z_ends_device[layer]
                                                : (layer == 0U
                                                       ? 0.0F
                                                       : slab_z_ends_device[layer - 1U]);
                                        position_z_mm = sycl::nextafter(
                                            interface_z,
                                            direction_z > 0.0F ? infinity : -infinity);
                                    }
                                    ++steps;
                                    continue;
                                }
                                const auto step_deposited_MeV = sycl::fmin(
                                    stopping_power_MeV_per_mm * path_step_mm, energy_MeV);
                                if (step_deposited_MeV <= 0.0F) {
                                    energy_MeV = 0.0F;
                                    break;
                                }
                                score_secondary_dose_device(
                                    step_deposited_MeV, is_neutral_lineage,
                                    species_index, neutral_origin, bin,
                                    number_of_bins, enable_voxel_scoring,
                                    enable_charged_origin_voxel_scoring, voxel_index,
                                    charged_origin_voxel_offset,
                                    neutral_origin_voxel_offset,
                                    fragment_dose_device, voxel_dose_device,
                                    charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                deposited_MeV += step_deposited_MeV;
                                energy_MeV -= step_deposited_MeV;
                                position_x_mm += direction_x * path_step_mm;
                                position_y_mm += direction_y * path_step_mm;
                                position_z_mm += direction_z * path_step_mm;
                                ++steps;
                            }
                            if (energy_MeV > 0.0F && position_z_mm >= 0.0F &&
                                position_z_mm < phantom_length_mm) {
                                auto bin =
                                    direction_z < 0.0F
                                        ? static_cast<int>(sycl::ceil(position_z_mm /
                                                                      depth_bin_width_mm)) -
                                              1
                                        : static_cast<int>(sycl::floor(position_z_mm /
                                                                       depth_bin_width_mm));
                                bin = sycl::max(
                                    0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                                auto voxel_x = static_cast<int>(voxel_bins_x / 2);
                                auto voxel_y = static_cast<int>(voxel_bins_y / 2);
                                if (enable_voxel_scoring) {
                                    const auto x_coordinate =
                                        (position_x_mm - voxel_min_x_mm) / voxel_size_x_mm;
                                    const auto y_coordinate =
                                        (position_y_mm - voxel_min_y_mm) / voxel_size_y_mm;
                                    voxel_x =
                                        direction_x < 0.0F
                                            ? static_cast<int>(sycl::ceil(x_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(x_coordinate));
                                    voxel_y =
                                        direction_y < 0.0F
                                            ? static_cast<int>(sycl::ceil(y_coordinate)) - 1
                                            : static_cast<int>(sycl::floor(y_coordinate));
                                    voxel_x = sycl::max(
                                        0,
                                        sycl::min(voxel_x, static_cast<int>(voxel_bins_x) - 1));
                                    voxel_y = sycl::max(
                                        0,
                                        sycl::min(voxel_y, static_cast<int>(voxel_bins_y) - 1));
                                }
                                const auto voxel_index =
                                    static_cast<std::size_t>(bin) * voxel_plane_size +
                                    static_cast<std::size_t>(voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(voxel_x);
                                score_secondary_dose_device(
                                    energy_MeV, is_neutral_lineage, species_index,
                                    neutral_origin, bin, number_of_bins, enable_voxel_scoring,
                                    enable_charged_origin_voxel_scoring, voxel_index,
                                    charged_origin_voxel_offset, neutral_origin_voxel_offset,
                                    fragment_dose_device, voxel_dose_device,
                                    charged_origin_voxel_dose_device,
                                    neutral_origin_dose_device,
                                    neutral_origin_voxel_dose_device);
                                deposited_MeV += energy_MeV;
                                energy_MeV = 0.0F;
                            }
                            escaped_MeV = energy_MeV;
                        }
                        secondary_deposited_device[particle_index] = deposited_MeV;
                        secondary_escaped_device[particle_index] = escaped_MeV;
                        secondary_steps_device[particle_index] = steps;
                    });
                secondary_kernel_event.wait_and_throw();
                charged_after_neutral_kernel_seconds +=
                    event_duration_seconds(secondary_kernel_event);
                std::cout << "  charged-after-neutral: [" << batch_begin << ','
                          << batch_end << ")/" << generation_end << '\n'
                          << std::flush;
                }  // batch loop
                transported_queue_count = generation_end;
            }
        }
    }

    // Device scorers may be FP32; promote to double for TransportResult.
    std::vector<DoseAtomicT> dose_atomic_host(number_of_bins);
    std::vector<DoseAtomicT> voxel_dose_atomic_host;
    std::vector<DoseAtomicT> charged_origin_voxel_dose_atomic_host;
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<float> nuclear_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<SecondaryGenerationSummary> secondary_summaries_host;
    std::vector<DoseAtomicT> fragment_dose_atomic_host;
    std::vector<float> secondary_deposited_host;
    std::vector<float> secondary_escaped_host;
    std::vector<std::uint32_t> secondary_steps_host;
    std::vector<CascadeTransportSummary> cascade_summaries_host;
    std::vector<NeutralTransportSummary> neutral_summaries_host;
    std::vector<DoseAtomicT> neutral_origin_dose_atomic_host;
    std::vector<DoseAtomicT> neutral_origin_voxel_dose_atomic_host;
    queue.copy(dose_device, dose_atomic_host.data(), number_of_bins);
    if (enable_voxel_scoring) {
        voxel_dose_atomic_host.resize(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_atomic_host.data(), number_of_voxels)
            .wait_and_throw();
    }
    if (enable_charged_origin_voxel_scoring) {
        charged_origin_voxel_dose_atomic_host.resize(
            charged_origin_category_count * number_of_voxels);
        queue.copy(charged_origin_voxel_dose_device,
                   charged_origin_voxel_dose_atomic_host.data(),
                   charged_origin_voxel_dose_atomic_host.size())
            .wait_and_throw();
    }
    const auto to_double_vec = [](const std::vector<DoseAtomicT>& src) {
        return std::vector<double>(src.begin(), src.end());
    };
    std::vector<double> dose_host = to_double_vec(dose_atomic_host);
    std::vector<double> voxel_dose_host = to_double_vec(voxel_dose_atomic_host);
    std::vector<double> charged_origin_voxel_dose_host =
        to_double_vec(charged_origin_voxel_dose_atomic_host);
    std::vector<double> fragment_dose_host;
    std::vector<double> neutral_origin_dose_host;
    std::vector<double> neutral_origin_voxel_dose_host;
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
        const auto transported_secondary_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(transported_queue_count, secondary_queue_capacity));
        fragment_dose_atomic_host.resize(fragment_species_count * number_of_bins);
        secondary_deposited_host.resize(transported_secondary_count);
        secondary_escaped_host.resize(transported_secondary_count);
        secondary_steps_host.resize(transported_secondary_count);
        queue.copy(fragment_dose_device, fragment_dose_atomic_host.data(),
                   fragment_dose_atomic_host.size());
        fragment_dose_host = to_double_vec(fragment_dose_atomic_host);
        if (transported_secondary_count > 0) {
            queue.copy(secondary_deposited_device, secondary_deposited_host.data(),
                       transported_secondary_count);
            queue.copy(secondary_escaped_device, secondary_escaped_host.data(),
                       transported_secondary_count);
            queue.copy(secondary_steps_device, secondary_steps_host.data(),
                       transported_secondary_count)
                .wait_and_throw();
        } else {
            queue.wait_and_throw();
        }
        if (enable_fragment_cascade) {
            cascade_summaries_host.resize(transported_secondary_count);
            if (transported_secondary_count > 0) {
                queue.copy(cascade_summaries_device, cascade_summaries_host.data(),
                           transported_secondary_count)
                    .wait_and_throw();
            }
        }
    }
    if (enable_neutral_transport) {
        const auto transported_neutral_summary_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(transported_neutral_count, neutral_queue_capacity));
        neutral_summaries_host.resize(transported_neutral_summary_count);
        neutral_origin_dose_atomic_host.resize(neutral_origin_category_count * number_of_bins);
        if (transported_neutral_summary_count > 0) {
            queue.copy(neutral_summaries_device, neutral_summaries_host.data(),
                       transported_neutral_summary_count);
        }
        queue.copy(neutral_origin_dose_device, neutral_origin_dose_atomic_host.data(),
                   neutral_origin_dose_atomic_host.size())
            .wait_and_throw();
        neutral_origin_dose_host = to_double_vec(neutral_origin_dose_atomic_host);
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_atomic_host.resize(
                neutral_origin_category_count * number_of_voxels);
            queue.copy(neutral_origin_voxel_dose_device,
                       neutral_origin_voxel_dose_atomic_host.data(),
                       neutral_origin_voxel_dose_atomic_host.size())
                .wait_and_throw();
            neutral_origin_voxel_dose_host =
                to_double_vec(neutral_origin_voxel_dose_atomic_host);
        }
    }

#ifdef CARBON_TRANSPORT_PROFILE
    std::array<std::uint64_t, static_cast<std::size_t>(TransportProfileSlot::count)>
        profile_host{};
    auto profile_enabled = false;
    if (profile_counters_device != nullptr) {
        queue
            .copy(profile_counters_device, profile_host.data(), transport_profile_slot_count())
            .wait_and_throw();
        profile_enabled = true;
    }
#else
    const auto profile_enabled = false;
    std::array<std::uint64_t, 1> profile_host{};
#endif

    free_immutable_device(table_device);
    free_immutable_device(cross_section_device);
    free_device(dose_device);
    free_device(voxel_dose_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(deposited_device);
    free_device(escaped_device);
    free_device(nuclear_device);
    free_device(steps_device);
    free_device(profile_counters_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_factor_lut_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);
    free_immutable_device(reaction_bins_device);
    free_immutable_device(reactions_device);
    free_immutable_device(reaction_secondaries_device);
    free_device(secondary_queue_device);
    free_device(secondary_queue_counter_device);
    free_device(secondary_queue_filled_device);
    free_device(secondary_work_counter_device);
    free_device(secondary_summaries_device);
    free_device(fragment_dose_device);
    free_device(secondary_deposited_device);
    free_device(secondary_escaped_device);
    free_device(secondary_steps_device);
    free_immutable_device(cascade_projectiles_device);
    free_immutable_device(cascade_cross_sections_device);
    free_immutable_device(cascade_interactions_device);
    free_immutable_device(cascade_products_device);
    free_device(cascade_summaries_device);
    free_device(cascade_xs_lut_device);
    free_immutable_device(neutral_projectiles_device);
    free_immutable_device(neutral_cross_sections_device);
    free_immutable_device(neutral_interactions_device);
    free_immutable_device(neutral_products_device);
    free_device(neutral_queue_device);
    free_device(neutral_queue_counter_device);
    free_device(neutral_queue_filled_device);
    free_device(neutral_summaries_device);
    free_device(neutral_origin_dose_device);
    free_device(neutral_origin_voxel_dose_device);

    TransportResult result;
    result.backend = "sycl-" + device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
#ifdef CARBON_TRANSPORT_PROFILE
    if (profile_enabled) {
        result.profile.enabled = true;
        result.profile.counters = profile_host;
        result.backend += "+profile";
    }
#else
    (void)profile_enabled;
    (void)profile_host;
#endif
    const auto batch_has_energy_spread = std::any_of(
        config.primary_spot_batch.begin(), config.primary_spot_batch.end(),
        [](const auto& spot) { return spot.floats[1] > 0.0F; });
    if (config.beam_energy_spread > 0.0 || batch_has_energy_spread) {
        result.backend += "+espread";
    }
    if (!config.primary_spot_batch.empty()) {
        result.backend += "+spot-batch";
    }
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_layered_phantom) {
        result.backend += use_material_tables ? "+layered-material" : "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += use_insert_material_tables ? "+hetero-insert-material"
                                                     : "+hetero-insert";
    }
    if (enable_ct_grid) {
        if (use_ct_mass_sp) {
            result.backend += "+ct-grid-mass-sp-lut+ct-dda";
        } else if (use_ct_material_sp) {
            result.backend += "+ct-grid-material+ct-dda";
        } else {
            result.backend += "+ct-grid+ct-dda";
        }
        if (use_ct_material_xs) {
            result.backend += "+ct-material-xs";
        }
    }
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring";
    }
    if (enable_charged_origin_voxel_scoring) {
        result.backend += "+charged-origin-voxel-scoring";
    }
    if (config.enable_primary_attenuation) {
        result.backend += "+attenuation";
    }
    if (enable_secondary_generation) {
        result.backend += "+secondary-generation";
    }
    if (enable_secondary_transport) {
        result.backend += "+secondary-transport-persistent";
    }
    if (enable_fragment_cascade) {
        result.backend += "+fragment-cascade";
        if (cascade_xs_lut_size > 0) {
            result.backend += "+cascade-xs-lut";
        }
    }
    if (enable_neutral_transport) {
        result.backend += neutral_allow_continuation ? "+neutral-transport-full"
                                                     : "+neutral-transport-first-interaction";
    }
    result.primary_c12_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    if (config.primary_spot_batch.empty()) {
        result.initial_energy_MeV =
            config.initial_total_energy_MeV() * static_cast<double>(number_of_histories);
    } else {
        result.initial_energy_MeV = 0.0;
        for (const auto& spot : config.primary_spot_batch) {
            result.initial_energy_MeV +=
                static_cast<double>(spot.floats[0]) *
                static_cast<double>(spot.history_end - spot.history_begin);
        }
    }
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
            result.queued_neutrals += summary.queued_neutral_count;
            result.queued_neutral_energy_MeV += summary.queued_neutral_energy_MeV;
            result.neutral_queue_overflow += summary.neutral_queue_overflow_count;
            result.neutral_queue_overflow_energy_MeV +=
                summary.neutral_queue_overflow_energy_MeV;
        }
        result.generated_direct_secondary_energy_MeV =
            result.queued_secondary_energy_MeV +
            result.secondary_queue_overflow_energy_MeV +
            result.untransported_neutral_energy_MeV +
            result.untransported_unsupported_charged_energy_MeV +
            result.queued_neutral_energy_MeV;
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
        double fragment_integral_MeV = 0.0;
        for (const auto* species : fragment_species) {
            for (std::size_t bin = 0; bin < number_of_bins; ++bin) {
                result.deposited_energy_MeV[bin] += (*species)[bin];
                fragment_integral_MeV += (*species)[bin];
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
        // Secondary-transport deposits are tracked in secondary_deposited_host.
        // Local neutral kerma is scored only into fragment_dose; include the extra.
        const auto neutral_kerma_deposit_MeV =
            std::max(0.0, fragment_integral_MeV - result.secondary_deposited_energy_MeV);
        result.total_deposited_energy_MeV +=
            result.secondary_deposited_energy_MeV + neutral_kerma_deposit_MeV;
        result.escaped_energy_MeV += result.secondary_escaped_energy_MeV;
        result.untracked_nuclear_energy_MeV -= result.queued_secondary_energy_MeV;
        if (enable_fragment_cascade) {
            for (std::size_t index = 0;
                 index < static_cast<std::size_t>(
                             std::min<std::uint64_t>(transported_queue_count,
                                                     cascade_summaries_host.size()));
                 ++index) {
                const auto& summary = cascade_summaries_host[index];
                result.cascade_interactions += summary.interaction_count;
                result.generated_cascade_products += summary.direct_count;
                result.queued_cascade_secondaries += summary.queued_count;
                result.cascade_queue_overflow += summary.overflow_count;
                result.queued_cascade_energy_MeV += summary.queued_energy_MeV;
                result.cascade_nuclear_energy_MeV += summary.incident_energy_MeV;
                result.queued_neutrals += summary.queued_neutral_count;
                result.queued_neutral_energy_MeV += summary.queued_neutral_energy_MeV;
                result.neutral_queue_overflow += summary.neutral_queue_overflow_count;
                result.neutral_queue_overflow_energy_MeV +=
                    summary.neutral_queue_overflow_energy_MeV;
                result.untransported_neutral_energy_MeV += summary.neutral_energy_MeV;
            }
            result.untracked_nuclear_energy_MeV +=
                result.cascade_nuclear_energy_MeV - result.queued_cascade_energy_MeV;
        }
        result.total_steps += result.secondary_transport_steps;
    }
    if (enable_neutral_transport) {
        const auto extract_neutral = [&](std::size_t origin_index) {
            const auto begin = neutral_origin_dose_host.begin() +
                               static_cast<std::ptrdiff_t>(origin_index * number_of_bins);
            return std::vector<double>(begin,
                                       begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.neutron_origin_deposited_energy_MeV = extract_neutral(0);
        result.gamma_origin_deposited_energy_MeV = extract_neutral(1);
        for (std::size_t bin = 0; bin < number_of_bins; ++bin) {
            result.deposited_energy_MeV[bin] +=
                result.neutron_origin_deposited_energy_MeV[bin] +
                result.gamma_origin_deposited_energy_MeV[bin];
        }
        result.neutral_origin_voxel_deposited_energy_MeV =
            std::move(neutral_origin_voxel_dose_host);
        result.transported_neutrals = transported_neutral_count;
        std::uint64_t continuation_count = 0;
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(
                         std::min<std::uint64_t>(transported_neutral_count,
                                                 neutral_summaries_host.size()));
             ++index) {
            const auto& summary = neutral_summaries_host[index];
            result.neutral_interactions += summary.interaction_count;
            result.neutral_deposited_energy_MeV += summary.local_deposit_MeV;
            result.neutral_escaped_energy_MeV += summary.escaped_energy_MeV;
            result.residual_neutral_energy_MeV += summary.residual_energy_MeV;
            result.charged_from_neutral_energy_MeV += summary.queued_charged_energy_MeV;
            result.neutral_queue_overflow += summary.neutral_overflow_count;
            result.neutral_queue_overflow_energy_MeV += summary.neutral_overflow_energy_MeV;
            result.secondary_queue_overflow += summary.charged_overflow_count;
            result.secondary_queue_overflow_energy_MeV += summary.charged_overflow_energy_MeV;
            continuation_count += summary.continuation_count;
            // Charged products are transported in the secondary pass; their deposits are
            // already included in secondary_deposited_energy_MeV.
        }
        result.queued_neutrals += continuation_count;
        // Birth neutral energy leaves untracked once. Residual continuation / nested
        // neutrals (mode D) return to the untracked nuclear residual, not phantom escape.
        result.untracked_nuclear_energy_MeV -= result.queued_neutral_energy_MeV;
        result.untracked_nuclear_energy_MeV += result.residual_neutral_energy_MeV;
        result.untransported_neutral_energy_MeV += result.residual_neutral_energy_MeV;
        result.total_deposited_energy_MeV += result.neutral_deposited_energy_MeV;
        result.escaped_energy_MeV += result.neutral_escaped_energy_MeV;
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    result.neutral_kernel_seconds = neutral_kernel_seconds;
    result.charged_after_neutral_kernel_seconds =
        charged_after_neutral_kernel_seconds;
    return result;
}

}  // namespace carbon

#endif
