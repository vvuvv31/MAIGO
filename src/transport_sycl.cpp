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

void score_secondary_dose_device(
    const float amount_MeV,
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
    double* fragment_dose_device,
    double* voxel_dose_device,
    double* charged_origin_voxel_dose_device,
    double* neutral_origin_dose_device,
    double* neutral_origin_voxel_dose_device) noexcept {
    if (amount_MeV <= 0.0F) {
        return;
    }
    if (is_neutral_lineage) {
        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_origin(neutral_origin_dose_device[neutral_origin * number_of_bins +
                                                     static_cast<std::size_t>(bin)]);
        atomic_origin.fetch_add(static_cast<double>(amount_MeV));
        if (enable_voxel_scoring && neutral_origin_voxel_dose_device != nullptr) {
            sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_voxel(
                    neutral_origin_voxel_dose_device[neutral_origin_voxel_offset +
                                                     voxel_index]);
            atomic_voxel.fetch_add(static_cast<double>(amount_MeV));
        }
        return;
    }
    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic_fragment(
            fragment_dose_device[species_index * number_of_bins +
                                 static_cast<std::size_t>(bin)]);
    atomic_fragment.fetch_add(static_cast<double>(amount_MeV));
    if (enable_voxel_scoring) {
        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_voxel(voxel_dose_device[voxel_index]);
        atomic_voxel.fetch_add(static_cast<double>(amount_MeV));
        if (enable_charged_origin_voxel_scoring) {
            sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_category(
                    charged_origin_voxel_dose_device[charged_origin_voxel_offset +
                                                     voxel_index]);
            atomic_category.fetch_add(static_cast<double>(amount_MeV));
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
                               const NeutralPackageTable* neutral_packages) {
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

    auto queue = make_sycl_queue(device_name);
    const auto start = std::chrono::steady_clock::now();
    const auto table_size = stopping_power.values().size();
    const auto cross_section_table_size = cross_section.values().size();
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_histories = config.number_of_histories;
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
    const auto voxel_min_x_mm = -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    const auto voxel_max_x_mm = -voxel_min_x_mm;
    const auto voxel_min_y_mm = -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    const auto voxel_max_y_mm = -voxel_min_y_mm;
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
    const auto secondary_queue_capacity =
        config.secondary_queue_capacity == 0 ? automatic_queue_capacity
                                             : config.secondary_queue_capacity;
    const auto automatic_neutral_queue_capacity =
        number_of_histories > std::numeric_limits<std::size_t>::max() / 32
            ? std::numeric_limits<std::size_t>::max()
            : number_of_histories * 32;
    const auto neutral_queue_capacity =
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
    if (!device.has(sycl::aspect::fp64) || !device.has(sycl::aspect::atomic64)) {
        throw std::runtime_error(
            "The current accurate SYCL scorer requires fp64 and atomic64 device aspects");
    }

    auto* table_device = sycl::malloc_device<float>(table_size, queue);
    auto* cross_section_device =
        sycl::malloc_device<float>(cross_section_table_size, queue);
    auto* dose_device = sycl::malloc_device<double>(number_of_bins, queue);
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? sycl::malloc_device<double>(number_of_voxels, queue)
                                  : nullptr;
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? sycl::malloc_device<double>(
                  charged_origin_category_count * number_of_voxels, queue)
            : nullptr;
    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* nuclear_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);
    ReactionEnergyBin* reaction_bins_device = nullptr;
    ReactionPackage* reactions_device = nullptr;
    ReactionSecondary* reaction_secondaries_device = nullptr;
    SecondaryParticle3D* secondary_queue_device = nullptr;
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
    NeutralProjectile* neutral_projectiles_device = nullptr;
    NeutralCrossSectionSample* neutral_cross_sections_device = nullptr;
    NeutralInteraction* neutral_interactions_device = nullptr;
    ReactionSecondary* neutral_products_device = nullptr;
    NeutralParticle3D* neutral_queue_device = nullptr;
    std::uint64_t* neutral_queue_counter_device = nullptr;
    std::uint64_t* neutral_queue_filled_device = nullptr;
    NeutralTransportSummary* neutral_summaries_device = nullptr;
    double* neutral_origin_dose_device = nullptr;
    double* neutral_origin_voxel_dose_device = nullptr;
    if (enable_secondary_generation) {
        reaction_bins_device = sycl::malloc_device<ReactionEnergyBin>(
            reaction_packages->energy_bins().size(), queue);
        reactions_device = sycl::malloc_device<ReactionPackage>(
            reaction_packages->reactions().size(), queue);
        reaction_secondaries_device = sycl::malloc_device<ReactionSecondary>(
            reaction_packages->secondaries().size(), queue);
        secondary_queue_device =
            sycl::malloc_device<SecondaryParticle3D>(secondary_queue_capacity, queue);
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
    if (enable_neutral_transport) {
        neutral_projectiles_device = sycl::malloc_device<NeutralProjectile>(
            neutral_packages->projectiles().size(), queue);
        neutral_cross_sections_device = sycl::malloc_device<NeutralCrossSectionSample>(
            neutral_packages->cross_sections().size(), queue);
        neutral_interactions_device = sycl::malloc_device<NeutralInteraction>(
            neutral_packages->interactions().size(), queue);
        neutral_products_device = sycl::malloc_device<ReactionSecondary>(
            neutral_packages->products().size(), queue);
        neutral_queue_device =
            sycl::malloc_device<NeutralParticle3D>(neutral_queue_capacity, queue);
        neutral_queue_counter_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_queue_filled_device = sycl::malloc_device<std::uint64_t>(1, queue);
        neutral_summaries_device =
            sycl::malloc_device<NeutralTransportSummary>(neutral_queue_capacity, queue);
        neutral_origin_dose_device = sycl::malloc_device<double>(
            neutral_origin_category_count * number_of_bins, queue);
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_device = sycl::malloc_device<double>(
                neutral_origin_category_count * number_of_voxels, queue);
        }
    }
    const auto free_device = [&queue](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
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
            free_device(table_device);
            free_device(cross_section_device);
            free_device(dose_device);
            free_device(voxel_dose_device);
            free_device(charged_origin_voxel_dose_device);
            free_device(deposited_device);
            free_device(escaped_device);
            free_device(nuclear_device);
            free_device(steps_device);
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
    float* ct_mass_sp_za_device = nullptr;
    float* ct_mass_sp_I_device = nullptr;
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
    // Absolute 4-class tables remain only if requested and no mass-SP factors.
    auto use_ct_mass_sp = false;
    auto use_ct_material_tables = false;
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
        use_ct_mass_sp = ct_grid_host.has_mass_sp_factors();
        use_ct_material_tables =
            !use_ct_mass_sp &&
            (!config.ct_air_stopping_power_file.empty() ||
             !config.ct_lung_stopping_power_file.empty() ||
             !config.ct_water_stopping_power_file.empty() ||
             !config.ct_bone_stopping_power_file.empty());
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
                I_host.assign(za_host.size(), 75.0F);
            }
            ct_mass_sp_za_device =
                sycl::malloc_device<float>(ct_n_mass_factors, queue);
            ct_mass_sp_I_device =
                sycl::malloc_device<float>(ct_n_mass_factors, queue);
            if (ct_mass_sp_za_device == nullptr || ct_mass_sp_I_device == nullptr) {
                free_device(ct_mass_sp_za_device);
                free_device(ct_mass_sp_I_device);
                free_device(ct_density_device);
                free_device(ct_material_device);
                throw std::bad_alloc();
            }
            queue
                .memcpy(ct_mass_sp_za_device, za_host.data(),
                        sizeof(float) * ct_n_mass_factors)
                .wait_and_throw();
            queue
                .memcpy(ct_mass_sp_I_device, I_host.data(),
                        sizeof(float) * ct_n_mass_factors)
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
        if (use_ct_material_tables) {
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
                const auto sp_table = StoppingPowerTable::from_csv(sp_paths[mat]);
                const auto xs_table = CrossSectionTable::from_csv(xs_paths[mat]);
                if (sp_table.values().size() != table_size ||
                    xs_table.values().size() != cross_section_table_size) {
                    throw std::invalid_argument("CT material tables must match water grid size");
                }
                for (std::size_t i = 0; i < table_size; ++i) {
                    ct_sp_host[mat * table_size + i] =
                        static_cast<float>(sp_table.values()[i]);
                }
                for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                    ct_xs_host[mat * cross_section_table_size + i] =
                        static_cast<float>(xs_table.values()[i]);
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
            free_device(ct_mass_sp_za_device);
            free_device(ct_mass_sp_I_device);
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
           (fragment_dose_device == nullptr || secondary_deposited_device == nullptr ||
           secondary_escaped_device == nullptr || secondary_steps_device == nullptr)) ||
         (enable_fragment_cascade &&
          (cascade_projectiles_device == nullptr || cascade_cross_sections_device == nullptr ||
           cascade_interactions_device == nullptr || cascade_products_device == nullptr ||
           cascade_summaries_device == nullptr)));
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
        secondary_allocation_failed || neutral_allocation_failed) {
        free_device(table_device);
        free_device(cross_section_device);
        free_device(dose_device);
        free_device(voxel_dose_device);
        free_device(charged_origin_voxel_dose_device);
        free_device(deposited_device);
        free_device(escaped_device);
        free_device(nuclear_device);
        free_device(steps_device);
        free_device(slab_z_ends_device);
        free_device(slab_densities_device);
        free_device(material_sp_device);
        free_device(material_xs_device);
        free_device(insert_sp_device);
        free_device(insert_xs_device);
        free_device(ct_density_device);
        free_device(ct_material_device);
        free_device(ct_mass_sp_za_device);
        free_device(ct_mass_sp_I_device);
        free_device(ct_sp_device);
        free_device(ct_xs_device);
        free_device(ct_ref_density_device);
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
        free_device(neutral_projectiles_device);
        free_device(neutral_cross_sections_device);
        free_device(neutral_interactions_device);
        free_device(neutral_products_device);
        free_device(neutral_queue_device);
        free_device(neutral_queue_counter_device);
        free_device(neutral_queue_filled_device);
        free_device(neutral_summaries_device);
        free_device(neutral_origin_dose_device);
        free_device(neutral_origin_voxel_dose_device);
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
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(double));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels * sizeof(double));
    }
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
    if (enable_neutral_transport) {
        queue.copy(neutral_packages->projectiles().data(), neutral_projectiles_device,
                   neutral_packages->projectiles().size());
        queue.copy(neutral_packages->cross_sections().data(), neutral_cross_sections_device,
                   neutral_packages->cross_sections().size());
        queue.copy(neutral_packages->interactions().data(), neutral_interactions_device,
                   neutral_packages->interactions().size());
        queue.copy(neutral_packages->products().data(), neutral_products_device,
                   neutral_packages->products().size());
        queue.memset(neutral_queue_counter_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_queue_filled_device, 0, sizeof(std::uint64_t));
        queue.memset(neutral_summaries_device, 0,
                     neutral_queue_capacity * sizeof(NeutralTransportSummary));
        queue.memset(neutral_origin_dose_device, 0,
                     neutral_origin_category_count * number_of_bins * sizeof(double));
        if (enable_voxel_scoring) {
            queue.memset(neutral_origin_voxel_dose_device, 0,
                         neutral_origin_category_count * number_of_voxels * sizeof(double));
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
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto random_seed = config.random_seed;
    const auto enable_primary_attenuation = config.enable_primary_attenuation;
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime =
        static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime =
        static_cast<float>(config.emittance_sigma_y_prime);
    const auto emittance_correlation_x =
        static_cast<float>(config.emittance_correlation_x);
    const auto emittance_correlation_y =
        static_cast<float>(config.emittance_correlation_y);
    const auto neutral_local_kerma_fraction =
        static_cast<float>(config.neutral_local_kerma_fraction);
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

    auto kernel_event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{global_size}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) {
            const auto history = item.get_global_linear_id();
            if (history >= number_of_histories) {
                return;
            }

            auto energy_MeV = initial_energy_MeV;
            auto position_x_mm = 0.0F;
            auto position_y_mm = 0.0F;
            auto position_z_mm = 0.0F;
            auto direction_x = 0.0F;
            auto direction_y = 0.0F;
            auto direction_z = 1.0F;
            if (enable_emittance_source) {
                // TOPAS BiGaussian: sample (x,x') and (y,y') from bivariate normals.
                // x' = dx/dz (unitless, rad-like). Independent axes with correlations.
                const auto u0 = sycl::fmax(
                    rng::uniform01(random_seed, history, 0, 30), 1.0e-12F);
                const auto u1 = rng::uniform01(random_seed, history, 0, 31);
                const auto u2 = sycl::fmax(
                    rng::uniform01(random_seed, history, 0, 32), 1.0e-12F);
                const auto u3 = rng::uniform01(random_seed, history, 0, 33);
                constexpr float two_pi = 6.2831853071795864769F;
                const auto g0 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                const auto g1 = sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::sin(two_pi * u1);
                const auto g2 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::cos(two_pi * u3);
                const auto g3 = sycl::sqrt(-2.0F * sycl::log(u2)) * sycl::sin(two_pi * u3);
                position_x_mm = emittance_sigma_x_mm * g0;
                position_y_mm = emittance_sigma_y_mm * g2;
                const auto rho_x = sycl::clamp(emittance_correlation_x, -0.9999F, 0.9999F);
                const auto rho_y = sycl::clamp(emittance_correlation_y, -0.9999F, 0.9999F);
                const auto x_prime =
                    emittance_sigma_x_prime *
                    (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                const auto y_prime =
                    emittance_sigma_y_prime *
                    (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                // Paraxial unit direction from slopes (dx/dz, dy/dz).
                const auto inv_norm =
                    sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                direction_x = x_prime * inv_norm;
                direction_y = y_prime * inv_norm;
                direction_z = inv_norm;
            }
            auto history_deposited_MeV = 0.0f;
            auto history_nuclear_MeV = 0.0f;
            SecondaryGenerationSummary secondary_summary{};
            std::uint32_t steps = 0;
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
                    if (use_ct_mass_sp && in_ct && ct_mass_sp_za_device != nullptr &&
                        ct_mass_sp_I_device != nullptr && ct_n_mass_factors > 0) {
                        // Schneider section energy-dep mass-SP × density (v3; v2≡I=75).
                        const auto sec = static_cast<std::uint32_t>(ct_material);
                        const auto fi =
                            sec < ct_n_mass_factors ? sec : (ct_n_mass_factors - 1U);
                        const auto mass_factor = ct_mass_sp_energy_factor_impl(
                            ct_mass_sp_za_device[fi], ct_mass_sp_I_device[fi],
                            energy_MeVu,
                            [](float x) { return sycl::log(x); });
                        stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                            water_sp, local_density_g_per_cm3, mass_factor);
                    } else if (use_ct_material_tables && in_ct && ct_sp_device != nullptr &&
                               ct_ref_density_device != nullptr) {
                        // Legacy absolute material SP × (local ρ / ρ_ref).
                        const auto mat =
                            static_cast<std::uint32_t>(sycl::min(
                                static_cast<int>(ct_material), 3));
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
                    step_mm = clamp_step_to_ct_faces_if_needed(
                        step_mm, position_x_mm, position_y_mm, position_z_mm,
                        direction_x, direction_y, direction_z, ct_origin_x,
                        ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                        ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                        ct_material_device, local_density_g_per_cm3, ct_material,
                        ct_skip_homogeneous_face_clamp);
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
                if (step_mm <= 1.0e-6F) {
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
                        continue;
                    }
                    break;
                }

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
                        effective_charge * water_Z_over_A * local_density_g_per_cm3 *
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
                if (enable_voxel_scoring) {
                    sycl::atomic_ref<double,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_voxel_dose(voxel_dose_device[voxel_index]);
                    atomic_voxel_dose.fetch_add(static_cast<double>(deposited_MeV));
                    if (enable_charged_origin_voxel_scoring) {
                        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_category_dose(
                                charged_origin_voxel_dose_device[voxel_index]);
                        atomic_category_dose.fetch_add(static_cast<double>(deposited_MeV));
                    }
                }
                history_deposited_MeV += deposited_MeV;
                const auto scattering_energy_MeV =
                    energy_MeV - 0.5F * deposited_MeV;
                energy_MeV -= deposited_MeV;
                position_x_mm += direction_x * step_mm;
                position_y_mm += direction_y * step_mm;
                position_z_mm += direction_z * step_mm;
                if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
                    const auto projected_rms_angle_rad =
                        highland_projected_rms_angle_device(
                            scattering_energy_MeV, 6, primary_mass_number, step_mm,
                            local_density_g_per_cm3);
                    const auto scattered = scatter_direction(
                        Direction3F{direction_x, direction_y, direction_z},
                        projected_rms_angle_rad, random_seed, history, steps, 4);
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
                        if (use_ct_material_tables && in_ct && ct_xs_device != nullptr &&
                            ct_ref_density_device != nullptr) {
                            const auto mat = static_cast<std::uint32_t>(
                                sycl::min(static_cast<int>(ct_material), 3));
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
                            std::uint32_t neutral_queueable_count = 0;
                            auto neutral_queueable_energy_MeV = 0.0F;
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
                                    if (enable_neutral_transport) {
                                        ++neutral_queueable_count;
                                        neutral_queueable_energy_MeV +=
                                            secondary.kinetic_energy_MeV;
                                    } else {
                                        const auto kerma_MeV =
                                            secondary.kinetic_energy_MeV *
                                            neutral_local_kerma_fraction;
                                        const auto residual_MeV =
                                            secondary.kinetic_energy_MeV - kerma_MeV;
                                        secondary_summary.neutral_energy_MeV += residual_MeV;
                                        if (kerma_MeV > 0.0F &&
                                            fragment_dose_device != nullptr) {
                                            // Score interim local kerma into "other" fragment
                                            // channel so total IDD includes it without
                                            // polluting primary C-12.
                                            constexpr std::size_t other_species = 6;
                                            sycl::atomic_ref<
                                                double, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                kerma_dose(
                                                    fragment_dose_device
                                                        [other_species * number_of_bins +
                                                         static_cast<std::size_t>(bin)]);
                                            kerma_dose.fetch_add(
                                                static_cast<double>(kerma_MeV));
                                            if (enable_voxel_scoring &&
                                                voxel_dose_device != nullptr) {
                                                const auto kerma_voxel_index =
                                                    static_cast<std::size_t>(bin) *
                                                        voxel_plane_size +
                                                    static_cast<std::size_t>(voxel_y) *
                                                        voxel_bins_x +
                                                    static_cast<std::size_t>(voxel_x);
                                                sycl::atomic_ref<
                                                    double, sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    kerma_voxel(
                                                        voxel_dose_device[kerma_voxel_index]);
                                                kerma_voxel.fetch_add(
                                                    static_cast<double>(kerma_MeV));
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
                                                        history,
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
                                                        history,
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
                        }
                        energy_MeV = 0.0f;
                    }
                }
                ++steps;
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
                sycl::atomic_ref<double,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dose(dose_device[bin]);
                atomic_dose.fetch_add(static_cast<double>(energy_MeV));
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
                    sycl::atomic_ref<double,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_voxel_dose(voxel_dose_device[voxel_index]);
                    atomic_voxel_dose.fetch_add(static_cast<double>(energy_MeV));
                    if (enable_charged_origin_voxel_scoring) {
                        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_category_dose(
                                charged_origin_voxel_dose_device[voxel_index]);
                        atomic_category_dose.fetch_add(static_cast<double>(energy_MeV));
                    }
                }
                history_deposited_MeV += energy_MeV;
                energy_MeV = 0.0F;
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

                    while (energy_MeV > energy_cutoff_MeV) {
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
                            if (use_ct_mass_sp && in_ct &&
                                ct_mass_sp_za_device != nullptr &&
                                ct_mass_sp_I_device != nullptr &&
                                ct_n_mass_factors > 0) {
                                const auto sec = static_cast<std::uint32_t>(ct_material);
                                const auto fi = sec < ct_n_mass_factors
                                                    ? sec
                                                    : (ct_n_mass_factors - 1U);
                                const auto mass_factor = ct_mass_sp_energy_factor_impl(
                                    ct_mass_sp_za_device[fi], ct_mass_sp_I_device[fi],
                                    energy_MeVu,
                                    [](float x) { return sycl::log(x); });
                                carbon_sp_local = ct_mass_scaled_stopping_power(
                                    carbon_stopping_power_MeV_per_mm,
                                    local_density_g_per_cm3, mass_factor);
                            } else if (use_ct_material_tables && in_ct &&
                                       ct_sp_device != nullptr &&
                                       ct_ref_density_device != nullptr) {
                                const auto mat = static_cast<std::uint32_t>(
                                    sycl::min(static_cast<int>(ct_material), 3));
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
                            path_step_mm = clamp_step_to_ct_faces_if_needed(
                                path_step_mm, position_x_mm, position_y_mm, position_z_mm,
                                direction_x, direction_y, direction_z, ct_origin_x,
                                ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                                ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                                ct_material_device, local_density_g_per_cm3, ct_material,
                                ct_skip_homogeneous_face_clamp);
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
                        if (path_step_mm <= 1.0e-6F) {
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
                        score_secondary_dose_device(
                            step_deposited_MeV, is_neutral_lineage, species_index,
                            neutral_origin, bin, number_of_bins, enable_voxel_scoring,
                            enable_charged_origin_voxel_scoring, voxel_index,
                            charged_origin_voxel_offset, neutral_origin_voxel_offset,
                            fragment_dose_device, voxel_dose_device,
                            charged_origin_voxel_dose_device, neutral_origin_dose_device,
                            neutral_origin_voxel_dose_device);
                        deposited_MeV += step_deposited_MeV;
                        const auto scattering_energy_MeV =
                            energy_MeV - 0.5F * step_deposited_MeV;
                        energy_MeV -= step_deposited_MeV;
                        position_x_mm += direction_x * path_step_mm;
                        position_y_mm += direction_y * path_step_mm;
                        position_z_mm += direction_z * path_step_mm;
                        if (enable_multiple_scattering && energy_MeV > energy_cutoff_MeV) {
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
                                if (slab_layer_count > 0) {
                                    macroscopic_cross_section_per_mm *=
                                        local_density_g_per_cm3;
                                }
                                const auto interaction_probability =
                                    1.0F - sycl::exp(-macroscopic_cross_section_per_mm *
                                                     path_step_mm);
                                if (rng::uniform01(random_seed, rng_stream, steps, 8) <
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
                                                const auto kerma_MeV =
                                                    scaled_energy * neutral_local_kerma_fraction;
                                                const auto residual_MeV =
                                                    scaled_energy - kerma_MeV;
                                                cascade_summary.neutral_energy_MeV +=
                                                    residual_MeV;
                                                if (kerma_MeV > 0.0F &&
                                                    fragment_dose_device != nullptr) {
                                                    constexpr std::size_t other_species = 6;
                                                    sycl::atomic_ref<
                                                        double, sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        kerma_dose(
                                                            fragment_dose_device
                                                                [other_species * number_of_bins +
                                                                 static_cast<std::size_t>(bin)]);
                                                    kerma_dose.fetch_add(
                                                        static_cast<double>(kerma_MeV));
                                                    if (enable_voxel_scoring &&
                                                        voxel_dose_device != nullptr) {
                                                        const auto kerma_voxel_index =
                                                            static_cast<std::size_t>(bin) *
                                                                voxel_plane_size +
                                                            static_cast<std::size_t>(voxel_y) *
                                                                voxel_bins_x +
                                                            static_cast<std::size_t>(voxel_x);
                                                        sycl::atomic_ref<
                                                            double, sycl::memory_order::relaxed,
                                                            sycl::memory_scope::device,
                                                            sycl::access::address_space::global_space>
                                                            kerma_voxel(
                                                                voxel_dose_device[kerma_voxel_index]);
                                                        kerma_voxel.fetch_add(
                                                            static_cast<double>(kerma_MeV));
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
                        }
                        ++steps;
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

                        while (energy_MeV > energy_cutoff_MeV) {
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
                            std::uint32_t upper = 0;
                            while (upper < projectile.cross_section_count &&
                                   neutral_cross_sections_device
                                           [projectile.cross_section_offset + upper]
                                               .energy_MeV < energy_MeV) {
                                ++upper;
                            }
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

                            std::uint32_t nearest = 0;
                            while (nearest + 1 < projectile.interaction_count &&
                                   neutral_interactions_device
                                           [projectile.interaction_offset + nearest + 1]
                                               .incident_energy_MeV < energy_MeV) {
                                ++nearest;
                            }
                            if (nearest + 1 < projectile.interaction_count) {
                                const auto lower_delta = sycl::fabs(
                                    neutral_interactions_device
                                        [projectile.interaction_offset + nearest]
                                            .incident_energy_MeV -
                                    energy_MeV);
                                const auto upper_delta = sycl::fabs(
                                    neutral_interactions_device
                                        [projectile.interaction_offset + nearest + 1]
                                            .incident_energy_MeV -
                                    energy_MeV);
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

                            while (energy_MeV > energy_cutoff_MeV) {
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
                                    charge * (1.0F - sycl::exp(-125.0F * beta *
                                                               sycl::pow(charge, -2.0F / 3.0F)));
                                constexpr float carbon_charge = 6.0F;
                                const auto carbon_effective_charge =
                                    carbon_charge *
                                    (1.0F - sycl::exp(-125.0F * beta *
                                                      sycl::pow(carbon_charge, -2.0F / 3.0F)));
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
                                    step_deposited_MeV, is_neutral_lineage, species_index,
                                    neutral_origin, bin, number_of_bins, enable_voxel_scoring,
                                    enable_charged_origin_voxel_scoring, voxel_index,
                                    charged_origin_voxel_offset, neutral_origin_voxel_offset,
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
                transported_queue_count = generation_end;
            }
        }
    }

    std::vector<double> dose_host(number_of_bins);
    std::vector<double> voxel_dose_host;
    std::vector<double> charged_origin_voxel_dose_host;
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
    std::vector<NeutralTransportSummary> neutral_summaries_host;
    std::vector<double> neutral_origin_dose_host;
    std::vector<double> neutral_origin_voxel_dose_host;
    queue.copy(dose_device, dose_host.data(), number_of_bins);
    if (enable_voxel_scoring) {
        voxel_dose_host.resize(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_host.data(), number_of_voxels)
            .wait_and_throw();
    }
    if (enable_charged_origin_voxel_scoring) {
        charged_origin_voxel_dose_host.resize(
            charged_origin_category_count * number_of_voxels);
        queue.copy(charged_origin_voxel_dose_device,
                   charged_origin_voxel_dose_host.data(),
                   charged_origin_voxel_dose_host.size())
            .wait_and_throw();
    }
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
    if (enable_neutral_transport) {
        neutral_summaries_host.resize(neutral_queue_capacity);
        neutral_origin_dose_host.resize(neutral_origin_category_count * number_of_bins);
        queue.copy(neutral_summaries_device, neutral_summaries_host.data(),
                   neutral_queue_capacity);
        queue.copy(neutral_origin_dose_device, neutral_origin_dose_host.data(),
                   neutral_origin_dose_host.size())
            .wait_and_throw();
        if (enable_voxel_scoring) {
            neutral_origin_voxel_dose_host.resize(
                neutral_origin_category_count * number_of_voxels);
            queue.copy(neutral_origin_voxel_dose_device,
                       neutral_origin_voxel_dose_host.data(),
                       neutral_origin_voxel_dose_host.size())
                .wait_and_throw();
        }
    }

    free_device(table_device);
    free_device(cross_section_device);
    free_device(dose_device);
    free_device(voxel_dose_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(deposited_device);
    free_device(escaped_device);
    free_device(nuclear_device);
    free_device(steps_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_za_device);
    free_device(ct_mass_sp_I_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);
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
    free_device(neutral_projectiles_device);
    free_device(neutral_cross_sections_device);
    free_device(neutral_interactions_device);
    free_device(neutral_products_device);
    free_device(neutral_queue_device);
    free_device(neutral_queue_counter_device);
    free_device(neutral_queue_filled_device);
    free_device(neutral_summaries_device);
    free_device(neutral_origin_dose_device);
    free_device(neutral_origin_voxel_dose_device);

    TransportResult result;
    result.backend = "sycl-" + device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
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
            result.backend += "+ct-grid-mass-sp-e";
        } else if (use_ct_material_tables) {
            result.backend += "+ct-grid-material";
        } else {
            result.backend += "+ct-grid";
        }
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
        result.backend += "+secondary-transport";
    }
    if (enable_fragment_cascade) {
        result.backend += "+fragment-cascade";
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
    return result;
}

}  // namespace carbon

#endif
