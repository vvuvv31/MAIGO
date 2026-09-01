#include "carbon/cross_section.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include "carbon/fred_event_library.hpp"
#include "carbon/fred_table1.hpp"
#include "carbon/inelastic.hpp"
#include "carbon/inelastic_package_v2.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_profile.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"
#include "detail/sycl_cinel02_device.inc"

namespace {

#include "detail/sycl_device_math.inc"
#include "detail/sycl_inelastic_device.inc"
#include "detail/sycl_score_device.inc"

float cuda_clock_warmup(sycl::queue& queue) {
    auto* dummy = sycl::malloc_device<float>(1024, queue);
    if (dummy == nullptr) {
        return 0.0F;
    }
    queue.memset(dummy, 0, 1024 * sizeof(float)).wait_and_throw();
    const auto start = std::chrono::steady_clock::now();
    auto event = queue.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{1024}, sycl::range<1>{128}},
        [=](sycl::nd_item<1> item) {
            const auto lane = item.get_global_linear_id();
            if (lane < 1024) {
                dummy[lane] = static_cast<float>(lane) * 1.001F;
            }
        });
    event.wait_and_throw();
    sycl::free(dummy, queue);
    return static_cast<float>(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace

TransportResult transport_sycl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    config.validate();
    const bool use_cinel02 = config.nuclear_model == "cinel02";
    std::optional<InelasticPackageV2Table> cinel02_package;
    std::optional<InelasticRateV2Table> cinel02_rates;
    std::optional<Cinel02DeviceTables> cinel02_host_tables;
    if (use_cinel02) {
        cinel02_package.emplace(InelasticPackageV2Table::from_binary(
            config.primary_inelastic_package_v2_file));
        cinel02_rates.emplace(InelasticRateV2Table::from_csv(
            config.primary_inelastic_rate_v2_file));
        cinel02_host_tables.emplace(cinel02_package->make_device_tables());
    }
    const auto start = std::chrono::steady_clock::now();
    const auto& table_energies = stopping_power.energies();
    if (table_energies.size() < 2 || !is_uniform_grid(table_energies)) {
        throw std::invalid_argument("Stopping power table must have a uniform energy grid");
    }

    const auto resolved_device_name =
        device_name.empty() ? std::string("gpu") : device_name;

    sycl::queue local_queue = (context != nullptr)
                                  ? context->impl_->queue
                                  : make_sycl_queue(resolved_device_name);
    auto& queue = local_queue;
    const auto device = queue.get_device();
    const auto backend = queue.get_backend();
    const bool is_cuda_backend = backend == sycl::backend::ext_oneapi_cuda;

    const auto free_device = [&](auto* pointer) {
        if (pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };

    const auto number_of_histories = config.number_of_histories;
    const auto number_of_bins = config.number_of_bins();
    const auto number_of_voxels = config.number_of_voxels();
    const auto table_size = stopping_power.values().size();
    // Device-side inelastic tables share the stopping-power transport grid.
    const auto cross_section_table_size = table_size;

    const bool reuse_immutable_buffers = context != nullptr;
    if (context != nullptr) {
        context->impl_->ensure_initialized(stopping_power, cross_section);
    }

    float* table_device = context != nullptr ? context->impl_->table_device
                                            : sycl::malloc_device<float>(table_size, queue);
    float* energy_grid_device =
        context != nullptr ? context->impl_->energy_grid_device
                           : (config.enable_csda_range_energy_loss
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr);
    float* cumulative_range_device =
        context != nullptr ? context->impl_->cumulative_range_device
                           : (config.enable_csda_range_energy_loss
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr);
    float* cross_section_device =
        context != nullptr ? context->impl_->cross_section_device
                           : sycl::malloc_device<float>(cross_section_table_size, queue);
    float* target_h_fraction_device =
        sycl::malloc_device<float>(cross_section_table_size, queue);

    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers && pointer != nullptr) {
            sycl::free(pointer, queue);
        }
    };

    Cinel02DeviceInteraction* cinel02_interactions_device = nullptr;
    Cinel02DeviceProduct* cinel02_products_device = nullptr;
    Cinel02EnergyNode* cinel02_energy_nodes_device = nullptr;
    std::uint32_t* cinel02_event_offsets_device = nullptr;
    std::uint32_t* cinel02_event_indices_device = nullptr;
    Cinel02RateGroup* cinel02_rate_groups_device = nullptr;
    Cinel02RateSample* cinel02_rate_samples_device = nullptr;
    std::uint32_t cinel02_interaction_count = 0U;
    std::uint32_t cinel02_product_count = 0U;
    std::uint32_t cinel02_energy_node_count = 0U;
    std::uint32_t cinel02_rate_group_count = 0U;
    std::uint32_t cinel02_rate_sample_count = 0U;
    constexpr std::size_t kCinel02DiagSlots =
        TransportResult::cinel02_diagnostic_slot_count;
    std::uint64_t* cinel02_diag_device = nullptr;
    constexpr std::size_t kCinel02EnergySlots = 8;
    constexpr std::size_t kCinel02SpeciesEnergySlots =
        TransportResult::species_ledger_species_count *
        TransportResult::species_ledger_metric_count;
    float* cinel02_energy_device = nullptr;
    float* cinel02_species_energy_device = nullptr;
    constexpr std::size_t kCinel02SpeciesTerminalSlots =
        Cinel02SpeciesLedgerSchema::species_count *
        Cinel02SpeciesLedgerSchema::terminal_reason_count;
    std::uint64_t* cinel02_species_terminal_device = nullptr;
    float* cinel02_replay_delta_device = nullptr;
    float* cinel02_replay_abs_delta_device = nullptr;
    std::uint64_t* cinel02_replay_delta_positive_device = nullptr;
    std::uint64_t* cinel02_replay_delta_negative_device = nullptr;
    std::uint64_t* cinel02_replay_valid_device = nullptr;
    constexpr std::size_t kCinel02ReplayStatusSlots =
        Cinel02ReplayLedgerSchema::status_slot_count;
    constexpr std::size_t kCinel02ParentOutcomeSlots =
        Cinel02ReplayLedgerSchema::parent_outcome_cell_count;
    constexpr std::size_t kCinel02TransitionSlots =
        Cinel02ReplayLedgerSchema::transition_cell_count;
    std::uint64_t* cinel02_replay_status_counts_device = nullptr;
    float* cinel02_replay_status_incident_device = nullptr;
    float* cinel02_replay_status_delta_device = nullptr;
    float* cinel02_replay_status_abs_delta_device = nullptr;
    std::uint64_t* cinel02_parent_outcome_counts_device = nullptr;
    float* cinel02_parent_outcome_incident_device = nullptr;
    float* cinel02_parent_outcome_after_device = nullptr;
    float* cinel02_parent_outcome_local_device = nullptr;
    float* cinel02_parent_outcome_export_device = nullptr;
    float* cinel02_parent_outcome_import_device = nullptr;
    std::uint64_t* cinel02_generated_transition_counts_device = nullptr;
    float* cinel02_generated_transition_energy_device = nullptr;
    std::uint64_t* cinel02_queued_transition_counts_device = nullptr;
    float* cinel02_queued_transition_energy_device = nullptr;
    if (use_cinel02) {
        const auto checked_u32 = [](const std::size_t value, const char* label) {
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(std::string("CINEL02 ") + label +
                                          " exceeds uint32 range");
            }
            return static_cast<std::uint32_t>(value);
        };
        cinel02_interaction_count = checked_u32(
            cinel02_host_tables->interactions.size(), "interaction count");
        cinel02_product_count = checked_u32(
            cinel02_host_tables->products.size(), "product count");
        cinel02_energy_node_count = checked_u32(
            cinel02_host_tables->energy_nodes.size(), "energy-node count");
        cinel02_rate_group_count = checked_u32(
            cinel02_rates->groups().size(), "rate-group count");
        cinel02_rate_sample_count = checked_u32(
            cinel02_rates->samples().size(), "rate-sample count");
        const auto immutable_bytes = cinel02_host_tables->bytes() +
            cinel02_rates->groups().size() * sizeof(Cinel02RateGroup) +
            cinel02_rates->samples().size() * sizeof(Cinel02RateSample);
        const auto global_bytes =
            device.get_info<sycl::info::device::global_mem_size>();
        const auto configured_limit = static_cast<std::uint64_t>(
            static_cast<double>(global_bytes) * config.max_device_memory_fraction);
        if (immutable_bytes > configured_limit) {
            throw std::runtime_error(
                "CINEL02 compact tables exceed max_device_memory_fraction: need " +
                std::to_string(immutable_bytes) + " bytes, limit " +
                std::to_string(configured_limit));
        }
        std::cout << "CINEL02 compact device tables: " << immutable_bytes
                  << " bytes (" << cinel02_interaction_count << " interactions, "
                  << cinel02_product_count << " products)\n";
        cinel02_interactions_device = sycl::malloc_device<Cinel02DeviceInteraction>(
            cinel02_interaction_count, queue);
        cinel02_products_device = sycl::malloc_device<Cinel02DeviceProduct>(
            cinel02_product_count, queue);
        cinel02_energy_nodes_device = sycl::malloc_device<Cinel02EnergyNode>(
            cinel02_energy_node_count, queue);
        cinel02_event_offsets_device = sycl::malloc_device<std::uint32_t>(
            cinel02_host_tables->event_offsets.size(), queue);
        cinel02_event_indices_device = sycl::malloc_device<std::uint32_t>(
            cinel02_host_tables->event_indices.size(), queue);
        cinel02_rate_groups_device = sycl::malloc_device<Cinel02RateGroup>(
            cinel02_rate_group_count, queue);
        cinel02_rate_samples_device = sycl::malloc_device<Cinel02RateSample>(
            cinel02_rate_sample_count, queue);
        cinel02_diag_device =
            sycl::malloc_device<std::uint64_t>(kCinel02DiagSlots, queue);
        cinel02_energy_device =
            sycl::malloc_device<float>(kCinel02EnergySlots, queue);
        cinel02_species_energy_device =
            sycl::malloc_device<float>(kCinel02SpeciesEnergySlots, queue);
        cinel02_species_terminal_device = sycl::malloc_device<std::uint64_t>(
            kCinel02SpeciesTerminalSlots, queue);
        cinel02_replay_delta_device = sycl::malloc_device<float>(
            TransportResult::species_ledger_species_count, queue);
        cinel02_replay_abs_delta_device = sycl::malloc_device<float>(
            TransportResult::species_ledger_species_count, queue);
        cinel02_replay_delta_positive_device = sycl::malloc_device<std::uint64_t>(
            TransportResult::species_ledger_species_count, queue);
        cinel02_replay_delta_negative_device = sycl::malloc_device<std::uint64_t>(
            TransportResult::species_ledger_species_count, queue);
        cinel02_replay_valid_device = sycl::malloc_device<std::uint64_t>(
            TransportResult::species_ledger_species_count, queue);
        cinel02_replay_status_counts_device = sycl::malloc_device<std::uint64_t>(
            kCinel02ReplayStatusSlots, queue);
        cinel02_replay_status_incident_device = sycl::malloc_device<float>(
            kCinel02ReplayStatusSlots, queue);
        cinel02_replay_status_delta_device = sycl::malloc_device<float>(
            kCinel02ReplayStatusSlots, queue);
        cinel02_replay_status_abs_delta_device = sycl::malloc_device<float>(
            kCinel02ReplayStatusSlots, queue);
        cinel02_parent_outcome_counts_device = sycl::malloc_device<std::uint64_t>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_parent_outcome_incident_device = sycl::malloc_device<float>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_parent_outcome_after_device = sycl::malloc_device<float>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_parent_outcome_local_device = sycl::malloc_device<float>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_parent_outcome_export_device = sycl::malloc_device<float>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_parent_outcome_import_device = sycl::malloc_device<float>(
            kCinel02ParentOutcomeSlots, queue);
        cinel02_generated_transition_counts_device = sycl::malloc_device<std::uint64_t>(
            kCinel02TransitionSlots, queue);
        cinel02_generated_transition_energy_device = sycl::malloc_device<float>(
            kCinel02TransitionSlots, queue);
        cinel02_queued_transition_counts_device = sycl::malloc_device<std::uint64_t>(
            kCinel02TransitionSlots, queue);
        cinel02_queued_transition_energy_device = sycl::malloc_device<float>(
            kCinel02TransitionSlots, queue);
        if (cinel02_interactions_device == nullptr || cinel02_products_device == nullptr ||
            cinel02_energy_nodes_device == nullptr || cinel02_event_offsets_device == nullptr ||
            cinel02_event_indices_device == nullptr || cinel02_rate_groups_device == nullptr ||
            cinel02_rate_samples_device == nullptr || cinel02_diag_device == nullptr ||
            cinel02_energy_device == nullptr || cinel02_species_energy_device == nullptr ||
            cinel02_species_terminal_device == nullptr || cinel02_replay_delta_device == nullptr ||
            cinel02_replay_abs_delta_device == nullptr ||
            cinel02_replay_delta_positive_device == nullptr ||
            cinel02_replay_delta_negative_device == nullptr ||
            cinel02_replay_valid_device == nullptr ||
            cinel02_replay_status_counts_device == nullptr ||
            cinel02_replay_status_incident_device == nullptr ||
            cinel02_replay_status_delta_device == nullptr ||
            cinel02_replay_status_abs_delta_device == nullptr ||
            cinel02_parent_outcome_counts_device == nullptr ||
            cinel02_parent_outcome_incident_device == nullptr ||
            cinel02_parent_outcome_after_device == nullptr ||
            cinel02_parent_outcome_local_device == nullptr ||
            cinel02_parent_outcome_export_device == nullptr ||
            cinel02_parent_outcome_import_device == nullptr ||
            cinel02_generated_transition_counts_device == nullptr ||
            cinel02_generated_transition_energy_device == nullptr ||
            cinel02_queued_transition_counts_device == nullptr ||
            cinel02_queued_transition_energy_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(cinel02_host_tables->interactions.data(),
                   cinel02_interactions_device, cinel02_interaction_count);
        queue.copy(cinel02_host_tables->products.data(),
                   cinel02_products_device, cinel02_product_count);
        queue.copy(cinel02_host_tables->energy_nodes.data(),
                   cinel02_energy_nodes_device, cinel02_energy_node_count);
        queue.copy(cinel02_host_tables->event_offsets.data(),
                   cinel02_event_offsets_device,
                   cinel02_host_tables->event_offsets.size());
        queue.copy(cinel02_host_tables->event_indices.data(),
                   cinel02_event_indices_device,
                   cinel02_host_tables->event_indices.size());
        queue.copy(cinel02_rates->groups().data(), cinel02_rate_groups_device,
                   cinel02_rate_group_count);
        queue.copy(cinel02_rates->samples().data(), cinel02_rate_samples_device,
                   cinel02_rate_sample_count);
        queue.fill(cinel02_diag_device, std::uint64_t{0}, kCinel02DiagSlots)
            .wait_and_throw();
        queue.fill(cinel02_energy_device, 0.0F, kCinel02EnergySlots)
            .wait_and_throw();
        queue.fill(cinel02_species_energy_device, 0.0F,
                   kCinel02SpeciesEnergySlots).wait_and_throw();
        queue.fill(cinel02_species_terminal_device, std::uint64_t{0},
                   kCinel02SpeciesTerminalSlots).wait_and_throw();
        queue.fill(cinel02_replay_delta_device, 0.0F,
                   TransportResult::species_ledger_species_count).wait_and_throw();
        queue.fill(cinel02_replay_abs_delta_device, 0.0F,
                   TransportResult::species_ledger_species_count).wait_and_throw();
        queue.fill(cinel02_replay_delta_positive_device, std::uint64_t{0},
                   TransportResult::species_ledger_species_count).wait_and_throw();
        queue.fill(cinel02_replay_delta_negative_device, std::uint64_t{0},
                   TransportResult::species_ledger_species_count).wait_and_throw();
        queue.fill(cinel02_replay_valid_device, std::uint64_t{0},
                   TransportResult::species_ledger_species_count).wait_and_throw();
        queue.fill(cinel02_replay_status_counts_device, std::uint64_t{0},
                   kCinel02ReplayStatusSlots).wait_and_throw();
        queue.fill(cinel02_replay_status_incident_device, 0.0F,
                   kCinel02ReplayStatusSlots).wait_and_throw();
        queue.fill(cinel02_replay_status_delta_device, 0.0F,
                   kCinel02ReplayStatusSlots).wait_and_throw();
        queue.fill(cinel02_replay_status_abs_delta_device, 0.0F,
                   kCinel02ReplayStatusSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_counts_device, std::uint64_t{0},
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_incident_device, 0.0F,
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_after_device, 0.0F,
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_local_device, 0.0F,
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_export_device, 0.0F,
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_parent_outcome_import_device, 0.0F,
                   kCinel02ParentOutcomeSlots).wait_and_throw();
        queue.fill(cinel02_generated_transition_counts_device, std::uint64_t{0},
                   kCinel02TransitionSlots).wait_and_throw();
        queue.fill(cinel02_generated_transition_energy_device, 0.0F,
                   kCinel02TransitionSlots).wait_and_throw();
        queue.fill(cinel02_queued_transition_counts_device, std::uint64_t{0},
                   kCinel02TransitionSlots).wait_and_throw();
        queue.fill(cinel02_queued_transition_energy_device, 0.0F,
                   kCinel02TransitionSlots).wait_and_throw();
        cinel02_host_tables.reset();
        cinel02_package.reset();
        cinel02_rates.reset();
    }

    // Slab layers
    const auto enable_layered_phantom = config.enable_layered_phantom;
    const auto slab_layer_count =
        enable_layered_phantom ? config.slab_layers.size() : std::size_t{0};
    const auto use_material_tables =
        enable_layered_phantom && !config.slab_stopping_power_files.empty();
    const auto material_table_count = use_material_tables ? slab_layer_count : std::size_t{0};

    std::vector<float> slab_z_ends_host(slab_layer_count);
    std::vector<float> slab_densities_host(slab_layer_count);
    std::vector<float> slab_radiation_lengths_host(slab_layer_count);
    for (std::size_t i = 0; i < slab_layer_count; ++i) {
        slab_z_ends_host[i] = static_cast<float>(config.slab_layers[i].z_end_mm);
        slab_densities_host[i] = static_cast<float>(config.slab_layers[i].density_g_per_cm3);
        slab_radiation_lengths_host[i] =
            i < config.slab_radiation_lengths_g_per_cm2.size()
                ? static_cast<float>(config.slab_radiation_lengths_g_per_cm2[i])
                : static_cast<float>(water_radiation_length_g_per_cm2);
    }
    float* slab_z_ends_device = slab_layer_count > 0
                                   ? sycl::malloc_device<float>(slab_layer_count, queue)
                                   : nullptr;
    float* slab_densities_device = slab_layer_count > 0
                                       ? sycl::malloc_device<float>(slab_layer_count, queue)
                                       : nullptr;
    float* slab_radiation_lengths_device =
        slab_layer_count > 0
            ? sycl::malloc_device<float>(slab_layer_count, queue)
            : nullptr;
    if (slab_layer_count > 0) {
        queue.copy(slab_z_ends_host.data(), slab_z_ends_device, slab_layer_count);
        queue.copy(slab_densities_host.data(), slab_densities_device, slab_layer_count);
        queue.copy(slab_radiation_lengths_host.data(), slab_radiation_lengths_device,
                   slab_layer_count).wait_and_throw();
    }

    std::vector<float> material_sp_host(material_table_count * table_size);
    std::vector<float> material_xs_host(material_table_count * cross_section_table_size);
    if (use_material_tables) {
        for (std::size_t layer = 0; layer < slab_layer_count; ++layer) {
            const auto sp = StoppingPowerTable::from_csv(config.slab_stopping_power_files[layer]);
            const auto xs = CrossSectionTable::from_csv(config.slab_cross_section_files[layer]);
            for (std::size_t i = 0; i < table_size; ++i) {
                material_sp_host[layer * table_size + i] = static_cast<float>(sp.values()[i]);
            }
            for (std::size_t i = 0; i < cross_section_table_size; ++i) {
                material_xs_host[layer * cross_section_table_size + i] =
                    static_cast<float>(xs.interpolate(table_energies[i]));
            }
        }
    }
    float* material_sp_device = material_table_count > 0
                                   ? sycl::malloc_device<float>(material_sp_host.size(), queue)
                                   : nullptr;
    float* material_xs_device = material_table_count > 0
                                   ? sycl::malloc_device<float>(material_xs_host.size(), queue)
                                   : nullptr;
    if (material_table_count > 0) {
        queue.copy(material_sp_host.data(), material_sp_device, material_sp_host.size());
        queue.copy(material_xs_host.data(), material_xs_device, material_xs_host.size())
            .wait_and_throw();
    }

    // Hetero insert
    const auto enable_hetero_insert = config.enable_hetero_insert;
    const auto insert_x_min = static_cast<float>(config.hetero_insert.x_min_mm);
    const auto insert_x_max = static_cast<float>(config.hetero_insert.x_max_mm);
    const auto insert_y_min = static_cast<float>(config.hetero_insert.y_min_mm);
    const auto insert_y_max = static_cast<float>(config.hetero_insert.y_max_mm);
    const auto insert_z_min = static_cast<float>(config.hetero_insert.z_min_mm);
    const auto insert_z_max = static_cast<float>(config.hetero_insert.z_max_mm);
    const auto insert_density_g_per_cm3 =
        static_cast<float>(config.hetero_insert.density_g_per_cm3);
    const auto insert_radiation_length_g_per_cm2 =
        static_cast<float>(config.insert_radiation_length_g_per_cm2);
    const auto use_insert_material_tables =
        enable_hetero_insert && !config.insert_stopping_power_file.empty();

    std::vector<float> insert_sp_host(use_insert_material_tables ? table_size : 0);
    std::vector<float> insert_xs_host(use_insert_material_tables ? cross_section_table_size : 0);
    if (use_insert_material_tables) {
        const auto sp = StoppingPowerTable::from_csv(config.insert_stopping_power_file);
        const auto xs = CrossSectionTable::from_csv(config.insert_cross_section_file);
        for (std::size_t i = 0; i < table_size; ++i) {
            insert_sp_host[i] = static_cast<float>(sp.values()[i]);
        }
        for (std::size_t i = 0; i < cross_section_table_size; ++i) {
            insert_xs_host[i] = static_cast<float>(xs.interpolate(table_energies[i]));
        }
    }
    float* insert_sp_device = use_insert_material_tables
                                  ? sycl::malloc_device<float>(table_size, queue)
                                  : nullptr;
    float* insert_xs_device = use_insert_material_tables
                                  ? sycl::malloc_device<float>(cross_section_table_size, queue)
                                  : nullptr;
    if (use_insert_material_tables) {
        queue.copy(insert_sp_host.data(), insert_sp_device, table_size);
        queue.copy(insert_xs_host.data(), insert_xs_device, cross_section_table_size)
            .wait_and_throw();
    }

    // CT grid
    const auto enable_ct_grid = config.enable_ct_grid;
    float ct_origin_x = 0.0F, ct_origin_y = 0.0F, ct_origin_z = 0.0F;
    float ct_spacing_x = 0.0F, ct_spacing_y = 0.0F, ct_spacing_z = 0.0F;
    std::uint32_t ct_nx = 0, ct_ny = 0, ct_nz = 0;
    float* ct_density_device = nullptr;
    std::uint8_t* ct_material_device = nullptr;
    float* ct_mass_sp_factor_lut_device = nullptr;
    float* ct_mass_sp_za_rel_device = nullptr;
    float* ct_sp_device = nullptr;
    float* ct_xs_device = nullptr;
    float* ct_ref_density_device = nullptr;
    std::uint32_t ct_n_mass_factors = 0;
    std::uint32_t ct_density_spr_n_rho = 0;
    float ct_mass_spr_log_rho_min = 0.0F;
    float ct_mass_spr_inv_dlog = 0.0F;
    bool use_ct_mass_sp = false;
    bool use_ct_density_mass_spr = false;
    bool use_ct_material_sp = false;
    bool use_ct_material_xs = false;
    bool ct_material_ids_are_schneider_sections = false;
    const auto ct_skip_homogeneous_face_clamp = config.ct_skip_homogeneous_face_clamp;

    if (enable_ct_grid) {
        const auto grid = CtGrid::load(
            config.ct_grid_file, config.ct_schneider_file, config.ct_dicom_origin_mode);
        ct_origin_x = grid.origin_x_mm;
        ct_origin_y = grid.origin_y_mm;
        ct_origin_z = grid.origin_z_mm;
        ct_spacing_x = grid.spacing_x_mm;
        ct_spacing_y = grid.spacing_y_mm;
        ct_spacing_z = grid.spacing_z_mm;
        ct_nx = grid.nx;
        ct_ny = grid.ny;
        ct_nz = grid.nz;
        ct_material_ids_are_schneider_sections =
            grid.file_version >= CtGrid::version_v2;

        const auto voxel_count = static_cast<std::size_t>(ct_nx) * ct_ny * ct_nz;
        ct_density_device = sycl::malloc_device<float>(voxel_count, queue);
        ct_material_device = sycl::malloc_device<std::uint8_t>(voxel_count, queue);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();

        use_ct_mass_sp = ct_material_ids_are_schneider_sections;
        use_ct_material_sp = !config.ct_water_stopping_power_file.empty() ||
                             !config.ct_bone_stopping_power_file.empty();
        use_ct_material_xs = !config.ct_bone_cross_section_file.empty() ||
                             !config.ct_schneider_cross_section_file.empty();

        if (use_ct_mass_sp) {
            ct_n_mass_factors = static_cast<std::uint32_t>(grid.mass_sp_za_rel.size());
            std::vector<float> mass_factor_lut(
                static_cast<std::size_t>(ct_n_mass_factors) * table_size);
            const auto sp_scale = static_cast<float>(config.ct_stopping_power_scale);

            const auto try_density_spr = [&]() -> bool {
                if (!config.ct_use_density_mass_spr) {
                    return false;
                }
                const auto resolve = [](const std::filesystem::path& user_path,
                                        const std::filesystem::path& fallback) {
                    return !user_path.empty() && std::filesystem::exists(user_path)
                               ? user_path
                               : (std::filesystem::exists(fallback) ? fallback
                                                                    : std::filesystem::path{});
                };
                const auto air_path = resolve(config.ct_air_stopping_power_file,
                                             "data/stopping_power_air_geant4_11_3_2.csv");
                const auto lung_path = resolve(config.ct_lung_stopping_power_file,
                                              "data/stopping_power_lung_geant4_11_3_2.csv");
                const auto bone_path = resolve(config.ct_bone_stopping_power_file,
                                              "data/stopping_power_bone_geant4_11_3_2.csv");
                if (air_path.empty() || lung_path.empty() || bone_path.empty()) {
                    return false;
                }
                const auto air_table = StoppingPowerTable::from_csv(air_path);
                const auto lung_table = StoppingPowerTable::from_csv(lung_path);
                const auto bone_table = StoppingPowerTable::from_csv(bone_path);
                const auto density_lut = build_density_mass_spr_lut(
                    stopping_power, air_table, lung_table, bone_table, sp_scale);
                if (density_lut.n_rho < 2 ||
                    density_lut.factors.size() !=
                        static_cast<std::size_t>(density_lut.n_rho) * table_size) {
                    return false;
                }
                mass_factor_lut = density_lut.factors;
                ct_density_spr_n_rho = density_lut.n_rho;
                ct_mass_spr_log_rho_min = density_lut.log_rho_min;
                ct_mass_spr_inv_dlog = density_lut.inv_dlog;
                use_ct_density_mass_spr = true;
                return true;
            };

            if (!try_density_spr()) {
                for (std::uint32_t sec = 0; sec < ct_n_mass_factors; ++sec) {
                    const auto za = grid.mass_sp_za_rel[sec];
                    const auto I_eV = grid.mass_sp_I_eV[sec];
                    const auto base = static_cast<std::size_t>(sec) * table_size;
                    for (std::size_t i = 0; i < table_size; ++i) {
                        mass_factor_lut[base + i] =
                            sp_scale * ct_mass_sp_energy_factor(
                                           za, I_eV,
                                           static_cast<float>(stopping_power.energies()[i]));
                    }
                }
            }
            const auto lut_bytes = mass_factor_lut.size();
            ct_mass_sp_factor_lut_device = sycl::malloc_device<float>(lut_bytes, queue);
            ct_mass_sp_za_rel_device = sycl::malloc_device<float>(ct_n_mass_factors, queue);
            queue.copy(mass_factor_lut.data(), ct_mass_sp_factor_lut_device, lut_bytes);
            queue.copy(grid.mass_sp_za_rel.data(), ct_mass_sp_za_rel_device, ct_n_mass_factors)
                .wait_and_throw();
        }
    }

    // Spot batching / TPS source
    const auto primary_spot_count = config.primary_spot_batch.size();
    PrimarySpotBatchEntry* primary_spots_device = nullptr;
    if (primary_spot_count > 0) {
        primary_spots_device =
            sycl::malloc_device<PrimarySpotBatchEntry>(primary_spot_count, queue);
        queue.copy(config.primary_spot_batch.data(), primary_spots_device, primary_spot_count)
            .wait_and_throw();
    }

    // Scorers & Result buffers
    const auto cinel02_max_secondary_inelastic_generations =
        config.cinel02_max_secondary_inelastic_generations;
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_let_scoring = config.enable_let_scoring;
    const auto voxel_scorer_clamps_transport = config.voxel_scorer_clamps_transport;
    const auto voxel_bins_x = config.voxel_bins_x;
    const auto voxel_bins_y = config.voxel_bins_y;
    const auto voxel_bins_z = config.voxel_bins_z;
    const auto voxel_size_x_mm = static_cast<float>(config.voxel_size_x_mm);
    const auto voxel_size_y_mm = static_cast<float>(config.voxel_size_y_mm);
    const auto voxel_size_z_mm = static_cast<float>(config.voxel_size_z_mm);
    const auto voxel_plane_size = voxel_bins_x * voxel_bins_y;
    float voxel_min_x_mm =
        -0.5F * static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_max_x_mm =
        voxel_min_x_mm + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
    float voxel_min_y_mm =
        -0.5F * static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    float voxel_max_y_mm =
        voxel_min_y_mm + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    if (enable_ct_grid &&
        std::abs(ct_spacing_x - voxel_size_x_mm) < 1.0e-5F &&
        std::abs(ct_spacing_y - voxel_size_y_mm) < 1.0e-5F) {
        voxel_min_x_mm = ct_origin_x;
        voxel_min_y_mm = ct_origin_y;
        voxel_max_x_mm =
            ct_origin_x + static_cast<float>(voxel_bins_x) * voxel_size_x_mm;
        voxel_max_y_mm =
            ct_origin_y + static_cast<float>(voxel_bins_y) * voxel_size_y_mm;
    }

    auto* dose_device = sycl::malloc_device<DoseAtomicT>(number_of_bins, queue);
    std::uint64_t* primary_survival_device = nullptr;
    std::uint64_t* inelastic_reaction_device = nullptr;
#ifdef CARBON_VALIDATION_SCORERS
    if (config.validation_scorers()) {
        primary_survival_device =
            sycl::malloc_device<std::uint64_t>(number_of_bins, queue);
        inelastic_reaction_device =
            sycl::malloc_device<std::uint64_t>(number_of_bins, queue);
        if (primary_survival_device == nullptr || inelastic_reaction_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(primary_survival_device, std::uint64_t{0}, number_of_bins);
        queue.fill(inelastic_reaction_device, std::uint64_t{0}, number_of_bins);
    }
#endif
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? sycl::malloc_device<DoseAtomicT>(number_of_voxels, queue)
                                  : nullptr;
    const auto enable_charged_origin_voxel_scoring =
        config.enable_charged_origin_voxel_scoring;
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? sycl::malloc_device<DoseAtomicT>(
                  charged_origin_category_count * number_of_voxels, queue)
            : nullptr;
    auto* be_isotope_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? sycl::malloc_device<DoseAtomicT>(
                  be_isotope_origin_category_count * number_of_voxels, queue)
            : nullptr;
    auto* let_moments_device = enable_let_scoring
                                   ? sycl::malloc_device<LetAtomicT>(4 * number_of_bins, queue)
                                   : nullptr;
    auto* voxel_let_moments_device =
        enable_let_scoring && enable_voxel_scoring
            ? sycl::malloc_device<LetAtomicT>(4 * number_of_voxels, queue)
            : nullptr;

    auto* deposited_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* escaped_device = sycl::malloc_device<float>(number_of_histories, queue);
    auto* steps_device = sycl::malloc_device<std::uint32_t>(number_of_histories, queue);
    auto* untracked_nuclear_device = sycl::malloc_device<float>(number_of_histories, queue);
    if (untracked_nuclear_device != nullptr) {
        queue.fill(untracked_nuclear_device, 0.0F, number_of_histories).wait_and_throw();
    }

    const auto enable_inelastic = config.enable_inelastic;
    const auto enable_nuclear_elastic = config.enable_nuclear_elastic;
    const auto enable_secondary_transport = config.enable_secondary_transport;
    constexpr std::size_t max_secondaries = 32000000;
    auto* secondary_queue_device =
        (enable_inelastic || enable_nuclear_elastic)
            ? sycl::malloc_device<SecondaryParticle>(max_secondaries, queue)
            : nullptr;
    auto* secondary_count_device =
        (enable_inelastic || enable_nuclear_elastic)
            ? sycl::malloc_device<uint32_t>(1, queue)
            : nullptr;
    uint32_t* secondary_overflow_count_device =
        (enable_inelastic || enable_nuclear_elastic)
            ? sycl::malloc_device<uint32_t>(1, queue)
            : nullptr;
    float* secondary_overflow_energy_device =
        (enable_inelastic || enable_nuclear_elastic)
            ? sycl::malloc_device<float>(1, queue)
            : nullptr;
    float* fred_model_residual_device =
        enable_inelastic ? sycl::malloc_device<float>(1, queue) : nullptr;
    float* fred_q_device = enable_inelastic ? sycl::malloc_device<float>(1, queue) : nullptr;
    float* fred_neutron_device = enable_inelastic ? sycl::malloc_device<float>(1, queue) : nullptr;
    float* fred_remnant_device = enable_inelastic ? sycl::malloc_device<float>(1, queue) : nullptr;
    uint32_t* fred_fail_count_device =
        enable_inelastic ? sycl::malloc_device<uint32_t>(1, queue) : nullptr;
    float* fred_fail_energy_device =
        enable_inelastic ? sycl::malloc_device<float>(1, queue) : nullptr;
    uint32_t* fred_cap_overflow_count_device = nullptr;
    float* fred_cap_overflow_energy_device = nullptr;
    if (enable_inelastic) {
        fred_cap_overflow_count_device = sycl::malloc_device<uint32_t>(1, queue);
        fred_cap_overflow_energy_device = sycl::malloc_device<float>(1, queue);
        const bool inelastic_ok =
            secondary_queue_device != nullptr && secondary_count_device != nullptr &&
            secondary_overflow_count_device != nullptr &&
            secondary_overflow_energy_device != nullptr &&
            fred_model_residual_device != nullptr && fred_q_device != nullptr &&
            fred_neutron_device != nullptr && fred_remnant_device != nullptr &&
            fred_fail_count_device != nullptr && fred_fail_energy_device != nullptr &&
            fred_cap_overflow_count_device != nullptr &&
            fred_cap_overflow_energy_device != nullptr;
        if (!inelastic_ok) {
            free_device(secondary_queue_device);
            free_device(secondary_count_device);
            free_device(secondary_overflow_count_device);
            free_device(secondary_overflow_energy_device);
            free_device(fred_model_residual_device);
            free_device(fred_q_device);
            free_device(fred_neutron_device);
            free_device(fred_remnant_device);
            free_device(fred_fail_count_device);
            free_device(fred_fail_energy_device);
            free_device(fred_cap_overflow_count_device);
            free_device(fred_cap_overflow_energy_device);
            throw std::bad_alloc();
        }
        queue.fill(secondary_count_device, 0U, 1);
        queue.fill(secondary_overflow_count_device, 0U, 1);
        queue.fill(secondary_overflow_energy_device, 0.0F, 1);
        queue.fill(fred_model_residual_device, 0.0F, 1);
        queue.fill(fred_q_device, 0.0F, 1);
        queue.fill(fred_neutron_device, 0.0F, 1);
        queue.fill(fred_remnant_device, 0.0F, 1);
        queue.fill(fred_fail_count_device, 0U, 1);
        queue.fill(fred_fail_energy_device, 0.0F, 1);
        queue.fill(fred_cap_overflow_count_device, 0U, 1);
        queue.fill(fred_cap_overflow_energy_device, 0.0F, 1).wait_and_throw();
    } else if (enable_nuclear_elastic) {
        if (secondary_queue_device == nullptr || secondary_count_device == nullptr ||
            secondary_overflow_count_device == nullptr ||
            secondary_overflow_energy_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(secondary_count_device, 0U, 1);
        queue.fill(secondary_overflow_count_device, 0U, 1);
        queue.fill(secondary_overflow_energy_device, 0.0F, 1).wait_and_throw();
    }

    constexpr std::size_t kFredDiagSlots = 26;
    float* fred_prob_proj_h_device = nullptr;
    float* fred_prob_proj_o_device = nullptr;
    float* fred_prob_tgt_h_device = nullptr;
    float* fred_prob_tgt_o_device = nullptr;
    std::uint64_t* fred_diag_device = nullptr;
    float invert_err_proj_h = 0.0F;
    float invert_err_proj_o = 0.0F;
    float invert_err_tgt_h = 0.0F;
    float invert_err_tgt_o = 0.0F;
    if (enable_inelastic) {
        static std::array<float, 18> cached_proj_h{};
        static std::array<float, 18> cached_proj_o{};
        static std::array<float, 18> cached_tgt_h{};
        static std::array<float, 18> cached_tgt_o{};
        static float cached_err_proj_h = 0.0F;
        static float cached_err_proj_o = 0.0F;
        static float cached_err_tgt_h = 0.0F;
        static float cached_err_tgt_o = 0.0F;
        static bool cached_sample_ready = false;
        if (!cached_sample_ready) {
            cached_proj_h = kFredProbH;
            cached_proj_o = kFredProbO;
            cached_tgt_h = kFredProbH;
            cached_tgt_o = kFredProbO;
            cached_err_proj_h = 0.0F;
            cached_err_proj_o = 0.0F;
            cached_err_tgt_h = 0.0F;
            cached_err_tgt_o = 0.0F;
            cached_sample_ready = true;
        }
        invert_err_proj_h = cached_err_proj_h;
        invert_err_proj_o = cached_err_proj_o;
        invert_err_tgt_h = cached_err_tgt_h;
        invert_err_tgt_o = cached_err_tgt_o;
        fred_prob_proj_h_device = sycl::malloc_device<float>(18, queue);
        fred_prob_proj_o_device = sycl::malloc_device<float>(18, queue);
        fred_prob_tgt_h_device = sycl::malloc_device<float>(18, queue);
        fred_prob_tgt_o_device = sycl::malloc_device<float>(18, queue);
        fred_diag_device = sycl::malloc_device<std::uint64_t>(kFredDiagSlots, queue);
        if (fred_prob_proj_h_device == nullptr || fred_prob_proj_o_device == nullptr ||
            fred_prob_tgt_h_device == nullptr || fred_prob_tgt_o_device == nullptr ||
            fred_diag_device == nullptr) {
            free_device(fred_prob_proj_h_device);
            free_device(fred_prob_proj_o_device);
            free_device(fred_prob_tgt_h_device);
            free_device(fred_prob_tgt_o_device);
            free_device(fred_diag_device);
            throw std::bad_alloc();
        }
        queue.copy(cached_proj_h.data(), fred_prob_proj_h_device, 18);
        queue.copy(cached_proj_o.data(), fred_prob_proj_o_device, 18);
        queue.copy(cached_tgt_h.data(), fred_prob_tgt_h_device, 18);
        queue.copy(cached_tgt_o.data(), fred_prob_tgt_o_device, 18);
        queue.fill(fred_diag_device, static_cast<std::uint64_t>(0), kFredDiagSlots)
            .wait_and_throw();
    }

    carbon::FredEventLibSetDevice lib_h_view{};
    carbon::FredEventLibSetDevice lib_o_view{};
    std::vector<void*> event_lib_device_allocations;
    const auto upload_event_set = [&](std::vector<std::filesystem::path> paths,
                                      const std::filesystem::path& legacy_path,
                                      const std::uint16_t expected_z,
                                      const std::uint16_t expected_a,
                                      carbon::FredEventLibSetDevice& set) {
        if (paths.empty() && !legacy_path.empty()) paths.push_back(legacy_path);
        std::vector<carbon::FredEventLibrary> hosts;
        for (const auto& path : paths) {
            if (path.empty() || !std::filesystem::exists(path))
                throw std::runtime_error("Missing FRED event library: " + path.string());
            auto host = carbon::load_fred_event_library(path);
            if (host.target_z != expected_z || host.target_a != expected_a)
                throw std::runtime_error("FRED event-library target identity mismatch: " +
                                         path.string());
            hosts.push_back(std::move(host));
        }
        std::sort(hosts.begin(), hosts.end(), [](const auto& a, const auto& b) {
            return a.reference_energy_MeVu < b.reference_energy_MeVu;
        });
        if (hosts.size() > 4) throw std::runtime_error("At most four event-library energies are supported");
        for (std::size_t k = 1; k < hosts.size(); ++k)
            if (hosts[k].reference_energy_MeVu <= hosts[k - 1].reference_energy_MeVu)
                throw std::runtime_error("Event-library energies must be unique");
        set.count = static_cast<std::uint32_t>(hosts.size());
        for (std::size_t k = 0; k < hosts.size(); ++k) {
            const auto& host = hosts[k];
            const auto n = host.event_count;
            const auto slots = static_cast<std::size_t>(n) * host.max_fragments;
            auto alloc = [&](auto*& pointer, std::size_t count) {
                using T = std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
                pointer = sycl::malloc_device<T>(count, queue);
                if (pointer == nullptr) throw std::bad_alloc();
                event_lib_device_allocations.push_back(static_cast<void*>(pointer));
            };
            std::uint8_t* nfrag{}; float* nke{}; std::int8_t* z{}; std::int8_t* a{};
            float* ke{}; float* ux{}; float* uy{}; float* uz{};
            alloc(nfrag, n); alloc(nke, n); alloc(z, slots); alloc(a, slots);
            alloc(ke, slots); alloc(ux, slots); alloc(uy, slots); alloc(uz, slots);
            queue.copy(host.fragment_count.data(), nfrag, n);
            queue.copy(host.neutron_ke_MeV.data(), nke, n);
            queue.copy(host.z.data(), z, slots); queue.copy(host.a.data(), a, slots);
            queue.copy(host.ke_MeV.data(), ke, slots); queue.copy(host.ux.data(), ux, slots);
            queue.copy(host.uy.data(), uy, slots); queue.copy(host.uz.data(), uz, slots)
                .wait_and_throw();
            set.libraries[k] = carbon::FredEventLibDevice{
                n, host.max_fragments, host.reference_energy_MeVu,
                nfrag, nke, z, a, ke, ux, uy, uz};
            std::cout << "Loaded FRED event library at " << host.reference_energy_MeVu
                      << " MeV/u (" << n << " events)\n";
        }
    };
    if (enable_inelastic) {
        upload_event_set(config.fred_event_library_h_files,
                         config.fred_event_library_h_file, 1, 1, lib_h_view);
        upload_event_set(config.fred_event_library_o_files,
                         config.fred_event_library_o_file, 8, 16, lib_o_view);
        for (const auto& path : config.fred_event_library_c_files) {
            const auto host = carbon::load_fred_event_library(path);
            if (host.target_z != 6 || host.target_a != 12)
                throw std::runtime_error("FRED carbon event-library target mismatch: " +
                                         path.string());
        }
    }

    float* fluct_energy_device = nullptr;
    float* fluct_density_device = nullptr;
    float* fluct_probability_device = nullptr;
    float* fluct_quantile_device = nullptr;
    std::size_t fluct_energy_count = 0;
    std::size_t fluct_density_count = 0;
    std::size_t fluct_probability_count = 0;
    if (config.uses_packaged_fluctuation()) {
        const auto host = carbon::EnergyLossFluctuationTable::from_csv(
            config.energy_straggling_package_file);
        if (host.projectile_atomic_number() != 6 ||
            host.projectile_mass_number() != 12 ||
            host.material_name() != "G4_WATER")
            throw std::runtime_error(
                "Packaged fluctuation must describe C-12 in G4_WATER");
        const auto to_float = [](const std::vector<double>& input) {
            std::vector<float> output(input.size());
            std::transform(input.begin(), input.end(), output.begin(),
                           [](double value) { return static_cast<float>(value); });
            return output;
        };
        const auto energies = to_float(host.energies_MeVu());
        const auto densities = to_float(host.areal_densities_g_per_cm2());
        const auto probabilities = to_float(host.probabilities());
        const auto quantiles = to_float(host.loss_ratio_quantiles());
        fluct_energy_count = energies.size();
        fluct_density_count = densities.size();
        fluct_probability_count = probabilities.size();
        fluct_energy_device = sycl::malloc_device<float>(energies.size(), queue);
        fluct_density_device = sycl::malloc_device<float>(densities.size(), queue);
        fluct_probability_device = sycl::malloc_device<float>(probabilities.size(), queue);
        fluct_quantile_device = sycl::malloc_device<float>(quantiles.size(), queue);
        if (!fluct_energy_device || !fluct_density_device ||
            !fluct_probability_device || !fluct_quantile_device)
            throw std::bad_alloc();
        queue.copy(energies.data(), fluct_energy_device, energies.size());
        queue.copy(densities.data(), fluct_density_device, densities.size());
        queue.copy(probabilities.data(), fluct_probability_device, probabilities.size());
        queue.copy(quantiles.data(), fluct_quantile_device, quantiles.size()).wait_and_throw();
        std::cout << "Loaded packaged C-12 fluctuation grid (" << fluct_energy_count
                  << " energies, " << fluct_density_count << " thicknesses, "
                  << fluct_probability_count << " quantiles)\n";
    }

    float* fred_2gr_mcs_device = nullptr;
    if (config.uses_fred_2gr_mcs()) {
        const auto host = carbon::Fred2GrMcsTable::from_binary(config.fred_2gr_mcs_file);
        fred_2gr_mcs_device = sycl::malloc_device<float>(host.values.size(), queue);
        if (fred_2gr_mcs_device == nullptr) throw std::bad_alloc();
        queue.copy(host.values.data(), fred_2gr_mcs_device, host.values.size())
            .wait_and_throw();
        std::cout << "Loaded FRED 3.76 2GR MCS table (51x48x6)\n";
    }

    float* ion_species_sp_device = nullptr;
    float* ion_energy_grid_device = nullptr;
    float* ion_csda_a1_device = nullptr;
    if (enable_inelastic) {
        std::filesystem::path ion_sp_path = "data/ion_stopping_power_water_geant4_11_3_2.csv";
        if (!std::filesystem::exists(ion_sp_path)) {
            const auto cur_p = std::filesystem::current_path();
            if (std::filesystem::exists(cur_p / "data" / "ion_stopping_power_water_geant4_11_3_2.csv")) {
                ion_sp_path = cur_p / "data" / "ion_stopping_power_water_geant4_11_3_2.csv";
            }
        }
        if (!std::filesystem::exists(ion_sp_path)) {
            throw std::runtime_error("Required ion stopping-power CSV not found: " + ion_sp_path.string());
        }
        const auto ion_sp_lut = load_ion_species_stopping_power_lut(ion_sp_path, table_size, 1.0F);
        ion_species_sp_device = sycl::malloc_device<float>(18 * table_size, queue);
        queue.copy(ion_sp_lut.data(), ion_species_sp_device, 18 * table_size).wait_and_throw();
        std::vector<float> energy_grid_host(table_size);
        std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                       energy_grid_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        ion_energy_grid_device = sycl::malloc_device<float>(table_size, queue);
        ion_csda_a1_device = sycl::malloc_device<float>(18 * table_size, queue);
        if (ion_energy_grid_device == nullptr || ion_csda_a1_device == nullptr) {
            throw std::bad_alloc();
        }
        std::vector<float> csda_host(18 * table_size, 0.0F);
        for (int species = 0; species < 18; ++species) {
            fill_a1_csda_range_mm(energy_grid_host.data(),
                                  ion_sp_lut.data() + static_cast<std::size_t>(species) * table_size,
                                  table_size,
                                  csda_host.data() + static_cast<std::size_t>(species) * table_size);
        }
        queue.copy(energy_grid_host.data(), ion_energy_grid_device, table_size);
        queue.copy(csda_host.data(), ion_csda_a1_device, 18 * table_size).wait_and_throw();
    }

    const auto primary_species_sp_idx = carbon::get_charged_species_idx(
        config.primary_atomic_number, config.primary_mass_number);
    if (enable_inelastic && primary_species_sp_idx < 0) {
        throw std::runtime_error(
            "Primary ion is absent from the explicit ion stopping-power table");
    }
    const auto primary_species_offset = primary_species_sp_idx >= 0
        ? static_cast<std::size_t>(primary_species_sp_idx) * table_size
        : 0U;
    const float* primary_water_sp_device = enable_inelastic
        ? ion_species_sp_device + primary_species_offset
        : table_device;
    const float* primary_csda_a1_device = enable_inelastic
        ? ion_csda_a1_device + primary_species_offset
        : cumulative_range_device;

    auto* in_fov_dose_device =
        enable_voxel_scoring ? sycl::malloc_device<DoseAtomicT>(number_of_bins, queue) : nullptr;
    if (in_fov_dose_device != nullptr) {
        queue.fill(in_fov_dose_device, DoseAtomicT{0}, number_of_bins).wait_and_throw();
    }

    if (dose_device == nullptr || deposited_device == nullptr ||
        escaped_device == nullptr || steps_device == nullptr ||
        (enable_voxel_scoring && (voxel_dose_device == nullptr || in_fov_dose_device == nullptr)) ||
        (enable_charged_origin_voxel_scoring &&
         (charged_origin_voxel_dose_device == nullptr ||
          be_isotope_origin_voxel_dose_device == nullptr)) ||
        (enable_let_scoring && let_moments_device == nullptr)) {
        throw std::bad_alloc();
    }

    if (!reuse_immutable_buffers) {
        std::vector<float> table_host(table_size);
        std::transform(stopping_power.values().begin(), stopping_power.values().end(),
                       table_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        queue.copy(table_host.data(), table_device, table_size);
        if (config.enable_csda_range_energy_loss) {
            std::vector<float> energy_grid_host(table_size);
            std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                           energy_grid_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(energy_grid_host.data(), energy_grid_device, table_size);
            std::vector<float> cumulative_range_host(table_size);
            std::transform(stopping_power.cumulative_ranges_mm().begin(),
                           stopping_power.cumulative_ranges_mm().end(),
                           cumulative_range_host.begin(),
                           [](double value) { return static_cast<float>(value); });
            queue.copy(cumulative_range_host.data(), cumulative_range_device, table_size);
        }
        const auto resampled_xs =
            resample_cross_section_grid(cross_section, table_energies);
        queue.copy(resampled_xs.macroscopic_per_mm.data(), cross_section_device,
                   cross_section_table_size);
        if (target_h_fraction_device != nullptr) {
            queue.copy(resampled_xs.target_h_fraction.data(), target_h_fraction_device,
                       cross_section_table_size);
        }
        queue.wait_and_throw();
    } else if (target_h_fraction_device != nullptr) {
        const auto resampled_xs =
            resample_cross_section_grid(cross_section, table_energies);
        queue.copy(resampled_xs.target_h_fraction.data(), target_h_fraction_device,
                   cross_section_table_size).wait_and_throw();
    }

    queue.memset(dose_device, 0, number_of_bins * sizeof(DoseAtomicT));
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
        queue.memset(be_isotope_origin_voxel_dose_device, 0,
                     be_isotope_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
    }
    if (enable_let_scoring) {
        queue.memset(let_moments_device, 0, 4 * number_of_bins * sizeof(LetAtomicT));
        if (voxel_let_moments_device != nullptr) {
            queue.memset(voxel_let_moments_device, 0, 4 * number_of_voxels * sizeof(LetAtomicT));
        }
    }

    const std::size_t local_size = is_cuda_backend ? 128U : (device.is_gpu() ? 256U : 128U);
    std::size_t history_chunk = config.history_chunk_size;
    if (history_chunk == 0) {
        history_chunk = is_cuda_backend ? (number_of_histories > 1000000 ? 16384 : 4096)
                                        : (device.is_gpu() ? 8192 : number_of_histories);
    }
    history_chunk = std::max<std::size_t>(1, history_chunk);

    const auto initial_energy_MeV = static_cast<float>(config.initial_total_energy_MeV());
    const auto beam_energy_spread = static_cast<float>(config.beam_energy_spread);
    const auto phantom_length_mm = static_cast<float>(config.phantom_length_mm);
    const auto depth_bin_width_mm = static_cast<float>(config.depth_bin_width_mm);
    const auto inverse_depth_bin_width_mm = 1.0F / depth_bin_width_mm;
    const auto inverse_voxel_size_x_mm = 1.0F / voxel_size_x_mm;
    const auto inverse_voxel_size_y_mm = 1.0F / voxel_size_y_mm;
    const auto maximum_step_mm = static_cast<float>(config.maximum_step_mm);
    const auto maximum_relative_energy_loss =
        static_cast<float>(config.maximum_relative_energy_loss);
    const auto energy_cutoff_MeV = static_cast<float>(config.energy_cutoff_MeV);

    if (is_cuda_backend && !config.enable_minibeam) {
        cuda_clock_warmup(queue);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto enable_step_stable_straggling =
        enable_energy_straggling && config.enable_step_stable_straggling;
    const auto straggling_sampling_length_mm =
        static_cast<float>(config.straggling_sampling_length_mm);
    const auto straggling_scale = static_cast<float>(config.straggling_scale);
    const auto straggling_sampler = config.straggling_sampler_id();
    const auto use_packaged_fluctuation = config.uses_packaged_fluctuation();
    const auto enable_secondary_energy_straggling =
        config.enable_secondary_energy_straggling;

    std::array<float, max_straggling_scale_points> straggling_scale_energies{};
    std::array<float, max_straggling_scale_points> straggling_scale_values{};
    const auto straggling_scale_point_count =
        config.straggling_scale_energies_MeVu.size();
    for (std::size_t index = 0; index < straggling_scale_point_count; ++index) {
        straggling_scale_energies[index] =
            static_cast<float>(config.straggling_scale_energies_MeVu[index]);
        straggling_scale_values[index] =
            static_cast<float>(config.straggling_scale_values[index]);
    }

    const auto multiple_scattering_scale =
        static_cast<float>(config.multiple_scattering_scale);
    const auto water_density_g_per_cm3 = static_cast<float>(config.water_density_g_per_cm3);
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
    const auto use_fred_2gr_mcs = config.uses_fred_2gr_mcs();
    const auto extrapolate_fred_2gr_high_energy =
        config.uses_fred_2gr_high_energy_extrapolation();
    const auto enable_ct_material_mcs = config.enable_ct_material_mcs;
    const auto enable_tps_source = config.uses_fixed_patient_coordinates();
    const auto random_seed = config.random_seed;
    const auto enable_flat_source = config.enable_flat_source;
    const auto flat_source_half_width_x_mm =
        static_cast<float>(config.flat_source_half_width_x_mm);
    const auto flat_source_half_width_y_mm =
        static_cast<float>(config.flat_source_half_width_y_mm);
    const auto enable_emittance_source = config.enable_emittance_source;
    const auto emittance_sigma_x_mm = static_cast<float>(config.emittance_sigma_x_mm);
    const auto emittance_sigma_y_mm = static_cast<float>(config.emittance_sigma_y_mm);
    const auto emittance_sigma_x_prime = static_cast<float>(config.emittance_sigma_x_prime);
    const auto emittance_sigma_y_prime = static_cast<float>(config.emittance_sigma_y_prime);
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
    const auto emittance_correlation_x = static_cast<float>(config.emittance_correlation_x);
    const auto emittance_correlation_y = static_cast<float>(config.emittance_correlation_y);
    const auto inverse_mass_number = 1.0f / static_cast<float>(config.primary_mass_number);
    const auto primary_mass_number = config.primary_mass_number;
    const auto enable_csda_range_energy_loss = config.enable_csda_range_energy_loss;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);

    double primary_kernel_seconds = 0.0;

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
                    const auto sigma_x = spot != nullptr ? spot->floats[2] : emittance_sigma_x_mm;
                    const auto sigma_y = spot != nullptr ? spot->floats[3] : emittance_sigma_y_mm;
                    const auto sigma_x_prime =
                        spot != nullptr ? spot->floats[4] : emittance_sigma_x_prime;
                    const auto sigma_y_prime =
                        spot != nullptr ? spot->floats[5] : emittance_sigma_y_prime;
                    local_x_mm = sigma_x * g0;
                    local_y_mm = sigma_y * g2;
                    const auto rho_x = sycl::clamp(
                        spot != nullptr ? spot->floats[6] : emittance_correlation_x,
                        -0.9999F, 0.9999F);
                    const auto rho_y = sycl::clamp(
                        spot != nullptr ? spot->floats[7] : emittance_correlation_y,
                        -0.9999F, 0.9999F);
                    const auto x_prime =
                        sigma_x_prime * (rho_x * g0 + sycl::sqrt(1.0F - rho_x * rho_x) * g1);
                    const auto y_prime =
                        sigma_y_prime * (rho_y * g2 + sycl::sqrt(1.0F - rho_y * rho_y) * g3);
                    const auto inv_norm =
                        sycl::rsqrt(1.0F + x_prime * x_prime + y_prime * y_prime);
                    local_dx = x_prime * inv_norm;
                    local_dy = y_prime * inv_norm;
                    local_dz = inv_norm;
                }

                const auto origin_x = spot != nullptr ? spot->floats[8] : source_origin_x_mm;
                const auto origin_y = spot != nullptr ? spot->floats[9] : source_origin_y_mm;
                const auto origin_z = spot != nullptr ? spot->floats[10] : source_origin_z_mm;
                const auto ux_x = spot != nullptr ? spot->floats[11] : beam_ux_x;
                const auto ux_y = spot != nullptr ? spot->floats[12] : beam_ux_y;
                const auto ux_z = spot != nullptr ? spot->floats[13] : beam_ux_z;
                const auto uy_x = spot != nullptr ? spot->floats[14] : beam_uy_x;
                const auto uy_y = spot != nullptr ? spot->floats[15] : beam_uy_y;
                const auto uy_z = spot != nullptr ? spot->floats[16] : beam_uy_z;
                const auto uz_x = spot != nullptr ? spot->floats[17] : beam_uz_x;
                const auto uz_y = spot != nullptr ? spot->floats[18] : beam_uz_y;
                const auto uz_z = spot != nullptr ? spot->floats[19] : beam_uz_z;

                auto position_x_mm = origin_x + ux_x * local_x_mm + uy_x * local_y_mm;
                auto position_y_mm = origin_y + ux_y * local_x_mm + uy_y * local_y_mm;
                auto position_z_mm = origin_z + ux_z * local_x_mm + uy_z * local_y_mm;
                auto direction_x = ux_x * local_dx + uy_x * local_dy + uz_x * local_dz;
                auto direction_y = ux_y * local_dx + uy_x * local_dy + uz_y * local_dz;
                auto direction_z = ux_z * local_dx + uy_z * local_dy + uz_z * local_dz;
                {
                    const auto inv_n = sycl::rsqrt(sycl::fmax(
                        1.0e-20F, direction_x * direction_x + direction_y * direction_y +
                                      direction_z * direction_z));
                    direction_x *= inv_n;
                    direction_y *= inv_n;
                    direction_z *= inv_n;
                }

                if (enable_tps_source) {
                    auto t_enter = 0.0F;
                    auto t_exit = 1.0e30F;
                    auto hit = true;
                    auto intersect_slab = [&](const float position, const float direction,
                                              const float lower, const float upper) {
                        if (sycl::fabs(direction) < 1.0e-8F) {
                            if (position < lower || position >= upper) {
                                hit = false;
                            }
                            return;
                        }
                        auto first = (lower - position) / direction;
                        auto second = (upper - position) / direction;
                        if (first > second) {
                            const auto temporary = first;
                            first = second;
                            second = temporary;
                        }
                        t_enter = sycl::fmax(t_enter, first);
                        t_exit = sycl::fmin(t_exit, second);
                        if (t_exit < t_enter) {
                            hit = false;
                        }
                    };
                    if (enable_voxel_scoring) {
                        intersect_slab(position_x_mm, direction_x, voxel_min_x_mm, voxel_max_x_mm);
                        intersect_slab(position_y_mm, direction_y, voxel_min_y_mm, voxel_max_y_mm);
                    }
                    intersect_slab(position_z_mm, direction_z, 0.0F, phantom_length_mm);
                    if (hit && t_exit >= t_enter) {
                        const auto entry = t_enter + 1.0e-4F;
                        position_x_mm += entry * direction_x;
                        position_y_mm += entry * direction_y;
                        position_z_mm += entry * direction_z;
                    }
                } else {
                    if (sycl::fabs(direction_z) > 1.0e-8F) {
                        const auto t_plane = -position_z_mm / direction_z;
                        position_x_mm += t_plane * direction_x;
                        position_y_mm += t_plane * direction_y;
                        position_z_mm = 0.0F;
                    }
                }

                auto history_deposited_MeV = 0.0F;
                std::uint32_t steps = 0;
                StepStableStragglingState<float> stable_straggling;
                stable_straggling.initialize(straggling_sampling_length_mm);

                double pending_primary_depth_MeV = 0.0;
                double pending_let_numerator = 0.0;
                double pending_let_denominator = 0.0;
                double pending_voxel_let_numerator = 0.0;
                double pending_voxel_let_denominator = 0.0;
                int pending_primary_bin = 0;
                double pending_primary_voxel_MeV = 0.0;
                std::size_t pending_primary_voxel = 0;
                auto last_primary_stopping_power_MeV_per_mm = 0.0F;
                auto last_primary_density_g_per_cm3 = 0.0F;

                constexpr std::uint32_t max_primary_steps = 2000000U;
                int last_survival_bin = -1;
                while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    const auto escaped_z =
                        position_z_mm < 0.0F || position_z_mm >= phantom_length_mm;
                    if (escaped_z) break;

                    const auto absolute_direction_x = sycl::fabs(direction_x);
                    const auto absolute_direction_y = sycl::fabs(direction_y);
                    const auto absolute_direction_z = sycl::fabs(direction_z);
                    auto bin = direction_z < 0.0F
                                   ? static_cast<int>(
                                         sycl::ceil(position_z_mm / depth_bin_width_mm)) - 1
                                   : static_cast<int>(
                                         sycl::floor(position_z_mm / depth_bin_width_mm));
                    bin = sycl::max(0, sycl::min(bin, static_cast<int>(number_of_bins) - 1));
                    if (primary_survival_device != nullptr && bin != last_survival_bin) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            survival(primary_survival_device[bin]);
                        survival.fetch_add(1U);
                        last_survival_bin = bin;
                    }

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
                    const auto fraction =
                        sycl::clamp(floating_index - static_cast<float>(index), 0.0F, 1.0F);

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
                            ? slab_layer_index(position_z_mm, slab_z_ends_device, slab_layer_count)
                            : 0U;
                    float stopping_power_MeV_per_mm = 0.0F;
                    if (enable_ct_grid) {
                        const auto water_sp =
                            table_device[index] +
                            fraction * (table_device[index + 1] - table_device[index]);
                        if (use_ct_mass_sp && in_ct && ct_mass_sp_factor_lut_device != nullptr &&
                            ct_n_mass_factors > 0) {
                            const auto mass_factor = ct_lookup_mass_sp_factor(
                                ct_mass_sp_factor_lut_device,
                                use_ct_density_mass_spr ? ct_density_spr_n_rho : ct_n_mass_factors,
                                table_size, use_ct_density_mass_spr,
                                ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                static_cast<std::uint32_t>(ct_material), local_density_g_per_cm3,
                                static_cast<std::size_t>(index), fraction,
                                [](float x) { return sycl::log(x); });
                            stopping_power_MeV_per_mm = ct_mass_scaled_stopping_power(
                                water_sp, local_density_g_per_cm3, mass_factor);
                        } else {
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
                            primary_water_sp_device[index] +
                            fraction * (primary_water_sp_device[index + 1] -
                                        primary_water_sp_device[index]);
                        const auto scale_density =
                            (slab_layer_count > 0 || in_insert) ? local_density_g_per_cm3 : 1.0F;
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
                        if (dz_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dz_step);
                        }
                    }
                    if (slab_layer_count > 0) {
                        const auto slab_step = distance_to_slab_interface_mm(
                            position_z_mm, direction_z, slab_z_ends_device, slab_layer_count,
                            phantom_length_mm);
                        if (slab_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, slab_step);
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                            insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                        if (insert_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, insert_step);
                        }
                    }
                    if (enable_ct_grid && in_ct) {
                        step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz, ct_density_device,
                            ct_material_device, local_density_g_per_cm3, ct_material,
                            ct_skip_homogeneous_face_clamp, nullptr);
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        const auto dx_step = (boundary_x_mm - position_x_mm) / direction_x;
                        if (dx_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dx_step);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        const auto dy_step = (boundary_y_mm - position_y_mm) / direction_y;
                        if (dy_step > 1.0e-5F) {
                            step_mm = sycl::fmin(step_mm, dy_step);
                        }
                    }

                    if (enable_energy_straggling && enable_step_stable_straggling) {
                        stable_straggling.prepare_step(
                            step_mm, phantom_length_mm - position_z_mm);
                    }
                    step_mm = sycl::fmax(step_mm, 1.0e-5F);

                    bool inelastic_this_step = false;
                    bool elastic_this_step = false;
                    float p_target_h_step = 0.5F;
                    std::int16_t cinel02_target_z_step = 0;
                    std::int16_t cinel02_target_a_step = 0;
                    if ((enable_inelastic || enable_nuclear_elastic) &&
                        energy_MeV > energy_cutoff_MeV) {
                        const auto cur_e_u = energy_MeV * inverse_mass_number;
                        float macro_xs = 0.0F;
                        if (use_cinel02) {
                            cinel02_diag_increment_device(cinel02_diag_device, 0U);
                            const auto target = cinel02_select_water_target_device(
                                cinel02_rate_groups_device, cinel02_rate_group_count,
                                cinel02_rate_samples_device, cinel02_rate_sample_count,
                                primary_atomic_number, primary_mass_number, cur_e_u,
                                local_density_g_per_cm3, water_density_g_per_cm3,
                                rng::uniform01(spot_seed, rng_history, steps, 12));
                            if (target.covered) {
                                cinel02_diag_increment_device(cinel02_diag_device, 1U);
                                macro_xs = target.total_rate_per_mm;
                                cinel02_target_z_step = target.target_z;
                                cinel02_target_a_step = target.target_a;
                            }
                        } else if (enable_inelastic && cross_section_device != nullptr) {
                            const auto xs_flt =
                                (cur_e_u - minimum_table_energy) * inverse_table_step;
                            auto xs_idx = static_cast<int>(sycl::floor(xs_flt));
                            xs_idx = sycl::max(
                                0, sycl::min(xs_idx,
                                             static_cast<int>(cross_section_table_size) - 2));
                            const auto xs_fr =
                                sycl::clamp(xs_flt - static_cast<float>(xs_idx), 0.0F, 1.0F);
                            macro_xs =
                                cross_section_device[xs_idx] +
                                xs_fr * (cross_section_device[xs_idx + 1] -
                                         cross_section_device[xs_idx]);
                            if (target_h_fraction_device != nullptr) {
                                p_target_h_step =
                                    target_h_fraction_device[xs_idx] +
                                    xs_fr * (target_h_fraction_device[xs_idx + 1] -
                                             target_h_fraction_device[xs_idx]);
                            }
                        }
                        const float macro_el =
                            enable_nuclear_elastic
                                ? carbon::water_elastic_h_macro_per_mm(
                                      cur_e_u, water_density_g_per_cm3)
                                : 0.0F;
                        const float macro_tot = macro_xs + macro_el;
                        const auto u_nuc = rng::uniform01(
                            spot_seed, rng_history, steps, 8);
                        float collision_s = step_mm;
                        const bool collision = carbon::inelastic_collision_in_step(
                            macro_tot, step_mm, u_nuc, &collision_s);
                        if (collision) {
                            step_mm = collision_s;
                            const float u_br = rng::uniform01(
                                spot_seed, rng_history, steps, 9);
                            if (enable_nuclear_elastic && macro_tot > 0.0F &&
                                u_br * macro_tot < macro_el) {
                                elastic_this_step = true;
                            } else if (enable_inelastic) {
                                if (use_cinel02) {
                                    cinel02_diag_increment_device(cinel02_diag_device, 2U);
                                    cinel02_diag_increment_device(
                                        cinel02_diag_device,
                                        cinel02_target_z_step == 1 ? 6U : 7U);
                                    cinel02_diag_increment_device(
                                        cinel02_diag_device,
                                        cinel02_hazard_diag_slot_device(
                                            primary_atomic_number, primary_mass_number,
                                            cinel02_target_z_step, 0U,
                                            energy_MeV * inverse_mass_number));
                                }
                                if (inelastic_reaction_device != nullptr) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        reaction(inelastic_reaction_device[bin]);
                                    reaction.fetch_add(1U);
                                }
                                inelastic_this_step = true;
                            }
                        }
                    }

                    float mean_loss_MeV = 0.0F;
                    if (enable_csda_range_energy_loss && energy_grid_device != nullptr &&
                        cumulative_range_device != nullptr) {
                        const auto end_energy_MeVu = csda_energy_after_distance_device(
                            ion_energy_grid_device, primary_water_sp_device,
                            primary_csda_a1_device,
                            table_size, energy_MeVu, step_mm, primary_mass_number);
                        mean_loss_MeV = sycl::clamp(
                            energy_MeV -
                                static_cast<float>(primary_mass_number) * end_energy_MeVu,
                            0.0F, energy_MeV);
                    } else if (material_table_count == 0 && !in_insert &&
                               (!use_ct_mass_sp || !in_ct)) {
                        const auto mid_energy_MeV = sycl::fmax(
                            minimum_table_energy * static_cast<float>(primary_mass_number),
                            energy_MeV - 0.5F * stopping_power_MeV_per_mm * step_mm);
                        const auto mid_energy_MeVu = mid_energy_MeV * inverse_mass_number;
                        const auto mid_floating_index =
                            (mid_energy_MeVu - minimum_table_energy) * inverse_table_step;
                        auto mid_index =
                            static_cast<int>(sycl::floor(mid_floating_index));
                        mid_index = sycl::max(
                            0, sycl::min(mid_index, static_cast<int>(table_size) - 2));
                        const auto mid_fraction = sycl::clamp(
                            mid_floating_index - static_cast<float>(mid_index),
                            0.0F, 1.0F);
                        const auto mid_sp =
                            (primary_water_sp_device[mid_index] +
                             mid_fraction * (primary_water_sp_device[mid_index + 1] -
                                             primary_water_sp_device[mid_index])) *
                            sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        mean_loss_MeV = mid_sp * step_mm;
                    } else {
                        mean_loss_MeV = stopping_power_MeV_per_mm * step_mm;
                    }
                    auto deposited_MeV = sycl::fmin(mean_loss_MeV, energy_MeV);

                    if (enable_energy_straggling) {
                        if (use_packaged_fluctuation) {
                            const auto u_loss = rng::uniform01(
                                spot_seed, rng_history, steps, 2);
                            const auto ratio = sample_energy_loss_ratio_from_grid(
                                fluct_energy_device, fluct_energy_count,
                                fluct_density_device, fluct_density_count,
                                fluct_probability_device, fluct_probability_count,
                                fluct_quantile_device, energy_MeVu,
                                local_density_g_per_cm3 * step_mm / 10.0F, u_loss);
                            const auto local_scale = interpolate_straggling_scale(
                                energy_MeVu, straggling_scale_energies,
                                straggling_scale_values,
                                straggling_scale_point_count, straggling_scale);
                            const auto scaled_ratio =
                                scale_energy_loss_ratio_preserving_mean(ratio, local_scale);
                            deposited_MeV = sycl::clamp(
                                mean_loss_MeV * scaled_ratio, 0.0F, energy_MeV);
                        } else {
                        const auto local_scale = interpolate_straggling_scale(
                            energy_MeVu, straggling_scale_energies, straggling_scale_values,
                            straggling_scale_point_count, straggling_scale);
                        const auto effective_charge =
                            ion_effective_charge_device(primary_atomic_number, energy_MeVu);
                        const auto variance_MeV2 =
                            condensed_total_loss_variance_with_mass_MeV2_device(
                                energy_MeV, primary_rest_mass_MeV, effective_charge, step_mm,
                                local_density_g_per_cm3);
                        if (enable_step_stable_straggling) {
                            if (!stable_straggling.block_active) {
                                const auto u0 = sycl::fmax(
                                    rng::uniform01(spot_seed, rng_history,
                                                   stable_straggling.block_index, 0),
                                    1.0e-12F);
                                const auto u1 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 1);
                                const auto u2 = rng::uniform01(
                                    spot_seed, rng_history, stable_straggling.block_index, 2);
                                constexpr float two_pi = 6.2831853071795864769F;
                                const auto gauss =
                                    sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                                stable_straggling.begin_block(
                                    mean_loss_MeV, variance_MeV2, step_mm, local_scale, gauss,
                                    energy_MeV, u2, straggling_sampler);
                            }
                            deposited_MeV = stable_straggling.consume_loss(step_mm, energy_MeV);
                        } else {
                            const auto u0 = sycl::fmax(
                                rng::uniform01(spot_seed, rng_history, steps, 0), 1.0e-12F);
                            const auto u1 = rng::uniform01(spot_seed, rng_history, steps, 1);
                            const auto u2 = rng::uniform01(spot_seed, rng_history, steps, 2);
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto gauss =
                                sycl::sqrt(-2.0F * sycl::log(u0)) * sycl::cos(two_pi * u1);
                            const auto sigma_MeV =
                                local_scale * sycl::sqrt(sycl::fmax(0.0F, variance_MeV2));
                            deposited_MeV = sample_condensed_energy_loss(
                                mean_loss_MeV, sigma_MeV, gauss, u2, energy_MeV, straggling_sampler);
                        }
                        }
                    }

                    if (bin != pending_primary_bin) {
                        if (pending_primary_depth_MeV > 0.0) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_primary_bin]);
                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                            pending_primary_depth_MeV = 0.0;
                        }
                        if (enable_let_scoring) {
                            flush_letd_moments_device(
                                let_moments_device, number_of_bins,
                                static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                                pending_let_numerator, pending_let_denominator, true);
                            pending_let_numerator = 0.0;
                            pending_let_denominator = 0.0;
                        }
                        pending_primary_bin = bin;
                    }
                    if (enable_voxel_scoring && voxel_index != pending_primary_voxel) {
                        if (pending_primary_voxel_MeV > 0.0 && pending_primary_voxel >= 0 &&
                            pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                            atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            if (enable_charged_origin_voxel_scoring) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_primary_origin(
                                        charged_origin_voxel_dose_device[pending_primary_voxel]);
                                atomic_primary_origin.fetch_add(
                                    static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                            }
                        }
                        pending_primary_voxel_MeV = 0.0;
                        if (enable_let_scoring && voxel_let_moments_device != nullptr && pending_primary_voxel >= 0) {
                            flush_letd_moments_device(
                                voxel_let_moments_device, number_of_voxels,
                                pending_primary_voxel, nullptr, 0, 0,
                                pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                            pending_voxel_let_numerator = 0.0;
                            pending_voxel_let_denominator = 0.0;
                        }
                        pending_primary_voxel = voxel_index;
                    }

                    pending_primary_depth_MeV += deposited_MeV;
                    if (enable_voxel_scoring && voxel_index >= 0) {
                        pending_primary_voxel_MeV += deposited_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(deposited_MeV));
                        }
                    }
                    history_deposited_MeV += deposited_MeV;
                    last_primary_stopping_power_MeV_per_mm = stopping_power_MeV_per_mm;
                    last_primary_density_g_per_cm3 = local_density_g_per_cm3;

                    if (enable_let_scoring) {
                        const auto step_numerator =
                            static_cast<double>(deposited_MeV) *
                            static_cast<double>(stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(local_density_g_per_cm3, 1.0e-6F));
                        const auto step_denominator = static_cast<double>(deposited_MeV);
                        pending_let_numerator += step_numerator;
                        pending_let_denominator += step_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += step_numerator;
                            pending_voxel_let_denominator += step_denominator;
                        }
                    }

                    if (enable_multiple_scattering) {
                        auto radiation_length_g_per_cm2 =
                            static_cast<float>(water_radiation_length_g_per_cm2);
                        if (enable_ct_grid && in_ct && enable_ct_material_mcs) {
                            radiation_length_g_per_cm2 =
                                ct_material_radiation_length_g_per_cm2(
                                    ct_material_class(ct_material, ct_material_ids_are_schneider_sections));
                        } else if (in_insert) {
                            radiation_length_g_per_cm2 = insert_radiation_length_g_per_cm2;
                        } else if (slab_layer_count > 0) {
                            radiation_length_g_per_cm2 =
                                slab_radiation_lengths_device[layer_for_material];
                        }
                        float theta_x = 0.0F;
                        float theta_y = 0.0F;
                        constexpr float two_pi = 6.2831853071795864769F;
                        if (use_fred_2gr_mcs) {
                            const auto mixture = rng::uniform01(
                                spot_seed, rng_history, steps, 3);
                            const auto radial = rng::uniform01(
                                spot_seed, rng_history, steps, 4);
                            const auto azimuth = two_pi * rng::uniform01(
                                spot_seed, rng_history, steps, 5);
                            const auto angle = fred_2gr_angle_device(
                                fred_2gr_mcs_device, energy_MeVu,
                                primary_atomic_number, primary_mass_number,
                                local_density_g_per_cm3 * step_mm / 10.0F,
                                radiation_length_g_per_cm2,
                                extrapolate_fred_2gr_high_energy,
                                multiple_scattering_scale, mixture, radial);
                            theta_x = angle * sycl::cos(azimuth);
                            theta_y = angle * sycl::sin(azimuth);
                        } else {
                            const auto theta0 = highland_projected_rms_angle_device(
                                energy_MeV, primary_atomic_number, primary_mass_number,
                                step_mm, local_density_g_per_cm3,
                                radiation_length_g_per_cm2) * multiple_scattering_scale;
                            const auto u0 = sycl::fmax(rng::uniform01(
                                spot_seed, rng_history, steps, 3), 1.0e-12F);
                            const auto u1 = rng::uniform01(
                                spot_seed, rng_history, steps, 4);
                            const auto u2 = sycl::fmax(rng::uniform01(
                                spot_seed, rng_history, steps, 5), 1.0e-12F);
                            const auto u3 = rng::uniform01(
                                spot_seed, rng_history, steps, 6);
                            theta_x = theta0 * sycl::sqrt(-2.0F * sycl::log(u0)) *
                                      sycl::cos(two_pi * u1);
                            theta_y = theta0 * sycl::sqrt(-2.0F * sycl::log(u2)) *
                                      sycl::cos(two_pi * u3);
                        }

                        const auto transverse_magnitude =
                            sycl::sqrt(theta_x * theta_x + theta_y * theta_y);
                        const auto local_direction_z =
                            sycl::cos(sycl::fmin(transverse_magnitude, 1.5707963F));
                        const auto transverse_scale =
                            transverse_magnitude > 0.0F
                                ? sycl::sin(sycl::fmin(transverse_magnitude, 1.5707963F)) /
                                      transverse_magnitude
                                : 1.0F;
                        const auto rotated = rotate_local_direction(
                            theta_x * transverse_scale, theta_y * transverse_scale,
                            local_direction_z,
                            Direction3F{direction_x, direction_y, direction_z});
                        direction_x = rotated.x;
                        direction_y = rotated.y;
                        direction_z = rotated.z;
                    }

                    position_x_mm += direction_x * step_mm;
                    position_y_mm += direction_y * step_mm;
                    position_z_mm += direction_z * step_mm;

                    if (absolute_direction_z >= 1.0e-6F) {
                        const auto boundary_z_mm =
                            direction_z < 0.0F
                                ? static_cast<float>(bin) * depth_bin_width_mm
                                : static_cast<float>(bin + 1) * depth_bin_width_mm;
                        if (sycl::fabs(boundary_z_mm - position_z_mm) <= 1.0e-5F) {
                            position_z_mm = sycl::nextafter(
                                boundary_z_mm,
                                direction_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        if (sycl::fabs(boundary_x_mm - position_x_mm) <= 1.0e-5F) {
                            position_x_mm = sycl::nextafter(
                                boundary_x_mm,
                                direction_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_y >= 1.0e-6F) {
                        const auto boundary_y_mm =
                            voxel_min_y_mm +
                            static_cast<float>(voxel_y + (direction_y > 0.0F ? 1 : 0)) *
                                voxel_size_y_mm;
                        if (sycl::fabs(boundary_y_mm - position_y_mm) <= 1.0e-5F) {
                            position_y_mm = sycl::nextafter(
                                boundary_y_mm,
                                direction_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

                    energy_MeV -= deposited_MeV;

                    if (enable_nuclear_elastic && elastic_this_step &&
                        energy_MeV > energy_cutoff_MeV) {
                        const float u_cos = rng::uniform01(
                            spot_seed, rng_history, steps, 10);
                        const float u_phi = rng::uniform01(
                            spot_seed, rng_history, steps, 11);
                        const auto scat = carbon::sample_c12_hydrogen_elastic(
                            energy_MeV, direction_x, direction_y, direction_z,
                            u_cos, u_phi);
                        energy_MeV = scat.projectile_ke_MeV;
                        direction_x = scat.proj_dir_x;
                        direction_y = scat.proj_dir_y;
                        direction_z = scat.proj_dir_z;
                        if (enable_secondary_transport &&
                            scat.proton_ke_MeV > energy_cutoff_MeV &&
                            secondary_queue_device != nullptr) {
                            auto count_ref = sycl::atomic_ref<
                                uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                *secondary_count_device);
                            const auto base_idx = count_ref.fetch_add(1U);
                            if (base_idx < max_secondaries) {
                                SecondaryParticle proton{};
                                proton.z = 1;
                                proton.a = 1;
                                proton.energy_MeV = scat.proton_ke_MeV;
                                proton.pos_x_mm = position_x_mm;
                                proton.pos_y_mm = position_y_mm;
                                proton.pos_z_mm = position_z_mm;
                                proton.dir_x = scat.proton_dir_x;
                                proton.dir_y = scat.proton_dir_y;
                                proton.dir_z = scat.proton_dir_z;
                                proton.weight = 1.0F;
                                proton.parent_history = rng_history;
                                proton.rng_stream = rng::child_stream(
                                    rng_history, rng::branch_tag(
                                        rng::branch_role_primary_charged, steps));
                                secondary_queue_device[base_idx] = proton;
                                cinel02_record_queued_secondary_birth_device(
                                    cinel02_species_energy_device, proton.z, proton.a,
                                    proton.energy_MeV);
                            } else if (secondary_overflow_count_device != nullptr) {
                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_ov(*secondary_overflow_count_device);
                                atomic_ov.fetch_add(1U);
                            }
                        }
                    }

                    if (enable_inelastic && inelastic_this_step &&
                        energy_MeV > energy_cutoff_MeV &&
                        (use_cinel02 || cross_section_device != nullptr)) {
                        if (use_cinel02) {
                            constexpr float cinel02_energy_tolerance_MeV_per_u = 0.51F;
                            const auto event_index = cinel02_find_event_device(
                                cinel02_energy_nodes_device, cinel02_energy_node_count,
                                cinel02_event_offsets_device, cinel02_event_indices_device,
                                cinel02_interactions_device, cinel02_interaction_count,
                                primary_atomic_number, primary_mass_number,
                                cinel02_target_z_step, cinel02_target_a_step,
                                energy_MeV * inverse_mass_number,
                                cinel02_energy_tolerance_MeV_per_u,
                                rng::uniform01(spot_seed, rng_history, steps, 13));
                            const auto replay_runtime_energy_MeV = sycl::fmax(0.0F, energy_MeV);
                            cinel02_record_replay_status_device(
                                cinel02_replay_status_counts_device,
                                cinel02_replay_status_incident_device,
                                cinel02_replay_status_delta_device,
                                cinel02_replay_status_abs_delta_device,
                                primary_atomic_number, primary_mass_number,
                                cinel02_target_z_step, 0U,
                                replay_runtime_energy_MeV, 0.0F,
                                static_cast<std::uint32_t>(
                                    Cinel02ReplayLedgerSchema::collision_candidate));
                            if (event_index != std::numeric_limits<std::uint32_t>::max()) {
                                cinel02_diag_increment_device(cinel02_diag_device, 3U);
                                const auto event = cinel02_interactions_device[event_index];
                                const auto product_end =
                                    static_cast<std::uint64_t>(event.product_offset) +
                                    event.product_count;
                                const auto replay_status =
                                    product_end <= cinel02_product_count &&
                                    (event.parent_status == 0 || event.parent_status == 2)
                                        ? Cinel02ReplayLedgerSchema::replay_valid
                                        : Cinel02ReplayLedgerSchema::replay_invalid_event;
                                cinel02_record_replay_status_device(
                                    cinel02_replay_status_counts_device,
                                    cinel02_replay_status_incident_device,
                                    cinel02_replay_status_delta_device,
                                    cinel02_replay_status_abs_delta_device,
                                    primary_atomic_number, primary_mass_number,
                                    cinel02_target_z_step, 0U,
                                    replay_runtime_energy_MeV,
                                    event.incident_energy_MeV_per_u -
                                        replay_runtime_energy_MeV * inverse_mass_number,
                                    static_cast<std::uint32_t>(replay_status));
                                if (product_end <= cinel02_product_count &&
                                    (event.parent_status == 0 || event.parent_status == 2)) {
                                    cinel02_diag_increment_device(
                                        cinel02_diag_device,
                                        event.parent_status == 0 ? 8U : 9U);
                                    cinel02_record_replay_delta_device(
                                        cinel02_replay_delta_device,
                                        cinel02_replay_abs_delta_device,
                                        cinel02_replay_delta_positive_device,
                                        cinel02_replay_delta_negative_device,
                                        cinel02_replay_valid_device,
                                        carbon::get_charged_species_idx(
                                            primary_atomic_number, primary_mass_number),
                                        event.incident_energy_MeV_per_u -
                                            sycl::fmax(0.0F,
                                                energy_MeV * inverse_mass_number));
                                    const auto local_deposit = sycl::fmax(
                                        0.0F, event.process_local_deposit_MeV);
                                    const auto parent_after = event.parent_status == 0
                                        ? sycl::fmax(0.0F, event.parent_energy_MeV)
                                        : 0.0F;
                                    const auto reaction_handoff_delta = energy_MeV -
                                        parent_after - local_deposit;
                                    cinel02_record_parent_outcome_device(
                                        cinel02_parent_outcome_counts_device,
                                        cinel02_parent_outcome_incident_device,
                                        cinel02_parent_outcome_after_device,
                                        cinel02_parent_outcome_local_device,
                                        cinel02_parent_outcome_export_device,
                                        cinel02_parent_outcome_import_device,
                                        primary_atomic_number, primary_mass_number,
                                        cinel02_target_z_step, 0U, event.parent_status,
                                        energy_MeV, parent_after, local_deposit,
                                        sycl::fmax(0.0F, reaction_handoff_delta),
                                        sycl::fmax(0.0F, -reaction_handoff_delta));
                                    cinel02_energy_add_device(cinel02_energy_device, 0U, energy_MeV);
                                    cinel02_energy_add_device(cinel02_energy_device, 1U, deposited_MeV);
                                    cinel02_energy_add_device(cinel02_energy_device, 2U, local_deposit);
                                    cinel02_energy_add_device(cinel02_energy_device, 3U,
                                                              event.nonionizing_deposit_MeV);
                                    cinel02_energy_add_device(cinel02_energy_device, 4U,
                                                              event.parent_energy_MeV);
                                    pending_primary_depth_MeV += local_deposit;
                                    if (enable_voxel_scoring && voxel_index >= 0) {
                                        pending_primary_voxel_MeV += local_deposit;
                                    }
                                    history_deposited_MeV += local_deposit;
                                    float untracked_MeV = 0.0F;
                                    if (primary_atomic_number == 6 &&
                                        primary_mass_number == 12 &&
                                        cinel02_target_z_step == 8 &&
                                        cinel02_target_a_step == 16) {
                                        const auto parent_energy_MeV_per_u = sycl::fmax(
                                            0.0F, energy_MeV * inverse_mass_number);
                                        const auto parent_energy_bin = sycl::min(
                                            static_cast<std::uint32_t>(
                                                parent_energy_MeV_per_u / 50.0F),
                                            7U);
                                        cinel02_diag_add_device(
                                            cinel02_diag_device, 1596U + parent_energy_bin,
                                            static_cast<std::uint64_t>(
                                                parent_energy_MeV_per_u * 1000.0F + 0.5F));
                                    }
                                    for (std::uint32_t ip = 0; ip < event.product_count; ++ip) {
                                        const auto product =
                                            cinel02_products_device[event.product_offset + ip];
                                        cinel02_energy_add_device(
                                            cinel02_energy_device,
                                            product.role == 2
                                                ? 7U
                                                : ((product.role == 0 && product.z > 0 &&
                                                    carbon::get_charged_species_idx(
                                                        product.z, product.a) >= 0)
                                                       ? 5U
                                                       : 6U),
                                            product.kinetic_energy_MeV);
                                        cinel02_diag_increment_device(
                                            cinel02_diag_device,
                                            product.role <= 2 ? 10U + product.role : 13U);
                                        if (product.role == 0 && product.z >= 1 && product.z <= 6) {
                                            cinel02_diag_increment_device(
                                                cinel02_diag_device,
                                                22U + static_cast<std::uint32_t>(product.z - 1));
                                        }
                                        if (product.role == 0) {
                                            cinel02_record_transition_device(
                                                cinel02_generated_transition_counts_device,
                                                cinel02_generated_transition_energy_device,
                                                primary_atomic_number, primary_mass_number,
                                                product.z, product.a,
                                                sycl::fmax(0.0F, product.kinetic_energy_MeV));
                                        }
                                        if (product.role == 0 && product.z == 4) {
                                            const auto birth_slot =
                                                cinel02_be_isotope_birth_diag_slot_device(
                                                    primary_atomic_number, cinel02_target_z_step,
                                                    0U, product.a);
                                            if (birth_slot !=
                                                std::numeric_limits<std::uint32_t>::max()) {
                                                cinel02_diag_increment_device(
                                                    cinel02_diag_device, birth_slot);
                                                cinel02_diag_add_device(
                                                    cinel02_diag_device, birth_slot + 48U,
                                                    static_cast<std::uint64_t>(sycl::fmax(
                                                        0.0F, product.kinetic_energy_MeV) *
                                                        1000.0F + 0.5F));
                                                cinel02_diag_add_device(
                                                    cinel02_diag_device, birth_slot + 96U,
                                                    static_cast<std::uint64_t>(sycl::fmax(
                                                        0.0F, position_z_mm) * 1000.0F + 0.5F));
                                                cinel02_diag_add_device(
                                                    cinel02_diag_device, birth_slot + 144U,
                                                    static_cast<std::uint64_t>((sycl::clamp(
                                                        product.local_direction_z, -1.0F, 1.0F) +
                                                        1.0F) * 1000000.0F + 0.5F));
                                            }
                                            if (product.a == 10 && cinel02_target_z_step == 8) {
                                                const auto parent_energy_bin = sycl::min(
                                                    static_cast<std::uint32_t>(sycl::fmax(
                                                        0.0F, energy_MeV * inverse_mass_number) /
                                                        50.0F), 7U);
                                                cinel02_diag_increment_device(
                                                    cinel02_diag_device, 1580U + parent_energy_bin);
                                                cinel02_diag_add_device(
                                                    cinel02_diag_device, 1588U + parent_energy_bin,
                                                    static_cast<std::uint64_t>(sycl::fmax(
                                                        0.0F, product.kinetic_energy_MeV) *
                                                        1000.0F + 0.5F));
                                            }
                                            if (cinel02_target_z_step == 8 &&
                                                (product.a == 6 || product.a == 7 ||
                                                 product.a == 9 || product.a == 10)) {
                                                const auto isotope_bin =
                                                    product.a == 6 ? 0U :
                                                    (product.a == 7 ? 1U :
                                                     (product.a == 9 ? 2U : 3U));
                                                const auto parent_energy_bin = sycl::min(
                                                    static_cast<std::uint32_t>(sycl::fmax(
                                                        0.0F, energy_MeV * inverse_mass_number) /
                                                        50.0F), 7U);
                                                const auto isotope_energy_slot =
                                                    isotope_bin * 8U + parent_energy_bin;
                                                cinel02_diag_increment_device(
                                                    cinel02_diag_device,
                                                    1604U + isotope_energy_slot);
                                                cinel02_diag_add_device(
                                                    cinel02_diag_device,
                                                    1636U + isotope_energy_slot,
                                                    static_cast<std::uint64_t>(sycl::fmax(
                                                        0.0F, product.kinetic_energy_MeV) *
                                                        1000.0F + 0.5F));
                                            }
                                        }
                                        if (product.role == 1) continue;
                                        if (product.role != 0 || product.z <= 0 || product.a <= 0) {
                                            untracked_MeV += sycl::fmax(
                                                0.0F, product.kinetic_energy_MeV);
                                            continue;
                                        }
                                        const auto child_direction = rotate_local_direction(
                                            product.local_direction_x,
                                            product.local_direction_y,
                                            product.local_direction_z,
                                            Direction3F{direction_x, direction_y, direction_z});
                                        if (secondary_queue_device != nullptr) {
                                            auto count_ref = sycl::atomic_ref<
                                                uint32_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>(
                                                *secondary_count_device);
                                            const auto output = count_ref.fetch_add(1U);
                                            if (output < max_secondaries) {
                                                SecondaryParticle child{};
                                                child.z = product.z;
                                                child.a = product.a;
                                                child.energy_MeV = product.kinetic_energy_MeV;
                                                child.pos_x_mm = position_x_mm;
                                                child.pos_y_mm = position_y_mm;
                                                child.pos_z_mm = position_z_mm;
                                                child.dir_x = child_direction.x;
                                                child.dir_y = child_direction.y;
                                                child.dir_z = child_direction.z;
                                                child.weight = product.weight;
                                                child.parent_history = rng_history;
                                                child.rng_stream = rng::child_stream(
                                                    rng_history, rng::branch_tag(
                                                        rng::branch_role_primary_charged, ip));
                                                if (product.z >= 1 && product.z <= 6) {
                                                    cinel02_diag_increment_device(
                                                        cinel02_diag_device,
                                                        28U + static_cast<std::uint32_t>(product.z - 1));
                                                }
                                                secondary_queue_device[output] = child;
                                                cinel02_record_queued_secondary_birth_device(
                                                    cinel02_species_energy_device, product.z, product.a,
                                                    sycl::fmax(0.0F, product.kinetic_energy_MeV));
                                                cinel02_record_transition_device(
                                                    cinel02_queued_transition_counts_device,
                                                    cinel02_queued_transition_energy_device,
                                                    primary_atomic_number, primary_mass_number,
                                                    product.z, product.a,
                                                    sycl::fmax(0.0F, product.kinetic_energy_MeV));
                                            } else {
                                                untracked_MeV += sycl::fmax(
                                                    0.0F, product.kinetic_energy_MeV);
                                                if (secondary_overflow_count_device != nullptr) {
                                                    sycl::atomic_ref<
                                                        uint32_t, sycl::memory_order::relaxed,
                                                        sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>
                                                        atomic_overflow(
                                                            *secondary_overflow_count_device);
                                                    atomic_overflow.fetch_add(1U);
                                                }
                                                if (secondary_overflow_energy_device != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_overflow_energy(
                                                            *secondary_overflow_energy_device);
                                                    atomic_overflow_energy.fetch_add(sycl::fmax(
                                                        0.0F, product.kinetic_energy_MeV));
                                                }
                                            }
                                        }
                                    }
                                    if (untracked_MeV > 0.0F &&
                                        untracked_nuclear_device != nullptr) {
                                        sycl::atomic_ref<
                                            float, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            atomic_untracked(
                                                untracked_nuclear_device[global_history]);
                                        atomic_untracked.fetch_add(untracked_MeV);
                                    }
                                    if (event.parent_status == 0) {
                                        energy_MeV = sycl::fmax(0.0F, event.parent_energy_MeV);
                                        const auto parent_direction = rotate_local_direction(
                                            event.parent_local_direction_x,
                                            event.parent_local_direction_y,
                                            event.parent_local_direction_z,
                                            Direction3F{direction_x, direction_y, direction_z});
                                        direction_x = parent_direction.x;
                                        direction_y = parent_direction.y;
                                        direction_z = parent_direction.z;
                                    } else {
                                        energy_MeV = 0.0F;
                                    }
                                }
                                else {
                                    cinel02_diag_increment_device(cinel02_diag_device, 5U);
                                    cinel02_record_replay_status_device(
                                        cinel02_replay_status_counts_device,
                                        cinel02_replay_status_incident_device,
                                        cinel02_replay_status_delta_device,
                                        cinel02_replay_status_abs_delta_device,
                                        primary_atomic_number, primary_mass_number,
                                        cinel02_target_z_step, 0U,
                                        replay_runtime_energy_MeV, 0.0F,
                                        static_cast<std::uint32_t>(
                                            Cinel02ReplayLedgerSchema::replay_no_event));
                                }
                            }
                            else {
                                cinel02_diag_increment_device(cinel02_diag_device, 4U);
                            }
                        } else {
                        const float p_target_h = p_target_h_step;
                        {
                            const Direction3F cur_dir{direction_x, direction_y,
                                                      direction_z};
                            const auto products =
                                sample_carbon_inelastic_products_device(
                                    energy_MeV, position_x_mm, position_y_mm,
                                    position_z_mm, cur_dir, spot_seed, rng_history, steps,
                                    fred_prob_proj_h_device, fred_prob_proj_o_device,
                                    fred_prob_tgt_h_device, fred_prob_tgt_o_device,
                                    ion_species_sp_device,
                                    static_cast<int>(table_size),
                                    minimum_table_energy, inverse_table_step,
                                    ion_energy_grid_device, ion_csda_a1_device,
                                    energy_cutoff_MeV, p_target_h,
                                    lib_h_view, lib_o_view);

                            if (products.local_deposit_MeV > 0.0F) {
                                pending_primary_depth_MeV +=
                                    products.local_deposit_MeV;
                                if (enable_voxel_scoring && voxel_index >= 0) {
                                    pending_primary_voxel_MeV +=
                                        products.local_deposit_MeV;
                                    if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                                        pending_primary_bin < static_cast<int>(number_of_bins)) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                                        atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(products.local_deposit_MeV));
                                    }
                                }
                                history_deposited_MeV +=
                                    products.local_deposit_MeV;
                            }

                            float total_charged_MeV = 0.0F;
                            for (uint8_t ip = 0; ip < products.count; ++ip) {
                                total_charged_MeV += products.products[ip].energy_MeV;
                            }
                            float untracked_MeV = products.untracked_energy_MeV;
                            const float residual_MeV = inelastic_numerical_residual_MeV(
                                energy_MeV, total_charged_MeV, products.local_deposit_MeV,
                                untracked_MeV, products.model_unassigned_MeV);
                            if (products.model_unassigned_MeV > 0.0F &&
                                fred_model_residual_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_un(*fred_model_residual_device);
                                atomic_un.fetch_add(products.model_unassigned_MeV);
                            }
                            if (residual_MeV > 0.0F && products.resample_failed == 0 &&
                                products.model_unassigned_MeV <= 0.0F &&
                                fred_model_residual_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_res(*fred_model_residual_device);
                                atomic_res.fetch_add(residual_MeV);
                            }

                            if (fred_q_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_q(*fred_q_device);
                                atomic_q.fetch_add(products.q_MeV);
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_nke(*fred_neutron_device);
                                atomic_nke.fetch_add(products.neutron_ke_MeV);
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_rem(*fred_remnant_device);
                                atomic_rem.fetch_add(products.remnant_local_MeV);
                            }
                            if (products.product_capacity_overflow != 0 &&
                                fred_cap_overflow_count_device != nullptr) {
                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_cap(*fred_cap_overflow_count_device);
                                atomic_cap.fetch_add(1U);
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_cape(*fred_cap_overflow_energy_device);
                                atomic_cape.fetch_add(products.product_capacity_overflow_MeV);
                            }
                            if (products.resample_failed != 0 && fred_fail_count_device != nullptr) {
                                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fail(*fred_fail_count_device);
                                atomic_fail.fetch_add(1U);
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fe(*fred_fail_energy_device);
                                atomic_fe.fetch_add(products.untracked_energy_MeV);
                            }

                            if (fred_diag_device != nullptr) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_n(fred_diag_device[18]);
                                atomic_n.fetch_add(1U);
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_retry(fred_diag_device[19]);
                                atomic_retry.fetch_add(static_cast<std::uint64_t>(products.retries_used));
                                if (products.energy_scaled != 0) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_sc(fred_diag_device[20]);
                                    atomic_sc.fetch_add(1U);
                                }
                                if (products.leftover_proj_a != 0 || products.leftover_proj_z != 0) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_open(fred_diag_device[21]);
                                    atomic_open.fetch_add(1U);
                                }
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_ta(fred_diag_device[22]);
                                atomic_ta.fetch_add(static_cast<std::uint64_t>(
                                    sycl::max(static_cast<int>(products.leftover_tgt_a), 0)));
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_tz(fred_diag_device[23]);
                                atomic_tz.fetch_add(static_cast<std::uint64_t>(
                                    sycl::max(static_cast<int>(products.leftover_tgt_z), 0)));
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_pa(fred_diag_device[24]);
                                atomic_pa.fetch_add(static_cast<std::uint64_t>(
                                    sycl::max(static_cast<int>(products.leftover_proj_a), 0)));
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_pz(fred_diag_device[25]);
                                atomic_pz.fetch_add(static_cast<std::uint64_t>(
                                    sycl::max(static_cast<int>(products.leftover_proj_z), 0)));
                                for (uint8_t s = 0; s < products.n_emitted && s < 12; ++s) {
                                    const auto idx = products.emitted_idx[s];
                                    if (idx < 18) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_iso(fred_diag_device[idx]);
                                        atomic_iso.fetch_add(1U);
                                    }
                                }
                            }

                            if (products.count > 0 &&
                                secondary_queue_device != nullptr) {
                                auto count_ref = sycl::atomic_ref<
                                    uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *secondary_count_device);
                                const auto base_idx =
                                    count_ref.fetch_add(products.count);
                                for (uint8_t ip = 0; ip < products.count; ++ip) {
                                    if (base_idx + ip < max_secondaries) {
                                        auto queued_product = products.products[ip];
                                        queued_product.rng_stream = rng::child_stream(
                                            rng_history, rng::branch_tag(
                                                rng::branch_role_primary_charged, ip));
                                        secondary_queue_device[base_idx + ip] = queued_product;
                                    } else if (secondary_overflow_count_device != nullptr) {
                                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_ov(*secondary_overflow_count_device);
                                        atomic_ov.fetch_add(1U);
                                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_ove(*secondary_overflow_energy_device);
                                        atomic_ove.fetch_add(products.products[ip].energy_MeV);
                                        untracked_MeV += products.products[ip].energy_MeV;
                                    }
                                }
                            }

                            if (untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_untracked(untracked_nuclear_device[global_history]);
                                atomic_untracked.fetch_add(untracked_MeV);
                            }

                            energy_MeV = 0.0F;
                            ++steps;
                            break;
                        }
                        }
                    }

                    ++steps;
                }

                if (energy_MeV > 0.0F && energy_MeV <= energy_cutoff_MeV &&
                    position_z_mm >= 0.0F && position_z_mm < phantom_length_mm) {
                    const auto cutoff_energy_MeV = energy_MeV;
                    pending_primary_depth_MeV += cutoff_energy_MeV;
                    if (enable_voxel_scoring && pending_primary_voxel >= 0 &&
                        pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                        pending_primary_voxel_MeV += cutoff_energy_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(cutoff_energy_MeV));
                        }
                    }
                    history_deposited_MeV += cutoff_energy_MeV;
                    if (enable_let_scoring) {
                        const auto cutoff_numerator =
                            static_cast<double>(cutoff_energy_MeV) *
                            static_cast<double>(last_primary_stopping_power_MeV_per_mm) /
                            static_cast<double>(sycl::fmax(last_primary_density_g_per_cm3, 1.0e-6F));
                        const auto cutoff_denominator = static_cast<double>(cutoff_energy_MeV);
                        pending_let_numerator += cutoff_numerator;
                        pending_let_denominator += cutoff_denominator;
                        if (voxel_let_moments_device != nullptr) {
                            pending_voxel_let_numerator += cutoff_numerator;
                            pending_voxel_let_denominator += cutoff_denominator;
                        }
                    }
                    energy_MeV = 0.0F;
                }

                if (pending_primary_depth_MeV > 0.0) {
                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_depth_MeV));
                }
                if (enable_let_scoring) {
                    flush_letd_moments_device(
                        let_moments_device, number_of_bins,
                        static_cast<std::size_t>(pending_primary_bin), nullptr, 0, 0,
                        pending_let_numerator, pending_let_denominator, true);
                }
                if (enable_voxel_scoring && pending_primary_voxel >= 0 &&
                    pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                    if (pending_primary_voxel_MeV > 0.0) {
                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_voxel_dose(voxel_dose_device[pending_primary_voxel]);
                        atomic_voxel_dose.fetch_add(static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        if (enable_charged_origin_voxel_scoring) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_primary_origin(
                                    charged_origin_voxel_dose_device[pending_primary_voxel]);
                            atomic_primary_origin.fetch_add(
                                static_cast<DoseAtomicT>(pending_primary_voxel_MeV));
                        }
                    }
                    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
                        flush_letd_moments_device(
                            voxel_let_moments_device, number_of_voxels,
                            pending_primary_voxel, nullptr, 0, 0,
                            pending_voxel_let_numerator, pending_voxel_let_denominator, true);
                    }
                }

                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV;
                steps_device[global_history] = steps;
            });
        kernel_event.wait_and_throw();
        primary_kernel_seconds += event_duration_seconds(kernel_event);
    }

    double secondary_kernel_seconds = 0.0;
    std::vector<SecondaryParticle> birth_secondaries_host;
    if (enable_inelastic && enable_secondary_transport &&
        secondary_count_device != nullptr && secondary_queue_device != nullptr) {
        uint32_t secondary_count_host = 0;
        queue.copy(secondary_count_device, &secondary_count_host, 1).wait_and_throw();
        if (secondary_count_host > max_secondaries) {
            secondary_count_host = static_cast<uint32_t>(max_secondaries);
        }
        if (secondary_count_host > 0) {
            std::uint32_t generation_begin = 0U;
            std::uint32_t generation_end = secondary_count_host;
            while (generation_begin < generation_end) {
            auto sec_event = queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<class CarbonSecondaryTransportKernel>(
                    sycl::range<1>(generation_end - generation_begin),
                    [=](sycl::id<1> item_id) {
                        const auto sec_idx = generation_begin + item_id[0];
                        const auto frag = secondary_queue_device[sec_idx];
                        if (frag.z <= 0 || frag.a <= 0) return;
                        const auto charged_origin_category =
                            charged_origin_category_from_fragment(
                                charged_dose_category(frag.z, frag.a));
                        const auto charged_origin_voxel_offset =
                            charged_origin_category * number_of_voxels;
                        const auto be_isotope_category =
                            be_isotope_origin_category(frag.z, frag.a);
                        if (frag.energy_MeV <= energy_cutoff_MeV) {
                            const auto ledger_species_idx = carbon::get_charged_species_idx(
                                static_cast<int>(frag.z), static_cast<int>(frag.a));
                            cinel02_species_energy_add_device(
                                cinel02_species_energy_device, ledger_species_idx, 5U,
                                frag.energy_MeV);
                            cinel02_species_terminal_increment_device(
                                cinel02_species_terminal_device, ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    Cinel02SpeciesLedgerSchema::initial_below_cutoff));
                            const auto bin_z = static_cast<int>(frag.pos_z_mm * inverse_depth_bin_width_mm);
                            if (bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dose(dose_device[bin_z]);
                                atomic_dose.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                            }
                            if (enable_voxel_scoring) {
                                const auto bin_x = static_cast<int>((frag.pos_x_mm - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto bin_y = static_cast<int>((frag.pos_y_mm - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                if (bin_x >= 0 && bin_x < static_cast<int>(voxel_bins_x) &&
                                    bin_y >= 0 && bin_y < static_cast<int>(voxel_bins_y) &&
                                    bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                    const auto cur_voxel = (bin_z * static_cast<int>(voxel_bins_y) + bin_y) * static_cast<int>(voxel_bins_x) + bin_x;
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_vox(voxel_dose_device[cur_voxel]);
                                    atomic_vox.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                    if (enable_charged_origin_voxel_scoring) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(charged_origin_voxel_dose_device[
                                                charged_origin_voxel_offset + cur_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            be_isotope_origin_voxel_dose_device,
                                            be_isotope_category, number_of_voxels, cur_voxel,
                                            static_cast<DoseAtomicT>(frag.energy_MeV));
                                    }
                                    if (in_fov_dose_device != nullptr) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[bin_z]);
                                        atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(frag.energy_MeV));
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx,
                                            6U, frag.energy_MeV);
                                    }
                                }
                            }
                            if (deposited_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(deposited_device[frag.parent_history]);
                                atomic_dep.fetch_add(frag.energy_MeV);
                            }
                            return;
                        }

                        const auto frag_a = static_cast<float>(frag.a);
                        const auto frag_inv_a = 1.0F / frag_a;
                        const auto charged_sp_idx = carbon::get_charged_species_idx(static_cast<int>(frag.z), static_cast<int>(frag.a));
                        const auto ledger_species_idx = charged_sp_idx;
                        if (charged_sp_idx < 0) {
                            cinel02_diag_increment_device(cinel02_diag_device, 34U);
                            if (untracked_nuclear_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_untracked(
                                        untracked_nuclear_device[frag.parent_history]);
                                atomic_untracked.fetch_add(sycl::fmax(0.0F, frag.energy_MeV));
                            }
                            return;
                        }
                        const float* ion_sp_table =
                            &ion_species_sp_device[
                                static_cast<std::size_t>(charged_sp_idx) * table_size];

                        bool sec_terminal_recorded = false;
                        float sec_e = frag.energy_MeV;
                        float sec_x = frag.pos_x_mm;
                        float sec_y = frag.pos_y_mm;
                        float sec_z = frag.pos_z_mm;
                        float sec_dx = frag.dir_x;
                        float sec_dy = frag.dir_y;
                        float sec_dz = frag.dir_z;

                        int pending_sec_bin = -1;
                        int pending_sec_voxel = -1;
                        float pending_sec_depth_MeV = 0.0F;
                        float pending_sec_voxel_MeV = 0.0F;

                        uint32_t sec_steps = 0;
                        constexpr uint32_t kSecondaryMaxSteps = 30000U;
                        while (sec_e > energy_cutoff_MeV && sec_z >= 0.0F && sec_z < phantom_length_mm &&
                               sec_steps < kSecondaryMaxSteps) {
                            const auto bin_z = static_cast<int>(sec_z * inverse_depth_bin_width_mm);

                            if (bin_z < 0 || bin_z >= static_cast<int>(number_of_bins)) break;

                            const auto sec_e_u = sec_e * frag_inv_a;
                            const auto flt_idx = (sec_e_u - minimum_table_energy) * inverse_table_step;
                            auto sp_idx = static_cast<int>(sycl::floor(flt_idx));
                            sp_idx = sycl::max(0, sycl::min(sp_idx, static_cast<int>(table_size) - 2));
                            const auto sp_frac = sycl::clamp(flt_idx - static_cast<float>(sp_idx), 0.0F, 1.0F);
                            const auto sec_sp = (ion_sp_table[sp_idx] + sp_frac * (ion_sp_table[sp_idx + 1] - ion_sp_table[sp_idx]));

                            if (sec_sp <= 1.0e-6F) break;

                            float sec_step_mm = maximum_step_mm;
                            if (sec_dz > 1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z + 1) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 1.0e-5F) sec_step_mm = sycl::fmin(sec_step_mm, dz_step);
                            } else if (sec_dz < -1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 1.0e-5F) sec_step_mm = sycl::fmin(sec_step_mm, dz_step);
                            }
                            sec_step_mm = sycl::fmax(sec_step_mm, 1.0e-5F);
                            bool secondary_inelastic = false;
                            std::int16_t secondary_target_z = 0;
                            std::int16_t secondary_target_a = 0;
                            if (use_cinel02 && frag.generation <
                                cinel02_max_secondary_inelastic_generations) {
                                cinel02_diag_increment_device(cinel02_diag_device, 14U);
                                const auto target = cinel02_select_water_target_device(
                                    cinel02_rate_groups_device, cinel02_rate_group_count,
                                    cinel02_rate_samples_device, cinel02_rate_sample_count,
                                    frag.z, frag.a, sec_e_u,
                                    water_density_g_per_cm3, water_density_g_per_cm3,
                                    rng::uniform01(2026, frag.rng_stream,
                                                   sec_steps, 12));
                                if (target.covered) {
                                    cinel02_diag_increment_device(cinel02_diag_device, 15U);
                                    float collision_distance = sec_step_mm;
                                    secondary_inelastic = inelastic_collision_in_step(
                                        target.total_rate_per_mm, sec_step_mm,
                                        rng::uniform01(2026, frag.rng_stream,
                                                       sec_steps, 13),
                                        &collision_distance);
                                    if (secondary_inelastic) {
                                        cinel02_diag_increment_device(cinel02_diag_device, 16U);
                                        cinel02_diag_increment_device(
                                            cinel02_diag_device,
                                            target.target_z == 1 ? 20U : 21U);
                                        cinel02_diag_increment_device(
                                            cinel02_diag_device,
                                            cinel02_hazard_diag_slot_device(
                                                frag.z, frag.a, target.target_z,
                                                static_cast<std::uint32_t>(frag.generation) + 1U,
                                                sec_e_u));
                                        sec_step_mm = collision_distance;
                                        secondary_target_z = target.target_z;
                                        secondary_target_a = target.target_a;
                                    }
                                }
                            }

                            // Midpoint loss
                            const auto mid_e_u = sycl::fmax(0.01F, (sec_e - 0.5F * sec_sp * sec_step_mm) * frag_inv_a);
                            const auto mid_flt = (mid_e_u - minimum_table_energy) * inverse_table_step;
                            auto mid_idx = static_cast<int>(sycl::floor(mid_flt));
                            mid_idx = sycl::max(0, sycl::min(mid_idx, static_cast<int>(table_size) - 2));
                            const auto mid_fr = sycl::clamp(mid_flt - static_cast<float>(mid_idx), 0.0F, 1.0F);
                            const auto mid_sp = (ion_sp_table[mid_idx] + mid_fr * (ion_sp_table[mid_idx + 1] - ion_sp_table[mid_idx]));

                            auto dE = sycl::fmin(mid_sp * sec_step_mm, sec_e);
                            if (enable_secondary_energy_straggling &&
                                use_packaged_fluctuation && frag.z == 6 && frag.a == 12) {
                                const auto u_loss = rng::uniform01(
                                    2026, frag.rng_stream, sec_steps, 2);
                                const auto ratio = sample_energy_loss_ratio_from_grid(
                                    fluct_energy_device, fluct_energy_count,
                                    fluct_density_device, fluct_density_count,
                                    fluct_probability_device, fluct_probability_count,
                                    fluct_quantile_device, sec_e_u,
                                    water_density_g_per_cm3 * sec_step_mm / 10.0F, u_loss);
                                const auto local_scale = interpolate_straggling_scale(
                                    sec_e_u, straggling_scale_energies,
                                    straggling_scale_values,
                                    straggling_scale_point_count, straggling_scale);
                                const auto scaled_ratio =
                                    scale_energy_loss_ratio_preserving_mean(ratio, local_scale);
                                dE = sycl::clamp(dE * scaled_ratio, 0.0F, sec_e);
                            }

                            // CINEL02 packages are captured at hadronic PostStepDoIt:
                            // advance the continuous EM state to the collision point
                            // before selecting and replaying the final state.
                            const auto collision_input_dx = sec_dx;
                            const auto collision_input_dy = sec_dy;
                            const auto collision_input_dz = sec_dz;
                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);
                            const auto post_em_x = sec_x + collision_input_dx * sec_step_mm;
                            const auto post_em_y = sec_y + collision_input_dy * sec_step_mm;
                            const auto post_em_z = sec_z + collision_input_dz * sec_step_mm;
                            const auto collision_bin = sycl::max(
                                0, sycl::min(
                                       static_cast<int>(post_em_z *
                                           inverse_depth_bin_width_mm),
                                       static_cast<int>(number_of_bins) - 1));
                            if (secondary_inelastic) {
                                sec_e = post_em_e;
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                                // Record the hazard before attempting package
                                // replay so isotope/target/generation misses
                                // remain distinguishable from valid events.
                                cinel02_record_replay_status_device(
                                    cinel02_replay_status_counts_device,
                                    cinel02_replay_status_incident_device,
                                    cinel02_replay_status_delta_device,
                                    cinel02_replay_status_abs_delta_device,
                                    frag.z, frag.a, secondary_target_z,
                                    static_cast<std::uint32_t>(frag.generation) + 1U,
                                    sycl::fmax(0.0F, sec_e), 0.0F,
                                    static_cast<std::uint32_t>(
                                        Cinel02ReplayLedgerSchema::collision_candidate));
                            }

                            if (bin_z != pending_sec_bin) {
                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dose(dose_device[pending_sec_bin]);
                                    atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_sec_depth_MeV));
                                    pending_sec_depth_MeV = 0.0F;
                                }
                                pending_sec_bin = bin_z;
                            }
                            pending_sec_depth_MeV += dE;
                            cinel02_species_energy_add_device(
                                cinel02_species_energy_device, ledger_species_idx, 1U, dE);

                            if (enable_voxel_scoring || secondary_inelastic) {
                                int cur_voxel = -1;
                                const auto score_bin_x = static_cast<int>(
                                    (sec_x - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto score_bin_y = static_cast<int>(
                                    (sec_y - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                if (score_bin_x >= 0 && score_bin_x < static_cast<int>(voxel_bins_x) && score_bin_y >= 0 && score_bin_y < static_cast<int>(voxel_bins_y)) {
                                    cur_voxel = (collision_bin * static_cast<int>(voxel_bins_y) + score_bin_y) * static_cast<int>(voxel_bins_x) + score_bin_x;
                                }
                                if (cur_voxel != pending_sec_voxel || secondary_inelastic) {
                                    if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_vox(voxel_dose_device[pending_sec_voxel]);
                                        atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    if (enable_charged_origin_voxel_scoring) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(charged_origin_voxel_dose_device[
                                                charged_origin_voxel_offset + pending_sec_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            be_isotope_origin_voxel_dose_device,
                                            be_isotope_category, number_of_voxels,
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                    }
                                    pending_sec_voxel_MeV = 0.0F;
                                    pending_sec_voxel = cur_voxel;
                            if (secondary_inelastic && sec_e > energy_cutoff_MeV) {
                                const auto event_index = cinel02_find_event_device(
                                    cinel02_energy_nodes_device, cinel02_energy_node_count,
                                    cinel02_event_offsets_device, cinel02_event_indices_device,
                                    cinel02_interactions_device, cinel02_interaction_count,
                                    frag.z, frag.a, secondary_target_z, secondary_target_a,
                                    sec_e * frag_inv_a, 0.51F,
                                    rng::uniform01(2026, frag.rng_stream,
                                                   sec_steps, 14));
                                if (event_index !=
                                    std::numeric_limits<std::uint32_t>::max()) {
                                    cinel02_diag_increment_device(cinel02_diag_device, 17U);
                                    const auto event =
                                        cinel02_interactions_device[event_index];
                                    const auto product_end =
                                        static_cast<std::uint64_t>(event.product_offset) +
                                        event.product_count;
                                    const auto replay_status =
                                        product_end <= cinel02_product_count &&
                                        (event.parent_status == 0 ||
                                         event.parent_status == 2)
                                            ? Cinel02ReplayLedgerSchema::replay_valid
                                            : Cinel02ReplayLedgerSchema::replay_invalid_event;
                                    cinel02_record_replay_status_device(
                                        cinel02_replay_status_counts_device,
                                        cinel02_replay_status_incident_device,
                                        cinel02_replay_status_delta_device,
                                        cinel02_replay_status_abs_delta_device,
                                        frag.z, frag.a, secondary_target_z,
                                        static_cast<std::uint32_t>(frag.generation) + 1U,
                                        sycl::fmax(0.0F, sec_e),
                                        event.incident_energy_MeV_per_u -
                                            sycl::fmax(0.0F, sec_e * frag_inv_a),
                                        static_cast<std::uint32_t>(replay_status));
                                    if (product_end <= cinel02_product_count &&
                                        (event.parent_status == 0 ||
                                         event.parent_status == 2)) {
                                        cinel02_diag_increment_device(
                                            cinel02_diag_device,
                                            event.parent_status == 0 ? 8U : 9U);
                                        if (frag.z >= 1 && frag.z <= 6) {
                                            const auto incident_keV =
                                                static_cast<std::uint64_t>(sycl::fmax(
                                                    0.0F, sec_e) * 1000.0F + 0.5F);
                                            cinel02_diag_add_device(
                                                cinel02_diag_device,
                                                1120U + sycl::min<std::uint32_t>(
                                                    frag.generation, 1U) * 6U +
                                                    static_cast<std::uint32_t>(frag.z - 1),
                                                incident_keV);
                                        }
                                        if (frag.z == 4) {
                                            const auto be_incident_slot =
                                                cinel02_be_incident_diag_slot_device(
                                                    secondary_target_z, frag.generation,
                                                    sec_e * frag_inv_a);
                                            cinel02_diag_increment_device(
                                                cinel02_diag_device, be_incident_slot);
                                            cinel02_diag_add_device(
                                                cinel02_diag_device, be_incident_slot + 32U,
                                                static_cast<std::uint64_t>(sycl::fmax(
                                                    0.0F, sec_e) * 1000.0F + 0.5F));
                                        }
                                        const auto parent_outcome_slot =
                                            cinel02_parent_outcome_diag_slot_device(
                                                frag.z, event.parent_status,
                                                static_cast<std::uint32_t>(frag.generation));
                                        if (parent_outcome_slot !=
                                            std::numeric_limits<std::uint32_t>::max()) {
                                            cinel02_diag_increment_device(
                                                cinel02_diag_device, parent_outcome_slot);
                                        }
                                        const auto local_deposit = sycl::fmax(
                                            0.0F, event.process_local_deposit_MeV);
                                        const auto parent_after = event.parent_status == 0
                                            ? sycl::fmax(0.0F, event.parent_energy_MeV)
                                            : 0.0F;
                                        const auto reaction_handoff_delta = sec_e -
                                            parent_after - local_deposit;
                                        cinel02_record_parent_outcome_device(
                                            cinel02_parent_outcome_counts_device,
                                            cinel02_parent_outcome_incident_device,
                                            cinel02_parent_outcome_after_device,
                                            cinel02_parent_outcome_local_device,
                                            cinel02_parent_outcome_export_device,
                                            cinel02_parent_outcome_import_device,
                                            frag.z, frag.a, secondary_target_z,
                                            static_cast<std::uint32_t>(frag.generation) + 1U,
                                            event.parent_status, sec_e, parent_after,
                                            local_deposit,
                                            sycl::fmax(0.0F, reaction_handoff_delta),
                                            sycl::fmax(0.0F, -reaction_handoff_delta));
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::nuclear_local_deposit_all),
                                            local_deposit);
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::reaction_export_kinetic),
                                            sycl::fmax(0.0F, reaction_handoff_delta));
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx,
                                            static_cast<std::uint32_t>(
                                                Cinel02SpeciesLedgerSchema::reaction_import_kinetic),
                                            sycl::fmax(0.0F, -reaction_handoff_delta));
                                        cinel02_record_replay_delta_device(
                                            cinel02_replay_delta_device,
                                            cinel02_replay_abs_delta_device,
                                            cinel02_replay_delta_positive_device,
                                            cinel02_replay_delta_negative_device,
                                            cinel02_replay_valid_device,
                                            ledger_species_idx,
                                            event.incident_energy_MeV_per_u -
                                                sycl::fmax(0.0F, sec_e * frag_inv_a));
                                        cinel02_energy_add_device(cinel02_energy_device, 0U, sec_e);
                                        cinel02_energy_add_device(cinel02_energy_device, 1U, dE);
                                        cinel02_energy_add_device(cinel02_energy_device, 2U, local_deposit);
                                        cinel02_energy_add_device(cinel02_energy_device, 3U,
                                                                  event.nonionizing_deposit_MeV);
                                        cinel02_energy_add_device(cinel02_energy_device, 4U,
                                                                  event.parent_energy_MeV);
                                        if (local_deposit > 0.0F) {
                                            sycl::atomic_ref<
                                                DoseAtomicT, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                atomic_local_depth(dose_device[collision_bin]);
                                            atomic_local_depth.fetch_add(
                                                static_cast<DoseAtomicT>(local_deposit));
                                        }
                                        if (enable_voxel_scoring && pending_sec_voxel >= 0) {
                                            pending_sec_voxel_MeV += local_deposit;
                                            cinel02_species_energy_add_device(
                                                cinel02_species_energy_device, ledger_species_idx,
                                                4U, local_deposit);
                                        }
                                        if (deposited_device != nullptr) {
                                            sycl::atomic_ref<
                                                float, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                atomic_dep(
                                                    deposited_device[frag.parent_history]);
                                            atomic_dep.fetch_add(local_deposit);
                                        }
                                        float untracked_MeV = 0.0F;
                                        for (std::uint32_t ip = 0;
                                             ip < event.product_count; ++ip) {
                                            const auto product = cinel02_products_device[
                                                event.product_offset + ip];
                                            cinel02_energy_add_device(
                                                cinel02_energy_device,
                                                product.role == 2
                                                ? 7U
                                                : ((product.role == 0 && product.z > 0 &&
                                                    carbon::get_charged_species_idx(
                                                        product.z, product.a) >= 0)
                                                       ? 5U
                                                       : 6U),
                                                product.kinetic_energy_MeV);
                                            cinel02_diag_increment_device(
                                                cinel02_diag_device,
                                                product.role <= 2 ? 10U + product.role : 13U);
                                            if (product.role == 0 && product.z >= 1 && product.z <= 6) {
                                                cinel02_diag_increment_device(
                                                    cinel02_diag_device,
                                                    22U + static_cast<std::uint32_t>(product.z - 1));
                                            }
                                            if (product.role == 0 && product.z == 4) {
                                                const auto birth_slot =
                                                    cinel02_be_isotope_birth_diag_slot_device(
                                                        frag.z, secondary_target_z,
                                                        frag.generation + 1U, product.a);
                                                if (birth_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                    cinel02_diag_increment_device(
                                                        cinel02_diag_device, birth_slot);
                                                    cinel02_diag_add_device(
                                                        cinel02_diag_device, birth_slot + 48U,
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, product.kinetic_energy_MeV) *
                                                            1000.0F + 0.5F));
                                                    cinel02_diag_add_device(
                                                        cinel02_diag_device, birth_slot + 96U,
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, sec_z) * 1000.0F + 0.5F));
                                                    cinel02_diag_add_device(
                                                        cinel02_diag_device, birth_slot + 144U,
                                                        static_cast<std::uint64_t>((sycl::clamp(
                                                            product.local_direction_z, -1.0F, 1.0F) +
                                                            1.0F) * 1000000.0F + 0.5F));
                                                }
                                                const auto be_channel_slot =
                                                    cinel02_be_channel_diag_slot_device(
                                                        frag.z, secondary_target_z,
                                                        frag.generation, sec_e * frag_inv_a);
                                                if (be_channel_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                    cinel02_diag_increment_device(
                                                        cinel02_diag_device, be_channel_slot);
                                                    cinel02_diag_add_device(
                                                        cinel02_diag_device,
                                                        be_channel_slot + 96U,
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, product.kinetic_energy_MeV) *
                                                            1000.0F + 0.5F));
                                                }
                                            }
                                            if (product.role == 0) {
                                                cinel02_record_transition_device(
                                                    cinel02_generated_transition_counts_device,
                                                    cinel02_generated_transition_energy_device,
                                                    frag.z, frag.a, product.z, product.a,
                                                    sycl::fmax(0.0F,
                                                        product.kinetic_energy_MeV));
                                                const auto transition_slot =
                                                    cinel02_transition_diag_slot_device(
                                                        frag.z, product.z,
                                                        static_cast<std::uint32_t>(frag.generation));
                                                if (transition_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                    cinel02_diag_increment_device(
                                                        cinel02_diag_device, transition_slot);
                                                    const auto kinetic_keV =
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, product.kinetic_energy_MeV) *
                                                            1000.0F + 0.5F);
                                                    cinel02_diag_add_device(
                                                        cinel02_diag_device,
                                                        transition_slot + 108U, kinetic_keV);
                                                }
                                            }
                                            if (product.role != 0 || product.z <= 0 ||
                                                product.a <= 0 ||
                                                secondary_queue_device == nullptr) {
                                                untracked_MeV += sycl::fmax(
                                                    0.0F, product.kinetic_energy_MeV);
                                                continue;
                                            }
                                            const auto child_direction =
                                                rotate_local_direction(
                                                    product.local_direction_x,
                                                    product.local_direction_y,
                                                    product.local_direction_z,
                                                    Direction3F{collision_input_dx, collision_input_dy,
                                                                collision_input_dz});
                                            auto count_ref = sycl::atomic_ref<
                                                uint32_t, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>(
                                                *secondary_count_device);
                                            const auto output = count_ref.fetch_add(1U);
                                            if (output < max_secondaries) {
                                                SecondaryParticle child{};
                                                child.z = product.z;
                                                child.a = product.a;
                                                child.energy_MeV =
                                                    product.kinetic_energy_MeV;
                                                child.pos_x_mm = sec_x;
                                                child.pos_y_mm = sec_y;
                                                child.pos_z_mm = sec_z;
                                                child.dir_x = child_direction.x;
                                                child.dir_y = child_direction.y;
                                                child.dir_z = child_direction.z;
                                                child.weight = product.weight;
                                                child.parent_history = frag.parent_history;
                                                const auto event_stream = rng::child_stream(
                                                    frag.rng_stream,
                                                    event_index ^
                                                        (sec_steps * 0x9E3779B9U));
                                                child.rng_stream = rng::child_stream(
                                                    event_stream, rng::branch_tag(
                                                        rng::branch_role_cascade_charged, ip));
                                                child.generation =
                                                    static_cast<std::uint16_t>(
                                                        frag.generation + 1U);
                                                if (product.z >= 1 && product.z <= 6) {
                                                    cinel02_diag_increment_device(
                                                        cinel02_diag_device,
                                                        28U + static_cast<std::uint32_t>(product.z - 1));
                                                }
                                                secondary_queue_device[output] = child;
                                                cinel02_record_queued_secondary_birth_device(
                                                    cinel02_species_energy_device, product.z, product.a,
                                                    sycl::fmax(0.0F, product.kinetic_energy_MeV));
                                                cinel02_record_transition_device(
                                                    cinel02_queued_transition_counts_device,
                                                    cinel02_queued_transition_energy_device,
                                                    frag.z, frag.a, product.z, product.a,
                                                    sycl::fmax(0.0F,
                                                        product.kinetic_energy_MeV));
                                            } else if (secondary_overflow_count_device !=
                                                       nullptr) {
                                                sycl::atomic_ref<
                                                    uint32_t, sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>
                                                    atomic_overflow(
                                                        *secondary_overflow_count_device);
                                                atomic_overflow.fetch_add(1U);
                                                const auto overflow_energy = sycl::fmax(
                                                    0.0F, product.kinetic_energy_MeV);
                                                untracked_MeV += overflow_energy;
                                                if (secondary_overflow_energy_device != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_overflow_energy(
                                                            *secondary_overflow_energy_device);
                                                    atomic_overflow_energy.fetch_add(overflow_energy);
                                                }
                                            }
                                        }
                                        if (untracked_MeV > 0.0F &&
                                            untracked_nuclear_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(
                                                    untracked_nuclear_device[frag.parent_history]);
                                            atomic_untracked.fetch_add(untracked_MeV);
                                        }
                                        if (event.parent_status == 0) {
                                            sec_e = sycl::fmax(
                                                0.0F, event.parent_energy_MeV);
                                            const auto parent_direction =
                                                rotate_local_direction(
                                                    event.parent_local_direction_x,
                                                    event.parent_local_direction_y,
                                                    event.parent_local_direction_z,
                                                    Direction3F{collision_input_dx, collision_input_dy,
                                                                collision_input_dz});
                                            sec_dx = parent_direction.x;
                                            sec_dy = parent_direction.y;
                                            sec_dz = parent_direction.z;
                                        } else {
                                            cinel02_species_terminal_increment_device(
                                                cinel02_species_terminal_device, ledger_species_idx,
                                                static_cast<std::uint32_t>(
                                                    Cinel02SpeciesLedgerSchema::reaction_killed));
                                            sec_terminal_recorded = true;
                                            sec_e = 0.0F;
                                        }
                                    }
                                    else {
                                        cinel02_diag_increment_device(cinel02_diag_device, 19U);
                                        cinel02_record_replay_status_device(
                                            cinel02_replay_status_counts_device,
                                            cinel02_replay_status_incident_device,
                                            cinel02_replay_status_delta_device,
                                            cinel02_replay_status_abs_delta_device,
                                            frag.z, frag.a, secondary_target_z,
                                            static_cast<std::uint32_t>(frag.generation) + 1U,
                                            sycl::fmax(0.0F, sec_e), 0.0F,
                                            static_cast<std::uint32_t>(
                                                Cinel02ReplayLedgerSchema::replay_invalid_event));
                                    }
                                }
                                else {
                                    cinel02_diag_increment_device(cinel02_diag_device, 18U);
                                    cinel02_record_replay_status_device(
                                        cinel02_replay_status_counts_device,
                                        cinel02_replay_status_incident_device,
                                        cinel02_replay_status_delta_device,
                                        cinel02_replay_status_abs_delta_device,
                                        frag.z, frag.a, secondary_target_z,
                                        static_cast<std::uint32_t>(frag.generation) + 1U,
                                        sycl::fmax(0.0F, sec_e), 0.0F,
                                        static_cast<std::uint32_t>(
                                            Cinel02ReplayLedgerSchema::replay_no_event));
                                }
                            } else if (secondary_inelastic) {
                                // The post-EM collision energy is already at
                                // or below the transport cutoff, so no package
                                // event is eligible.  Keep the candidate in
                                // the status partition as no_event.
                                cinel02_record_replay_status_device(
                                    cinel02_replay_status_counts_device,
                                    cinel02_replay_status_incident_device,
                                    cinel02_replay_status_delta_device,
                                    cinel02_replay_status_abs_delta_device,
                                    frag.z, frag.a, secondary_target_z,
                                    static_cast<std::uint32_t>(frag.generation) + 1U,
                                    sycl::fmax(0.0F, sec_e), 0.0F,
                                    static_cast<std::uint32_t>(
                                        Cinel02ReplayLedgerSchema::replay_no_event));
                            }
                                    pending_sec_voxel = cur_voxel;
                                }
                                if (cur_voxel >= 0) {
                                    pending_sec_voxel_MeV += dE;
                                    if (in_fov_dose_device != nullptr && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[pending_sec_bin]);
                                        atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(dE));
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx,
                                            2U, dE);
                                    }
                                }
                            }

                            if (!secondary_inelastic) sec_e = post_em_e;
                            if (deposited_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(deposited_device[frag.parent_history]);
                                atomic_dep.fetch_add(dE);
                            }
                            if (!secondary_inelastic) {
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                            }

                            if (!secondary_inelastic && enable_multiple_scattering &&
                                sec_e > energy_cutoff_MeV) {
                                constexpr float two_pi = 6.2831853071795864769F;
                                float theta_scat = 0.0F;
                                float phi_scat = 0.0F;
                                if (use_fred_2gr_mcs) {
                                    const auto mixture = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 0);
                                    const auto radial = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 1);
                                    phi_scat = two_pi * rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 3);
                                    theta_scat = fred_2gr_angle_device(
                                        fred_2gr_mcs_device, sec_e / frag_a,
                                        static_cast<int>(frag.z), static_cast<int>(frag.a),
                                        water_density_g_per_cm3 * sec_step_mm / 10.0F,
                                        static_cast<float>(water_radiation_length_g_per_cm2),
                                        extrapolate_fred_2gr_high_energy,
                                        multiple_scattering_scale, mixture, radial);
                                } else {
                                    const auto theta_rms = highland_projected_rms_angle_device(
                                        sec_e, static_cast<int>(frag.z),
                                        static_cast<int>(frag.a), sec_step_mm,
                                        water_density_g_per_cm3) * multiple_scattering_scale;
                                    const auto u_msc0 = sycl::fmax(rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 0),
                                        1.0e-10F);
                                    theta_scat = theta_rms *
                                        sycl::sqrt(-2.0F * sycl::log(u_msc0));
                                    phi_scat = two_pi * rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 1);
                                }

                                const auto sin_scat = sycl::sin(theta_scat);
                                const auto cos_scat = sycl::cos(theta_scat);
                                const auto rotated = rotate_local_direction(
                                    sin_scat * sycl::cos(phi_scat),
                                    sin_scat * sycl::sin(phi_scat),
                                    cos_scat,
                                    Direction3F{collision_input_dx, collision_input_dy,
                                                                collision_input_dz});
                                sec_dx = rotated.x;
                                sec_dy = rotated.y;
                                sec_dz = rotated.z;
                            }

                            ++sec_steps;
                        }

                        const bool step_limited =
                            sec_e > energy_cutoff_MeV && sec_steps >= kSecondaryMaxSteps;
                        if (sec_e > 0.0F) {
                            if (step_limited) {
                                cinel02_species_energy_add_device(
                                    cinel02_species_energy_device, ledger_species_idx,  9U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    cinel02_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::step_limit));
                                sec_terminal_recorded = true;
                                if (escaped_device != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_esc(escaped_device[frag.parent_history]);
                                    atomic_esc.fetch_add(sec_e);
                                }
                            } else if (sec_z >= 0.0F && sec_z < phantom_length_mm) {
                                cinel02_species_energy_add_device(
                                    cinel02_species_energy_device, ledger_species_idx, 5U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    cinel02_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::terminal_deposit));
                                sec_terminal_recorded = true;
                                const auto bin_x = static_cast<int>((sec_x - voxel_min_x_mm) * inverse_voxel_size_x_mm);
                                const auto bin_y = static_cast<int>((sec_y - voxel_min_y_mm) * inverse_voxel_size_y_mm);
                                const auto bin_z = static_cast<int>(sec_z * inverse_depth_bin_width_mm);
                                if (bin_z >= 0 && bin_z < static_cast<int>(number_of_bins)) {
                                    if (bin_z != pending_sec_bin) {
                                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 &&
                                            pending_sec_bin < static_cast<int>(number_of_bins)) {
                                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dose(dose_device[pending_sec_bin]);
                                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_sec_depth_MeV));
                                            pending_sec_depth_MeV = 0.0F;
                                        }
                                        pending_sec_bin = bin_z;
                                    }
                                    pending_sec_depth_MeV += sec_e;
                                    if (enable_voxel_scoring) {
                                        int cur_voxel = -1;
                                        if (bin_x >= 0 && bin_x < static_cast<int>(voxel_bins_x) &&
                                            bin_y >= 0 && bin_y < static_cast<int>(voxel_bins_y)) {
                                            cur_voxel = (bin_z * static_cast<int>(voxel_bins_y) + bin_y) *
                                                            static_cast<int>(voxel_bins_x) +
                                                        bin_x;
                                        }
                                        if (cur_voxel != pending_sec_voxel) {
                                            if (pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 &&
                                                pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_vox(voxel_dose_device[pending_sec_voxel]);
                                                atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    if (enable_charged_origin_voxel_scoring) {
                                        sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_origin(charged_origin_voxel_dose_device[
                                                charged_origin_voxel_offset + pending_sec_voxel]);
                                        atomic_origin.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                        score_be_isotope_origin_voxel_device(
                                            be_isotope_origin_voxel_dose_device,
                                            be_isotope_category, number_of_voxels,
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                            }
                                            pending_sec_voxel_MeV = 0.0F;
                                            pending_sec_voxel = cur_voxel;
                                        }
                                        if (cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += sec_e;
                                            if (in_fov_dose_device != nullptr) {
                                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_in_fov(in_fov_dose_device[bin_z]);
                                                atomic_in_fov.fetch_add(static_cast<DoseAtomicT>(sec_e));
                                                cinel02_species_energy_add_device(
                                                    cinel02_species_energy_device, ledger_species_idx,
                                                    6U, sec_e);
                                            }
                                        }
                                    }
                                }
                                if (deposited_device != nullptr) {
                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dep(deposited_device[frag.parent_history]);
                                    atomic_dep.fetch_add(sec_e);
                                }
                            } else {
                                cinel02_species_energy_add_device(
                                    cinel02_species_energy_device, ledger_species_idx, 7U, sec_e);
                                cinel02_species_terminal_increment_device(
                                    cinel02_species_terminal_device, ledger_species_idx,
                                    static_cast<std::uint32_t>(Cinel02SpeciesLedgerSchema::boundary_escape));
                                sec_terminal_recorded = true;
                                if (escaped_device != nullptr) {
                                // Escaped phantom boundaries
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_esc(escaped_device[frag.parent_history]);
                                atomic_esc.fetch_add(sec_e);
                                }
                            }
                        }

                        if (!sec_terminal_recorded) {
                            cinel02_species_terminal_increment_device(
                                cinel02_species_terminal_device, ledger_species_idx,
                                static_cast<std::uint32_t>(
                                    Cinel02SpeciesLedgerSchema::continuous_stop));
                        }

                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_sec_bin]);
                            atomic_dose.fetch_add(static_cast<DoseAtomicT>(pending_sec_depth_MeV));
                        }
                        if (enable_voxel_scoring && pending_sec_voxel_MeV > 0.0F && pending_sec_voxel >= 0 && pending_sec_voxel < static_cast<int>(number_of_voxels)) {
                            sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_vox(voxel_dose_device[pending_sec_voxel]);
                            atomic_vox.fetch_add(static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            if (enable_charged_origin_voxel_scoring) {
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_origin(charged_origin_voxel_dose_device[
                                        charged_origin_voxel_offset + pending_sec_voxel]);
                                atomic_origin.fetch_add(
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                score_be_isotope_origin_voxel_device(
                                    be_isotope_origin_voxel_dose_device,
                                    be_isotope_category, number_of_voxels,
                                    pending_sec_voxel,
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            }
                        }
                    });
            });
            sec_event.wait_and_throw();
                secondary_kernel_seconds += event_duration_seconds(sec_event);
                generation_begin = generation_end;
                queue.copy(secondary_count_device, &secondary_count_host, 1)
                    .wait_and_throw();
                generation_end = sycl::min(
                    secondary_count_host, static_cast<std::uint32_t>(max_secondaries));
            }
            if (!config.fragment_birth_spectrum_output_file.empty()) {
                birth_secondaries_host.resize(generation_end);
                queue.copy(secondary_queue_device, birth_secondaries_host.data(),
                           generation_end).wait_and_throw();
            }
        }
    }
    if (birth_secondaries_host.empty() &&
        !config.fragment_birth_spectrum_output_file.empty() &&
        secondary_count_device != nullptr && secondary_queue_device != nullptr) {
        std::uint32_t birth_count = 0U;
        queue.copy(secondary_count_device, &birth_count, 1).wait_and_throw();
        birth_count = sycl::min(
            birth_count, static_cast<std::uint32_t>(max_secondaries));
        if (birth_count > 0U) {
            birth_secondaries_host.resize(birth_count);
            queue.copy(secondary_queue_device, birth_secondaries_host.data(),
                       birth_count).wait_and_throw();
        }
    }

    std::vector<DoseAtomicT> dose_device_host(number_of_bins);
    queue.copy(dose_device, dose_device_host.data(), number_of_bins).wait_and_throw();
    std::vector<double> dose_host(number_of_bins);
    std::transform(dose_device_host.begin(), dose_device_host.end(), dose_host.begin(),
                   [](DoseAtomicT val) { return static_cast<double>(val); });

    std::vector<double> voxel_dose_host;
    if (enable_voxel_scoring) {
        std::vector<DoseAtomicT> voxel_dose_device_host(number_of_voxels);
        queue.copy(voxel_dose_device, voxel_dose_device_host.data(), number_of_voxels)
            .wait_and_throw();
        voxel_dose_host.resize(number_of_voxels);
        std::transform(voxel_dose_device_host.begin(), voxel_dose_device_host.end(),
                       voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> charged_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            charged_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(charged_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        charged_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       charged_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> be_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            be_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(be_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        be_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       be_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> in_fov_dose_host;
    if (enable_voxel_scoring && in_fov_dose_device != nullptr) {
        std::vector<DoseAtomicT> in_fov_device_host(number_of_bins);
        queue.copy(in_fov_dose_device, in_fov_device_host.data(), number_of_bins).wait_and_throw();
        in_fov_dose_host.resize(number_of_bins);
        std::transform(in_fov_device_host.begin(), in_fov_device_host.end(),
                       in_fov_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<LetAtomicT> let_moments_host;
    if (enable_let_scoring) {
        let_moments_host.resize(4 * number_of_bins);
        queue.copy(let_moments_device, let_moments_host.data(), 4 * number_of_bins)
            .wait_and_throw();
    }

    std::vector<LetAtomicT> voxel_let_moments_host;
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        voxel_let_moments_host.resize(4 * number_of_voxels);
        queue.copy(voxel_let_moments_device, voxel_let_moments_host.data(), 4 * number_of_voxels)
            .wait_and_throw();
    }

    std::vector<std::uint64_t> primary_survival_host;
    std::vector<std::uint64_t> inelastic_reaction_host;
    if (primary_survival_device != nullptr) {
        primary_survival_host.resize(number_of_bins);
        inelastic_reaction_host.resize(number_of_bins);
        queue.copy(primary_survival_device, primary_survival_host.data(), number_of_bins);
        queue.copy(inelastic_reaction_device, inelastic_reaction_host.data(), number_of_bins);
    }
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<float> untracked_host(number_of_histories, 0.0F);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories);
    if (untracked_nuclear_device != nullptr) {
        queue.copy(untracked_nuclear_device, untracked_host.data(), number_of_histories);
    }
    std::array<float, kCinel02EnergySlots> cinel02_energy_host{};
    std::array<float, kCinel02SpeciesEnergySlots> cinel02_species_energy_host{};
    std::array<std::uint64_t, kCinel02SpeciesTerminalSlots>
        cinel02_species_terminal_host{};
    std::array<float, TransportResult::species_ledger_species_count>
        cinel02_replay_delta_host{};
    std::array<float, TransportResult::species_ledger_species_count>
        cinel02_replay_abs_delta_host{};
    std::array<std::uint64_t, TransportResult::species_ledger_species_count>
        cinel02_replay_delta_positive_host{};
    std::array<std::uint64_t, TransportResult::species_ledger_species_count>
        cinel02_replay_delta_negative_host{};
    std::array<std::uint64_t, TransportResult::species_ledger_species_count>
        cinel02_replay_valid_host{};
    std::array<std::uint64_t, kCinel02ReplayStatusSlots>
        cinel02_replay_status_counts_host{};
    std::array<float, kCinel02ReplayStatusSlots>
        cinel02_replay_status_incident_host{};
    std::array<float, kCinel02ReplayStatusSlots>
        cinel02_replay_status_delta_host{};
    std::array<float, kCinel02ReplayStatusSlots>
        cinel02_replay_status_abs_delta_host{};
    std::array<std::uint64_t, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_counts_host{};
    std::array<float, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_incident_host{};
    std::array<float, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_after_host{};
    std::array<float, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_local_host{};
    std::array<float, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_export_host{};
    std::array<float, kCinel02ParentOutcomeSlots>
        cinel02_parent_outcome_import_host{};
    std::array<std::uint64_t, kCinel02TransitionSlots>
        cinel02_generated_transition_counts_host{};
    std::array<float, kCinel02TransitionSlots>
        cinel02_generated_transition_energy_host{};
    std::array<std::uint64_t, kCinel02TransitionSlots>
        cinel02_queued_transition_counts_host{};
    std::array<float, kCinel02TransitionSlots>
        cinel02_queued_transition_energy_host{};
    if (cinel02_energy_device != nullptr) {
        queue.copy(cinel02_energy_device, cinel02_energy_host.data(),
                   kCinel02EnergySlots);
    }
    if (cinel02_species_energy_device != nullptr) {
        queue.copy(cinel02_species_energy_device,
                   cinel02_species_energy_host.data(),
                   kCinel02SpeciesEnergySlots);
    }
    if (cinel02_species_terminal_device != nullptr) {
        queue.copy(cinel02_species_terminal_device,
                   cinel02_species_terminal_host.data(),
                   kCinel02SpeciesTerminalSlots);
    }
    if (cinel02_replay_delta_device != nullptr) {
        queue.copy(cinel02_replay_delta_device, cinel02_replay_delta_host.data(),
                   TransportResult::species_ledger_species_count);
        queue.copy(cinel02_replay_abs_delta_device, cinel02_replay_abs_delta_host.data(),
                   TransportResult::species_ledger_species_count);
        queue.copy(cinel02_replay_delta_positive_device,
                   cinel02_replay_delta_positive_host.data(),
                   TransportResult::species_ledger_species_count);
        queue.copy(cinel02_replay_delta_negative_device,
                   cinel02_replay_delta_negative_host.data(),
                   TransportResult::species_ledger_species_count);
        queue.copy(cinel02_replay_valid_device, cinel02_replay_valid_host.data(),
                   TransportResult::species_ledger_species_count);
    }
    if (cinel02_replay_status_counts_device != nullptr) {
        queue.copy(cinel02_replay_status_counts_device,
                   cinel02_replay_status_counts_host.data(), kCinel02ReplayStatusSlots);
        queue.copy(cinel02_replay_status_incident_device,
                   cinel02_replay_status_incident_host.data(), kCinel02ReplayStatusSlots);
        queue.copy(cinel02_replay_status_delta_device,
                   cinel02_replay_status_delta_host.data(), kCinel02ReplayStatusSlots);
        queue.copy(cinel02_replay_status_abs_delta_device,
                   cinel02_replay_status_abs_delta_host.data(), kCinel02ReplayStatusSlots);
        queue.copy(cinel02_parent_outcome_counts_device,
                   cinel02_parent_outcome_counts_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_parent_outcome_incident_device,
                   cinel02_parent_outcome_incident_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_parent_outcome_after_device,
                   cinel02_parent_outcome_after_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_parent_outcome_local_device,
                   cinel02_parent_outcome_local_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_parent_outcome_export_device,
                   cinel02_parent_outcome_export_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_parent_outcome_import_device,
                   cinel02_parent_outcome_import_host.data(), kCinel02ParentOutcomeSlots);
        queue.copy(cinel02_generated_transition_counts_device,
                   cinel02_generated_transition_counts_host.data(), kCinel02TransitionSlots);
        queue.copy(cinel02_generated_transition_energy_device,
                   cinel02_generated_transition_energy_host.data(), kCinel02TransitionSlots);
        queue.copy(cinel02_queued_transition_counts_device,
                   cinel02_queued_transition_counts_host.data(), kCinel02TransitionSlots);
        queue.copy(cinel02_queued_transition_energy_device,
                   cinel02_queued_transition_energy_host.data(), kCinel02TransitionSlots);
    }
    std::array<std::uint64_t, kCinel02DiagSlots> cinel02_diag_host{};
    if (cinel02_diag_device != nullptr) {
        queue.copy(cinel02_diag_device, cinel02_diag_host.data(), kCinel02DiagSlots);
    }
    std::array<std::uint64_t, 26> fred_diag_host{};
    if (fred_diag_device != nullptr) {
        queue.copy(fred_diag_device, fred_diag_host.data(), 26);
    }
    uint32_t overflow_count_host = 0;
    float overflow_energy_host = 0.0F;
    float model_residual_host = 0.0F;
    float fred_q_host = 0.0F;
    float fred_neutron_host = 0.0F;
    float fred_remnant_host = 0.0F;
    uint32_t fred_fail_count_host = 0;
    float fred_fail_energy_host = 0.0F;
    uint32_t fred_cap_count_host = 0;
    float fred_cap_energy_host = 0.0F;
    if (secondary_overflow_count_device != nullptr) {
        queue.copy(secondary_overflow_count_device, &overflow_count_host, 1);
        queue.copy(secondary_overflow_energy_device, &overflow_energy_host, 1);
        queue.copy(fred_model_residual_device, &model_residual_host, 1);
        queue.copy(fred_q_device, &fred_q_host, 1);
        queue.copy(fred_neutron_device, &fred_neutron_host, 1);
        queue.copy(fred_remnant_device, &fred_remnant_host, 1);
        queue.copy(fred_fail_count_device, &fred_fail_count_host, 1);
        queue.copy(fred_fail_energy_device, &fred_fail_energy_host, 1);
        queue.copy(fred_cap_overflow_count_device, &fred_cap_count_host, 1);
        queue.copy(fred_cap_overflow_energy_device, &fred_cap_energy_host, 1);
    }
    queue.wait_and_throw();

    // Free buffers
    free_immutable_device(table_device);
    free_immutable_device(energy_grid_device);
    free_immutable_device(cumulative_range_device);
    free_immutable_device(cross_section_device);
    if (target_h_fraction_device != nullptr) {
        sycl::free(target_h_fraction_device, queue);
    }
    free_device(dose_device);
    free_device(primary_survival_device);
    free_device(inelastic_reaction_device);
    free_device(in_fov_dose_device);
    free_device(voxel_dose_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(be_isotope_origin_voxel_dose_device);
    free_device(let_moments_device);
    free_device(voxel_let_moments_device);
    free_device(deposited_device);
    free_device(secondary_queue_device);
    free_device(secondary_count_device);
    free_device(ion_species_sp_device);
    free_device(ion_energy_grid_device);
    free_device(ion_csda_a1_device);
    free_device(cinel02_interactions_device);
    free_device(cinel02_products_device);
    free_device(cinel02_energy_nodes_device);
    free_device(cinel02_event_offsets_device);
    free_device(cinel02_event_indices_device);
    free_device(cinel02_rate_groups_device);
    free_device(cinel02_rate_samples_device);
    free_device(cinel02_diag_device);
    free_device(cinel02_energy_device);
    free_device(cinel02_species_energy_device);
    free_device(cinel02_species_terminal_device);
    free_device(cinel02_replay_delta_device);
    free_device(cinel02_replay_abs_delta_device);
    free_device(cinel02_replay_delta_positive_device);
    free_device(cinel02_replay_delta_negative_device);
    free_device(cinel02_replay_valid_device);
    free_device(cinel02_replay_status_counts_device);
    free_device(cinel02_replay_status_incident_device);
    free_device(cinel02_replay_status_delta_device);
    free_device(cinel02_replay_status_abs_delta_device);
    free_device(cinel02_parent_outcome_counts_device);
    free_device(cinel02_parent_outcome_incident_device);
    free_device(cinel02_parent_outcome_after_device);
    free_device(cinel02_parent_outcome_local_device);
    free_device(cinel02_parent_outcome_export_device);
    free_device(cinel02_parent_outcome_import_device);
    free_device(cinel02_generated_transition_counts_device);
    free_device(cinel02_generated_transition_energy_device);
    free_device(cinel02_queued_transition_counts_device);
    free_device(cinel02_queued_transition_energy_device);
    free_device(untracked_nuclear_device);
    free_device(fred_prob_proj_h_device);
    free_device(fred_prob_proj_o_device);
    free_device(fred_prob_tgt_h_device);
    free_device(fred_prob_tgt_o_device);
    for (auto* allocation : event_lib_device_allocations) {
        if (allocation != nullptr) sycl::free(allocation, queue);
    }
    free_device(fluct_energy_device);
    free_device(fluct_density_device);
    free_device(fluct_probability_device);
    free_device(fluct_quantile_device);
    free_device(fred_2gr_mcs_device);
    free_device(fred_diag_device);
    free_device(secondary_overflow_count_device);
    free_device(secondary_overflow_energy_device);
    free_device(fred_model_residual_device);
    free_device(fred_q_device);
    free_device(fred_neutron_device);
    free_device(fred_remnant_device);
    free_device(fred_fail_count_device);
    free_device(fred_fail_energy_device);
    free_device(fred_cap_overflow_count_device);
    free_device(fred_cap_overflow_energy_device);
    free_device(escaped_device);
    free_device(steps_device);
    free_device(primary_spots_device);
    free_device(slab_z_ends_device);
    free_device(slab_densities_device);
    free_device(slab_radiation_lengths_device);
    free_device(material_sp_device);
    free_device(material_xs_device);
    free_device(insert_sp_device);
    free_device(insert_xs_device);
    free_device(ct_density_device);
    free_device(ct_material_device);
    free_device(ct_mass_sp_factor_lut_device);
    free_device(ct_mass_sp_za_rel_device);
    free_device(ct_sp_device);
    free_device(ct_xs_device);
    free_device(ct_ref_density_device);

    TransportResult result;
    result.backend = "sycl-" + resolved_device_name +
                     (config.enable_energy_straggling ? "+straggling" : "");
    if (config.uses_moment_matched_straggling()) {
        result.backend += "+moment-matched-straggling";
    }
    if (config.enable_step_stable_straggling) {
        result.backend += "+step-stable-primary-straggling";
    }
    const auto batch_has_energy_spread = std::any_of(
        config.primary_spot_batch.begin(), config.primary_spot_batch.end(),
        [](const auto& spot) { return spot.floats[1] > 0.0F; });
    if (config.beam_energy_spread > 0.0 || batch_has_energy_spread) {
        result.backend += "+espread";
    }
    if (!config.primary_spot_batch.empty()) {
        result.backend += "+spot-batch";
    }
    if (config.enable_tps_source) {
        result.backend += "+tps-source";
    }
    if (enable_multiple_scattering) {
        result.backend += "+multiple-scattering";
    }
    if (enable_ct_grid && enable_ct_material_mcs) {
        result.backend += "+ct-material-mcs";
    }
    if (enable_layered_phantom) {
        result.backend += use_material_tables ? "+layered-material" : "+layered-slab";
    }
    if (enable_hetero_insert) {
        result.backend += use_insert_material_tables ? "+hetero-insert-material" : "+hetero-insert";
    }
    if (enable_ct_grid) {
        if (use_ct_mass_sp) {
            result.backend += use_ct_density_mass_spr ? "+ct-grid-density-spr+ct-dda"
                                                     : "+ct-grid-mass-sp-lut+ct-dda";
        } else if (use_ct_material_sp) {
            result.backend += "+ct-grid-material+ct-dda";
        } else {
            result.backend += "+ct-grid+ct-dda";
        }
    }
    if constexpr (k_dose_atomic_fp32) {
        result.backend += "+fp32-dose";
    } else {
        result.backend += "+fp64-dose";
    }
    if (enable_voxel_scoring) {
        result.backend += "+voxel-scoring";
        if (!voxel_scorer_clamps_transport) {
            result.backend += "+scorer-decoupled";
        }
    }
    if (enable_let_scoring) {
        result.backend += "+letd-scoring";
    }

    if (config.enable_fragment_species_scoring && !enable_secondary_transport) {
        result.secondary_carbon_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_boron_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_beryllium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_lithium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_helium_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_proton_deposited_energy_MeV.assign(number_of_bins, 0.0);
        result.secondary_other_charged_deposited_energy_MeV.assign(number_of_bins, 0.0);
    }
    result.primary_survival_counts = std::move(primary_survival_host);
    result.inelastic_reaction_counts = std::move(inelastic_reaction_host);
    result.primary_deposited_energy_MeV = dose_host;
    result.deposited_energy_MeV = std::move(dose_host);
    result.voxel_deposited_energy_MeV = std::move(voxel_dose_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    result.be_isotope_origin_voxel_deposited_energy_MeV =
        std::move(be_isotope_origin_voxel_dose_host);
    result.in_fov_deposited_energy_MeV = std::move(in_fov_dose_host);
    if (!config.fragment_birth_spectrum_output_file.empty()) {
        constexpr std::size_t categories = light_isotope_category_count;
        constexpr std::size_t generations = birth_generation_bin_count;
        result.birth_counts_by_generation.assign(categories * generations, 0);
        result.birth_ke_sum_MeV_by_generation.assign(categories * generations, 0.0);
        result.birth_mevu_hist.assign(birth_hist_plane_size(birth_mevu_bin_count), 0);
        result.birth_depth_hist.assign(birth_hist_plane_size(number_of_bins), 0);
        result.birth_cos_hist.assign(birth_hist_plane_size(birth_cos_bin_count), 0);
        result.birth_parent_mevu_hist.assign(
            birth_hist_plane_size(birth_parent_mevu_bin_count), 0);
        result.birth_parent_z_hist.assign(
            birth_hist_plane_size(birth_parent_z_bin_count), 0);
        result.birth_parent_product_mevu_hist.assign(birth_joint_plane_size(), 0);

        const auto parent_mevu_bin = birth_parent_mevu_bin(
            config.initial_total_energy_MeV(), config.primary_mass_number);
        const auto parent_z_bin = birth_parent_z_bin(config.primary_atomic_number);
        for (const auto& fragment : birth_secondaries_host) {
            const auto generation = birth_generation_bin(
                static_cast<std::uint8_t>(fragment.generation));
            const auto category =
                light_isotope_category(fragment.z, fragment.a);
            if (category >= categories || fragment.a <= 0) {
                continue;
            }
            const auto summary_index = category * generations + generation;
            const auto mevu_bin =
                birth_mevu_bin(fragment.energy_MeV, fragment.a);
            const auto depth_bin = std::min(
                number_of_bins - 1,
                static_cast<std::size_t>(std::max(
                    0.0F, fragment.pos_z_mm / depth_bin_width_mm)));
            const auto cos_bin = birth_cos_bin(fragment.dir_z);
            ++result.birth_counts_by_generation[summary_index];
            result.birth_ke_sum_MeV_by_generation[summary_index] +=
                fragment.energy_MeV;
            ++result.birth_mevu_hist[birth_hist_index(
                category, generation, mevu_bin, birth_mevu_bin_count)];
            ++result.birth_depth_hist[birth_hist_index(
                category, generation, depth_bin, number_of_bins)];
            ++result.birth_cos_hist[birth_hist_index(
                category, generation, cos_bin, birth_cos_bin_count)];
            ++result.birth_parent_mevu_hist[birth_hist_index(
                category, generation, parent_mevu_bin, birth_parent_mevu_bin_count)];
            ++result.birth_parent_z_hist[birth_hist_index(
                category, generation, parent_z_bin, birth_parent_z_bin_count)];

            ++result.birth_parent_product_mevu_hist[birth_joint_index(
                category, generation, parent_mevu_bin, mevu_bin)];
        }
    }
    if (enable_let_scoring) {
        const auto extract_let_moment = [&](const std::size_t moment) {
            const auto begin = let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_bins);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_bins));
        };
        result.primary_letd_numerator = extract_let_moment(0);
        result.primary_letd_denominator = extract_let_moment(1);
        result.all_hadron_letd_numerator = extract_let_moment(2);
        result.all_hadron_letd_denominator = extract_let_moment(3);
    }
    if (enable_let_scoring && voxel_let_moments_device != nullptr) {
        const auto extract_voxel_let_moment = [&](const std::size_t moment) {
            const auto begin = voxel_let_moments_host.begin() +
                               static_cast<std::ptrdiff_t>(moment * number_of_voxels);
            return std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(number_of_voxels));
        };
        result.primary_voxel_letd_numerator = extract_voxel_let_moment(0);
        result.primary_voxel_letd_denominator = extract_voxel_let_moment(1);
        result.all_hadron_voxel_letd_numerator = extract_voxel_let_moment(2);
        result.all_hadron_voxel_letd_denominator = extract_voxel_let_moment(3);
    }

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
        std::accumulate(untracked_host.begin(), untracked_host.end(), 0.0);
    for (std::size_t i = 0; i < 18; ++i) {
        result.fred_isotope_counts[i] = fred_diag_host[i];
    }
    result.fred_inelastic_events = fred_diag_host[18];
    result.fred_retry_sum = fred_diag_host[19];
    result.fred_energy_scaled_events = fred_diag_host[20];
    result.fred_projectile_az_open_events = fred_diag_host[21];
    result.fred_leftover_target_a_sum = fred_diag_host[22];
    result.fred_leftover_target_z_sum = fred_diag_host[23];
    result.fred_leftover_projectile_a_sum = fred_diag_host[24];
    result.fred_leftover_projectile_z_sum = fred_diag_host[25];
    result.fred_model_residual_MeV = static_cast<double>(model_residual_host);
    result.fred_model_unassigned_MeV = static_cast<double>(model_residual_host);
    result.fred_q_MeV = static_cast<double>(fred_q_host);
    result.fred_neutron_ke_MeV = static_cast<double>(fred_neutron_host);
    result.fred_remnant_local_MeV = static_cast<double>(fred_remnant_host);
    result.fred_resample_failed_events = fred_fail_count_host;
    result.fred_resample_failed_energy_MeV = static_cast<double>(fred_fail_energy_host);
    result.fred_product_capacity_overflow_events = fred_cap_count_host;
    result.fred_product_capacity_overflow_energy_MeV = static_cast<double>(fred_cap_energy_host);
    result.fred_invert_error_proj_h = invert_err_proj_h;
    result.fred_invert_error_proj_o = invert_err_proj_o;
    result.fred_invert_error_tgt_h = invert_err_tgt_h;
    result.fred_invert_error_tgt_o = invert_err_tgt_o;
    result.secondary_queue_overflow = overflow_count_host;
    result.secondary_queue_overflow_energy_MeV = static_cast<double>(overflow_energy_host);
    result.cinel02_diagnostics = cinel02_diag_host;
    for (std::size_t i = 0; i < kCinel02EnergySlots; ++i) {
        result.cinel02_energy_ledger_MeV[i] =
            static_cast<double>(cinel02_energy_host[i]);
    }
    for (std::size_t i = 0; i < kCinel02SpeciesEnergySlots; ++i) {
        result.cinel02_species_transport_ledger_MeV[i] =
            static_cast<double>(cinel02_species_energy_host[i]);
    }
    result.cinel02_species_terminal_reason_counts =
        cinel02_species_terminal_host;
    for (std::size_t i = 0; i < TransportResult::species_ledger_species_count; ++i) {
        result.cinel02_replay_delta_MeV_per_u[i] =
            static_cast<double>(cinel02_replay_delta_host[i]);
        result.cinel02_replay_abs_delta_MeV_per_u[i] =
            static_cast<double>(cinel02_replay_abs_delta_host[i]);
        result.cinel02_replay_delta_positive_counts[i] =
            cinel02_replay_delta_positive_host[i];
        result.cinel02_replay_delta_negative_counts[i] =
            cinel02_replay_delta_negative_host[i];
        result.cinel02_replay_valid_counts[i] = cinel02_replay_valid_host[i];
    }
    for (std::size_t i = 0; i < kCinel02ReplayStatusSlots; ++i) {
        result.cinel02_replay_status_counts[i] = cinel02_replay_status_counts_host[i];
        result.cinel02_replay_status_incident_energy_MeV[i] =
            static_cast<double>(cinel02_replay_status_incident_host[i]);
        result.cinel02_replay_status_delta_MeV_per_u[i] =
            static_cast<double>(cinel02_replay_status_delta_host[i]);
        result.cinel02_replay_status_abs_delta_MeV_per_u[i] =
            static_cast<double>(cinel02_replay_status_abs_delta_host[i]);
    }
    for (std::size_t i = 0; i < kCinel02ParentOutcomeSlots; ++i) {
        result.cinel02_parent_outcome_counts[i] = cinel02_parent_outcome_counts_host[i];
        result.cinel02_parent_outcome_incident_energy_MeV[i] =
            static_cast<double>(cinel02_parent_outcome_incident_host[i]);
        result.cinel02_parent_outcome_after_energy_MeV[i] =
            static_cast<double>(cinel02_parent_outcome_after_host[i]);
        result.cinel02_parent_outcome_local_deposit_MeV[i] =
            static_cast<double>(cinel02_parent_outcome_local_host[i]);
        result.cinel02_parent_outcome_export_MeV[i] =
            static_cast<double>(cinel02_parent_outcome_export_host[i]);
        result.cinel02_parent_outcome_import_MeV[i] =
            static_cast<double>(cinel02_parent_outcome_import_host[i]);
    }
    for (std::size_t i = 0; i < kCinel02TransitionSlots; ++i) {
        result.cinel02_generated_transition_counts[i] =
            cinel02_generated_transition_counts_host[i];
        result.cinel02_generated_transition_kinetic_MeV[i] =
            static_cast<double>(cinel02_generated_transition_energy_host[i]);
        result.cinel02_queued_transition_counts[i] =
            cinel02_queued_transition_counts_host[i];
        result.cinel02_queued_transition_kinetic_MeV[i] =
            static_cast<double>(cinel02_queued_transition_energy_host[i]);
    }
    if (use_cinel02 && config.cinel02_strict_match &&
        (result.cinel02_diagnostics[4] != 0U || result.cinel02_diagnostics[5] != 0U ||
         result.cinel02_diagnostics[18] != 0U || result.cinel02_diagnostics[19] != 0U ||
         result.cinel02_diagnostics[34] != 0U ||
         result.cinel02_diagnostics[2] != result.cinel02_diagnostics[3] + result.cinel02_diagnostics[4] + result.cinel02_diagnostics[5] ||
         result.cinel02_diagnostics[16] != result.cinel02_diagnostics[17] + result.cinel02_diagnostics[18] + result.cinel02_diagnostics[19])) {
        throw std::runtime_error("CINEL02 strict match failed");
    }
    result.nuclear_interactions = use_cinel02
        ? result.cinel02_diagnostics[2] + result.cinel02_diagnostics[16]
        : result.fred_inelastic_events;
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;
}

}  // namespace carbon

#endif
