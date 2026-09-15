#include "carbon/nuclear_collision.hpp"
#include "carbon/charged_species.hpp"
#include "carbon/secondary_schedule.hpp"
#include "carbon/delta_moments_data.hpp"
#include "carbon/runtime_timing.hpp"
#include "carbon/unified_em_view.hpp"
#ifndef CARBON_SECONDARY_STEP_PROFILE
#define CARBON_SECONDARY_STEP_PROFILE 0
#endif
#include <fstream>
#include <sstream>
#include "carbon/cross_section.hpp"
#include "carbon/hadronic_cache_candidate.hpp"
#include "carbon/ct_grid.hpp"
#include "carbon/device.hpp"
#include "carbon/electron_transport.hpp"
#include "carbon/electron_joint_response.hpp"
#include "carbon/water_electron_response.hpp"
#include "carbon/material_electron_response.hpp"
#include "carbon/material_electron_device.hpp"
#include "carbon/electron_packet_transport.hpp"
#include "carbon/electron_short_range.hpp"
#include "carbon/energy_loss_fluctuation.hpp"
#include "carbon/inelastic.hpp"
#include "carbon/inelastic_package_v2.hpp"
#include "carbon/multiple_scattering.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"
#include "carbon/slab_phantom.hpp"
#include "carbon/stopping_power.hpp"
#include "carbon/tps_source.hpp"
#include "carbon/schneider_stopping_table.hpp"
#include "carbon/schneider_ion_stopping_table.hpp"
#include "carbon/nuclear_elastic_scattering.hpp"
#include "carbon/all_ion_elastic.hpp"
#include "carbon/detail/device_memory_tracker.hpp"
#include "carbon/straggling.hpp"
#include "carbon/transport.hpp"
#include "carbon/sha256.hpp"
#include "carbon/schneider_rate_table.hpp"
#include "carbon/schneider_delta_tail.hpp"
#include "carbon/longitudinal_ray.hpp"
#include "carbon/schneider_target_sampler.hpp"
#include "carbon/secondary_rate_table.hpp"
#include "carbon/inelastic_package_v3.hpp"
#include "carbon/cinel03_event_rotation.hpp"
#include "carbon/schneider_ct_device_context.hpp"

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
template<int EmMode> class CarbonSecondaryTransportKernel;

#include "detail/sycl_dose_atomic.inc"
#include "detail/sycl_profile.inc"
#include "detail/sycl_transport_context_impl.inc"
#include "detail/sycl_transport_context_methods.inc"
#include "detail/sycl_cinel02_device.inc"

namespace {

// State persists across launch boundaries; a pause must not finalize dose or
// diagnostics. Only survivor indices move during stable compaction.
struct alignas(16) SecondaryResumeState {
    bool sec_terminal_recorded{};
    bool unified_secondary_escaped_ct{};
    ContinuousSpeciesTrackTally continuous_species_tally{};
    float sec_e{};
    float sec_x{};
    float sec_y{};
    float sec_z{};
    float sec_dx{};
    float sec_dy{};
    float sec_dz{};
    float pending_sec_depth_MeV{};
    float pending_sec_voxel_MeV{};
    int pending_sec_bin{};
    int pending_sec_voxel{};
    std::uint64_t unified_secondary_counter{};
    UnifiedEmState unified_secondary_state{};
    std::array<std::uint64_t,8> unified_secondary_audit{};
    uint32_t sec_steps{};
    std::uint64_t local_sec_rate_queries{};
    std::uint64_t local_sec_steps{};
    std::array<double,6> he4_audit{};
    std::array<std::uint64_t,CARBON_SECONDARY_STEP_PROFILE?60:0> sec_prof{};
};
static_assert(sizeof(SecondaryResumeState) % 16 == 0);
struct UnifiedEmFailureRecord {int reason,section,z,a;float energy,density,step;};
inline void record_unified_em_failure(unsigned* count,UnifiedEmFailureRecord* records,
    int reason,int section,int z,int a,float energy,float density,float step) {
    if(!count)return;
    sycl::atomic_ref<unsigned,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> counter(*count);
    const auto slot=counter.fetch_add(1);if(slot<16)records[slot]={reason,section,z,a,energy,density,step};
}
inline void flush_unified_em_audit(std::uint64_t* global,
                                  const std::array<std::uint64_t,8>& local) {
#pragma unroll
    for(int i=0;i<8;++i) {
        if(local[i]) {
            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                sycl::memory_scope::device,sycl::access::address_space::global_space>
                count(global[i]);
            count.fetch_add(local[i]);
        }
    }
}

#include "detail/sycl_device_math.inc"
#include "detail/sycl_score_device.inc"

using carbon::detail::DeviceMemoryTracker;

// Independent Schneider-CT nuclear diagnostics (named schema, NOT the
// CINEL02 water array). All increments are relaxed device atomics.
inline void schneider_diag_add_device(std::uint64_t* diag, SchneiderDiagSlot slot,
                                      std::uint64_t value) {
    if (diag == nullptr || value == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(diag[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(value);
}

inline void schneider_diag_increment_device(std::uint64_t* diag, SchneiderDiagSlot slot) {
    schneider_diag_add_device(diag, slot, 1U);
}

inline void schneider_float_add_device(float* floats, SchneiderFloatSlot slot, float value) {
    if (floats == nullptr || value == 0.0F) {
        return;
    }
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(floats[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(value);
}

inline void schneider_energy_add_device(std::uint64_t* diag, SchneiderDiagSlot slot,
                                          float value_MeV) {
    if (diag == nullptr || !(value_MeV > 0.0F)) {
        return;
    }
    const auto fixed =
        static_cast<std::uint64_t>(static_cast<double>(value_MeV) * 1.0e6);
    if (fixed == 0) {
        return;
    }
    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(diag[static_cast<std::uint32_t>(slot)]);
    ref.fetch_add(fixed);
}

inline void schneider_float_max_device(float* floats, SchneiderFloatSlot slot, float value) {
    if (floats == nullptr || !(value > 0.0F)) {
        return;
    }
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(floats[static_cast<std::uint32_t>(slot)]);
    float current = ref.load();
    while (value > current) {
        if (ref.compare_exchange_strong(current, value)) {
            break;
        }
    }
}

inline std::uint8_t schneider_lookup_status_code(Cinel03LookupStatus status) noexcept {
    switch (status) {
    case Cinel03LookupStatus::Hit: return 255;
    case Cinel03LookupStatus::MissingProjectile: return 2;
    case Cinel03LookupStatus::MissingTarget: return 3;
    case Cinel03LookupStatus::BelowEnergyDomain: return 0;
    case Cinel03LookupStatus::AboveEnergyDomain: return 1;
    case Cinel03LookupStatus::EnergyGapTooLarge: return 4;
    case Cinel03LookupStatus::EmptyNode: return 5;
    }
    return 5;
}

// Bounded per-miss record log. Misses are rare (10^2 in 50k runs); the
// buffer holds 2^19 entries and counts drops instead of wrapping.
inline void schneider_log_miss_device(SchneiderMissRecord* log, std::uint32_t* counts,
                                      std::uint32_t cap, bool is_primary,
                                      int pz, int pa, int tz, std::uint8_t section,
                                      std::uint8_t generation,
                                      const Cinel03LookupResult& result,
                                      float query_e_u, float step_dE_MeV,
                                      float total_rate_per_mm, float hazard_step_mm,
                                      float incident_e_MeV, float birth_e_MeV) {
    if (log == nullptr || counts == nullptr) {
        return;
    }
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        count_ref(counts[0]);
    const std::uint32_t slot = count_ref.fetch_add(1U);
    if (slot >= cap) {
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            drop_ref(counts[1]);
        drop_ref.fetch_add(1U);
        return;
    }
    SchneiderMissRecord rec{};
    rec.projectile_z = static_cast<std::int16_t>(pz);
    rec.projectile_a = static_cast<std::int16_t>(pa);
    rec.target_z = static_cast<std::int16_t>(tz);
    rec.status = schneider_lookup_status_code(result.status);
    rec.section_id = section;
    rec.is_primary = is_primary ? 1 : 0;
    rec.generation = generation;
    rec.query_energy_MeV_per_u = query_e_u;
    rec.step_dE_MeV = step_dE_MeV;
    rec.total_rate_per_mm = total_rate_per_mm;
    rec.hazard_step_mm = hazard_step_mm;
    rec.incident_energy_MeV = incident_e_MeV;
    rec.birth_energy_MeV = birth_e_MeV;
    log[slot] = rec;
}

// Bounded per-track log for registry-unknown projectiles (isotope census).
inline void schneider_log_unsupported_track_device(
    SchneiderUnsupportedTrack* log, std::uint32_t* counts, std::uint32_t cap,
    int pz, int pa, std::uint16_t generation, float birth_e_MeV,
    float x_mm, float y_mm, float z_mm) {
    if (log == nullptr || counts == nullptr) {
        return;
    }
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        count_ref(counts[0]);
    const std::uint32_t slot = count_ref.fetch_add(1U);
    if (slot >= cap) {
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            drop_ref(counts[1]);
        drop_ref.fetch_add(1U);
        return;
    }
    SchneiderUnsupportedTrack rec{};
    rec.projectile_z = static_cast<std::int16_t>(pz);
    rec.projectile_a = static_cast<std::int16_t>(pa);
    rec.generation = generation;
    rec.birth_energy_MeV = birth_e_MeV;
    rec.birth_x_mm = x_mm;
    rec.birth_y_mm = y_mm;
    rec.birth_z_mm = z_mm;
    log[slot] = rec;
}

// Record one CINEL03 lookup outcome into the named Schneider counters.
// incident_energy_MeV is the parent kinetic energy at the vertex; domain
// misses itemize it under OutOfDomainEnergy, all other lookup failures
// under LookupFailureEnergy (never silently inside untracked).
inline void schneider_record_lookup_device(std::uint64_t* diag, float* floats,
                                           bool is_primary,
                                           const Cinel03LookupResult& result,
                                           float incident_energy_MeV) {
    if (diag == nullptr) {
        return;
    }
    const float incident = incident_energy_MeV > 0.0F ? incident_energy_MeV : 0.0F;
    switch (result.status) {
    case Cinel03LookupStatus::Hit:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryExactTargetHits
                             : SchneiderDiagSlot::SecondaryExactTargetHits);
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEventsReplayed
                             : SchneiderDiagSlot::SecondaryEventsReplayed);
        schneider_float_add_device(
            floats, is_primary ? SchneiderFloatSlot::PrimaryMismatchSum
                               : SchneiderFloatSlot::SecondaryMismatchSum,
            result.absolute_energy_mismatch_MeV_per_u);
        schneider_float_max_device(
            floats, is_primary ? SchneiderFloatSlot::PrimaryMismatchMax
                               : SchneiderFloatSlot::SecondaryMismatchMax,
            result.absolute_energy_mismatch_MeV_per_u);
        break;
    case Cinel03LookupStatus::MissingProjectile:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryMissingProjectile
                             : SchneiderDiagSlot::SecondaryMissingProjectile);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::MissingTarget:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryMissingTarget
                             : SchneiderDiagSlot::SecondaryMissingTarget);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::BelowEnergyDomain:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryBelowDomain
                             : SchneiderDiagSlot::SecondaryBelowDomain);
        schneider_float_add_device(floats, SchneiderFloatSlot::OutOfDomainEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::AboveEnergyDomain:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryAboveDomain
                             : SchneiderDiagSlot::SecondaryAboveDomain);
        schneider_float_add_device(floats, SchneiderFloatSlot::OutOfDomainEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::EnergyGapTooLarge:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEnergyGapMisses
                             : SchneiderDiagSlot::SecondaryEnergyGapMisses);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    case Cinel03LookupStatus::EmptyNode:
        schneider_diag_increment_device(
            diag, is_primary ? SchneiderDiagSlot::PrimaryEmptyNodes
                             : SchneiderDiagSlot::SecondaryEmptyNodes);
        schneider_float_add_device(floats, SchneiderFloatSlot::LookupFailureEnergy,
                                   incident);
        break;
    }
}

float cuda_clock_warmup(sycl::queue& queue, DeviceMemoryTracker& tracker) {
    auto* dummy = tracker.allocate<float>(1024);
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
    tracker.free(dummy);
    return static_cast<float>(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace

[[gnu::noinline]] SchneiderCtDeviceContext upload_schneider_ct_device_context(
    sycl::queue& queue,
    carbon::detail::DeviceMemoryTracker& mem_tracker,
    const TransportConfig& config,
    const float* schneider_stopping_device,
    std::uint32_t schneider_sp_sections,
    std::uint32_t schneider_sp_energies,
    float schneider_sp_e_min,
    float schneider_sp_e_max,
    float schneider_sp_inv_dE) {

    SchneiderCtDeviceContext ctx{};
    ctx.mode = config.material_physics_mode;
    ctx.unified_water = config.unified_water_nuclear_transport;

    if (!ctx.uses_cinel03() || config.is_primary_attenuation_only_mode()) {
        return ctx;
    }
    PureWaterMaterial water_material{};
    SchneiderMaterialTable water_source_materials{};
    if (ctx.unified_water) {
        water_material = PureWaterMaterial::from_probe(config.unified_water_material_file,
                                                       config.unified_water_material_sha256);
        water_source_materials = SchneiderMaterialTable::from_topas_file(
            config.ct_schneider_file.empty() ? std::filesystem::path("data/HUtoMaterialSchneider.txt") : config.ct_schneider_file);
        ctx.water_radiation_length_g_cm2 = water_material.radiation_length_g_cm2;
    }
    const auto upload_water_rates = [&](const MaterialNuclearRates& rates) {
        auto view = rates.view();
        auto* partials = mem_tracker.allocate<double>(rates.partials().size());
        auto* domains = mem_tracker.allocate<MaterialRateDomain>(rates.domains().size());
        if (!partials || !domains) throw std::bad_alloc();
        queue.copy(rates.partials().data(),partials,rates.partials().size()).wait_and_throw();
        queue.copy(rates.domains().data(),domains,rates.domains().size()).wait_and_throw();
        view.partials=partials;view.domains=domains;return view;
    };

    std::filesystem::path primary_rate_file = config.ct_schneider_primary_rate_file;
    std::filesystem::path c12_cinel_file = config.ct_schneider_c12_cinel03_file;
    std::filesystem::path sec_rate_file = config.ct_schneider_secondary_rate_file;
    std::filesystem::path sec_cinel_file = config.ct_schneider_secondary_cinel03_file;

    // 1. Primary Target Sampler
    if (std::filesystem::exists(primary_rate_file)) {
        const auto rate_table = SchneiderRateTable::from_binary(primary_rate_file);
        if (ctx.unified_water) ctx.water_primary_rates = upload_water_rates(
            MaterialNuclearRates::water_primary(rate_table,water_source_materials,water_material.hydrogen_mass_fraction));
        const SchneiderTargetSampler target_sampler(rate_table);
        const auto cdf_size = target_sampler.cdf_table().size();
        const auto tot_size = target_sampler.total_mass_rates().size();
        float* dev_cdf = mem_tracker.allocate<float>(cdf_size);
        float* dev_total_rates = mem_tracker.allocate<float>(tot_size);
        if (dev_cdf == nullptr || dev_total_rates == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(target_sampler.cdf_table().data(), dev_cdf, cdf_size);
        queue.copy(target_sampler.total_mass_rates().data(), dev_total_rates, tot_size);
        ctx.primary_sampler = target_sampler.device_table();
        ctx.primary_sampler.cdf_table = dev_cdf;
        ctx.primary_sampler.total_mass_rates = dev_total_rates;
        // v3: raw partials + per-target domain for the masked device path.
        if (target_sampler.rate_version() == 3) {
            const auto& partials = target_sampler.partial_rates();
            float* dev_partial = mem_tracker.allocate<float>(partials.size());
            float* dev_emin = mem_tracker.allocate<float>(target_sampler.domain_emin().size());
            float* dev_emax = mem_tracker.allocate<float>(target_sampler.domain_emax().size());
            unsigned char* dev_has =
                mem_tracker.allocate<unsigned char>(target_sampler.domain_has().size());
            if (dev_partial == nullptr || dev_emin == nullptr || dev_emax == nullptr ||
                dev_has == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(partials.data(), dev_partial, partials.size());
            queue.copy(target_sampler.domain_emin().data(), dev_emin,
                       target_sampler.domain_emin().size());
            queue.copy(target_sampler.domain_emax().data(), dev_emax,
                       target_sampler.domain_emax().size());
            queue.copy(target_sampler.domain_has().data(), dev_has,
                       target_sampler.domain_has().size());
            ctx.primary_sampler.partial_rates = dev_partial;
            ctx.primary_sampler.domain_emin = dev_emin;
            ctx.primary_sampler.domain_emax = dev_emax;
            ctx.primary_sampler.domain_has = dev_has;
        }
    }

    // 1b. Elastic C12 SCHNELXS sampler (diagnostic only; absent file
    // keeps elastic disabled with zero behavior change).
    if (!config.ct_elastic_section_rate_file.empty()) {
        const auto elastic_table = SchneiderRateTable::from_binary(
            config.ct_elastic_section_rate_file, {}, "SCHNELXS", 3);
        if (compute_file_sha256_hex(config.ct_elastic_section_rate_file) !=
            config.ct_elastic_section_rate_sha256)
            throw std::runtime_error("Elastic rate data SHA256 mismatch");
        const SchneiderTargetSampler elastic_sampler(elastic_table);
        const auto cdf_size = elastic_sampler.cdf_table().size();
        const auto tot_size = elastic_sampler.total_mass_rates().size();
        float* dev_cdf = mem_tracker.allocate<float>(cdf_size);
        float* dev_total_rates = mem_tracker.allocate<float>(tot_size);
        if (dev_cdf == nullptr || dev_total_rates == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(elastic_sampler.cdf_table().data(), dev_cdf, cdf_size);
        queue.copy(elastic_sampler.total_mass_rates().data(), dev_total_rates, tot_size);
        ctx.elastic_sampler = elastic_sampler.device_table();
        ctx.elastic_sampler.cdf_table = dev_cdf;
        ctx.elastic_sampler.total_mass_rates = dev_total_rates;
        const auto& partials = elastic_sampler.partial_rates();
        float* dev_partial = mem_tracker.allocate<float>(partials.size());
        float* dev_emin = mem_tracker.allocate<float>(elastic_sampler.domain_emin().size());
        float* dev_emax = mem_tracker.allocate<float>(elastic_sampler.domain_emax().size());
        unsigned char* dev_has =
            mem_tracker.allocate<unsigned char>(elastic_sampler.domain_has().size());
        if (dev_partial == nullptr || dev_emin == nullptr || dev_emax == nullptr ||
            dev_has == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(partials.data(), dev_partial, partials.size());
        queue.copy(elastic_sampler.domain_emin().data(), dev_emin,
                   elastic_sampler.domain_emin().size());
        queue.copy(elastic_sampler.domain_emax().data(), dev_emax,
                   elastic_sampler.domain_emax().size());
        queue.copy(elastic_sampler.domain_has().data(), dev_has,
                   elastic_sampler.domain_has().size());
        ctx.elastic_sampler.partial_rates = dev_partial;
        ctx.elastic_sampler.domain_emin = dev_emin;
        ctx.elastic_sampler.domain_emax = dev_emax;
        ctx.elastic_sampler.domain_has = dev_has;
        std::cout << "[schneider-elastic-rates] mode=elastic-diagnostic sections=25 "
                  << "targets=13 energies=921 source="
                  << config.ct_elastic_section_rate_file << '\n';
    }

    // 2. Primary C12 CINEL03 Package
    if (std::filesystem::exists(c12_cinel_file)) {
        const auto c12_pkg = InelasticPackageV3Table::from_binary(c12_cinel_file);
        const auto c12_dev = c12_pkg.make_device_tables();
        auto* dev_c12_nodes = mem_tracker.allocate<Cinel03EnergyNode>(c12_dev.energy_nodes.size());
        auto* dev_c12_offsets = mem_tracker.allocate<std::uint32_t>(c12_dev.event_offsets.size());
        auto* dev_c12_indices = mem_tracker.allocate<std::uint32_t>(c12_dev.event_indices.size());
        auto* dev_c12_ints = mem_tracker.allocate<Cinel03DeviceInteraction>(c12_dev.interactions.size());
        auto* dev_c12_prods = mem_tracker.allocate<Cinel03DeviceProduct>(c12_dev.products.size());
        if (dev_c12_nodes == nullptr || dev_c12_offsets == nullptr || dev_c12_indices == nullptr ||
            dev_c12_ints == nullptr || dev_c12_prods == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(c12_dev.energy_nodes.data(), dev_c12_nodes, c12_dev.energy_nodes.size());
        queue.copy(c12_dev.event_offsets.data(), dev_c12_offsets, c12_dev.event_offsets.size());
        queue.copy(c12_dev.event_indices.data(), dev_c12_indices, c12_dev.event_indices.size());
        queue.copy(c12_dev.interactions.data(), dev_c12_ints, c12_dev.interactions.size());
        queue.copy(c12_dev.products.data(), dev_c12_prods, c12_dev.products.size());

        ctx.c12_energy_nodes = dev_c12_nodes;
        ctx.c12_event_offsets = dev_c12_offsets;
        ctx.c12_event_indices = dev_c12_indices;
        ctx.c12_interactions = dev_c12_ints;
        ctx.c12_products = dev_c12_prods;
        ctx.c12_node_count = static_cast<std::uint32_t>(c12_dev.energy_nodes.size());
        ctx.c12_total_events = static_cast<std::uint32_t>(c12_dev.interactions.size());
        ctx.c12_total_products = static_cast<std::uint32_t>(c12_dev.products.size());
    }

    // 3. Secondary Rates (when secondary transport or production mode is active)
    if (std::filesystem::exists(sec_rate_file) &&
        (config.enable_secondary_transport || config.run_mode == RunMode::production || !config.ct_schneider_secondary_rate_file.empty())) {
        const auto sec_rate_table = SecondaryRateTable::from_binary(sec_rate_file);
        if (ctx.unified_water) ctx.water_secondary_rates = upload_water_rates(
            MaterialNuclearRates::water_secondary(sec_rate_table,water_source_materials,water_material.hydrogen_mass_fraction));
        std::vector<float> sec_total_rates_float(sec_rate_table.mass_total_rates().size());
        for (std::size_t i = 0; i < sec_rate_table.mass_total_rates().size(); ++i) {
            sec_total_rates_float[i] = static_cast<float>(sec_rate_table.mass_total_rates()[i]);
        }
        std::vector<float> sec_partial_rates_float(sec_rate_table.mass_partial_rates().size());
        for (std::size_t i = 0; i < sec_rate_table.mass_partial_rates().size(); ++i) {
            sec_partial_rates_float[i] = static_cast<float>(sec_rate_table.mass_partial_rates()[i]);
        }
        float* dev_sec_total = mem_tracker.allocate<float>(sec_total_rates_float.size());
        float* dev_sec_partial = mem_tracker.allocate<float>(sec_partial_rates_float.size());
        if (dev_sec_total == nullptr || dev_sec_partial == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(sec_total_rates_float.data(), dev_sec_total, sec_total_rates_float.size());
        queue.copy(sec_partial_rates_float.data(), dev_sec_partial, sec_partial_rates_float.size());

        ctx.sec_total_rates = dev_sec_total;
        ctx.sec_partial_rates = dev_sec_partial;
        // v3: bundle-ordered projectile registry + per-(projectile,target)
        // domain for the only supported, masked device path.
        {
            const auto& projs = sec_rate_table.projectiles();
            std::vector<std::int32_t> keys;
            keys.reserve(projs.size() * 2);
            for (const auto& p : projs) {
                keys.push_back(p.z);
                keys.push_back(p.a);
            }
            std::int32_t* dev_keys = mem_tracker.allocate<std::int32_t>(keys.size());
            if (dev_keys == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(keys.data(), dev_keys, keys.size());
            ctx.sec_proj_keys = dev_keys;
            const auto& doms = sec_rate_table.channel_domains();
            std::vector<float> demin;
            std::vector<float> demax;
            std::vector<unsigned char> dhas;
            demin.reserve(doms.size());
            demax.reserve(doms.size());
            dhas.reserve(doms.size());
            for (const auto& d : doms) {
                demin.push_back(static_cast<float>(d.energy_min_mevu));
                demax.push_back(static_cast<float>(d.energy_max_mevu));
                dhas.push_back(d.has_support);
            }
            float* dev_demin = mem_tracker.allocate<float>(demin.size());
            float* dev_demax = mem_tracker.allocate<float>(demax.size());
            unsigned char* dev_dhas = mem_tracker.allocate<unsigned char>(dhas.size());
            if (dev_demin == nullptr || dev_demax == nullptr || dev_dhas == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(demin.data(), dev_demin, demin.size());
            queue.copy(demax.data(), dev_demax, demax.size());
            queue.copy(dhas.data(), dev_dhas, dhas.size());
            ctx.sec_domain_emin = dev_demin;
            ctx.sec_domain_emax = dev_demax;
            ctx.sec_domain_has = dev_dhas;
        }
        ctx.sec_num_projectiles = static_cast<std::uint32_t>(sec_rate_table.num_projectiles());
        ctx.sec_num_sections = static_cast<std::uint32_t>(kSecondaryNumSections);
        ctx.sec_num_targets = static_cast<std::uint32_t>(kSecondaryNumTargets);
        ctx.sec_num_energies = static_cast<std::uint32_t>(sec_rate_table.num_energies());
        ctx.sec_energy_min_MeV_per_u = static_cast<float>(sec_rate_table.energy_min_mevu());
        ctx.sec_energy_step_MeV_per_u = static_cast<float>(sec_rate_table.energy_step_mevu());
        ctx.sec_inv_energy_step = static_cast<float>(1.0 / sec_rate_table.energy_step_mevu());
    }

    // 4. Secondary CINEL03 Package
    if (std::filesystem::exists(sec_cinel_file) &&
        (config.enable_secondary_transport || config.run_mode == RunMode::production || !config.ct_schneider_secondary_cinel03_file.empty())) {
        const auto sec_pkg = InelasticPackageV3Table::from_binary(sec_cinel_file);
        const auto sec_dev = sec_pkg.make_device_tables();
        auto* dev_sec_nodes = mem_tracker.allocate<Cinel03EnergyNode>(sec_dev.energy_nodes.size());
        auto* dev_sec_offsets = mem_tracker.allocate<std::uint32_t>(sec_dev.event_offsets.size());
        auto* dev_sec_indices = mem_tracker.allocate<std::uint32_t>(sec_dev.event_indices.size());
        auto* dev_sec_ints = mem_tracker.allocate<Cinel03DeviceInteraction>(sec_dev.interactions.size());
        auto* dev_sec_prods = mem_tracker.allocate<Cinel03DeviceProduct>(sec_dev.products.size());
        if (dev_sec_nodes == nullptr || dev_sec_offsets == nullptr || dev_sec_indices == nullptr ||
            dev_sec_ints == nullptr || dev_sec_prods == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(sec_dev.energy_nodes.data(), dev_sec_nodes, sec_dev.energy_nodes.size());
        queue.copy(sec_dev.event_offsets.data(), dev_sec_offsets, sec_dev.event_offsets.size());
        queue.copy(sec_dev.event_indices.data(), dev_sec_indices, sec_dev.event_indices.size());
        queue.copy(sec_dev.interactions.data(), dev_sec_ints, sec_dev.interactions.size());
        queue.copy(sec_dev.products.data(), dev_sec_prods, sec_dev.products.size());

        ctx.sec_energy_nodes = dev_sec_nodes;
        ctx.sec_event_offsets = dev_sec_offsets;
        ctx.sec_event_indices = dev_sec_indices;
        ctx.sec_interactions = dev_sec_ints;
        ctx.sec_products = dev_sec_prods;
        ctx.sec_node_count = static_cast<std::uint32_t>(sec_dev.energy_nodes.size());
        ctx.sec_total_events = static_cast<std::uint32_t>(sec_dev.interactions.size());
        ctx.sec_total_products = static_cast<std::uint32_t>(sec_dev.products.size());
    }

    // 5. Stopping power
    ctx.stopping_power_device = schneider_stopping_device;
    ctx.sp_sections = schneider_sp_sections;
    ctx.sp_energies = schneider_sp_energies;
    ctx.sp_e_min = schneider_sp_e_min;
    ctx.sp_e_max = schneider_sp_e_max;
    ctx.sp_inv_dE = schneider_sp_inv_dE;

    queue.wait_and_throw();

    return ctx;
}

template<int EmMode>
[[gnu::noinline]] TransportResult transport_sycl_impl(const TransportConfig& config,
                               const StoppingPowerTable& stopping_power,
                               const CrossSectionTable& cross_section,
                               const std::string& device_name,
                               SyclTransportContext* context) {
    RuntimeScope runtime_transport("transport_total");
    RuntimeScope runtime_setup("transport_setup_including_upload");
    config.validate();
    validate_schneider_ct_startup(config);

    // Verify provenance and metadata early before ANY GPU queue creation or device memory allocation
    auto is_valid_64hex = [](const std::string& s) -> bool {
        if (s.size() != 64) return false;
        for (char c : s) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                return false;
            }
        }
        return true;
    };

    std::string early_verified_sha256 = "";
    if (config.is_primary_attenuation_only_mode()) {
        const auto xs_file = config.ct_schneider_cross_section_file;
        if (xs_file.empty() || !std::filesystem::exists(xs_file)) {
            throw std::runtime_error("Schneider cross section file missing in validation mode: " + xs_file.string());
        }
        const auto meta_path = xs_file.parent_path() / (xs_file.stem().string() + ".metadata.json");
        if (!std::filesystem::exists(meta_path)) {
            throw std::runtime_error("Schneider cross section metadata file missing: " + meta_path.string());
        }
        std::ifstream meta_in(meta_path);
        if (!meta_in.is_open()) {
            throw std::runtime_error("Failed to open Schneider cross section metadata file: " + meta_path.string());
        }
        std::string source_sha256 = "";
        std::string line;
        while (std::getline(meta_in, line)) {
            if (line.find("\"data_sha256\"") != std::string::npos) {
                const auto colon = line.find(':');
                const auto quote1 = line.find('"', colon);
                const auto quote2 = line.find('"', quote1 + 1);
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                    break;
                }
            }
        }
        if (source_sha256.empty() || !is_valid_64hex(source_sha256)) {
            throw std::runtime_error(
                "Schneider cross section metadata missing or malformed data_sha256 in: " + meta_path.string());
        }
        const auto actual_sha256 = compute_file_sha256_hex(xs_file);
        if (source_sha256 != actual_sha256) {
            throw std::runtime_error(
                "Schneider cross section runtime SHA256 mismatch: file=" + actual_sha256 +
                ", metadata=" + source_sha256);
        }
        early_verified_sha256 = actual_sha256;
    } else if (!config.ct_schneider_cross_section_file.empty() &&
               std::filesystem::exists(config.ct_schneider_cross_section_file)) {
        const auto xs_file = config.ct_schneider_cross_section_file;
        const auto meta_path = xs_file.parent_path() / (xs_file.stem().string() + ".metadata.json");
        if (std::filesystem::exists(meta_path)) {
            std::ifstream meta_in(meta_path);
            if (meta_in.is_open()) {
                std::string source_sha256 = "";
                std::string line;
                while (std::getline(meta_in, line)) {
                    if (line.find("\"data_sha256\"") != std::string::npos) {
                        const auto colon = line.find(':');
                        const auto quote1 = line.find('"', colon);
                        const auto quote2 = line.find('"', quote1 + 1);
                        if (quote1 != std::string::npos && quote2 != std::string::npos) {
                            source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                            break;
                        }
                    }
                }
                if (!source_sha256.empty() && is_valid_64hex(source_sha256)) {
                    const auto actual_sha256 = compute_file_sha256_hex(xs_file);
                    if (source_sha256 != actual_sha256) {
                        throw std::runtime_error(
                            "Schneider cross section runtime SHA256 mismatch: file=" + actual_sha256 +
                            ", metadata=" + source_sha256);
                    }
                    early_verified_sha256 = actual_sha256;
                }
            }
        }
    }

    std::string early_verified_stopping_sha256 = "";
    if (config.is_primary_attenuation_only_mode()) {
        const auto sp_file = config.ct_schneider_stopping_power_file;
        if (sp_file.empty() || !std::filesystem::exists(sp_file)) {
            throw std::runtime_error("Schneider stopping power file missing in validation mode: " + sp_file.string());
        }
        const auto meta_path = sp_file.parent_path() / (sp_file.stem().string() + ".metadata.json");
        if (!std::filesystem::exists(meta_path)) {
            throw std::runtime_error("Schneider stopping power metadata file missing: " + meta_path.string());
        }
        std::ifstream meta_in(meta_path);
        if (!meta_in.is_open()) {
            throw std::runtime_error("Failed to open Schneider stopping power metadata file: " + meta_path.string());
        }
        std::string source_sha256 = "";
        std::string line;
        while (std::getline(meta_in, line)) {
            if (line.find("\"data_sha256\"") != std::string::npos) {
                const auto colon = line.find(':');
                const auto quote1 = line.find('"', colon);
                const auto quote2 = line.find('"', quote1 + 1);
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                    break;
                }
            }
        }
        if (source_sha256.empty() || !is_valid_64hex(source_sha256)) {
            throw std::runtime_error(
                "Schneider stopping power metadata missing or malformed data_sha256 in: " + meta_path.string());
        }
        const auto actual_sha256 = compute_file_sha256_hex(sp_file);
        if (source_sha256 != actual_sha256) {
            throw std::runtime_error(
                "Schneider stopping power runtime SHA256 mismatch: file=" + actual_sha256 +
                ", metadata=" + source_sha256);
        }
        early_verified_stopping_sha256 = actual_sha256;
    } else if (!config.ct_schneider_stopping_power_file.empty() &&
               std::filesystem::exists(config.ct_schneider_stopping_power_file)) {
        const auto sp_file = config.ct_schneider_stopping_power_file;
        const auto meta_path = sp_file.parent_path() / (sp_file.stem().string() + ".metadata.json");
        if (std::filesystem::exists(meta_path)) {
            std::ifstream meta_in(meta_path);
            if (meta_in.is_open()) {
                std::string source_sha256 = "";
                std::string line;
                while (std::getline(meta_in, line)) {
                    if (line.find("\"data_sha256\"") != std::string::npos) {
                        const auto colon = line.find(':');
                        const auto quote1 = line.find('"', colon);
                        const auto quote2 = line.find('"', quote1 + 1);
                        if (quote1 != std::string::npos && quote2 != std::string::npos) {
                            source_sha256 = line.substr(quote1 + 1, quote2 - quote1 - 1);
                            break;
                        }
                    }
                }
                if (!source_sha256.empty() && is_valid_64hex(source_sha256)) {
                    const auto actual_sha256 = compute_file_sha256_hex(sp_file);
                    if (source_sha256 != actual_sha256) {
                        throw std::runtime_error(
                            "Schneider stopping power runtime SHA256 mismatch: file=" + actual_sha256 +
                            ", metadata=" + source_sha256);
                    }
                    early_verified_stopping_sha256 = actual_sha256;
                }
            }
        }
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

    DeviceMemoryTracker mem_tracker{queue};

    const auto free_device = [&](auto* pointer) {
        mem_tracker.free(pointer);
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
                                            : mem_tracker.allocate<float>(table_size);
    float* energy_grid_device =
        context != nullptr ? context->impl_->energy_grid_device
                           : (config.enable_csda_range_energy_loss
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr);
    float* cumulative_range_device =
        context != nullptr ? context->impl_->cumulative_range_device
                           : (config.enable_csda_range_energy_loss
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr);
    float* cross_section_device =
        context != nullptr ? context->impl_->cross_section_device
                           : mem_tracker.allocate<float>(cross_section_table_size);
    float* target_h_fraction_device =
        mem_tracker.allocate<float>(cross_section_table_size);

    const auto free_immutable_device = [&](auto* pointer) {
        if (!reuse_immutable_buffers && pointer != nullptr) {
            mem_tracker.free(pointer);
        }
    };

    float* schneider_primary_xs_device = nullptr;
    std::uint32_t schneider_xs_sections = 0;
    std::uint32_t schneider_xs_energies = 0;
    float schneider_xs_e_min = 0.0F;
    float schneider_xs_inv_dE = 0.0F;
    const bool use_unified_water = config.unified_water_nuclear_transport;

    bool use_schneider_primary_xs = use_unified_water;
    float* schneider_stopping_device = nullptr;
    std::uint32_t schneider_sp_sections = 0;
    std::uint32_t schneider_sp_energies = 0;
    float schneider_sp_e_min = 0.0F;
    float schneider_sp_e_max = 0.0F;
    float schneider_sp_inv_dE = 0.0F;
    bool use_schneider_stopping = false;
    // Scheme-2 secondary section stopping (SCHNIOSP v1). Null = current
    // legacy water path when the material bank is not explicitly configured.
    float* schneider_ion_sp_device = nullptr;
    bool use_schneider_ion_sp = false;
    std::uint32_t* ion_stopping_failures = nullptr;

    Cinel02DeviceInteraction* cinel02_interactions_device = nullptr;
    Cinel02DeviceProduct* cinel02_products_device = nullptr;
    Cinel02EnergyNode* cinel02_energy_nodes_device = nullptr;
    std::uint32_t* cinel02_event_offsets_device = nullptr;
    std::uint32_t* cinel02_event_indices_device = nullptr;
    Cinel02RateGroup* cinel02_rate_groups_device = nullptr;
    Cinel02RateSample* cinel02_rate_samples_device = nullptr;
    Cinel02MaterialRateGroup* cinel02_ct_rate_groups_device = nullptr;
    Cinel02RateSample* cinel02_ct_rate_samples_device = nullptr;
    std::uint32_t cinel02_interaction_count = 0U;
    std::uint32_t cinel02_product_count = 0U;
    std::uint32_t cinel02_energy_node_count = 0U;
    std::uint32_t cinel02_rate_group_count = 0U;
    std::uint32_t cinel02_rate_sample_count = 0U;
    std::uint32_t cinel02_ct_rate_group_count = 0U;
    std::uint32_t cinel02_ct_rate_sample_count = 0U;
    float cinel02_ct_rate_reference_density_g_per_cm3 = 1.0F;
    constexpr std::size_t kSchneiderDiagSlots =
        static_cast<std::size_t>(SchneiderDiagSlot::Count);
    constexpr std::size_t kSchneiderFloatSlots =
        static_cast<std::size_t>(SchneiderFloatSlot::Count);
    std::uint64_t* schneider_diag_device = nullptr;
    float* schneider_float_device = nullptr;
    // Bounded per-record logs (Schneider CT only). Capacities are generous
    // for full-shard runs (50k needs ~10^2 entries); overflow is counted,
    // never silently wrapped.
    constexpr std::uint32_t kSchneiderMissLogCap = 1U << 19;   // 524288
    constexpr std::uint32_t kSchneiderTrackLogCap = 1U << 19;  // 524288
    SchneiderMissRecord* schneider_miss_device = nullptr;
    SchneiderUnsupportedTrack* schneider_track_log_device = nullptr;
    std::uint32_t* schneider_miss_count_device = nullptr;
    std::uint32_t* schneider_track_count_device = nullptr;
    constexpr std::size_t kCinel02SpeciesEnergySlots =
        TransportResult::species_ledger_species_count *
        TransportResult::species_ledger_metric_count;
    float* cinel02_species_energy_device = nullptr;
    double* grid_deposited_in_device = nullptr;
    double* grid_deposited_out_device = nullptr;
    constexpr std::size_t kCinel02SpeciesTerminalSlots =
        Cinel02SpeciesLedgerSchema::species_count *
        Cinel02SpeciesLedgerSchema::terminal_reason_count;
    std::uint64_t* cinel02_species_terminal_device = nullptr;
    // CINEL02-only package/rate uploads and diagnostic allocations retired.

    // Read-only species ledger is mode-independent: secondary EM transport
    // records per-species deposited/escaped energy unconditionally on device
    // (null-checked increments), so these two buffers remain in the shared
    // CINEL03 framework. No physics effect:
    // ledger writes are isolated atomics into dedicated buffers.
    if (cinel02_species_energy_device == nullptr) {
        cinel02_species_energy_device =
            mem_tracker.allocate<float>(kCinel02SpeciesEnergySlots);
        if (cinel02_species_energy_device == nullptr) throw std::bad_alloc();
    }
    if (cinel02_species_terminal_device == nullptr) {
        cinel02_species_terminal_device =
            mem_tracker.allocate<std::uint64_t>(kCinel02SpeciesTerminalSlots);
        if (cinel02_species_terminal_device == nullptr) throw std::bad_alloc();
    }
    queue.fill(cinel02_species_energy_device, 0.0F,
               kCinel02SpeciesEnergySlots).wait_and_throw();
    queue.fill(cinel02_species_terminal_device, std::uint64_t{0},
               kCinel02SpeciesTerminalSlots).wait_and_throw();

    // Explicit in-grid/outside-grid deposited-energy sinks (global MeV).
    // Every site crediting deposited ledgers splits the same amount here
    // using that site's paired voxel-scorer guard, so total ≈ in + outside
    // and voxel_sum ≈ in_grid hold by construction. Double precision:
    // global accumulators reach 1e10 MeV where float32 atomics would absorb
    // MeV-scale adds.
    grid_deposited_in_device = mem_tracker.allocate<double>(1);
    grid_deposited_out_device = mem_tracker.allocate<double>(1);
    if (grid_deposited_in_device == nullptr ||
        grid_deposited_out_device == nullptr) {
        throw std::bad_alloc();
    }
    queue.fill(grid_deposited_in_device, 0.0, 1).wait_and_throw();
    queue.fill(grid_deposited_out_device, 0.0, 1).wait_and_throw();

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
                                   ? mem_tracker.allocate<float>(slab_layer_count)
                                   : nullptr;
    float* slab_densities_device = slab_layer_count > 0
                                       ? mem_tracker.allocate<float>(slab_layer_count)
                                       : nullptr;
    float* slab_radiation_lengths_device =
        slab_layer_count > 0
            ? mem_tracker.allocate<float>(slab_layer_count)
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
                                   ? mem_tracker.allocate<float>(material_sp_host.size())
                                   : nullptr;
    float* material_xs_device = material_table_count > 0
                                   ? mem_tracker.allocate<float>(material_xs_host.size())
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
                                  ? mem_tracker.allocate<float>(table_size)
                                  : nullptr;
    float* insert_xs_device = use_insert_material_tables
                                  ? mem_tracker.allocate<float>(cross_section_table_size)
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
    const auto ct_primary_midpoint_stopping = config.ct_primary_midpoint_stopping;

    if (std::getenv("CARBON_JOINT_EM_DATA") || std::getenv("CARBON_DIAGNOSTIC_PRIMARY_START_DEDX"))
        throw std::invalid_argument("Obsolete isolated EM environment switch; use explicit primary_em_model configuration");
    // Explicit research model; legacy transport remains the default.
    const bool unified_em=EmMode<0?config.em_model=="g4_material_joint_v1":EmMode==1;
    const float em_primary_step_scale=static_cast<float>(config.em_primary_step_scale);
    const float em_secondary_step_scale=static_cast<float>(config.em_secondary_step_scale);
    if(unified_em)std::cout<<"[em-performance] material_cache="<<CARBON_EM_MATERIAL_CACHE
        <<" step_cache="<<CARBON_EM_STEP_CACHE<<" local_audit="<<CARBON_EM_LOCAL_AUDIT
        <<" exact_index="<<CARBON_EM_EXACT_INDEX<<"\n";
    UnifiedEmDevice unified_device;
    UnifiedEmMaterial* unified_materials=nullptr;
    UnifiedEmSpecies* unified_species=nullptr;
    UnifiedEmSectionRange* unified_sections=nullptr;
    unsigned* unified_energy_index=nullptr;
    unsigned* unified_failure_count=nullptr;
    UnifiedEmFailureRecord* unified_failure_records=nullptr;
    UnifiedEmRecord* unified_records=nullptr;
    UnifiedEmNode* unified_nodes=nullptr;
    float* delta_mean_device=nullptr;
    EmCubicSegmentCandidate<float>* unified_segments=nullptr;
    std::uint64_t* unified_audit=nullptr;
    std::uint64_t* sec_step_profile_device=nullptr;
    if(CARBON_SECONDARY_STEP_PROFILE) {
        sec_step_profile_device=mem_tracker.allocate<std::uint64_t>(60);
        if(!sec_step_profile_device)throw std::bad_alloc();
        queue.fill(sec_step_profile_device,std::uint64_t{0},60).wait_and_throw();
    }
    if(unified_em){
        auto package=runtime_call("unified_em_package_load_verify",[&]{return UnifiedEmPackage::load(config.em_package_file,config.em_package_sha256);});
        auto upload=[&]<class T>(const std::vector<T>& values){
            auto* ptr=mem_tracker.allocate<T>(values.size());if(!ptr)throw std::bad_alloc();
            queue.copy(values.data(),ptr,values.size()).wait_and_throw();return ptr;
        };
        const auto delta_path=config.em_delta_moments_file.empty()
            ? config.em_package_file.parent_path()/delta_moments_filename
            : config.em_delta_moments_file;
        const auto delta_means=runtime_call("delta_moments_load_verify",[&]{
            return load_delta_moments(delta_path,config.em_package_sha256,package.nodes.size());
        });
        delta_mean_device=upload(delta_means);
        std::cout<<"[delta-moments] condensed_partition_v1; aggregate Gamma; no delta clock; nodes="
                 <<package.nodes.size()<<" SHA256="<<delta_moments_sha256<<"\n";
        std::vector<UnifiedEmSectionRange> sections(26);
        for(unsigned i=0;i<package.materials.size();++i) {
            auto& range=sections[package.materials[i].section+1];
            if(range.begin<0)range.begin=i;
            range.end=i;
        }
        unified_failure_count=mem_tracker.allocate<unsigned>(1);
        unified_failure_records=mem_tracker.allocate<UnifiedEmFailureRecord>(16);
        if(!unified_failure_count || !unified_failure_records)throw std::bad_alloc();
        queue.fill(unified_failure_count,0u,1).wait_and_throw();
        unified_sections=upload(sections);
        if constexpr(CARBON_EM_EXACT_INDEX) {
            const auto index=build_unified_em_index(package);
            unified_energy_index=upload(index);
        }
        unified_materials=upload(package.materials);unified_species=upload(package.species);
        unified_records=upload(package.records);unified_nodes=upload(package.nodes);unified_segments=upload(package.segments);
        unified_audit=mem_tracker.allocate<std::uint64_t>(8);if(!unified_audit)throw std::bad_alloc();
        queue.fill(unified_audit,std::uint64_t{0},8).wait_and_throw();
        unified_device={unified_materials,unified_species,unified_records,unified_nodes,unified_segments,static_cast<unsigned>(package.materials.size()),unified_sections,unified_energy_index,delta_mean_device};
        std::cout<<"[unified-em] all 18 charged ions; water + Schneider density nodes; native particle step parameters plus 1% combined mean-loss guard; local aggregate delta deposition; density cut-onset/patient accuracy validation pending\n";
    }
    const auto ct_secondary_exact_faces = config.ct_secondary_exact_faces;
    std::cout << "[stopping-config] primary_midpoint=" << ct_primary_midpoint_stopping
              << " secondary_exact_faces=" << ct_secondary_exact_faces
              << " secondary_ion_section_file=" << config.ct_secondary_ion_section_stopping_file
              << " sha256=" << config.ct_secondary_ion_section_stopping_sha256 << '\n';
    const auto ct_secondary_mcs_off = config.ct_secondary_mcs_off_diagnostic;

    const bool use_schneider_delta_tail =
        EmMode!=1 && !config.ct_schneider_delta_tail_file.empty();
    std::optional<SchneiderDeltaTailTable> schneider_delta_tail;
    float* schneider_delta_energies_device = nullptr;
    float* schneider_delta_fractions_device = nullptr;
    float* schneider_delta_radii_device = nullptr;
    std::uint8_t* schneider_delta_source_eligible_device = nullptr;
    std::uint64_t* schneider_delta_energy_device = nullptr;
    std::size_t schneider_delta_energy_count = 0;
    std::size_t schneider_delta_quantile_count = 0;
    // Optional longitudinal (forward) supplement. Shares the transverse
    // eligibility mask and section-0 scope; empty file disables it exactly.
    const bool use_schneider_delta_longitudinal =
        use_schneider_delta_tail &&
        !config.ct_schneider_delta_longitudinal_file.empty();
    const bool use_longitudinal_interface_mass = config.ct_longitudinal_interface_mass_diagnostic;
    const bool use_electron_joint = EmMode!=1 && !config.ct_electron_joint_response_diagnostic_file.empty();
    // Legacy C12-only diagnostic is separate from the all-projectile bank.
    const bool use_ct_elastic = config.ct_elastic_diagnostic;
    const bool ct_elastic_all_targets = config.ct_elastic_all_targets;
    // Production speed switch: skip reporting-only joint diagnostic atomics.
    // Slots 1 (energy domain) and 2 (miss/blocked) stay always-on: they feed
    // fail-closed quality gates. Skipped counters never feed transport.
    const bool enable_electron_joint_diagnostics = config.electron_joint_diagnostics;
    const bool use_material_electron=EmMode!=1 && !config.material_electron_response_index_file.empty();
    const bool use_material_ct=use_material_electron && config.enable_ct_grid;
    const bool use_water_electron=EmMode!=1 && ((use_material_electron && !use_material_ct) || !config.water_electron_response_diagnostic_file.empty());
    std::optional<MaterialElectronDeviceBank> material_electron_bank;
    const MaterialElectronResponseView* material_electron_views=nullptr;
    std::size_t material_electron_view_count=0;
    std::uint64_t* material_untracked_device=nullptr; // photon / other, micro-MeV
    struct MaterialPacketFailure {
        std::uint64_t history{},step{},rng{};
        ElectronEnergyPacket before{},after{};
        ElectronContinuationAdvance advance{};
    };
    MaterialPacketFailure* material_failure_device=nullptr;
    std::uint64_t* short_range_hits_device=nullptr;
    const double material_response_density=config.water_density_g_per_cm3;
    WaterElectronChannel* water_electron_channels_device=nullptr;
    WaterElectronSample* water_electron_samples_device=nullptr;
    WaterElectronPathNode* water_electron_nodes_device=nullptr;
    std::uint32_t* water_electron_heads_device=nullptr;
    double* water_electron_radius_device=nullptr;
    std::size_t water_electron_node_count=0;
    std::size_t water_electron_channel_count=0;
    ElectronJointChannel* electron_joint_channels_device=nullptr;
    ElectronJointSample* electron_joint_samples_device=nullptr;
    ElectronPathRange* electron_path_ranges_device=nullptr;
    std::array<double,3>* electron_path_vectors_device=nullptr;
    MassPathBounds* electron_path_bounds_device=nullptr;
    std::string electron_path_sha256;
    std::uint64_t* electron_joint_diag_device=nullptr;
    std::array<double,kElectronJointSections> electron_joint_minimum{},electron_joint_maximum{};
    double electron_joint_ceiling=185;
    if(use_electron_joint) {
        const auto table=ElectronJointResponseTable::from_csv(config.ct_electron_joint_response_diagnostic_file,
            config.ct_electron_joint_response_sha256,config.ct_electron_joint_response_metadata_sha256);
        if(config.ct_electron_joint_patient_experiment && !table.full_schneider_scope)
            throw std::invalid_argument("Patient experiment requires full 25-section multi-energy schema 3, not a two-material pilot");
        // Explicitly authorized ISOLATED smoke diagnostic only. Config still
        // rejects research/production, spots, other energies and nuclear-on;
        // quality still always reports unvalidated_electron_joint_response.
        electron_joint_channels_device=mem_tracker.allocate<ElectronJointChannel>(kElectronJointChannels);
        electron_joint_samples_device=mem_tracker.allocate<ElectronJointSample>(table.samples.size());
        electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
        if(!electron_joint_channels_device || !electron_joint_samples_device || !electron_joint_diag_device)throw std::bad_alloc();
        queue.copy(table.channels.data(),electron_joint_channels_device,kElectronJointChannels).wait_and_throw();
        queue.copy(table.samples.data(),electron_joint_samples_device,table.samples.size()).wait_and_throw();
        if(!table.path_ranges.empty()) {
            electron_path_sha256=table.path_sha256;
            electron_path_ranges_device=mem_tracker.allocate<ElectronPathRange>(table.path_ranges.size());
            electron_path_vectors_device=mem_tracker.allocate<std::array<double,3>>(table.path_vectors.size());
            electron_path_bounds_device=mem_tracker.allocate<MassPathBounds>(table.path_ranges.size());
            if(!electron_path_ranges_device || !electron_path_vectors_device || !electron_path_bounds_device)throw std::bad_alloc();
            std::vector<MassPathBounds> bounds(table.path_ranges.size());
            for(std::size_t i=0;i<bounds.size();++i) {
                const auto r=table.path_ranges[i];
                bounds[i]=mass_path_bounds(r.count,[&](std::size_t j){return table.path_vectors[r.offset+j];});
            }
            queue.copy(bounds.data(),electron_path_bounds_device,bounds.size()).wait_and_throw();
            queue.copy(table.path_ranges.data(),electron_path_ranges_device,table.path_ranges.size()).wait_and_throw();
            queue.copy(table.path_vectors.data(),electron_path_vectors_device,table.path_vectors.size()).wait_and_throw();
        }
        queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
        electron_joint_minimum=table.minimum;electron_joint_maximum=table.maximum;
        electron_joint_ceiling=table.energy_ceiling_MeVu;
    }
    if(use_material_electron && !use_material_ct) {
            const auto index=MaterialElectronResponseIndex::load(config.material_electron_response_index_file,
                config.material_electron_response_index_sha256);
            const bool mapped=config.material_electron_response_memory_mode=="host_mapped";
            const auto budget=static_cast<std::size_t>(mapped?config.material_electron_response_host_budget_MiB:
                config.material_electron_response_device_budget_MiB)*1024*1024;
            material_electron_bank.emplace(queue,mapped?MaterialElectronMemory::host_mapped:MaterialElectronMemory::device);
            material_electron_bank->load(index,{{-1,material_response_density}},budget-7*sizeof(std::uint64_t));
            material_electron_views=material_electron_bank->views();
            material_electron_view_count=material_electron_bank->size();
            electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
            if(!electron_joint_diag_device)throw std::bad_alloc();
            queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
    } else if(use_water_electron) {
        const auto table=WaterElectronResponseTable::load(config.water_electron_response_diagnostic_file,
            config.water_electron_response_sha256,config.water_electron_response_metadata_sha256);
        water_electron_channel_count=table.channels.size();
        const double required_ceiling=(use_material_electron || config.water_electron_high_energy_diagnostic) ? 450.0 : 300.0;
        if(table.channels.empty() || table.channels.back().high<required_ceiling)
            throw std::invalid_argument("Water electron loaded table does not cover declared source domain");
        water_electron_channels_device=mem_tracker.allocate<WaterElectronChannel>(water_electron_channel_count);
        water_electron_samples_device=mem_tracker.allocate<WaterElectronSample>(table.samples.size());
        water_electron_nodes_device=mem_tracker.allocate<WaterElectronPathNode>(table.nodes.size());
        water_electron_heads_device=mem_tracker.allocate<std::uint32_t>(table.heads.size());
        water_electron_radius_device=mem_tracker.allocate<double>(table.prefix_radius.size());
        electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
        if(!water_electron_channels_device || !water_electron_samples_device || !water_electron_nodes_device ||
           !water_electron_heads_device || !water_electron_radius_device || !electron_joint_diag_device)throw std::bad_alloc();
        queue.copy(table.channels.data(),water_electron_channels_device,water_electron_channel_count).wait_and_throw();
        queue.copy(table.samples.data(),water_electron_samples_device,table.samples.size()).wait_and_throw();
        queue.copy(table.nodes.data(),water_electron_nodes_device,table.nodes.size()).wait_and_throw();
        queue.copy(table.heads.data(),water_electron_heads_device,table.heads.size()).wait_and_throw();
        queue.copy(table.prefix_radius.data(),water_electron_radius_device,table.prefix_radius.size()).wait_and_throw();
        queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
        water_electron_node_count=table.nodes.size();
    }
    if (use_schneider_delta_longitudinal && k_dose_atomic_fp32) {
        throw std::runtime_error(
            "ct_schneider_delta_longitudinal_file requires an FP64 dose build "
            "(CARBON_DOSE_FP32=OFF): the distributed forward shares are far below "
            "FP32 atomic granularity at clinical per-bin totals and would be "
            "silently dropped, failing energy closure");
    }
    const auto schneider_long_fraction_scale = static_cast<float>(
        config.ct_schneider_delta_longitudinal_scale);
    std::optional<SchneiderLongitudinalTable> schneider_longitudinal;
    double schneider_long_diagnostic_density = kLongitudinalReferenceDensityGPerCm3;
    LongitudinalDomainRecord* longitudinal_domain_device = nullptr;
    if (config.ct_longitudinal_homogeneous_density_diagnostic || use_longitudinal_interface_mass) {
        longitudinal_domain_device = mem_tracker.allocate<LongitudinalDomainRecord>(
            kLongitudinalDomainLogCap);
        if (!longitudinal_domain_device) throw std::bad_alloc();
    }
    float* schneider_long_energies_device = nullptr;
    float* schneider_long_fractions_device = nullptr;
    float* schneider_long_lambdas_device = nullptr;
    std::size_t schneider_long_energy_count = 0;
    if (use_schneider_delta_tail) {
        schneider_delta_tail = SchneiderDeltaTailTable::from_csv(
            config.ct_schneider_delta_tail_file);
        schneider_delta_energy_count = schneider_delta_tail->energy_count();
        schneider_delta_quantile_count = schneider_delta_tail->quantile_count();
        schneider_delta_energies_device =
            mem_tracker.allocate<float>(schneider_delta_energy_count);
        schneider_delta_fractions_device =
            mem_tracker.allocate<float>(schneider_delta_energy_count);
        schneider_delta_radii_device = mem_tracker.allocate<float>(
            schneider_delta_energy_count * schneider_delta_quantile_count);
        schneider_delta_energy_device = mem_tracker.allocate<std::uint64_t>(10);
        if (schneider_delta_energies_device == nullptr ||
            schneider_delta_fractions_device == nullptr ||
            schneider_delta_radii_device == nullptr ||
            schneider_delta_energy_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.copy(schneider_delta_tail->energies_MeV_per_u().data(),
                   schneider_delta_energies_device, schneider_delta_energy_count);
        queue.copy(schneider_delta_tail->moved_fractions().data(),
                   schneider_delta_fractions_device, schneider_delta_energy_count);
        queue.copy(schneider_delta_tail->radii_mm().data(),
                   schneider_delta_radii_device,
                   schneider_delta_energy_count * schneider_delta_quantile_count);
        queue.fill(schneider_delta_energy_device, std::uint64_t{0}, 10)
            .wait_and_throw();
        if (use_schneider_delta_longitudinal) {
            schneider_longitudinal = SchneiderLongitudinalTable::from_csv(
                config.ct_schneider_delta_longitudinal_file);
            schneider_long_energy_count = schneider_longitudinal->energy_count();
            schneider_long_energies_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            schneider_long_fractions_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            schneider_long_lambdas_device =
                mem_tracker.allocate<float>(schneider_long_energy_count);
            if (schneider_long_energies_device == nullptr ||
                schneider_long_fractions_device == nullptr ||
                schneider_long_lambdas_device == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(schneider_longitudinal->energies_MeV_per_u().data(),
                       schneider_long_energies_device, schneider_long_energy_count);
            queue.copy(schneider_longitudinal->forward_fractions().data(),
                       schneider_long_fractions_device, schneider_long_energy_count);
            queue.copy(schneider_longitudinal->lambdas_mm().data(),
                       schneider_long_lambdas_device, schneider_long_energy_count)
                .wait_and_throw();
        }
    }

    if (enable_ct_grid) {
        const auto grid = CtGrid::load(
            config.ct_grid_file, config.ct_schneider_file, config.ct_dicom_origin_mode);
        if(use_material_ct) {
            if(grid.file_version<CtGrid::version_v2)
                throw std::invalid_argument("Material electron transport requires Schneider section IDs");
            if(grid.nx!=config.voxel_bins_x || grid.ny!=config.voxel_bins_y || grid.nz!=config.voxel_bins_z ||
               std::abs(grid.spacing_x_mm-config.voxel_size_x_mm)>1e-6 ||
               std::abs(grid.spacing_y_mm-config.voxel_size_y_mm)>1e-6 ||
               std::abs(grid.spacing_z_mm-config.voxel_size_z_mm)>1e-6 || std::abs(grid.origin_z_mm)>1e-6)
                throw std::invalid_argument("Material CT packet scorer must align with CT grid, z origin zero");
            const auto response_index=MaterialElectronResponseIndex::load(config.material_electron_response_index_file,
                config.material_electron_response_index_sha256);
            std::vector<std::pair<int,double>> demand;
            for(std::size_t v=0;v<grid.material_id.size();++v)
                demand.emplace_back(grid.material_id[v],grid.density_g_per_cm3[v]);
            std::sort(demand.begin(),demand.end());demand.erase(std::unique(demand.begin(),demand.end()),demand.end());
            // Load both measured brackets in the same Schneider section.
            // required_tables rejects uncovered densities; no material fallback.
            response_index.required_tables(demand);
            const bool mapped=config.material_electron_response_memory_mode=="host_mapped";
            const auto budget=static_cast<std::size_t>(mapped?config.material_electron_response_host_budget_MiB:
                config.material_electron_response_device_budget_MiB)*1024*1024;
            material_electron_bank.emplace(queue,mapped?MaterialElectronMemory::host_mapped:MaterialElectronMemory::device);
            material_electron_bank->load(response_index,demand,
                budget-10*sizeof(std::uint64_t)-sizeof(MaterialPacketFailure),true,
                config.material_electron_short_range_mm>0);
            material_electron_views=material_electron_bank->views();material_electron_view_count=material_electron_bank->size();
            electron_joint_diag_device=mem_tracker.allocate<std::uint64_t>(7);
            material_untracked_device=mem_tracker.allocate<std::uint64_t>(3);
            material_failure_device=mem_tracker.allocate<MaterialPacketFailure>(1);
            if(config.material_electron_short_range_mm>0) {
                short_range_hits_device=mem_tracker.allocate<std::uint64_t>(1);
                if(!short_range_hits_device)throw std::bad_alloc();
                queue.fill(short_range_hits_device,std::uint64_t{0},1).wait_and_throw();
            }
            if(!electron_joint_diag_device || !material_untracked_device)throw std::bad_alloc();
            queue.fill(electron_joint_diag_device,std::uint64_t{0},7).wait_and_throw();
            if(!material_failure_device)throw std::bad_alloc();
            queue.fill(material_untracked_device,std::uint64_t{0},3).wait_and_throw();
        }
        if (use_longitudinal_interface_mass || (use_electron_joint && !config.ct_electron_joint_patient_experiment))
            validate_longitudinal_interface_grid(grid.density_g_per_cm3, grid.material_id);
        else if(use_schneider_delta_longitudinal)
            reject_unvalidated_longitudinal_heterogeneity(grid.density_g_per_cm3, grid.material_id);
        if (config.ct_longitudinal_homogeneous_density_diagnostic) {
            schneider_long_diagnostic_density = longitudinal_probe_density(
                grid.density_g_per_cm3, grid.material_id);
        }
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
        ct_density_device = mem_tracker.allocate<float>(voxel_count);
        ct_material_device = mem_tracker.allocate<std::uint8_t>(voxel_count);
        queue.copy(grid.density_g_per_cm3.data(), ct_density_device, voxel_count);
        queue.copy(grid.material_id.data(), ct_material_device, voxel_count).wait_and_throw();

        if (use_schneider_delta_tail) {
            if (grid.file_version < CtGrid::version_v2 || grid.material_id.size() != voxel_count) {
                throw std::runtime_error(
                    "Schneider delta-tail requires exact Schneider section IDs");
            }
            const auto aligned =
                config.voxel_bins_x == grid.nx && config.voxel_bins_y == grid.ny &&
                config.voxel_bins_z == grid.nz &&
                std::abs(config.voxel_size_x_mm - grid.spacing_x_mm) < 1.0e-6 &&
                std::abs(config.voxel_size_y_mm - grid.spacing_y_mm) < 1.0e-6 &&
                std::abs(config.voxel_size_z_mm - grid.spacing_z_mm) < 1.0e-6 &&
                std::abs(grid.origin_z_mm) < 1.0e-6;
            if (!aligned) {
                throw std::runtime_error(
                    "Schneider delta-tail requires a scorer exactly aligned to the CCTG grid");
            }
            // Grid edges are not material interfaces: inspect existing neighbours only.
            // Sampled endpoints still use the scorer-escape and material checks below.
            std::vector<std::uint8_t> eligible(voxel_count, 0U);
            const auto min_spacing = std::min({grid.spacing_x_mm, grid.spacing_y_mm,
                                               grid.spacing_z_mm});
            for (std::uint32_t iz = 0; iz < grid.nz; ++iz) {
                for (std::uint32_t iy = 0; iy < grid.ny; ++iy) {
                    for (std::uint32_t ix = 0; ix < grid.nx; ++ix) {
                        const auto index = ct_linear_index(ix, iy, iz, grid.nx, grid.ny);
                        if (grid.material_id[index] != 0U) continue;
                        bool clear = true;
                        if (grid.spacing_x_mm <= min_spacing * 1.001F) {
                            clear = clear && (ix == 0 || grid.material_id[index - 1] == 0U) &&
                                    (ix + 1 == grid.nx || grid.material_id[index + 1] == 0U);
                        }
                        if (grid.spacing_y_mm <= min_spacing * 1.001F) {
                            clear = clear &&
                                (iy == 0 || grid.material_id[index - grid.nx] == 0U) &&
                                (iy + 1 == grid.ny || grid.material_id[index + grid.nx] == 0U);
                        }
                        if (grid.spacing_z_mm <= min_spacing * 1.001F) {
                            const auto plane = static_cast<std::size_t>(grid.nx) * grid.ny;
                            clear = clear && (iz == 0 || grid.material_id[index - plane] == 0U) &&
                                    (iz + 1 == grid.nz || grid.material_id[index + plane] == 0U);
                        }
                        eligible[index] = clear ? 1U : 0U;
                    }
                }
            }
            schneider_delta_source_eligible_device =
                mem_tracker.allocate<std::uint8_t>(voxel_count);
            if (schneider_delta_source_eligible_device == nullptr) throw std::bad_alloc();
            queue.copy(eligible.data(), schneider_delta_source_eligible_device,
                       voxel_count).wait_and_throw();
        }

        use_ct_mass_sp = ct_material_ids_are_schneider_sections;
        use_ct_material_sp = !config.ct_water_stopping_power_file.empty() ||
                             !config.ct_bone_stopping_power_file.empty();
        use_ct_material_xs = !config.ct_bone_cross_section_file.empty() ||
                             !config.ct_schneider_cross_section_file.empty();

        use_schneider_primary_xs =
            ((ct_material_ids_are_schneider_sections && grid.mass_sp_za_rel.size() == 25) ||
             !config.ct_schneider_file.empty() ||
             !config.ct_schneider_cross_section_file.empty() ||
             config.is_schneider_ct_mode()) &&
            config.nuclear_model != "none";

        if (use_schneider_primary_xs) {
            if (config.primary_atomic_number != 6 || config.primary_mass_number != 12) {
                throw std::runtime_error(
                    "Schneider primary cross section is validated for C12 (Z=6, A=12) primaries only, got Z=" +
                    std::to_string(config.primary_atomic_number) + ", A=" + std::to_string(config.primary_mass_number));
            }

            // Validate all voxel material_id < 25 before kernel launch
            for (std::size_t i = 0; i < grid.material_id.size(); ++i) {
                if (grid.material_id[i] >= SchneiderResampledCrossSectionGrid::kExpectedSections) {
                    throw std::runtime_error(
                        "Invalid Schneider material_id " +
                        std::to_string(static_cast<unsigned>(grid.material_id[i])) +
                        " at voxel " + std::to_string(i) + " (must be < 25)");
                }
            }

            // v3 primary rate: the hazard comes from the masked rate-binary
            // partials (single source with the target sampler), so no CSV XS
            // table is loaded. The CSV key MUST be empty for v3 (fail-fast on
            // v1/v2.1 mixing is enforced at startup); v1 keeps this path.
            const std::filesystem::path v3_primary_rate_probe =
                config.ct_schneider_primary_rate_file;
            const bool primary_rate_is_v3 =
                std::filesystem::exists(v3_primary_rate_probe) &&
                schneider_rate_binary_version(v3_primary_rate_probe) == 3;
            if (primary_rate_is_v3) {
                if (!config.ct_schneider_cross_section_file.empty()) {
                    throw std::runtime_error(
                        "v3 primary rate requires empty ct_schneider_cross_section_file "
                        "(masked-binary hazard; refusing CSV/v3 mixing)");
                }
                if (config.is_primary_attenuation_only_mode()) {
                    throw std::runtime_error(
                        "primary-attenuation-only mode requires the CSV XS table; v3 has none");
                }
                std::cout << "[schneider-primary-xs] mode=primary-c12-masked-binary-v3 (no CSV)\n";
            } else {
                const auto schneider_host_grid = prepare_schneider_primary_xs(config);
                schneider_xs_sections =
                    static_cast<std::uint32_t>(SchneiderResampledCrossSectionGrid::kExpectedSections);
                schneider_xs_energies =
                    static_cast<std::uint32_t>(schneider_host_grid.energy_nodes());
                schneider_xs_e_min = static_cast<float>(schneider_host_grid.transport_energies_MeVu.front());
                const float dE = static_cast<float>(
                    schneider_host_grid.transport_energies_MeVu[1] - schneider_host_grid.transport_energies_MeVu[0]);
                schneider_xs_inv_dE = 1.0F / dE;

                const std::size_t total_elements =
                    static_cast<std::size_t>(schneider_xs_sections) * schneider_xs_energies;
                if (schneider_host_grid.mass_xs_per_mm_at_1g_cm3.size() != total_elements) {
                    throw std::runtime_error("Schneider cross section host payload size mismatch");
                }
                if (schneider_xs_sections != 25) {
                    throw std::runtime_error("schneider_xs_sections must be exactly 25");
                }
                if (schneider_xs_energies != schneider_host_grid.energy_nodes()) {
                    throw std::runtime_error("schneider_xs_energies must match grid size");
                }

                schneider_primary_xs_device = mem_tracker.allocate<float>(total_elements);
                if (schneider_primary_xs_device == nullptr) {
                    throw std::bad_alloc();
                }
                queue.copy(schneider_host_grid.mass_xs_per_mm_at_1g_cm3.data(),
                           schneider_primary_xs_device, total_elements).wait_and_throw();

                // Log table dimensions, byte count, source file, mode, and source SHA256 once
                std::cout << "[schneider-primary-xs] mode=primary-c12-section-resolved\n"
                          << "  sections=" << schneider_xs_sections << "\n"
                          << "  energies=" << schneider_xs_energies << "\n"
                          << "  bytes=" << total_elements * sizeof(float) << "\n"
                          << "  E_min=" << schneider_xs_e_min << " MeV/u\n"
                          << "  inv_dE=" << schneider_xs_inv_dE << "\n"
                          << "  source=" << config.ct_schneider_cross_section_file << "\n"
                          << "  source_sha256=" << (early_verified_sha256.empty() ? "unknown" : early_verified_sha256) << "\n";
            }
            if (config.is_primary_attenuation_only_mode()) {
                std::cout << "[ct-validation-mode] primary-attenuation-only\n"
                          << "  verified_source_sha256=" << early_verified_sha256 << "\n";
            }
        }

        if (ct_material_ids_are_schneider_sections) {
            // 1. Fail-closed Projectile Guard: Table v1 is validated exclusively for C12 (Z=6, A=12)
            if (config.primary_atomic_number != 6 || config.primary_mass_number != 12) {
                throw std::runtime_error(
                    "Schneider stopping power table v1 is currently validated exclusively for C12 (Z=6, A=12); "
                    "received primary ion Z=" + std::to_string(config.primary_atomic_number) +
                    ", A=" + std::to_string(config.primary_mass_number));
            }

            // 2. Fail-closed Energy Domain Guard: Primary birth energies must be within [0.01, 430.0] MeV/u
            const double initial_e = config.initial_energy_MeVu;
            if (!std::isfinite(initial_e) || initial_e < kSchneiderStoppingEnergyMin || initial_e > 430.0 + 1e-5) {
                throw std::invalid_argument(
                    "Schneider stopping power mode requires finite initial_energy_MeVu in [" +
                    std::to_string(kSchneiderStoppingEnergyMin) + ", 430.0] MeV/u, got " +
                    std::to_string(initial_e));
            }
            if (!std::isfinite(config.beam_energy_spread) || config.beam_energy_spread < 0.0) {
                throw std::invalid_argument("Schneider stopping power mode requires finite beam_energy_spread >= 0.0");
            }
            constexpr double kMaxGaussianSupport = 7.433851508; // sqrt(-2 ln 1e-12)
            if (config.beam_energy_spread > 0.0) {
                const double max_possible_e = initial_e * (1.0 + kMaxGaussianSupport * config.beam_energy_spread);
                const double min_possible_e = initial_e * (1.0 - kMaxGaussianSupport * config.beam_energy_spread);
                if (max_possible_e > kSchneiderStoppingEnergyMax) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: beam energy spread allows birth energies up to " +
                        std::to_string(max_possible_e) + " MeV/u, exceeding table maximum " +
                        std::to_string(kSchneiderStoppingEnergyMax) + " MeV/u");
                }
                if (min_possible_e < kSchneiderStoppingEnergyMin) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: beam energy spread allows birth energies down to " +
                        std::to_string(min_possible_e) + " MeV/u, below table minimum " +
                        std::to_string(kSchneiderStoppingEnergyMin) + " MeV/u");
                }
            }

            // Audit all primary spots in spot batch
            for (std::size_t si = 0; si < config.primary_spot_batch.size(); ++si) {
                const auto& spot = config.primary_spot_batch[si];
                const double spot_e = static_cast<double>(spot.floats[0]) / 12.0; // total MeV to MeV/u for C12
                if (!std::isfinite(spot_e) || spot_e < kSchneiderStoppingEnergyMin || spot_e > 430.0 + 1e-5) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: spot " + std::to_string(si) +
                        " energy " + std::to_string(spot_e) + " MeV/u is outside valid domain [" +
                        std::to_string(kSchneiderStoppingEnergyMin) + ", 430.0] MeV/u");
                }
                const double spot_spread = static_cast<double>(spot.floats[1]);
                if (!std::isfinite(spot_spread) || spot_spread < 0.0) {
                    throw std::invalid_argument(
                        "Schneider stopping power mode: spot " + std::to_string(si) +
                        " energy spread must be finite and >= 0.0");
                }
                if (spot_spread > 0.0) {
                    const double max_spot_e = spot_e * (1.0 + kMaxGaussianSupport * spot_spread);
                    const double min_spot_e = spot_e * (1.0 - kMaxGaussianSupport * spot_spread);
                    if (max_spot_e > kSchneiderStoppingEnergyMax) {
                        throw std::invalid_argument(
                            "Schneider stopping power mode: spot " + std::to_string(si) +
                            " energy spread allows birth energies up to " +
                            std::to_string(max_spot_e) + " MeV/u, exceeding table maximum " +
                            std::to_string(kSchneiderStoppingEnergyMax) + " MeV/u");
                    }
                    if (min_spot_e < kSchneiderStoppingEnergyMin) {
                        throw std::invalid_argument(
                            "Schneider stopping power mode: spot " + std::to_string(si) +
                            " energy spread allows birth energies down to " +
                            std::to_string(min_spot_e) + " MeV/u, below table minimum " +
                            std::to_string(kSchneiderStoppingEnergyMin) + " MeV/u");
                    }
                }
            }

            if (std::abs(config.ct_stopping_power_scale - 1.0) > 1e-6) {
                throw std::invalid_argument(
                    "Exact Schneider stopping power mode requires ct_stopping_power_scale == 1.0; calibration scaling is forbidden");
            }

            auto stopping_file = config.ct_schneider_stopping_power_file;
            if (stopping_file.empty()) {
                const auto default_stopping_bin = std::filesystem::path("data/schneider/schneider_stopping_v1.bin");
                if (std::filesystem::exists(default_stopping_bin)) {
                    stopping_file = default_stopping_bin;
                } else {
                    throw std::runtime_error(
                        "Schneider CT grid requires ct_schneider_stopping_power_file; silent fallback to legacy SP is strictly forbidden");
                }
            }
            if (!std::filesystem::exists(stopping_file)) {
                throw std::runtime_error(
                    "Schneider stopping power file missing: " + stopping_file.string());
            }

            // Verify all voxel material IDs are strictly in [0, 24]
            for (std::size_t vi = 0; vi < grid.material_id.size(); ++vi) {
                if (grid.material_id[vi] >= kSchneiderStoppingNumSections) {
                    throw std::runtime_error(
                        "Schneider CT grid contains invalid material ID " + std::to_string(grid.material_id[vi]) +
                        " at voxel " + std::to_string(vi) + " (must be in [0, 24])");
                }
            }

            const auto meta_path = stopping_file.parent_path() /
                                   (stopping_file.stem().string() + ".metadata.json");
            const auto stopping_table = SchneiderStoppingTable::from_binary(
                stopping_file, meta_path);
            schneider_sp_sections = static_cast<std::uint32_t>(stopping_table.num_sections());
            schneider_sp_energies = static_cast<std::uint32_t>(stopping_table.num_energies());
            schneider_sp_e_min = static_cast<float>(stopping_table.energy_min_mevu());
            schneider_sp_e_max = static_cast<float>(stopping_table.energy_max_mevu());
            const float dE = static_cast<float>(stopping_table.energy_step_mevu());
            schneider_sp_inv_dE = 1.0F / dE;

            if (schneider_sp_sections != 25 || schneider_sp_energies != 4302) {
                throw std::runtime_error(
                    "Schneider stopping table dimension mismatch: sections=" + std::to_string(schneider_sp_sections) +
                    ", energies=" + std::to_string(schneider_sp_energies) + " (expected 25x4302)");
            }

            const auto flat_sp = stopping_table.to_flat_mass_stopping_float();
            const std::size_t total_sp_elements = flat_sp.size();
            schneider_stopping_device = mem_tracker.allocate<float>(total_sp_elements);
            if (schneider_stopping_device == nullptr) {
                throw std::bad_alloc();
            }
            queue.copy(flat_sp.data(), schneider_stopping_device, total_sp_elements).wait_and_throw();
            use_schneider_stopping = true;

            if (!config.ct_secondary_ion_section_stopping_file.empty()) {
                const auto ion_meta_path =
                    config.ct_secondary_ion_section_stopping_file.parent_path() /
                    (config.ct_secondary_ion_section_stopping_file.stem().string() +
                     ".metadata.json");
                if (compute_file_sha256_hex(ion_meta_path) !=
                    config.ct_secondary_ion_section_stopping_metadata_sha256)
                    throw std::runtime_error(
                        "Section ion stopping metadata SHA256 mismatch");
                if (compute_file_sha256_hex(
                        config.ct_secondary_ion_section_stopping_file) !=
                    config.ct_secondary_ion_section_stopping_sha256)
                    throw std::runtime_error(
                        "Section ion stopping data SHA256 mismatch");
                SchneiderIonStoppingTable ion_table = SchneiderIonStoppingTable::from_binary(
                    config.ct_secondary_ion_section_stopping_file, ion_meta_path);
                const auto flat_ion = ion_table.to_flat_float();
                schneider_ion_sp_device = mem_tracker.allocate<float>(flat_ion.size());
                if (schneider_ion_sp_device == nullptr) {
                    throw std::bad_alloc();
                }
                queue.copy(flat_ion.data(), schneider_ion_sp_device, flat_ion.size())
                    .wait_and_throw();
                use_schneider_ion_sp = true;
                ion_stopping_failures = mem_tracker.allocate<std::uint32_t>(8);
                queue.fill(ion_stopping_failures, std::uint32_t{0}, 8).wait_and_throw();
                std::cout << "[schneider-ion-stopping] mode=section-ion-v1 sections=25 "
                          << "species=18 energies=" << kSchneiderIonEnergies << " source="
                          << config.ct_secondary_ion_section_stopping_file << '\n';
            }

            std::cout << "[schneider-stopping-power] mode=exact-schneider-v1\n"
                      << "  sections=" << schneider_sp_sections << "\n"
                      << "  energies=" << schneider_sp_energies << "\n"
                      << "  bytes=" << total_sp_elements * sizeof(float) << "\n"
                      << "  E_min=" << schneider_sp_e_min << " MeV/u\n"
                      << "  E_max=" << schneider_sp_e_max << " MeV/u\n"
                      << "  inv_dE=" << schneider_sp_inv_dE << "\n"
                      << "  source=" << config.ct_schneider_stopping_power_file << "\n"
                      << "  source_sha256=" << (early_verified_stopping_sha256.empty() ? "unknown" : early_verified_stopping_sha256) << "\n";
        }

        // Exact C12 stopping only replaces the PRIMARY lookup. Secondary ions
        // still need their 25-section material factors; previously this guard
        // also suppressed their LUT, leaving water stopping times density.
        if (use_ct_mass_sp && (!use_schneider_stopping || config.ct_secondary_schneider_sp_diagnostic)) {
            if (config.ct_secondary_schneider_sp_diagnostic) {
                if (!ct_material_ids_are_schneider_sections || grid.mass_sp_za_rel.size()!=25 || grid.mass_sp_I_eV.size()!=25)
                    throw std::invalid_argument("Secondary Schneider stopping requires 25 exact section Z/A and I entries");
                for (std::size_t s=0;s<25;++s)
                    if (!std::isfinite(grid.mass_sp_za_rel[s]) || grid.mass_sp_za_rel[s]<=0 ||
                        !std::isfinite(grid.mass_sp_I_eV[s]) || grid.mass_sp_I_eV[s]<=0)
                        throw std::invalid_argument("Invalid secondary Schneider material parameters");
                std::cout << "[secondary-schneider-stopping] diagnostic: 25-section Z/A,I Bethe factors; exact primary table unchanged; no four-class LUT\n";
            }
            ct_n_mass_factors = static_cast<std::uint32_t>(grid.mass_sp_za_rel.size());
            std::vector<float> mass_factor_lut(
                static_cast<std::size_t>(ct_n_mass_factors) * table_size);
            const auto sp_scale = static_cast<float>(config.ct_stopping_power_scale);

            const auto try_density_spr = [&]() -> bool {
                // This diagnostic is exclusively Schneider-25. Do not enter
                // the legacy air/lung/bone density-LUT branch below.
                if (config.ct_secondary_schneider_sp_diagnostic) return false;
                if (!config.ct_use_density_mass_spr) {
                    return false;
                }
                const auto resolve = [](const std::filesystem::path& user_path,
                                        const std::filesystem::path& fallback) {
                    if (!user_path.empty()) {
                        return std::filesystem::exists(user_path) ? user_path : std::filesystem::path{};
                    }
                    return std::filesystem::exists(fallback) ? fallback : std::filesystem::path{};
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
                if (config.is_primary_attenuation_only_mode()) {
                    throw std::runtime_error(
                        "primary-attenuation-only mode requires density-mass-SPR LUT to be successfully constructed without fallback");
                }
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
            ct_mass_sp_factor_lut_device = mem_tracker.allocate<float>(lut_bytes);
            ct_mass_sp_za_rel_device = mem_tracker.allocate<float>(ct_n_mass_factors);
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
            mem_tracker.allocate<PrimarySpotBatchEntry>(primary_spot_count);
        queue.copy(config.primary_spot_batch.data(), primary_spots_device, primary_spot_count)
            .wait_and_throw();
    }

    const bool use_all_elastic = !config.all_ion_elastic_file.empty();
    AllIonElasticView all_elastic{};
    ElasticRecoilStoppingView recoil_stopping{};
    float* elastic_energy = nullptr; // local, queued charged, overflow
    std::uint32_t* elastic_audit = nullptr; // primary, secondary, bad query, unsupported recoil
    if(use_all_elastic) {
        const auto stop=ElasticRecoilStoppingTable::load(config.elastic_recoil_stopping_file,config.elastic_recoil_stopping_sha256);
        auto* keys=mem_tracker.allocate<std::int32_t>(stop.keys.size());
        auto* se=mem_tracker.allocate<float>(stop.energies.size());auto* sv=mem_tracker.allocate<float>(stop.values.size());
        if(!keys||!se||!sv)throw std::bad_alloc();
        queue.copy(stop.keys.data(),keys,stop.keys.size());queue.copy(stop.energies.data(),se,stop.energies.size());queue.copy(stop.values.data(),sv,stop.values.size()).wait_and_throw();
        recoil_stopping={keys,se,sv,static_cast<std::uint32_t>(stop.keys.size()/2),static_cast<std::uint32_t>(stop.energies.size())};
        const auto bank=AllIonElasticTable::load(config.all_ion_elastic_file,config.all_ion_elastic_sha256);
        auto* e=mem_tracker.allocate<float>(bank.energies.size());
        auto* r=mem_tracker.allocate<float>(bank.rates.size());
        auto* a=mem_tracker.allocate<ElasticSample>(bank.samples.size());
        auto* m=mem_tracker.allocate<double>(18);
        elastic_audit=mem_tracker.allocate<std::uint32_t>(12);
        elastic_energy=mem_tracker.allocate<float>(3);
        if(!elastic_energy)throw std::bad_alloc();
        queue.fill(elastic_energy,0.0F,3);
        if(!e||!r||!a||!m||!elastic_audit)throw std::bad_alloc();
        queue.copy(bank.energies.data(),e,bank.energies.size());queue.copy(bank.rates.data(),r,bank.rates.size());
        queue.copy(bank.samples.data(),a,bank.samples.size());queue.copy(bank.masses.data(),m,18);
        queue.fill(elastic_audit,std::uint32_t{0},12).wait_and_throw();
        all_elastic={e,r,a,m,static_cast<std::uint32_t>(bank.energies.size()),bank.nq};
    }
    const SchneiderCtDeviceContext schneider_ct_device_ctx = upload_schneider_ct_device_context(
        queue, mem_tracker, config,
        schneider_stopping_device,
        schneider_sp_sections, schneider_sp_energies,
        schneider_sp_e_min, schneider_sp_e_max, schneider_sp_inv_dE);
    if (schneider_ct_device_ctx.uses_cinel03() &&
        !config.is_primary_attenuation_only_mode()) {
        schneider_diag_device = mem_tracker.allocate<std::uint64_t>(kSchneiderDiagSlots);
        schneider_float_device = mem_tracker.allocate<float>(kSchneiderFloatSlots);
        schneider_miss_device =
            mem_tracker.allocate<SchneiderMissRecord>(kSchneiderMissLogCap);
        schneider_track_log_device =
            mem_tracker.allocate<SchneiderUnsupportedTrack>(kSchneiderTrackLogCap);
        schneider_miss_count_device = mem_tracker.allocate<std::uint32_t>(2);
        schneider_track_count_device = mem_tracker.allocate<std::uint32_t>(2);
        if (schneider_diag_device == nullptr || schneider_float_device == nullptr ||
            schneider_miss_device == nullptr || schneider_track_log_device == nullptr ||
            schneider_miss_count_device == nullptr || schneider_track_count_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(schneider_diag_device, std::uint64_t{0}, kSchneiderDiagSlots)
            .wait_and_throw();
        queue.fill(schneider_float_device, 0.0F, kSchneiderFloatSlots).wait_and_throw();
        // [0] = written count, [1] = dropped (over-capacity) count.
        queue.fill(schneider_miss_count_device, std::uint32_t{0}, 2).wait_and_throw();
        queue.fill(schneider_track_count_device, std::uint32_t{0}, 2).wait_and_throw();
    }

    // Scorers & Result buffers
    const auto cinel02_max_secondary_inelastic_generations =
        config.cinel02_max_secondary_inelastic_generations;
    constexpr bool cinel02_topas_compatibility_mode = false;
    const auto enable_voxel_scoring = config.enable_voxel_scoring;
    const auto enable_let_scoring = EmMode!=1 && config.enable_let_scoring;
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

    auto* dose_device = mem_tracker.allocate<DepthAtomicT>(number_of_bins);
    std::uint64_t* primary_survival_device = nullptr;
    std::uint64_t* inelastic_reaction_device = nullptr;
    // Diagnostic-only: fragment-species scoring also needs per-depth
    // survival/reaction counts. No transport effect.
    if (config.needs_primary_survival_buffers()) {
        primary_survival_device =
            mem_tracker.allocate<std::uint64_t>(number_of_bins);
        inelastic_reaction_device =
            mem_tracker.allocate<std::uint64_t>(number_of_bins);
        if (primary_survival_device == nullptr || inelastic_reaction_device == nullptr) {
            throw std::bad_alloc();
        }
        queue.fill(primary_survival_device, std::uint64_t{0}, number_of_bins);
        queue.fill(inelastic_reaction_device, std::uint64_t{0}, number_of_bins);
    }
    auto* voxel_dose_device = enable_voxel_scoring
                                  ? mem_tracker.allocate<DoseAtomicT>(number_of_voxels)
                                  : nullptr;
    const auto enable_primary_voxel_fluence =
        !config.primary_voxel_fluence_mhd_output_file.empty();
    auto* primary_voxel_track_length_device =
        enable_primary_voxel_fluence
            ? mem_tracker.allocate<DoseAtomicT>(number_of_voxels)
            : nullptr;
    const auto enable_charged_origin_voxel_scoring =
        config.enable_charged_origin_voxel_scoring;
    auto* he4_hazard_audit_device = config.fragment_birth_spectrum_output_file.empty()
        ? nullptr : mem_tracker.allocate<double>(6);
    if (!config.fragment_birth_spectrum_output_file.empty()) {
        if (!he4_hazard_audit_device) throw std::bad_alloc();
        queue.memset(he4_hazard_audit_device, 0, 6 * sizeof(double)).wait_and_throw();
    }
    auto* charged_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(charged_origin_category_count * number_of_voxels)
            : nullptr;
    auto* be_isotope_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(be_isotope_origin_category_count * number_of_voxels)
            : nullptr;
    auto* he_isotope_origin_voxel_dose_device =
        enable_charged_origin_voxel_scoring
            ? mem_tracker.allocate<DoseAtomicT>(he_isotope_origin_category_count * number_of_voxels)
            : nullptr;
    auto* let_moments_device = enable_let_scoring
                                   ? mem_tracker.allocate<LetAtomicT>(4 * number_of_bins)
                                   : nullptr;
    auto* voxel_let_moments_device =
        enable_let_scoring && enable_voxel_scoring
            ? mem_tracker.allocate<LetAtomicT>(4 * number_of_voxels)
            : nullptr;

    const bool is_primary_attenuation_only = config.is_primary_attenuation_only_mode();
    const auto enable_inelastic = config.enable_inelastic;
    if (enable_inelastic && !use_schneider_primary_xs) {
        throw std::runtime_error("Nuclear transport requires the validated Schneider/unified-water CINEL03 path");
    }
    const auto enable_secondary_transport = config.enable_secondary_transport;

    auto* deposited_device = mem_tracker.allocate<float>(number_of_histories);
    auto* sampled_incident_device = mem_tracker.allocate<float>(number_of_histories);
    auto* escaped_device = mem_tracker.allocate<float>(number_of_histories);
    auto* steps_device = mem_tracker.allocate<std::uint32_t>(number_of_histories);
    auto* untracked_nuclear_device = mem_tracker.allocate<float>(number_of_histories);
    auto* other_terminal_energy_device = mem_tracker.allocate<float>(number_of_histories);
    auto* cutoff_stopped_energy_device = mem_tracker.allocate<float>(number_of_histories);
    auto* schneider_inelastic_device =
        use_schneider_primary_xs ? mem_tracker.allocate<std::uint64_t>(1) : nullptr;
    auto* primary_terminal_counts_device = mem_tracker.allocate<std::uint64_t>(4);

    if (deposited_device == nullptr || sampled_incident_device == nullptr || escaped_device == nullptr || steps_device == nullptr ||
        untracked_nuclear_device == nullptr || other_terminal_energy_device == nullptr ||
        cutoff_stopped_energy_device == nullptr || primary_terminal_counts_device == nullptr ||
        (use_schneider_primary_xs && schneider_inelastic_device == nullptr)) {
        free_device(deposited_device);
        free_device(sampled_incident_device);
        free_device(escaped_device);
        free_device(steps_device);
        free_device(untracked_nuclear_device);
        free_device(other_terminal_energy_device);
        free_device(cutoff_stopped_energy_device);
        free_device(schneider_inelastic_device);
        free_device(primary_terminal_counts_device);
        throw std::bad_alloc();
    }

    queue.fill(untracked_nuclear_device, 0.0F, number_of_histories);
    queue.fill(other_terminal_energy_device, 0.0F, number_of_histories);
    queue.fill(cutoff_stopped_energy_device, 0.0F, number_of_histories);
    if (schneider_inelastic_device != nullptr) {
        queue.fill(schneider_inelastic_device, std::uint64_t{0}, 1);
    }
    queue.fill(primary_terminal_counts_device, std::uint64_t{0}, 4).wait_and_throw();

    const bool record_first_interactions =
        is_primary_attenuation_only || config.validation_scorers();
    PrimaryFirstInteractionRecord* first_interactions_device = nullptr;
    std::uint32_t* first_interactions_count_device = nullptr;
    if (record_first_interactions) {
        first_interactions_device = mem_tracker.allocate<PrimaryFirstInteractionRecord>(number_of_histories);
        first_interactions_count_device = mem_tracker.allocate<std::uint32_t>(1);
        if (first_interactions_device == nullptr || first_interactions_count_device == nullptr) {
            free_device(primary_terminal_counts_device);
            free_device(deposited_device);
            free_device(sampled_incident_device);
            free_device(escaped_device);
            free_device(steps_device);
            free_device(untracked_nuclear_device);
            free_device(other_terminal_energy_device);
            free_device(cutoff_stopped_energy_device);
            free_device(schneider_inelastic_device);
            free_device(first_interactions_device);
            free_device(first_interactions_count_device);
            throw std::bad_alloc();
        }
        queue.fill(first_interactions_count_device, 0U, 1).wait_and_throw();
    }

    const bool need_secondary_buffers =
        !is_primary_attenuation_only && enable_inelastic;

    constexpr std::size_t max_secondaries = 32000000;
    auto* secondary_queue_device =
        need_secondary_buffers
            ? mem_tracker.allocate<SecondaryParticle>(max_secondaries)
            : nullptr;
    auto* secondary_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    uint32_t* secondary_overflow_count_device =
        need_secondary_buffers
            ? mem_tracker.allocate<uint32_t>(1)
            : nullptr;
    float* secondary_overflow_energy_device =
        need_secondary_buffers
            ? mem_tracker.allocate<float>(1)
            : nullptr;
    if (need_secondary_buffers) {
        if (!secondary_queue_device || !secondary_count_device ||
            !secondary_overflow_count_device || !secondary_overflow_energy_device)
            throw std::bad_alloc();
        queue.fill(secondary_count_device, 0U, 1);
        queue.fill(secondary_overflow_count_device, 0U, 1);
        queue.fill(secondary_overflow_energy_device, 0.0F, 1).wait_and_throw();
    }

    float* fluct_energy_device = nullptr;
    float* fluct_density_device = nullptr;
    float* fluct_probability_device = nullptr;
    float* fluct_quantile_device = nullptr;
    std::size_t fluct_energy_count = 0;
    std::size_t fluct_density_count = 0;
    std::size_t fluct_probability_count = 0;
    const bool fluct_fraction_hybrid = config.energy_straggling_model == "packaged_fluctuation_fraction_hybrid";
    const bool fluct_fraction_axis = config.energy_straggling_model == "packaged_fluctuation_fraction" || fluct_fraction_hybrid;
    std::uint32_t* fluct_domain_failures = nullptr;
    std::uint64_t* primary_loss_query_audit = nullptr;
    if (config.enable_primary_loss_query_audit) {
        primary_loss_query_audit = mem_tracker.allocate<std::uint64_t>(28);
        if (!primary_loss_query_audit) throw std::bad_alloc();
        queue.fill(primary_loss_query_audit, std::uint64_t{0}, 28).wait_and_throw();
    }
    if (config.uses_packaged_fluctuation()) {
        const auto host = carbon::EnergyLossFluctuationTable::from_csv(
            config.energy_straggling_package_file, fluct_fraction_axis);
        if (host.projectile_atomic_number() != 6 ||
            host.projectile_mass_number() != 12 ||
            host.material_name() != (fluct_fraction_axis ? "Water_75eV" : "G4_WATER"))
            throw std::runtime_error(
                "Packaged fluctuation must describe C-12 in G4_WATER");
        const auto to_float = [](const std::vector<double>& input) {
            std::vector<float> output(input.size());
            std::transform(input.begin(), input.end(), output.begin(),
                           [](double value) { return static_cast<float>(value); });
            return output;
        };
        const auto energies = to_float(host.energies_MeVu());
        const auto densities = to_float(host.second_axis_values());
        if (fluct_fraction_hybrid && densities.front() > 1.0e-4F)
            throw std::runtime_error("Hybrid fraction grid must cover f >= 1e-4");
        if (fluct_fraction_axis) {
            if (!use_unified_water || config.water_density_g_per_cm3 != 1.0)
                throw std::runtime_error("Fraction-axis candidate requires native homogeneous density-1 water");
            fluct_domain_failures = mem_tracker.allocate<std::uint32_t>(1);
            queue.fill(fluct_domain_failures, std::uint32_t{0}, 1).wait_and_throw();
        }
        const auto probabilities = to_float(host.probabilities());
        const auto quantiles = to_float(host.loss_ratio_quantiles());
        fluct_energy_count = energies.size();
        fluct_density_count = densities.size();
        fluct_probability_count = probabilities.size();
        fluct_energy_device = mem_tracker.allocate<float>(energies.size());
        fluct_density_device = mem_tracker.allocate<float>(densities.size());
        fluct_probability_device = mem_tracker.allocate<float>(probabilities.size());
        fluct_quantile_device = mem_tracker.allocate<float>(quantiles.size());
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

    float* ion_species_sp_device = nullptr;
    float* ion_energy_grid_device = nullptr;
    float* ion_csda_a1_device = nullptr;
    if (enable_inelastic) {
        // Native unified water must honor its explicit material-specific table.
        // Preserve the frozen CT/legacy route until separately validated.
        std::filesystem::path ion_sp_path = use_unified_water
            ? config.particle_stopping_power_file
            : std::filesystem::path("data/ion_stopping_power_water_geant4_11_3_2.csv");
        if (use_unified_water) {
            if (ion_sp_path.empty() || !std::filesystem::exists(ion_sp_path)) {
                throw std::runtime_error("Unified water requires its configured ion stopping table");
            }
            // The device loader indexes rows by primary-table index. Reject
            // different energy grids before uploading rather than misindexing.
            (void)IonStoppingPowerTables::from_csv(ion_sp_path, stopping_power);
            std::cout << "Unified water active ion stopping table: " << ion_sp_path
                      << " SHA256=" << compute_file_sha256_hex(ion_sp_path) << '\n';
        }
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
        ion_species_sp_device = mem_tracker.allocate<float>(18 * table_size);
        queue.copy(ion_sp_lut.data(), ion_species_sp_device, 18 * table_size).wait_and_throw();
        std::vector<float> energy_grid_host(table_size);
        std::transform(stopping_power.energies().begin(), stopping_power.energies().end(),
                       energy_grid_host.begin(),
                       [](double value) { return static_cast<float>(value); });
        ion_energy_grid_device = mem_tracker.allocate<float>(table_size);
        ion_csda_a1_device = mem_tracker.allocate<float>(18 * table_size);
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
    // EM-only transport never allocates the ion-species grid. Use the
    // primary grid uploaded with its cumulative-range table in that case.
    const float* primary_csda_energy_grid_device = enable_inelastic
        ? ion_energy_grid_device : energy_grid_device;

    auto* in_fov_dose_device =
        enable_voxel_scoring ? mem_tracker.allocate<DepthAtomicT>(number_of_bins) : nullptr;
    if (in_fov_dose_device != nullptr) {
        queue.fill(in_fov_dose_device, DepthAtomicT{0}, number_of_bins).wait_and_throw();
    }

    if (dose_device == nullptr || deposited_device == nullptr ||
        escaped_device == nullptr || steps_device == nullptr ||
        (enable_voxel_scoring && (voxel_dose_device == nullptr || in_fov_dose_device == nullptr)) ||
        (enable_primary_voxel_fluence &&
         primary_voxel_track_length_device == nullptr) ||
        (enable_charged_origin_voxel_scoring &&
         (charged_origin_voxel_dose_device == nullptr ||
          be_isotope_origin_voxel_dose_device == nullptr ||
          he_isotope_origin_voxel_dose_device == nullptr)) ||
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

    queue.memset(dose_device, 0, number_of_bins * sizeof(DepthAtomicT));
    if (enable_voxel_scoring) {
        queue.memset(voxel_dose_device, 0, number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_primary_voxel_fluence) {
        queue.memset(primary_voxel_track_length_device, 0,
                     number_of_voxels * sizeof(DoseAtomicT));
    }
    if (enable_charged_origin_voxel_scoring) {
        queue.memset(charged_origin_voxel_dose_device, 0,
                     charged_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
        queue.memset(be_isotope_origin_voxel_dose_device, 0,
                     be_isotope_origin_category_count * number_of_voxels *
                         sizeof(DoseAtomicT));
        queue.memset(he_isotope_origin_voxel_dose_device, 0,
                     he_isotope_origin_category_count * number_of_voxels *
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
    const auto maximum_primary_steps = config.maximum_primary_steps;
    const auto energy_cutoff_MeV = static_cast<float>(
        use_schneider_stopping ? std::max(config.energy_cutoff_MeV, 12.0 * static_cast<double>(schneider_sp_e_min))
                               : config.energy_cutoff_MeV);

    if (is_cuda_backend && !config.enable_minibeam) {
        cuda_clock_warmup(queue, mem_tracker);
    }

    const auto enable_energy_straggling = config.enable_energy_straggling;
    const auto terminal_generation_em = config.enable_terminal_generation_em_transport;
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
    const auto active_water_radiation_length = use_unified_water
        ? schneider_ct_device_ctx.water_radiation_length_g_cm2 : water_radiation_length_g_per_cm2;
    const auto enable_multiple_scattering = config.enable_multiple_scattering;
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
    const double electron_short_range_mm=config.material_electron_short_range_mm;
    if(electron_short_range_mm>0)
        std::cout<<"[research-short-range] threshold_mm="<<electron_short_range_mm
                 <<" childless complete tail, strict same-voxel containment; not accuracy validated\n";
    const auto enable_csda_range_energy_loss = config.enable_csda_range_energy_loss;
    const auto primary_atomic_number = config.primary_atomic_number;
    const auto primary_rest_mass_MeV =
        static_cast<float>(config.resolved_primary_rest_mass_MeV());
    const auto minimum_table_energy = static_cast<float>(stopping_power.energies().front());
    const auto inverse_table_step =
        1.0f / static_cast<float>(stopping_power.energies()[1] - stopping_power.energies()[0]);

    runtime_setup.finish();
    RuntimeScope runtime_steps("transport_loop_including_scoring_and_queue_transfers");
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

                const auto sampled_initial_energy_MeVu = energy_MeV * inverse_mass_number;
                // Store the energy actually transported after source sampling/cutoff.
                // This does not consume RNG or change the source distribution.
                sampled_incident_device[global_history] = energy_MeV;
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
                const auto composed_direction = compose_tps_direction(
                    local_dx, local_dy, local_dz, ux_x, ux_y, ux_z,
                    uy_x, uy_y, uy_z, uz_x, uz_y, uz_z);
                auto direction_x = composed_direction.x;
                auto direction_y = composed_direction.y;
                auto direction_z = composed_direction.z;
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
                    if (position_z_mm < 0.0F && sycl::fabs(direction_z) > 1.0e-8F) {
                        const auto t_plane = -position_z_mm / direction_z;
                        position_x_mm += t_plane * direction_x;
                        position_y_mm += t_plane * direction_y;
                        position_z_mm = 0.0F;
                    }
                }

                auto history_deposited_MeV = 0.0F;
                auto history_water_electron_escaped_MeV=0.0F;
                std::uint64_t local_schneider_rate_queries = 0;
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

                float nuclear_tau_remaining = 0.0F;
                HadronicIncreasingCacheCandidate primary_hadronic_cache;
                bool nuclear_tau_active = false;
                std::uint32_t nuclear_tau_rng_step = 0;
                bool primary_inelastic_occurred = false;
                bool unified_primary_escaped_ct = false;
                UnifiedEmClock unified_primary_clock;
                UnifiedEmState unified_primary_state;
                std::array<std::uint64_t,8> unified_primary_audit{};
                std::uint64_t unified_primary_counter=0;
                const int unified_primary_species=unified_em?unified_device.species_index(primary_atomic_number,primary_mass_number):-1;

                std::uint32_t interaction_section = 0;
                float interaction_density = 0.0F;

                const std::uint32_t max_primary_steps = maximum_primary_steps;
                int last_survival_bin = -1;
                while (energy_MeV > energy_cutoff_MeV && steps < max_primary_steps) {
                    // The rate/target hazard is evaluated from the energy at
                    // the beginning of this step.  The replay event is
                    // queried after the continuous EM loss below, so retain
                    // both values for the compact replay ledger.
                    const auto primary_rate_query_energy_MeV =
                        sycl::fmax(0.0F, energy_MeV);
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
                                          ct_material, direction_x, direction_y, direction_z);
                        if (in_ct) {
                            local_density_g_per_cm3 = ct_rho;
                        }
                    }

                    // Mirror secondary CT escape: the EM package has no exterior material.
                    if(unified_em && enable_ct_grid && !in_ct) {
                        unified_primary_escaped_ct=true;
                        break;
                    }

                    const auto layer_for_material =
                        slab_layer_count > 0
                            ? slab_layer_index(position_z_mm, slab_z_ends_device, slab_layer_count)
                            : 0U;
                    float stopping_power_MeV_per_mm = 0.0F;
                    if (enable_ct_grid) {
                        if (use_schneider_stopping && in_ct) {
                            const auto floating_sp_index = (energy_MeVu - schneider_sp_e_min) * schneider_sp_inv_dE;
                            auto sp_index = static_cast<int>(sycl::floor(floating_sp_index));
                            sp_index = sycl::max(0, sycl::min(sp_index, static_cast<int>(schneider_sp_energies) - 2));
                            const auto sp_fraction = sycl::clamp(floating_sp_index - static_cast<float>(sp_index), 0.0F, 1.0F);

                            const auto sec_id = static_cast<std::size_t>(
                                sycl::min(static_cast<std::uint32_t>(ct_material), schneider_sp_sections - 1));
                            const auto base = sec_id * schneider_sp_energies;
                            const auto mass_sp = schneider_stopping_device[base + sp_index] +
                                                 sp_fraction * (schneider_stopping_device[base + sp_index + 1] -
                                                                schneider_stopping_device[base + sp_index]);
                            stopping_power_MeV_per_mm = mass_sp * sycl::fmax(local_density_g_per_cm3, 1.0e-6F);
                        } else {
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
                    step_mm = sycl::fmax(step_mm, 1.0e-5F);
                    UnifiedEmStep unified_primary_pre;
                    float unified_primary_rate=0,unified_primary_distance=std::numeric_limits<float>::infinity();
                    auto unified_uniform=[&](){return (rng::random_u32(spot_seed,rng_history,unified_primary_counter++,120)>>8)*0x1p-24f;};
                    auto unified_count=[&](int index,std::uint64_t count=1){
                        if constexpr(CARBON_EM_LOCAL_AUDIT) unified_primary_audit[index]+=count;
                            else {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(unified_audit[index]);a.fetch_add(count);
                            }
                    };
                    if(unified_em){
                        const int section=enable_ct_grid?(in_ct?int(ct_material):-2):-1;
                        if(!CARBON_EM_MATERIAL_CACHE || !unified_primary_state.valid || unified_primary_state.section!=section ||
                           unified_primary_state.density!=local_density_g_per_cm3)
                            unified_primary_state=unified_device.select(section,local_density_g_per_cm3,unified_primary_species);
                        if(!unified_primary_state.covers(energy_MeV)){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,1,section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,0);
                            unified_count(0);break;}
                        if(unified_primary_clock.last_section!=unified_primary_state.section || unified_primary_clock.last_density!=unified_primary_state.density)primary_hadronic_cache.valid=false;
                        // Preserve material cache invalidation without updating a delta clock.
                        unified_primary_clock.last_section=unified_primary_state.section;
                        unified_primary_clock.last_density=unified_primary_state.density;
                        unified_primary_pre=unified_primary_state.prepare(energy_MeV);
                        step_mm=sycl::fmin((em_primary_step_scale!=1.f?unified_primary_state.research_step(energy_MeV,CARBON_EM_STEP_CACHE?unified_primary_pre:unified_primary_state.prepare(energy_MeV),em_primary_step_scale):(CARBON_EM_STEP_CACHE?unified_primary_state.step(unified_primary_pre):unified_primary_state.step(energy_MeV))),unified_primary_distance);
                        // Bound combined mean loss, not the old restricted range alone.
                        const float delta_sp=unified_primary_state.mix(unified_primary_pre.lo.delta_stopping,unified_primary_pre.hi.delta_stopping);
                        if(delta_sp>0) step_mm=sycl::fmin(step_mm,.01f*energy_MeV/sycl::fmax(1e-12f,delta_sp+unified_primary_state.mix(unified_primary_pre.lo.stopping,unified_primary_pre.hi.stopping)));
                    }

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
                        if (slab_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, slab_step);
                        }
                    }
                    if (enable_hetero_insert) {
                        const auto insert_step = distance_to_insert_interface_mm(
                            position_x_mm, position_y_mm, position_z_mm, direction_x,
                            direction_y, direction_z, insert_x_min, insert_x_max, insert_y_min,
                            insert_y_max, insert_z_min, insert_z_max, phantom_length_mm);
                        if (insert_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, insert_step);
                        }
                    }
                    CtFaceClampResult ct_clamp_res{step_mm, false, 0};
                    if (enable_ct_grid && in_ct) {
                        ct_clamp_res = clamp_step_to_ct_faces_exact(
                            step_mm, position_x_mm, position_y_mm, position_z_mm,
                            direction_x, direction_y, direction_z, ct_origin_x,
                            ct_origin_y, ct_origin_z, ct_spacing_x, ct_spacing_y,
                            ct_spacing_z, ct_nx, ct_ny, ct_nz);
                        step_mm = ct_clamp_res.step_mm;
                    }
                    if (enable_voxel_scoring && voxel_scorer_clamps_transport &&
                        absolute_direction_x >= 1.0e-6F) {
                        const auto boundary_x_mm =
                            voxel_min_x_mm +
                            static_cast<float>(voxel_x + (direction_x > 0.0F ? 1 : 0)) *
                                voxel_size_x_mm;
                        const auto dx_step = (boundary_x_mm - position_x_mm) / direction_x;
                        if (dx_step > 0.0F) {
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
                        if (dy_step > 0.0F) {
                            step_mm = sycl::fmin(step_mm, dy_step);
                        }
                    }

                    if (enable_energy_straggling && enable_step_stable_straggling) {
                        stable_straggling.prepare_step(
                            step_mm, phantom_length_mm - position_z_mm);
                    }

                    bool inelastic_this_step = false;
                    // Elastic candidates are resolved only by the configured CT/water bank.
                    bool ct_elastic_this_step = false;
                    std::int16_t cinel02_target_z_step = 0;
                    std::int16_t cinel02_target_a_step = 0;
                    if (enable_inelastic &&
                        energy_MeV > energy_cutoff_MeV) {
                        const auto cur_e_u = energy_MeV * inverse_mass_number;
                        if (use_schneider_primary_xs) {
                            if (in_ct || use_unified_water) {
                                const std::uint32_t section_id = use_unified_water ? 255U : static_cast<std::uint32_t>(
                                    sycl::min(static_cast<std::uint32_t>(ct_material), 24U));
                                // v3: hazard from the masked rate-binary partials
                                // (single source with the target sampler); v1
                                // keeps the CSV XS table EXACTLY.
                                float mass_rate = 0.0F;
                                if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                                    mass_rate = schneider_ct_device_ctx.primary_rates(
                                        section_id, cur_e_u).total;
                                } else {
                                    mass_rate = schneider_primary_mass_xs(
                                        schneider_primary_xs_device, schneider_xs_sections,
                                        schneider_xs_energies, schneider_xs_e_min, schneider_xs_inv_dE,
                                        section_id, cur_e_u);
                                }
                                // Density enters exactly once, in the total
                                // hazard; target fractions from the sampler CDF
                                // are density-independent.
                                // Full-section elastic hazard (diagnostic): same
                                // optical-depth competition as inelastic; the
                                // branch below splits them exclusively, so no
                                // double counting of the total rate.
                                float macro_el_ct = 0.0F;
                                const bool elastic_armed =
                                    (use_all_elastic && (in_ct || use_unified_water)) ||
                                    (use_ct_elastic && in_ct &&
                                    schneider_ct_device_ctx.elastic_sampler.rate_version == 3);
                                if (use_all_elastic && elastic_armed) {
                                    const float rate=all_elastic.rate(16,use_unified_water?25:section_id,cur_e_u);
                                    if(rate<0) {
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[4]).fetch_add(1);
                                        break;
                                    }
                                    macro_el_ct=local_density_g_per_cm3*rate;
                                } else if (elastic_armed) {
                                    const auto el_rates =
                                        schneider_ct_device_ctx.elastic_rates(
                                            section_id, cur_e_u);
                                    // Legacy diagnostic H-only target selection;
                                    // all-target isotropic needs explicit
                                    // opt-in (unphysical for heavy nuclei).
                                    macro_el_ct = local_density_g_per_cm3 *
                                        (ct_elastic_all_targets
                                             ? el_rates.total
                                             : el_rates.partials[0]);
                                }
                                float macro_inelastic = local_density_g_per_cm3 * mass_rate;
                                if (unified_em) {
                                // Cache only the process whose post-step acceptance uses
                                // this rate. Elastic retains its own local hazard and must
                                // never enter the inelastic acceptance denominator.
                                macro_inelastic = enable_inelastic
                                    ? primary_hadronic_cache.update(cur_e_u, macro_inelastic)
                                    : 0.0F;
                                }
                                const float macro_tot = macro_inelastic + macro_el_ct;
                                ++local_schneider_rate_queries;

                                float u_nuc = 1.0F;
                                if (!nuclear_tau_active) {
                                    u_nuc = rng::uniform01(
                                        spot_seed, rng_history, nuclear_tau_rng_step++, 8);
                                }
                                const bool collision = consume_schneider_optical_depth_segment(
                                    nuclear_tau_remaining, nuclear_tau_active, step_mm, macro_tot, u_nuc);
                                bool elastic_branch = false;
                                if (collision && elastic_armed && macro_tot > 0.0F) {
                                    const float u_br_el = rng::uniform01(
                                        spot_seed, rng_history, steps, 9);
                                    elastic_branch = (u_br_el * macro_tot < macro_el_ct);
                                }
                                if (elastic_branch) {
                                    ct_elastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                } else if (collision && enable_inelastic) {
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryHazards);
                                    if (schneider_inelastic_device != nullptr) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            nuc_ref(*schneider_inelastic_device);
                                        nuc_ref.fetch_add(1U);
                                    }
                                    if (inelastic_reaction_device != nullptr) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            reaction(inelastic_reaction_device[bin]);
                                        reaction.fetch_add(1U);
                                    }
                                    inelastic_this_step = true;
                                    interaction_section = section_id;
                                    interaction_density = local_density_g_per_cm3;
                                }
                            }
                        }
                    }

                    // Diagnostic only: CT face clamping above guarantees the
                    // segment stays in its starting CT voxel. Do not score a
                    // laterally escaped segment into a clamped edge voxel.
                    if (enable_primary_voxel_fluence &&
                        (!enable_ct_grid || in_ct)) {
                        sycl::atomic_ref<
                            DoseAtomicT, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_track_length(
                                primary_voxel_track_length_device[voxel_index]);
                        atomic_track_length.fetch_add(
                            static_cast<DoseAtomicT>(step_mm));
                    }

                    float mean_loss_MeV = 0.0F;
                    if(unified_em){
                        // The unified loss sampler computes the physical mean below.
                        // This separate query is needed only by the optional audit.
                        if (primary_loss_query_audit)
                            mean_loss_MeV=unified_primary_state.mean(energy_MeV,step_mm);
                    } else if (ct_primary_midpoint_stopping && use_schneider_stopping && in_ct) {
                        mean_loss_MeV=midpoint_continuous_energy_loss(energy_MeV,step_mm,
                            stopping_power_MeV_per_mm,[&](float mid_energy) {
                                const float f=(mid_energy*inverse_mass_number-schneider_sp_e_min)*schneider_sp_inv_dE;
                                const int i=sycl::clamp(static_cast<int>(sycl::floor(f)),0,static_cast<int>(schneider_sp_energies)-2);
                                const float w=sycl::clamp(f-static_cast<float>(i),0.0F,1.0F);
                                const auto base=static_cast<std::size_t>(ct_material)*schneider_sp_energies;
                                const float mass_sp=schneider_stopping_device[base+i]+w*(schneider_stopping_device[base+i+1]-schneider_stopping_device[base+i]);
                                return mass_sp*sycl::fmax(local_density_g_per_cm3,1.0e-6F);
                            });
                    } else if (enable_csda_range_energy_loss && energy_grid_device != nullptr &&
                        cumulative_range_device != nullptr) {
                        const auto end_energy_MeVu = csda_energy_after_distance_device(
                            primary_csda_energy_grid_device, primary_water_sp_device,
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

                    if (primary_loss_query_audit && mean_loss_MeV > 0 && energy_MeV > 0) {
                        // Bin 0: f<1e-12; bins 1..12: decades; bin 13: f>=1.
                        const float f = mean_loss_MeV / energy_MeV;
                        const int b = sycl::clamp(static_cast<int>(sycl::floor(sycl::log10(f)))+13, 0, 13);
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>
                            count(primary_loss_query_audit[b]);
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device, sycl::access::address_space::global_space>
                            energy_sum(primary_loss_query_audit[b+14]);
                        count.fetch_add(1);
                        energy_sum.fetch_add(static_cast<std::uint64_t>(mean_loss_MeV*1.0e6F+0.5F));
                    }

                    if (enable_energy_straggling && !unified_em) {
                        // Explicit smoke hybrid: retain the existing Gaussian
                        // only for f<1e-4; never substitute it for missing energy
                        // coverage or other out-of-domain package queries.
                        if (use_packaged_fluctuation &&
                            (!fluct_fraction_hybrid || mean_loss_MeV / energy_MeV >= 1.0e-4F)) {
                            const auto fluct_coordinate = fluct_fraction_axis
                                ? mean_loss_MeV / energy_MeV
                                : local_density_g_per_cm3 * step_mm / 10.0F;
                            if (fluct_fraction_axis && mean_loss_MeV > 0 &&
                                (energy_MeVu < fluct_energy_device[0] ||
                                 energy_MeVu > fluct_energy_device[fluct_energy_count-1] ||
                                 fluct_coordinate < fluct_density_device[0] ||
                                 fluct_coordinate > fluct_density_device[fluct_density_count-1])) {
                                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, sycl::access::address_space::global_space>
                                    fail(*fluct_domain_failures);
                                fail.fetch_add(1);
                            }
                            const auto u_loss = rng::uniform01(
                                spot_seed, rng_history, steps, 2);
                            const auto ratio = sample_energy_loss_ratio_from_grid(
                                fluct_energy_device, fluct_energy_count,
                                fluct_density_device, fluct_density_count,
                                fluct_probability_device, fluct_probability_count,
                                fluct_quantile_device, energy_MeVu,
                                fluct_coordinate, u_loss);
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

                    if(unified_em){
                        auto draw=unified_em_loss(unified_primary_state,unified_primary_clock,energy_MeV,step_mm,unified_primary_rate,unified_primary_distance,enable_energy_straggling,unified_primary_pre,unified_uniform);
                        if(!draw.valid){
                            record_unified_em_failure(unified_failure_count,unified_failure_records,2,unified_primary_state.section,primary_atomic_number,primary_mass_number,energy_MeV,local_density_g_per_cm3,step_mm);
                            unified_count(0);break;}
                        deposited_MeV=draw.loss;unified_count(1);unified_count(3,draw.proposed);unified_count(4,draw.accepted);
                        unified_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                    }
                    auto local_voxel_deposit_MeV = deposited_MeV;
                    auto same_voxel_electron_packet_MeV = 0.0F;
                    auto delta_tail_escaped_scorer_MeV = 0.0F;
                    // Forward-redistributed energy leaves the source depth bin, so the
                    // 1-D depth scorers must not credit it at the source bin (the 3-D
                    // march deposits below credit the destination bins instead).
                    auto forward_shifted_MeV = 0.0F;
                    auto transverse_relocated_MeV = 0.0F;
                    auto transverse_escaped_MeV = 0.0F;
                    auto water_physical_escape_MeV=0.0F;
                    float material_untracked_MeV=0;
                    if(use_material_ct && in_ct && deposited_MeV>0) {
                        auto count=[&](int slot,std::uint64_t value) {
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);a.fetch_add(value);
                        };
                        auto untracked=[&](double weight,bool photon) {
                            material_untracked_MeV+=static_cast<float>(weight);
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(material_untracked_device[photon?0:1]);
                            a.fetch_add(static_cast<std::uint64_t>(weight*1e6+0.5));
                        };
                        // A separate deterministic packet stream does not advance the
                        // carbon transport RNG. Rebinding consumes this stream only.
                        std::uint64_t packet_rng=spot_seed ^ ((rng_history+1)*0x9e3779b97f4a7c15ULL) ^
                            (static_cast<std::uint64_t>(steps)<<32) ^ 0x6a09e667f3bcc909ULL;
                        auto uniform=[&]() {packet_rng=packet_rng*6364136223846793005ULL+1442695040888963407ULL;
                            return double(packet_rng>>11)*0x1.0p-53;};
                        count(0,1);
                        const double density_u=uniform(),birth_u=uniform();
                        const double along=uniform();
                        const std::array<double,3> birth_position{
                            position_x_mm+along*step_mm*direction_x,position_y_mm+along*step_mm*direction_y,
                            position_z_mm+along*step_mm*direction_z};
                        const ElectronCtGeometry geometry{{ct_nx,ct_ny,ct_nz},{ct_origin_x,ct_origin_y,ct_origin_z},
                            {ct_spacing_x,ct_spacing_y,ct_spacing_z},ct_density_device,ct_material_device};
                        const auto birth_material=electron_ct_point_material(geometry,birth_position,
                            {direction_x,direction_y,direction_z});
                        MaterialElectronBirthDraw material_birth;
                        if(birth_material.valid)material_birth=sample_material_electron_birth(birth_material.section,
                            birth_material.density,energy_MeVu,density_u,birth_u,
                            material_electron_views,material_electron_view_count);
                        const auto birth=material_birth.birth;
                        const auto ti=material_birth.table;
                        if(!birth.valid) {
                            count(1,1);untracked(deposited_MeV,false);
                            local_voxel_deposit_MeV=0;forward_shifted_MeV+=deposited_MeV;
                        } else if(birth.fraction>0) {
                            const float weight=deposited_MeV*static_cast<float>(birth.fraction);
                            local_voxel_deposit_MeV-=weight;forward_shifted_MeV+=weight;
                            ElectronEnergyPacket packet;packet.weight_MeV=weight;
                            packet.material_table_index=ti;
                            packet.cursor=bind_electron_birth(birth,material_electron_views[ti],
                                birth_position,{direction_x,direction_y,direction_z},uniform());
                            packet.cursor.physical_density_g_cm3=birth_material.density;
                            packet.status=packet.cursor.valid?ElectronPacketStatus::active:ElectronPacketStatus::invalid;
                            if(electron_short_range_mm>0 && electron_short_range_contained(
                                packet,material_electron_views[ti],geometry,electron_short_range_mm)) {
                                packet.status=ElectronPacketStatus::deposited;
                                packet.deposit_position=packet.cursor.position;
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> hits(*short_range_hits_device);
                                hits.fetch_add(1);
                            }
                            auto previous=packet;auto previous_rng=packet_rng;
                            for(unsigned j=0;j<4096 && packet.status==ElectronPacketStatus::active;++j) {
                                previous=packet;previous_rng=packet_rng;
                                packet=transport_electron_packet_step(packet,material_electron_views,
                                    material_electron_view_count,geometry,uniform);
                            }
                            count(6,1);
                            if(packet.status==ElectronPacketStatus::deposited) {
                                const auto p=packet.deposit_position;
                                const int ix=static_cast<int>(sycl::floor((p[0]-voxel_min_x_mm)/voxel_size_x_mm));
                                const int iy=static_cast<int>(sycl::floor((p[1]-voxel_min_y_mm)/voxel_size_y_mm));
                                const int iz=static_cast<int>(sycl::floor(p[2]/voxel_size_z_mm));
                                auto add=[&](auto* address) {
                                    using T=std::remove_pointer_t<decltype(address)>;
                                    sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(weight));
                                };
                                if(ix>=0 && iy>=0 && iz>=0 && ix<static_cast<int>(voxel_bins_x) &&
                                   iy<static_cast<int>(voxel_bins_y) && iz<static_cast<int>(voxel_bins_z)) {
                                    const auto target=(static_cast<std::size_t>(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;
                                    add(voxel_dose_device+target);
                                    const int dz=static_cast<int>(sycl::floor(p[2]/depth_bin_width_mm));
                                    if(dz>=0 && dz<static_cast<int>(number_of_bins)) {
                                        add(dose_device+dz);if(in_fov_dose_device)add(in_fov_dose_device+dz);
                                    }
                                    if(enable_charged_origin_voxel_scoring)add(charged_origin_voxel_dose_device+target);
                                    count(3,static_cast<std::uint64_t>(double(weight)*1e6));
                                } else {delta_tail_escaped_scorer_MeV+=weight;count(4,static_cast<std::uint64_t>(double(weight)*1e6));}
                            } else if(packet.status==ElectronPacketStatus::escaped) {
                                water_physical_escape_MeV+=weight;history_water_electron_escaped_MeV+=weight;
                                count(4,static_cast<std::uint64_t>(double(weight)*1e6));
                            } else {
                                const bool photon=packet.status==ElectronPacketStatus::coverage_missing &&
                                    packet.gap==ElectronPacketGap::photon_continuation;
                                untracked(weight,photon);
                                if(!photon) {
                                    count(2,1); // unresolved electrons/invalid/cap remain hard failures
                                    sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space> first(material_untracked_device[2]);
                                    if(first.fetch_add(1)==0) {
                                        MaterialPacketFailure f;f.history=global_history;f.step=steps;f.rng=previous_rng;
                                        f.before=previous;f.after=packet;
                                        for(std::size_t t=0;t<material_electron_view_count;++t)
                                            if(material_electron_views[t].section==previous.cursor.section &&
                                               material_electron_views[t].density_g_cm3==previous.cursor.density_g_cm3)
                                                f.advance=advance_electron_continuation(previous.cursor,material_electron_views[t],geometry);
                                        *material_failure_device=f;
                                    }
                                }
                            }
                        }
                    }
                    if(use_water_electron && deposited_MeV>0) {
                        auto counter=[&](int slot,double value) {
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);
                            a.fetch_add(static_cast<std::uint64_t>(value));
                        };
                        counter(0,1);
                        auto draw=sample_water_electron_response(energy_MeVu,
                            rng::uniform01(spot_seed,rng_history,steps,21),rng::uniform01(spot_seed,rng_history,steps,23),
                            water_electron_channels_device,water_electron_samples_device,water_electron_heads_device,
                            water_electron_channel_count);
                        const WaterElectronPathNode* response_nodes=water_electron_nodes_device;
                        const double* response_radius=water_electron_radius_device;
                        auto response_node_count=water_electron_node_count;
                        if(use_material_electron) {
                            const auto selected=sample_material_electron_response(-1,material_response_density,energy_MeVu,
                                rng::uniform01(spot_seed,rng_history,steps,24),
                                rng::uniform01(spot_seed,rng_history,steps,21),rng::uniform01(spot_seed,rng_history,steps,23),
                                material_electron_views,material_electron_view_count);
                            draw=selected.response;
                            if(selected.status!=MaterialElectronStatus::hit)draw.valid=false;
                            else {
                                const auto table=material_electron_views[selected.table_index];
                                response_nodes=table.nodes;response_radius=table.prefix_radius;
                                response_node_count=table.node_count;
                            }
                        }
                        if(!draw.valid) {counter(1,1);counter(5,static_cast<double>(deposited_MeV)*1e6);}
                        else {
                            // Finite-source photon remainder is EXPLICITLY unresolved.
                            // In this unvalidated EM-only pilot it is carried as escaping
                            // energy, never renormalized into the charged response.
                            const float unresolved=deposited_MeV*static_cast<float>(draw.unresolved);
                            local_voxel_deposit_MeV-=unresolved;forward_shifted_MeV+=unresolved;
                            water_physical_escape_MeV+=unresolved;
                            const float packet=deposited_MeV*static_cast<float>(draw.fraction);
                            if(packet>0) {
                                const double phi=6.2831853071795864769*rng::uniform01(spot_seed,rng_history,steps,22);
                                const auto axis=Direction3F{direction_x,direction_y,direction_z};
                                const auto ex=rotate_local_direction(static_cast<float>(sycl::cos(phi)),static_cast<float>(sycl::sin(phi)),0,axis);
                                const auto ey=rotate_local_direction(static_cast<float>(-sycl::sin(phi)),static_cast<float>(sycl::cos(phi)),0,axis);
                                const double birth=rng::uniform01(spot_seed,rng_history,steps,20);
                                const double bz=position_z_mm+birth*step_mm*direction_z;
                                const auto status=water_electron_path_in_slab(draw,bz,phantom_length_mm,
                                    {ex.z,ey.z,direction_z},response_nodes,response_node_count,response_radius);
                                if(status==WaterElectronPathStatus::invalid) {counter(2,1);counter(5,static_cast<double>(packet)*1e6);}
                                else {
                                    local_voxel_deposit_MeV-=packet;forward_shifted_MeV+=packet;
                                    counter(6,1);
                                    if(status==WaterElectronPathStatus::escaped) {
                                        water_physical_escape_MeV+=packet;counter(4,static_cast<double>(packet)*1e6);
                                    } else {
                                        const auto p=draw.point;
                                        const double x=position_x_mm+birth*step_mm*direction_x+p[0]*ex.x+p[1]*ey.x+p[2]*direction_x;
                                        const double y=position_y_mm+birth*step_mm*direction_y+p[0]*ex.y+p[1]*ey.y+p[2]*direction_y;
                                        const double z=bz+p[0]*ex.z+p[1]*ey.z+p[2]*direction_z;
                                        const int ix=static_cast<int>(sycl::floor((x-voxel_min_x_mm)/voxel_size_x_mm));
                                        const int iy=static_cast<int>(sycl::floor((y-voxel_min_y_mm)/voxel_size_y_mm));
                                        const int iz=static_cast<int>(sycl::floor(z/voxel_size_z_mm));
                                        auto add=[&](auto* address) {
                                            using T=std::remove_pointer_t<decltype(address)>;
                                            sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(packet));
                                        };
                                        // Native water has no transverse material edge at
                                        // the dose ROI. Only the final point determines its tally.
                                        if(ix>=0 && iy>=0 && iz>=0 && ix<static_cast<int>(voxel_bins_x) &&
                                           iy<static_cast<int>(voxel_bins_y) && iz<static_cast<int>(voxel_bins_z)) {
                                            const auto target=(static_cast<std::size_t>(iz)*voxel_bins_y+iy)*voxel_bins_x+ix;
                                            const int dz=static_cast<int>(sycl::floor(z/depth_bin_width_mm));
                                            add(voxel_dose_device+target);
                                            if(dz>=0 && dz<static_cast<int>(number_of_bins)) {
                                                add(dose_device+dz);if(in_fov_dose_device)add(in_fov_dose_device+dz);
                                            }
                                            if(enable_charged_origin_voxel_scoring)add(charged_origin_voxel_dose_device+target);
                                            counter(3,static_cast<double>(packet)*1e6);
                                        } else {delta_tail_escaped_scorer_MeV+=packet;counter(4,static_cast<double>(packet)*1e6);}
                                    }
                                }
                            }
                        }
                        history_water_electron_escaped_MeV+=water_physical_escape_MeV;
                    }
                    if(use_electron_joint && in_ct && enable_voxel_scoring &&
                       voxel_index<number_of_voxels && deposited_MeV>0) {
                        auto counter=[&](int slot,std::uint64_t amount) {
                            if(!enable_electron_joint_diagnostics && slot!=1 && slot!=2) return;
                            sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                                sycl::memory_scope::device,sycl::access::address_space::global_space> a(electron_joint_diag_device[slot]);a.fetch_add(amount);
                        };
                        counter(0,1);
                        const auto draw=sample_electron_joint_device(ct_material,energy_MeVu,
                            rng::uniform01(spot_seed,rng_history,steps,21),electron_joint_channels_device,
                            electron_joint_samples_device,electron_joint_minimum,electron_joint_maximum,electron_joint_ceiling);
                        if(draw.status!=ElectronJointStatus::hit) {
                            counter(draw.status==ElectronJointStatus::energy_domain ? 1 : 2,1);
                            counter(5,static_cast<std::uint64_t>(static_cast<double>(deposited_MeV)*1e6));
                        } else if(draw.fraction>0) {
                            // This sampled endpoint is already an energy-weighted FULL
                            // response deposit. Do NOT distribute it uniformly along the
                            // ray again, and do NOT also apply the old transverse tail.
                            const float packet=deposited_MeV*static_cast<float>(draw.fraction);
                            const double radius=sycl::sqrt(draw.longitudinal_mass_g_cm2*draw.longitudinal_mass_g_cm2+
                                                           draw.radial_mass_g_cm2*draw.radial_mass_g_cm2);
                            if((radius>0 || electron_path_vectors_device) && packet>0) {
                                const double phi=6.2831853071795864769*rng::uniform01(spot_seed,rng_history,steps,22);
                                const auto ray=radius>0 ? rotate_local_direction(
                                    static_cast<float>(draw.radial_mass_g_cm2*sycl::cos(phi)/radius),
                                    static_cast<float>(draw.radial_mass_g_cm2*sycl::sin(phi)/radius),
                                    static_cast<float>(draw.longitudinal_mass_g_cm2/radius),
                                    Direction3F{direction_x,direction_y,direction_z}) : Direction3F{0,0,1};
                                const double birth=rng::uniform01(spot_seed,rng_history,steps,20);
                                std::size_t target=voxel_index;int target_z=bin;
                                LongitudinalMarchResult march;
                                if(electron_path_vectors_device) {
                                    counter(6,1);
                                    const auto range=electron_path_ranges_device[draw.sample_index];
                                    const auto ex=rotate_local_direction(static_cast<float>(sycl::cos(phi)),static_cast<float>(sycl::sin(phi)),0,Direction3F{direction_x,direction_y,direction_z});
                                    const auto ey=rotate_local_direction(static_cast<float>(-sycl::sin(phi)),static_cast<float>(sycl::cos(phi)),0,Direction3F{direction_x,direction_y,direction_z});
                                    // FP32 electron march: identical algorithm to the
                                    // validated double march; endpoints agree to
                                    // ~1e-4 mm (voxel scale 0.5 mm). Validated by
                                    // gamma equivalence, not bitwise identity.
                                    const std::array<float,3> birth_f{
                                        position_x_mm+static_cast<float>(birth)*step_mm*direction_x,
                                        position_y_mm+static_cast<float>(birth)*step_mm*direction_y,
                                        position_z_mm+static_cast<float>(birth)*step_mm*direction_z};
                                    const std::array<float,3> origin_f{ct_origin_x,ct_origin_y,ct_origin_z};
                                    const std::array<float,3> spacing_f{ct_spacing_x,ct_spacing_y,ct_spacing_z};
                                    const std::array<int,3> dims_i{static_cast<int>(ct_nx),static_cast<int>(ct_ny),static_cast<int>(ct_nz)};
                                    auto density_f=[&](const std::array<int,3>& c) {
                                        return ct_density_device[(static_cast<std::size_t>(c[2])*ct_ny+c[1])*ct_nx+c[0]];};
                                    MassPathResult path;
                                    path.endpoint={birth_f[0],birth_f[1],birth_f[2]};
                                    const auto stored_bounds=electron_path_bounds_device[draw.sample_index];
                                    MassPathBoundsT<float> bounds_f;
                                    for(int bi=0;bi<3;++bi) {
                                        bounds_f.low[bi]=static_cast<float>(stored_bounds.low[bi]);
                                        bounds_f.high[bi]=static_cast<float>(stored_bounds.high[bi]);
                                        bounds_f.net[bi]=static_cast<float>(stored_bounds.net[bi]);
                                    }
                                    const std::array<std::array<float,3>,3> basis_f{
                                        {{ex.x,ex.y,ex.z},{ey.x,ey.y,ey.z},{direction_x,direction_y,direction_z}}};
                                    std::array<float,3> probe_f{birth_f[0],birth_f[1],birth_f[2]};
                                    const bool same_cell=try_same_voxel_mass_path<float>(probe_f,
                                        bounds_f,basis_f,origin_f,spacing_f,dims_i,density_f);
                                    if(same_cell) {
                                        path.endpoint={probe_f[0],probe_f[1],probe_f[2]};
                                    }
                                    if(same_cell)path.completed_segments=range.count;
                                    else path=replay_mass_polyline_indexed<float>(
                                        birth_f,range.count,
                                        [&](std::size_t j) {
                                            const auto v=electron_path_vectors_device[range.offset+j];
                                            return std::array<float,3>{
                                                static_cast<float>(v[0])*ex.x+static_cast<float>(v[1])*ey.x+static_cast<float>(v[2])*direction_x,
                                                static_cast<float>(v[0])*ex.y+static_cast<float>(v[1])*ey.y+static_cast<float>(v[2])*direction_y,
                                                static_cast<float>(v[0])*ex.z+static_cast<float>(v[1])*ey.z+static_cast<float>(v[2])*direction_z};
                                        },origin_f,spacing_f,dims_i,density_f);
                                    march.invalid=path.invalid;march.escaped=path.escaped;
                                    if(!march.invalid && !march.escaped) {
                                        const int x=static_cast<int>(sycl::floor((path.endpoint[0]-ct_origin_x)/ct_spacing_x));
                                        const int y=static_cast<int>(sycl::floor((path.endpoint[1]-ct_origin_y)/ct_spacing_y));
                                        const int z=static_cast<int>(sycl::floor((path.endpoint[2]-ct_origin_z)/ct_spacing_z));
                                        if(x<0 || y<0 || z<0 || x>=static_cast<int>(ct_nx) || y>=static_cast<int>(ct_ny) || z>=static_cast<int>(ct_nz))march.escaped=true;
                                        else {target=(static_cast<std::size_t>(z)*ct_ny+y)*ct_nx+x;target_z=z;}
                                    }
                                } else march=march_longitudinal_mass_segments(
                                    {position_x_mm+birth*step_mm*direction_x,position_y_mm+birth*step_mm*direction_y,position_z_mm+birth*step_mm*direction_z},
                                    {ray.x,ray.y,ray.z},{ct_origin_x,ct_origin_y,ct_origin_z},
                                    {ct_spacing_x,ct_spacing_y,ct_spacing_z},
                                    {static_cast<int>(ct_nx),static_cast<int>(ct_ny),static_cast<int>(ct_nz)},radius,
                                    [&](const std::array<int,3>& cell) {
                                        return static_cast<double>(ct_density_device[(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0]]);
                                    },[&](const std::array<int,3>& cell,double) {
                                        target=(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0];target_z=cell[2];
                                    });
                                if(march.invalid || march.blocked)counter(2,1);
                                else {
                                    if(march.escaped) {
                                        delta_tail_escaped_scorer_MeV+=packet;
                                        counter(4,static_cast<std::uint64_t>(static_cast<double>(packet)*1e6));
                                    } else {
                                        auto add=[&](auto* address) {
                                            using T=std::remove_pointer_t<decltype(address)>;
                                            sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                                sycl::access::address_space::global_space> a(*address);a.fetch_add(static_cast<T>(packet));
                                        };
                                        if(target==voxel_index && target_z==bin) {
                                            // The replay still happened. Cache its original
                                            // packet separately, preserving rounded dE-p
                                            // and p rather than replacing them by dE.
                                            same_voxel_electron_packet_MeV=packet;
                                        } else {
                                            add(voxel_dose_device+target);add(dose_device+target_z);
                                            if(in_fov_dose_device)add(in_fov_dose_device+target_z);
                                            if(enable_charged_origin_voxel_scoring)add(charged_origin_voxel_dose_device+target);
                                        }
                                        counter(3,static_cast<std::uint64_t>(static_cast<double>(packet)*1e6));
                                    }
                                    local_voxel_deposit_MeV-=packet;forward_shifted_MeV+=packet;
                                }
                            }
                        }
                    }
                    if (!use_electron_joint && use_schneider_delta_tail && in_ct && ct_material == 0U &&
                        enable_voxel_scoring && voxel_index < number_of_voxels &&
                        schneider_delta_source_eligible_device[voxel_index] != 0U) {
                        float moved_fraction = 0.0F;
                        float radius_mm = 0.0F;
                        schneider_delta_tail_lookup_device(
                            energy_MeVu,
                            rng::uniform01(spot_seed, rng_history, steps, 17),
                            schneider_delta_energies_device,
                            schneider_delta_fractions_device,
                            schneider_delta_radii_device,
                            schneider_delta_energy_count,
                            schneider_delta_quantile_count,
                            moved_fraction, radius_mm);
                        const auto moved_MeV =
                            deposited_MeV * sycl::clamp(moved_fraction, 0.0F, 0.5F);
                        if (moved_MeV > 0.0F && radius_mm > 0.0F) {
                            constexpr float two_pi = 6.2831853071795864769F;
                            const auto phi = two_pi * rng::uniform01(
                                spot_seed, rng_history, steps, 18);
                            const auto transverse = rotate_local_direction(
                                sycl::cos(phi), sycl::sin(phi), 0.0F,
                                Direction3F{direction_x, direction_y, direction_z});
                            const auto source_x = position_x_mm + 0.5F * step_mm * direction_x;
                            const auto source_y = position_y_mm + 0.5F * step_mm * direction_y;
                            const auto source_z = position_z_mm + 0.5F * step_mm * direction_z;
                            const auto destination_x = source_x + radius_mm * transverse.x;
                            const auto destination_y = source_y + radius_mm * transverse.y;
                            const auto destination_z = source_z + radius_mm * transverse.z;
                            float destination_density = 0.0F;
                            std::uint8_t destination_material = 255U;
                            const auto destination_in_section0 = ct_sample(
                                destination_x, destination_y, destination_z,
                                ct_origin_x, ct_origin_y, ct_origin_z,
                                ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                ct_nx, ct_ny, ct_nz, ct_density_device,
                                ct_material_device, destination_density,
                                destination_material) && destination_material == 0U;
                            const auto destination_voxel_x = static_cast<int>(sycl::floor(
                                (destination_x - voxel_min_x_mm) / voxel_size_x_mm));
                            const auto destination_voxel_y = static_cast<int>(sycl::floor(
                                (destination_y - voxel_min_y_mm) / voxel_size_y_mm));
                            const auto destination_bin = static_cast<int>(sycl::floor(
                                destination_z / depth_bin_width_mm));
                            const auto destination_in_scorer =
                                destination_voxel_x >= 0 &&
                                destination_voxel_x < static_cast<int>(voxel_bins_x) &&
                                destination_voxel_y >= 0 &&
                                destination_voxel_y < static_cast<int>(voxel_bins_y) &&
                                destination_bin >= 0 &&
                                destination_bin < static_cast<int>(number_of_bins);
                            const auto fixed = static_cast<std::uint64_t>(
                                static_cast<double>(moved_MeV) * 1.0e6);
                            if (destination_in_section0 && destination_in_scorer) {
                                const auto destination_voxel =
                                    static_cast<std::size_t>(destination_bin) * voxel_plane_size +
                                    static_cast<std::size_t>(destination_voxel_y) * voxel_bins_x +
                                    static_cast<std::size_t>(destination_voxel_x);
                                sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_delta(voxel_dose_device[destination_voxel]);
                                atomic_delta.fetch_add(static_cast<DoseAtomicT>(moved_MeV));
                                // Mirror the actual 3-D destination. Even a pencil beam
                                // can cross a depth face between step start and midpoint.
                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    target_depth(dose_device[destination_bin]);
                                target_depth.fetch_add(static_cast<DepthAtomicT>(moved_MeV));
                                if (in_fov_dose_device) {
                                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        target_fov(in_fov_dose_device[destination_bin]);
                                    target_fov.fetch_add(static_cast<DepthAtomicT>(moved_MeV));
                                }
                                transverse_relocated_MeV = moved_MeV;
                                if (enable_charged_origin_voxel_scoring) {
                                    sycl::atomic_ref<DoseAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_origin(charged_origin_voxel_dose_device[
                                            destination_voxel]);
                                    atomic_origin.fetch_add(
                                        static_cast<DoseAtomicT>(moved_MeV));
                                }
                                local_voxel_deposit_MeV -= moved_MeV;
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_moved(schneider_delta_energy_device[0]);
                                atomic_moved.fetch_add(fixed);
                            } else if (!destination_in_scorer) {
                                // The aligned CCTG and dose scorer share the same bounds.
                                // A sampled delta-electron endpoint outside those bounds must
                                // leave the voxel score instead of being folded back into the
                                // edge voxel. Transfer it from the in-grid sink to outside-grid.
                                local_voxel_deposit_MeV -= moved_MeV;
                                delta_tail_escaped_scorer_MeV += moved_MeV;
                                transverse_escaped_MeV = moved_MeV;
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_escape(schneider_delta_energy_device[2]);
                                atomic_escape.fetch_add(fixed);
                            } else {
                                // Cross-material electron transport is outside this section-0
                                // LUT. Preserve the energy locally rather than aliasing a target.
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fallback(schneider_delta_energy_device[1]);
                                atomic_fallback.fetch_add(fixed);
                            }
                        }
                    }
                        if (use_schneider_delta_longitudinal && in_ct && enable_voxel_scoring &&
                            voxel_index < number_of_voxels && deposited_MeV > 0.0F &&
                            (use_longitudinal_interface_mass || (ct_material==0U &&
                             schneider_delta_source_eligible_device[voxel_index]!=0U))) {
                            // Forward supplement: carry a fitted fraction of the local
                            // deposit downstream along the particle direction with an
                            // exponential range, distributed uniformly along the ray
                            // (diagnostic approximation, not validated electron physics).
                            // Only reference-density section-0 voxels receive exact
                            // path shares. At unsupported density/material, retain the
                            // remainder at source; at scorer exit, book escape. This
                            // does not establish correct air-tissue interface transport.
                            float forward_fraction = 0.0F;
                            float forward_lambda_mm = 0.0F;
                            const bool longitudinal_covered = schneider_longitudinal_lookup_device(
                                energy_MeVu,
                                schneider_long_energies_device,
                                schneider_long_fractions_device,
                                schneider_long_lambdas_device,
                                schneider_long_energy_count,
                                forward_fraction, forward_lambda_mm);
                            if (!longitudinal_covered) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    count(schneider_delta_energy_device[6]),
                                    energy(schneider_delta_energy_device[7]);
                                const auto domain_slot = count.fetch_add(1);
                                if (longitudinal_domain_device &&
                                    domain_slot < kLongitudinalDomainLogCap) {
                                    longitudinal_domain_device[domain_slot] = {
                                        global_history, static_cast<std::uint32_t>(steps),
                                        sampled_initial_energy_MeVu, energy_MeVu,
                                        position_z_mm, step_mm, deposited_MeV};
                                }
                                energy.fetch_add(static_cast<std::uint64_t>(
                                    static_cast<double>(deposited_MeV) * 1e6));
                            }
                            const auto forward_MeV =
                                deposited_MeV * sycl::clamp(
                                    forward_fraction * schneider_long_fraction_scale,
                                    0.0F, 0.5F);
                            // Default remains reference-only. Optional probe has already
                            // verified the ENTIRE grid is homogeneous at one known density.
                            const double rho_ref = schneider_long_diagnostic_density;
                            if (forward_MeV > 0.0F && forward_lambda_mm > 0.0F) {
                                const auto u = rng::uniform01(spot_seed, rng_history, steps, 19);
                                const double range = -static_cast<double>(forward_lambda_mm) *
                                    (kLongitudinalReferenceDensityGPerCm3 / rho_ref) *
                                    sycl::log(1.0 - sycl::clamp(static_cast<double>(u), 0.0, 0.99999988));
                                double fwd_escaped_MeV = 0, fwd_kept_MeV = 0, fwd_scored_MeV = 0;
                                if (use_longitudinal_interface_mass && range > 0) {
                                    // Diagnostic hypothesis: the fixed air kernel expressed
                                    // in mass thickness, with sources on BOTH sides. Same
                                    // fraction/range table; no interface-fitted multiplier.
                                    // The ion loses energy along the ENTIRE step, not at
                                    // its midpoint. In tissue lambda is sub-voxel; midpoint
                                    // emission biases cross-face fluence. Independent tag20
                                    // integrates uniform per-step births without a grid phase.
                                    const double birth_fraction=rng::uniform01(spot_seed,rng_history,steps,20);
                                    const auto march = march_longitudinal_mass_segments(
                                        {position_x_mm + birth_fraction * step_mm * direction_x,
                                         position_y_mm + birth_fraction * step_mm * direction_y,
                                         position_z_mm + birth_fraction * step_mm * direction_z},
                                        {direction_x, direction_y, direction_z},
                                        {ct_origin_x, ct_origin_y, ct_origin_z},
                                        {ct_spacing_x, ct_spacing_y, ct_spacing_z},
                                        {static_cast<int>(ct_nx), static_cast<int>(ct_ny),
                                         static_cast<int>(ct_nz)}, range * rho_ref / 10.0,
                                        [&](const std::array<int,3>& cell) {
                                            return static_cast<double>(ct_density_device[
                                                (static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0]]);
                                        },
                                        [&](const std::array<int,3>& cell,double fraction) {
                                            const auto index=(static_cast<std::size_t>(cell[2])*ct_ny+cell[1])*ct_nx+cell[0];
                                            const double share=static_cast<double>(forward_MeV)*fraction;
                                            auto add=[&](auto* address) {
                                                using T=std::remove_pointer_t<decltype(address)>;
                                                sycl::atomic_ref<T,sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space> atom(*address);
                                                atom.fetch_add(static_cast<T>(share));
                                            };
                                            add(voxel_dose_device+index);
                                            add(dose_device+cell[2]);
                                            if(in_fov_dose_device) add(in_fov_dose_device+cell[2]);
                                            if(enable_charged_origin_voxel_scoring)
                                                add(charged_origin_voxel_dose_device+index);
                                            fwd_scored_MeV+=share;
                                        });
                                    const double remainder=sycl::fmax(0.0,static_cast<double>(forward_MeV)-fwd_scored_MeV);
                                    if(march.escaped) fwd_escaped_MeV=remainder;
                                    else fwd_kept_MeV=remainder;
                                    if(march.invalid || march.blocked) {
                                        sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space> invalid(schneider_delta_energy_device[9]);
                                        invalid.fetch_add(1);
                                    }
                                } else if (range > 0 &&
                                    sycl::fabs(static_cast<double>(local_density_g_per_cm3)-rho_ref)
                                        <= rho_ref * 1e-4) {
                                    const auto march = march_longitudinal_segments(
                                        {position_x_mm + 0.5 * step_mm * direction_x,
                                         position_y_mm + 0.5 * step_mm * direction_y,
                                         position_z_mm + 0.5 * step_mm * direction_z},
                                        {direction_x, direction_y, direction_z},
                                        {ct_origin_x, ct_origin_y, ct_origin_z},
                                        {ct_spacing_x, ct_spacing_y, ct_spacing_z},
                                        {static_cast<int>(ct_nx), static_cast<int>(ct_ny),
                                         static_cast<int>(ct_nz)}, range,
                                        [&](const std::array<int,3>& cell, double length) {
                                            const auto index = (static_cast<std::size_t>(cell[2]) *
                                                ct_ny + cell[1]) * ct_nx + cell[0];
                                            if (ct_material_device[index] != 0U ||
                                                sycl::fabs(static_cast<double>(ct_density_device[index]) -
                                                           rho_ref) > rho_ref * 1e-4) return false;
                                            const double share = static_cast<double>(forward_MeV) * length / range;
                                            auto add = [&](auto* address) {
                                                using T = std::remove_pointer_t<decltype(address)>;
                                                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space> atom(*address);
                                                atom.fetch_add(static_cast<T>(share));
                                            };
                                            add(voxel_dose_device + index);
                                            add(dose_device + cell[2]);
                                            if (in_fov_dose_device) add(in_fov_dose_device + cell[2]);
                                            if (enable_charged_origin_voxel_scoring)
                                                add(charged_origin_voxel_dose_device + index);
                                            fwd_scored_MeV += share;
                                            return true;
                                        });
                                    const double remainder = sycl::fmax(
                                        0.0, static_cast<double>(forward_MeV)-fwd_scored_MeV);
                                    if (march.escaped) fwd_escaped_MeV = remainder;
                                    else fwd_kept_MeV = remainder;
                                    if (march.invalid) {
                                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                                            invalid(schneider_delta_energy_device[9]);
                                        invalid.fetch_add(1);
                                    }
                                } else fwd_kept_MeV = forward_MeV;
                                local_voxel_deposit_MeV -= forward_MeV - fwd_kept_MeV;
                                delta_tail_escaped_scorer_MeV += fwd_escaped_MeV;
                                forward_shifted_MeV = forward_MeV - fwd_kept_MeV;
                                const auto fwd_fixed = static_cast<std::uint64_t>(
                                    fwd_scored_MeV * 1.0e6);
                                const auto fwd_esc_fixed = static_cast<std::uint64_t>(
                                    static_cast<double>(fwd_escaped_MeV) * 1.0e6);
                                const auto fwd_kept_fixed = static_cast<std::uint64_t>(
                                    static_cast<double>(fwd_kept_MeV) * 1.0e6);
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_fwd_moved(schneider_delta_energy_device[3]);
                                atomic_fwd_moved.fetch_add(fwd_fixed);
                                if (fwd_kept_fixed > 0U) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_fwd_kept(schneider_delta_energy_device[4]);
                                    atomic_fwd_kept.fetch_add(fwd_kept_fixed);
                                }
                                if (fwd_esc_fixed > 0U) {
                                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_fwd_escape(schneider_delta_energy_device[5]);
                                    atomic_fwd_escape.fetch_add(fwd_esc_fixed);
                                }
                            }
                        }
                    if (bin != pending_primary_bin) {
                        if (pending_primary_depth_MeV > 0.0) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_dose(dose_device[pending_primary_bin]);
                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
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

                    // Keep the legacy unrestricted-depth escape convention, but
                    // remove relocated energy now tallied at its destination.
                    pending_primary_depth_MeV += deposited_MeV - forward_shifted_MeV -
                        transverse_relocated_MeV;
                    pending_primary_depth_MeV += same_voxel_electron_packet_MeV;
                    if (enable_voxel_scoring && voxel_index >= 0) {
                        pending_primary_voxel_MeV += local_voxel_deposit_MeV;
                        pending_primary_voxel_MeV += same_voxel_electron_packet_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(
                                deposited_MeV - forward_shifted_MeV -
                                transverse_relocated_MeV - transverse_escaped_MeV)+
                                static_cast<DepthAtomicT>(same_voxel_electron_packet_MeV));
                        }
                    }
                    history_deposited_MeV += deposited_MeV-water_physical_escape_MeV-material_untracked_MeV;
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device,
                        enable_voxel_scoring && voxel_index >= 0,
                        deposited_MeV - delta_tail_escaped_scorer_MeV-water_physical_escape_MeV-material_untracked_MeV);
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device, false,
                        delta_tail_escaped_scorer_MeV);
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

                    const float seg_dir_x = direction_x;
                    const float seg_dir_y = direction_y;
                    const float seg_dir_z = direction_z;

                    if (enable_multiple_scattering) {
                        auto radiation_length_g_per_cm2 =
                            static_cast<float>(active_water_radiation_length);
                        if (enable_ct_grid && in_ct && enable_ct_material_mcs) {
                            if (ct_material_ids_are_schneider_sections) {
                                radiation_length_g_per_cm2 = static_cast<float>(
                                    select_transport_radiation_length_g_per_cm2(
                                        ct_material_ids_are_schneider_sections, in_ct,
                                        ct_material, enable_ct_material_mcs,
                                        water_radiation_length_g_per_cm2));
                            } else {
                                radiation_length_g_per_cm2 = static_cast<float>(
                                    ct_material_radiation_length_g_per_cm2(
                                        ct_material_class(ct_material, false)));
                            }
                        } else if (in_insert) {
                            radiation_length_g_per_cm2 = insert_radiation_length_g_per_cm2;
                        } else if (slab_layer_count > 0) {
                            radiation_length_g_per_cm2 =
                                slab_radiation_lengths_device[layer_for_material];
                        }
                        float theta_x = 0.0F;
                        float theta_y = 0.0F;
                        constexpr float two_pi = 6.2831853071795864769F;
                        {
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

                    position_x_mm += seg_dir_x * step_mm;
                    position_y_mm += seg_dir_y * step_mm;
                    position_z_mm += seg_dir_z * step_mm;

                    if (enable_ct_grid && in_ct && ct_clamp_res.hit_face && !inelastic_this_step && !ct_elastic_this_step) {
                        if ((ct_clamp_res.axis_mask & 1) != 0 && sycl::fabs(seg_dir_x) > 1.0e-6F) {
                            const float fx = (position_x_mm - ct_origin_x) / ct_spacing_x;
                            const int face_x = static_cast<int>(sycl::round(fx));
                            const float b_x = ct_origin_x + static_cast<float>(face_x) * ct_spacing_x;
                            position_x_mm = sycl::nextafter(b_x, seg_dir_x > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 2) != 0 && sycl::fabs(seg_dir_y) > 1.0e-6F) {
                            const float fy = (position_y_mm - ct_origin_y) / ct_spacing_y;
                            const int face_y = static_cast<int>(sycl::round(fy));
                            const float b_y = ct_origin_y + static_cast<float>(face_y) * ct_spacing_y;
                            position_y_mm = sycl::nextafter(b_y, seg_dir_y > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                        if ((ct_clamp_res.axis_mask & 4) != 0 && sycl::fabs(seg_dir_z) > 1.0e-6F) {
                            const float fz = (position_z_mm - ct_origin_z) / ct_spacing_z;
                            const int face_z = static_cast<int>(sycl::round(fz));
                            const float b_z = ct_origin_z + static_cast<float>(face_z) * ct_spacing_z;
                            position_z_mm = sycl::nextafter(b_z, seg_dir_z > 0.0F ? 1.0e30F : -1.0e30F);
                        }
                    }

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

                    // Elastic endpoint: full bank uses target-specific TOPAS samples;
                    // the separate legacy diagnostic retains its explicit isotropic law.
                    // Supported recoils are queued; below-cutoff energy is scored locally.
                    if(use_all_elastic && ct_elastic_this_step && energy_MeV>energy_cutoff_MeV &&
                       all_elastic.rate(16,use_unified_water?25:interaction_section,energy_MeV*inverse_mass_number)==0) {
                        ct_elastic_this_step=false;
                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[10]).fetch_add(1);
                    }
                    if ((use_ct_elastic || use_all_elastic) && ct_elastic_this_step &&
                        energy_MeV > energy_cutoff_MeV) {
                        const float u_tgt = rng::uniform01(
                            spot_seed, rng_history, steps, 12);
                        const auto draw = use_all_elastic ? all_elastic.draw(16,
                            use_unified_water?25:interaction_section,energy_MeV,direction_x,direction_y,direction_z,
                            u_tgt,rng::uniform01(spot_seed,rng_history,steps,70),
                            rng::uniform01(spot_seed,rng_history,steps,71),rng::uniform01(spot_seed,rng_history,steps,11)) : ElasticDraw{};
                        if(use_all_elastic && !draw.valid) {
                            sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[5]).fetch_add(1);
                            break;
                        }
                        const int target_z=use_all_elastic?draw.target_z:(ct_elastic_all_targets?
                            sample_schneider_target_device(schneider_ct_device_ctx.elastic_sampler,interaction_section,
                                energy_MeV*inverse_mass_number,u_tgt):1);
                        const int target_idx=carbon::elastic_target_index_from_z(target_z);
                        const int target_a=use_all_elastic?draw.target_a:static_cast<int>(carbon::kElasticTargetMassU[target_idx<0?0:target_idx]);
                        if (target_idx >= 0) {
                            const auto scat=use_all_elastic?draw.outcome:carbon::sample_c12_target_elastic(
                                energy_MeV,direction_x,direction_y,direction_z,
                                rng::uniform01(spot_seed,rng_history,steps,10),rng::uniform01(spot_seed,rng_history,steps,11),
                                carbon::kElasticTargetMassU[target_idx]);
                            if(use_all_elastic)sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[0]).fetch_add(1);
                            energy_MeV = scat.projectile_ke_MeV;
                            direction_x = scat.proj_dir_x;
                            direction_y = scat.proj_dir_y;
                            direction_z = scat.proj_dir_z;
                            if ((carbon::get_charged_species_idx(target_z,target_a)>=0 ||
                                 (use_all_elastic&&recoil_stopping.projectile(target_z,target_a)>=0)) && enable_secondary_transport &&
                                scat.recoil_ke_MeV > energy_cutoff_MeV &&
                                secondary_queue_device != nullptr) {
                                // Elastic recoil born accounting: mirrors the
                                // inelastic born/queued pair so the born
                                // conservation gate stays exact.
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedBorn);
                                auto count_ref = sycl::atomic_ref<
                                    uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                                    *secondary_count_device);
                                const auto base_idx = count_ref.fetch_add(1U);
                                if (base_idx < max_secondaries) {
                                    SecondaryParticle proton{};
                                    proton.z = target_z;
                                    proton.a = target_a;
                                    if(carbon::get_charged_species_idx(target_z,target_a)<0)proton.generation=cinel02_max_secondary_inelastic_generations;
                                    proton.energy_MeV = scat.recoil_ke_MeV;
                                    proton.pos_x_mm = position_x_mm;
                                    proton.pos_y_mm = position_y_mm;
                                    proton.pos_z_mm = position_z_mm;
                                    proton.dir_x = scat.recoil_dir_x;
                                    proton.dir_y = scat.recoil_dir_y;
                                    proton.dir_z = scat.recoil_dir_z;
                                    proton.weight = 1.0F;
                                    proton.parent_history = global_history;
                                    proton.rng_stream = rng::child_stream(
                                        rng_history, rng::branch_tag(
                                            rng::branch_role_primary_charged, steps));
                                    secondary_queue_device[base_idx] = proton;
                                    if(use_all_elastic){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[1]).fetch_add(scat.recoil_ke_MeV);}
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::PrimaryChargedQueued);
                                    if (cinel02_species_energy_device != nullptr) {
                                        cinel02_record_queued_secondary_birth_device(
                                            cinel02_species_energy_device, proton.z, proton.a,
                                            proton.energy_MeV);
                                    }
                                } else if (secondary_overflow_count_device != nullptr) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_ov(*secondary_overflow_count_device);
                                    atomic_ov.fetch_add(1U);
                                    if(use_all_elastic){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[2]).fetch_add(scat.recoil_ke_MeV);sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[11]).fetch_add(1);}
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_energy_device).fetch_add(scat.recoil_ke_MeV);
                                }
                            } else if(use_all_elastic && scat.recoil_ke_MeV>energy_cutoff_MeV) {
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[3]).fetch_add(1);
                                sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(untracked_nuclear_device[global_history]).fetch_add(scat.recoil_ke_MeV);
                            } else if (scat.recoil_ke_MeV > 0.0F) {
                                if (enable_voxel_scoring && voxel_index >= 0 &&
                                    voxel_index < number_of_voxels) {
                                    pending_primary_voxel_MeV += scat.recoil_ke_MeV;
                                }
                                history_deposited_MeV += scat.recoil_ke_MeV;
                                if(use_all_elastic){sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[0]).fetch_add(scat.recoil_ke_MeV);}
                                pending_primary_depth_MeV += scat.recoil_ke_MeV;
                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                    enable_voxel_scoring&&voxel_index>=0,scat.recoil_ke_MeV);
                            }
                        }
                    }


                    if (use_schneider_primary_xs && (in_ct || use_unified_water) && inelastic_this_step) {
                        if (is_primary_attenuation_only) {
                            primary_inelastic_occurred = true;
                            if (first_interactions_device != nullptr && first_interactions_count_device != nullptr) {
                                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    count_ref(*first_interactions_count_device);
                                const auto slot = count_ref.fetch_add(1U);
                                if (slot < number_of_histories) {
                                    first_interactions_device[slot] = PrimaryFirstInteractionRecord{
                                        position_x_mm,
                                        position_y_mm,
                                        position_z_mm,
                                        energy_MeV * inverse_mass_number,
                                        interaction_section,
                                        interaction_density
                                    };
                                }
                            }
                            if (primary_terminal_counts_device != nullptr) {
                                sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    term_ref(primary_terminal_counts_device[0]);
                                term_ref.fetch_add(1U);
                            }
                            if (untracked_nuclear_device != nullptr) {
                                untracked_nuclear_device[global_history] = energy_MeV;
                            }
                            energy_MeV = 0.0F;
                            break;
                        }

                        // Production Schneider CT primary inelastic collision.
                        // Exact (C12, target_Z) channel, bounded energy domain,
                        // stochastic bracketing. Tag 15 drives the bracket
                        // choice, tag 16 the intra-node event pick.
                        const float cur_primary_e_u = energy_MeV * inverse_mass_number;
                        const float u_target = rng::uniform01(spot_seed, rng_history, steps, 14);
                        // v3: masked draw from the same partials as the hazard.
                        // A non-positive result (masked hazard fired but
                        // slowing emptied every channel before the collision
                        // point) is a primary post-EM null collision: no
                        // package query, track continues, never
                        // UnsupportedTargets. v1 keeps the legacy call and
                        // flows to lookup EXACTLY as before.
                        int target_z = 0;
                        if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                            const auto masked = schneider_ct_device_ctx.primary_rates(
                                interaction_section, cur_primary_e_u);
                            target_z = sample_masked_schneider_target_device(
                                masked.partials, u_target);
                            if (unified_em) {
                            // Dedicated dimension 44; do not reuse source, loss,
                            // target, event or electron-response variates.
                            const bool accepted = primary_hadronic_cache.accept(
                                interaction_density * masked.total,
                                rng::uniform01(spot_seed, rng_history, steps, 44));
                            // Candidate-only: rejected proposal reuses the
                            // existing energy-preserving post-EM null route.
                            if (!accepted) target_z = 0;
                            }
                        } else {
                            target_z = sample_schneider_target_device(
                                schneider_ct_device_ctx.primary_sampler,
                                interaction_section,
                                cur_primary_e_u,
                                u_target);
                        }
                        if (schneider_ct_device_ctx.primary_sampler.rate_version == 3 &&
                            target_z <= 0) {
                            // Null collision: retain the surviving track and complete this EM step.
                            inelastic_this_step = false;
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::PrimaryPostEmNullCollisions);
                            schneider_float_add_device(
                                schneider_float_device,
                                SchneiderFloatSlot::PostEmNullEnergy,
                                sycl::fmax(0.0F, energy_MeV));
                        } else {

                        const float u_bracket = rng::uniform01(spot_seed, rng_history, steps, 15);
                        const float u_event = rng::uniform01(spot_seed, rng_history, steps, 16);
                        const auto primary_lookup = cinel03_lookup_event_device(
                            schneider_ct_device_ctx.c12_energy_nodes,
                            schneider_ct_device_ctx.c12_node_count,
                            schneider_ct_device_ctx.c12_event_offsets,
                            schneider_ct_device_ctx.c12_event_indices,
                            schneider_ct_device_ctx.c12_total_events,
                            6, 12, target_z,
                            cur_primary_e_u,
                            u_bracket, u_event);
                        schneider_record_lookup_device(schneider_diag_device,
                                                       schneider_float_device,
                                                       true, primary_lookup,
                                                       energy_MeV);

                        if (primary_lookup.status != Cinel03LookupStatus::Hit) {
                            // Per-miss log: recompute the macro total exactly
                            // as at hazard time (density x mass, once).
                            // Per-miss macro total recomputed exactly as at
                            // hazard time: v3 uses the masked binary total
                            // (no CSV exists for v3), v1 the CSV XS table.
                            float primary_miss_macro = 0.0F;
                            if (schneider_ct_device_ctx.primary_sampler.rate_version == 3) {
                                primary_miss_macro = interaction_density *
                                    schneider_ct_device_ctx.primary_rates(
                                        interaction_section, cur_primary_e_u).total;
                            } else {
                                primary_miss_macro = interaction_density *
                                    schneider_primary_mass_xs(
                                        schneider_primary_xs_device, schneider_xs_sections,
                                        schneider_xs_energies, schneider_xs_e_min, schneider_xs_inv_dE,
                                        interaction_section, cur_primary_e_u);
                            }
                            schneider_log_miss_device(
                                schneider_miss_device, schneider_miss_count_device,
                                kSchneiderMissLogCap, true,
                                6, 12, target_z,
                                interaction_section > 255 ? 255
                                                          : static_cast<std::uint8_t>(interaction_section),
                                0, primary_lookup, cur_primary_e_u, deposited_MeV,
                                primary_miss_macro, 0.0F, energy_MeV,
                                spot_initial_energy_MeV);
                            if (untracked_nuclear_device != nullptr) {
                                untracked_nuclear_device[global_history] = energy_MeV;
                            }
                            energy_MeV = 0.0F;
                            break;
                        }


                        // A hazard proposal (including a post-EM null collision)
                        // is not an inelastic reaction. Mark only an event hit.
                        primary_inelastic_occurred = true;
                        const auto& event = schneider_ct_device_ctx.c12_interactions[primary_lookup.event_index];
                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);

                        pending_primary_depth_MeV += local_deposit;
                        if (enable_voxel_scoring && voxel_index >= 0) {
                            pending_primary_voxel_MeV += local_deposit;
                        }
                        history_deposited_MeV += local_deposit;
                        grid_deposit_split_device(
                            grid_deposited_in_device, grid_deposited_out_device,
                            enable_voxel_scoring && voxel_index >= 0, local_deposit);

                        float charged_accounted_MeV = 0.0F;
                        float neutral_accounted_MeV = 0.0F;
                        float unsupported_accounted_MeV = 0.0F;

                        const std::uint32_t prod_offset = event.product_offset;
                        const std::uint32_t prod_count = event.direct_product_count;

                        // Finite event-library azimuths are not a preferred
                        // laboratory direction. Rotate the whole event once.
                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                            spot_seed, rng_history, steps, cinel03_event_azimuth_dimension);
                        const float event_cos = sycl::cos(event_phi);
                        const float event_sin = sycl::sin(event_phi);

                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                            if (prod_offset + ip >= schneider_ct_device_ctx.c12_total_products) break;
                            const auto& product = schneider_ct_device_ctx.c12_products[prod_offset + ip];

                            if (product.role == 2) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                unsupported_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                continue;
                            }
                            if (product.z <= 0 || product.a <= 0) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                neutral_accounted_MeV += ke;
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                continue;
                            }
                            if (product.role != 0) {
                                continue;
                            }
                            // Be6 keeps the frozen TopasCompatKill policy:
                            // explicit counter + energy, never queued.
                            if (product.z == 4 && product.a == 6) {
                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryBe6Kills);
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::Be6TopasCompatKills);
                                schneider_float_add_device(
                                    schneider_float_device,
                                    SchneiderFloatSlot::Be6KillEnergy, ke);
                                continue;
                            }
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::PrimaryChargedBorn);

                            if (enable_secondary_transport && product.kinetic_energy_MeV > energy_cutoff_MeV) {
                                const auto local_direction = rotate_cinel03_event_azimuth(
                                    product.local_direction_x, product.local_direction_y,
                                    product.local_direction_z, event_cos, event_sin);
                                const auto child_direction = rotate_local_direction(
                                    local_direction.x,
                                    local_direction.y,
                                    local_direction.z,
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
                                        child.weight = 1.0F;
                                        child.parent_history = global_history;
                                        child.rng_stream = rng::event_product_stream(
                                            rng_history, steps, rng::branch_role_primary_charged, ip);
                                        secondary_queue_device[output] = child;
                                        charged_accounted_MeV += product.kinetic_energy_MeV;
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryChargedQueued);
                                        if (cinel02_species_energy_device != nullptr) {
                                            cinel02_record_queued_secondary_birth_device(
                                                cinel02_species_energy_device, child.z, child.a,
                                                child.energy_MeV);
                                        }
                                    } else {
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::PrimaryQueueOverflows);
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::QueueOverflows);
                                        if (secondary_overflow_count_device != nullptr) {
                                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_ov(*secondary_overflow_count_device);
                                            atomic_ov.fetch_add(1U);
                                        }
                                        if (secondary_overflow_energy_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_ove(*secondary_overflow_energy_device);
                                            atomic_ove.fetch_add(product.kinetic_energy_MeV);
                                        }
                                    }
                                }
                            } else {
                                pending_primary_depth_MeV += product.kinetic_energy_MeV;
                                if (enable_voxel_scoring && voxel_index >= 0) {
                                    pending_primary_voxel_MeV += product.kinetic_energy_MeV;
                                }
                                history_deposited_MeV += product.kinetic_energy_MeV;
                                grid_deposit_split_device(
                                    grid_deposited_in_device,
                                    grid_deposited_out_device,
                                    enable_voxel_scoring && voxel_index >= 0,
                                    product.kinetic_energy_MeV);
                                charged_accounted_MeV += product.kinetic_energy_MeV;
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::PrimaryChargedCutoffKills);
                            }
                        }

                        schneider_float_add_device(
                            schneider_float_device, SchneiderFloatSlot::ReactionQResidual,
                            sycl::fmax(0.0F, energy_MeV - local_deposit -
                                                  charged_accounted_MeV -
                                                  neutral_accounted_MeV -
                                                  unsupported_accounted_MeV));
                        // NO-DOUBLE-COUNT RULE: the legacy untracked sink below
                        // already contains neutral + unsupported + Q-residual
                        // energy (E - local - charged). The split float slots
                        // above are INFORMATIONAL ONLY and must never be added
                        // into the global closure alongside untracked; the
                        // closure in run_quality uses the legacy sink alone.
                        // A unit test pins this (split fields leave the
                        // residual bitwise unchanged).
                        const float untracked_MeV = sycl::fmax(0.0F, energy_MeV - local_deposit - charged_accounted_MeV);
                        if (untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_untracked(untracked_nuclear_device[global_history]);
                            atomic_untracked.fetch_add(untracked_MeV);
                        }

                        if (primary_terminal_counts_device != nullptr) {
                            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                term_ref(primary_terminal_counts_device[0]);
                            term_ref.fetch_add(1U);
                        }
                        energy_MeV = 0.0F;
                        break;
                        }  // else of the v3 masked-sampler empty-draw guard
                    }



                    ++steps;
                }

                if(unified_em && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit(unified_audit,unified_primary_audit);

                const bool inside_phantom = !unified_primary_escaped_ct && (position_z_mm >= 0.0F && position_z_mm < phantom_length_mm);

                if (energy_MeV > 0.0F && energy_MeV <= energy_cutoff_MeV && inside_phantom) {
                    const auto cutoff_energy_MeV = energy_MeV;
                    pending_primary_depth_MeV += cutoff_energy_MeV;
                    if (enable_voxel_scoring && pending_primary_voxel >= 0 &&
                        pending_primary_voxel < static_cast<std::size_t>(number_of_voxels)) {
                        pending_primary_voxel_MeV += cutoff_energy_MeV;
                        if (in_fov_dose_device != nullptr && pending_primary_bin >= 0 &&
                            pending_primary_bin < static_cast<int>(number_of_bins)) {
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_in_fov(in_fov_dose_device[pending_primary_bin]);
                            atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(cutoff_energy_MeV));
                        }
                    }
                    history_deposited_MeV += cutoff_energy_MeV;
                    grid_deposit_split_device(
                        grid_deposited_in_device, grid_deposited_out_device,
                        enable_voxel_scoring && pending_primary_voxel >= 0 &&
                            pending_primary_voxel <
                                static_cast<std::size_t>(number_of_voxels),
                        cutoff_energy_MeV);
                    if (cutoff_stopped_energy_device != nullptr) {
                        cutoff_stopped_energy_device[global_history] = cutoff_energy_MeV;
                    }
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
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[2]); // [2] = stopped_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                    energy_MeV = 0.0F;
                } else if (energy_MeV > 0.0F && !inside_phantom) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[1]); // [1] = escaped_ct_without_inelastic
                        term_ref.fetch_add(1U);
                    }
                } else if (!primary_inelastic_occurred) {
                    if (primary_terminal_counts_device != nullptr) {
                        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            term_ref(primary_terminal_counts_device[3]); // [3] = other_terminal
                        term_ref.fetch_add(1U);
                    }
                    if (energy_MeV > 0.0F) {
                        if (other_terminal_energy_device != nullptr) {
                            other_terminal_energy_device[global_history] = energy_MeV;
                        }
                        energy_MeV = 0.0F;
                    }
                }

                if (pending_primary_depth_MeV > 0.0) {
                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_dose(dose_device[pending_primary_bin]);
                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_primary_depth_MeV));
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

                schneider_diag_add_device(schneider_diag_device,
                                              SchneiderDiagSlot::PrimaryRateQueries,
                                              local_schneider_rate_queries);
                deposited_device[global_history] = history_deposited_MeV;
                escaped_device[global_history] = energy_MeV+history_water_electron_escaped_MeV;
                steps_device[global_history] = steps;
            });
        kernel_event.wait_and_throw();
        primary_kernel_seconds += event_duration_seconds(kernel_event);
        std::cout << "[progress] primary batch completed: "
                  << (hist_offset + chunk_count) << "/" << number_of_histories
                  << " histories" << std::endl;
    }

    double secondary_kernel_seconds = 0.0;
    const bool enable_secondary_unified_em = config.enable_secondary_unified_em;
    const bool segment_secondaries = config.secondary_step_chunking;
    // Reorder indices only: RNG streams and parent histories belong to particles.
    const bool group_secondaries = config.secondary_species_grouping &&
        enable_inelastic && enable_secondary_transport;
    std::uint32_t* secondary_order = nullptr;
    std::uint32_t* secondary_group_counts = nullptr;
    std::uint32_t* secondary_group_cursors = nullptr;
    double secondary_group_seconds = 0.0;
    if (group_secondaries && secondary_queue_device) {
        secondary_order = mem_tracker.allocate<std::uint32_t>(max_secondaries);
        secondary_group_counts = mem_tracker.allocate<std::uint32_t>(19);
        secondary_group_cursors = mem_tracker.allocate<std::uint32_t>(19);
        if (!secondary_order || !secondary_group_counts || !secondary_group_cursors)
            throw std::bad_alloc();
    }
    std::cout << "[secondary-schedule] group=" << group_secondaries
              << "; particle state and RNG identities preserved\n";
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
                const auto batch_begin = generation_begin;
                if (group_secondaries) {
                    const auto grouping_start = std::chrono::steady_clock::now();
                    queue.fill(secondary_group_counts, 0u, 19).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const auto frag = secondary_queue_device[generation_begin + id[0]];
                            const int species = carbon::get_charged_species_idx(frag.z, frag.a);
                            const unsigned bucket = species >= 0 ? unsigned(species) : 18u;
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                count(secondary_group_counts[bucket]);
                            count.fetch_add(1);
                        }).wait_and_throw();
                    queue.single_task([=]() {
                        unsigned total = 0;
                        for (unsigned k = 0; k < 19; ++k) {
                            secondary_group_cursors[k] = total;
                            total += secondary_group_counts[k];
                        }
                    }).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(generation_end - generation_begin),
                        [=](sycl::id<1> id) {
                            const unsigned index = generation_begin + id[0];
                            const auto frag = secondary_queue_device[index];
                            const int species = carbon::get_charged_species_idx(frag.z, frag.a);
                            const unsigned bucket = species >= 0 ? unsigned(species) : 18u;
                            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>
                                cursor(secondary_group_cursors[bucket]);
                            secondary_order[cursor.fetch_add(1)] = index;
                        }).wait_and_throw();
                    secondary_group_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - grouping_start).count();
                }
                const unsigned resume_capacity = generation_end - generation_begin;
                const unsigned state_capacity = segment_secondaries ? resume_capacity : 0;
                auto* resume_states = mem_tracker.allocate<SecondaryResumeState>(state_capacity);
                auto* resume_ready = mem_tracker.allocate<unsigned>(state_capacity);
                auto* active_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* next_order = mem_tracker.allocate<unsigned>(state_capacity);
                auto* keep = mem_tracker.allocate<unsigned>(state_capacity);
                auto* ranks = mem_tracker.allocate<unsigned>(state_capacity);
                auto* block_counts = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                auto* block_offsets = mem_tracker.allocate<unsigned>((state_capacity + 255) / 256);
                auto* active_count = mem_tracker.allocate<unsigned>(segment_secondaries ? 1 : 0);
                if (segment_secondaries) {
                    if (!resume_states || !resume_ready || !active_order || !next_order ||
                        !keep || !ranks || !block_counts || !block_offsets || !active_count)
                        throw std::runtime_error("Secondary continuation allocation failed; reduce histories per shard");
                    queue.fill(resume_ready, 0u, resume_capacity).wait_and_throw();
                    queue.parallel_for(sycl::range<1>(resume_capacity), [=](sycl::id<1> i) {
                        active_order[i[0]] = i[0];
                    }).wait_and_throw();
                    std::cout << "[segmented-secondary] step_limit=" << kSecondarySegmentSteps << " tail_threshold=8192 state_bytes="
                              << sizeof(SecondaryResumeState) << " capacity=" << resume_capacity << "\n";
                }
                unsigned resume_active = resume_capacity, segment_rounds = 0;
                double compact_seconds = 0;
                while (resume_active) {
                const bool finish_tail = !segment_secondaries || resume_active < 8192;
                auto sec_event = queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<CarbonSecondaryTransportKernel<EmMode>>(
                    sycl::range<1>(resume_active),
                    [=](sycl::id<1> item_id) {
                        const unsigned state_idx = segment_secondaries ? active_order[item_id[0]] : item_id[0];
                        const bool resumed = segment_secondaries && resume_ready[state_idx] != 0;
                        if (segment_secondaries) keep[item_id[0]] = 0;
                        const auto sec_idx = group_secondaries ? secondary_order[state_idx]
                            : generation_begin + state_idx;
                        const auto frag = secondary_queue_device[sec_idx];
                        if (frag.z <= 0 || frag.a <= 0) return;
                        const auto charged_origin_category =
                            charged_origin_category_from_fragment(
                                charged_dose_category(frag.z, frag.a));
                        const auto charged_origin_voxel_offset =
                            charged_origin_category * number_of_voxels;
                        const auto be_isotope_category =
                            be_isotope_origin_category(frag.z, frag.a);
                        const auto he_isotope_category =
                            he_isotope_origin_category(frag.z, frag.a);
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
                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dose(dose_device[bin_z]);
                                atomic_dose.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
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
                                        score_he_isotope_origin_voxel_device(
                                            he_isotope_origin_voxel_dose_device,
                                            he_isotope_category, number_of_voxels, cur_voxel,
                                            static_cast<DoseAtomicT>(frag.energy_MeV));
                                    }
                                    if (in_fov_dose_device != nullptr) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[bin_z]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(frag.energy_MeV));
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
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                    frag.energy_MeV);
                                {
                                    const auto sqx = static_cast<int>(
                                        (frag.pos_x_mm - voxel_min_x_mm) *
                                        inverse_voxel_size_x_mm);
                                    const auto sqy = static_cast<int>(
                                        (frag.pos_y_mm - voxel_min_y_mm) *
                                        inverse_voxel_size_y_mm);
                                    const auto sqz = static_cast<int>(
                                        frag.pos_z_mm * inverse_depth_bin_width_mm);
                                    const bool sq_in_grid =
                                        enable_voxel_scoring && sqx >= 0 &&
                                        sqx < static_cast<int>(voxel_bins_x) &&
                                        sqy >= 0 &&
                                        sqy < static_cast<int>(voxel_bins_y) &&
                                        sqz >= 0 &&
                                        sqz < static_cast<int>(number_of_bins);
                                    grid_deposit_split_device(
                                        grid_deposited_in_device,
                                        grid_deposited_out_device, sq_in_grid,
                                        frag.energy_MeV);
                                }
                            }
                            return;
                        }

                        const auto frag_a = static_cast<float>(frag.a);
                        const auto frag_inv_a = 1.0F / frag_a;
                        const auto charged_sp_idx = carbon::get_charged_species_idx(static_cast<int>(frag.z), static_cast<int>(frag.a));
                        const auto ledger_species_idx = charged_sp_idx;
                        const int recoil_sp_idx=use_all_elastic?recoil_stopping.projectile(frag.z,frag.a):-1;
                        const bool generic_recoil=charged_sp_idx<0&&recoil_sp_idx>=0;
                        if (charged_sp_idx < 0 && !generic_recoil) {
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
                                static_cast<std::size_t>(charged_sp_idx<0?0:charged_sp_idx) * table_size];

                        bool sec_terminal_recorded = false;
                        bool unified_secondary_escaped_ct = false;
                        ContinuousSpeciesTrackTally continuous_species_tally;
                        double he4_audit[6]{};
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

                        const bool unified_secondary=unified_em && !generic_recoil && enable_secondary_unified_em;
                        const int unified_secondary_species=unified_secondary?unified_device.species_index(frag.z,frag.a):-1;
                        UnifiedEmClock unified_secondary_clock;
                        std::uint64_t unified_secondary_counter=0;
                        // Diagnostic species profile: 20 slots x {steps,
                        // short-range steps, deposited uMeV}. Zero production
                        // impact (compiled out when profiling is off).
                        std::uint64_t sec_prof[60];
                        if constexpr(CARBON_SECONDARY_STEP_PROFILE)
                            for(int sec_pi=0;sec_pi<60;++sec_pi)sec_prof[sec_pi]=0;
                        UnifiedEmState unified_secondary_state;
                        std::array<std::uint64_t,8> unified_secondary_audit{};
                        auto unified_secondary_uniform=[&](){return (rng::random_u32(2026,frag.rng_stream,unified_secondary_counter++,121)>>8)*0x1p-24f;};
                        auto unified_secondary_count=[&](int index,std::uint64_t count=1){
                            if constexpr(CARBON_EM_LOCAL_AUDIT) unified_secondary_audit[index]+=count;
                            else {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> a(unified_audit[index]);a.fetch_add(count);
                            }
                        };
                        uint32_t sec_steps = 0;
                        constexpr uint32_t kSecondaryMaxSteps = 30000U;
                        std::uint64_t local_sec_rate_queries = 0;
                        std::uint64_t local_sec_steps = 0;
                        const int schneider_reg_idx =
                            secondary_projectile_lut_index_device(
                                      schneider_ct_device_ctx.sec_proj_keys,
                                      schneider_ct_device_ctx.sec_num_projectiles,
                                      frag.z, frag.a);
                        if(!resumed){
                        // A track that reaches the stepping loop has actually
                        // started charged transport: count it and itemize its
                        // birth kinetic energy (not a config-switch inference).
                        schneider_diag_increment_device(
                            schneider_diag_device,
                            SchneiderDiagSlot::SecondaryTracksStarted);
                        schneider_float_add_device(
                            schneider_float_device,
                            SchneiderFloatSlot::SecondaryTransportBirthEnergy,
                            sycl::fmax(0.0F, frag.energy_MeV));
                        // Per-track unsupported-projectile accounting (once per
                        // track, not per step): generation-eligible tracks whose
                        // (Z/A) has no secondary rate-table registry entry.
                        // The per-step evaluation counter below
                        // (UnsupportedProjectileSteps) is diagnostic only and
                        // must NOT drive coverage gates.
                        // v3: registry lookup over the uploaded bundle-ordered
                        // keys (any (Z,A) in the bundle is supported).

                        if (schneider_ct_device_ctx.uses_cinel03() &&
                            frag.generation < cinel02_max_secondary_inelastic_generations &&
                            schneider_reg_idx < 0) {
                            schneider_diag_increment_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::UnsupportedProjectileTracks);
                            if (frag.z == 4 && frag.a == 6) {
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::UnsupportedBe6Tracks);
                            }
                            schneider_energy_add_device(
                                schneider_diag_device,
                                SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV,
                                sycl::fmax(0.0F, frag.energy_MeV));
                            // Per-(Z/A) census record for the coverage audit.
                            schneider_log_unsupported_track_device(
                                schneider_track_log_device, schneider_track_count_device,
                                kSchneiderTrackLogCap,
                                frag.z, frag.a, frag.generation,
                                sycl::fmax(0.0F, frag.energy_MeV),
                                frag.pos_x_mm, frag.pos_y_mm, frag.pos_z_mm);
                        }
                        }
                        if(resumed){
                            const auto saved=resume_states[state_idx];
                            sec_terminal_recorded=saved.sec_terminal_recorded;
                            unified_secondary_escaped_ct=saved.unified_secondary_escaped_ct;
                            continuous_species_tally=saved.continuous_species_tally;
                            sec_e=saved.sec_e;
                            sec_x=saved.sec_x;
                            sec_y=saved.sec_y;
                            sec_z=saved.sec_z;
                            sec_dx=saved.sec_dx;
                            sec_dy=saved.sec_dy;
                            sec_dz=saved.sec_dz;
                            pending_sec_depth_MeV=saved.pending_sec_depth_MeV;
                            pending_sec_voxel_MeV=saved.pending_sec_voxel_MeV;
                            pending_sec_bin=saved.pending_sec_bin;
                            pending_sec_voxel=saved.pending_sec_voxel;
                            unified_secondary_counter=saved.unified_secondary_counter;
                            unified_secondary_state=saved.unified_secondary_state;
                            unified_secondary_state.tables=&unified_device;
                            unified_secondary_audit=saved.unified_secondary_audit;
                            sec_steps=saved.sec_steps;
                            local_sec_rate_queries=saved.local_sec_rate_queries;
                            local_sec_steps=saved.local_sec_steps;
                            for(int j=0;j<6;++j)he4_audit[j]=saved.he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)sec_prof[j]=saved.sec_prof[j];
                        }
                        unsigned segment_steps=0;bool segment_paused=false;
                        while (sec_e > energy_cutoff_MeV && sec_z >= 0.0F && sec_z < phantom_length_mm &&
                               sec_steps < kSecondaryMaxSteps) {
                            if(!finish_tail && segment_steps>=kSecondarySegmentSteps){segment_paused=true;break;}
                            ++segment_steps;
                            const auto bin_z = static_cast<int>(sec_z * inverse_depth_bin_width_mm);

                            if (bin_z < 0 || bin_z >= static_cast<int>(number_of_bins)) break;

                            const auto secondary_rate_query_energy_MeV =
                                sycl::fmax(0.0F, sec_e);
                            const auto sec_e_u = sec_e * frag_inv_a;
                            float sec_local_density_g_per_cm3 = water_density_g_per_cm3;
                            std::uint8_t sec_ct_material = 2U;
                            bool sec_in_ct = false;
                            if (enable_ct_grid) {
                                sec_in_ct = ct_sample(
                                    sec_x, sec_y, sec_z, ct_origin_x, ct_origin_y,
                                    ct_origin_z, ct_spacing_x, ct_spacing_y, ct_spacing_z,
                                    ct_nx, ct_ny, ct_nz, ct_density_device,
                                    ct_material_device, sec_local_density_g_per_cm3,
                                    sec_ct_material, sec_dx, sec_dy, sec_dz);
                            }
                            // The unified package has no exterior material. Leaving the
                            // CT volume is a charged-particle escape, not a lookup error.
                            if (unified_secondary && enable_ct_grid && !sec_in_ct) {
                                unified_secondary_escaped_ct = true;
                                break;
                            }
                            constexpr auto exposure_cell = std::numeric_limits<std::uint32_t>::max();
                            constexpr bool secondary_generation_eligible = false;
                            Cinel02DeviceRateLookup exposure_h_lookup{};
                            Cinel02DeviceRateLookup exposure_o_lookup{};
                            const bool exposure_h_covered = exposure_h_lookup.covered;
                            const bool exposure_o_covered = exposure_o_lookup.covered;
                            const bool exposure_rate_covered =
                                exposure_h_covered && exposure_o_covered;
                            const auto flt_idx = (sec_e_u - minimum_table_energy) * inverse_table_step;
                            auto sp_idx = static_cast<int>(sycl::floor(flt_idx));
                            sp_idx = sycl::max(0, sycl::min(sp_idx, static_cast<int>(table_size) - 2));
                            const auto sp_frac = sycl::clamp(flt_idx - static_cast<float>(sp_idx), 0.0F, 1.0F);
                            auto sec_sp = (ion_sp_table[sp_idx] +
                                           sp_frac * (ion_sp_table[sp_idx + 1] -
                                                      ion_sp_table[sp_idx]));
                            const bool sec_use_mass_sp_factor =
                                enable_ct_grid && sec_in_ct && use_ct_mass_sp &&
                                ct_mass_sp_factor_lut_device != nullptr &&
                                ct_n_mass_factors > 0U;
                            float sec_material_factor = 1.0F;
                            // Enabled bank is mandatory inside CT. A failed query aborts
                            // the run at readback; never substitute a water/material factor.
                            auto scheme2_linear_sp = [&](float energy_mevu) {
                                if(generic_recoil) {
                                    const float v=recoil_stopping.stopping(recoil_sp_idx,sec_in_ct?sec_ct_material:25,energy_mevu);
                                    if(!(v>0)){sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[6]).fetch_add(1);}
                                    return v*sec_local_density_g_per_cm3;
                                }
                                if (!(use_schneider_ion_sp && sec_in_ct)) return -1.0F;
                                const float unit_sp = schneider_ion_stopping_lookup(
                                    schneider_ion_sp_device, static_cast<std::size_t>(charged_sp_idx),
                                    static_cast<std::size_t>(sec_ct_material), energy_mevu);
                                if (!(unit_sp > 0) || !sycl::isfinite(sec_local_density_g_per_cm3) ||
                                    !(sec_local_density_g_per_cm3 > 0)) {
                                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>
                                        failures(*ion_stopping_failures);
                                    const auto failure_index = failures.fetch_add(1);
                                    if (failure_index == 0) {
                                        ion_stopping_failures[1] = frag.z;
                                        ion_stopping_failures[2] = frag.a;
                                        ion_stopping_failures[3] = sec_ct_material;
                                        ion_stopping_failures[4] = sycl::bit_cast<std::uint32_t>(energy_mevu);
                                        ion_stopping_failures[5] = sycl::bit_cast<std::uint32_t>(sec_local_density_g_per_cm3);
                                    }
                                    return -1.0F;
                                }
                                return unit_sp * sec_local_density_g_per_cm3;
                            };
                            bool sec_scheme2_hit = false;
                            {
                                const float scheme2_sp = scheme2_linear_sp(sec_e_u);
                                if ((generic_recoil||(use_schneider_ion_sp && sec_in_ct)) && !(scheme2_sp > 0)) break;
                                if (scheme2_sp > 0.0F) {
                                    sec_sp = scheme2_sp;
                                    sec_scheme2_hit = true;
                                }
                            }
                            if (sec_use_mass_sp_factor && !sec_scheme2_hit) {
                                sec_material_factor = ct_lookup_mass_sp_factor(
                                    ct_mass_sp_factor_lut_device,
                                    use_ct_density_mass_spr ? ct_density_spr_n_rho
                                                            : ct_n_mass_factors,
                                    table_size, use_ct_density_mass_spr,
                                    ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                    static_cast<std::uint32_t>(sec_ct_material),
                                    sec_local_density_g_per_cm3,
                                    static_cast<std::size_t>(sp_idx), sp_frac,
                                    [](float x) { return sycl::log(x); });
                            }
                            // Scheme-2 value is already linear stopping at
                            // local density; bypass water x density scaling.
                            if (!sec_scheme2_hit) {
                                sec_sp = secondary_material_stopping_power(
                                    sec_sp, enable_ct_grid && sec_in_ct,
                                    sec_local_density_g_per_cm3,
                                    sec_use_mass_sp_factor, sec_material_factor);
                            }

                            if (sec_sp <= 1.0e-6F) break;

                            float sec_step_mm = maximum_step_mm;
                            UnifiedEmStep unified_secondary_pre;
                            float unified_secondary_rate=0,unified_secondary_distance=std::numeric_limits<float>::infinity();
                            if(unified_secondary){
                                const int section=enable_ct_grid?(sec_in_ct?int(sec_ct_material):-2):-1;
                                if(!CARBON_EM_MATERIAL_CACHE || !unified_secondary_state.valid || unified_secondary_state.section!=section ||
                                   unified_secondary_state.density!=sec_local_density_g_per_cm3)
                                    unified_secondary_state=unified_device.select(section,sec_local_density_g_per_cm3,unified_secondary_species);
                                if(!unified_secondary_state.covers(sec_e)){
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,3,section,frag.z,frag.a,sec_e,sec_local_density_g_per_cm3,0);
                                    unified_secondary_count(0);break;}
                                unified_secondary_pre=unified_secondary_state.prepare(sec_e);
                                sec_step_mm=sycl::fmin((em_secondary_step_scale!=1.f?unified_secondary_state.research_step(sec_e,CARBON_EM_STEP_CACHE?unified_secondary_pre:unified_secondary_state.prepare(sec_e),em_secondary_step_scale):(CARBON_EM_STEP_CACHE?unified_secondary_state.step(unified_secondary_pre):unified_secondary_state.step(sec_e))),unified_secondary_distance);
                                // Bound combined mean loss, not the old restricted range alone.
                                const float delta_sp=unified_secondary_state.mix(unified_secondary_pre.lo.delta_stopping,unified_secondary_pre.hi.delta_stopping);
                                if(delta_sp>0) sec_step_mm=sycl::fmin(sec_step_mm,.01f*sec_e/sycl::fmax(1e-12f,delta_sp+unified_secondary_state.mix(unified_secondary_pre.lo.stopping,unified_secondary_pre.hi.stopping)));
                            }

                            if (sec_dz > 1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z + 1) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 1.0e-5F) sec_step_mm = sycl::fmin(sec_step_mm, dz_step);
                            } else if (sec_dz < -1.0e-6F) {
                                const auto bz = static_cast<float>(bin_z) * depth_bin_width_mm;
                                const auto dz_step = (bz - sec_z) / sec_dz;
                                if (dz_step > 1.0e-5F) sec_step_mm = sycl::fmin(sec_step_mm, dz_step);
                            }
                            CtFaceClampResult sec_face_clamp{sec_step_mm,false,0};
                            if (enable_ct_grid && sec_in_ct && ct_secondary_exact_faces) {
                                sec_face_clamp = clamp_step_to_ct_faces_exact(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, ct_origin_x, ct_origin_y, ct_origin_z,
                                    ct_spacing_x, ct_spacing_y, ct_spacing_z, ct_nx, ct_ny, ct_nz);
                                sec_step_mm = sec_face_clamp.step_mm;
                            } else if (enable_ct_grid && sec_in_ct) {
                                sec_step_mm = clamp_step_to_ct_faces_near_z_if_needed(
                                    sec_step_mm, sec_x, sec_y, sec_z, sec_dx, sec_dy,
                                    sec_dz, ct_origin_x, ct_origin_y, ct_origin_z,
                                    ct_spacing_x, ct_spacing_y, ct_spacing_z, ct_nx, ct_ny,
                                    ct_nz, ct_density_device, ct_material_device,
                                    sec_local_density_g_per_cm3, sec_ct_material,
                                    ct_skip_homogeneous_face_clamp, nullptr);
                            }
                            // Do not enlarge a real face distance to the legacy
                            // minimum step: that would cross the material again.
                            if (!unified_secondary && (!ct_secondary_exact_faces || !sec_face_clamp.hit_face))
                                sec_step_mm = sycl::fmax(sec_step_mm, 1.0e-5F);
                            // Generic EM-only recoils take CSDA steps without
                            // fluctuations: a 5% relative-loss cap keeps
                            // stopping variation second-order while cutting
                            // the 0.5%-capped micro-step count ~10x.
                            if(generic_recoil)sec_step_mm=sycl::fmin(sec_step_mm,0.05F*sec_e/sec_sp);
                            bool secondary_inelastic = false;
                            bool secondary_elastic = false;
                            // A nuclear hazard is not necessarily a replayed
                            // event. Keep this separate so lookup misses and
                            // invalid records follow null-collision transport
                            // semantics and still receive normal MCS.
                            bool secondary_replay_succeeded = false;
                            std::int16_t secondary_target_z = 0;
                            std::int16_t secondary_target_a = 0;
                            // v3: set when the post-EM sampler finds no
                            // in-domain channel (lookup skipped, never queried).
                            bool skip_schneider_lookup = false;
                            // Stash for the miss log: macro total rate at
                            // sampling (density x mass, density exactly once),
                            // sampled collision distance, and section.
                            float schneider_hazard_total_rate = 0.0F;
                            float schneider_hazard_step_mm = 0.0F;
                            std::uint8_t schneider_hazard_section = 255;
                            if (schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || use_unified_water) &&
                                frag.generation < cinel02_max_secondary_inelastic_generations) {
                                // Bundle-ordered registry LUT; no hardcoded isotope fallback.
                                const int proj_idx =
                                    secondary_projectile_lut_index_device(
                                              schneider_ct_device_ctx.sec_proj_keys,
                                              schneider_ct_device_ctx.sec_num_projectiles,
                                              frag.z, frag.a);
                                if (proj_idx < 0) {
                                    // Unsupported secondary projectile: explicit
                                    // per-STEP evaluation counter, never a silent
                                    // zero-rate step. (Generation-ineligible tracks
                                    // skip nuclear evaluation entirely: stage-C
                                    // transport without secondary reactions.)
                                    // Coverage gates use the per-TRACK counter
                                    // recorded at track start, not this value.
                                    schneider_diag_increment_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::UnsupportedProjectileSteps);
                                } else if (schneider_ct_device_ctx.sec_total_rates != nullptr) {
                                    const std::size_t section_id = use_unified_water ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    ++local_sec_rate_queries;
                                    // v3: single masked computation drives hazard
                                    // AND target sampling (same partials, same
                                    // mask), without an older total/sampler alternative.
                                    {
                                        // v3: hazard from the masked total at the
                                        // step-start energy E_h. Target sampling
                                        // is deferred to the lookup site (post-EM
                                        // collision energy E_c) so the mask and
                                        // the package query share one energy.
                                        // A post-EM empty draw remains a classified null collision.
                                        const auto sec_masked = schneider_ct_device_ctx.secondary_rates(
                                            proj_idx, section_id, sec_e_u);
                                        const float sec_macro_xs =
                                            sec_local_density_g_per_cm3 * sec_masked.total;
                                        schneider_hazard_total_rate = sec_macro_xs;
                                        if (sec_macro_xs > 0.0F) {
                                            float collision_distance = sec_step_mm;
                                            const auto collision = sample_exponential_collision(
                                                sec_macro_xs, sec_step_mm,
                                                rng::uniform01(2026, frag.rng_stream, sec_steps, 13));
                                            secondary_inelastic = collision.occurred;
                                            collision_distance = collision.distance_mm;
                                            if (secondary_inelastic) {
                                                sec_step_mm = collision_distance;
                                                schneider_hazard_total_rate = sec_macro_xs;
                                                schneider_hazard_step_mm = sec_step_mm;
                                                schneider_hazard_section = use_unified_water ? 255U : sec_ct_material;
                                            }
                                        }
                                    }
                                }
                            }

                            // Independent exponential elastic clock competes with the already
                            // sampled inelastic distance. Elastic is NOT generation-limited.
                            if(use_all_elastic && !generic_recoil && (sec_in_ct || use_unified_water)) {
                                const int p=all_elastic.projectile(frag.z,frag.a);
                                const float rate=all_elastic.rate(p,use_unified_water?25:sec_ct_material,sec_e_u);
                                if(rate<0) {
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[7]).fetch_add(1);
                                    break;
                                }
                                const auto collision = sample_exponential_collision(
                                    rate * sec_local_density_g_per_cm3, sec_step_mm,
                                    rng::uniform01(2026, frag.rng_stream, sec_steps, 70));
                                secondary_elastic = collision.occurred;
                                if (secondary_elastic) {
                                    sec_step_mm = collision.distance_mm;
                                    secondary_inelastic = false;
                                }
                            }
                            if(secondary_inelastic)schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryHazards);

                            // Record the actual charged-track exposure after any
                            // collision truncation.  Coverage is evaluated at the
                            // same step-start energy used by the runtime hazard;
                            // generation-blocked path is kept separate and never
                            // folded into an apparent uncovered rate segment.
                            if (exposure_cell != std::numeric_limits<std::uint32_t>::max()) {
                                if (secondary_generation_eligible) {
                                    if (exposure_rate_covered) {
                                    } else {
                                    }
                                    if (!exposure_h_covered) {
                                    }
                                    if (!exposure_o_covered) {
                                    }
                                    if (exposure_h_covered) {
                                    }
                                    if (exposure_o_covered) {
                                    }
                                    if (exposure_rate_covered) {
                                    }
                                } else {
                                    // Counterfactual hazard for this blocked segment.
                                    // Keep H/O separate when only one target is covered;
                                    // total is defined only for a complete H/O pair, exactly
                                    // as in the runtime target selector.
                                    if (exposure_h_covered) {
                                    }
                                    if (exposure_o_covered) {
                                    }
                                    if (exposure_rate_covered) {
                                    }
                                }
                            }

                            // Midpoint loss
                            const auto mid_e_u = sycl::fmax(generic_recoil?recoil_stopping.energies[0]:0.01F, (sec_e - 0.5F * sec_sp * sec_step_mm) * frag_inv_a);
                            const auto mid_flt = (mid_e_u - minimum_table_energy) * inverse_table_step;
                            auto mid_idx = static_cast<int>(sycl::floor(mid_flt));
                            mid_idx = sycl::max(0, sycl::min(mid_idx, static_cast<int>(table_size) - 2));
                            const auto mid_fr = sycl::clamp(mid_flt - static_cast<float>(mid_idx), 0.0F, 1.0F);
                            // Unified EM overwrites dE below: keep only the
                            // scheme2 break check, skip the dead midpoint
                            // stopping evaluation.
                            auto mid_sp = 0.0F;
                            // Query the same material bank at the predicted midpoint.
                            // Already linear at local density: bypass the
                            // water x density scaling entirely.
                            bool mid_scheme2_hit = false;
                            {
                                const float scheme2_mid = scheme2_linear_sp(mid_e_u);
                                if ((generic_recoil||(use_schneider_ion_sp && sec_in_ct)) && !(scheme2_mid > 0)) break;
                                if (scheme2_mid > 0.0F) {
                                    mid_sp = scheme2_mid;
                                    mid_scheme2_hit = true;
                                }
                            }
                            if (!unified_secondary) {
                                mid_sp = (ion_sp_table[mid_idx] +
                                          mid_fr * (ion_sp_table[mid_idx + 1] -
                                                    ion_sp_table[mid_idx]));
                                // The midpoint value drives dE and must use the same
                                // Schneider density/material scaling as sec_sp at the
                                // step start. Previously this remained a density-1
                                // water value, over-stopping secondaries by ~1/rho
                                // (about 25x in the RT06423 air section).
                                float mid_material_factor = 1.0F;
                                if (sec_use_mass_sp_factor) {
                                    mid_material_factor = ct_lookup_mass_sp_factor(
                                        ct_mass_sp_factor_lut_device,
                                        use_ct_density_mass_spr ? ct_density_spr_n_rho
                                                                : ct_n_mass_factors,
                                        table_size, use_ct_density_mass_spr,
                                        ct_mass_spr_log_rho_min, ct_mass_spr_inv_dlog,
                                        static_cast<std::uint32_t>(sec_ct_material),
                                        sec_local_density_g_per_cm3,
                                        static_cast<std::size_t>(mid_idx), mid_fr,
                                        [](float x) { return sycl::log(x); });
                                }
                            if (!mid_scheme2_hit) {
                                mid_sp = secondary_material_stopping_power(
                                    mid_sp, enable_ct_grid && sec_in_ct,
                                    sec_local_density_g_per_cm3,
                                    sec_use_mass_sp_factor, mid_material_factor);
                            }
                            }

                            float dE = sycl::fmin(mid_sp * sec_step_mm, sec_e);
                            if (!unified_secondary && enable_secondary_energy_straggling &&
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
                            if(unified_secondary){
                                auto draw=unified_em_loss(unified_secondary_state,unified_secondary_clock,sec_e,sec_step_mm,unified_secondary_rate,unified_secondary_distance,enable_secondary_energy_straggling,unified_secondary_pre,unified_secondary_uniform);
                                if(!draw.valid){
                                    record_unified_em_failure(unified_failure_count,unified_failure_records,4,unified_secondary_state.section,frag.z,frag.a,sec_e,sec_local_density_g_per_cm3,sec_step_mm);
                                    unified_secondary_count(0);unified_secondary_count(7);break;}
                                dE=draw.loss;unified_secondary_count(2);unified_secondary_count(3,draw.proposed);unified_secondary_count(4,draw.accepted);
                                unified_secondary_count(5,static_cast<std::uint64_t>(draw.continuous*1e6f));unified_secondary_count(6,static_cast<std::uint64_t>(draw.delta*1e6f));
                            }
                            const auto post_em_e = sycl::fmax(0.0F, sec_e - dE);
                            if (he4_hazard_audit_device && frag.z == 2 && frag.a == 4 &&
                                frag.generation < cinel02_max_secondary_inelastic_generations &&
                                schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || use_unified_water)) {
                                const auto p = secondary_projectile_lut_index_device(
                                    schneider_ct_device_ctx.sec_proj_keys,
                                    schneider_ct_device_ctx.sec_num_projectiles, 2, 4);
                                const auto section = use_unified_water ? 255U : sec_ct_material;
                                const double start = schneider_hazard_total_rate;
                                const double middle = sec_local_density_g_per_cm3 *
                                    schneider_ct_device_ctx.secondary_rates(p, section,
                                        (sec_e - 0.5F*dE)*frag_inv_a).total;
                                const double end = sec_local_density_g_per_cm3 *
                                    schneider_ct_device_ctx.secondary_rates(p, section,
                                        post_em_e*frag_inv_a).total;
                                he4_audit[0] += start*sec_step_mm;
                                he4_audit[1] += (start+4*middle+end)*sec_step_mm/6;
                                he4_audit[2] += (start*sec_e+4*middle*(sec_e-0.5F*dE)+end*post_em_e)*sec_step_mm/6;
                                he4_audit[3] += secondary_inelastic ? 1 : 0;
                                he4_audit[4] += secondary_inelastic ? post_em_e : 0;
                                he4_audit[5] += sec_step_mm;
                            }
                            auto post_em_x = sec_x + collision_input_dx * sec_step_mm;
                            auto post_em_y = sec_y + collision_input_dy * sec_step_mm;
                            auto post_em_z = sec_z + collision_input_dz * sec_step_mm;
                            if (ct_secondary_exact_faces && sec_face_clamp.hit_face && !secondary_inelastic && !secondary_elastic) {
                                const auto endpoint=ct_finish_exact_face_step(sec_face_clamp,sec_step_mm,
                                    {sec_x,sec_y,sec_z},{collision_input_dx,collision_input_dy,collision_input_dz},
                                    {ct_origin_x,ct_origin_y,ct_origin_z},{ct_spacing_x,ct_spacing_y,ct_spacing_z});
                                post_em_x=endpoint[0];post_em_y=endpoint[1];post_em_z=endpoint[2];
                            }

                            // Independent continuous optical-depth audit.  The
                            // runtime sampler uses the step-start rates above;
                            // this diagnostic evaluates the same H/O rates at
                            // the start, midpoint and post-EM endpoint and
                            // integrates them with Simpson's rule over the
                            // actual (possibly collision-truncated) segment.
                            // It never feeds back into target selection or
                            // collision sampling.
                            const auto continuous_mid_e_u = sycl::fmax(
                                0.0F, (sec_e - 0.5F * dE) * frag_inv_a);
                            const auto continuous_end_e_u = sycl::fmax(
                                0.0F, post_em_e * frag_inv_a);
                            Cinel02DeviceRateLookup continuous_h_mid{};
                            Cinel02DeviceRateLookup continuous_o_mid{};
                            Cinel02DeviceRateLookup continuous_h_end{};
                            Cinel02DeviceRateLookup continuous_o_end{};
                            const bool continuous_rate_covered =
                                exposure_h_lookup.covered && exposure_o_lookup.covered &&
                                continuous_h_mid.covered && continuous_o_mid.covered &&
                                continuous_h_end.covered && continuous_o_end.covered;
                            if (exposure_cell != std::numeric_limits<std::uint32_t>::max()) {
                                if (continuous_rate_covered) {
                                    const auto lambda_start =
                                        exposure_h_lookup.value_per_mm +
                                        exposure_o_lookup.value_per_mm;
                                    const auto lambda_mid =
                                        continuous_h_mid.value_per_mm +
                                        continuous_o_mid.value_per_mm;
                                    const auto lambda_end =
                                        continuous_h_end.value_per_mm +
                                        continuous_o_end.value_per_mm;
                                    const auto tau_continuous =
                                        cinel02_simpson_hazard(
                                            lambda_start, lambda_mid, lambda_end,
                                            sec_step_mm);
                                } else {
                                }
                            }
                            const auto collision_bin = sycl::max(
                                0, sycl::min(
                                       static_cast<int>(post_em_z *
                                           inverse_depth_bin_width_mm),
                                       static_cast<int>(number_of_bins) - 1));
                            // Exact-face transport keeps continuous loss inside
                            // the SOURCE voxel. Do not mix source x/y with the
                            // endpoint z, or credit this loss at a nuclear vertex.
                            // Commit once here before a replay can update sec_x/y/z
                            // or break; the legacy dE commits below are suppressed.
                            const bool source_voxel_continuous_loss =
                                ct_secondary_exact_faces && enable_ct_grid && sec_in_ct && enable_voxel_scoring;
                            int continuous_step_voxel = -1;
                            if (source_voxel_continuous_loss) {
                                const int sx=static_cast<int>(sycl::floor((sec_x-voxel_min_x_mm)*inverse_voxel_size_x_mm));
                                const int sy=static_cast<int>(sycl::floor((sec_y-voxel_min_y_mm)*inverse_voxel_size_y_mm));
                                if (sx>=0 && sy>=0 && bin_z>=0 && sx<static_cast<int>(voxel_bins_x) &&
                                    sy<static_cast<int>(voxel_bins_y) && bin_z<static_cast<int>(number_of_bins)) {
                                    continuous_step_voxel=(bin_z*static_cast<int>(voxel_bins_y)+sy)*static_cast<int>(voxel_bins_x)+sx;
                                    sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,sycl::access::address_space::global_space>
                                        a(voxel_dose_device[continuous_step_voxel]);
                                    a.fetch_add(static_cast<DoseAtomicT>(dE));
                                    if (enable_charged_origin_voxel_scoring) {
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,
                                            sycl::memory_scope::device,sycl::access::address_space::global_space>
                                            origin(charged_origin_voxel_dose_device[charged_origin_voxel_offset+continuous_step_voxel]);
                                        origin.fetch_add(static_cast<DoseAtomicT>(dE));
                                        score_be_isotope_origin_voxel_device(be_isotope_origin_voxel_dose_device,
                                            be_isotope_category,number_of_voxels,continuous_step_voxel,static_cast<DoseAtomicT>(dE));
                                        score_he_isotope_origin_voxel_device(he_isotope_origin_voxel_dose_device,
                                            he_isotope_category,number_of_voxels,continuous_step_voxel,static_cast<DoseAtomicT>(dE));
                                    }
                                }
                            }
                            if (secondary_inelastic) {
                                sec_e = post_em_e;
                                // v3: sample the target HERE at the post-EM
                                // collision energy E_c (sec_e just updated),
                                // so the replay-status record below and the
                                // package query share one energy with the mask.
                                if (schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || use_unified_water) &&
                                    sec_e > energy_cutoff_MeV &&
                                    frag.generation <
                                        cinel02_max_secondary_inelastic_generations) {
                                    const float sec_e_c = sec_e * frag_inv_a;
                                    const int proj_idx_c =
                                        secondary_projectile_lut_index_device(
                                            schneider_ct_device_ctx.sec_proj_keys,
                                            schneider_ct_device_ctx.sec_num_projectiles,
                                            frag.z, frag.a);
                                    const std::size_t section_c = use_unified_water ? 255U : static_cast<std::size_t>(
                                        sycl::min(static_cast<std::uint32_t>(sec_ct_material), 24U));
                                    const auto sec_masked_c = schneider_ct_device_ctx.secondary_rates(
                                        proj_idx_c, section_c, sec_e_c);
                                    // Same tag-14 draw as the legacy site (same
                                    // step counter: sec_steps increments after
                                    // the lookup); only the energy moves E_h ->
                                    // E_c. Empty draw (slowing left every
                                    // channel domain): counted, never queried.
                                    const float u_target_c = rng::uniform01(
                                        2026, frag.rng_stream, sec_steps, 14);
                                    const int sampled_z_c =
                                        sample_masked_secondary_target_device(
                                            sec_masked_c.partials, u_target_c);
                                    if (sampled_z_c <= 0) {
                                        // Post-EM null collision (declared research
                                        // approximation): no UnsupportedTargets,
                                        // no StoppedBeforeReplay. The track keeps
                                        // its post-EM energy/position, continues
                                        // transport, loses no energy, deposits
                                        // nothing locally, issues no lookup.
                                        schneider_diag_increment_device(
                                            schneider_diag_device,
                                            SchneiderDiagSlot::SecondaryPostEmNullCollisions);
                                        schneider_float_add_device(
                                            schneider_float_device,
                                            SchneiderFloatSlot::PostEmNullEnergy,
                                            sycl::fmax(0.0F, sec_e));
                                        secondary_inelastic = false;
                                        skip_schneider_lookup = true;
                                    } else {
                                        secondary_target_z =
                                            static_cast<std::int16_t>(sampled_z_c);
                                    }
                                }
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                                // Record the hazard before attempting package
                                // replay so isotope/target/generation misses
                                // remain distinguishable from valid events.
                                // v3 sampler-empty skips the record exactly
                                // like the v1 hazard-site empty draw (which
                                // never reaches this block).
                                if (!skip_schneider_lookup) {
                                }  // !skip_schneider_lookup (v3 sampler-empty records nothing)
                            }

                            if (bin_z != pending_sec_bin) {
                        if (pending_sec_depth_MeV > 0.0F && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                    sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                     sycl::memory_scope::device,
                                                     sycl::access::address_space::global_space>
                                        atomic_dose(dose_device[pending_sec_bin]);
                                    atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
                                    pending_sec_depth_MeV = 0.0F;
                                }
                                pending_sec_bin = bin_z;
                            }
                            pending_sec_depth_MeV += dE;
                            continuous_species_tally.add_all(dE);

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
                                        score_he_isotope_origin_voxel_device(
                                            he_isotope_origin_voxel_dose_device,
                                            he_isotope_category, number_of_voxels,
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                    }
                                    pending_sec_voxel_MeV = 0.0F;
                                    pending_sec_voxel = cur_voxel;
                            if (secondary_inelastic && !(sec_e > energy_cutoff_MeV) &&
                                schneider_ct_device_ctx.uses_cinel03() && (sec_in_ct || use_unified_water) &&
                                frag.generation < cinel02_max_secondary_inelastic_generations) {
                                // Sampled Schneider collision whose post-EM
                                // energy is already below cutoff: continuous
                                // stopping owns the energy, no replay attempted.
                                schneider_diag_increment_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
                            }
                            if (secondary_inelastic && sec_e > energy_cutoff_MeV) {
                                // v3 sampling already happened at the post-EM
                                // update above; skip the query only when the
                                // sampler found no in-domain channel.
                                if (schneider_ct_device_ctx.uses_cinel03() && !skip_schneider_lookup) {
                                    // Exact (Z/A, target_Z) channel with the
                                    // secondary's own sampled target: never the
                                    // primary target, never water/O fallback.
                                    // Tag 15 drives the bracket choice, tag 16
                                    // the intra-node event pick.
                                    const float sec_u_bracket = rng::uniform01(2026, frag.rng_stream, sec_steps, 15);
                                    const float sec_u_event = rng::uniform01(2026, frag.rng_stream, sec_steps, 16);
                                    const auto sec_lookup = cinel03_lookup_event_device(
                                        schneider_ct_device_ctx.sec_energy_nodes,
                                        schneider_ct_device_ctx.sec_node_count,
                                        schneider_ct_device_ctx.sec_event_offsets,
                                        schneider_ct_device_ctx.sec_event_indices,
                                        schneider_ct_device_ctx.sec_total_events,
                                        frag.z, frag.a, secondary_target_z,
                                        sec_e * frag_inv_a,
                                        sec_u_bracket, sec_u_event);
                                    schneider_record_lookup_device(
                                        schneider_diag_device, schneider_float_device,
                                        false, sec_lookup, sec_e);

                                    if (sec_lookup.status == Cinel03LookupStatus::Hit) {
                                        const std::uint32_t event_idx = sec_lookup.event_index;
                                        secondary_replay_succeeded = true;
                                        cinel02_species_energy_add_device(
                                            cinel02_species_energy_device, ledger_species_idx, 8U, sec_e);
                                        // Use the collision vertex, not the source
                                        // voxel or legacy pre-x/post-z mixed index.
                                        // These are subsets, never extra energy sinks.
                                        if (enable_voxel_scoring && replay_vertex_in_scoring_box(
                                                {post_em_x,post_em_y,post_em_z},
                                                {voxel_min_x_mm,voxel_min_y_mm,0.0F},
                                                {voxel_max_x_mm,voxel_max_y_mm,
                                                 static_cast<float>(number_of_bins)*depth_bin_width_mm})) {
                                            cinel02_species_energy_add_device(
                                                cinel02_species_energy_device, ledger_species_idx,
                                                Cinel02SpeciesLedgerSchema::cinel03_replay_input_fov, sec_e);
                                            cinel02_species_energy_add_device(
                                                cinel02_species_energy_device, ledger_species_idx,
                                                Cinel02SpeciesLedgerSchema::cinel03_replay_step_dE_fov, dE);
                                        }
                                        const auto& event = schneider_ct_device_ctx.sec_interactions[event_idx];
                                        const float event_phi = 6.2831853071795864769F * rng::uniform01(
                                            2026, frag.rng_stream, sec_steps, cinel03_event_azimuth_dimension);
                                        const float event_cos = sycl::cos(event_phi);
                                        const float event_sin = sycl::sin(event_phi);
                                        const float local_deposit = sycl::fmax(0.0F, event.process_local_deposit_MeV);
                                        pending_sec_depth_MeV += local_deposit;
                                        if (enable_voxel_scoring && cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += local_deposit;
                                        }
                                        if (deposited_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(deposited_device[frag.parent_history]);
                                            // Ledger keeps local + step dE here; the
                                            // common path is skipped by the break
                                            // below, so this is the single dE credit.
                                            atomic_dep.fetch_add(local_deposit + dE);
                                            schneider_energy_add_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                local_deposit + dE);
                                            if (source_voxel_continuous_loss) {
                                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                                    cur_voxel>=0,local_deposit);
                                                grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,
                                                    continuous_step_voxel>=0,dE);
                                            } else grid_deposit_split_device(
                                                grid_deposited_in_device,grid_deposited_out_device,
                                                enable_voxel_scoring && cur_voxel>=0,local_deposit+dE);
                                        }
                                        // The break below skips the common-path
                                        // voxel dE commit: commit it here so the
                                        // collision step reaches history ledger,
                                        // depth, voxel and species scorers.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            enable_voxel_scoring && !source_voxel_continuous_loss, cur_voxel, dE);

                                        float sec_charged_accounted_MeV = 0.0F;
                                        float sec_neutral_accounted_MeV = 0.0F;
                                        float sec_unsupported_accounted_MeV = 0.0F;

                                        const std::uint32_t prod_offset = event.product_offset;
                                        const std::uint32_t prod_count = event.direct_product_count;
                                        for (std::uint32_t ip = 0; ip < prod_count; ++ip) {
                                            if (prod_offset + ip >= schneider_ct_device_ctx.sec_total_products) break;
                                            const auto& product = schneider_ct_device_ctx.sec_products[prod_offset + ip];

                                            if (product.role == 2) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_unsupported_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::UnsupportedProductEnergy, ke);
                                                continue;
                                            }
                                            if (product.z <= 0 || product.a <= 0) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                sec_neutral_accounted_MeV += ke;
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::NeutralProductKinetic, ke);
                                                continue;
                                            }
                                            if (product.role != 0) {
                                                continue;
                                            }

                                            // TopasCompatKill for Be6 (Z=4, A=6):
                                            // independent counter + energy, never queued.
                                            if (product.z == 4 && product.a == 6) {
                                                const float ke = sycl::fmax(0.0F, product.kinetic_energy_MeV);
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::SecondaryBe6Kills);
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::Be6TopasCompatKills);
                                                schneider_float_add_device(
                                                    schneider_float_device,
                                                    SchneiderFloatSlot::Be6KillEnergy, ke);
                                                continue;
                                            }
                                            schneider_diag_increment_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryChargedBorn);

                                            if (product.kinetic_energy_MeV > energy_cutoff_MeV &&
                                                (terminal_generation_em ||
                                                 frag.generation + 1U < cinel02_max_secondary_inelastic_generations)) {
                                                // Reaching the reaction cap need not imply
                                                // local deposition: the child's existing
                                                // generation guard suppresses further nuclear
                                                // hazards but leaves EM slowing/MCS active.
                                                const auto local_direction = rotate_cinel03_event_azimuth(
                                                    product.local_direction_x, product.local_direction_y,
                                                    product.local_direction_z, event_cos, event_sin);
                                                const auto child_direction = rotate_local_direction(
                                                    local_direction.x,
                                                    local_direction.y,
                                                    local_direction.z,
                                                    Direction3F{collision_input_dx, collision_input_dy, collision_input_dz});

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
                                                        child.pos_x_mm = post_em_x;
                                                        child.pos_y_mm = post_em_y;
                                                        child.pos_z_mm = post_em_z;
                                                        child.dir_x = child_direction.x;
                                                        child.dir_y = child_direction.y;
                                                        child.dir_z = child_direction.z;
                                                        child.weight = 1.0F;
                                                        child.generation = static_cast<std::uint16_t>(frag.generation + 1U);
                                                        child.parent_history = frag.parent_history;
                                                        child.rng_stream = rng::event_product_stream(
                                                            frag.rng_stream, sec_steps,
                                                            rng::branch_role_cascade_charged, ip);
                                                        secondary_queue_device[output] = child;
                                                        sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::SecondaryChargedQueued);
                                                        if (cinel02_species_energy_device != nullptr) {
                                                            cinel02_record_queued_secondary_birth_device(
                                                                cinel02_species_energy_device, child.z, child.a,
                                                                child.energy_MeV);
                                                            const auto child_species_idx = carbon::get_charged_species_idx(child.z, child.a);
                                                            if (child_species_idx < 18) {
                                                                cinel02_species_energy_add_device(
                                                                    cinel02_species_energy_device, child_species_idx, 10U, child.energy_MeV);
                                                            }
                                                        }
                                                    } else {
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::SecondaryQueueOverflows);
                                                        schneider_diag_increment_device(
                                                            schneider_diag_device,
                                                            SchneiderDiagSlot::QueueOverflows);
                                                        if (secondary_overflow_count_device != nullptr) {
                                                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ov(*secondary_overflow_count_device);
                                                            atomic_ov.fetch_add(1U);
                                                        }
                                                        if (secondary_overflow_energy_device != nullptr) {
                                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                atomic_ove(*secondary_overflow_energy_device);
                                                            atomic_ove.fetch_add(product.kinetic_energy_MeV);
                                                        }
                                                    }
                                                }
                                            } else {
                                                pending_sec_depth_MeV += product.kinetic_energy_MeV;
                                                if (enable_voxel_scoring && cur_voxel >= 0) {
                                                    pending_sec_voxel_MeV += product.kinetic_energy_MeV;
                                                }
                                                if (deposited_device != nullptr) {
                                                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                                     sycl::memory_scope::device,
                                                                     sycl::access::address_space::global_space>
                                                        atomic_dep(deposited_device[frag.parent_history]);
                                                    atomic_dep.fetch_add(product.kinetic_energy_MeV);
                                                    schneider_energy_add_device(
                                                        schneider_diag_device,
                                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                        product.kinetic_energy_MeV);
                                                    grid_deposit_split_device(
                                                        grid_deposited_in_device,
                                                        grid_deposited_out_device,
                                                        enable_voxel_scoring && cur_voxel >= 0,
                                                        product.kinetic_energy_MeV);
                                                }
                                                sec_charged_accounted_MeV += product.kinetic_energy_MeV;
                                                schneider_diag_increment_device(
                                                    schneider_diag_device,
                                                    SchneiderDiagSlot::SecondaryChargedCutoffKills);
                                            }
                                        }

                                        schneider_float_add_device(
                                            schneider_float_device, SchneiderFloatSlot::ReactionQResidual,
                                            sycl::fmax(0.0F, sec_e - local_deposit -
                                                                  sec_charged_accounted_MeV -
                                                                  sec_neutral_accounted_MeV -
                                                                  sec_unsupported_accounted_MeV));
                                        // NO-DOUBLE-COUNT RULE (same as primary
                                        // vertex): legacy untracked sink already
                                        // contains neutral + unsupported + Q;
                                        // split slots are informational only.
                                        const float sec_untracked_MeV = sycl::fmax(0.0F, sec_e - local_deposit - sec_charged_accounted_MeV);
                                        if (sec_untracked_MeV > 0.0F && untracked_nuclear_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(untracked_nuclear_device[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_untracked_MeV);
                                        }
                                        sec_e = 0.0F;
                                        break;
                                    } else {
                                        // CINEL03 miss: fail closed + per-miss log.
                                        schneider_log_miss_device(
                                            schneider_miss_device, schneider_miss_count_device,
                                            kSchneiderMissLogCap, false,
                                            frag.z, frag.a, secondary_target_z,
                                            schneider_hazard_section,
                                            frag.generation > 255 ? 255
                                                                  : static_cast<std::uint8_t>(frag.generation),
                                            sec_lookup, sec_e * frag_inv_a, dE,
                                            schneider_hazard_total_rate,
                                            schneider_hazard_step_mm, sec_e,
                                            frag.energy_MeV);
                                        if (untracked_nuclear_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_untracked(untracked_nuclear_device[frag.parent_history]);
                                            atomic_untracked.fetch_add(sec_e);
                                        }
                                        // cur_voxel is out of scope on the miss
                                        // path: recompute the collision voxel
                                        // with the same formula for both the
                                        // split and the commit below.
                                        int miss_voxel = -1;
                                        {
                                            const auto miss_sbx = static_cast<int>(
                                                (sec_x - voxel_min_x_mm) *
                                                inverse_voxel_size_x_mm);
                                            const auto miss_sby = static_cast<int>(
                                                (sec_y - voxel_min_y_mm) *
                                                inverse_voxel_size_y_mm);
                                            if (miss_sbx >= 0 &&
                                                miss_sbx < static_cast<int>(voxel_bins_x) &&
                                                miss_sby >= 0 &&
                                                miss_sby < static_cast<int>(voxel_bins_y)) {
                                                miss_voxel =
                                                    (collision_bin *
                                                     static_cast<int>(voxel_bins_y) +
                                                     miss_sby) *
                                                        static_cast<int>(voxel_bins_x) +
                                                    miss_sbx;
                                            }
                                        }
                                        if (deposited_device != nullptr) {
                                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dep(deposited_device[frag.parent_history]);
                                            atomic_dep.fetch_add(dE);
                                            schneider_energy_add_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                            grid_deposit_split_device(
                                                grid_deposited_in_device,
                                                grid_deposited_out_device,
                                                enable_voxel_scoring && (source_voxel_continuous_loss ? continuous_step_voxel>=0 : miss_voxel>=0),
                                                dE);
                                        }
                                        // Same bypass as the replay-hit path:
                                        // the break below skips the common-path
                                        // voxel dE commit.
                                        carbon::secondary_step_voxel_commit(
                                            pending_sec_voxel_MeV,
                                            enable_voxel_scoring && !source_voxel_continuous_loss, miss_voxel, dE);
                                        sec_e = 0.0F;
                                        break;
                                    }
                                } else {
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
                                    if (product_end <= cinel02_product_count &&
                                        (event.parent_status == 0 ||
                                         event.parent_status == 2)) {
                                        secondary_replay_succeeded = true;
                                        if (frag.z >= 1 && frag.z <= 6) {
                                            const auto incident_keV =
                                                static_cast<std::uint64_t>(sycl::fmax(
                                                    0.0F, sec_e) * 1000.0F + 0.5F);
                                        }
                                        if (frag.z == 4) {
                                            const auto be_incident_slot =
                                                cinel02_be_incident_diag_slot_device(
                                                    secondary_target_z, frag.generation,
                                                    sec_e * frag_inv_a);
                                        }
                                        const auto parent_outcome_slot =
                                            cinel02_parent_outcome_diag_slot_device(
                                                frag.z, event.parent_status,
                                                static_cast<std::uint32_t>(frag.generation));
                                        if (parent_outcome_slot !=
                                            std::numeric_limits<std::uint32_t>::max()) {
                                        }
                                        const auto local_deposit = sycl::fmax(
                                            0.0F, event.process_local_deposit_MeV);
                                        const auto parent_after = event.parent_status == 0
                                            ? sycl::fmax(0.0F, event.parent_energy_MeV)
                                            : 0.0F;
                                        const auto reaction_handoff_delta = sec_e -
                                            parent_after - local_deposit;
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
                                        if (local_deposit > 0.0F) {
                                            sycl::atomic_ref<
                                                DepthAtomicT, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>
                                                atomic_local_depth(dose_device[collision_bin]);
                                            atomic_local_depth.fetch_add(
                                                static_cast<DepthAtomicT>(local_deposit));
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
                                            schneider_energy_add_device(
                                                schneider_diag_device,
                                                SchneiderDiagSlot::SecondaryDepositedMicroMeV,
                                                local_deposit);
                                            grid_deposit_split_device(
                                                grid_deposited_in_device,
                                                grid_deposited_out_device,
                                                enable_voxel_scoring &&
                                                    pending_sec_voxel >= 0,
                                                local_deposit);
                                        }
                                        float untracked_MeV = 0.0F;
                                        for (std::uint32_t ip = 0;
                                             ip < event.product_count; ++ip) {
                                            const auto product = cinel02_products_device[
                                                event.product_offset + ip];
                                            if (product.role == 0 && product.z >= 1 && product.z <= 6) {
                                            }
                                            if (product.role == 0 && product.z == 4) {
                                                const auto birth_slot =
                                                    cinel02_be_isotope_birth_diag_slot_device(
                                                        frag.z, secondary_target_z,
                                                        frag.generation + 1U, product.a);
                                                if (birth_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                }
                                                const auto be_channel_slot =
                                                    cinel02_be_channel_diag_slot_device(
                                                        frag.z, secondary_target_z,
                                                        frag.generation, sec_e * frag_inv_a);
                                                if (be_channel_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                }
                                            }
                                            if (product.role == 0) {
                                                const auto transition_slot =
                                                    cinel02_transition_diag_slot_device(
                                                        frag.z, product.z,
                                                        static_cast<std::uint32_t>(frag.generation));
                                                if (transition_slot !=
                                                    std::numeric_limits<std::uint32_t>::max()) {
                                                    const auto kinetic_keV =
                                                        static_cast<std::uint64_t>(sycl::fmax(
                                                            0.0F, product.kinetic_energy_MeV) *
                                                            1000.0F + 0.5F);
                                                }
                                            }
                                            if (product.role == 0 && product.z > 0 &&
                                                product.a > 0 &&
                                                cinel02_should_topas_compat_kill(
                                                    cinel02_topas_compatibility_mode,
                                                    product.z, product.a)) {
                                                // Match TOPAS/Geant4's unsupported prompt-ion
                                                // fallback: generated, then killed before queue,
                                                // with no daughter and no local deposit.
                                                continue;
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
                                                }
                                                secondary_queue_device[output] = child;
                                                cinel02_record_queued_secondary_birth_device(
                                                    cinel02_species_energy_device, product.z, product.a,
                                                    sycl::fmax(0.0F, product.kinetic_energy_MeV));
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
                                    }
                                }
                                else {
                                }
                                }
                            } else if (secondary_inelastic) {
                                // The post-EM collision energy is already at
                                // or below the transport cutoff, so no package
                                // event is eligible.  Keep the candidate in
                                // the status partition as post_em_below_cutoff.
                            }
                                    pending_sec_voxel = cur_voxel;
                                }
                                if (cur_voxel >= 0) {
                                    if (!source_voxel_continuous_loss) pending_sec_voxel_MeV += dE;
                                    if (in_fov_dose_device != nullptr && pending_sec_bin >= 0 && pending_sec_bin < static_cast<int>(number_of_bins)) {
                                        sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            atomic_in_fov(in_fov_dose_device[pending_sec_bin]);
                                        atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(dE));
                                        continuous_species_tally.add_fov(dE);
                                    }
                                }
                            }

                            if (!secondary_inelastic) sec_e = post_em_e;
                            // cur_voxel is out of scope on the common path:
                            // recompute the step voxel with the same formula.
                            int com_voxel = -1;
                            {
                                const auto com_sbx = static_cast<int>(
                                    (sec_x - voxel_min_x_mm) *
                                    inverse_voxel_size_x_mm);
                                const auto com_sby = static_cast<int>(
                                    (sec_y - voxel_min_y_mm) *
                                    inverse_voxel_size_y_mm);
                                if (com_sbx >= 0 &&
                                    com_sbx < static_cast<int>(voxel_bins_x) &&
                                    com_sby >= 0 &&
                                    com_sby < static_cast<int>(voxel_bins_y)) {
                                    com_voxel =
                                        (collision_bin *
                                         static_cast<int>(voxel_bins_y) +
                                         com_sby) *
                                            static_cast<int>(voxel_bins_x) +
                                        com_sbx;
                                }
                            }
                            if (deposited_device != nullptr) {
                                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    atomic_dep(deposited_device[frag.parent_history]);
                                atomic_dep.fetch_add(dE);
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryDepositedMicroMeV, dE);
                                grid_deposit_split_device(
                                    grid_deposited_in_device,
                                    grid_deposited_out_device, source_voxel_continuous_loss ? continuous_step_voxel>=0 : com_voxel>=0,
                                    dE);
                            }
                            if (!secondary_inelastic) {
                                sec_x = post_em_x;
                                sec_y = post_em_y;
                                sec_z = post_em_z;
                            }

                            if (cinel02_should_apply_secondary_mcs(
                                    secondary_inelastic, secondary_replay_succeeded,
                                    enable_multiple_scattering && !ct_secondary_mcs_off, sec_e,
                                    energy_cutoff_MeV)) {
                                constexpr float two_pi = 6.2831853071795864769F;
                                float theta_scat = 0.0F;
                                float phi_scat = 0.0F;
                                // Secondary MCS material selection shares the
                                // primary helper: Schneider CT voxels use the
                                // section X0, everywhere else falls back to
                                // water. Density is the local CT density
                                // (== water outside CT), never double-counted.
                                const auto sec_radiation_length_g_per_cm2 =
                                    static_cast<float>(
                                        select_transport_radiation_length_g_per_cm2(
                                            ct_material_ids_are_schneider_sections,
                                            sec_in_ct,
                                            static_cast<unsigned>(sec_ct_material),
                                            enable_ct_material_mcs,
                                            active_water_radiation_length));
                                {
                                    const auto theta_rms = highland_projected_rms_angle_device(
                                        sec_e, static_cast<int>(frag.z),
                                        static_cast<int>(frag.a), sec_step_mm,
                                        sec_local_density_g_per_cm3,
                                        sec_radiation_length_g_per_cm2) * multiple_scattering_scale;
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

                            // A hazard sampled at step start can leave a supported rate
                            // interval after EM loss. Treat the zero-rate endpoint as a
                            // null collision; retain post-EM energy, position and MCS.
                            if(secondary_elastic && sec_e>energy_cutoff_MeV &&
                               all_elastic.rate(all_elastic.projectile(frag.z,frag.a),use_unified_water?25:sec_ct_material,sec_e*frag_inv_a)==0) {
                                secondary_elastic=false;
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[9]).fetch_add(1);
                            }
                            if(secondary_elastic && sec_e>energy_cutoff_MeV) {
                                const int p=all_elastic.projectile(frag.z,frag.a);
                                const auto draw=all_elastic.draw(p,use_unified_water?25:sec_ct_material,
                                    sec_e,sec_dx,sec_dy,sec_dz,
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,71),rng::uniform01(2026,frag.rng_stream,sec_steps,72),
                                    rng::uniform01(2026,frag.rng_stream,sec_steps,73),rng::uniform01(2026,frag.rng_stream,sec_steps,74));
                                if(!draw.valid){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[2]).fetch_add(1);
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[8]).fetch_add(1);break;
                                }
                                sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[1]).fetch_add(1);
                                const auto scat=draw.outcome;sec_e=scat.projectile_ke_MeV;
                                sec_dx=scat.proj_dir_x;sec_dy=scat.proj_dir_y;sec_dz=scat.proj_dir_z;
                                const float recoil=scat.recoil_ke_MeV;
                                const int child_species=carbon::get_charged_species_idx(draw.target_z,draw.target_a);
                                if(recoil>energy_cutoff_MeV && (child_species>=0||recoil_stopping.projectile(draw.target_z,draw.target_a)>=0)) {
                                    schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryChargedBorn);
                                    const auto output=sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_count_device).fetch_add(1);
                                    if(output<max_secondaries){
                                        SecondaryParticle child{};child.z=draw.target_z;child.a=draw.target_a;child.energy_MeV=recoil;
                                        child.pos_x_mm=sec_x;child.pos_y_mm=sec_y;child.pos_z_mm=sec_z;
                                        child.dir_x=scat.recoil_dir_x;child.dir_y=scat.recoil_dir_y;child.dir_z=scat.recoil_dir_z;
                                        child.weight=1;child.generation=child_species>=0?frag.generation:cinel02_max_secondary_inelastic_generations;child.parent_history=frag.parent_history;
                                        child.rng_stream=rng::event_product_stream(frag.rng_stream,sec_steps,rng::branch_role_cascade_charged,0xE1U);
                                        secondary_queue_device[output]=child;
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[1]).fetch_add(recoil);
                                        schneider_diag_increment_device(schneider_diag_device,SchneiderDiagSlot::SecondaryChargedQueued);
                                        cinel02_record_queued_secondary_birth_device(cinel02_species_energy_device,child.z,child.a,recoil);
                                        cinel02_species_energy_add_device(cinel02_species_energy_device,child_species,10U,recoil);
                                    }else{
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_count_device).fetch_add(1);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(*secondary_overflow_energy_device).fetch_add(recoil);
                                        sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[2]).fetch_add(recoil);
                                        sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[11]).fetch_add(1);
                                    }
                                    cinel02_species_energy_add_device(cinel02_species_energy_device,ledger_species_idx,8U,recoil);
                                }else if(recoil>energy_cutoff_MeV){
                                    sycl::atomic_ref<std::uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_audit[3]).fetch_add(1);
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(untracked_nuclear_device[frag.parent_history]).fetch_add(recoil);
                                    cinel02_species_energy_add_device(cinel02_species_energy_device,ledger_species_idx,8U,recoil);
                                }else if(recoil>0){
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(elastic_energy[0]).fetch_add(recoil);
                                    const int vx=static_cast<int>(sycl::floor((sec_x-voxel_min_x_mm)/voxel_size_x_mm));
                                    const int vy=static_cast<int>(sycl::floor((sec_y-voxel_min_y_mm)/voxel_size_y_mm));
                                    const int vz=static_cast<int>(sycl::floor(sec_z*inverse_depth_bin_width_mm));
                                    const bool inside=enable_voxel_scoring&&vx>=0&&vy>=0&&vz>=0&&vx<static_cast<int>(voxel_bins_x)&&vy<static_cast<int>(voxel_bins_y)&&vz<static_cast<int>(number_of_bins);
                                    if(inside){const int v=(vz*voxel_bins_y+vy)*voxel_bins_x+vx;
                                        sycl::atomic_ref<DoseAtomicT,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(voxel_dose_device[v]).fetch_add(static_cast<DoseAtomicT>(recoil));}
                                    sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(deposited_device[frag.parent_history]).fetch_add(recoil);
                                    grid_deposit_split_device(grid_deposited_in_device,grid_deposited_out_device,inside,recoil);
                                    schneider_energy_add_device(schneider_diag_device,SchneiderDiagSlot::SecondaryDepositedMicroMeV,recoil);
                                    cinel02_species_energy_add_device(cinel02_species_energy_device,ledger_species_idx,3U,recoil);
                                    if(inside)cinel02_species_energy_add_device(cinel02_species_energy_device,ledger_species_idx,4U,recoil);
                                }
                            }

                            ++sec_steps;
                            ++local_sec_steps;
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
                                const int sec_slot = unified_secondary_species>=0 ? unified_secondary_species : (generic_recoil ? 18 : 19);
                                sec_prof[sec_slot]+=1;
                                const float sec_range = unified_secondary ? unified_secondary_state.range(sec_e) : 1.0e30f;
                                if(sec_range<0.5f)sec_prof[20+sec_slot]+=1;
                                sec_prof[40+sec_slot]+=static_cast<std::uint64_t>(dE*1e6f);
                            }
                        }
                        if(segment_paused){
                            SecondaryResumeState saved;
                            saved.sec_terminal_recorded=sec_terminal_recorded;
                            saved.unified_secondary_escaped_ct=unified_secondary_escaped_ct;
                            saved.continuous_species_tally=continuous_species_tally;
                            saved.sec_e=sec_e;
                            saved.sec_x=sec_x;
                            saved.sec_y=sec_y;
                            saved.sec_z=sec_z;
                            saved.sec_dx=sec_dx;
                            saved.sec_dy=sec_dy;
                            saved.sec_dz=sec_dz;
                            saved.pending_sec_depth_MeV=pending_sec_depth_MeV;
                            saved.pending_sec_voxel_MeV=pending_sec_voxel_MeV;
                            saved.pending_sec_bin=pending_sec_bin;
                            saved.pending_sec_voxel=pending_sec_voxel;
                            saved.unified_secondary_counter=unified_secondary_counter;
                            saved.unified_secondary_state=unified_secondary_state;
                            saved.unified_secondary_audit=unified_secondary_audit;
                            saved.sec_steps=sec_steps;
                            saved.local_sec_rate_queries=local_sec_rate_queries;
                            saved.local_sec_steps=local_sec_steps;
                            for(int j=0;j<6;++j)saved.he4_audit[j]=he4_audit[j];
                            if constexpr(CARBON_SECONDARY_STEP_PROFILE)for(int j=0;j<60;++j)saved.sec_prof[j]=sec_prof[j];
                            resume_states[state_idx]=saved;resume_ready[state_idx]=1;keep[item_id[0]]=1;
                            return; // Suspend: no terminal scoring or audit flush.
                        }
                        if(unified_secondary && CARBON_EM_LOCAL_AUDIT)flush_unified_em_audit(unified_audit,unified_secondary_audit);
                        if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
                            for(int sec_pi=0;sec_pi<60;++sec_pi) if(sec_prof[sec_pi]) {
                                sycl::atomic_ref<std::uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> pa(sec_step_profile_device[sec_pi]);pa.fetch_add(sec_prof[sec_pi]);
                            }
                        }
                        // Small per-step deposits must not contend directly on
                        if (he4_hazard_audit_device && frag.z == 2 && frag.a == 4) {
                            for (int i=0; i<6; ++i) {
                                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device, sycl::access::address_space::global_space>
                                    tally(he4_hazard_audit_device[i]);
                                tally.fetch_add(he4_audit[i]);
                            }
                        }
                        // Small per-step deposits must not contend directly on
                        // one global species scalar. Preserve original scoring
                        // guards and reduce per track; all replay breaks arrive
                        // here too. This changes diagnostics only, not dose.
                        cinel02_species_energy_add_device(
                            cinel02_species_energy_device, ledger_species_idx, 1U,
                            continuous_species_tally.all_MeV);
                        cinel02_species_energy_add_device(
                            cinel02_species_energy_device, ledger_species_idx, 2U,
                            continuous_species_tally.fov_MeV);
                        schneider_diag_add_device(schneider_diag_device,
                                                  SchneiderDiagSlot::SecondaryRateQueries,
                                                  local_sec_rate_queries);
                        schneider_diag_add_device(schneider_diag_device,
                                                  SchneiderDiagSlot::SecondarySteps,
                                                  local_sec_steps);

                        const bool step_limited =
                            sec_e > energy_cutoff_MeV && sec_steps >= kSecondaryMaxSteps;
                        if (sec_e > 0.0F) {
                            bool sec_terminal_scored = false;
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
                                    schneider_energy_add_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
                                }
                            } else if (!unified_secondary_escaped_ct && sec_z >= 0.0F && sec_z < phantom_length_mm) {
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
                                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                             sycl::memory_scope::device,
                                                             sycl::access::address_space::global_space>
                                                atomic_dose(dose_device[pending_sec_bin]);
                                            atomic_dose.fetch_add(static_cast<DepthAtomicT>(pending_sec_depth_MeV));
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
                                        score_he_isotope_origin_voxel_device(
                                            he_isotope_origin_voxel_dose_device,
                                            he_isotope_category, number_of_voxels,
                                            pending_sec_voxel,
                                            static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                                    }
                                            }
                                            pending_sec_voxel_MeV = 0.0F;
                                            pending_sec_voxel = cur_voxel;
                                        }
                                        if (cur_voxel >= 0) {
                                            pending_sec_voxel_MeV += sec_e;
                                            sec_terminal_scored = true;
                                            if (in_fov_dose_device != nullptr) {
                                                sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
                                                                 sycl::memory_scope::device,
                                                                 sycl::access::address_space::global_space>
                                                    atomic_in_fov(in_fov_dose_device[bin_z]);
                                                atomic_in_fov.fetch_add(static_cast<DepthAtomicT>(sec_e));
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
                                    schneider_energy_add_device(
                                        schneider_diag_device,
                                        SchneiderDiagSlot::SecondaryDepositedMicroMeV, sec_e);
                                    grid_deposit_split_device(
                                        grid_deposited_in_device,
                                        grid_deposited_out_device,
                                        sec_terminal_scored, sec_e);
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
                                schneider_energy_add_device(
                                    schneider_diag_device,
                                    SchneiderDiagSlot::SecondaryEscapedMicroMeV, sec_e);
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
                            sycl::atomic_ref<DepthAtomicT, sycl::memory_order::relaxed,
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
                                score_he_isotope_origin_voxel_device(
                                    he_isotope_origin_voxel_dose_device,
                                    he_isotope_category, number_of_voxels,
                                    pending_sec_voxel,
                                    static_cast<DoseAtomicT>(pending_sec_voxel_MeV));
                            }
                        }
                    });
            });
            sec_event.wait_and_throw();
                secondary_kernel_seconds += event_duration_seconds(sec_event);
                ++segment_rounds;
                if (finish_tail) break; // No suspended tracks; avoid an empty compaction.
                const auto compact_start=std::chrono::steady_clock::now();
                const unsigned resume_groups=(resume_active+255)/256;
                queue.parallel_for(sycl::nd_range<1>(resume_groups*256,256),[=](sycl::nd_item<1> it){
                    const auto i=it.get_global_linear_id();const unsigned flag=i<resume_active?keep[i]:0;
                    const auto rank=sycl::exclusive_scan_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    const auto count=sycl::reduce_over_group(it.get_group(),flag,sycl::plus<unsigned>());
                    if(i<resume_active)ranks[i]=rank;
                    if(it.get_local_linear_id()==0)block_counts[it.get_group_linear_id()]=count;
                }).wait_and_throw();
                queue.single_task([=](){unsigned sum=0;for(unsigned i=0;i<resume_groups;++i){block_offsets[i]=sum;sum+=block_counts[i];}*active_count=sum;}).wait_and_throw();
                queue.parallel_for(sycl::range<1>(resume_active),[=](sycl::id<1> id){const auto i=id[0];if(keep[i])next_order[block_offsets[i/256]+ranks[i]]=active_order[i];}).wait_and_throw();
                queue.copy(active_count,&resume_active,1).wait_and_throw();std::swap(active_order,next_order);
                compact_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-compact_start).count();
                }
                if (segment_secondaries) std::cout<<"[segmented-secondary] rounds="<<segment_rounds<<" compaction_s="<<compact_seconds<<"\n";
                mem_tracker.free(resume_states);
                for(auto* ptr:{resume_ready,active_order,next_order,keep,ranks,block_counts,block_offsets,active_count})mem_tracker.free(ptr);
                const auto batch_end = generation_end;
                generation_begin = batch_end;
                queue.copy(secondary_count_device, &secondary_count_host, 1)
                    .wait_and_throw();
                generation_end = sycl::min(
                    secondary_count_host, static_cast<std::uint32_t>(max_secondaries));
                std::cout << "[progress] secondary batch completed: ["
                          << batch_begin << ", " << batch_end << ") particles; queued="
                          << generation_end << std::endl;
            }
            std::cout << "[secondary-schedule] grouping_seconds="
                      << secondary_group_seconds << "\n";
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

    runtime_steps.finish();
    RuntimeScope runtime_post("transport_readback_and_host_finalize");
    std::vector<DepthAtomicT> dose_device_host(number_of_bins);
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

    std::vector<double> primary_voxel_track_length_host;
    if (enable_primary_voxel_fluence) {
        std::vector<DoseAtomicT> device_host(number_of_voxels);
        queue.copy(primary_voxel_track_length_device, device_host.data(),
                   number_of_voxels).wait_and_throw();
        primary_voxel_track_length_host.resize(number_of_voxels);
        std::transform(device_host.begin(), device_host.end(),
                       primary_voxel_track_length_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> charged_origin_voxel_dose_host;
    std::array<double, 6> he4_hazard_audit_host{};
    if (he4_hazard_audit_device)
        queue.copy(he4_hazard_audit_device, he4_hazard_audit_host.data(), 6).wait_and_throw();
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
    std::vector<double> he_isotope_origin_voxel_dose_host;
    if (enable_charged_origin_voxel_scoring) {
        const auto value_count =
            he_isotope_origin_category_count * number_of_voxels;
        std::vector<DoseAtomicT> device_host(value_count);
        queue.copy(he_isotope_origin_voxel_dose_device, device_host.data(),
                   value_count).wait_and_throw();
        he_isotope_origin_voxel_dose_host.resize(value_count);
        std::transform(device_host.begin(), device_host.end(),
                       he_isotope_origin_voxel_dose_host.begin(),
                       [](DoseAtomicT val) { return static_cast<double>(val); });
    }

    std::vector<double> in_fov_dose_host;
    if (enable_voxel_scoring && in_fov_dose_device != nullptr) {
        std::vector<DepthAtomicT> in_fov_device_host(number_of_bins);
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
    std::array<std::uint32_t,12> elastic_counts_host{};
    std::array<float,3> elastic_energy_host{};
    if(use_all_elastic) {
        auto& counts=elastic_counts_host;queue.copy(elastic_audit,counts.data(),12);
        queue.copy(elastic_energy,elastic_energy_host.data(),3).wait_and_throw();
        std::cout<<"[all-ion-elastic] primary="<<counts[0]<<" secondary="<<counts[1]
                 <<" bad_queries="<<counts[2]<<" unsupported_recoils="<<counts[3]<<'\n';
        std::cout<<"[elastic-failure-sites] primary_rate="<<counts[4]<<" primary_draw="<<counts[5]<<" recoil_stopping="<<counts[6]<<" secondary_rate="<<counts[7]<<" secondary_draw="<<counts[8]<<" null_post_em="<<counts[9]<<'\n';
        if(counts[2]||counts[3])throw std::runtime_error("All-ion elastic coverage failure: missing bank query or recoil transport; dose rejected");
    }
    std::vector<std::uint64_t> inelastic_reaction_host;
    if (primary_survival_device != nullptr) {
        primary_survival_host.resize(number_of_bins);
        inelastic_reaction_host.resize(number_of_bins);
        queue.copy(primary_survival_device, primary_survival_host.data(), number_of_bins);
        queue.copy(inelastic_reaction_device, inelastic_reaction_host.data(), number_of_bins);
    }
    std::vector<float> deposited_host(number_of_histories);
    std::vector<float> sampled_incident_host(number_of_histories);
    std::vector<float> escaped_host(number_of_histories);
    std::vector<std::uint32_t> steps_host(number_of_histories);
    std::vector<float> untracked_host(number_of_histories, 0.0F);
    queue.copy(deposited_device, deposited_host.data(), number_of_histories);
    queue.copy(sampled_incident_device, sampled_incident_host.data(), number_of_histories);
    queue.copy(escaped_device, escaped_host.data(), number_of_histories);
    queue.copy(steps_device, steps_host.data(), number_of_histories);
    if (untracked_nuclear_device != nullptr) {
        queue.copy(untracked_nuclear_device, untracked_host.data(), number_of_histories);
    }
    std::vector<float> other_terminal_host(number_of_histories, 0.0F);
    if (other_terminal_energy_device != nullptr) {
        queue.copy(other_terminal_energy_device, other_terminal_host.data(), number_of_histories);
    }
    std::vector<float> cutoff_stopped_host(number_of_histories, 0.0F);
    if (cutoff_stopped_energy_device != nullptr) {
        queue.copy(cutoff_stopped_energy_device, cutoff_stopped_host.data(), number_of_histories);
    }
        if(unified_em){
        std::uint64_t audit[8]{};queue.copy(unified_audit,audit,8).wait_and_throw();
        std::cout<<"[unified-em-audit]";for(auto count:audit)std::cout<<" "<<count;std::cout<<"\n";
        if(audit[0]) {
            unsigned count=0;queue.copy(unified_failure_count,&count,1).wait_and_throw();
            std::vector<UnifiedEmFailureRecord> records(std::min(count,16u));
            queue.copy(unified_failure_records,records.data(),records.size()).wait_and_throw();
            for(const auto& f:records)std::cerr<<"[unified-em-failure] reason="<<f.reason<<" section="<<f.section<<" Z="<<f.z<<" A="<<f.a<<" kinetic_MeV="<<f.energy<<" density="<<f.density<<" step_mm="<<f.step<<"\n";
            std::cerr<<"[runtime-rejected-kernel] primary_s="<<primary_kernel_seconds<<" secondary_s="<<secondary_kernel_seconds<<" histories="<<number_of_histories<<"\n";
            throw std::runtime_error("Unified EM missing domain or sampling failure; dose rejected");
        }
        mem_tracker.free(unified_failure_count);mem_tracker.free(unified_failure_records);
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) {
            std::uint64_t secprof[60]{};queue.copy(sec_step_profile_device,secprof,60).wait_and_throw();
            std::cout<<"[secondary-step-profile] slot steps short depMeV\n";
            for(int sec_si=0;sec_si<20;++sec_si)
                std::cout<<"[secondary-step-profile] "<<sec_si<<" "<<secprof[sec_si]<<" "<<secprof[20+sec_si]<<" "<<(static_cast<double>(secprof[40+sec_si])*1e-6)<<"\n";
        }
        mem_tracker.free(delta_mean_device);mem_tracker.free(unified_energy_index);mem_tracker.free(unified_sections);mem_tracker.free(unified_materials);mem_tracker.free(unified_species);mem_tracker.free(unified_records);
        mem_tracker.free(unified_nodes);mem_tracker.free(unified_segments);mem_tracker.free(unified_audit);
        if constexpr(CARBON_SECONDARY_STEP_PROFILE) mem_tracker.free(sec_step_profile_device);
    }
    std::uint64_t schneider_inelastic_host = 0;
    if (schneider_inelastic_device != nullptr) {
        queue.copy(schneider_inelastic_device, &schneider_inelastic_host, 1);
    }
    std::array<std::uint64_t, 4> terminal_counts_host{};
    if (primary_terminal_counts_device != nullptr) {
        queue.copy(primary_terminal_counts_device, terminal_counts_host.data(), 4).wait_and_throw();
    }
    std::vector<PrimaryFirstInteractionRecord> first_interactions_host;
    if (first_interactions_device != nullptr && first_interactions_count_device != nullptr) {
        std::uint32_t first_int_count = 0;
        queue.copy(first_interactions_count_device, &first_int_count, 1).wait_and_throw();
        const auto actual_count = std::min(first_int_count, static_cast<std::uint32_t>(number_of_histories));
        first_interactions_host.resize(actual_count);
        if (actual_count > 0) {
            queue.copy(first_interactions_device, first_interactions_host.data(), actual_count).wait_and_throw();
        }
    }
    // Only shared species and grid ledgers are backed by device buffers.
    std::array<float, kCinel02SpeciesEnergySlots> cinel02_species_energy_host{};
    std::array<std::uint64_t, kCinel02SpeciesTerminalSlots> cinel02_species_terminal_host{};
    double grid_deposited_in_host = 0.0;
    double grid_deposited_out_host = 0.0;
    queue.copy(cinel02_species_energy_device, cinel02_species_energy_host.data(),
               kCinel02SpeciesEnergySlots);
    queue.copy(cinel02_species_terminal_device, cinel02_species_terminal_host.data(),
               kCinel02SpeciesTerminalSlots);
    if (grid_deposited_in_device != nullptr)
        queue.copy(grid_deposited_in_device, &grid_deposited_in_host, 1);
    if (grid_deposited_out_device != nullptr)
        queue.copy(grid_deposited_out_device, &grid_deposited_out_host, 1);
    std::array<std::uint64_t, 44> schneider_diag_host{};
    static_assert(44 == static_cast<std::size_t>(SchneiderDiagSlot::Count),
                  "Schneider host mirror must match the device schema");
    std::array<float, 12> schneider_float_host{};
    static_assert(12 == static_cast<std::size_t>(SchneiderFloatSlot::Count),
                  "Schneider float mirror must match the device schema");
    if (schneider_diag_device != nullptr) {
        queue.copy(schneider_diag_device, schneider_diag_host.data(), kSchneiderDiagSlots);
    }
    if (schneider_float_device != nullptr) {
        queue.copy(schneider_float_device, schneider_float_host.data(), kSchneiderFloatSlots);
    }
    std::array<std::uint32_t, 2> schneider_miss_counts_host{0, 0};
    std::array<std::uint32_t, 2> schneider_track_counts_host{0, 0};
    if (schneider_miss_count_device != nullptr) {
        queue.copy(schneider_miss_count_device, schneider_miss_counts_host.data(), 2);
    }
    if (schneider_track_count_device != nullptr) {
        queue.copy(schneider_track_count_device, schneider_track_counts_host.data(), 2);
    }
    uint32_t overflow_count_host = 0;
    float overflow_energy_host = 0.0F;
    if (secondary_overflow_count_device != nullptr) {
        queue.copy(secondary_overflow_count_device, &overflow_count_host, 1);
        queue.copy(secondary_overflow_energy_device, &overflow_energy_host, 1);
    }
    queue.wait_and_throw();

    // Per-record Schneider logs: counts are exact after the fence above.
    // Staged into host vectors here (device buffers are freed below);
    // moved into TransportResult after its declaration.
    std::vector<SchneiderMissRecord> schneider_miss_host;
    std::vector<SchneiderUnsupportedTrack> schneider_track_host;
    if (schneider_miss_device != nullptr && schneider_miss_counts_host[0] > 0) {
        const auto n_miss = std::min<std::uint32_t>(
            schneider_miss_counts_host[0], kSchneiderMissLogCap);
        schneider_miss_host.resize(n_miss);
        queue.copy(schneider_miss_device, schneider_miss_host.data(), n_miss)
            .wait_and_throw();
    }
    if (schneider_track_log_device != nullptr && schneider_track_counts_host[0] > 0) {
        const auto n_trk = std::min<std::uint32_t>(
            schneider_track_counts_host[0], kSchneiderTrackLogCap);
        schneider_track_host.resize(n_trk);
        queue.copy(schneider_track_log_device, schneider_track_host.data(), n_trk)
            .wait_and_throw();
    }

    if (primary_loss_query_audit) {
        std::array<std::uint64_t,28> audit{};
        queue.copy(primary_loss_query_audit, audit.data(), audit.size()).wait_and_throw();
        free_device(primary_loss_query_audit);
        for (int b=0;b<14;++b)
            std::cout << "PRIMARY_LOSS_QUERY," << b << ',' << audit[b] << ',' << audit[b+14] << '\n';
    }
    if (ion_stopping_failures) {
        std::array<std::uint32_t, 8> failures{};
        queue.copy(ion_stopping_failures, failures.data(), failures.size()).wait_and_throw();
        free_device(ion_stopping_failures);
        free_device(schneider_ion_sp_device);
        if (failures[0]) {
            float energy, density;
            std::memcpy(&energy, &failures[4], sizeof(float));
            std::memcpy(&density, &failures[5], sizeof(float));
            throw std::runtime_error("Schneider ion stopping domain failure: " + std::to_string(failures[0]) +
                " first Z=" + std::to_string(failures[1]) + " A=" + std::to_string(failures[2]) +
                " section=" + std::to_string(failures[3]) + " E_MeVu=" + std::to_string(energy) +
                " rho=" + std::to_string(density));
        }
    }
    if (fluct_domain_failures) {
        std::uint32_t failures=0;
        queue.copy(fluct_domain_failures, &failures, 1).wait_and_throw();
        free_device(fluct_domain_failures);
        if (failures) throw std::runtime_error("Fraction-axis fluctuation outside validated grid: " + std::to_string(failures));
    }
    std::array<std::uint64_t, 10> schneider_delta_energy_host{};
    std::array<std::uint64_t,7> electron_joint_diag_host{};
    std::array<std::uint64_t,3> material_untracked_host{};
    if(material_untracked_device)queue.copy(material_untracked_device,material_untracked_host.data(),3).wait_and_throw();
    if(material_failure_device && material_untracked_host[2]) {
        MaterialPacketFailure f;queue.copy(material_failure_device,&f,1).wait_and_throw();
        const auto& c=f.before.cursor;
        std::cerr.precision(17);
        std::cerr<<"[material-packet-failure] history="<<f.history<<" step="<<f.step<<" rng="<<f.rng
                 <<" status="<<int(f.after.status)<<" gap="<<int(f.after.gap)<<" advance="<<int(f.advance.status)
                 <<" boundary="<<int(f.advance.boundary.status)<<" section="<<c.section
                 <<" rho="<<c.density_g_cm3<<" physical_rho="<<c.physical_density_g_cm3
                 <<" source="<<c.source<<" row="<<c.row<<" pdg="<<c.pdg<<" KE="<<c.energy_MeV
                 <<" W="<<f.before.weight_MeV<<" fraction="<<c.fraction
                 <<" position="<<c.position[0]<<','<<c.position[1]<<','<<c.position[2]
                 <<" advances="<<f.after.advances<<" crossings="<<f.after.boundary_restarts
                 <<" restarts="<<f.after.source_restarts<<" residual="<<f.advance.energy_residual_MeV<<'\n';
    }
    if(electron_joint_diag_device)queue.copy(electron_joint_diag_device,electron_joint_diag_host.data(),7).wait_and_throw();
    if (schneider_delta_energy_device != nullptr) {
        queue.copy(schneider_delta_energy_device,
                   schneider_delta_energy_host.data(),
                   schneider_delta_energy_host.size()).wait_and_throw();
    }

    std::vector<LongitudinalDomainRecord> longitudinal_domain_host;
    if (longitudinal_domain_device) {
        longitudinal_domain_host.resize(std::min<std::uint64_t>(
            schneider_delta_energy_host[6], kLongitudinalDomainLogCap));
        if (!longitudinal_domain_host.empty())
            queue.copy(longitudinal_domain_device, longitudinal_domain_host.data(),
                       longitudinal_domain_host.size()).wait_and_throw();
        std::sort(longitudinal_domain_host.begin(), longitudinal_domain_host.end(),
            [](const auto& a, const auto& b) {
                return a.history < b.history || (a.history == b.history && a.step < b.step);
            });
    }
    // Free buffers
    free_device(electron_path_bounds_device);
    free_device(water_electron_channels_device);free_device(water_electron_samples_device);
    free_device(water_electron_nodes_device);free_device(water_electron_heads_device);free_device(water_electron_radius_device);
    free_device(electron_joint_channels_device);free_device(electron_joint_samples_device);free_device(electron_joint_diag_device);
    free_device(electron_path_ranges_device);free_device(electron_path_vectors_device);
    free_immutable_device(table_device);
    free_immutable_device(energy_grid_device);
    free_immutable_device(cumulative_range_device);
    free_immutable_device(cross_section_device);
    free_device(target_h_fraction_device);
    free_device(dose_device);
    free_device(primary_survival_device);
    free_device(inelastic_reaction_device);
    free_device(primary_terminal_counts_device);
    free_device(first_interactions_device);
    free_device(first_interactions_count_device);
    free_device(in_fov_dose_device);
    free_device(voxel_dose_device);
    free_device(primary_voxel_track_length_device);
    free_device(charged_origin_voxel_dose_device);
    free_device(be_isotope_origin_voxel_dose_device);
    free_device(he4_hazard_audit_device);
    free_device(he_isotope_origin_voxel_dose_device);
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
    free_device(cinel02_ct_rate_groups_device);
    free_device(cinel02_ct_rate_samples_device);
    free_device(cinel02_species_energy_device);
    free_device(cinel02_species_terminal_device);
    free_device(untracked_nuclear_device);
    free_device(other_terminal_energy_device);
    free_device(cutoff_stopped_energy_device);
    free_device(fluct_energy_device);
    free_device(fluct_density_device);
    free_device(fluct_probability_device);
    free_device(fluct_quantile_device);
    free_device(secondary_overflow_count_device);
    free_device(secondary_overflow_energy_device);
    free_device(escaped_device);
    free_device(sampled_incident_device);
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
    free_device(schneider_delta_energies_device);
    free_device(schneider_delta_fractions_device);
    free_device(schneider_delta_radii_device);
    free_device(schneider_delta_source_eligible_device);
    free_device(schneider_delta_energy_device);
    free_device(longitudinal_domain_device);
    free_device(schneider_long_energies_device);
    free_device(schneider_long_fractions_device);
    free_device(schneider_long_lambdas_device);
    free_device(schneider_primary_xs_device);
    free_device(schneider_inelastic_device);
    free_device(schneider_diag_device);
    free_device(schneider_float_device);
    free_device(schneider_miss_device);
    free_device(schneider_track_log_device);
    free_device(schneider_miss_count_device);
    free_device(schneider_track_count_device);

    TransportResult result;
    result.primary_elastic_interactions=elastic_counts_host[0];
    result.secondary_elastic_interactions=elastic_counts_host[1];
    result.elastic_post_em_null_collisions=elastic_counts_host[9]+elastic_counts_host[10];
    result.elastic_local_deposited_energy_MeV=elastic_energy_host[0];
    result.elastic_queued_charged_energy_MeV=elastic_energy_host[1];
    result.elastic_queue_overflow_energy_MeV=elastic_energy_host[2];
    result.elastic_queue_overflow=elastic_counts_host[11];
    result.schneider_miss_log = std::move(schneider_miss_host);
    result.schneider_primary_delta_tail_moved_MeV =
        static_cast<double>(schneider_delta_energy_host[0]) * 1.0e-6;
    result.schneider_primary_delta_tail_fallback_MeV =
        static_cast<double>(schneider_delta_energy_host[1]) * 1.0e-6;
    result.schneider_primary_delta_tail_escaped_scorer_MeV =
        static_cast<double>(schneider_delta_energy_host[2]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_moved_MeV =
        static_cast<double>(schneider_delta_energy_host[3]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_fallback_MeV =
        static_cast<double>(schneider_delta_energy_host[4]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_escaped_scorer_MeV =
        static_cast<double>(schneider_delta_energy_host[5]) * 1.0e-6;
    result.schneider_primary_delta_longitudinal_domain_queries = schneider_delta_energy_host[6];
    result.schneider_primary_delta_longitudinal_domain_energy_MeV =
        static_cast<double>(schneider_delta_energy_host[7]) * 1e-6;
    result.schneider_primary_delta_longitudinal_invalid_marches = schneider_delta_energy_host[9];
    result.longitudinal_diagnostic_density_g_cm3 = schneider_long_diagnostic_density;
    result.longitudinal_domain_log = std::move(longitudinal_domain_host);
    result.electron_joint_diagnostics={electron_joint_diag_host[0],electron_joint_diag_host[1],electron_joint_diag_host[2],
        static_cast<double>(electron_joint_diag_host[3])*1e-6,static_cast<double>(electron_joint_diag_host[4])*1e-6,static_cast<double>(electron_joint_diag_host[5])*1e-6,electron_joint_diag_host[6]};
    result.electron_ordered_path_sha256=electron_path_sha256;
    result.material_electron_photon_untracked_MeV=double(material_untracked_host[0])*1e-6;
    result.material_electron_untracked_MeV=double(material_untracked_host[0]+material_untracked_host[1])*1e-6;
    free_device(material_untracked_device);
    free_device(material_failure_device);
    if(short_range_hits_device) {
        std::uint64_t hits=0;
        queue.copy(short_range_hits_device,&hits,1).wait_and_throw();
        std::cout<<"[research-short-range] shortcut_packets="<<hits<<'\n';
        free_device(short_range_hits_device);
    }
    result.schneider_unsupported_tracks = std::move(schneider_track_host);
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
    result.primary_voxel_track_length_mm =
        std::move(primary_voxel_track_length_host);
    result.charged_origin_voxel_deposited_energy_MeV =
        std::move(charged_origin_voxel_dose_host);
    result.be_isotope_origin_voxel_deposited_energy_MeV =
        std::move(be_isotope_origin_voxel_dose_host);
    result.he_isotope_origin_voxel_deposited_energy_MeV =
        std::move(he_isotope_origin_voxel_dose_host);
    result.helium4_hazard_audit = he4_hazard_audit_host;
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
            if (fragment.z == 2 && (fragment.a == 3 || fragment.a == 4 || fragment.a == 6)) {
                result.helium_birth_records.push_back({
                    fragment.parent_history, static_cast<unsigned>(fragment.generation),
                    static_cast<unsigned>(fragment.a), fragment.energy_MeV,
                    fragment.pos_x_mm, fragment.pos_y_mm, fragment.pos_z_mm,
                    fragment.dir_x, fragment.dir_y, fragment.dir_z, fragment.weight});
            }
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

    result.initial_energy_MeV =
        std::accumulate(sampled_incident_host.begin(), sampled_incident_host.end(), 0.0);
    result.total_deposited_energy_MeV =
        std::accumulate(deposited_host.begin(), deposited_host.end(), 0.0);
    result.escaped_energy_MeV =
        std::accumulate(escaped_host.begin(), escaped_host.end(), 0.0);
    if (config.enable_primary_loss_query_audit && !enable_inelastic) {
        // Host-only summary of existing readback: no device, RNG or scoring
        // changes. With nuclear transport disabled this is primary energy.
        const double mean = result.escaped_energy_MeV / escaped_host.size();
        double second = 0.0, third = 0.0;
        std::size_t positive = 0;
        for (const float energy : escaped_host) {
            const double delta = static_cast<double>(energy) - mean;
            second += delta * delta;
            third += delta * delta * delta;
            positive += energy > 0.0F;
        }
        second /= escaped_host.size();
        third /= escaped_host.size();
        auto sorted = escaped_host;
        std::sort(sorted.begin(), sorted.end());
        const auto quantile = [&](double p) {
            const double index = p * (sorted.size()-1);
            const auto low = static_cast<std::size_t>(index);
            const auto high = std::min(low+1,sorted.size()-1);
            return sorted[low] + (index-low)*(sorted[high]-sorted[low]);
        };
        const auto old_precision = std::cout.precision(17);
        std::cout << "PRIMARY_ESCAPE_ENERGY," << escaped_host.size() << ','
                  << positive << ',' << mean << ',' << second << ','
                  << (second > 0 ? third/std::pow(second,1.5) : 0.0) << ','
                  << quantile(.05) << ',' << quantile(.5) << ',' << quantile(.95) << '\n';
        std::cout.precision(old_precision);
    }
    result.untracked_nuclear_energy_MeV =
        std::accumulate(untracked_host.begin(), untracked_host.end(), 0.0);
    result.primary_inelastic_terminated_count = terminal_counts_host[0];
    result.primary_escaped_ct_count = terminal_counts_host[1];
    result.primary_stopped_count = terminal_counts_host[2];
    result.primary_other_terminal_count = terminal_counts_host[3];
    result.primary_inelastic_removed_kinetic_MeV = result.untracked_nuclear_energy_MeV;
    result.primary_other_terminal_kinetic_MeV =
        std::accumulate(other_terminal_host.begin(), other_terminal_host.end(), 0.0);
    result.primary_cutoff_stopped_energy_MeV =
        std::accumulate(cutoff_stopped_host.begin(), cutoff_stopped_host.end(), 0.0);
    result.primary_first_interactions = std::move(first_interactions_host);
    result.secondary_queue_overflow = overflow_count_host;
    result.secondary_queue_overflow_energy_MeV = static_cast<double>(overflow_energy_host);
    // Retired CINEL02 diagnostics retain their zero-initialized result fields.
    // Current shared species/grid measurements must still be copied.
    for (std::size_t i = 0; i < kCinel02SpeciesEnergySlots; ++i) {
        result.cinel02_species_transport_ledger_MeV[i] =
            static_cast<double>(cinel02_species_energy_host[i]);
    }
    result.in_grid_deposited_energy_MeV = grid_deposited_in_host;
    result.outside_grid_deposited_energy_MeV = grid_deposited_out_host;
    result.cinel02_species_terminal_reason_counts = cinel02_species_terminal_host;

    // Aggregate the flat Schneider device schema into named diagnostics.
    {
        auto& sch = result.schneider_diagnostics;
        const auto slot = [&](SchneiderDiagSlot s) -> std::uint64_t {
            return schneider_diag_host[static_cast<std::uint32_t>(s)];
        };
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        sch.primary_rate_queries = slot(SchneiderDiagSlot::PrimaryRateQueries);
        sch.primary_hazards = slot(SchneiderDiagSlot::PrimaryHazards);
        sch.primary_exact_target_hits = slot(SchneiderDiagSlot::PrimaryExactTargetHits);
        sch.primary_missing_projectile = slot(SchneiderDiagSlot::PrimaryMissingProjectile);
        sch.primary_missing_target = slot(SchneiderDiagSlot::PrimaryMissingTarget);
        sch.primary_below_domain = slot(SchneiderDiagSlot::PrimaryBelowDomain);
        sch.primary_above_domain = slot(SchneiderDiagSlot::PrimaryAboveDomain);
        sch.primary_energy_gap_misses = slot(SchneiderDiagSlot::PrimaryEnergyGapMisses);
        sch.primary_empty_nodes = slot(SchneiderDiagSlot::PrimaryEmptyNodes);
        sch.primary_events_replayed = slot(SchneiderDiagSlot::PrimaryEventsReplayed);
        sch.primary_charged_products_born = slot(SchneiderDiagSlot::PrimaryChargedBorn);
        sch.primary_charged_products_queued = slot(SchneiderDiagSlot::PrimaryChargedQueued);
        sch.primary_charged_cutoff_kills = slot(SchneiderDiagSlot::PrimaryChargedCutoffKills);
        sch.primary_be6_kills = slot(SchneiderDiagSlot::PrimaryBe6Kills);
        sch.primary_queue_overflows = slot(SchneiderDiagSlot::PrimaryQueueOverflows);
        sch.secondary_tracks_started = slot(SchneiderDiagSlot::SecondaryTracksStarted);
        sch.secondary_steps = slot(SchneiderDiagSlot::SecondarySteps);
        sch.secondary_rate_queries = slot(SchneiderDiagSlot::SecondaryRateQueries);
        sch.secondary_hazards = slot(SchneiderDiagSlot::SecondaryHazards);
        sch.secondary_exact_target_hits = slot(SchneiderDiagSlot::SecondaryExactTargetHits);
        sch.secondary_missing_projectile = slot(SchneiderDiagSlot::SecondaryMissingProjectile);
        sch.secondary_missing_target = slot(SchneiderDiagSlot::SecondaryMissingTarget);
        sch.secondary_below_domain = slot(SchneiderDiagSlot::SecondaryBelowDomain);
        sch.secondary_above_domain = slot(SchneiderDiagSlot::SecondaryAboveDomain);
        sch.secondary_energy_gap_misses = slot(SchneiderDiagSlot::SecondaryEnergyGapMisses);
        sch.secondary_empty_nodes = slot(SchneiderDiagSlot::SecondaryEmptyNodes);
        sch.secondary_events_replayed = slot(SchneiderDiagSlot::SecondaryEventsReplayed);
        sch.secondary_charged_products_born = slot(SchneiderDiagSlot::SecondaryChargedBorn);
        sch.secondary_charged_products_queued = slot(SchneiderDiagSlot::SecondaryChargedQueued);
        sch.secondary_charged_cutoff_kills = slot(SchneiderDiagSlot::SecondaryChargedCutoffKills);
        sch.secondary_be6_kills = slot(SchneiderDiagSlot::SecondaryBe6Kills);
        sch.secondary_queue_overflows = slot(SchneiderDiagSlot::SecondaryQueueOverflows);
        sch.secondary_stopped_before_replay = slot(SchneiderDiagSlot::SecondaryStoppedBeforeReplay);
        sch.secondary_post_em_null_collisions = slot(SchneiderDiagSlot::SecondaryPostEmNullCollisions);
        sch.secondary_post_em_null_energy_MeV = fslot(SchneiderFloatSlot::PostEmNullEnergy);
        sch.primary_post_em_null_collisions = slot(SchneiderDiagSlot::PrimaryPostEmNullCollisions);
        // Exact fixed-point secondary sub-ledger (micro-MeV -> MeV). These
        // fields were historically always zero (never assigned); on the
        // Schneider path they now accumulate every secondary deposit/escape
        // at the same sites as the per-history arrays.
        result.secondary_deposited_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryDepositedMicroMeV)) * 1.0e-6;
        result.secondary_escaped_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::SecondaryEscapedMicroMeV)) * 1.0e-6;
        sch.be6_topas_compat_kills = slot(SchneiderDiagSlot::Be6TopasCompatKills);
        sch.unsupported_projectile_steps = slot(SchneiderDiagSlot::UnsupportedProjectileSteps);
        sch.unsupported_projectile_tracks = slot(SchneiderDiagSlot::UnsupportedProjectileTracks);
        sch.unsupported_be6_tracks = slot(SchneiderDiagSlot::UnsupportedBe6Tracks);
        sch.unsupported_projectile_birth_energy_MeV =
            static_cast<double>(slot(SchneiderDiagSlot::UnsupportedProjectileBirthEnergyMicroMeV)) * 1.0e-6;
        // Dropped counts come from the log-count buffers (device-side
        // overflow tallies), not the diag slots (unused for these two).
        sch.miss_log_dropped = schneider_miss_counts_host[1];
        sch.unsupported_log_dropped = schneider_track_counts_host[1];
        sch.unsupported_targets = slot(SchneiderDiagSlot::UnsupportedTargets);
        sch.queue_overflows = slot(SchneiderDiagSlot::QueueOverflows);
        sch.primary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::PrimaryMismatchSum);
        sch.primary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::PrimaryMismatchMax);
        sch.secondary_selected_energy_mismatch_sum = fslot(SchneiderFloatSlot::SecondaryMismatchSum);
        sch.secondary_selected_energy_mismatch_max = fslot(SchneiderFloatSlot::SecondaryMismatchMax);
        sch.lookup_failure_energy_MeV = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        sch.be6_kill_energy_MeV = fslot(SchneiderFloatSlot::Be6KillEnergy);
        sch.neutral_product_kinetic_MeV = fslot(SchneiderFloatSlot::NeutralProductKinetic);
        sch.reaction_q_residual_MeV = fslot(SchneiderFloatSlot::ReactionQResidual);
    }

    // 8-part energy accounting ledger
    result.energy_ledger.E_continuous_ionizing = result.total_deposited_energy_MeV;
    result.energy_ledger.E_nuclear_local = 0.0;
    result.energy_ledger.E_transported_secondaries = result.queued_secondary_energy_MeV;
    result.energy_ledger.E_escaped_charged = result.escaped_energy_MeV;
    result.energy_ledger.E_neutral = result.untransported_neutral_energy_MeV;
    result.energy_ledger.E_cutoff_kill = result.primary_cutoff_stopped_energy_MeV;
    result.energy_ledger.E_unsupported = result.untransported_unsupported_charged_energy_MeV;
    result.energy_ledger.E_queue_overflow = result.secondary_queue_overflow_energy_MeV;
    // Split Schneider nuclear-vertex ledger, accumulated on device from
    // actually executed vertices (informational itemization; the legacy
    // untracked sink behavior above is unchanged this step).
    {
        const auto fslot = [&](SchneiderFloatSlot s) -> double {
            return static_cast<double>(schneider_float_host[static_cast<std::uint32_t>(s)]);
        };
        result.energy_ledger.E_be6_kill = fslot(SchneiderFloatSlot::Be6KillEnergy);
        result.energy_ledger.E_lookup_failure = fslot(SchneiderFloatSlot::LookupFailureEnergy);
        result.energy_ledger.E_neutral_product_kinetic =
            fslot(SchneiderFloatSlot::NeutralProductKinetic);
        result.energy_ledger.E_reaction_q_residual = fslot(SchneiderFloatSlot::ReactionQResidual);
        result.energy_ledger.E_unsupported_charged =
            fslot(SchneiderFloatSlot::UnsupportedProductEnergy);
        result.energy_ledger.E_out_of_domain = fslot(SchneiderFloatSlot::OutOfDomainEnergy);
        if (schneider_diag_device != nullptr) {
            result.energy_ledger.E_transported_secondaries =
                fslot(SchneiderFloatSlot::SecondaryTransportBirthEnergy);
        }
    }

    if (config.quality_reject_any_queue_overflow && overflow_count_host > 0) {
        throw std::runtime_error("Secondary particle queue overflow detected: discarded " +
                                 std::to_string(overflow_count_host) + " particles (" +
                                 std::to_string(overflow_energy_host) + " MeV). Shard must be split and rerun with fewer particles.");
    }

    result.nuclear_interactions =
        schneider_inelastic_host;
    result.total_steps = std::accumulate(steps_host.begin(), steps_host.end(), std::uint64_t{0});

    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.primary_kernel_seconds = primary_kernel_seconds;
    result.secondary_kernel_seconds = secondary_kernel_seconds;
    return result;
}

[[gnu::noinline]] TransportResult transport_sycl(const TransportConfig& config,
    const StoppingPowerTable& stopping_power,const CrossSectionTable& cross_section,
    const std::string& device_name,SyclTransportContext* context) {
#if CARBON_EM_SPECIALIZE
    if(config.em_model=="g4_material_joint_v1")
        return transport_sycl_impl<1>(config,stopping_power,cross_section,device_name,context);
    return transport_sycl_impl<0>(config,stopping_power,cross_section,device_name,context);
#else
    return transport_sycl_impl<-1>(config,stopping_power,cross_section,device_name,context);
#endif
}

}  // namespace carbon

#endif
