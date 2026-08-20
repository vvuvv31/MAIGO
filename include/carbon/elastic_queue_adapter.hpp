#pragma once

#include "carbon/elastic_application.hpp"
#include "carbon/particle.hpp"
#include "carbon/rng.hpp"

#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace carbon {

// Queue records intentionally keep the existing 48-byte transport ABI.  The
// route lineage in SecondaryParticle3D::reserved is therefore the existing
// charged/neutron/gamma tag; elastic origin is carried by this adapter's
// contribution and parent metadata rather than by changing the queue record.
enum class ElasticQueueOrigin : std::uint8_t {
    elastic = 1,
};

struct ElasticQueueParentMetadata {
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float time_ns{0.0F};
    std::int32_t parent_pdg_id{0};
    std::int16_t parent_atomic_number{0};
    std::int16_t parent_mass_number{0};
    std::uint8_t parent_generation{0};
    std::uint16_t parent_lineage{0};
    std::uint64_t parent_rng_stream{0};
    std::uint64_t interaction_index{0};
    ElasticQueueOrigin origin{ElasticQueueOrigin::elastic};
};

static_assert(std::is_trivially_copyable_v<ElasticQueueParentMetadata>);
static_assert(std::is_standard_layout_v<ElasticQueueParentMetadata>);

// Counter is uint64_t for device storage and std::atomic<uint64_t> for the
// host concurrency tests.  No queue ownership or allocation is hidden here.
template <typename Counter = std::uint64_t>
struct ElasticQueueStorage {
    SecondaryParticle3D* charged_records{nullptr};
    std::uint64_t charged_capacity{0};
    Counter* charged_counter{nullptr};
    Counter* charged_filled{nullptr};
    NeutralParticle3D* neutral_records{nullptr};
    std::uint64_t neutral_capacity{0};
    Counter* neutral_counter{nullptr};
    Counter* neutral_filled{nullptr};
    // Disable this route when neutral transport is not configured. Products
    // then remain explicit dropped ledger energy while charged products can
    // still use the existing queue.
    bool neutral_route_enabled{true};
};

using ElasticDeviceQueueStorage = ElasticQueueStorage<std::uint64_t>;
using ElasticHostAtomicQueueStorage = ElasticQueueStorage<std::atomic<std::uint64_t>>;

enum class ElasticQueueCommitStatus : std::uint8_t {
    success = 0,
    invalid_input,
    charged_route_overflow,
    neutral_route_overflow,
    both_routes_overflow,
    partial_route_commit,
};

struct ElasticScorerContribution {
    ElasticQueueOrigin origin{ElasticQueueOrigin::elastic};
    float position_x_mm{0.0F};
    float position_y_mm{0.0F};
    float position_z_mm{0.0F};
    float time_ns{0.0F};
    float local_deposit_MeV{0.0F};
    float primary_continuation_energy_MeV{0.0F};
    float queued_charged_energy_MeV{0.0F};
    float queued_neutral_energy_MeV{0.0F};
    float overflow_charged_energy_MeV{0.0F};
    float overflow_neutral_energy_MeV{0.0F};
};

struct ElasticQueueCommitResult {
    ElasticQueueCommitStatus status{ElasticQueueCommitStatus::invalid_input};
    std::uint32_t written_charged_count{0};
    std::uint32_t written_neutral_count{0};
    std::uint32_t dropped_charged_count{0};
    std::uint32_t dropped_neutral_count{0};
    std::uint32_t written_count{0};
    std::uint32_t dropped_count{0};
    float written_charged_energy_MeV{0.0F};
    float written_neutral_energy_MeV{0.0F};
    float dropped_charged_energy_MeV{0.0F};
    float dropped_neutral_energy_MeV{0.0F};
    float written_energy_MeV{0.0F};
    float dropped_energy_MeV{0.0F};
    float local_deposit_MeV{0.0F};
    float primary_continuation_energy_MeV{0.0F};
    float input_ledger_closure_residual_MeV{0.0F};
    float committed_ledger_closure_residual_MeV{0.0F};
    ElasticScorerContribution scorer{};

    [[nodiscard]] bool success() const noexcept {
        return status == ElasticQueueCommitStatus::success;
    }
};

static_assert(std::is_trivially_copyable_v<ElasticQueueStorage<>>);
static_assert(std::is_trivially_copyable_v<ElasticScorerContribution>);
static_assert(std::is_trivially_copyable_v<ElasticQueueCommitResult>);

namespace elastic_queue_detail {

inline constexpr std::uint32_t branch_role_elastic_charged = 0x70000000U;
inline constexpr std::uint32_t branch_role_elastic_neutral = 0x80000000U;

template <typename Counter>
inline std::uint64_t queue_atomic_fetch_add(Counter* counter,
                                            const std::uint64_t amount) noexcept {
    using CounterValue = std::remove_cv_t<Counter>;
    if constexpr (std::is_same_v<CounterValue, std::atomic<std::uint64_t>>) {
        return counter->fetch_add(amount, std::memory_order_relaxed);
    } else {
#ifdef CARBON_HAS_SYCL
        sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic(*counter);
        return atomic.fetch_add(amount);
#else
        // Raw uint64_t storage is only used on a SYCL device.  This fallback
        // keeps the header usable in a non-SYCL compile for serial tests.
        const auto previous = *counter;
        *counter += amount;
        return previous;
#endif
    }
}

inline bool finite(const float value) noexcept {
    return value == value && value <= std::numeric_limits<float>::max() &&
           value >= -std::numeric_limits<float>::max();
}

inline bool valid_position_and_time(const ElasticQueueParentMetadata& parent) noexcept {
    return finite(parent.position_x_mm) && finite(parent.position_y_mm) &&
           finite(parent.position_z_mm) && finite(parent.time_ns);
}

inline bool energy_close(const float left, const float right) noexcept {
    if (!finite(left) || !finite(right)) return false;
    const auto scale = left > right ? left : right;
    const auto tolerance = 1.0e-2F + 2.0e-3F * (scale > 1.0F ? scale : 1.0F);
    return left - right <= tolerance && right - left <= tolerance;
}

inline bool valid_product(const ElasticQueuedProduct& product) noexcept {
    if (!finite(product.charge_e) || !finite(product.kinetic_energy_MeV) ||
        product.kinetic_energy_MeV < 0.0F || product.atomic_number < 0 ||
        product.mass_number < 0 ||
        (product.atomic_number > 0 && product.mass_number < product.atomic_number) ||
        product.route == ElasticProductRoute::none) {
        return false;
    }
    if (!finite(product.global_direction.x) || !finite(product.global_direction.y) ||
        !finite(product.global_direction.z)) {
        return false;
    }
    const auto norm2 = product.global_direction.x * product.global_direction.x +
                       product.global_direction.y * product.global_direction.y +
                       product.global_direction.z * product.global_direction.z;
    if (!(norm2 > 1.0e-12F && finite(norm2))) return false;
    constexpr float charge_tolerance = 1.0e-3F;
    if (product.route == ElasticProductRoute::charged) {
        return product.charge_e < -charge_tolerance || product.charge_e > charge_tolerance;
    }
    return product.charge_e >= -charge_tolerance && product.charge_e <= charge_tolerance;
}

inline std::uint64_t child_stream(const ElasticQueueParentMetadata& parent,
                                  const std::uint32_t role,
                                  const std::uint32_t product_index) noexcept {
    constexpr auto interaction_role = 0x90000000U;
    const auto interaction_stream = rng::child_stream(
        parent.parent_rng_stream,
        rng::branch_tag(interaction_role,
                        static_cast<std::uint32_t>(parent.interaction_index)));
    return rng::child_stream(interaction_stream,
                             rng::branch_tag(role, product_index));
}

template <typename Counter>
inline bool reserve_route(Counter* counter,
                          const std::uint64_t capacity,
                          const std::uint64_t count,
                          std::uint64_t& offset) noexcept {
    if (count == 0) {
        offset = 0;
        return true;
    }
    if (counter == nullptr) return false;
    offset = queue_atomic_fetch_add(counter, count);
    return offset <= capacity && count <= capacity - offset;
}

inline ElasticQueueCommitStatus status_for_routes(const bool charged_attempted,
                                                  const bool neutral_attempted,
                                                  const bool charged_fit,
                                                  const bool neutral_fit) noexcept {
    if ((!charged_attempted || charged_fit) && (!neutral_attempted || neutral_fit)) {
        return ElasticQueueCommitStatus::success;
    }
    if (charged_attempted && neutral_attempted) {
        if (charged_fit != neutral_fit) return ElasticQueueCommitStatus::partial_route_commit;
        return ElasticQueueCommitStatus::both_routes_overflow;
    }
    return charged_attempted ? ElasticQueueCommitStatus::charged_route_overflow
                             : ElasticQueueCommitStatus::neutral_route_overflow;
}

}  // namespace elastic_queue_detail

// Commit descriptors produced by apply_elastic_event to the two existing
// queues.  Each route is reserved exactly once.  A failed route never writes a
// record, but the other route may already have committed; this is explicit in
// status=partial_route_commit and is not presented as a global transaction.
template <typename Counter>
inline ElasticQueueCommitResult commit_elastic_queues(
    const ElasticApplicationResult& application,
    const ElasticProductOutputSpan descriptors,
    const ElasticQueueParentMetadata& parent,
    const ElasticQueueStorage<Counter>& storage) noexcept {
    ElasticQueueCommitResult result{};
    result.input_ledger_closure_residual_MeV = application.energy_closure_residual_MeV;
    result.local_deposit_MeV = application.local_deposit_MeV;
    result.primary_continuation_energy_MeV = application.outgoing_primary_energy_MeV;
    result.scorer = ElasticScorerContribution{
        ElasticQueueOrigin::elastic,
        parent.position_x_mm,
        parent.position_y_mm,
        parent.position_z_mm,
        parent.time_ns,
        application.local_deposit_MeV,
        application.outgoing_primary_energy_MeV,
        0.0F,
        0.0F,
        0.0F,
        0.0F};

    const auto descriptor_count =
        static_cast<std::size_t>(application.queued_charged_count) +
        static_cast<std::size_t>(application.queued_neutral_count);
    if (!application.success() ||
        (descriptor_count > 0 && descriptors.data == nullptr) ||
        !elastic_queue_detail::valid_position_and_time(parent) ||
        descriptors.capacity < descriptor_count) {
        return result;
    }
    if (!elastic_queue_detail::energy_close(
            application.energy_closure_residual_MeV, 0.0F) ||
        !elastic_queue_detail::finite(application.incident_energy_MeV) ||
        application.incident_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.outgoing_primary_energy_MeV) ||
        application.outgoing_primary_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.local_deposit_MeV) ||
        application.local_deposit_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.queued_charged_energy_MeV) ||
        application.queued_charged_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.queued_neutral_energy_MeV) ||
        application.queued_neutral_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.discarded_energy_MeV) ||
        application.discarded_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.other_disposition_energy_MeV) ||
        application.other_disposition_energy_MeV < 0.0F ||
        !elastic_queue_detail::finite(application.escaped_energy_MeV) ||
        application.escaped_energy_MeV < 0.0F) {
        return result;
    }

    std::uint32_t charged_count = 0;
    std::uint32_t neutral_count = 0;
    float charged_energy = 0.0F;
    float neutral_energy = 0.0F;
    for (std::size_t index = 0; index < descriptor_count; ++index) {
        const auto& product = descriptors.data[index];
        if (!elastic_queue_detail::valid_product(product)) return result;
        if (product.route == ElasticProductRoute::charged) {
            ++charged_count;
            charged_energy += product.kinetic_energy_MeV;
        } else {
            ++neutral_count;
            neutral_energy += product.kinetic_energy_MeV;
        }
    }
    if (charged_count != application.queued_charged_count ||
        neutral_count != application.queued_neutral_count ||
        !elastic_queue_detail::energy_close(
            charged_energy, application.queued_charged_energy_MeV) ||
        !elastic_queue_detail::energy_close(
            neutral_energy, application.queued_neutral_energy_MeV) ||
        !elastic_queue_detail::finite(charged_energy) ||
        !elastic_queue_detail::finite(neutral_energy)) {
        return result;
    }

    const auto charged_attempted = charged_count > 0;
    const auto neutral_attempted = neutral_count > 0;
    if (charged_attempted &&
        (storage.charged_records == nullptr || storage.charged_filled == nullptr)) {
        return result;
    }
    if (neutral_attempted && storage.neutral_route_enabled &&
        (storage.neutral_records == nullptr || storage.neutral_filled == nullptr)) {
        return result;
    }
    if (charged_attempted && storage.charged_counter == nullptr) return result;
    if (neutral_attempted && storage.neutral_route_enabled &&
        storage.neutral_counter == nullptr) return result;

    std::uint64_t charged_offset = 0;
    std::uint64_t neutral_offset = 0;
    const auto charged_fit = elastic_queue_detail::reserve_route(
        storage.charged_counter, storage.charged_capacity, charged_count, charged_offset);
    const auto neutral_fit = storage.neutral_route_enabled &&
                             elastic_queue_detail::reserve_route(
                                 storage.neutral_counter, storage.neutral_capacity,
                                 neutral_count, neutral_offset);

    result.status = elastic_queue_detail::status_for_routes(
        charged_attempted, neutral_attempted, charged_fit, neutral_fit);

    // A route is committed only after its complete reservation fits.  The
    // other route is independent and may still commit in a mixed result.
    if (charged_fit && charged_attempted) {
        std::uint64_t output_index = charged_offset;
        for (std::size_t index = 0; index < descriptor_count; ++index) {
            const auto& product = descriptors.data[index];
            if (product.route != ElasticProductRoute::charged) continue;
            storage.charged_records[output_index++] = SecondaryParticle3D{
                parent.position_x_mm,
                parent.position_y_mm,
                parent.position_z_mm,
                product.kinetic_energy_MeV,
                product.global_direction.x,
                product.global_direction.y,
                product.global_direction.z,
                product.pdg_id,
                product.atomic_number,
                product.mass_number,
                charged_dose_category(product.atomic_number, product.mass_number),
                product.generation,
                charged_lineage,
                elastic_queue_detail::child_stream(
                    parent, elastic_queue_detail::branch_role_elastic_charged,
                    static_cast<std::uint32_t>(index))};
        }
        (void)elastic_queue_detail::queue_atomic_fetch_add(storage.charged_filled,
                                                            charged_count);
        result.written_charged_count = charged_count;
        result.written_charged_energy_MeV = charged_energy;
    } else if (charged_attempted) {
        result.dropped_charged_count = charged_count;
        result.dropped_charged_energy_MeV = charged_energy;
    }

    if (neutral_fit && neutral_attempted) {
        std::uint64_t output_index = neutral_offset;
        for (std::size_t index = 0; index < descriptor_count; ++index) {
            const auto& product = descriptors.data[index];
            if (product.route != ElasticProductRoute::neutral) continue;
            const auto lineage = neutral_lineage_from_pdg(product.pdg_id);
            storage.neutral_records[output_index++] = NeutralParticle3D{
                parent.position_x_mm,
                parent.position_y_mm,
                parent.position_z_mm,
                product.kinetic_energy_MeV,
                product.global_direction.x,
                product.global_direction.y,
                product.global_direction.z,
                product.pdg_id,
                static_cast<std::uint8_t>(neutral_origin_category_from_lineage(lineage)),
                product.generation,
                lineage,
                elastic_queue_detail::child_stream(
                    parent, elastic_queue_detail::branch_role_elastic_neutral,
                    static_cast<std::uint32_t>(index))};
        }
        (void)elastic_queue_detail::queue_atomic_fetch_add(storage.neutral_filled,
                                                            neutral_count);
        result.written_neutral_count = neutral_count;
        result.written_neutral_energy_MeV = neutral_energy;
    } else if (neutral_attempted) {
        result.dropped_neutral_count = neutral_count;
        result.dropped_neutral_energy_MeV = neutral_energy;
    }

    result.scorer.queued_charged_energy_MeV = result.written_charged_energy_MeV;
    result.scorer.queued_neutral_energy_MeV = result.written_neutral_energy_MeV;
    result.scorer.overflow_charged_energy_MeV = result.dropped_charged_energy_MeV;
    result.scorer.overflow_neutral_energy_MeV = result.dropped_neutral_energy_MeV;
    result.written_count = result.written_charged_count + result.written_neutral_count;
    result.dropped_count = result.dropped_charged_count + result.dropped_neutral_count;
    result.written_energy_MeV = result.written_charged_energy_MeV +
                                result.written_neutral_energy_MeV;
    result.dropped_energy_MeV = result.dropped_charged_energy_MeV +
                                result.dropped_neutral_energy_MeV;
    // Queue admission does not alter the event ledger: dropped products remain
    // explicit overflow energy, so closure is retained for both outcomes.
    const auto committed_total = application.outgoing_primary_energy_MeV +
                                  application.local_deposit_MeV +
                                  result.written_charged_energy_MeV +
                                  result.written_neutral_energy_MeV +
                                  result.dropped_charged_energy_MeV +
                                  result.dropped_neutral_energy_MeV +
                                  application.discarded_energy_MeV +
                                  application.other_disposition_energy_MeV +
                                  application.escaped_energy_MeV;
    result.committed_ledger_closure_residual_MeV =
        application.incident_energy_MeV - committed_total;
    return result;
}

}  // namespace carbon
