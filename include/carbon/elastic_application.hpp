#pragma once

#include "carbon/elastic_sampling.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace carbon {

// ELPKG v1 disposition codes.  Keep these values in sync with the compiler
// and reader; application code should use the named constants below.
inline constexpr std::int32_t elastic_disposition_transport = 1;
inline constexpr std::int32_t elastic_disposition_queue = 2;
inline constexpr std::int32_t elastic_disposition_local_deposit = 3;
inline constexpr std::int32_t elastic_disposition_discard = 4;
inline constexpr std::int32_t elastic_disposition_primary_continuation = 5;
inline constexpr std::int32_t elastic_disposition_recoil = 6;

enum class ElasticApplicationStatus : std::uint8_t {
    success = 0,
    invalid_view,
    invalid_primary,
    invalid_event,
    invalid_product_range,
    invalid_energy,
    invalid_direction,
    invalid_generation,
    invalid_disposition,
    invalid_charge_identity,
    invalid_continuation,
    energy_imbalance,
    capacity_exceeded,
};

enum class ElasticProductRoute : std::uint8_t {
    none = 0,
    charged = 1,
    neutral = 2,
};

// This is deliberately independent of SecondaryParticle3D.  The later
// transport adapter supplies position, lineage and RNG stream information.
struct ElasticQueuedProduct {
    std::int32_t pdg_id{0};
    std::int16_t atomic_number{0};
    std::int16_t mass_number{0};
    float charge_e{0.0F};
    float kinetic_energy_MeV{0.0F};
    ElasticDirection3F global_direction{};
    std::uint8_t generation{0};
    ElasticProductRoute route{ElasticProductRoute::none};
};

struct ElasticProductOutputSpan {
    ElasticQueuedProduct* data{nullptr};
    std::size_t capacity{0};
};

// The incoming fields are supplied by the caller.  The outgoing fields are
// committed only after every event/product and ledger check succeeds.
struct ElasticPrimaryState {
    float incoming_energy_MeV{0.0F};
    ElasticDirection3F incoming_direction{};
    float outgoing_energy_MeV{0.0F};
    ElasticDirection3F outgoing_direction{};
    bool active{false};
};

struct ElasticApplicationResult {
    ElasticApplicationStatus status{ElasticApplicationStatus::invalid_view};
    std::uint32_t queued_charged_count{0};
    std::uint32_t queued_neutral_count{0};
    float incident_energy_MeV{0.0F};
    float outgoing_primary_energy_MeV{0.0F};
    float local_deposit_MeV{0.0F};
    float queued_charged_energy_MeV{0.0F};
    float queued_neutral_energy_MeV{0.0F};
    // ELPKG v1 calls disposition 4 "discard".  It is kept as an explicit
    // ledger term; no product energy is silently dropped.
    float discarded_energy_MeV{0.0F};
    // Disposition 5 is not a second primary in this adapter.  Its energy is
    // explicit here until a future format defines a multi-primary contract.
    float other_disposition_energy_MeV{0.0F};
    // ELPKG v1 has no escape disposition.  This field makes that absence
    // explicit for a later adapter and remains zero for v1 events.
    float escaped_energy_MeV{0.0F};
    float energy_closure_residual_MeV{0.0F};

    [[nodiscard]] bool success() const noexcept {
        return status == ElasticApplicationStatus::success;
    }
};

static_assert(std::is_trivially_copyable_v<ElasticQueuedProduct>);
static_assert(std::is_standard_layout_v<ElasticQueuedProduct>);
static_assert(std::is_trivially_copyable_v<ElasticProductOutputSpan>);
static_assert(std::is_trivially_copyable_v<ElasticPrimaryState>);
static_assert(std::is_trivially_copyable_v<ElasticApplicationResult>);

namespace elastic_application_detail {

inline constexpr float charge_tolerance = 1.0e-3F;
inline constexpr float energy_absolute_tolerance_MeV = 1.0e-2F;
inline constexpr float energy_relative_tolerance = 2.0e-3F;

inline bool known_neutral_pdg(const std::int32_t pdg_id) noexcept {
    const auto pdg = pdg_id < 0 ? -static_cast<std::int64_t>(pdg_id)
                                : static_cast<std::int64_t>(pdg_id);
    switch (pdg) {
    case 12:   // electron neutrino
    case 14:   // muon neutrino
    case 16:   // tau neutrino
    case 18:   // tau-prime neutrino
    case 22:   // photon
    case 111:  // pi0
    case 130:  // K0L
    case 2112: // neutron
    case 311:  // K0
    case 310:  // K0S
    case 3122: // Lambda
        return true;
    default:
        return false;
    }
}

inline bool valid_direction(const ElasticDirection3F direction) noexcept {
    if (!elastic_detail::finite(direction.x) || !elastic_detail::finite(direction.y) ||
        !elastic_detail::finite(direction.z)) return false;
    const auto norm2 = direction.x * direction.x + direction.y * direction.y +
                       direction.z * direction.z;
    return elastic_detail::finite(norm2) && norm2 > 1.0e-12F;
}

inline bool valid_generation(const std::int32_t generation) noexcept {
    return generation >= 0 && generation <= 255;
}

inline bool valid_projectile(const ElasticEvent& event) noexcept {
    return event.projectile_atomic_number > 0 &&
           event.projectile_mass_number >= event.projectile_atomic_number;
}

inline bool valid_product_identity(const ElasticProduct& product) noexcept {
    if (product.atomic_number < 0 || product.mass_number < 0 ||
        (product.atomic_number == 0 && product.mass_number != 0) ||
        (product.mass_number > 0 && product.atomic_number > product.mass_number)) {
        return false;
    }
    if (!elastic_detail::finite(product.charge_e)) return false;
    if (known_neutral_pdg(product.pdg_id) &&
        (product.charge_e < -charge_tolerance || product.charge_e > charge_tolerance)) {
        return false;
    }
    if (product.atomic_number == 0 && product.mass_number == 0 &&
        (product.charge_e < -charge_tolerance || product.charge_e > charge_tolerance)) {
        return false;
    }
    return true;
}

inline ElasticProductRoute route_for_product(const ElasticProduct& product) noexcept {
    return product.charge_e < -charge_tolerance || product.charge_e > charge_tolerance
               ? ElasticProductRoute::charged
               : ElasticProductRoute::neutral;
}

inline bool is_transport_disposition(const std::int32_t disposition) noexcept {
    return disposition == elastic_disposition_transport ||
           disposition == elastic_disposition_queue ||
           disposition == elastic_disposition_recoil;
}

inline bool known_disposition(const std::int32_t disposition) noexcept {
    return disposition >= elastic_disposition_transport &&
           disposition <= elastic_disposition_recoil;
}

inline float energy_tolerance(const float incident_energy_MeV) noexcept {
    const auto scale = incident_energy_MeV > 1.0F ? incident_energy_MeV : 1.0F;
    const auto relative = energy_relative_tolerance * scale;
    return relative > energy_absolute_tolerance_MeV ? relative
                                                     : energy_absolute_tolerance_MeV;
}

}  // namespace elastic_application_detail

// Apply one sampled ELPKG event without touching any transport queue or
// scorer.  Product validation and the complete energy/capacity preflight run
// before either the primary state or output descriptors are modified.
inline ElasticApplicationResult apply_elastic_event(
    const ElasticPackageDeviceView view,
    const SampledElasticEvent sampled,
    const ElasticPrimaryState& current,
    const ElasticProductOutputSpan output,
    ElasticPrimaryState& next) noexcept {
    ElasticApplicationResult result{};
    result.status = ElasticApplicationStatus::invalid_view;
    if (view.events == nullptr || view.event_count == 0 ||
        (view.product_count > 0 && view.products == nullptr)) {
        return result;
    }
    if (!elastic_detail::finite(current.incoming_energy_MeV) ||
        current.incoming_energy_MeV <= 0.0F ||
        !elastic_application_detail::valid_direction(current.incoming_direction)) {
        result.status = ElasticApplicationStatus::invalid_primary;
        return result;
    }
    if (!sampled.valid() || sampled.event_index >= view.event_count) {
        result.status = ElasticApplicationStatus::invalid_event;
        return result;
    }
    const auto event = view.events[sampled.event_index];
    if (!elastic_application_detail::valid_projectile(event) || event.continuation != 1 ||
        !elastic_application_detail::valid_generation(event.generation)) {
        result.status = event.continuation == 1 ? ElasticApplicationStatus::invalid_event
                                                : ElasticApplicationStatus::invalid_continuation;
        return result;
    }
    if (!elastic_detail::finite(event.incident_energy_MeV_per_u) ||
        !elastic_detail::finite(event.outgoing_projectile_energy_MeV_per_u) ||
        !elastic_detail::finite(event.local_deposit_MeV) ||
        event.incident_energy_MeV_per_u <= 0.0F ||
        event.outgoing_projectile_energy_MeV_per_u < 0.0F || event.local_deposit_MeV < 0.0F ||
        !elastic_application_detail::valid_direction(
            ElasticDirection3F{event.outgoing_direction_x, event.outgoing_direction_y,
                               event.outgoing_direction_z})) {
        result.status = ElasticApplicationStatus::invalid_energy;
        return result;
    }
    if (static_cast<std::size_t>(event.product_offset) > view.product_count ||
        static_cast<std::size_t>(event.product_count) >
            view.product_count - static_cast<std::size_t>(event.product_offset)) {
        result.status = ElasticApplicationStatus::invalid_product_range;
        return result;
    }

    const auto projectile_mass = static_cast<float>(event.projectile_mass_number);
    const auto sampled_incident_energy_MeV = event.incident_energy_MeV_per_u * projectile_mass;
    if (!elastic_detail::finite(sampled_incident_energy_MeV) ||
        sampled_incident_energy_MeV <= 0.0F) {
        result.status = ElasticApplicationStatus::invalid_energy;
        return result;
    }
    // ELPKG events are sampled from a finite 1 MeV/u bin.  Their incident
    // energy therefore need not equal the current track energy.  Scale the
    // complete final-state energy ledger by the ratio of total energies while
    // preserving all event topology, identities, directions and dispositions.
    const auto energy_scale = current.incoming_energy_MeV / sampled_incident_energy_MeV;
    if (!elastic_detail::finite(energy_scale) || energy_scale <= 0.0F) {
        result.status = ElasticApplicationStatus::invalid_energy;
        return result;
    }
    result.incident_energy_MeV = current.incoming_energy_MeV;
    const auto sampled_outgoing_energy_MeV =
        event.outgoing_projectile_energy_MeV_per_u * projectile_mass;
    if (!elastic_detail::finite(sampled_outgoing_energy_MeV) ||
        sampled_outgoing_energy_MeV < 0.0F) {
        result.status = ElasticApplicationStatus::invalid_energy;
        return result;
    }
    result.outgoing_primary_energy_MeV = sampled_outgoing_energy_MeV * energy_scale;
    result.local_deposit_MeV = event.local_deposit_MeV * energy_scale;
    if (!elastic_detail::finite(result.outgoing_primary_energy_MeV) ||
        result.outgoing_primary_energy_MeV < 0.0F ||
        !elastic_detail::finite(result.local_deposit_MeV) ||
        result.local_deposit_MeV < 0.0F) {
        result.status = ElasticApplicationStatus::invalid_energy;
        return result;
    }

    const auto begin = static_cast<std::size_t>(event.product_offset);
    const auto end = begin + static_cast<std::size_t>(event.product_count);
    std::size_t output_count = 0;
    for (std::size_t index = begin; index < end; ++index) {
        const auto product = view.products[index];
        if (!elastic_application_detail::valid_product_identity(product)) {
            result.status = ElasticApplicationStatus::invalid_charge_identity;
            return result;
        }
        if (!elastic_detail::finite(product.kinetic_energy_MeV) ||
            product.kinetic_energy_MeV < 0.0F) {
            result.status = ElasticApplicationStatus::invalid_energy;
            return result;
        }
        const auto scaled_product_energy = product.kinetic_energy_MeV * energy_scale;
        if (!elastic_detail::finite(scaled_product_energy) || scaled_product_energy < 0.0F) {
            result.status = ElasticApplicationStatus::invalid_energy;
            return result;
        }
        if (!elastic_application_detail::valid_direction(
                ElasticDirection3F{product.direction_x, product.direction_y,
                                   product.direction_z})) {
            result.status = ElasticApplicationStatus::invalid_direction;
            return result;
        }
        if (!elastic_application_detail::valid_generation(product.generation)) {
            result.status = ElasticApplicationStatus::invalid_generation;
            return result;
        }
        if (!elastic_application_detail::known_disposition(product.transport_disposition)) {
            result.status = ElasticApplicationStatus::invalid_disposition;
            return result;
        }
        if (product.transport_disposition == elastic_disposition_local_deposit) {
            result.local_deposit_MeV += scaled_product_energy;
        } else if (product.transport_disposition == elastic_disposition_discard) {
            result.discarded_energy_MeV += scaled_product_energy;
        } else if (product.transport_disposition == elastic_disposition_primary_continuation) {
            result.other_disposition_energy_MeV += scaled_product_energy;
        } else if (elastic_application_detail::is_transport_disposition(
                       product.transport_disposition)) {
            ++output_count;
            const auto route = elastic_application_detail::route_for_product(product);
            if (route == ElasticProductRoute::charged) {
                ++result.queued_charged_count;
                result.queued_charged_energy_MeV += scaled_product_energy;
            } else {
                ++result.queued_neutral_count;
                result.queued_neutral_energy_MeV += scaled_product_energy;
            }
        }
    }
    const auto ledger_total = result.outgoing_primary_energy_MeV + result.local_deposit_MeV +
                              result.queued_charged_energy_MeV +
                              result.queued_neutral_energy_MeV + result.discarded_energy_MeV +
                              result.other_disposition_energy_MeV + result.escaped_energy_MeV;
    result.energy_closure_residual_MeV = result.incident_energy_MeV - ledger_total;
    if (!elastic_detail::finite(ledger_total) ||
        !elastic_detail::finite(result.energy_closure_residual_MeV) ||
        (result.energy_closure_residual_MeV >
             elastic_application_detail::energy_tolerance(result.incident_energy_MeV) ||
         -result.energy_closure_residual_MeV >
             elastic_application_detail::energy_tolerance(result.incident_energy_MeV))) {
        result.status = ElasticApplicationStatus::energy_imbalance;
        return result;
    }
    if ((output.capacity > 0 && output.data == nullptr) || output_count > output.capacity) {
        result.status = ElasticApplicationStatus::capacity_exceeded;
        return result;
    }

    // Commit is intentionally the final phase.  Rotation is deterministic and
    // cannot fail after direction preflight has succeeded.
    std::size_t output_index = 0;
    for (std::size_t index = begin; index < end; ++index) {
        const auto product = view.products[index];
        if (!elastic_application_detail::is_transport_disposition(
                product.transport_disposition)) {
            continue;
        }
        const auto route = elastic_application_detail::route_for_product(product);
        output.data[output_index++] = ElasticQueuedProduct{
            product.pdg_id,
            product.atomic_number,
            product.mass_number,
            product.charge_e,
            product.kinetic_energy_MeV * energy_scale,
            rotate_elastic_local_direction(
                ElasticDirection3F{product.direction_x, product.direction_y, product.direction_z},
                current.incoming_direction),
            static_cast<std::uint8_t>(product.generation),
            route};
    }
    next = current;
    next.outgoing_energy_MeV = result.outgoing_primary_energy_MeV;
    next.outgoing_direction = rotate_elastic_local_direction(
        ElasticDirection3F{event.outgoing_direction_x, event.outgoing_direction_y,
                           event.outgoing_direction_z},
        current.incoming_direction);
    next.active = true;
    result.status = ElasticApplicationStatus::success;
    return result;
}

}  // namespace carbon
