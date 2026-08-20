#pragma once

#include "carbon/elastic_package.hpp"
#include "carbon/rng.hpp"

#ifdef CARBON_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

namespace carbon {

// Flat, non-owning representation suitable for copying to a SYCL device.
// The pointed-to records keep the ELPKG v1 layout already used by the reader.
struct ElasticPackageDeviceView {
    const ElasticEnergyBin* energy_bins{nullptr};
    std::size_t energy_bin_count{0};
    const ElasticEvent* events{nullptr};
    std::size_t event_count{0};
    const ElasticProduct* products{nullptr};
    std::size_t product_count{0};
    float minimum_energy_MeV_per_u{0.0F};
    float energy_bin_width_MeV_per_u{1.0F};
};

inline ElasticPackageDeviceView make_elastic_package_device_view(
    const ElasticPackageTable& package) noexcept {
    return ElasticPackageDeviceView{package.energy_bins().data(),
                                    package.energy_bins().size(),
                                    package.events().data(),
                                    package.events().size(),
                                    package.products().data(),
                                    package.products().size(),
                                    package.minimum_energy_MeV_per_u(),
                                    package.energy_bin_width_MeV_per_u()};
}

enum class ElasticSampleStatus : std::uint8_t {
    success = 0,
    invalid_view,
    invalid_energy,
    empty_bin,
    invalid_event_range,
    invalid_product_range,
    invalid_event,
};

struct ElasticDirection3F {
    float x{0.0F};
    float y{0.0F};
    float z{1.0F};
};

struct SampledElasticEvent {
    ElasticSampleStatus status{ElasticSampleStatus::invalid_view};
    std::uint32_t event_index{0};
    float outgoing_energy_MeV_per_u{0.0F};
    ElasticDirection3F outgoing_local_direction{};
    float local_deposit_MeV{0.0F};
    std::uint32_t product_offset{0};
    std::uint32_t product_count{0};

    [[nodiscard]] bool valid() const noexcept {
        return status == ElasticSampleStatus::success;
    }
};

namespace elastic_detail {

inline bool finite(const float value) noexcept {
    // This arithmetic test is valid in both host C++ and SYCL device code and
    // avoids pulling a host-only classification routine into kernels.
    return value == value && value <= std::numeric_limits<float>::max() &&
           value >= -std::numeric_limits<float>::max();
}

inline float logarithm(const float value) noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::log(value);
#else
    return __builtin_logf(value);
#endif
}

inline float square_root(const float value) noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(value);
#else
    return __builtin_sqrtf(value);
#endif
}

inline float clamp_unit_open(const float value) noexcept {
    constexpr float epsilon = 5.960464477539063e-8F;
    if (!finite(value)) return 0.5F;
    return value <= epsilon ? epsilon : (value >= 1.0F ? 1.0F - epsilon : value);
}

inline ElasticDirection3F normalize(const ElasticDirection3F value,
                                    const ElasticDirection3F fallback) noexcept {
    const auto norm2 = value.x * value.x + value.y * value.y + value.z * value.z;
    if (!finite(norm2) || norm2 <= 1.0e-24F) return fallback;
    const auto inverse = 1.0F / square_root(norm2);
    return ElasticDirection3F{value.x * inverse, value.y * inverse, value.z * inverse};
}

inline ElasticDirection3F cross(const ElasticDirection3F a,
                                const ElasticDirection3F b) noexcept {
    return ElasticDirection3F{a.y * b.z - a.z * b.y,
                              a.z * b.x - a.x * b.z,
                              a.x * b.y - a.y * b.x};
}

}  // namespace elastic_detail

inline SampledElasticEvent sample_elastic_event(
    const ElasticPackageDeviceView view,
    const std::size_t energy_bin_index,
    const float uniform01) noexcept {
    SampledElasticEvent result{};
    if (view.energy_bins == nullptr || view.events == nullptr || view.energy_bin_count == 0 ||
        view.event_count == 0 || (view.product_count > 0 && view.products == nullptr)) {
        result.status = ElasticSampleStatus::invalid_view;
        return result;
    }
    if (energy_bin_index >= view.energy_bin_count) {
        result.status = ElasticSampleStatus::invalid_energy;
        return result;
    }
    const auto bin = view.energy_bins[energy_bin_index];
    if (bin.event_count == 0) {
        result.status = ElasticSampleStatus::empty_bin;
        return result;
    }
    if (static_cast<std::size_t>(bin.event_offset) > view.event_count ||
        static_cast<std::size_t>(bin.event_count) >
            view.event_count - static_cast<std::size_t>(bin.event_offset)) {
        result.status = ElasticSampleStatus::invalid_event_range;
        return result;
    }
    const auto bounded_u = elastic_detail::clamp_unit_open(uniform01);
    auto selected_offset = static_cast<std::uint32_t>(
        bounded_u * static_cast<float>(bin.event_count));
    if (selected_offset >= bin.event_count) selected_offset = bin.event_count - 1U;
    const auto event_index = static_cast<std::size_t>(bin.event_offset) + selected_offset;
    const auto event = view.events[event_index];
    if (static_cast<std::size_t>(event.product_offset) > view.product_count ||
        static_cast<std::size_t>(event.product_count) >
            view.product_count - static_cast<std::size_t>(event.product_offset)) {
        result.status = ElasticSampleStatus::invalid_product_range;
        return result;
    }
    const auto direction_norm2 = event.outgoing_direction_x * event.outgoing_direction_x +
                                 event.outgoing_direction_y * event.outgoing_direction_y +
                                 event.outgoing_direction_z * event.outgoing_direction_z;
    if (!elastic_detail::finite(event.outgoing_projectile_energy_MeV_per_u) ||
        event.outgoing_projectile_energy_MeV_per_u < 0.0F ||
        !elastic_detail::finite(event.outgoing_direction_x) ||
        !elastic_detail::finite(event.outgoing_direction_y) ||
        !elastic_detail::finite(event.outgoing_direction_z) ||
        !elastic_detail::finite(direction_norm2) || direction_norm2 <= 1.0e-12F ||
        !elastic_detail::finite(event.local_deposit_MeV) || event.local_deposit_MeV < 0.0F) {
        result.status = ElasticSampleStatus::invalid_event;
        return result;
    }
    result.status = ElasticSampleStatus::success;
    result.event_index = static_cast<std::uint32_t>(event_index);
    result.outgoing_energy_MeV_per_u = event.outgoing_projectile_energy_MeV_per_u;
    result.outgoing_local_direction = ElasticDirection3F{event.outgoing_direction_x,
                                                         event.outgoing_direction_y,
                                                         event.outgoing_direction_z};
    result.local_deposit_MeV = event.local_deposit_MeV;
    result.product_offset = event.product_offset;
    result.product_count = event.product_count;
    return result;
}

inline SampledElasticEvent sample_elastic_event_for_energy(
    const ElasticPackageDeviceView view,
    const float energy_MeV_per_u,
    const float uniform01) noexcept {
    if (!elastic_detail::finite(energy_MeV_per_u) || view.energy_bins == nullptr ||
        view.energy_bin_count == 0 || !elastic_detail::finite(view.minimum_energy_MeV_per_u) ||
        !elastic_detail::finite(view.energy_bin_width_MeV_per_u) ||
        view.energy_bin_width_MeV_per_u <= 0.0F) {
        return SampledElasticEvent{ElasticSampleStatus::invalid_energy};
    }
    const auto scaled = (energy_MeV_per_u - view.minimum_energy_MeV_per_u) /
                        view.energy_bin_width_MeV_per_u;
    auto index = scaled <= 0.0F ? std::size_t{0}
                                : static_cast<std::size_t>(scaled);
    if (index >= view.energy_bin_count) index = view.energy_bin_count - 1U;
    return sample_elastic_event(view, index, uniform01);
}

inline ElasticDirection3F rotate_elastic_local_direction(
    const ElasticDirection3F local_direction,
    const ElasticDirection3F parent_direction) noexcept {
    const auto w = elastic_detail::normalize(parent_direction, ElasticDirection3F{0.0F, 0.0F, 1.0F});
    const auto local = elastic_detail::normalize(local_direction, ElasticDirection3F{0.0F, 0.0F, 1.0F});
    // Pick the Cartesian axis least aligned with w. This remains well
    // conditioned for tracks parallel to z and for near-degenerate inputs.
    const auto ax = w.x < 0.0F ? -w.x : w.x;
    const auto ay = w.y < 0.0F ? -w.y : w.y;
    const auto az = w.z < 0.0F ? -w.z : w.z;
    const ElasticDirection3F reference =
        ax <= ay && ax <= az ? ElasticDirection3F{1.0F, 0.0F, 0.0F}
        : ay <= az ? ElasticDirection3F{0.0F, 1.0F, 0.0F}
                   : ElasticDirection3F{0.0F, 0.0F, 1.0F};
    const auto u = elastic_detail::normalize(elastic_detail::cross(reference, w),
                                             ElasticDirection3F{1.0F, 0.0F, 0.0F});
    const auto v = elastic_detail::normalize(elastic_detail::cross(w, u),
                                             ElasticDirection3F{0.0F, 1.0F, 0.0F});
    return elastic_detail::normalize(
        ElasticDirection3F{local.x * u.x + local.y * v.x + local.z * w.x,
                           local.x * u.y + local.y * v.y + local.z * w.y,
                           local.x * u.z + local.y * v.z + local.z * w.z},
        w);
}

enum class CompetingInteractionChannel : std::uint8_t { none = 0, elastic, inelastic };
enum class CompetingInteractionStatus : std::uint8_t {
    no_event = 0,
    elastic_event,
    inelastic_event,
    invalid_state,
    invalid_cross_section,
    invalid_step,
    invalid_random,
};

struct CompetingInteractionState {
    float remaining_optical_depth{0.0F};
    bool target_sampled{false};
};

struct CompetingInteractionRng {
    std::uint64_t seed{0};
    std::uint64_t history_id{0};
    std::uint64_t interaction_index{0};
    std::uint32_t target_dimension{0};
    std::uint32_t channel_dimension{1};
};

struct CompetingInteractionResult {
    CompetingInteractionStatus status{CompetingInteractionStatus::no_event};
    CompetingInteractionChannel channel{CompetingInteractionChannel::none};
    float distance_mm{0.0F};
    float step_fraction{0.0F};
    float optical_depth_consumed{0.0F};
    bool target_rng_consumed{false};
    bool channel_rng_consumed{false};

    [[nodiscard]] bool event() const noexcept {
        return status == CompetingInteractionStatus::elastic_event ||
               status == CompetingInteractionStatus::inelastic_event;
    }
};

inline CompetingInteractionResult advance_competing_interaction(
    CompetingInteractionState& state,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const float step_mm,
    const CompetingInteractionRng& random) noexcept {
    CompetingInteractionResult result{};
    if (!elastic_detail::finite(sigma_elastic_per_mm) ||
        !elastic_detail::finite(sigma_inelastic_per_mm) || sigma_elastic_per_mm < 0.0F ||
        sigma_inelastic_per_mm < 0.0F) {
        result.status = CompetingInteractionStatus::invalid_cross_section;
        return result;
    }
    if (!elastic_detail::finite(step_mm) || step_mm < 0.0F) {
        result.status = CompetingInteractionStatus::invalid_step;
        return result;
    }
    if (state.target_sampled &&
        (!elastic_detail::finite(state.remaining_optical_depth) ||
         state.remaining_optical_depth < 0.0F)) {
        result.status = CompetingInteractionStatus::invalid_state;
        return result;
    }
    const auto total = sigma_elastic_per_mm + sigma_inelastic_per_mm;
    if (!elastic_detail::finite(total)) {
        result.status = CompetingInteractionStatus::invalid_cross_section;
        return result;
    }
    if (total <= 0.0F || step_mm <= 0.0F) return result;
    if (!state.target_sampled) {
        const auto target_uniform = rng::uniform01(random.seed, random.history_id,
                                                   random.interaction_index,
                                                   random.target_dimension);
        state.remaining_optical_depth =
            -elastic_detail::logarithm(elastic_detail::clamp_unit_open(target_uniform));
        state.target_sampled = true;
        result.target_rng_consumed = true;
    }
    const auto available_optical_depth = total * step_mm;
    if (!elastic_detail::finite(available_optical_depth)) {
        result.status = CompetingInteractionStatus::invalid_cross_section;
        return result;
    }
    if (state.remaining_optical_depth > available_optical_depth) {
        state.remaining_optical_depth -= available_optical_depth;
        result.optical_depth_consumed = available_optical_depth;
        return result;
    }
    const auto distance = total > 0.0F ? state.remaining_optical_depth / total : 0.0F;
    result.distance_mm = distance < 0.0F ? 0.0F : (distance > step_mm ? step_mm : distance);
    result.step_fraction = step_mm > 0.0F ? result.distance_mm / step_mm : 0.0F;
    result.optical_depth_consumed = total * result.distance_mm;
    state.remaining_optical_depth = 0.0F;
    if (sigma_elastic_per_mm <= 0.0F) {
        result.channel = CompetingInteractionChannel::inelastic;
        result.status = CompetingInteractionStatus::inelastic_event;
        return result;
    }
    if (sigma_inelastic_per_mm <= 0.0F) {
        result.channel = CompetingInteractionChannel::elastic;
        result.status = CompetingInteractionStatus::elastic_event;
        return result;
    }
    const auto channel_uniform = rng::uniform01(random.seed, random.history_id,
                                                random.interaction_index,
                                                random.channel_dimension);
    result.channel_rng_consumed = true;
    if (channel_uniform < sigma_elastic_per_mm / total) {
        result.channel = CompetingInteractionChannel::elastic;
        result.status = CompetingInteractionStatus::elastic_event;
    } else {
        result.channel = CompetingInteractionChannel::inelastic;
        result.status = CompetingInteractionStatus::inelastic_event;
    }
    return result;
}

// Independent clocks for processes whose hazards must not be merged into a
// single total-hazard clock.  The draw counters are part of the state so a
// winner can be resampled without changing the losing process's RNG stream.
enum class IndependentInteractionChannel : std::uint8_t { none = 0, elastic, inelastic };
enum class IndependentInteractionStatus : std::uint8_t {
    no_event = 0,
    elastic_event,
    inelastic_event,
    invalid_state,
    invalid_cross_section,
    invalid_step,
    invalid_random,
};

struct IndependentInteractionClocks {
    float elastic_remaining_tau{0.0F};
    float inelastic_remaining_tau{0.0F};
    bool elastic_initialized{false};
    bool inelastic_initialized{false};
    std::uint64_t elastic_rng_draws{0};
    std::uint64_t inelastic_rng_draws{0};
};

// This deterministic default is device-compatible and gives each process its
// own random-dimension stream.  Applications can use the callback overloads
// below when a different process-specific RNG is required.
struct IndependentInteractionRng {
    std::uint64_t seed{0};
    std::uint64_t history_id{0};
    std::uint64_t interaction_index{0};
    std::uint32_t elastic_dimension{0};
    std::uint32_t inelastic_dimension{1};

    [[nodiscard]] float draw_elastic(const std::uint64_t draw_index) const noexcept {
        return rng::uniform01(seed, history_id, interaction_index,
                              elastic_dimension + static_cast<std::uint32_t>(draw_index));
    }

    [[nodiscard]] float draw_inelastic(const std::uint64_t draw_index) const noexcept {
        return rng::uniform01(seed, history_id, interaction_index,
                              inelastic_dimension + static_cast<std::uint32_t>(draw_index));
    }
};

struct IndependentInteractionProposal {
    IndependentInteractionStatus status{IndependentInteractionStatus::no_event};
    IndependentInteractionChannel channel{IndependentInteractionChannel::none};
    float elastic_distance_mm{std::numeric_limits<float>::infinity()};
    float inelastic_distance_mm{std::numeric_limits<float>::infinity()};
    float distance_mm{std::numeric_limits<float>::infinity()};

    [[nodiscard]] bool event() const noexcept {
        return status == IndependentInteractionStatus::elastic_event ||
               status == IndependentInteractionStatus::inelastic_event;
    }
};

struct IndependentInteractionResult {
    IndependentInteractionStatus status{IndependentInteractionStatus::no_event};
    IndependentInteractionChannel channel{IndependentInteractionChannel::none};
    float distance_mm{0.0F};
    float step_fraction{0.0F};
    bool elastic_rng_consumed{false};
    bool inelastic_rng_consumed{false};

    [[nodiscard]] bool event() const noexcept {
        return status == IndependentInteractionStatus::elastic_event ||
               status == IndependentInteractionStatus::inelastic_event;
    }
};

namespace independent_detail {

inline bool valid_sigma(const float sigma) noexcept {
    return elastic_detail::finite(sigma) && sigma >= 0.0F;
}

inline float absolute(const float value) noexcept {
    return value < 0.0F ? -value : value;
}

inline float roundoff_tolerance(const float a, const float b) noexcept {
    const auto scale_a = absolute(a);
    const auto scale_b = absolute(b);
    const auto scale = scale_a > scale_b ? scale_a : scale_b;
    return 16.0F * std::numeric_limits<float>::epsilon() * scale;
}

template <typename Draw>
inline float invoke_draw(Draw& draw, const std::uint64_t draw_index) noexcept {
    if constexpr (requires { draw(draw_index); }) {
        return static_cast<float>(draw(draw_index));
    } else {
        return static_cast<float>(draw());
    }
}

template <typename Draw>
inline IndependentInteractionStatus ensure_clock(float& remaining_tau,
                                                 bool& initialized,
                                                 std::uint64_t& draw_count,
                                                 const float sigma,
                                                 Draw draw) noexcept {
    if (!valid_sigma(sigma)) return IndependentInteractionStatus::invalid_cross_section;
    if (initialized &&
        (!elastic_detail::finite(remaining_tau) || remaining_tau < 0.0F)) {
        return IndependentInteractionStatus::invalid_state;
    }
    // A zero hazard neither creates a clock nor consumes its random stream.
    if (sigma == 0.0F || initialized) return IndependentInteractionStatus::no_event;

    const auto uniform = invoke_draw(draw, draw_count);
    if (!elastic_detail::finite(uniform) || uniform <= 0.0F || uniform >= 1.0F) {
        if (draw_count != std::numeric_limits<std::uint64_t>::max()) ++draw_count;
        return IndependentInteractionStatus::invalid_random;
    }
    const auto tau = -elastic_detail::logarithm(uniform);
    if (!elastic_detail::finite(tau) || tau < 0.0F) {
        if (draw_count != std::numeric_limits<std::uint64_t>::max()) ++draw_count;
        return IndependentInteractionStatus::invalid_random;
    }
    if (draw_count != std::numeric_limits<std::uint64_t>::max()) ++draw_count;
    remaining_tau = tau;
    initialized = true;
    return IndependentInteractionStatus::no_event;
}

inline IndependentInteractionStatus validate_clocks(
    const IndependentInteractionClocks& clocks) noexcept {
    if (clocks.elastic_initialized &&
        (!elastic_detail::finite(clocks.elastic_remaining_tau) ||
         clocks.elastic_remaining_tau < 0.0F)) {
        return IndependentInteractionStatus::invalid_state;
    }
    if (clocks.inelastic_initialized &&
        (!elastic_detail::finite(clocks.inelastic_remaining_tau) ||
         clocks.inelastic_remaining_tau < 0.0F)) {
        return IndependentInteractionStatus::invalid_state;
    }
    return IndependentInteractionStatus::no_event;
}

inline void reset_clock(float& remaining_tau, bool& initialized) noexcept {
    remaining_tau = 0.0F;
    initialized = false;
}

}  // namespace independent_detail

template <typename Draw>
inline IndependentInteractionStatus ensure_independent_interaction_clock(
    IndependentInteractionClocks& clocks,
    const IndependentInteractionChannel channel,
    const float sigma_per_mm,
    Draw draw) noexcept {
    if (channel == IndependentInteractionChannel::elastic) {
        return independent_detail::ensure_clock(clocks.elastic_remaining_tau,
                                                clocks.elastic_initialized,
                                                clocks.elastic_rng_draws, sigma_per_mm, draw);
    }
    if (channel == IndependentInteractionChannel::inelastic) {
        return independent_detail::ensure_clock(clocks.inelastic_remaining_tau,
                                                clocks.inelastic_initialized,
                                                clocks.inelastic_rng_draws, sigma_per_mm, draw);
    }
    return IndependentInteractionStatus::invalid_state;
}

template <typename ElasticDraw, typename InelasticDraw>
inline IndependentInteractionStatus ensure_independent_interaction_clocks(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    ElasticDraw elastic_draw,
    InelasticDraw inelastic_draw) noexcept {
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        return IndependentInteractionStatus::invalid_cross_section;
    }
    const auto elastic_status = ensure_independent_interaction_clock(
        clocks, IndependentInteractionChannel::elastic, sigma_elastic_per_mm, elastic_draw);
    if (elastic_status != IndependentInteractionStatus::no_event) return elastic_status;
    return ensure_independent_interaction_clock(
        clocks, IndependentInteractionChannel::inelastic, sigma_inelastic_per_mm, inelastic_draw);
}

inline IndependentInteractionStatus ensure_independent_interaction_clocks(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const IndependentInteractionRng& random) noexcept {
    const auto elastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_elastic(index);
    };
    const auto inelastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_inelastic(index);
    };
    return ensure_independent_interaction_clocks(clocks, sigma_elastic_per_mm,
                                                  sigma_inelastic_per_mm, elastic_draw,
                                                  inelastic_draw);
}

template <typename ElasticDraw, typename InelasticDraw>
inline IndependentInteractionStatus initialize_independent_interaction_clocks(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    ElasticDraw elastic_draw,
    InelasticDraw inelastic_draw) noexcept {
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        return IndependentInteractionStatus::invalid_cross_section;
    }
    clocks = IndependentInteractionClocks{};
    return ensure_independent_interaction_clocks(clocks, sigma_elastic_per_mm,
                                                  sigma_inelastic_per_mm, elastic_draw,
                                                  inelastic_draw);
}

inline IndependentInteractionStatus initialize_independent_interaction_clocks(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const IndependentInteractionRng& random) noexcept {
    const auto elastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_elastic(index);
    };
    const auto inelastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_inelastic(index);
    };
    return initialize_independent_interaction_clocks(clocks, sigma_elastic_per_mm,
                                                      sigma_inelastic_per_mm, elastic_draw,
                                                      inelastic_draw);
}

inline IndependentInteractionProposal propose_independent_interaction_step(
    const IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm) noexcept {
    IndependentInteractionProposal result{};
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        result.status = IndependentInteractionStatus::invalid_cross_section;
        return result;
    }
    const auto state_status = independent_detail::validate_clocks(clocks);
    if (state_status != IndependentInteractionStatus::no_event) {
        result.status = state_status;
        return result;
    }
    if (sigma_elastic_per_mm > 0.0F && !clocks.elastic_initialized) {
        result.status = IndependentInteractionStatus::invalid_state;
        return result;
    }
    if (sigma_inelastic_per_mm > 0.0F && !clocks.inelastic_initialized) {
        result.status = IndependentInteractionStatus::invalid_state;
        return result;
    }
    if (sigma_elastic_per_mm > 0.0F) {
        result.elastic_distance_mm = clocks.elastic_remaining_tau / sigma_elastic_per_mm;
    }
    if (sigma_inelastic_per_mm > 0.0F) {
        result.inelastic_distance_mm = clocks.inelastic_remaining_tau / sigma_inelastic_per_mm;
    }
    if (!elastic_detail::finite(result.elastic_distance_mm) &&
        !elastic_detail::finite(result.inelastic_distance_mm)) {
        return result;
    }
    // An exact tie intentionally selects inelastic for deterministic replay.
    if (result.inelastic_distance_mm <= result.elastic_distance_mm) {
        result.channel = IndependentInteractionChannel::inelastic;
        result.distance_mm = result.inelastic_distance_mm;
        result.status = IndependentInteractionStatus::inelastic_event;
    } else {
        result.channel = IndependentInteractionChannel::elastic;
        result.distance_mm = result.elastic_distance_mm;
        result.status = IndependentInteractionStatus::elastic_event;
    }
    return result;
}

inline IndependentInteractionStatus advance_independent_interaction_clocks(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const float travel_distance_mm) noexcept {
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        return IndependentInteractionStatus::invalid_cross_section;
    }
    if (!elastic_detail::finite(travel_distance_mm) || travel_distance_mm < 0.0F) {
        return IndependentInteractionStatus::invalid_step;
    }
    const auto state_status = independent_detail::validate_clocks(clocks);
    if (state_status != IndependentInteractionStatus::no_event) return state_status;
    if (sigma_elastic_per_mm > 0.0F && !clocks.elastic_initialized) {
        return IndependentInteractionStatus::invalid_state;
    }
    if (sigma_inelastic_per_mm > 0.0F && !clocks.inelastic_initialized) {
        return IndependentInteractionStatus::invalid_state;
    }

    float elastic_tau = clocks.elastic_remaining_tau;
    float inelastic_tau = clocks.inelastic_remaining_tau;
    if (sigma_elastic_per_mm > 0.0F) {
        const auto consumed = sigma_elastic_per_mm * travel_distance_mm;
        if (!elastic_detail::finite(consumed)) return IndependentInteractionStatus::invalid_step;
        elastic_tau -= consumed;
        if (elastic_tau < 0.0F) {
            if (-elastic_tau <= independent_detail::roundoff_tolerance(
                                    clocks.elastic_remaining_tau, consumed)) {
                elastic_tau = 0.0F;
            } else {
                return IndependentInteractionStatus::invalid_step;
            }
        }
    }
    if (sigma_inelastic_per_mm > 0.0F) {
        const auto consumed = sigma_inelastic_per_mm * travel_distance_mm;
        if (!elastic_detail::finite(consumed)) return IndependentInteractionStatus::invalid_step;
        inelastic_tau -= consumed;
        if (inelastic_tau < 0.0F) {
            if (-inelastic_tau <= independent_detail::roundoff_tolerance(
                                      clocks.inelastic_remaining_tau, consumed)) {
                inelastic_tau = 0.0F;
            } else {
                return IndependentInteractionStatus::invalid_step;
            }
        }
    }
    if (sigma_elastic_per_mm > 0.0F) clocks.elastic_remaining_tau = elastic_tau;
    if (sigma_inelastic_per_mm > 0.0F) clocks.inelastic_remaining_tau = inelastic_tau;
    return IndependentInteractionStatus::no_event;
}

inline IndependentInteractionStatus commit_independent_interaction_winner(
    IndependentInteractionClocks& clocks,
    const IndependentInteractionChannel winner) noexcept {
    if (winner == IndependentInteractionChannel::elastic) {
        independent_detail::reset_clock(clocks.elastic_remaining_tau,
                                         clocks.elastic_initialized);
        return IndependentInteractionStatus::no_event;
    }
    if (winner == IndependentInteractionChannel::inelastic) {
        independent_detail::reset_clock(clocks.inelastic_remaining_tau,
                                        clocks.inelastic_initialized);
        return IndependentInteractionStatus::no_event;
    }
    return IndependentInteractionStatus::invalid_state;
}

template <typename ElasticDraw, typename InelasticDraw>
inline IndependentInteractionStatus commit_independent_interaction_winner_and_resample(
    IndependentInteractionClocks& clocks,
    const IndependentInteractionChannel winner,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    ElasticDraw elastic_draw,
    InelasticDraw inelastic_draw) noexcept {
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        return IndependentInteractionStatus::invalid_cross_section;
    }
    const auto commit_status = commit_independent_interaction_winner(clocks, winner);
    if (commit_status != IndependentInteractionStatus::no_event) return commit_status;
    if (winner == IndependentInteractionChannel::elastic) {
        return ensure_independent_interaction_clock(
            clocks, winner, sigma_elastic_per_mm, elastic_draw);
    }
    if (winner == IndependentInteractionChannel::inelastic) {
        return ensure_independent_interaction_clock(
            clocks, winner, sigma_inelastic_per_mm, inelastic_draw);
    }
    return IndependentInteractionStatus::invalid_state;
}

inline IndependentInteractionStatus commit_independent_interaction_winner_and_resample(
    IndependentInteractionClocks& clocks,
    const IndependentInteractionChannel winner,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const IndependentInteractionRng& random) noexcept {
    const auto elastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_elastic(index);
    };
    const auto inelastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_inelastic(index);
    };
    return commit_independent_interaction_winner_and_resample(
        clocks, winner, sigma_elastic_per_mm, sigma_inelastic_per_mm, elastic_draw,
        inelastic_draw);
}

template <typename ElasticDraw, typename InelasticDraw>
inline IndependentInteractionResult advance_independent_interaction_step(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const float travel_distance_mm,
    ElasticDraw elastic_draw,
    InelasticDraw inelastic_draw) noexcept {
    IndependentInteractionResult result{};
    if (!independent_detail::valid_sigma(sigma_elastic_per_mm) ||
        !independent_detail::valid_sigma(sigma_inelastic_per_mm)) {
        result.status = IndependentInteractionStatus::invalid_cross_section;
        return result;
    }
    if (!elastic_detail::finite(travel_distance_mm) || travel_distance_mm < 0.0F) {
        result.status = IndependentInteractionStatus::invalid_step;
        return result;
    }
    const auto state_status = independent_detail::validate_clocks(clocks);
    if (state_status != IndependentInteractionStatus::no_event) {
        result.status = state_status;
        return result;
    }
    // A zero-length step cannot reach a collision and must not draw a clock.
    if (travel_distance_mm == 0.0F ||
        (sigma_elastic_per_mm == 0.0F && sigma_inelastic_per_mm == 0.0F)) {
        return result;
    }
    const auto before_elastic = clocks.elastic_rng_draws;
    const auto before_inelastic = clocks.inelastic_rng_draws;
    const auto ensure_status = ensure_independent_interaction_clocks(
        clocks, sigma_elastic_per_mm, sigma_inelastic_per_mm, elastic_draw, inelastic_draw);
    result.elastic_rng_consumed = clocks.elastic_rng_draws != before_elastic;
    result.inelastic_rng_consumed = clocks.inelastic_rng_draws != before_inelastic;
    if (ensure_status != IndependentInteractionStatus::no_event) {
        result.status = ensure_status;
        return result;
    }
    const auto proposal = propose_independent_interaction_step(
        clocks, sigma_elastic_per_mm, sigma_inelastic_per_mm);
    if (!proposal.event()) {
        result.status = proposal.status;
        return result;
    }
    if (proposal.distance_mm > travel_distance_mm) {
        result.status = advance_independent_interaction_clocks(
            clocks, sigma_elastic_per_mm, sigma_inelastic_per_mm, travel_distance_mm);
        return result;
    }
    result.distance_mm = proposal.distance_mm;
    result.step_fraction = travel_distance_mm > 0.0F
                               ? proposal.distance_mm / travel_distance_mm
                               : 0.0F;
    const auto advance_status = advance_independent_interaction_clocks(
        clocks, sigma_elastic_per_mm, sigma_inelastic_per_mm, proposal.distance_mm);
    if (advance_status != IndependentInteractionStatus::no_event) {
        result.status = advance_status;
        return result;
    }
    result.channel = proposal.channel;
    result.status = proposal.status;
    const auto resample_status = commit_independent_interaction_winner_and_resample(
        clocks, proposal.channel, sigma_elastic_per_mm, sigma_inelastic_per_mm, elastic_draw,
        inelastic_draw);
    result.elastic_rng_consumed = clocks.elastic_rng_draws != before_elastic;
    result.inelastic_rng_consumed = clocks.inelastic_rng_draws != before_inelastic;
    if (resample_status != IndependentInteractionStatus::no_event) result.status = resample_status;
    return result;
}

inline IndependentInteractionResult advance_independent_interaction_step(
    IndependentInteractionClocks& clocks,
    const float sigma_elastic_per_mm,
    const float sigma_inelastic_per_mm,
    const float travel_distance_mm,
    const IndependentInteractionRng& random) noexcept {
    const auto elastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_elastic(index);
    };
    const auto inelastic_draw = [random](const std::uint64_t index) noexcept {
        return random.draw_inelastic(index);
    };
    return advance_independent_interaction_step(
        clocks, sigma_elastic_per_mm, sigma_inelastic_per_mm, travel_distance_mm, elastic_draw,
        inelastic_draw);
}

}  // namespace carbon
