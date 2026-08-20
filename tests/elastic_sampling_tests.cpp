#include "carbon/elastic_sampling.hpp"

#ifdef CARBON_HAS_SYCL
#include "carbon/device.hpp"
#include <sycl/sycl.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(const float actual, const float expected, const float tolerance,
                 const char* message) {
    if (std::abs(actual - expected) > tolerance) throw std::runtime_error(message);
}

carbon::ElasticPackageDeviceView fixture_view(std::vector<carbon::ElasticEnergyBin>& bins,
                                              std::vector<carbon::ElasticEvent>& events,
                                              std::vector<carbon::ElasticProduct>& products) {
    bins = {{0.0F, 1.0F, 0U, 2U}, {1.0F, 2.0F, 2U, 1U}};
    events = {
        {1, 1, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0U, 0U, 1, 0},
        {1, 1, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0U, 0U, 1, 0},
        {1, 1, 1.0F, 0.5F, 0.0F, 1.0F, 0.0F, 0.25F, 0U, 1U, 1, 0},
    };
    products = {{2212, 1, 1, 0.25F, 0.0F, 0.0F, 1.0F, 1.0F, 0, 1}};
    return carbon::ElasticPackageDeviceView{bins.data(), bins.size(), events.data(), events.size(),
                                            products.data(), products.size(), 0.0F, 1.0F};
}

void test_uniform_event_sampling() {
    std::vector<carbon::ElasticEnergyBin> bins;
    std::vector<carbon::ElasticEvent> events;
    std::vector<carbon::ElasticProduct> products;
    const auto view = fixture_view(bins, events, products);
    const auto first = carbon::sample_elastic_event(view, 0, 0.0F);
    const auto second = carbon::sample_elastic_event(view, 0, 0.999999F);
    require(first.valid() && second.valid(), "valid ELPKG events were not sampled");
    require(first.event_index == 0U && second.event_index == 1U,
            "ELPKG event selection was not uniform over the bin");
    require(first.product_count == 0U, "unexpected first event products");
    const auto third = carbon::sample_elastic_event_for_energy(view, 1.5F, 0.2F);
    require(third.valid() && third.event_index == 2U && third.product_count == 1U,
            "energy-bin event lookup failed");
    const auto bad_bin = carbon::sample_elastic_event(view, 2, 0.5F);
    require(bad_bin.status == carbon::ElasticSampleStatus::invalid_energy,
            "invalid ELPKG bin did not return status");
    auto empty_bins = bins;
    empty_bins[1].event_count = 0U;
    const carbon::ElasticPackageDeviceView empty_view{empty_bins.data(), empty_bins.size(),
                                                      events.data(), events.size(), products.data(),
                                                      products.size(), 0.0F, 1.0F};
    const auto empty = carbon::sample_elastic_event(empty_view, 1, 0.5F);
    require(empty.status == carbon::ElasticSampleStatus::empty_bin,
            "empty ELPKG bin did not return status");
    auto bad_events = events;
    bad_events[0].product_offset = 99U;
    const carbon::ElasticPackageDeviceView bad_product_view{bins.data(), bins.size(),
                                                             bad_events.data(), bad_events.size(),
                                                             products.data(), products.size(),
                                                             0.0F, 1.0F};
    const auto bad_product = carbon::sample_elastic_event(bad_product_view, 0, 0.1F);
    require(bad_product.status == carbon::ElasticSampleStatus::invalid_product_range,
            "invalid ELPKG product range did not return status");
    const auto bad_view = carbon::sample_elastic_event(
        carbon::ElasticPackageDeviceView{}, 0, 0.5F);
    require(bad_view.status == carbon::ElasticSampleStatus::invalid_view,
            "invalid ELPKG view did not return status");
}

void test_competing_interaction() {
    const carbon::CompetingInteractionRng random{1234U, 7U, 2U, 4U, 5U};
    carbon::CompetingInteractionState state{};
    constexpr float elastic_sigma = 0.25F;
    constexpr float inelastic_sigma = 0.75F;
    const auto target_uniform = carbon::rng::uniform01(
        random.seed, random.history_id, random.interaction_index, random.target_dimension);
    const auto expected_target = -std::log(target_uniform);
    const auto first_step = expected_target * 0.5F;
    const auto first = carbon::advance_competing_interaction(
        state, elastic_sigma, inelastic_sigma, first_step, random);
    require(!first.event() && first.target_rng_consumed && !first.channel_rng_consumed,
            "competing sampler consumed the wrong RNGs before an event");
    require_near(state.remaining_optical_depth, expected_target - first_step, 2.0e-6F,
                 "competing sampler did not consume total optical depth");
    const auto second = carbon::advance_competing_interaction(
        state, elastic_sigma, inelastic_sigma, expected_target, random);
    require(second.event(), "competing sampler missed a boundary event");
    require(second.channel_rng_consumed, "mixed channels did not consume channel RNG");
    require_near(second.distance_mm, expected_target - first_step, 2.0e-6F,
                 "competing event distance is not total-hazard distance");
    require_near(second.step_fraction, 0.5F, 2.0e-6F,
                 "competing event fraction is incorrect");
    carbon::CompetingInteractionState boundary{1.0F, true};
    const auto boundary_event = carbon::advance_competing_interaction(
        boundary, 1.0F, 0.0F, 1.0F, random);
    require(boundary_event.event() && boundary_event.channel ==
                carbon::CompetingInteractionChannel::elastic,
            "exact optical-depth boundary did not produce an event");
    require_near(boundary_event.distance_mm, 1.0F, 1.0e-6F,
                 "exact optical-depth boundary distance is incorrect");
    carbon::CompetingInteractionState tiny{};
    const auto tiny_result = carbon::advance_competing_interaction(
        tiny, 1.0e-30F, 0.0F, 1.0F, random);
    require(!tiny_result.event(), "tiny cross section caused a spurious event");

    // A zero channel is deterministic and cannot consume the channel stream.
    carbon::CompetingInteractionState no_elastic{};
    const auto inelastic = carbon::advance_competing_interaction(
        no_elastic, 0.0F, 1.0F, 2.0F, random);
    require(inelastic.status == carbon::CompetingInteractionStatus::inelastic_event &&
                !inelastic.channel_rng_consumed,
            "zero elastic channel did not preserve inelastic RNG parity");
    carbon::CompetingInteractionState no_inelastic{};
    const auto elastic = carbon::advance_competing_interaction(
        no_inelastic, 1.0F, 0.0F, 2.0F, random);
    require(elastic.status == carbon::CompetingInteractionStatus::elastic_event &&
                !elastic.channel_rng_consumed,
            "zero inelastic channel did not preserve elastic RNG parity");
    carbon::CompetingInteractionState no_channels{};
    const auto none = carbon::advance_competing_interaction(
        no_channels, 0.0F, 0.0F, 2.0F, random);
    require(none.status == carbon::CompetingInteractionStatus::no_event &&
                !none.target_rng_consumed && !none.channel_rng_consumed &&
                !no_channels.target_sampled,
            "zero hazards consumed RNG or changed state");
    const auto zero_step = carbon::advance_competing_interaction(
        no_channels, 1.0F, 1.0F, 0.0F, random);
    require(zero_step.status == carbon::CompetingInteractionStatus::no_event &&
                !zero_step.target_rng_consumed,
            "zero step consumed target RNG");

    carbon::CompetingInteractionState invalid{};
    const auto invalid_xs = carbon::advance_competing_interaction(
        invalid, -1.0F, 1.0F, 1.0F, random);
    require(invalid_xs.status == carbon::CompetingInteractionStatus::invalid_cross_section,
            "negative cross section was not rejected");

    std::uint32_t elastic_count = 0;
    std::uint32_t inelastic_count = 0;
    for (std::uint64_t history = 0; history < 20000U; ++history) {
        carbon::CompetingInteractionState sample_state{};
        const carbon::CompetingInteractionRng sample_rng{91U, history, 0U, 0U, 1U};
        const auto sample = carbon::advance_competing_interaction(
            sample_state, 0.25F, 0.75F, 100.0F, sample_rng);
        require(sample.event(), "large-hazard branching sample missed an event");
        if (sample.channel == carbon::CompetingInteractionChannel::elastic) {
            ++elastic_count;
        } else if (sample.channel == carbon::CompetingInteractionChannel::inelastic) {
            ++inelastic_count;
        }
    }
    const auto elastic_frequency = static_cast<float>(elastic_count) /
                                   static_cast<float>(elastic_count + inelastic_count);
    require(std::abs(elastic_frequency - 0.25F) < 0.015F,
            "competing channel frequency does not match cross-section ratio");
}

void test_independent_interaction_clocks() {
    std::uint32_t elastic_draws = 0;
    std::uint32_t inelastic_draws = 0;
    const auto elastic_uniform = [&](const std::uint64_t index) {
        ++elastic_draws;
        return index == 0U ? 0.5F : 0.75F;
    };
    const auto inelastic_uniform = [&](const std::uint64_t index) {
        ++inelastic_draws;
        return index == 0U ? 0.25F : 0.5F;
    };

    carbon::IndependentInteractionClocks clocks{};
    const auto initialized = carbon::initialize_independent_interaction_clocks(
        clocks, 2.0F, 1.0F, elastic_uniform, inelastic_uniform);
    require(initialized == carbon::IndependentInteractionStatus::no_event &&
                clocks.elastic_initialized && clocks.inelastic_initialized &&
                clocks.elastic_rng_draws == 1U && clocks.inelastic_rng_draws == 1U &&
                elastic_draws == 1U && inelastic_draws == 1U,
            "independent clock initialization did not draw per process");
    const auto initial_elastic_tau = clocks.elastic_remaining_tau;
    const auto initial_inelastic_tau = clocks.inelastic_remaining_tau;
    require(carbon::advance_independent_interaction_clocks(
                clocks, 2.0F, 1.0F, 0.125F) ==
                carbon::IndependentInteractionStatus::no_event,
            "independent clock advance failed");
    require_near(clocks.elastic_remaining_tau, initial_elastic_tau - 0.25F, 2.0e-6F,
                 "elastic clock did not decrement by its own hazard");
    require_near(clocks.inelastic_remaining_tau, initial_inelastic_tau - 0.125F, 2.0e-6F,
                 "inelastic clock did not decrement by its own hazard");

    // Elastic wins while the inelastic clock is advanced and retained.
    carbon::IndependentInteractionClocks elastic_winner{1.0F, 2.0F, true, true, 1U, 1U};
    const auto elastic_loser_tau = elastic_winner.inelastic_remaining_tau;
    const auto elastic_proposal = carbon::propose_independent_interaction_step(
        elastic_winner, 2.0F, 1.0F);
    require(elastic_proposal.channel == carbon::IndependentInteractionChannel::elastic &&
                elastic_proposal.distance_mm == 0.5F,
            "elastic independent proposal did not select the shorter clock");
    require(carbon::advance_independent_interaction_clocks(
                elastic_winner, 2.0F, 1.0F, elastic_proposal.distance_mm) ==
                carbon::IndependentInteractionStatus::no_event,
            "elastic winner advance failed");
    const auto commit_status = carbon::commit_independent_interaction_winner_and_resample(
        elastic_winner, elastic_proposal.channel, 2.0F, 1.0F, elastic_uniform,
        inelastic_uniform);
    require(commit_status == carbon::IndependentInteractionStatus::no_event &&
                elastic_winner.elastic_initialized && elastic_winner.elastic_rng_draws == 2U &&
                elastic_winner.inelastic_initialized && elastic_winner.inelastic_rng_draws == 1U &&
                elastic_draws == 2U && inelastic_draws == 1U,
            "winner-only resample consumed the wrong process RNG");
    require_near(elastic_winner.inelastic_remaining_tau, elastic_loser_tau - 0.5F, 2.0e-6F,
                 "loser clock was not preserved after elastic event");

    // Inelastic wins ties deterministically, and only its clock is reset.
    carbon::IndependentInteractionClocks tie{1.0F, 1.0F, true, true, 3U, 4U};
    const auto tie_proposal = carbon::propose_independent_interaction_step(tie, 1.0F, 1.0F);
    require(tie_proposal.channel == carbon::IndependentInteractionChannel::inelastic &&
                tie_proposal.distance_mm == 1.0F,
            "independent exact tie did not select inelastic deterministically");
    const auto tie_loser_tau = tie.elastic_remaining_tau;
    require(carbon::commit_independent_interaction_winner(tie,
                                                           tie_proposal.channel) ==
                carbon::IndependentInteractionStatus::no_event &&
                !tie.inelastic_initialized && tie.inelastic_remaining_tau == 0.0F &&
                tie.elastic_initialized && tie.elastic_remaining_tau == tie_loser_tau,
            "inelastic winner reset the wrong clock");

    // A zero region leaves an existing clock alone and does not draw a new one.
    carbon::IndependentInteractionClocks local_region{};
    std::uint32_t local_elastic_draws = 0;
    std::uint32_t local_inelastic_draws = 0;
    const auto local_elastic_uniform = [&](const std::uint64_t) {
        ++local_elastic_draws;
        return 0.5F;
    };
    const auto local_inelastic_uniform = [&](const std::uint64_t) {
        ++local_inelastic_draws;
        return 0.5F;
    };
    require(carbon::initialize_independent_interaction_clocks(
                local_region, 0.0F, 1.0F, local_elastic_uniform, local_inelastic_uniform) ==
                carbon::IndependentInteractionStatus::no_event &&
                !local_region.elastic_initialized && local_region.inelastic_initialized,
            "zero local cross section initialized a new clock");
    const auto preserved_tau = local_region.inelastic_remaining_tau;
    require(local_inelastic_draws == 1U && local_elastic_draws == 0U,
            "zero local cross section consumed RNG");
    require(carbon::advance_independent_interaction_clocks(
                local_region, 0.0F, 0.0F, 0.25F) ==
                carbon::IndependentInteractionStatus::no_event &&
                local_region.inelastic_remaining_tau == preserved_tau,
            "zero local cross section changed an existing clock");
    require(carbon::ensure_independent_interaction_clocks(
                local_region, 1.0F, 1.0F, local_elastic_uniform, local_inelastic_uniform) ==
                carbon::IndependentInteractionStatus::no_event &&
                local_region.elastic_initialized && local_region.inelastic_rng_draws == 1U &&
                local_elastic_draws == 1U && local_inelastic_draws == 1U,
            "zero-to-positive cross section did not preserve/init clocks correctly");

    carbon::IndependentInteractionClocks no_channels{};
    require(carbon::initialize_independent_interaction_clocks(
                no_channels, 0.0F, 0.0F, local_elastic_uniform, local_inelastic_uniform) ==
                carbon::IndependentInteractionStatus::no_event &&
                !no_channels.elastic_initialized && !no_channels.inelastic_initialized &&
                no_channels.elastic_rng_draws == 0U && no_channels.inelastic_rng_draws == 0U,
            "both zero channels initialized or consumed RNG");
    require(carbon::advance_independent_interaction_clocks(
                no_channels, 0.0F, 0.0F, 10.0F) ==
                carbon::IndependentInteractionStatus::no_event,
            "both zero channels did not remain a no-event step");
    carbon::IndependentInteractionClocks invalid{};
    require(carbon::propose_independent_interaction_step(invalid, -1.0F, 1.0F).status ==
                carbon::IndependentInteractionStatus::invalid_cross_section,
            "negative independent cross section was not rejected");
    require(carbon::advance_independent_interaction_clocks(
                invalid, 1.0F, 1.0F, std::numeric_limits<float>::infinity()) ==
                carbon::IndependentInteractionStatus::invalid_step,
            "nonfinite independent step was not rejected");

    // The default RNG gives reproducible exponential distances and the expected
    // independent branch probability for a large enough transport step.
    float first_distance_sum = 0.0F;
    std::uint32_t elastic_count = 0;
    std::uint32_t inelastic_count = 0;
    for (std::uint64_t history = 0; history < 10000U; ++history) {
        const carbon::IndependentInteractionRng random{91U, history, 0U, 0U, 1U};
        carbon::IndependentInteractionClocks sample{};
        const auto result = carbon::advance_independent_interaction_step(
            sample, 0.25F, 0.75F, 100.0F, random);
        require(result.event(), "independent sampler missed a large-step event");
        first_distance_sum += result.distance_mm;
        if (result.channel == carbon::IndependentInteractionChannel::elastic) {
            ++elastic_count;
        } else if (result.channel == carbon::IndependentInteractionChannel::inelastic) {
            ++inelastic_count;
        }
    }
    const auto branch_total = static_cast<float>(elastic_count + inelastic_count);
    require(std::abs(static_cast<float>(elastic_count) / branch_total - 0.25F) < 0.02F,
            "independent branch frequency does not match cross-section ratio");
    require(std::abs(first_distance_sum / branch_total - 1.0F) < 0.05F,
            "independent first distance does not match exponential hazard");

    const carbon::IndependentInteractionRng reproducible_rng{123U, 456U, 7U, 2U, 11U};
    carbon::IndependentInteractionClocks first_state{};
    carbon::IndependentInteractionClocks second_state{};
    const auto first_result = carbon::advance_independent_interaction_step(
        first_state, 0.25F, 0.75F, 3.0F, reproducible_rng);
    const auto second_result = carbon::advance_independent_interaction_step(
        second_state, 0.25F, 0.75F, 3.0F, reproducible_rng);
    require(first_result.status == second_result.status &&
                first_result.channel == second_result.channel &&
                first_result.distance_mm == second_result.distance_mm &&
                first_state.elastic_remaining_tau == second_state.elastic_remaining_tau &&
                first_state.inelastic_remaining_tau == second_state.inelastic_remaining_tau &&
                first_state.elastic_rng_draws == second_state.elastic_rng_draws &&
                first_state.inelastic_rng_draws == second_state.inelastic_rng_draws,
            "independent clock default RNG is not reproducible");
}

void test_direction_rotation() {
    const auto local = carbon::ElasticDirection3F{0.6F, 0.8F, 0.0F};
    for (const auto parent : {carbon::ElasticDirection3F{0.0F, 0.0F, 1.0F},
                              carbon::ElasticDirection3F{0.0F, 0.0F, -1.0F},
                              carbon::ElasticDirection3F{1.0e-8F, 0.0F, 1.0F},
                              carbon::ElasticDirection3F{1.0F, 1.0F, 1.0F}}) {
        const auto output = carbon::rotate_elastic_local_direction(local, parent);
        const auto parent_norm = std::sqrt(parent.x * parent.x + parent.y * parent.y +
                                           parent.z * parent.z);
        const auto w_x = parent.x / parent_norm;
        const auto w_y = parent.y / parent_norm;
        const auto w_z = parent.z / parent_norm;
        const auto norm = std::sqrt(output.x * output.x + output.y * output.y +
                                    output.z * output.z);
        require_near(norm, 1.0F, 2.0e-5F, "rotated direction is not unit length");
        require_near(output.x * w_x + output.y * w_y + output.z * w_z, 0.0F, 2.0e-5F,
                     "relative local direction changed its polar angle");
    }
}

#ifdef CARBON_HAS_SYCL
void test_sycl_device_smoke() {
    auto queue = carbon::make_sycl_queue("cpu");
    auto* bins = sycl::malloc_shared<carbon::ElasticEnergyBin>(1, queue);
    auto* events = sycl::malloc_shared<carbon::ElasticEvent>(1, queue);
    auto* products = sycl::malloc_shared<carbon::ElasticProduct>(1, queue);
    auto* output = sycl::malloc_shared<carbon::SampledElasticEvent>(1, queue);
    auto* independent_output =
        sycl::malloc_shared<carbon::IndependentInteractionResult>(1, queue);
    require(bins != nullptr && events != nullptr && products != nullptr && output != nullptr &&
                independent_output != nullptr,
            "SYCL smoke allocation failed");
    bins[0] = {0.0F, 1.0F, 0U, 1U};
    events[0] = {1, 1, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0U, 0U, 1, 0};
    products[0] = {};
    *output = {};
    *independent_output = {};
    const carbon::ElasticPackageDeviceView view{bins, 1, events, 1, products, 1, 0.0F, 1.0F};
    queue.single_task([=] { *output = carbon::sample_elastic_event(view, 0, 0.25F); }).wait_and_throw();
    require(output->valid() && output->event_index == 0U, "SYCL elastic sampler smoke failed");
    queue.single_task([=] {
        carbon::IndependentInteractionClocks clocks{};
        const carbon::IndependentInteractionRng random{123U, 7U, 2U, 0U, 1U};
        *independent_output = carbon::advance_independent_interaction_step(
            clocks, 0.25F, 0.75F, 100.0F, random);
    }).wait_and_throw();
    require(independent_output->event(), "SYCL independent clock smoke failed");
    sycl::free(bins, queue);
    sycl::free(events, queue);
    sycl::free(products, queue);
    sycl::free(output, queue);
    sycl::free(independent_output, queue);
}
#endif

}  // namespace

int main() {
    try {
        test_uniform_event_sampling();
        test_competing_interaction();
        test_independent_interaction_clocks();
        test_direction_rotation();
#ifdef CARBON_HAS_SYCL
        test_sycl_device_smoke();
#endif
        std::cout << "elastic_sampling_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
