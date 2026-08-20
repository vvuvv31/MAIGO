#include "carbon/elastic_application.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef CARBON_HAS_SYCL
#include "carbon/device.hpp"
#include <sycl/sycl.hpp>
#endif

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(const float actual, const float expected, const float tolerance,
                 const char* message) {
    if (std::abs(actual - expected) > tolerance) throw std::runtime_error(message);
}

carbon::ElasticPrimaryState primary(const float energy,
                                    const carbon::ElasticDirection3F direction = {}) {
    carbon::ElasticPrimaryState state{};
    state.incoming_energy_MeV = energy;
    state.incoming_direction = direction;
    return state;
}

carbon::ElasticPackageDeviceView fixture_view(
    std::vector<carbon::ElasticEnergyBin>& bins,
    std::vector<carbon::ElasticEvent>& events,
    std::vector<carbon::ElasticProduct>& products) {
    bins = {{0.0F, 20.0F, 0U, 1U}};
    events = {{1, 1, 10.0F, 7.0F, 0.0F, 0.0F, 1.0F, 0.25F, 0U, 4U, 1, 2}};
    products = {
        {2212, 1, 1, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 3, 1},
        {22, 0, 0, 0.5F, 0.0F, 1.0F, 0.0F, 0.0F, 4, 2},
        {0, 0, 0, 0.75F, 1.0F, 0.0F, 0.0F, 0.0F, 5, 5},
        {2112, 0, 0, 0.5F, 0.0F, 0.0F, -1.0F, 0.0F, 6, 4},
    };
    return {bins.data(), bins.size(), events.data(), events.size(), products.data(),
            products.size(), 0.0F, 20.0F};
}

carbon::ElasticPackageDeviceView one_mevu_bin_fixture(
    std::vector<carbon::ElasticEnergyBin>& bins,
    std::vector<carbon::ElasticEvent>& events,
    std::vector<carbon::ElasticProduct>& products) {
    // The event is deliberately in the middle of a 1 MeV/u package bin.
    // Applying it at the lower, middle and upper parts of the bin exercises
    // the same mismatch that occurs in the merged proton ELPKG.
    bins = {{10.0F, 11.0F, 0U, 1U}};
    events = {{1, 1, 10.25F, 7.25F, 0.0F, 0.0F, 1.0F, 0.25F, 0U, 4U, 1, 2}};
    products = {
        {2212, 1, 1, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 3, 1},
        {22, 0, 0, 0.5F, 0.0F, 1.0F, 0.0F, 0.0F, 4, 2},
        {0, 0, 0, 0.75F, 1.0F, 0.0F, 0.0F, 0.0F, 5, 5},
        {2112, 0, 0, 0.5F, 0.0F, 0.0F, -1.0F, 0.0F, 6, 4},
    };
    return {bins.data(), bins.size(), events.data(), events.size(), products.data(),
            products.size(), 10.0F, 1.0F};
}

void test_mixed_application_and_ledger() {
    std::vector<carbon::ElasticEnergyBin> bins;
    std::vector<carbon::ElasticEvent> events;
    std::vector<carbon::ElasticProduct> products;
    auto view = fixture_view(bins, events, products);
    const auto sampled = carbon::sample_elastic_event(view, 0, 0.1F);
    require(sampled.valid(), "fixture event did not sample");
    auto current = primary(10.0F, {1.0F, 0.0F, 0.0F});
    auto next = current;
    std::vector<carbon::ElasticQueuedProduct> output(4);
    const auto result = carbon::apply_elastic_event(
        view, sampled, current, {output.data(), output.size()}, next);
    require(result.success(), "mixed elastic event application failed");
    require(result.queued_charged_count == 1U && result.queued_neutral_count == 1U,
            "charged/neutral queue counts were misclassified");
    require_near(result.queued_charged_energy_MeV, 1.0F, 1.0e-6F,
                 "charged queue energy mismatch");
    require_near(result.queued_neutral_energy_MeV, 0.5F, 1.0e-6F,
                 "neutral queue energy mismatch");
    require_near(result.local_deposit_MeV, 0.25F, 1.0e-6F,
                 "event local deposit was not recorded");
    require_near(result.discarded_energy_MeV, 0.5F, 1.0e-6F,
                 "discarded energy ledger mismatch");
    require_near(result.other_disposition_energy_MeV, 0.75F, 1.0e-6F,
                 "other disposition energy ledger mismatch");
    require_near(result.energy_closure_residual_MeV, 0.0F, 1.0e-5F,
                 "elastic energy ledger does not close");
    require(output[0].route == carbon::ElasticProductRoute::charged &&
                output[1].route == carbon::ElasticProductRoute::neutral,
            "output routes are not explicit charged/neutral descriptors");
    require(output[0].generation == 3U && output[1].generation == 4U,
            "product generation was not preserved");
    require(next.active && next.outgoing_energy_MeV == 7.0F,
            "primary continuation was not committed");
    require_near(next.outgoing_direction.x, 1.0F, 1.0e-5F,
                 "primary outgoing direction rotation is incorrect");
    require_near(next.outgoing_direction.y, 0.0F, 1.0e-5F,
                 "primary outgoing direction rotation is incorrect");
    require_near(next.outgoing_direction.z, 0.0F, 1.0e-5F,
                 "primary outgoing direction rotation is incorrect");
}

void test_no_recoil_and_direction_rotation() {
    std::vector<carbon::ElasticEnergyBin> bins{{0.0F, 20.0F, 0U, 1U}};
    std::vector<carbon::ElasticEvent> events{{1, 1, 5.0F, 5.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                              0U, 0U, 1, 0}};
    std::vector<carbon::ElasticProduct> products;
    const carbon::ElasticPackageDeviceView view{bins.data(), bins.size(), events.data(),
                                                events.size(), nullptr, 0U, 0.0F, 20.0F};
    const auto sampled = carbon::sample_elastic_event(view, 0, 0.5F);
    auto current = primary(5.0F, {0.0F, 0.0F, 1.0F});
    auto next = current;
    const auto result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.success() && result.queued_charged_count == 0U &&
                result.queued_neutral_count == 0U,
            "proton elastic no-recoil event failed");
    require(next.outgoing_direction.x > 0.9999F && std::abs(next.outgoing_direction.y) < 1e-5F,
            "direction rotation did not preserve the local transverse direction");

    // A non-axis parent exercises the stable local-to-global basis.
    events[0].outgoing_direction_x = 0.0F;
    events[0].outgoing_direction_y = 0.0F;
    events[0].outgoing_direction_z = 1.0F;
    current = primary(5.0F, {1.0F, 1.0F, 1.0F});
    next = current;
    const auto rotated = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(rotated.success(), "non-axis direction rotation failed");
    const auto norm = std::sqrt(next.outgoing_direction.x * next.outgoing_direction.x +
                                next.outgoing_direction.y * next.outgoing_direction.y +
                                next.outgoing_direction.z * next.outgoing_direction.z);
    require_near(norm, 1.0F, 2.0e-5F, "rotated primary direction is not normalized");
    require_near(next.outgoing_direction.x, 1.0F / std::sqrt(3.0F), 2.0e-5F,
                 "local forward direction did not follow parent direction");
}

void test_correlated_energy_scaling_across_one_mevu_bin() {
    std::vector<carbon::ElasticEnergyBin> bins;
    std::vector<carbon::ElasticEvent> events;
    std::vector<carbon::ElasticProduct> products;
    const auto view = one_mevu_bin_fixture(bins, events, products);
    const auto sampled = carbon::sample_elastic_event_for_energy(view, 10.5F, 0.5F);
    require(sampled.valid(), "1 MeV/u fixture event did not sample");

    const float current_energies[] = {10.01F, 10.50F, 10.99F};
    for (const auto current_energy : current_energies) {
        auto current = primary(current_energy, {1.0F, 0.0F, 0.0F});
        auto next = current;
        std::vector<carbon::ElasticQueuedProduct> output(4);
        const auto result = carbon::apply_elastic_event(
            view, sampled, current, {output.data(), output.size()}, next);
        const auto scale = current_energy / 10.25F;
        require(result.success(), "correlated elastic energy scaling failed");
        require_near(result.incident_energy_MeV, current_energy, 1.0e-6F,
                     "application result did not report current incident energy");
        require_near(result.outgoing_primary_energy_MeV, 7.25F * scale, 2.0e-5F,
                     "primary energy was not correlated-scaled");
        require_near(result.local_deposit_MeV, 0.25F * scale, 2.0e-5F,
                     "local deposit was not correlated-scaled");
        require_near(result.queued_charged_energy_MeV, 1.0F * scale, 2.0e-5F,
                     "charged product energy was not correlated-scaled");
        require_near(result.queued_neutral_energy_MeV, 0.5F * scale, 2.0e-5F,
                     "neutral product energy was not correlated-scaled");
        require_near(result.discarded_energy_MeV, 0.5F * scale, 2.0e-5F,
                     "discarded product energy was not correlated-scaled");
        require_near(result.other_disposition_energy_MeV, 0.75F * scale, 2.0e-5F,
                     "other disposition energy was not correlated-scaled");
        require_near(result.energy_closure_residual_MeV, 0.0F, 2.0e-5F,
                     "scaled elastic event did not close its energy ledger");
        require(output[0].pdg_id == 2212 && output[1].pdg_id == 22,
                "scaled output changed product identity or order");
        require_near(output[0].kinetic_energy_MeV, 1.0F * scale, 2.0e-5F,
                     "charged output energy was not scaled");
        require_near(output[1].kinetic_energy_MeV, 0.5F * scale, 2.0e-5F,
                     "neutral output energy was not scaled");
    }

    // The scale-one case is an exact compatibility check for the former path.
    auto current = primary(10.25F, {1.0F, 0.0F, 0.0F});
    auto next = current;
    std::vector<carbon::ElasticQueuedProduct> output(4);
    const auto result = carbon::apply_elastic_event(
        view, sampled, current, {output.data(), output.size()}, next);
    require(result.success(), "scale-one elastic application failed");
    require_near(result.outgoing_primary_energy_MeV, 7.25F, 1.0e-6F,
                 "scale-one primary energy changed");
    require_near(output[0].kinetic_energy_MeV, 1.0F, 1.0e-6F,
                 "scale-one product energy changed");
}

void test_capacity_is_atomic() {
    std::vector<carbon::ElasticEnergyBin> bins;
    std::vector<carbon::ElasticEvent> events;
    std::vector<carbon::ElasticProduct> products;
    const auto view = fixture_view(bins, events, products);
    const auto sampled = carbon::sample_elastic_event(view, 0, 0.1F);
    auto current = primary(10.0F);
    auto next = current;
    next.outgoing_energy_MeV = 77.0F;
    carbon::ElasticQueuedProduct sentinel{999, 9, 9, 9.0F, 9.0F, {9.0F, 9.0F, 9.0F}, 9,
                                          carbon::ElasticProductRoute::charged};
    carbon::ElasticQueuedProduct output = sentinel;
    const auto result = carbon::apply_elastic_event(view, sampled, current, {&output, 0U}, next);
    require(result.status == carbon::ElasticApplicationStatus::capacity_exceeded,
            "insufficient output capacity was not rejected");
    require(output.pdg_id == sentinel.pdg_id && next.outgoing_energy_MeV == 77.0F,
            "capacity failure partially modified output or primary state");
}

void test_invalid_inputs_are_rejected() {
    std::vector<carbon::ElasticEnergyBin> bins;
    std::vector<carbon::ElasticEvent> events;
    std::vector<carbon::ElasticProduct> products;
    auto view = fixture_view(bins, events, products);
    auto current = primary(10.0F);
    auto next = current;
    const auto sampled = carbon::sample_elastic_event(view, 0, 0.1F);
    events[0].product_offset = 99U;
    auto result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_product_range,
            "bad product range was accepted");
    events[0].product_offset = 0U;
    products[0].charge_e = 1.0F;
    products[0].pdg_id = 22;
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_charge_identity,
            "neutral PDG with charge was accepted");
    products[0] = {2212, 1, 1, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 3, 99};
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_disposition,
            "unknown disposition was accepted");
    products[0].transport_disposition = 1;
    products[0].generation = 256;
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_generation,
            "out-of-range generation was accepted");
    products[0].generation = 0;
    events[0].continuation = 0;
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_continuation,
            "invalid primary continuation was accepted");
    events[0].continuation = 1;
    current.incoming_energy_MeV = 9.0F;
    std::vector<carbon::ElasticQueuedProduct> scaled_output(4);
    result = carbon::apply_elastic_event(
        view, sampled, current, {scaled_output.data(), scaled_output.size()}, next);
    require(result.success() && result.incident_energy_MeV == 9.0F,
            "non-matching incoming energy was not correlated-scaled");

    current.incoming_energy_MeV = 0.0F;
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_primary,
            "zero incoming energy was accepted");
    current.incoming_energy_MeV = 10.0F;
    events[0].incident_energy_MeV_per_u = 0.0F;
    result = carbon::apply_elastic_event(view, sampled, current, {}, next);
    require(result.status == carbon::ElasticApplicationStatus::invalid_energy,
            "zero sampled incident energy was accepted");
}

#ifdef CARBON_HAS_SYCL
void test_sycl_device_smoke() {
    auto queue = carbon::make_sycl_queue("cpu");
    auto* bins = sycl::malloc_shared<carbon::ElasticEnergyBin>(1, queue);
    auto* events = sycl::malloc_shared<carbon::ElasticEvent>(1, queue);
    auto* products = sycl::malloc_shared<carbon::ElasticProduct>(2, queue);
    auto* output = sycl::malloc_shared<carbon::ElasticQueuedProduct>(10, queue);
    auto* result = sycl::malloc_shared<carbon::ElasticApplicationResult>(5, queue);
    auto* current = sycl::malloc_shared<carbon::ElasticPrimaryState>(5, queue);
    auto* next = sycl::malloc_shared<carbon::ElasticPrimaryState>(5, queue);
    require(bins && events && products && output && result && current && next,
            "elastic application SYCL allocation failed");
    *bins = {10.0F, 11.0F, 0U, 1U};
    *events = {1, 1, 10.25F, 8.0F, 0.0F, 0.0F, 1.0F, 0.25F, 0U, 2U, 1, 0};
    products[0] = {2212, 1, 1, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0, 1};
    products[1] = {22, 0, 0, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0, 2};
    current[0] = primary(10.01F);
    current[1] = primary(10.50F);
    current[2] = primary(10.99F);
    current[3] = primary(10.25F);
    current[4] = primary(0.0F);
    for (std::size_t index = 0; index < 5; ++index) next[index] = current[index];
    const carbon::ElasticPackageDeviceView view{bins, 1U, events, 1U, products, 2U, 10.0F,
                                                1.0F};
    queue.single_task([=] {
        const auto sampled = carbon::sample_elastic_event(view, 0U, 0.5F);
        for (std::size_t index = 0; index < 4; ++index) {
            result[index] = carbon::apply_elastic_event(
                view, sampled, current[index], {output + index * 2U, 2U}, next[index]);
        }
        result[4] = carbon::apply_elastic_event(view, sampled, current[4], {output + 8U, 2U},
                                                 next[4]);
    }).wait_and_throw();
    for (std::size_t index = 0; index < 4; ++index) {
        const auto scale = current[index].incoming_energy_MeV / 10.25F;
        require(result[index].success() && next[index].active,
                "elastic application SYCL scaling failed");
        require_near(result[index].incident_energy_MeV,
                     current[index].incoming_energy_MeV, 2.0e-5F,
                     "SYCL result did not report current incident energy");
        require_near(result[index].local_deposit_MeV, 0.25F * scale, 2.0e-5F,
                     "SYCL local deposit was not scaled");
        require_near(result[index].queued_charged_energy_MeV, scale, 2.0e-5F,
                     "SYCL charged energy was not scaled");
        require_near(result[index].queued_neutral_energy_MeV, scale, 2.0e-5F,
                     "SYCL neutral energy was not scaled");
        require_near(result[index].energy_closure_residual_MeV, 0.0F, 2.0e-5F,
                     "SYCL scaled event did not close");
        require_near(output[index * 2U].kinetic_energy_MeV, scale, 2.0e-5F,
                     "SYCL charged output was not scaled");
        require_near(output[index * 2U + 1U].kinetic_energy_MeV, scale, 2.0e-5F,
                     "SYCL neutral output was not scaled");
    }
    require(result[4].status == carbon::ElasticApplicationStatus::invalid_primary,
            "SYCL zero incoming energy was accepted");
    sycl::free(bins, queue);
    sycl::free(events, queue);
    sycl::free(products, queue);
    sycl::free(output, queue);
    sycl::free(result, queue);
    sycl::free(current, queue);
    sycl::free(next, queue);
}
#endif

}  // namespace

int main() {
    try {
        test_mixed_application_and_ledger();
        test_no_recoil_and_direction_rotation();
        test_correlated_energy_scaling_across_one_mevu_bin();
        test_capacity_is_atomic();
        test_invalid_inputs_are_rejected();
#ifdef CARBON_HAS_SYCL
        test_sycl_device_smoke();
#endif
        std::cout << "elastic_application_tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
